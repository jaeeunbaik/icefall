#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                  Wei Kang
#                                                  Mingshuang Luo)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Usage:
  export CUDA_VISIBLE_DEVICES="0,1,2,3"
  ./conformer_ctc/train.py \
     --exp-dir ./conformer_ctc/exp \
     --world-size 4 \
     --full-libri 1 \
     --max-duration 200 \
     --num-epochs 20
"""

import argparse
import logging
import time
from pathlib import Path
from shutil import copyfile
from typing import Optional, Tuple, Dict, List
import random
import numpy as np

import k2
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import sentencepiece as spm
from collections import defaultdict
from asr_datamodule import LibriSpeechAsrDataModule, LibriLightAsrDataModule
from conformer import Conformer
from ema_teacher import EMATeacher
from k_means_clustering import PrototypeKMeansManager
from lhotse.cut import Cut
from lhotse.utils import fix_random_seed
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from transformer import Noam
from decode import decode_dataset, save_results

from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.checkpoint import load_checkpoint
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.graph_compiler import CtcTrainingGraphCompiler
from icefall.lexicon import Lexicon
from icefall.rnn_lm.model import RnnLmModel
from icefall.utils import (
    AttributeDict,
    load_averaged_model,
    MetricsTracker,
    encode_supervisions,
    setup_logger,
    str2bool,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a checkpoint .pt file to resume training from. If set, overrides --start-epoch.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="""Resume training from from this epoch.
        If it is positive, it will load checkpoint from
        conformer_ctc/exp/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="./conformer_ctc_sd/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--lang-dir",
        type=str,
        default="./data/lang_phone",
        help="""The lang dir
        It contains language related input files such as
        "lexicon.txt"
        """,
    )
    
    parser.add_argument(
        "--bpe-dir",
        type=str,
        default="./data/lang_bpe_1024",
        help="""The lang dir
        It contains language related input files such as
        "lexicon.txt"
        """,
    )

    parser.add_argument(
        "--att-rate",
        type=float,
        default=0.0,
        help="""The attention rate.
        The total loss is (1 -  att_rate) * ctc_loss + att_rate * att_loss
        """,
    )

    parser.add_argument(
        "--num-decoder-layers",
        type=int,
        default=0,
        help="""Number of decoder layer of transformer decoder.
        Setting this to 0 will not create the decoder at all (pure CTC model)
        """,
    )

    parser.add_argument(
        "--lr-factor",
        type=float,
        default=2.0,
        help="The lr_factor for Noam optimizer. "
        "Lower values (1.0-2.0) recommended for fine-tuning pretrained models",
    )

    parser.add_argument(
        "--warm-step",
        type=int,
        default=30000,
        help="Number of warmup steps for Noam optimizer. "
        "Recommended: 30000 (with data aug), 15000-20000 (without data aug)",
    )

    # Fine-tuning scheduler options
    parser.add_argument(
        "--scheduler-type",
        type=str,
        default="noam",
        choices=["noam", "plateau", "constant"],
        help="Type of learning rate scheduler. "
        "noam: Noam scheduler (default), "
        "plateau: ReduceLROnPlateau, "
        "constant: Fixed learning rate",
    )
    
    parser.add_argument(
        "--base-lr",
        type=float,
        default=5e-5,
        help="Base learning rate for plateau and constant schedulers. "
        "For fine-tuning pretrained models: 5e-5 (encoder) to 1e-4 (new CTC layer)",
    )
    
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=3,
        help="Patience for ReduceLROnPlateau scheduler",
    )
    
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="Factor for ReduceLROnPlateau scheduler",
    )
    
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for ReduceLROnPlateau scheduler",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )
    parser.add_argument(
        "--sanity-check",
        type=str2bool,
        default=True,
        help="About Sanity check process",
    )
    
    # Self-distillation arguments
    parser.add_argument(
        "--enable-self-distillation",
        type=str2bool,
        default=True,
        help="Enable self-distillation training between clean and noisy samples",
    )
    
    parser.add_argument(
        "--distill-layers",
        type=str,
        default="6",
        help="Which encoder layer(s) to use for distillation (0-based). "
             "Can be a single layer (e.g., '6') or comma-separated list (e.g., '4,6,8'). "
             "Clean and noisy outputs from these layers will be compared.",
    )
    
    parser.add_argument(
        "--distill-loss-type",
        type=str,
        default="mse",
        choices=["mse", "cos", "kl"],
        help="Type of loss for self-distillation: 'mse' for Mean Squared Error, "
             "'cosine' for cosine similarity loss.",
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Weight for self-distillation loss. Total loss = ctc_loss + distill_weight * distill_loss",
    )
    
    parser.add_argument(
        "--distill-aggregation",
        type=str,
        default="layer_avg",
        choices=["layer_avg", "output_avg"],
        help="How to aggregate multi-layer distillation losses: "
             "'layer_avg' averages the layer outputs first then computes a single loss, "
             "'output_avg' computes loss for each layer and averages them.",
    )
    
    
    parser.add_argument(
        "--distill-temperature",
        type=float,
        default=1.0,
        help="Temperature for attention map distillation (used with KL divergence). "
             "Higher values make attention distributions smoother.",
    )
    
    parser.add_argument(
        "--layer-weights",
        type=str,
        default=None,
        help="Comma-separated weights for each distillation layer loss. "
             "Should match the number of layers in --distill-layers. "
             "Example: '0.5,0.7,1.0' for three layers. "
             "If not provided, all layers get equal weights (1.0).",
    )
    
    # EMA Teacher Model Arguments
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay rate for teacher model updates. "
             "Higher values (closer to 1.0) make teacher model change more slowly. "
             "Typical values: 0.999, 0.9999",
    )
    
    parser.add_argument(
        "--ema-start-step",
        type=int,
        default=1000,
        help="Step number to start EMA teacher model updates. "
             "Before this step, teacher model equals student model.",
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="ctc-decoding",
        help="""Decoding method.
        Supported values are:
        - ctc-decoding: CTC greedy search or beam search.
        - nbest-rescoring: Use N-best list for LM rescoring.
        - whole-lattice-rescoring: Use whole lattice for LM rescoring.
        - attention-decoder: Use attention decoder rescoring.
        - rnn-lm: Use RNN LM for rescoring.
        """,
    )
    
    parser.add_argument(
        "--enable-validation",
        type=str2bool,
        default=True,
        help="Enable validation during training. Set to False to disable validation completely.",
    )
    
    parser.add_argument(
        "--valid-interval",
        type=int,
        default=3000,
        help="Run validation every N batches. Increase this to validate less frequently.",
    )
    
    parser.add_argument(
        "--validation-decoding-method",
        type=str,
        default="greedy",
        choices=["greedy", "beam"],
        help="Decoding method for validation: 'greedy' for faster validation, 'beam' for more accurate WER.",
    )
    
    parser.add_argument(
        "--validation-search-beam",
        type=float,
        default=10.0,
        help="Search beam size for validation decoding (only used with beam search).",
    )
    
    parser.add_argument(
        "--validation-output-beam",
        type=float,
        default=5.0,
        help="Output beam size for validation decoding (only used with beam search).",
    )
    
    parser.add_argument(
        "--validation-skip-wer",
        type=str2bool,
        default=False,
        help="Skip WER computation during validation for faster validation (only compute loss).",
    )
    
    parser.add_argument(
        "--use-proj-layer",
        type=str2bool,
        default=True,
        help="Whether to use projection layer between encoder and decoder for self-distillation.",
    )
    
    parser.add_argument(
        "--learning-type",
        default="encoder-only",
        choices=["encoder-only", "hybrid", "asr"],
        help="""Training method, encoder-only for training only encoder, hybrid when loss is weighted sum of self-distillation and asr
        asr when training loss is only composed of asr loss
        """
    )
    
    parser.add_argument(
        "--clean-ratio",
        type=float,
        default=0.1,
    )
    
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="auto",
        choices=["auto", "librispeech", "librilight"],
        help="Dataset type to use. 'auto' selects LibriLight for encoder-only mode, LibriSpeech otherwise.",
    )
    
    # Prototype-based KL divergence arguments
    parser.add_argument(
        "--prototype-dir",
        type=str,
        default="./prototypes/librilight-512",
        help="Directory to save/load prototypes. If directory doesn't exist, prototypes will be initialized.",
    )
    
    parser.add_argument(
        "--num-prototypes",
        type=int,
        default=512,
        help="Number of prototypes per layer for KL-based distillation (K value for K-means)",
    )
    
    parser.add_argument(
        "--prototype-samples",
        type=int,
        default=100000,
        help="Number of feature samples per layer for prototype initialization using K-means",
    )
    
    parser.add_argument(
        "--recluster-prototypes-interval",
        type=int,
        default=0,
        help="Re-cluster prototypes every N epochs. 0 means no reclustering (use initial prototypes only). "
             "Recommended: 2-5 epochs for frequent updates, or 0 to disable.",
    )
    
    parser.add_argument(
        "--recluster-start-epoch",
        type=int,
        default=1,
        help="Start re-clustering prototypes from this epoch. Default: 1 (re-cluster after first epoch)",
    )
    
    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - subsampling_factor:  The subsampling factor for the model.

        - use_feat_batchnorm: Normalization for the input features, can be a
                              boolean indicating whether to do batch
                              normalization, or a float which means just scaling
                              the input features with this float value.
                              If given a float value, we will remove batchnorm
                              layer in `ConvolutionModule` as well.

        - attention_dim: Hidden dim for multi-head attention model.

        - head: Number of heads of multi-head attention model.

        - num_decoder_layers: Number of decoder layer of transformer decoder.

        - beam_size: It is used in k2.ctc_loss

        - reduction: It is used in k2.ctc_loss

        - use_double_scores: It is used in k2.ctc_loss

        - weight_decay:  The weight_decay for the optimizer.

        - warm_step: The warm_step for Noam optimizer.
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 3000,  # Default value, will be overridden by args
            # parameters for conformer
            "feature_dim": 80,
            "subsampling_factor": 4,
            "use_feat_batchnorm": True,
            "attention_dim": 256,
            "nhead": 4,
            # parameters for loss
            "beam_size": 4,  # Reduced from 10 to 4 for numerical stability
            "reduction": "sum",
            "use_double_scores": False,  # Changed to False for stability
            # parameters for decoding/validation
            "search_beam": 20.0,
            "output_beam": 8.0,
            "min_active_states": 30,
            "max_active_states": 10000,
            # parameters for Noam
            "weight_decay": 1e-6,
            "warm_step": 30000,
            "env_info": get_env_info()
        }
    )

    return params


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ema_teacher: Optional[EMATeacher] = None,
) -> Optional[dict]:
    """Load checkpoint from file.

    If params.start_epoch is positive, it will load the checkpoint from
    `params.start_epoch - 1`. Otherwise, this function does nothing.

    Apart from loading state dict for `model`, `optimizer` and `scheduler`,
    it also updates `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The learning rate scheduler we are using.
    Returns:
      Return None.
    """

    # Define models_dir consistently at the beginning
    models_dir = params.exp_dir / "models"

    # If resume-from is set, use that path directly
    resume_path = params.resume_from
    if resume_path:
        filename = Path(resume_path)
        if not filename.exists():
            logging.warning(f"Resume checkpoint not found at {filename}")
            return None
        
        logging.info(f"Loading pretrained checkpoint from: {filename}")
        
        # Load checkpoint with special handling for fine-tuning from pretrained encoder
        checkpoint = torch.load(filename, map_location='cpu')
        
        if 'model' in checkpoint:
            pretrained_state_dict = checkpoint['model']
        else:
            pretrained_state_dict = checkpoint
        
        # Get current model state dict
        model_state_dict = model.state_dict()
        
        # Filter out keys that don't match or shouldn't be loaded
        filtered_state_dict = {}
        skipped_keys = []
        new_keys = []
        
        for key, value in pretrained_state_dict.items():
            if key in model_state_dict:
                # Check if shapes match
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    skipped_keys.append(f"{key} (shape mismatch: {value.shape} vs {model_state_dict[key].shape})")
            else:
                skipped_keys.append(f"{key} (not in current model)")
        
        # Find keys in current model but not in pretrained checkpoint
        for key in model_state_dict:
            if key not in pretrained_state_dict:
                new_keys.append(key)
        
        # Load the filtered state dict with strict=False
        model.load_state_dict(filtered_state_dict, strict=False)
        
        logging.info(f"✓ Loaded {len(filtered_state_dict)} parameters from pretrained checkpoint")
        if skipped_keys:
            logging.info(f"⚠ Skipped {len(skipped_keys)} pretrained parameters (shape mismatch or not needed):")
            for key in skipped_keys[:10]:  # Show first 10
                logging.info(f"    - {key}")
            if len(skipped_keys) > 10:
                logging.info(f"    ... and {len(skipped_keys) - 10} more")
        
        if new_keys:
            logging.info(f"⚠ {len(new_keys)} parameters will be randomly initialized (not in pretrained model):")
            for key in new_keys[:10]:  # Show first 10
                logging.info(f"    - {key}")
            if len(new_keys) > 10:
                logging.info(f"    ... and {len(new_keys) - 10} more")
        
        # Don't load training state from pretrained checkpoint
        logging.info("Note: Training state (epoch, loss, optimizer) NOT loaded from pretrained checkpoint")
        logging.info("      Starting fresh training with pretrained encoder weights")
        
        return None  # Don't load training state
        
    else:
        if params.start_epoch <= 0:
            return None
        # First try to find checkpoint in models directory
        filename = models_dir / f"epoch-{params.start_epoch-1}.pt"
        # If not found in models directory, try the old location for backward compatibility
        if not filename.exists():
            filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
        if not filename.exists():
            logging.warning(f"Checkpoint not found at {filename}")
            return None
    
    saved_params = load_checkpoint(
        filename,
        model=model,
        optimizer=None,  # Don't load optimizer here, we'll handle it separately
        scheduler=scheduler,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch", 
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    # Load the full checkpoint data
    # We don't load optimizer state to avoid issues when optimizer type changes.
    # full_checkpoint = torch.load(filename, map_location='cpu')
    
    # if 'optimizer' in full_checkpoint:
    #     saved_params['optimizer'] = full_checkpoint['optimizer']

    # Try to load EMA teacher checkpoint if it exists
    if ema_teacher is not None:
        if resume_path:
            # If resume_path was used, try to find EMA checkpoint next to it
            resume_file = Path(resume_path)
            ema_filename = resume_file.parent / f"{resume_file.stem}-ema-teacher.pt"
        else:
            # Standard epoch-based EMA checkpoint naming
            ema_filename = models_dir / f"epoch-{params.start_epoch-1}-ema-teacher.pt"
        
        # If not found, try old location for backward compatibility
        if not ema_filename.exists():
            ema_filename = params.exp_dir / f"epoch-{params.start_epoch-1}-ema-teacher.pt"
        
        if ema_filename.exists():
            try:
                ema_state_dict = torch.load(ema_filename, map_location='cpu')
                ema_teacher.load_state_dict(ema_state_dict)
                logging.info(f"Loaded EMA teacher checkpoint from {ema_filename}")
                saved_params['ema_teacher'] = ema_state_dict
            except Exception as e:
                logging.warning(f"Failed to load EMA teacher checkpoint: {e}")
        else:
            logging.info("EMA teacher checkpoint not found, will initialize from student model")

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    rank: int = 0,
    suffix: str = "",
    wer_value: Optional[float] = None,
    step: Optional[int] = None,
    ema_teacher: Optional[EMATeacher] = None,
    epoch: Optional[int] = None,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      wer_value:
        WER value to include in filename (optional).
      step:
        Training step to include in filename instead of epoch (optional).
    """
    if rank != 0:
        return
    
    # Create models directory if it doesn't exist
    models_dir = params.exp_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    if suffix:
        # Use step instead of epoch for validation checkpoints
        epoch_or_step = step if step is not None else params.cur_epoch
        if wer_value is not None:
            filename = models_dir / f"step-{epoch_or_step}-{suffix}-wer{wer_value:.2f}-epoch{epoch}.pt"
        else:
            filename = models_dir / f"step-{epoch_or_step}-{suffix}.pt"
    else:
        filename = models_dir / f"epoch-{params.cur_epoch}.pt"
    
    # Save main checkpoint
    save_checkpoint_impl(
        filename=filename,
        model=model,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        rank=rank,
    )
    
    # Save EMA teacher model separately if it exists
    if ema_teacher is not None:
        ema_filename = models_dir / f"epoch-{params.cur_epoch}-ema-teacher.pt"
        torch.save(ema_teacher.state_dict(), ema_filename)
        logging.info(f"EMA teacher checkpoint saved to {ema_filename}")

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = models_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = models_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)
    
    logging.info(f"Checkpoint saved successfully to {filename}")
    # Remove the print statement that might be causing issues
    # print("Saving All Done!")


def _unwrap_ddp_model(model: nn.Module) -> nn.Module:
    """
    Unwrap DDP model to access the underlying model.
    
    Args:
        model: The model, which might be wrapped with DDP
        
    Returns:
        The underlying model without DDP wrapper
    """
    if isinstance(model, DDP):
        return model.module
    return model


def _extract_projected_outputs(
    model: nn.Module,
    layer_results: Optional[list],
    distill_layers: list,
    detach: bool = False,
) -> list:
    """
    Extract and project intermediate layer outputs for distillation.
    
    Args:
        model: The model (possibly wrapped with DDP)
        layer_results: List of intermediate layer outputs from encoder
        distill_layers: List of layer indices to use for distillation
        detach: If True, detach outputs from computation graph (for teacher)
        
    Returns:
        List of projected embeddings for distillation
    """
    if layer_results is None or not distill_layers:
        return []
    
    unwrapped_model = _unwrap_ddp_model(model)
    
    # Extract outputs from selected layers
    selected_outputs = []
    for layer_idx in distill_layers:
        if layer_idx < len(layer_results):
            output = layer_results[layer_idx]
            if detach:
                output = output.detach()
            selected_outputs.append(output)
    
    # Apply projection heads if available
    # CRITICAL: If detach=True (teacher), wrap projection in no_grad()
    if detach:
        with torch.no_grad():
            projected_embeddings = _apply_projection_heads(
                unwrapped_model, selected_outputs, detach=True
            )
    else:
        projected_embeddings = _apply_projection_heads(
            unwrapped_model, selected_outputs, detach=False
        )
    
    return projected_embeddings


def _apply_projection_heads(
    unwrapped_model: nn.Module,
    selected_outputs: list,
    detach: bool,
) -> list:
    """Apply projection heads to selected layer outputs."""
    projected_embeddings = []
    
    if (hasattr(unwrapped_model, 'use_proj_layer') and unwrapped_model.use_proj_layer 
        and hasattr(unwrapped_model, 'distill_projection_heads') 
        and unwrapped_model.distill_projection_heads
        and unwrapped_model.learning_type != "asr"):
        
        for i, layer_output in enumerate(selected_outputs):
            if i < len(unwrapped_model.distill_projection_heads):
                # Apply normalization if needed
                if unwrapped_model.normalize_before and hasattr(unwrapped_model, 'after_norm'):
                    normalized_output = unwrapped_model.after_norm(layer_output)
                else:
                    normalized_output = layer_output
                
                # Apply projection
                projected = unwrapped_model.distill_projection_heads[i](normalized_output)
                if detach:
                    projected = projected.detach()
                projected_embeddings.append(projected)
            else:
                projected_embeddings.append(layer_output)
    else:
        projected_embeddings = selected_outputs
    
    return projected_embeddings


def compute_loss(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
    graph_compiler: BpeCtcTrainingGraphCompiler,
    is_training: bool,
    ema_teacher: Optional[EMATeacher] = None,
    prototype_manager: Optional[PrototypeKMeansManager] = None,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute loss optimized by learning-type to avoid unnecessary computations.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of ConformerCTC.
      batch:
        A batch of data. Can contain both clean and noisy samples for self-distillation.
      graph_compiler:
        It is used to build a decoding graph from a ctc topo and training
        transcript.
      is_training:
        True for training. False for validation.
    """
    device = graph_compiler.device
    model_device = next(model.parameters()).device
    
    # Initialize metrics for tracking
    info = MetricsTracker()
    
    # Early branch based on learning_type to optimize computation
    if params.learning_type == "encoder-only":
        return _compute_encoder_only_loss(
            params, model, batch, graph_compiler, is_training, 
            ema_teacher, prototype_manager, device, model_device, info
        )
    elif params.learning_type == "hybrid":
        return _compute_hybrid_loss(
            params, model, batch, graph_compiler, is_training,
            ema_teacher, prototype_manager, device, model_device, info
        )
    elif params.learning_type == "asr":
        return _compute_asr_only_loss(
            params, model, batch, graph_compiler, is_training,
            device, model_device, info
        )
    else:
        raise ValueError(f"Unknown learning_type: {params.learning_type}")


def _compute_encoder_only_loss(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
    graph_compiler: BpeCtcTrainingGraphCompiler,
    is_training: bool,
    ema_teacher: Optional[EMATeacher],
    prototype_manager: Optional[PrototypeKMeansManager],
    device,
    model_device,
    info: MetricsTracker,
) -> Tuple[Tensor, MetricsTracker]:
    """Compute only distillation loss for encoder-only training."""
    
    # Handle clean-noisy batch structure
    if 'clean' not in batch or 'noisy' not in batch:
        raise ValueError("encoder-only mode requires clean-noisy batch structure")
    
    clean_feature = batch['clean']['inputs'].to(model_device)
    noisy_feature = batch['noisy']['inputs'].to(model_device)
    clean_supervisions = batch['clean']['supervisions']
    noisy_supervisions = batch['noisy']['supervisions']
    
    assert clean_feature.ndim == 3 and noisy_feature.ndim == 3
    
    with torch.set_grad_enabled(is_training):
        # Student forward pass (noisy sample) - single forward call
        _, _, _, student_layer_results, _ = model(noisy_feature, noisy_supervisions)
        
        # Get output lengths for distillation
        if isinstance(noisy_supervisions, dict) and 'num_frames' in noisy_supervisions:
            original_lengths = noisy_supervisions['num_frames']
            if not isinstance(original_lengths, torch.Tensor):
                original_lengths = torch.tensor(original_lengths, device=model_device)
            output_lens = (original_lengths + params.subsampling_factor - 1) // params.subsampling_factor
            output_lens = output_lens.cpu().tolist()
        else:
            batch_size = clean_feature.size(0)
            seq_len = clean_feature.size(1) // params.subsampling_factor
            output_lens = [seq_len] * batch_size
        
        # Teacher forward pass (with no_grad)
        teacher_layer_results = None
        with torch.no_grad():
            if ema_teacher is not None and params.batch_idx_train >= params.ema_start_step:
                teacher_model = ema_teacher.get_teacher_model()
                _, _, _, teacher_layer_results, _ = teacher_model(clean_feature, clean_supervisions)
            else:
                _, _, _, teacher_layer_results, _ = model(clean_feature, clean_supervisions)
        
        # Parse distillation layers
        try:
            distill_layers = [int(x.strip()) for x in params.distill_layers.split(',')]
        except:
            distill_layers = [int(params.distill_layers)]
        
        # Extract projected outputs from already computed layer results
        # IMPORTANT: Teacher must use detach=True to prevent gradient computation
        teacher_projected_outputs = _extract_projected_outputs(
            model=ema_teacher.get_teacher_model() if (ema_teacher is not None and params.batch_idx_train >= params.ema_start_step) else model,
            layer_results=teacher_layer_results,
            distill_layers=distill_layers,
            detach=True,  # Detach teacher to prevent gradient
        )
        
        # Student projection (needs gradient)
        student_projected_outputs = _extract_projected_outputs(
            model=model,
            layer_results=student_layer_results,
            distill_layers=distill_layers,
            detach=False,  # Keep gradient for student
        )
        
        # Compute distillation loss
        if teacher_projected_outputs and student_projected_outputs:
            from conformer import compute_multi_layer_distillation_loss
            
            # Parse layer weights
            layer_weights = None
            if params.layer_weights is not None:
                try:
                    layer_weights = [float(x.strip()) for x in params.layer_weights.split(',')]
                    if len(layer_weights) != len(distill_layers):
                        layer_weights = None
                except ValueError:
                    layer_weights = None
            
            distillation_loss = compute_multi_layer_distillation_loss(
                teacher_knowledge=teacher_projected_outputs,
                student_knowledge=student_projected_outputs,
                knowledge_lens=output_lens,
                layer_indices=list(range(len(distill_layers))),
                loss_type=params.distill_loss_type,
                aggregation=params.distill_aggregation,
                temperature=params.distill_temperature,
                prototype_manager=prototype_manager,
                target_layers=distill_layers,
                layer_weights=layer_weights,
            )
        else:
            distillation_loss = torch.tensor(0.0, device=model_device, requires_grad=is_training)
    
    # Update metrics
    info["frames"] = clean_feature.size(0) * clean_feature.size(1)  # Approximate
    info["ctc_loss"] = 0.0  # Not computed in encoder-only mode
    info["att_loss"] = 0.0  # Not computed in encoder-only mode
    info["distill_loss"] = distillation_loss.detach().cpu().item()
    info["loss"] = distillation_loss.detach().cpu().item()
    info["utterances"] = clean_feature.size(0)
    info["utt_duration"] = noisy_supervisions["num_frames"].sum().item() if isinstance(noisy_supervisions, dict) and 'num_frames' in noisy_supervisions else clean_feature.size(0) * clean_feature.size(1)
    info["utt_pad_proportion"] = 0.0  # Approximate
    
    return distillation_loss, info


def _compute_hybrid_loss(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
    graph_compiler: BpeCtcTrainingGraphCompiler,
    is_training: bool,
    ema_teacher: Optional[EMATeacher],
    prototype_manager: Optional[PrototypeKMeansManager],
    device,
    model_device,
    info: MetricsTracker,
) -> Tuple[Tensor, MetricsTracker]:
    """Compute CTC + distillation loss for hybrid training."""
    
    # Handle batch structure
    if 'clean' in batch and 'noisy' in batch:
        clean_feature = batch['clean']['inputs'].to(model_device)
        noisy_feature = batch['noisy']['inputs'].to(model_device)
        clean_supervisions = batch['clean']['supervisions']
        noisy_supervisions = batch['noisy']['supervisions']
        feature = noisy_feature
        supervisions = noisy_supervisions
    else:
        feature = batch["inputs"].to(model_device)
        supervisions = batch["supervisions"]
        clean_feature = None
        noisy_feature = None
        clean_supervisions = None
    
    assert feature.ndim == 3
    
    with torch.set_grad_enabled(is_training):
        # Student forward pass (noisy sample) - single forward call
        nnet_output, encoder_memory, memory_mask, student_layer_results, _ = model(feature, supervisions)
        
        # Compute student CTC loss
        s_ctc_loss, supervision_segments = compute_ctc_loss(params, graph_compiler, nnet_output, supervisions)
        
        # Teacher forward pass if clean samples available (with no_grad)
        t_output = None
        teacher_layer_results = None
        if clean_feature is not None:
            with torch.no_grad():
                if ema_teacher is not None and params.batch_idx_train >= params.ema_start_step:
                    teacher_model = ema_teacher.get_teacher_model()
                    t_output, _, _, teacher_layer_results, _ = teacher_model(clean_feature, clean_supervisions)
                else:
                    t_output, _, _, teacher_layer_results, _ = model(clean_feature, clean_supervisions)
                    
            if t_output is not None:
                t_ctc_loss, _ = compute_ctc_loss(params, graph_compiler, t_output, clean_supervisions)
                ctc_loss = params.clean_ratio * t_ctc_loss + (1 - params.clean_ratio) * s_ctc_loss
            else:
                ctc_loss = s_ctc_loss
        else:
            ctc_loss = s_ctc_loss
        
        # Compute attention loss if needed
        att_loss = torch.tensor([0], device=model_device)
        if params.att_rate != 0.0:
            mmodel = model.module if hasattr(model, "module") else model
            unsorted_token_ids = graph_compiler.texts_to_ids(supervisions["text"])
            att_loss = mmodel.decoder_forward(
                encoder_memory, memory_mask, token_ids=unsorted_token_ids,
                sos_id=graph_compiler.sos_id, eos_id=graph_compiler.eos_id,
            )
            ctc_att_loss = (1.0 - params.att_rate) * ctc_loss + params.att_rate * att_loss
        else:
            ctc_att_loss = ctc_loss
        
        # Compute distillation loss if clean samples available
        distillation_loss = torch.tensor(0.0, device=model_device)
        if clean_feature is not None and teacher_layer_results is not None and student_layer_results is not None:
            # Get output lengths
            if isinstance(supervisions, dict) and 'num_frames' in supervisions:
                original_lengths = supervisions['num_frames']
                if not isinstance(original_lengths, torch.Tensor):
                    original_lengths = torch.tensor(original_lengths, device=model_device)
                output_lens = (original_lengths + params.subsampling_factor - 1) // params.subsampling_factor
                output_lens = output_lens.cpu().tolist()
            else:
                batch_size = feature.size(0)
                seq_len = feature.size(1) // params.subsampling_factor
                output_lens = [seq_len] * batch_size
            
            # Parse distillation layers
            try:
                distill_layers = [int(x.strip()) for x in params.distill_layers.split(',')]
            except:
                distill_layers = [int(params.distill_layers)]
            
            # Extract projected outputs from already computed layer results
            # IMPORTANT: Teacher must use detach=True to prevent gradient computation
            teacher_projected_outputs = _extract_projected_outputs(
                model=ema_teacher.get_teacher_model() if (ema_teacher is not None and params.batch_idx_train >= params.ema_start_step) else model,
                layer_results=teacher_layer_results,
                distill_layers=distill_layers,
                detach=True,  # Detach teacher to prevent gradient
            )
            
            # Student projection (needs gradient)
            student_projected_outputs = _extract_projected_outputs(
                model=model,
                layer_results=student_layer_results,
                distill_layers=distill_layers,
                detach=False,  # Keep gradient for student
            )
            
            if teacher_projected_outputs and student_projected_outputs:
                from conformer import compute_multi_layer_distillation_loss
                
                # Parse layer weights
                layer_weights = None
                if params.layer_weights is not None:
                    try:
                        layer_weights = [float(x.strip()) for x in params.layer_weights.split(',')]
                        if len(layer_weights) != len(distill_layers):
                            layer_weights = None
                    except ValueError:
                        layer_weights = None
                
                distillation_loss = compute_multi_layer_distillation_loss(
                    teacher_knowledge=teacher_projected_outputs,
                    student_knowledge=student_projected_outputs,
                    knowledge_lens=output_lens,
                    layer_indices=list(range(len(distill_layers))),
                    loss_type=params.distill_loss_type,
                    aggregation=params.distill_aggregation,
                    temperature=params.distill_temperature,
                    prototype_manager=prototype_manager,
                    target_layers=distill_layers,
                    layer_weights=layer_weights,
                )
    
    # Combine losses
    total_loss = ctc_att_loss + params.alpha * distillation_loss
    
    # Update metrics
    info["frames"] = supervision_segments[:, 2].sum().item()
    info["ctc_loss"] = ctc_loss.detach().cpu().item()
    info["att_loss"] = att_loss.detach().cpu().item()
    info["distill_loss"] = distillation_loss.detach().cpu().item()
    info["loss"] = total_loss.detach().cpu().item()
    info["utterances"] = feature.size(0)
    info["utt_duration"] = supervisions["num_frames"].sum().item() if isinstance(supervisions, dict) and 'num_frames' in supervisions else feature.size(0) * feature.size(1)
    info["utt_pad_proportion"] = (
        ((feature.size(1) - supervisions["num_frames"]) / feature.size(1)).sum().item()
        if isinstance(supervisions, dict) and 'num_frames' in supervisions
        else 0.0
    )
    
    return total_loss, info


def _compute_asr_only_loss(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
    graph_compiler: BpeCtcTrainingGraphCompiler,
    is_training: bool,
    device,
    model_device,
    info: MetricsTracker,
) -> Tuple[Tensor, MetricsTracker]:
    """Compute only CTC + attention loss for ASR training (no distillation)."""
    
    feature = batch["inputs"].to(model_device)
    supervisions = batch["supervisions"]
    
    assert feature.ndim == 3
    
    with torch.set_grad_enabled(is_training):
        # Forward pass
        nnet_output, encoder_memory, memory_mask, _, _ = model(feature, supervisions)
        
        # Compute CTC loss
        ctc_loss, supervision_segments = compute_ctc_loss(params, graph_compiler, nnet_output, supervisions)
        
        # Compute attention loss if needed
        att_loss = torch.tensor([0], device=model_device)
        if params.att_rate != 0.0:
            mmodel = model.module if hasattr(model, "module") else model
            unsorted_token_ids = graph_compiler.texts_to_ids(supervisions["text"])
            att_loss = mmodel.decoder_forward(
                encoder_memory, memory_mask, token_ids=unsorted_token_ids,
                sos_id=graph_compiler.sos_id, eos_id=graph_compiler.eos_id,
            )
            total_loss = (1.0 - params.att_rate) * ctc_loss + params.att_rate * att_loss
        else:
            total_loss = ctc_loss
    
    # Update metrics
    info["frames"] = supervision_segments[:, 2].sum().item()
    info["ctc_loss"] = ctc_loss.detach().cpu().item()
    info["att_loss"] = att_loss.detach().cpu().item()
    info["distill_loss"] = 0.0  # No distillation in ASR mode
    info["loss"] = total_loss.detach().cpu().item()
    info["utterances"] = feature.size(0)
    info["utt_duration"] = supervisions["num_frames"].sum().item() if isinstance(supervisions, dict) and 'num_frames' in supervisions else feature.size(0) * feature.size(1)
    info["utt_pad_proportion"] = (
        ((feature.size(1) - supervisions["num_frames"]) / feature.size(1)).sum().item()
        if isinstance(supervisions, dict) and 'num_frames' in supervisions
        else 0.0
    )
    
    return total_loss, info

def compute_ctc_loss(
    params: AttributeDict,
    graph_compiler: BpeCtcTrainingGraphCompiler,
    nnet_output,
    supervisions,
):
    supervision_segments, texts = encode_supervisions(
        supervisions, subsampling_factor=params.subsampling_factor
    )

    if isinstance(graph_compiler, BpeCtcTrainingGraphCompiler):
        # Works with a BPE model
        token_ids = graph_compiler.texts_to_ids(texts)
        decoding_graph = graph_compiler.compile(token_ids)
    elif isinstance(graph_compiler, CtcTrainingGraphCompiler):
        # Works with a phone lexicon
        decoding_graph = graph_compiler.compile(texts)
    else:
        raise ValueError(f"Unsupported type of graph compiler: {type(graph_compiler)}")

    # Compute CTC loss
    dense_fsa_vec = k2.DenseFsaVec(
        nnet_output,
        supervision_segments,
        allow_truncate=max(params.subsampling_factor - 1, 10),
    )
    
    ctc_loss = k2.ctc_loss(
        decoding_graph=decoding_graph,
        dense_fsa_vec=dense_fsa_vec,
        # target_lengths=target_lengths,  # Removed - not in original
        output_beam=params.beam_size,
        reduction=params.reduction,  # Use original params.reduction
        use_double_scores=params.use_double_scores,
    )
    
    # Stabilize CTC loss for fine-tuning with converged models
    if ctc_loss.item() < 0:
        # For fine-tuning with converged models, negative CTC loss can occur
        # Apply a conservative fix: clamp to small positive value
        min_loss = 0.1  # Minimum positive loss value
        ctc_loss = torch.clamp(torch.abs(ctc_loss), min=min_loss)
    
    if torch.isnan(ctc_loss) or torch.isinf(ctc_loss):
        ctc_loss = torch.tensor(1.0, device=ctc_loss.device, requires_grad=True)
    
    # Additional check: if loss is too large, it might indicate numerical issues
    if ctc_loss.item() > 1000000:
        if params.batch_idx_train % 100 == 0:
            logging.warning(f"Very large CTC loss detected: {ctc_loss.item()}, clamping to reasonable range")
        ctc_loss = torch.clamp(ctc_loss, max=100000.0)
    
    return ctc_loss, supervision_segments


def compute_validation_loss(
    params: AttributeDict,
    model: nn.Module,
    graph_compiler: BpeCtcTrainingGraphCompiler,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
    epoch: int = 1,
    quick_validation: bool = True,
    rank: int = 0,
    tb_writer: Optional[SummaryWriter] = None,
) -> Tuple[MetricsTracker, Optional[float]]:
    """
    Compute validation loss and optionally WER.
    Returns:
        Tuple of (loss_metrics, wer_value)
    """
    
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Step 1: Compute validation loss
        tot_loss = MetricsTracker()
        
        for batch_idx, batch in enumerate(valid_dl):
            loss, loss_info = compute_loss(
                params=params,
                model=model,
                batch=batch,
                graph_compiler=graph_compiler,
                is_training=False,
                prototype_manager=None,
            )
            
            assert loss.requires_grad is False
            tot_loss = tot_loss + loss_info

        loss_value = tot_loss["loss"] / tot_loss["frames"]
        if loss_value < params.best_valid_loss:
            params.best_valid_epoch = params.cur_epoch
            params.best_valid_loss = loss_value

        logging.info(f"Validation loss: {loss_value:.4f}")

        # Step 2: Check if WER computation should be skipped
        if params.validation_skip_wer:
            logging.info("Skipping WER computation as requested")
            return tot_loss, None

        # Step 3: WER computation based on decode.py
        logging.info("Starting WER computation...")
        
        try:
            # Setup for decoding (similar to decode.py)
            sos_id = graph_compiler.sos_id
            eos_id = graph_compiler.eos_id
            
            # Read vocab size from tokens.txt
            tokens_file = params.lang_dir / "tokens.txt"
            with open(tokens_file, 'r', encoding='utf-8') as f:
                vocab_size = len(f.readlines())
            max_token_id = vocab_size - 1

            # Setup decoding components (from decode.py pattern)
            if params.att_rate == 0.0:
                # CTC decoding mode
                HLG = None
                H = k2.ctc_topo(
                    max_token=max_token_id,
                    modified=False,
                    device=device,
                )
                bpe_model = spm.SentencePieceProcessor()
                bpe_model.load(str(params.lang_dir / "bpe.model"))
            else:
                # Attention decoding mode
                H = None
                bpe_model = None
                HLG = k2.Fsa.from_dict(
                    torch.load(f"{params.lang_dir}/HLG.pt", map_location=device)
                )
                assert HLG.requires_grad is False
                if not hasattr(HLG, "lm_scores"):
                    HLG.lm_scores = HLG.scores.clone()

            # Create word table (from decode.py)
            word_table = {}
            with open(tokens_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            token, idx = parts[0], parts[1]
                            word_table[int(idx)] = token

            # Set validation-specific decoding parameters
            original_search_beam = getattr(params, 'search_beam', 20.0)
            original_output_beam = getattr(params, 'output_beam', 8.0)
            
            if params.validation_decoding_method == "greedy":
                params.search_beam = 1.0
                params.output_beam = 1.0
            else:
                params.search_beam = getattr(params, 'validation_search_beam', 10.0)
                params.output_beam = getattr(params, 'validation_output_beam', 5.0)

            # Decode validation set (using decode.py logic)
            results_dict = _decode_validation_dataset(
                dl=valid_dl,
                params=params,
                model=model,
                HLG=HLG,
                H=H,
                bpe_model=bpe_model,
                word_table=word_table,
                sos_id=sos_id,
                eos_id=eos_id,
                device=device,
            )
            
            # Compute WER and save results
            wer_value = None
            if results_dict:
                # Use first decoding method for WER calculation
                key = list(results_dict.keys())[0]
                results = results_dict[key]
                
                total_errors = 0
                total_words = 0
                error_details = []  # Store detailed error information
                
                # Calculate WER and collect error details
                for cut_id, ref_words, hyp_words in results:
                    # Simple WER calculation
                    ref_len = len(ref_words)
                    hyp_len = len(hyp_words)
                    
                    # Calculate edit distance (simplified)
                    errors = _calculate_edit_distance(ref_words, hyp_words)
                    
                    total_errors += errors
                    total_words += ref_len
                    
                    # Store error details for this utterance
                    utt_wer = (errors / ref_len * 100) if ref_len > 0 else 0.0
                    error_details.append({
                        'cut_id': cut_id,
                        'ref': ref_words,
                        'hyp': hyp_words,
                        'errors': errors,
                        'ref_len': ref_len,
                        'wer': utt_wer
                    })
                
                wer_value = (total_errors / total_words * 100) if total_words > 0 else 0.0
                logging.info(f"Validation WER: {wer_value:.2f}% (Errors: {total_errors}, Words: {total_words})")
                
                # Save results to files (only rank 0)
                if rank == 0:
                    # Create file names
                    recogs_filename = params.exp_dir / f"recogs-valid-{key}-epoch{epoch}-batch{params.batch_idx_train}.txt"
                    errs_filename = params.exp_dir / f"errs-valid-{key}-epoch{epoch}-batch{params.batch_idx_train}.txt"
                    wer_summary_file = params.exp_dir / f"wer-summary-epoch_{epoch}_validation.txt"
                    
                    # Save recognition results (ref and hyp pairs)
                    with open(recogs_filename, "w") as f:
                        for detail in error_details:
                            ref_text = " ".join(detail['ref'])
                            hyp_text = " ".join(detail['hyp'])
                            f.write(f"ref {detail['cut_id']}: {ref_text}\n")
                            f.write(f"hyp {detail['cut_id']}: {hyp_text}\n")
                            f.write("\n")
                    
                    logging.info(f"Saved validation transcripts to {recogs_filename}")
                    
                    # Save error statistics (detailed per utterance)
                    with open(errs_filename, "w") as f:
                        f.write("CUT_ID\tREF_LEN\tHYP_LEN\tERRORS\tWER(%)\n")
                        for detail in error_details:
                            f.write(f"{detail['cut_id']}\t{detail['ref_len']}\t{len(detail['hyp'])}\t{detail['errors']}\t{detail['wer']:.2f}\n")
                        f.write(f"\nOVERALL\t{total_words}\t-\t{total_errors}\t{wer_value:.2f}\n")
                    
                    logging.info(f"Saved error statistics to {errs_filename}")
                    
                    # Save WER summary
                    with open(wer_summary_file, "w") as f:
                        f.write("method\tWER\n")
                        f.write(f"{key}\t{wer_value:.2f}\n")
                    
                    logging.info(f"Saved WER summary to {wer_summary_file}")
            
            # Restore original beam parameters
            params.search_beam = original_search_beam
            params.output_beam = original_output_beam
            
            return tot_loss, wer_value
            
        except Exception as e:
            logging.error(f"WER computation failed: {e}")
            logging.error(f"Error type: {type(e).__name__}")
            # Restore beam parameters even on error
            if 'original_search_beam' in locals():
                params.search_beam = original_search_beam
                params.output_beam = original_output_beam
            return tot_loss, None


def _decode_validation_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    HLG: Optional[k2.Fsa],
    H: Optional[k2.Fsa],
    bpe_model: Optional[spm.SentencePieceProcessor],
    word_table: dict,
    sos_id: int,
    eos_id: int,
    device: torch.device,
    max_batches: int = 10,  # Limit batches for validation
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """
    Decode validation dataset (simplified version of decode.py's decode_dataset).
    Limited to max_batches for efficiency during training.
    """
    from collections import defaultdict
    from icefall.decode import get_lattice, one_best_decoding
    from icefall.utils import get_texts
    
    results = defaultdict(list)
    
    for batch_idx, batch in enumerate(dl):
        if batch_idx >= max_batches:
            break
            
        try:
            texts = batch["supervisions"]["text"]
            cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
            
            hyps_dict = _decode_one_validation_batch(
                params=params,
                model=model,
                HLG=HLG,
                H=H,
                bpe_model=bpe_model,
                batch=batch,
                word_table=word_table,
                sos_id=sos_id,
                eos_id=eos_id,
                device=device,
            )
            
            if hyps_dict is not None:
                for lm_scale, hyps in hyps_dict.items():
                    this_batch = []
                    for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                        ref_words = ref_text.split()
                        this_batch.append((cut_id, ref_words, hyp_words))
                    results[lm_scale].extend(this_batch)
                    
        except Exception as e:
            logging.warning(f"Failed to decode validation batch {batch_idx}: {e}")
            continue
    
    return results


def _decode_one_validation_batch(
    params: AttributeDict,
    model: nn.Module,
    HLG: Optional[k2.Fsa],
    H: Optional[k2.Fsa],
    bpe_model: Optional[spm.SentencePieceProcessor],
    batch: dict,
    word_table: dict,
    sos_id: int,
    eos_id: int,
    device: torch.device,
) -> Optional[Dict[str, List[List[str]]]]:
    """
    Decode one validation batch (simplified version of decode.py's decode_one_batch).
    """
    try:
        from icefall.decode import get_lattice, one_best_decoding
        from icefall.utils import get_texts
        
        feature = batch["inputs"].to(device)
        supervisions = batch["supervisions"]
        
        # Model forward pass
        nnet_output, memory, memory_key_padding_mask, _, _ = model(feature, None)
        
        # Prepare supervision segments
        supervision_segments = torch.stack(
            (
                supervisions["sequence_idx"],
                supervisions["start_frame"] // params.subsampling_factor,
                supervisions["num_frames"] // params.subsampling_factor,
            ),
            1,
        ).to(torch.int32)
        
        # Clamp supervision segments
        max_allowed_frames = nnet_output.size(1)
        supervision_segments[:, 2] = torch.clamp(supervision_segments[:, 2], max=max_allowed_frames)
        supervision_segments = supervision_segments.cpu()
        
        # Select decoding graph
        if H is None:
            decoding_graph = HLG
        else:
            decoding_graph = H
        
        # Get lattice
        lattice = get_lattice(
            nnet_output=nnet_output,
            decoding_graph=decoding_graph,
            supervision_segments=supervision_segments,
            search_beam=params.search_beam,
            output_beam=params.output_beam,
            min_active_states=getattr(params, 'min_active_states', 30),
            max_active_states=getattr(params, 'max_active_states', 1000),
            subsampling_factor=params.subsampling_factor,
        )
        
        # CTC decoding (simplified)
        best_path = one_best_decoding(
            lattice=lattice, 
            use_double_scores=getattr(params, 'use_double_scores', True)
        )
        
        if params.att_rate == 0.0 and bpe_model is not None:
            # CTC decoding with BPE
            token_ids = get_texts(best_path)
            hyps = bpe_model.decode(token_ids)
            hyps = [s.split() for s in hyps]
        else:
            # Standard decoding
            hyps = get_texts(best_path)
            hyps = [[word_table.get(i, f"<UNK{i}>") for i in ids] for ids in hyps]
        
        return {"validation": hyps}
        
    except Exception as e:
        logging.warning(f"Batch decoding failed: {e}")
        return None


def _calculate_edit_distance(ref_words: List[str], hyp_words: List[str]) -> int:
    """
    Calculate simple edit distance (Levenshtein distance).
    """
    m, n = len(ref_words), len(hyp_words)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def log_prediction_examples(results_dict, max_examples=5, force_log=False):
    """
    Log a few examples of ground truth vs predicted text for validation inspection.
    Only logs to terminal every 50 validation samples to reduce clutter.
    
    Args:
        results_dict: Dictionary containing decoding results
        max_examples: Maximum number of examples to log
        force_log: Force logging regardless of sample counter
    """
    
    if not results_dict:
        return
    
    # Get the first method's results (usually there's only one method in validation)
    first_method = list(results_dict.keys())[0]
    results = results_dict[first_method]
    
    if not results:
        return
    
    
    # Still compute and log basic statistics, just not the detailed examples
    total_sample_wer = 0
    valid_samples = 0
    
    for result in results:
        if len(result) >= 3:
            cut_id, ref_words, hyp_words = result[0], result[1], result[2]
            ref_text = " ".join(ref_words) if isinstance(ref_words, list) else str(ref_words)
            hyp_text = " ".join(hyp_words) if isinstance(hyp_words, list) else str(hyp_words)
            
            ref_word_list = ref_text.split()
            hyp_word_list = hyp_text.split()
            
            if len(ref_word_list) > 0:
                import difflib
                matcher = difflib.SequenceMatcher(None, ref_word_list, hyp_word_list)
                word_errors = len(ref_word_list) + len(hyp_word_list) - 2 * sum(triple.size for triple in matcher.get_matching_blocks())
                utt_wer = (word_errors / len(ref_word_list)) * 100
                total_sample_wer += utt_wer
                valid_samples += 1
    

    # Select diverse examples: some short, some long, some with errors, some perfect
    selected_examples = []
    
    # Try to get diverse examples by length and error type
    perfect_matches = []
    error_cases = []
    
    for result in results:
        if len(result) >= 3:
            cut_id, ref_words, hyp_words = result[0], result[1], result[2]
            ref_text = " ".join(ref_words) if isinstance(ref_words, list) else str(ref_words)
            hyp_text = " ".join(hyp_words) if isinstance(hyp_words, list) else str(hyp_words)
            
            if ref_text.split() == hyp_text.split():
                perfect_matches.append(result)
            else:
                error_cases.append(result)
    
    # Mix perfect matches and error cases
    selected_examples = error_cases[:max_examples-1] + perfect_matches[:1]
    if len(selected_examples) < max_examples:
        selected_examples.extend(results[:max_examples - len(selected_examples)])
    
    selected_examples = selected_examples[:max_examples]
    
    logging.info("=" * 80)
    logging.info(f"VALIDATION EXAMPLES (showing {len(selected_examples)} samples):")
    logging.info("=" * 80)
    
    total_sample_wer = 0
    valid_samples = 0
    
    for i, result in enumerate(selected_examples):
        if len(result) >= 3:
            cut_id, ref_words, hyp_words = result[0], result[1], result[2]
            
            # Convert word lists to strings
            ref_text = " ".join(ref_words) if isinstance(ref_words, list) else str(ref_words)
            hyp_text = " ".join(hyp_words) if isinstance(hyp_words, list) else str(hyp_words)
            
            logging.info(f"Example {i+1} (ID: {cut_id}):")
            logging.info(f"  REF: {ref_text}")
            logging.info(f"  HYP: {hyp_text}")
            
            # Simple word error analysis
            ref_word_list = ref_text.split()
            hyp_word_list = hyp_text.split()
            
            if ref_word_list == hyp_word_list:
                logging.info(f"  --> ✅ PERFECT MATCH ({len(ref_word_list)} words, WER: 0.0%)")
                total_sample_wer += 0.0
                valid_samples += 1
            else:
                # Basic error analysis
                ref_len = len(ref_word_list)
                hyp_len = len(hyp_word_list)
                
                # Calculate simple WER for this utterance
                import difflib
                matcher = difflib.SequenceMatcher(None, ref_word_list, hyp_word_list)
                word_errors = ref_len + hyp_len - 2 * sum(triple.size for triple in matcher.get_matching_blocks())
                utt_wer = (word_errors / ref_len * 100) if ref_len > 0 else 0
                total_sample_wer += utt_wer
                valid_samples += 1
                
                # Find common words for basic analysis
                ref_set = set(ref_word_list)
                hyp_set = set(hyp_word_list)
                missing_words = ref_set - hyp_set
                extra_words = hyp_set - ref_set
                
                error_info = f"WER: {utt_wer:.1f}%, REF: {ref_len} words, HYP: {hyp_len} words"
                if missing_words and len(missing_words) <= 3:
                    error_info += f", Missing: {list(missing_words)}"
                elif missing_words:
                    error_info += f", Missing: {len(missing_words)} words"
                    
                if extra_words and len(extra_words) <= 3:
                    error_info += f", Extra: {list(extra_words)}"
                elif extra_words:
                    error_info += f", Extra: {len(extra_words)} words"
                
                logging.info(f"  --> ❌ ERRORS ({error_info})")
            logging.info("")
    
    # Log average WER for the examples
    if valid_samples > 0:
        avg_example_wer = total_sample_wer / valid_samples
        logging.info(f"Average WER for these {valid_samples} examples: {avg_example_wer:.2f}%")
    
    logging.info("=" * 80)


def log_validation_examples_to_tensorboard(results_dict, tb_writer, step, max_examples=5):
    """
    Log validation examples to TensorBoard as text.
    
    Args:
        results_dict: Dictionary containing decoding results
        tb_writer: TensorBoard writer
        step: Current training step
        max_examples: Maximum number of examples to log
    """
    if not results_dict or tb_writer is None:
        return
    
    # Get the first method's results
    first_method = list(results_dict.keys())[0]
    results = results_dict[first_method]
    
    if not results:
        return
    
    # Select diverse examples
    selected_examples = []
    perfect_matches = []
    error_cases = []
    
    for result in results:
        if len(result) >= 3:
            cut_id, ref_words, hyp_words = result[0], result[1], result[2]
            ref_text = " ".join(ref_words) if isinstance(ref_words, list) else str(ref_words)
            hyp_text = " ".join(hyp_words) if isinstance(hyp_words, list) else str(hyp_words)
            
            if ref_text.split() == hyp_text.split():
                perfect_matches.append(result)
            else:
                error_cases.append(result)
    
    # Mix error cases and perfect matches
    selected_examples = error_cases[:max_examples-1] + perfect_matches[:1]
    if len(selected_examples) < max_examples:
        selected_examples.extend(results[:max_examples - len(selected_examples)])
    
    selected_examples = selected_examples[:max_examples]
    
    # Create text to log to TensorBoard
    tb_text = "## Validation Examples\n\n"
    
    total_wer = 0
    valid_count = 0
    
    for i, result in enumerate(selected_examples):
        if len(result) >= 3:
            cut_id, ref_words, hyp_words = result[0], result[1], result[2]
            
            ref_text = " ".join(ref_words) if isinstance(ref_words, list) else str(ref_words)
            hyp_text = " ".join(hyp_words) if isinstance(hyp_words, list) else str(hyp_words)
            
            tb_text += f"**Example {i+1} (ID: {cut_id})**\n\n"
            tb_text += f"- **REF:** {ref_text}\n"
            tb_text += f"- **HYP:** {hyp_text}\n"
            
            # Calculate simple WER for this utterance
            ref_word_list = ref_text.split()
            hyp_word_list = hyp_text.split()
            
            if ref_word_list == hyp_word_list:
                tb_text += f"- **Result:** ✅ PERFECT MATCH ({len(ref_word_list)} words, WER: 0.0%)\n\n"
                total_wer += 0.0
                valid_count += 1
            else:
                import difflib
                matcher = difflib.SequenceMatcher(None, ref_word_list, hyp_word_list)
                word_errors = len(ref_word_list) + len(hyp_word_list) - 2 * sum(triple.size for triple in matcher.get_matching_blocks())
                utt_wer = (word_errors / len(ref_word_list) * 100) if len(ref_word_list) > 0 else 0
                tb_text += f"- **Result:** ❌ WER: {utt_wer:.1f}% (REF: {len(ref_word_list)} words, HYP: {len(hyp_word_list)} words)\n\n"
                total_wer += utt_wer
                valid_count += 1
    
    # Add summary statistics
    if valid_count > 0:
        avg_wer = total_wer / valid_count
        tb_text += f"**Summary:** Average WER for {valid_count} examples: {avg_wer:.2f}%\n\n"
    
    # Log to TensorBoard
    tb_writer.add_text("Validation/Examples", tb_text, step)


def train_one_epoch(
    params: AttributeDict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    graph_compiler: BpeCtcTrainingGraphCompiler,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
    ema_teacher: Optional[EMATeacher] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    prototype_manager: Optional[PrototypeKMeansManager] = None,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      graph_compiler:
        It is used to convert transcripts to FSAs.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
    """
    model.train()
    
    # Log parameter training status at the start of each epoch
    if rank == 0:  # Only log from main process
        actual_model = model.module if hasattr(model, 'module') else model
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        logging.info(f"Epoch {params.cur_epoch} - Training mode: {params.learning_type}")
        logging.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        # Check if projection layer is being trained
        if hasattr(actual_model, 'proj_layer') and actual_model.use_proj_layer:
            proj_trainable = any(p.requires_grad for p in actual_model.proj_layer.parameters())
            logging.info(f"Projection layer trainable: {proj_trainable}")

    tot_loss = MetricsTracker()
    
    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])

        loss, loss_info = compute_loss(
            params=params,
            model=model,
            batch=batch,
            graph_compiler=graph_compiler,
            is_training=True,
            ema_teacher=ema_teacher,
            prototype_manager=prototype_manager,
        )
        # summary stats
        tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

        # NOTE: We use reduction==sum and loss is computed over utterances
        # in the batch and there is no normalization to it so far.

        optimizer.zero_grad()
        loss.backward()
        # More conservative gradient clipping for fine-tuning
        clip_grad_norm_(model.parameters(), 1.0, 2.0)  # Reduced from 5.0 to 1.0
        optimizer.step()
        
        
        # Update EMA teacher model after optimizer step
        if ema_teacher is not None and params.batch_idx_train >= params.ema_start_step:
            ema_teacher.update(model)
            if params.batch_idx_train % 1000 == 0:  # Log every 1000 steps instead of 100
                logging.info(f"EMA teacher updated at step {params.batch_idx_train}")

        if batch_idx % params.log_interval == 0:
            # Get current learning rate
            if params.scheduler_type == "noam" and hasattr(optimizer, '_rate'):
                cur_lr = optimizer._rate
            else:
                cur_lr = optimizer.param_groups[0]['lr']
            
            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}"
            )

        if batch_idx % params.log_interval == 0:
            # Get current learning rate for TensorBoard
            if params.scheduler_type == "noam" and hasattr(optimizer, '_rate'):
                cur_lr = optimizer._rate
            else:
                cur_lr = optimizer.param_groups[0]['lr']
                
            if tb_writer is not None:
                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                # Add learning rate to TensorBoard
                tb_writer.add_scalar("train/learning_rate", cur_lr, params.batch_idx_train)

        if batch_idx > 0 and batch_idx % params.valid_interval == 0 and params.enable_validation:
            logging.info(f"Computing validation loss (rank {rank})")
            
            
            # Use quick validation for frequent checks, full validation less frequently
            quick_val = (params.batch_idx_train % (params.valid_interval * 5) != 0)
            valid_info, validation_wer = compute_validation_loss(
                params=params,
                model=model,
                graph_compiler=graph_compiler,
                valid_dl=valid_dl,
                world_size=world_size,
                epoch=params.cur_epoch,
                quick_validation=quick_val,
                rank=rank,
                tb_writer=tb_writer,
            )

            
            # Log validation results with WER if available
            if validation_wer is not None:
                logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}, WER: {validation_wer:.2f}%")
            else:
                logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            
            # Update scheduler if using ReduceLROnPlateau
            if scheduler is not None and params.scheduler_type == "plateau":
                validation_loss = valid_info["loss"] / valid_info["frames"]
                scheduler.step(validation_loss)
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f"Scheduler step: validation_loss={validation_loss:.6f}, current_lr={current_lr:.2e}")
                        
            # Save checkpoint after validation (only rank 0)
            if rank == 0:
                logging.info(f"Saving checkpoint after validation at batch {batch_idx}")
                try:
                    save_checkpoint(
                        params=params,
                        model=model,
                        optimizer=optimizer,
                        rank=rank,
                        suffix=f"val-{batch_idx}",
                        wer_value=validation_wer,
                        step=batch_idx,
                        epoch=params.cur_epoch
                    )
                    logging.info(f"Checkpoint saved successfully for batch {batch_idx}")
                except Exception as e:
                    logging.error(f"Failed to save checkpoint: {e}")
                    # Continue training even if checkpoint saving fails
            model.train()
            
            
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )
                
                # Write WER to TensorBoard if validation results file exists and contains WER
                wer_summary_file = params.exp_dir / f"wer-summary-epoch_{params.cur_epoch}_validation.txt"
                if wer_summary_file.exists():
                    try:
                        with open(wer_summary_file, 'r') as f:
                            lines = f.readlines()
                            for line in lines[1:]:  # Skip header line
                                if line.strip():
                                    parts = line.strip().split('\t')
                                    if len(parts) >= 2:
                                        method_name = parts[0]
                                        wer_value = float(parts[1])
                                        tb_writer.add_scalar(f"train/valid_WER_{method_name}", wer_value, params.batch_idx_train)
                    except Exception as e:
                        logging.warning(f"Could not log WER to TensorBoard: {e}")


    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")
    logging.info(f"Warmup steps: {params.warm_step}")
    
    # Log projection layer training configuration
    logging.info("=" * 60)
    logging.info("PROJECTION LAYER TRAINING CONFIGURATION")
    logging.info("=" * 60)
    logging.info(f"Use projection layer: {params.use_proj_layer}")
    logging.info(f"Self-distillation enabled: {params.enable_self_distillation}")
    
    logging.info(params)

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)

    if "lang_bpe" in str(params.lang_dir):
        graph_compiler = BpeCtcTrainingGraphCompiler(
            params.lang_dir,
            device=device,
            sos_token="<sos/eos>",
            eos_token="<sos/eos>",
        )
        # Read vocab size from tokens.txt
        tokens_file = params.lang_dir / "tokens.txt"
        with open(tokens_file, 'r', encoding='utf-8') as f:
            num_classes = len(f.readlines())
        max_token_id = num_classes - 1
    elif "lang_phone" in str(params.lang_dir):
        assert params.att_rate == 0, (
            "Attention decoder training does not support phone lang dirs "
            "at this time due to a missing <sos/eos> symbol. Set --att-rate=0 "
            "for pure CTC training when using a phone-based lang dir."
        )
        assert params.num_decoder_layers == 0, (
            "Attention decoder training does not support phone lang dirs "
            "at this time due to a missing <sos/eos> symbol. "
            "Set --num-decoder-layers=0 for pure CTC training when using "
            "a phone-based lang dir."
        )
        lexicon = Lexicon(params.lang_dir)
        max_token_id = max(lexicon.tokens)
        num_classes = max_token_id + 1  # +1 for the blank
        graph_compiler = CtcTrainingGraphCompiler(
            lexicon,
            device=device,
        )
        # Manually add the sos/eos ID with their default values
        # from the BPE recipe which we're adapting here.
        graph_compiler.sos_id = 1
        graph_compiler.eos_id = 1
    else:
        raise ValueError(
            f"Unsupported type of lang dir (we expected it to have "
            f"'lang_bpe' or 'lang_phone' in its name): {params.lang_dir}"
        )

    logging.info("About to create model")
    
    # Parse distill_layers argument from string to List[int] if needed
    distill_layers = params.distill_layers
    if isinstance(distill_layers, str):
        distill_layers = [int(x) for x in distill_layers.split(',') if x.strip()]
    
    logging.info(f"Model creation parameters: distill_layers={distill_layers}")
    logging.info(f"Memory optimization: creating model for learning_type={params.learning_type}")

    model = Conformer(
        num_features=params.feature_dim,
        num_classes=num_classes,
        nhead=params.nhead,
        subsampling_factor=params.subsampling_factor,
        vgg_frontend=False,
        use_feat_batchnorm=params.use_feat_batchnorm,
        use_proj_layer=params.use_proj_layer,
        distill_layers=distill_layers,
        proj_dim=128,  # Compressed dimension for prototype-based distillation
        learning_type=params.learning_type,  # Memory optimization parameter
    )
    model.to(device)

    # Configure parameter training based on learning_type
    if params.learning_type == "encoder-only":
        logging.info("=" * 60)
        logging.info("ENCODER-ONLY MODE: Configuring parameter freezing")
        logging.info("=" * 60)
        logging.info("Freezing all parameters except encoder and distillation projection layers")
        
        # First freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Get the actual model (handle DDP wrapper)
        actual_model = model.module if hasattr(model, 'module') else model
        
        unfrozen_components = []
        
        # Unfreeze encoder parameters
        if hasattr(actual_model, 'encoder'):
            for param in actual_model.encoder.parameters():
                param.requires_grad = True
            unfrozen_components.append("encoder")
        
        # Unfreeze encoder embedding layers
        if hasattr(actual_model, 'encoder_embed'):
            for param in actual_model.encoder_embed.parameters():
                param.requires_grad = True
            unfrozen_components.append("encoder_embed")
        
        # Unfreeze encoder positional encoding
        if hasattr(actual_model, 'encoder_pos'):
            for param in actual_model.encoder_pos.parameters():
                param.requires_grad = True
            unfrozen_components.append("encoder_pos")
        
        # Unfreeze layer normalization layers (after_norm, etc.)
        if hasattr(actual_model, 'after_norm'):
            for param in actual_model.after_norm.parameters():
                param.requires_grad = True
            unfrozen_components.append("after_norm")
        
        # Unfreeze distillation-related projection layers based on use_proj_layer option
        if params.use_proj_layer:
            # Unfreeze main projection layer for distillation
            if hasattr(actual_model, 'proj_layer'):
                for param in actual_model.proj_layer.parameters():
                    param.requires_grad = True
                unfrozen_components.append("proj_layer")
            
            # Unfreeze distillation projection heads if they exist
            if hasattr(actual_model, 'distill_projection_heads'):
                for param in actual_model.distill_projection_heads.parameters():
                    param.requires_grad = True
                unfrozen_components.append("distill_projection_heads")
        
        # CTC-related layers remain FROZEN (ctc_output, linear, etc.)
        # These are only used for CTC loss computation, not for distillation
        
        logging.info(f"use_proj_layer: {params.use_proj_layer}")
        logging.info(f"Unfrozen components: {', '.join(unfrozen_components)}")
        logging.info("Frozen components: CTC projection heads (ctc_output, linear, etc.)")
        logging.info("=" * 60)
            
    else:
        logging.info(f"Mode: {params.learning_type} - All parameters trainable")
        # Ensure all parameters are trainable for other modes
        for param in model.parameters():
            param.requires_grad = True
        
    # Count and log trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params_count = total_params - trainable_params_count
    
    logging.info(f"Parameter Summary:")
    logging.info(f"  Total parameters: {total_params:,}")
    logging.info(f"  Trainable parameters: {trainable_params_count:,}")
    logging.info(f"  Frozen parameters: {frozen_params_count:,}")
    logging.info(f"  Trainable ratio: {trainable_params_count/total_params:.2%}")
        
    logging.info(f"Model created: distill_layers={model.distill_layers if hasattr(model, 'distill_layers') else 'NOT FOUND'}")
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Get trainable parameters for optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    logging.info(f"Optimizer will update {len(trainable_params)} parameter tensors")
        
    # Create optimizer and scheduler based on scheduler_type
    logging.info(f"Scheduler type: {params.scheduler_type}")
    logging.info(f"Base LR: {params.base_lr}")
    
    if params.scheduler_type == "noam":
        optimizer = Noam(
            trainable_params,  # Only pass trainable parameters
            model_size=params.attention_dim,
            factor=params.lr_factor,
            warm_step=params.warm_step,
            weight_decay=params.weight_decay,
        )
        scheduler = None  # Noam optimizer handles scheduling internally
        logging.info(f"Using Noam optimizer with lr_factor={params.lr_factor}")
    else:
        # Use Adam optimizer for plateau and constant schedulers
        base_lr = params.base_lr  # Default fallback
        logging.info(f"Using Adam optimizer with base_lr={base_lr}")
        
        optimizer = torch.optim.Adam(
            trainable_params,  # Only pass trainable parameters
            lr=base_lr,
            weight_decay=params.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        if params.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=params.scheduler_factor,
                patience=params.scheduler_patience,
                min_lr=params.min_lr,
                verbose=True
            )
            logging.info(f"Using ReduceLROnPlateau scheduler: "
                        f"lr={params.base_lr}, patience={params.scheduler_patience}, "
                        f"factor={params.scheduler_factor}, min_lr={params.min_lr}")
        elif params.scheduler_type == "constant":
            scheduler = None  # No scheduler for constant learning rate
            logging.info(f"Using constant learning rate: {params.base_lr}")
        else:
            raise ValueError(f"Unknown scheduler type: {params.scheduler_type}")
    
    logging.info(f"Optimizer type: {type(optimizer).__name__}")
    logging.info(f"Scheduler type: {params.scheduler_type}")
    
    # Debug optimizer state
    for param_group in optimizer.param_groups:
        logging.info(f"Optimizer param group lr: {param_group['lr']}")
        break  # Just check first param group

    # Determine dataset type based on learning_type and dataset_type argument
    if params.dataset_type == "auto":
        use_librilight = (params.learning_type == "encoder-only")
    else:
        use_librilight = (params.dataset_type == "librilight")
    
    # Create appropriate data module based on dataset type
    if use_librilight:
        logging.info("Using LibriLight dataset for prototype-based knowledge distillation")
        data_module = LibriLightAsrDataModule(args)
        # Use LibriLight training data for K-means clustering and training
        train_cuts = data_module.librilight_train_cuts()
        logging.info(f"Loaded LibriLight training cuts: {len(train_cuts)} utterances")
    else:
        logging.info("Using LibriSpeech dataset")
        data_module = LibriSpeechAsrDataModule(args)
        
        # Load appropriate LibriSpeech training cuts based on configuration
        if params.mini_libri:
            logging.info("Using mini LibriSpeech (train-clean-5)")
            train_cuts = data_module.train_clean_5_cuts()
        elif params.full_libri:
            logging.info("Using full LibriSpeech (960h)")
            train_cuts = (
                data_module.train_clean_100_cuts()
                + data_module.train_clean_360_cuts()
                + data_module.train_other_500_cuts()
            )
        else:
            logging.info("Using LibriSpeech clean-100h subset")
            train_cuts = data_module.train_clean_100_cuts()

    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 20 seconds
        #
        # Caution: There is a reason to select 20.0 here. Please see
        # ../local/display_manifest_statistics.py
        #
        # You should use ../local/display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        return 1.0 <= c.duration <= 20.0

    train_cuts = train_cuts.filter(remove_short_and_long_utt)
    
    # For self-distillation: use shuffle=True with fixed seed for deterministic but diverse ordering
    train_dl = data_module.train_dataloaders(train_cuts, shuffle=True)

    # Create validation dataloader based on dataset type
    if use_librilight:
        # LibriLight uses LibriSpeech dev-clean for validation
        valid_cuts = data_module.librilight_dev_cuts()
        logging.info("Using LibriSpeech dev-clean for LibriLight validation")
    else:
        # Use LibriSpeech dev sets
        if params.mini_libri:
            valid_cuts = data_module.dev_clean_2_cuts()
            logging.info("Using mini LibriSpeech dev-clean-2 for validation")
        else:
            valid_cuts = data_module.dev_clean_cuts()
            # valid_cuts += data_module.dev_other_cuts()  # Comment out for faster validation
            logging.info("Using LibriSpeech dev-clean for validation")
    
    valid_dl = data_module.valid_dataloaders(valid_cuts)
    logging.info(f"Validation set size: {len(valid_cuts)} utterances")
    
    if params.sanity_check:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            graph_compiler=graph_compiler,
            params=params,
        )
    else: pass
    
    # Initialize EMA Teacher Model for self-distillation (must be before checkpoint loading)
    # Memory optimization: Skip EMA teacher for ASR-only mode
    ema_teacher = None
    if params.enable_self_distillation and params.learning_type != "asr":
        logging.info(f"Initializing EMA teacher model with decay={params.ema_decay}, start_step={params.ema_start_step}")
        ema_teacher = EMATeacher(model, decay=params.ema_decay, device=device)    
        logging.info(f"Memory optimization: EMA teacher created for {params.learning_type} mode")
    elif params.learning_type == "asr":
        logging.info(f"Memory optimization: Skipped EMA teacher for ASR-only mode")
    else:
        logging.info(f"EMA teacher disabled (enable_self_distillation={params.enable_self_distillation})")    

    checkpoints = load_checkpoint_if_available(
        params=params, 
        model=model, 
        ema_teacher=ema_teacher
    )
    
    # For fine-tuning, we DO NOT load the optimizer state.
    # The optimizer is created fresh.
    # if checkpoints and "optimizer" in checkpoints: 
    #     try:
    #         optimizer.load_state_dict(checkpoints["optimizer"])
    #         logging.info("Successfully loaded optimizer state from checkpoint")
    #     except (ValueError, KeyError, RuntimeError) as e:
    #         logging.warning(f"Failed to load optimizer state: {e}")
    #         logging.warning("Starting with fresh optimizer state")
    #         # Continue training with fresh optimizer state
    
    # Initialize prototype manager for KL-based distillation
    # Memory optimization: Skip prototype manager for ASR-only mode
    prototype_manager = None
    if params.learning_type in ["encoder-only", "hybrid"] and params.distill_loss_type == "kl":
        try:
            # Parse distillation layers
            distill_layers = [int(x.strip()) for x in params.distill_layers.split(',')]
        except:
            distill_layers = [int(params.distill_layers)]
        
        if distill_layers and rank == 0:  # Only initialize on rank 0
            logging.info("Initializing prototype manager for KL-based distillation")
            logging.info(f"Target layers for prototypes: {distill_layers}")
            logging.info(f"Prototype directory: {params.prototype_dir}")
            
            prototype_manager = PrototypeKMeansManager(
                target_layers=distill_layers,
                num_prototypes=params.num_prototypes,
                proj_dim=128,  # Use compressed dimension for prototype-based distillation
                temperature=params.distill_temperature,
                save_dir=params.prototype_dir
            )
            
            # Check if prototype directory exists and has required prototype files
            prototype_dir = Path(params.prototype_dir)
            prototype_files_exist = all(
                (prototype_dir / f"prototypes_layer_{layer_idx}.pt").exists() 
                for layer_idx in distill_layers
            )
            
            if not prototype_files_exist:
                logging.info("Prototype files not found. Initializing prototypes using K-means clustering")
                teacher_model = ema_teacher.get_teacher_model() if ema_teacher else model
                prototype_manager.initialize_prototypes(
                    teacher_model=teacher_model,
                    dataloader=train_dl,
                    num_samples_per_layer=params.prototype_samples,
                    load_if_exists=False  # Force initialization since files don't exist
                )
                logging.info("Prototype initialization completed")
            else:
                logging.info("Prototype files found. Loading existing prototypes")
                prototype_manager.load_prototypes(params.prototype_dir)
                logging.info("Prototype loading completed")
            
            logging.info(f"Memory optimization: Prototype manager created for {params.learning_type} mode")
    elif params.learning_type == "asr":
        logging.info(f"Memory optimization: Skipped prototype manager for ASR-only mode")

    # Log memory optimization summary after all components are initialized
    actual_model = model.module if hasattr(model, 'module') else model
    logging.info("=" * 60)
    logging.info("MEMORY OPTIMIZATION SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Learning type: {params.learning_type}")
    
    # Check which components are enabled/disabled
    has_decoder = hasattr(actual_model, 'decoder') and actual_model.decoder is not None
    has_proj_layer = hasattr(actual_model, 'proj_layer') and actual_model.proj_layer is not None
    has_distill_heads = hasattr(actual_model, 'distill_projection_heads') and len(actual_model.distill_projection_heads) > 0
    has_ctc_output = hasattr(actual_model, 'ctc_output') and actual_model.ctc_output is not None
    
    logging.info(f"  Decoder: {'Enabled' if has_decoder else 'Disabled (Memory Saved)'}")
    logging.info(f"  Projection layer: {'Enabled' if has_proj_layer else 'Disabled (Memory Saved)'}")
    logging.info(f"  Distillation heads: {'Enabled' if has_distill_heads else 'Disabled (Memory Saved)'}")
    logging.info(f"  CTC output: {'Enabled' if has_ctc_output else 'Disabled (Memory Saved)'}")
    logging.info(f"  EMA teacher: {'Enabled' if ema_teacher is not None else 'Disabled (Memory Saved)'}")
    logging.info(f"  Prototype manager: {'Enabled' if prototype_manager is not None else 'Disabled (Memory Saved)'}")
    
    # Estimate memory savings
    savings_info = []
    if not has_decoder and params.learning_type == "encoder-only":
        savings_info.append("Decoder layers")
    if not has_proj_layer and params.learning_type == "asr":
        savings_info.append("Projection layers")
    if not has_distill_heads and params.learning_type == "asr":
        savings_info.append("Distillation heads")
    if ema_teacher is None and params.learning_type == "asr":
        savings_info.append("EMA teacher model")
    if prototype_manager is None and params.learning_type == "asr":
        savings_info.append("Prototype manager")
    
    if savings_info:
        logging.info(f"  Memory saved by disabling: {', '.join(savings_info)}")
    else:
        logging.info(f"  All components enabled for {params.learning_type} mode")
    
    logging.info("=" * 60)

    for epoch in range(params.start_epoch, params.num_epochs):
        fix_random_seed(params.seed + epoch)
        train_dl.sampler.set_epoch(epoch)

        # Re-cluster prototypes at specified intervals
        if (
            prototype_manager is not None
            and params.recluster_prototypes_interval > 0
            and epoch >= params.recluster_start_epoch
            and (epoch - params.recluster_start_epoch) % params.recluster_prototypes_interval == 0
            and epoch > params.start_epoch  # Don't recluster on the very first epoch we're resuming from
        ):
            if rank == 0:
                logging.info(f"=" * 60)
                logging.info(f"Re-clustering prototypes at epoch {epoch}")
                logging.info(f"  - Interval: every {params.recluster_prototypes_interval} epochs")
                logging.info(f"  - Started from: epoch {params.recluster_start_epoch}")
                logging.info(f"=" * 60)
            
            # Re-initialize prototypes using K-means on fresh feature samples
            start_time = time.time()
            prototype_manager.initialize_prototypes(
                model=model,
                train_dl=train_dl,
                num_samples=params.prototype_samples,
                device=device,
                rank=rank,
                world_size=world_size,
            )
            
            if rank == 0:
                elapsed = time.time() - start_time
                logging.info(f"Prototype re-clustering completed in {elapsed:.2f} seconds")
                logging.info(f"=" * 60)

        # Get current learning rate based on optimizer type
        if params.scheduler_type == "noam" and hasattr(optimizer, '_rate'):
            cur_lr = optimizer._rate
        else:
            cur_lr = optimizer.param_groups[0]['lr']
            
        if tb_writer is not None:
            tb_writer.add_scalar("train/learning_rate", cur_lr, params.batch_idx_train)
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        if rank == 0:
            logging.info("epoch {}, learning rate {}".format(epoch, cur_lr))

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            optimizer=optimizer,
            graph_compiler=graph_compiler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
            ema_teacher=ema_teacher,
            scheduler=scheduler,
            prototype_manager=prototype_manager,
        )

        save_checkpoint(
            params=params,
            model=model,
            optimizer=optimizer,
            rank=rank,
            ema_teacher=ema_teacher,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()

def scan_pessimistic_batches_for_oom(
    model: nn.Module,
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    graph_compiler: BpeCtcTrainingGraphCompiler,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 0 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        try:
            optimizer.zero_grad()
            loss, _ = compute_loss(
                params=params,
                model=model,
                batch=batch,
                graph_compiler=graph_compiler,
                is_training=True,
                prototype_manager=None,  # Sanity check doesn't need prototype manager
            )
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0, 2.0)
            optimizer.step()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            raise


def log_prediction_examples(results_dict, max_examples=5, force_log=False):
    """
    Log a few examples of ground truth vs predicted text for validation inspection.
    Only logs to terminal every 50 validation samples to reduce clutter.
    
    Args:
        results_dict: Dictionary containing decoding results
        max_examples: Maximum number of examples to log
        force_log: Force logging regardless of sample counter
    """
    
    if not results_dict:
        return
    
    # Get the first method's results (usually there's only one method in validation)
    first_method = list(results_dict.keys())[0]
    results = results_dict[first_method]
    
    if not results:
        return
    
    
    # Still compute and log basic statistics, just not the detailed examples
    total_sample_wer = 0
    valid_samples = 0
    
    for result in results:
        if len(result) >= 3:
            cut_id, ref_words, hyp_words = result[0], result[1], result[2]
            ref_text = " ".join(ref_words) if isinstance(ref_words, list) else str(ref_words)
            hyp_text = " ".join(hyp_words) if isinstance(hyp_words, list) else str(hyp_words)
            
            ref_word_list = ref_text.split()
            hyp_word_list = hyp_text.split()
            
            if len(ref_word_list) > 0:
                import difflib
                matcher = difflib.SequenceMatcher(None, ref_word_list, hyp_word_list)
                word_errors = len(ref_word_list) + len(hyp_word_list) - 2 * sum(triple.size for triple in matcher.get_matching_blocks())
                utt_wer = (word_errors / len(ref_word_list)) * 100
                total_sample_wer += utt_wer
                valid_samples += 1
    

    # Select diverse examples: some short, some long, some with errors, some perfect
    selected_examples = []
    
    # Try to get diverse examples by length and error type
    perfect_matches = []
    error_cases = []
    
    for result in results:
        if len(result) >= 3:
            cut_id, ref_words, hyp_words = result[0], result[1], result[2]
            ref_text = " ".join(ref_words) if isinstance(ref_words, list) else str(ref_words)
            hyp_text = " ".join(hyp_words) if isinstance(hyp_words, list) else str(hyp_words)
            
            if ref_text.split() == hyp_text.split():
                perfect_matches.append(result)
            else:
                error_cases.append(result)
    
    # Mix perfect matches and error cases
    selected_examples = error_cases[:max_examples-1] + perfect_matches[:1]
    if len(selected_examples) < max_examples:
        selected_examples.extend(results[:max_examples - len(selected_examples)])
    
    selected_examples = selected_examples[:max_examples]
    
    logging.info("=" * 80)
    logging.info(f"VALIDATION EXAMPLES (showing {len(selected_examples)} samples):")
    logging.info("=" * 80)
    
    total_sample_wer = 0
    valid_samples = 0
    
    for i, result in enumerate(selected_examples):
        if len(result) >= 3:
            cut_id, ref_words, hyp_words = result[0], result[1], result[2]
            
            # Convert word lists to strings
            ref_text = " ".join(ref_words) if isinstance(ref_words, list) else str(ref_words)
            hyp_text = " ".join(hyp_words) if isinstance(hyp_words, list) else str(hyp_words)
            
            logging.info(f"Example {i+1} (ID: {cut_id}):")
            logging.info(f"  REF: {ref_text}")
            logging.info(f"  HYP: {hyp_text}")
            
            # Simple word error analysis
            ref_word_list = ref_text.split()
            hyp_word_list = hyp_text.split()
            
            if ref_word_list == hyp_word_list:
                logging.info(f"  --> ✅ PERFECT MATCH ({len(ref_word_list)} words, WER: 0.0%)")
                total_sample_wer += 0.0
                valid_samples += 1
            else:
                # Basic error analysis
                ref_len = len(ref_word_list)
                hyp_len = len(hyp_word_list)
                
                # Calculate simple WER for this utterance
                import difflib
                matcher = difflib.SequenceMatcher(None, ref_word_list, hyp_word_list)
                word_errors = ref_len + hyp_len - 2 * sum(triple.size for triple in matcher.get_matching_blocks())
                utt_wer = (word_errors / ref_len * 100) if ref_len > 0 else 0
                total_sample_wer += utt_wer
                valid_samples += 1
                
                # Find common words for basic analysis
                ref_set = set(ref_word_list)
                hyp_set = set(hyp_word_list)
                missing_words = ref_set - hyp_set
                extra_words = hyp_set - ref_set
                
                error_info = f"WER: {utt_wer:.1f}%, REF: {ref_len} words, HYP: {hyp_len} words"
                if missing_words and len(missing_words) <= 3:
                    error_info += f", Missing: {list(missing_words)}"
                elif missing_words:
                    error_info += f", Missing: {len(missing_words)} words"
                    
                if extra_words and len(extra_words) <= 3:
                    error_info += f", Extra: {list(extra_words)}"
                elif extra_words:
                    error_info += f", Extra: {len(extra_words)} words"
                
                logging.info(f"  --> ❌ ERRORS ({error_info})")
            logging.info("")
    
    # Log average WER for the examples
    if valid_samples > 0:
        avg_example_wer = total_sample_wer / valid_samples
        logging.info(f"Average WER for these {valid_samples} examples: {avg_example_wer:.2f}%")
    
    logging.info("=" * 80)


def log_validation_examples_to_tensorboard(results_dict, tb_writer, step, max_examples=5):
    """
    Log validation examples to TensorBoard as text.
    
    Args:
        results_dict: Dictionary containing decoding results
        tb_writer: TensorBoard writer
        step: Current training step
        max_examples: Maximum number of examples to log
    """
    if not results_dict or tb_writer is None:
        return
    
    # Get the first method's results
    first_method = list(results_dict.keys())[0]
    results = results_dict[first_method]
    
    if not results:
        return
    
    # Select diverse examples
    selected_examples = []
    perfect_matches = []
    error_cases = []
    
    for result in results:
        if len(result) >= 3:
            cut_id, ref_words, hyp_words = result[0], result[1], result[2]
            ref_text = " ".join(ref_words) if isinstance(ref_words, list) else str(ref_words)
            hyp_text = " ".join(hyp_words) if isinstance(hyp_words, list) else str(hyp_words)
            
            if ref_text.split() == hyp_text.split():
                perfect_matches.append(result)
            else:
                error_cases.append(result)
    
    # Mix error cases and perfect matches
    selected_examples = error_cases[:max_examples-1] + perfect_matches[:1]
    if len(selected_examples) < max_examples:
        selected_examples.extend(results[:max_examples - len(selected_examples)])
    
    selected_examples = selected_examples[:max_examples]
    
    # Create text to log to TensorBoard
    tb_text = "## Validation Examples\n\n"
    
    total_wer = 0
    valid_count = 0
    
    for i, result in enumerate(selected_examples):
        if len(result) >= 3:
            cut_id, ref_words, hyp_words = result[0], result[1], result[2]
            
            ref_text = " ".join(ref_words) if isinstance(ref_words, list) else str(ref_words)
            hyp_text = " ".join(hyp_words) if isinstance(hyp_words, list) else str(hyp_words)
            
            tb_text += f"**Example {i+1} (ID: {cut_id})**\n\n"
            tb_text += f"- **REF:** {ref_text}\n"
            tb_text += f"- **HYP:** {hyp_text}\n"
            
            # Calculate simple WER for this utterance
            ref_word_list = ref_text.split()
            hyp_word_list = hyp_text.split()
            
            if ref_word_list == hyp_word_list:
                tb_text += f"- **Result:** ✅ PERFECT MATCH ({len(ref_word_list)} words, WER: 0.0%)\n\n"
                total_wer += 0.0
                valid_count += 1
            else:
                import difflib
                matcher = difflib.SequenceMatcher(None, ref_word_list, hyp_word_list)
                word_errors = len(ref_word_list) + len(hyp_word_list) - 2 * sum(triple.size for triple in matcher.get_matching_blocks())
                utt_wer = (word_errors / len(ref_word_list) * 100) if len(ref_word_list) > 0 else 0
                tb_text += f"- **Result:** ❌ WER: {utt_wer:.1f}% (REF: {len(ref_word_list)} words, HYP: {len(hyp_word_list)} words)\n\n"
                total_wer += utt_wer
                valid_count += 1
    
    # Add summary statistics
    if valid_count > 0:
        avg_wer = total_wer / valid_count
        tb_text += f"**Summary:** Average WER for {valid_count} examples: {avg_wer:.2f}%\n\n"
    
    # Log to TensorBoard
    tb_writer.add_text("Validation/Examples", tb_text, step)


def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    LibriLightAsrDataModule.add_arguments(parser)  # Add LibriLight arguments
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)
    args.bpe_dir = Path(args.bpe_dir)
    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)



if __name__ == "__main__":
    main()
