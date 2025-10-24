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
from pathlib import Path
from shutil import copyfile
from typing import Optional, Tuple
import random
import numpy as np

import k2
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import sentencepiece as spm
from asr_datamodule import LibriSpeechAsrDataModule
from conformer import Conformer
from ema_teacher import EMATeacher
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
        default=0.8,
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
        default=5.0,
        help="The lr_factor for Noam optimizer",
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
        default=2e-5,
        help="Base learning rate for plateau and constant schedulers",
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
        "--knowledge",
        type=str,
        default="encoder-output",
        choices=["encoder-output", "attention-map"],
        help="Type of knowledge to use for self-distillation: "
             "'encoder-output' uses intermediate encoder layer outputs, "
             "'attention-map' uses attention weights from self-attention layers.",
    )
    
    parser.add_argument(
        "--distill-temperature",
        type=float,
        default=1.0,
        help="Temperature for attention map distillation (used with KL divergence). "
             "Higher values make attention distributions smoother.",
    )
    
    # Clean Augmentation Arguments for Self-Distillation
    parser.add_argument(
        "--enable-clean-augmentation",
        type=str2bool,
        default=False,
        help="Enable light augmentation for clean samples in self-distillation. "
             "When enabled, clean samples get lighter noise/augmentation compared to noisy samples.",
    )
    
    parser.add_argument(
        "--clean-augmentation-prob",
        type=float,
        default=0.1,
        help="Probability of applying augmentation to clean samples. "
             "Lower values maintain cleaner samples while adding slight variation. "
             "Typical values: 0.05-0.2",
    )
    
    parser.add_argument(
        "--clean-specaugment-time-mask-max-frames",
        type=int,
        default=5,
        help="Maximum time mask length for clean SpecAugment (much smaller than noisy). "
             "Clean samples use lighter SpecAugment for slight variation.",
    )
    
    parser.add_argument(
        "--clean-specaugment-freq-mask-max-bins",
        type=int,
        default=3,
        help="Maximum frequency mask bins for clean SpecAugment (much smaller than noisy). "
             "Clean samples use lighter SpecAugment for slight variation.",
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
        "--proj-layer-training",
        type=str,
        default="full-finetuning",
        choices=["full-finetuning", "only-proj"],
        help="Training method for self distillation: 'full-finetuning' for fine-tuning all parameters, 'only-proj' for training only proj layer, else freezed.",
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
    resume_path = getattr(params, "resume_from", None)
    if resume_path:
        filename = Path(resume_path)
        if not filename.exists():
            logging.warning(f"Resume checkpoint not found at {filename}")
            return None
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

    # Load the full checkpoint data including optimizer state
    full_checkpoint = torch.load(filename, map_location='cpu')
    
    # Add optimizer state to saved_params if it exists
    if 'optimizer' in full_checkpoint:
        saved_params['optimizer'] = full_checkpoint['optimizer']

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


def compute_loss(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
    graph_compiler: BpeCtcTrainingGraphCompiler,
    is_training: bool,
    ema_teacher: Optional[EMATeacher] = None,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute CTC loss with optional self-distillation.

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
    
    # Handle clean-noisy batch structure for self-distillation
    if 'clean' in batch and 'noisy' in batch and params.enable_self_distillation:
        # Self-distillation mode with clean-noisy samples
        clean_feature = batch['clean']['inputs']
        noisy_feature = batch['noisy']['inputs']
        clean_supervisions = batch['clean']['supervisions']
        noisy_supervisions = batch['noisy']['supervisions']
        
        # Use noisy samples as primary for CTC loss computation
        feature = noisy_feature
        supervisions = noisy_supervisions
        
        # Note: We'll move to the correct device later after getting model device
        use_self_distillation = True
        # Clean-noisy pairs are being used for self-distillation
    else:
        # Normal mode or self-distillation disabled
        feature = batch["inputs"]
        supervisions = batch["supervisions"]
        # Note: We'll move to the correct device later after getting model device
        
        clean_feature = None
        noisy_feature = None
        clean_supervisions = None
        use_self_distillation = False
    
    # at entry, feature is (N, T, C)
    assert feature.ndim == 3
    
    # Ensure model and feature are on the same device
    model_device = next(model.parameters()).device
    
    # Move feature to the correct device if needed
    if feature.device != model_device:
        feature = feature.to(model_device)
    
    # Also move clean_feature if it exists
    if clean_feature is not None and clean_feature.device != model_device:
        clean_feature = clean_feature.to(model_device)
    t_output = None
    with torch.set_grad_enabled(is_training):
        # Forward pass through model (noisy sample)
        nnet_output, encoder_memory, memory_mask, hiddens, att_maps = model(feature, supervisions)
        
        # Get the original lengths from supervisions
        if isinstance(supervisions, dict) and 'num_frames' in supervisions:
            original_lengths = supervisions['num_frames']
            if not isinstance(original_lengths, torch.Tensor):
                original_lengths = torch.tensor(original_lengths, device=model_device)
            # Apply subsampling factor to get actual output lengths
            output_lens = (original_lengths + params.subsampling_factor - 1) // params.subsampling_factor
            output_lens = output_lens.cpu().tolist()  # Convert to list for distillation functions
        else:
            # Fallback: use actual nnet_output sequence lengths for each sample
            batch_size = nnet_output.size(1)  # N from (T, N, C)
            seq_len = nnet_output.size(0)     # T from (T, N, C) 
            output_lens = [seq_len] * batch_size
        
        # Self-distillation computation
        distillation_loss = torch.tensor(0.0, device=model_device)
        
        if use_self_distillation and params.enable_self_distillation and clean_feature is not None:
            # Log only occasionally to reduce spam
            if params.batch_idx_train % 1000 == 0:
                logging.info(f"Self-distillation active: ema_teacher={ema_teacher is not None}, step={params.batch_idx_train}")
            
            # Use EMA teacher model if available, otherwise use the same model with clean samples
            if ema_teacher is not None and params.batch_idx_train >= params.ema_start_step:
                teacher_model = ema_teacher.get_teacher_model()
                with torch.no_grad():
                    t_output, _, _, t_hidden, t_maps = teacher_model(clean_feature, clean_supervisions)
                if params.batch_idx_train % 1000 == 0:
                    logging.info(f"Using EMA teacher model for distillation (step {params.batch_idx_train})")
            else:
                # 학습 초반부에 EMA 모델 없으면 student 모델로 계산
                with torch.no_grad():
                    t_output, _, _, t_hidden, t_maps = model(clean_feature, clean_supervisions)
                if ema_teacher is not None and params.batch_idx_train % 1000 == 0:
                    logging.info(f"EMA teacher exists but step {params.batch_idx_train} < start_step {params.ema_start_step}, using same model")
                elif ema_teacher is None and params.batch_idx_train % 1000 == 0:
                    logging.info(f"No EMA teacher, using same model with clean samples as teacher")
            
            # Parse distillation layers from comma-separated string
            try:
                distill_layers = [int(x.strip()) for x in params.distill_layers.split(',')]
            except:
                distill_layers = [int(params.distill_layers)]
            
            if params.knowledge == "encoder-output":
                # Extract encoder outputs for distillation with projection layers
                if params.use_proj_layer and len(distill_layers) > 0:
                    # Use get_intermediate_outputs for projection-applied distillation layers
                    if ema_teacher is not None and params.batch_idx_train >= params.ema_start_step:
                        teacher_model = ema_teacher.get_teacher_model()
                        with torch.no_grad():
                            teacher_projected_outputs = teacher_model.get_intermediate_outputs(clean_feature, clean_supervisions)
                    else:
                        with torch.no_grad():
                            teacher_projected_outputs = model.get_intermediate_outputs(clean_feature, clean_supervisions)
                    
                    # Get student projected outputs
                    student_projected_outputs = model.get_intermediate_outputs(feature, supervisions)
                    
                    if teacher_projected_outputs and student_projected_outputs:
                        from conformer import compute_multi_layer_distillation_loss
                        distillation_loss = compute_multi_layer_distillation_loss(
                            teacher_knowledge=teacher_projected_outputs,
                            student_knowledge=student_projected_outputs,
                            knowledge_lens=output_lens,
                            layer_indices=list(range(len(distill_layers))),  # Use sequential indices since we already filtered layers
                            loss_type=params.distill_loss_type,
                            knowledge_type="encoder-output",
                            aggregation=params.distill_aggregation,
                        )
                    else:
                        logging.warning("Failed to get projected outputs for distillation")
                        distillation_loss = torch.tensor(0.0, device=model_device)
                
                elif hiddens is not None and t_hidden is not None:
                    # Fallback to original method when projection is disabled
                    # Import the multi-layer distillation function
                    from conformer import compute_multi_layer_distillation_loss
                    
                    if len(distill_layers) == 1:
                        from conformer import compute_distillation_loss
                        distillation_loss = compute_distillation_loss(
                            teacher_knowledge=t_output,
                            student_knowledge=nnet_output,
                            knowledge_lens=output_lens,
                            loss_type=params.distill_loss_type,
                            knowledge_type="encoder-output",
                        )
                    else:
                        from conformer import compute_multi_layer_distillation_loss
                        distillation_loss = compute_multi_layer_distillation_loss(
                            teacher_knowledge=t_hidden,
                            student_knowledge=hiddens,
                            knowledge_lens=output_lens,
                            layer_indices=distill_layers,
                            loss_type=params.distill_loss_type,
                            knowledge_type="encoder-output",
                            aggregation=params.distill_aggregation,
                        )
                    
            elif params.knowledge == "attention-map":
                # Extract attention maps for distillation
                if att_maps is not None and t_maps is not None:
                    
                    # Import the multi-layer distillation function
                    from conformer import compute_multi_layer_distillation_loss
                    
                    distillation_loss = compute_multi_layer_distillation_loss(
                        teacher_knowledge=t_maps,
                        student_knowledge=att_maps,
                        knowledge_lens=output_lens,
                        layer_indices=distill_layers,
                        loss_type="kl",  # Always use KL divergence for attention maps
                        knowledge_type="attention-map",
                        aggregation=params.distill_aggregation,
                        temperature=params.distill_temperature,
                    )
                    logging.debug(f"Attention-map distillation loss computed: {distillation_loss.item():.6f}")
                else:
                    logging.warning("Attention maps not found in model output. Distillation disabled for this batch.")
                    distillation_loss = torch.tensor(0.0, device=model_device)
        else:
            if not use_self_distillation:
                logging.debug("Self-distillation disabled (use_self_distillation=False)")
            elif clean_feature is None:
                logging.warning("Clean feature is None, skipping self-distillation")

    # NOTE: We need `encode_supervisions` to sort sequences with
    # different duration in decreasing order, required by
    # `k2.intersect_dense` called in `k2.ctc_loss`
    s_ctc_loss, supervision_segments = compute_ctc_loss(params, graph_compiler, nnet_output, supervisions)
    if t_output is not None:
        t_ctc_loss, _ = compute_ctc_loss(params, graph_compiler, t_output, clean_supervisions)
        ctc_loss = 0.1 * t_ctc_loss + 0.9 * s_ctc_loss
    else:
        ctc_loss = s_ctc_loss



    # Attention loss computation (if applicable)
    if params.att_rate != 0.0:
        with torch.set_grad_enabled(is_training):
            mmodel = model.module if hasattr(model, "module") else model
            # Note: We need to generate an unsorted version of token_ids
            unsorted_token_ids = graph_compiler.texts_to_ids(supervisions["text"])
            att_loss = mmodel.decoder_forward(
                encoder_memory,
                memory_mask,
                token_ids=unsorted_token_ids,
                sos_id=graph_compiler.sos_id,
                eos_id=graph_compiler.eos_id,
            )
        total_loss = (1.0 - params.att_rate) * ctc_loss + params.att_rate * att_loss
    else:
        att_loss = torch.tensor([0])
        total_loss = ctc_loss

    # Add self-distillation loss with proper scale matching
    if use_self_distillation and distillation_loss.item() > 0:
        # With reduction='mean', CTC loss scale is more manageable
        total_loss = total_loss + params.alpha * distillation_loss
        logging.debug(f"Loss combination: ctc_loss={total_loss-params.alpha*distillation_loss:.4f}, "
                     f"distill_loss={distillation_loss:.4f}, alpha={params.alpha}, "
                     f"total_loss={total_loss:.4f}")

    assert total_loss.requires_grad == is_training

    # Metrics tracking
    info = MetricsTracker()
    info["frames"] = supervision_segments[:, 2].sum().item()
    info["ctc_loss"] = ctc_loss.detach().cpu().item()
    info["att_loss"] = att_loss.detach().cpu().item()
    info["distill_loss"] = distillation_loss.detach().cpu().item()
    info["loss"] = total_loss.detach().cpu().item()

    # `utt_duration` and `utt_pad_proportion` would be normalized by `utterances`
    info["utterances"] = feature.size(0)
    # averaged input duration in frames over utterances
    info["utt_duration"] = supervisions["num_frames"].sum().item()
    # averaged padding proportion over utterances
    info["utt_pad_proportion"] = (
        ((feature.size(1) - supervisions["num_frames"]) / feature.size(1)).sum().item()
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
    quick_validation: bool = True,  # Add option for quick validation
    rank: int = 0,  # Add rank parameter
    tb_writer: Optional[SummaryWriter] = None,  # Add TensorBoard writer parameter
) -> MetricsTracker:

    
    model.eval()
    
    with torch.no_grad():
        device = next(model.parameters()).device
        tot_loss = MetricsTracker()
        
        for batch_idx, batch in enumerate(valid_dl):
            loss, loss_info = compute_loss(
                params=params,
                model=model,
                batch=batch,
                graph_compiler=graph_compiler,
                is_training=False,
            )
            
            assert loss.requires_grad is False
            tot_loss = tot_loss + loss_info

        loss_value = tot_loss["loss"] / tot_loss["frames"]
        if loss_value < params.best_valid_loss:
            params.best_valid_epoch = params.cur_epoch
            params.best_valid_loss = loss_value

        logging.info("Validation loss computation completed")

        # Check if WER computation should be skipped
        if params.validation_skip_wer:
            logging.info("Skipping WER computation as requested")
            return tot_loss, None

        # TODO: Re-enable WER computation after fixing decode_dataset issues
        # Always compute WER for analysis
        logging.info("Starting WER computation...")
        
        # Use the existing graph_compiler instead of creating a new one
        # to ensure device compatibility in DDP training
        sos_id = graph_compiler.sos_id
        eos_id = graph_compiler.eos_id
        
        # Read vocab size from tokens.txt
        tokens_file = params.lang_dir / "tokens.txt"
        with open(tokens_file, 'r', encoding='utf-8') as f:
            vocab_size = len(f.readlines())
        max_token_id = vocab_size - 1

        # WER calculation with proper device handling
        if params.att_rate == 0.0:
            HLG = None
            H = k2.ctc_topo(
                max_token=max_token_id,
                modified=False,
                device=device,
            )
            bpe_model = spm.SentencePieceProcessor()
            bpe_model.load(str(params.lang_dir / "bpe.model"))
        else:
            H = None
            bpe_model = None
            HLG = k2.Fsa.from_dict(
                torch.load(f"{params.lang_dir}/HLG.pt", map_location=device)
            )
            assert HLG.requires_grad is False

            if not hasattr(HLG, "lm_scores"):
                HLG.lm_scores = HLG.scores.clone()
        
        # For BPE mode, create a simple word table from tokens
        if "lang_bpe" in str(params.lang_dir):
            # Read tokens and create a simple word table mapping
            tokens_file = params.lang_dir / "tokens.txt"
            if tokens_file.exists():
                word_table = {}
                with open(tokens_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                token, idx = parts[0], parts[1]
                                word_table[token] = int(idx)
            else:
                word_table = None
        else:
            # Phone mode: use lexicon word table
            lexicon = Lexicon(params.lang_dir)
            word_table = lexicon.word_table
        

        
        # Use validation-specific decoding parameters
        if params.validation_decoding_method == "greedy":
            logging.info("Starting decode_dataset with GREEDY decoding...")
            # Override beam parameters for greedy decoding
            original_search_beam = params.search_beam
            original_output_beam = params.output_beam
            params.search_beam = 1.0  # Greedy = beam size 1
            params.output_beam = 1.0
        else:
            logging.info(f"Starting decode_dataset with BEAM search (search_beam={params.validation_search_beam}, output_beam={params.validation_output_beam})...")
            # Use validation-specific beam parameters
            original_search_beam = params.search_beam
            original_output_beam = params.output_beam
            params.search_beam = params.validation_search_beam
            params.output_beam = params.validation_output_beam
        
        try:
            results_dict = decode_dataset(
                dl=valid_dl,
                params=params,
                model=model,
                rnn_lm_model=None,  # For CTC validation, we don't use RNN LM
                HLG=HLG,
                H=H,
                bpe_model=bpe_model,
                word_table=word_table,
                sos_id=sos_id,
                eos_id=eos_id,
            )
            
        except Exception as e:
            logging.error(f"decode_dataset failed: {e}")
            logging.error("Skipping WER computation for this validation")
            # Restore original beam parameters
            if params.validation_decoding_method == "greedy":
                params.search_beam = original_search_beam
                params.output_beam = original_output_beam
            else:
                params.search_beam = original_search_beam
                params.output_beam = original_output_beam
            
            logging.info(f"Validation loss: {loss_value:.4f}")
            return tot_loss, None
        
        # Restore original beam parameters
        if params.validation_decoding_method == "greedy":
            params.search_beam = original_search_beam
            params.output_beam = original_output_beam
        else:
            params.search_beam = original_search_beam
            params.output_beam = original_output_beam
        
        logging.info("Starting save_results...")
        
        try:
            wer_results = save_results(params=params, test_set_name=f"epoch_{epoch}_validation", results_dict=results_dict)
        except Exception as e:
            logging.error(f"save_results failed: {e}")
            logging.error("Skipping WER computation due to save_results error")
            logging.info(f"Validation loss: {loss_value:.4f}")
            return tot_loss, None
        
        # Log WER results
        if wer_results:
            for method, wer_value in wer_results.items():
                logging.info(f"Dataset-level WER ({method}): {wer_value:.2f}% (total errors/total words)")
                # Log each WER method to TensorBoard
                if rank == 0 and tb_writer is not None:
                    tb_writer.add_scalar(f"validation/wer_{method}", wer_value, params.batch_idx_train)
        else:
            logging.info("Validation WER: N/A")
        
        # Log some example predictions vs ground truth for inspection
        log_prediction_examples(results_dict, max_examples=3)
        
        # Log examples to TensorBoard if available
        if rank == 0 and tb_writer is not None:
            log_validation_examples_to_tensorboard(results_dict, tb_writer, params.batch_idx_train, max_examples=5)
        
        # Calculate overall WER statistics if we have results
        overall_wer = None
        if wer_results:
            # Find the main WER method (usually the first one or the one with 'wer' in the name)
            main_wer_key = None
            for key in wer_results.keys():
                if 'wer' in key.lower() or 'word_error_rate' in key.lower():
                    main_wer_key = key
                    break
            
            if main_wer_key is None and wer_results:
                # If no specific WER key found, use the first one
                main_wer_key = list(wer_results.keys())[0]
            
            if main_wer_key:
                overall_wer = wer_results[main_wer_key]
                logging.info(f"Main dataset-level WER ({main_wer_key}): {overall_wer:.2f}% (total errors/total words)")
                # Log the main/total WER to TensorBoard
                if rank == 0 and tb_writer is not None:
                    tb_writer.add_scalar("validation/total_wer", overall_wer, params.batch_idx_train)
                    tb_writer.add_scalar("validation/wer_dataset_level", overall_wer, params.batch_idx_train)
        
        # Final logging of validation results
        logging.info(f"Validation loss: {loss_value:.4f}")
        if overall_wer is not None:
            logging.info(f"Total validation WER: {overall_wer:.2f}% (dataset-level)")
            # Log the final total WER to TensorBoard
            if rank == 0 and tb_writer is not None:
                tb_writer.add_scalar("validation/loss", loss_value, params.batch_idx_train)
                tb_writer.add_scalar("validation/total_wer", overall_wer, params.batch_idx_train)
        else:
            logging.info("Validation WER: N/A")

        return tot_loss, overall_wer


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
        
        logging.info(f"Epoch {params.cur_epoch} - Training mode: {params.proj_layer_training}")
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
        
        # Debug: Check parameter updates for "only-proj" mode
        if params.proj_layer_training == "only-proj" and batch_idx % 1000 == 0 and rank == 0:
            actual_model = model.module if hasattr(model, 'module') else model
            if hasattr(actual_model, 'proj_layer') and actual_model.use_proj_layer:
                # Check if projection layer parameters have gradients
                proj_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                                  for p in actual_model.proj_layer.parameters() if p.requires_grad)
                
                # Check other parameters don't have gradients (should be None or zero)
                other_params_with_grad = 0
                for name, param in actual_model.named_parameters():
                    if 'proj_layer' not in name and param.requires_grad and param.grad is not None:
                        if param.grad.abs().sum() > 0:
                            other_params_with_grad += 1
                
                logging.info(f"Step {params.batch_idx_train}: proj_layer has grad: {proj_has_grad}, "
                           f"other params with grad: {other_params_with_grad}")
        
        # Update EMA teacher model after optimizer step
        if ema_teacher is not None and params.batch_idx_train >= params.ema_start_step:
            ema_teacher.update(model)
            if params.batch_idx_train % 1000 == 0:  # Log every 1000 steps instead of 100
                logging.info(f"EMA teacher updated at step {params.batch_idx_train}")

        if batch_idx % params.log_interval == 0:
            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}"
            )

        if batch_idx % params.log_interval == 0:
            if tb_writer is not None:
                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)

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
    logging.info(f"Projection layer training mode: {params.proj_layer_training}")
    logging.info(f"Self-distillation enabled: {params.enable_self_distillation}")
    if params.proj_layer_training == "only-proj":
        logging.info("→ Only projection layer parameters will be updated")
        logging.info("→ All other parameters will be frozen")
    elif params.proj_layer_training == "full-finetuning":
        logging.info("→ All model parameters will be updated")
    logging.info("=" * 60)
    
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
    
    logging.info(f"Model creation parameters: distill_layers={distill_layers}, knowledge_type={params.knowledge}")

    model = Conformer(
        num_features=params.feature_dim,
        num_classes=num_classes,
        nhead=params.nhead,
        subsampling_factor=params.subsampling_factor,
        vgg_frontend=False,
        use_feat_batchnorm=params.use_feat_batchnorm,
        use_proj_layer=params.use_proj_layer,
        distill_layers=distill_layers,
    )
    model.to(device)
    
    # Configure parameter training based on proj_layer_training setting
    if params.proj_layer_training == "only-proj":
        logging.info("Setting up projection layer only training - freezing all parameters except projection layer")
        
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze only projection layer parameters
        if hasattr(model, 'proj_layer') and model.use_proj_layer:
            for param in model.proj_layer.parameters():
                param.requires_grad = True
            logging.info("Projection layer parameters unfrozen for training")
        else:
            logging.warning("Projection layer not found or disabled! No parameters will be trained.")
        
        # Unfreeze distillation projection heads
        if hasattr(model, 'distill_projection_heads') and model.distill_projection_heads:
            for param in model.distill_projection_heads.parameters():
                param.requires_grad = True
            logging.info(f"Distillation projection heads parameters unfrozen for training (layers: {distill_layers})")
            
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
    elif params.proj_layer_training == "full-finetuning":
        logging.info("Setting up full fine-tuning - all parameters will be trained")
        
        # Ensure all parameters are trainable (default behavior)
        for param in model.parameters():
            param.requires_grad = True
            
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"All parameters are trainable: {trainable_params:,}")
        
    else:
        raise ValueError(f"Unknown proj_layer_training mode: {params.proj_layer_training}")
    
    logging.info(f"Model created: distill_layers={model.distill_layers if hasattr(model, 'distill_layers') else 'NOT FOUND'}")
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Get trainable parameters for optimizer (important for "only-proj" mode)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found! Check proj_layer_training configuration.")
    
    logging.info(f"Optimizer will update {len(trainable_params)} parameter tensors")
        
    # Create optimizer and scheduler based on scheduler_type
    logging.info(f"Scheduler type: {params.scheduler_type}")
    logging.info(f"Base LR: {getattr(params, 'base_lr', 'NOT SET')}")
    
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
        base_lr = getattr(params, 'base_lr', 2e-5)  # Default fallback
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

    librispeech = LibriSpeechAsrDataModule(args)

    if params.full_libri:
        train_cuts = librispeech.train_all_shuf_cuts()
    else:
        train_cuts = librispeech.train_clean_100_cuts()

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
    train_dl = librispeech.train_dataloaders(train_cuts, shuffle=True)

    # Use only dev_clean for faster validation (dev_other can be added later)
    valid_cuts = librispeech.dev_clean_cuts()
    # valid_cuts += librispeech.dev_other_cuts()  # Comment out for faster validation
    valid_dl = librispeech.valid_dataloaders(valid_cuts)
    
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
    ema_teacher = None
    if params.enable_self_distillation:
        logging.info(f"Initializing EMA teacher model with decay={params.ema_decay}, start_step={params.ema_start_step}")
        ema_teacher = EMATeacher(model, decay=params.ema_decay, device=device)    

    checkpoints = load_checkpoint_if_available(
        params=params, 
        model=model, 
        ema_teacher=ema_teacher
    )
    
    # Re-apply parameter freezing after checkpoint loading (checkpoint might overwrite requires_grad)
    if params.proj_layer_training == "only-proj":
        logging.info("Re-applying projection layer only training after checkpoint loading")
        
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze only projection layer parameters
        actual_model = model.module if hasattr(model, 'module') else model
        if hasattr(actual_model, 'proj_layer') and actual_model.use_proj_layer:
            for param in actual_model.proj_layer.parameters():
                param.requires_grad = True
            logging.info("Projection layer parameters re-unfrozen after checkpoint loading")
        
        # Verify parameter state after checkpoint loading
        trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params_count = sum(p.numel() for p in model.parameters())
        logging.info(f"After checkpoint loading - Trainable: {trainable_params_count:,} / {total_params_count:,}")
    
    # Load optimizer state from checkpoint if available
    if checkpoints and "optimizer" in checkpoints: 
        try:
            optimizer.load_state_dict(checkpoints["optimizer"])
            logging.info("Successfully loaded optimizer state from checkpoint")
        except (ValueError, KeyError, RuntimeError) as e:
            logging.warning(f"Failed to load optimizer state: {e}")
            logging.warning("Starting with fresh optimizer state")
            # Continue training with fresh optimizer state
    

    for epoch in range(params.start_epoch, params.num_epochs):
        fix_random_seed(params.seed + epoch)
        train_dl.sampler.set_epoch(epoch)

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
