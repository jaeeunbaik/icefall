#!/usr/bin/env python3
# Copyright    2021-2024  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Mingshuang Luo,
#                                                       Zengwei Yao,
#                                                       Yifan Yang,
#                                                       Daniel Povey)
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
SSL (Self-Supervised Learning) training for encoder-only mode using LibriLight dataset.

Usage:
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./conformer_ctc_sd_proj/train_enc.py \
  --world-size 4 \
  --num-epochs 100 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir ./conformer_ctc_sd_proj/exp_ssl \
  --max-duration 200 \
  --accum-grad 4 \
  --learning-type encoder-only
"""

import argparse
import copy
import logging
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from asr_datamodule import LibriLightAsrDataModule
from conformer import Conformer
from ema_teacher import EMATeacher
from k_means_clustering import PrototypeKMeansManager
from transformer import Noam

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
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
        default=100,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )


    parser.add_argument(
        "--exp-dir",
        type=str,
        default="./conformer_ctc_sd_proj/exp_ssl",
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
        "--base-lr",
        type=float,
        default=2e-5,
        help="Base learning rate for plateau and constant schedulers",
    )

    parser.add_argument(
        "--warmup-batches",
        type=float,
        default=30000,
        help="Eden warmup steps",
    )

    parser.add_argument(
        "--warmup-start",
        type=float,
        default=0,
        help="Eden warmup start learning rate",
    )

    parser.add_argument(
        "--ref-duration",
        type=float,
        default=600,
        help="Reference batch duration for purposes of adjusting batch counts for setting various "
        "schedules inside the model",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--sanity-check",
        type=str2bool,
        default=False,
        help="Check if any of the batches in epoch 1 would cause OOM.",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=100000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 1.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=30,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=30,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=200,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--accum-grad",
        type=int,
        default=4,
        help="""update gradient when batch_idx_train % accum_grad == 0.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    # SSL-specific parameters (Noise Mixing)
    parser.add_argument(
        "--enable-musan",
        type=str2bool,
        default=True,
        help="Enable MUSAN noise augmentation for SSL noise mixing",
    )

    parser.add_argument(
        "--musan-ratio",
        type=float,
        default=0.5,
        help="Probability of applying MUSAN noise augmentation",
    )

    parser.add_argument(
        "--snr-range",
        type=str,
        default="10,20",
        help="SNR range for noise mixing (min,max)",
    )

    # Model parameters
    parser.add_argument(
        "--attention-dim",
        type=int,
        default=256,
        help="Hidden dim for multi-head attention model.",
    )

    parser.add_argument(
        "--nhead",
        type=int,
        default=4,
        help="Number of heads of multi-head attention model.",
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
        "--feature-dim",
        type=int,
        default=80,
        help="The model input dim. It has to match the one used in computing features.",
    )

    parser.add_argument(
        "--subsampling-factor",
        type=int,
        default=4,
        help="The subsampling factor for the model.",
    )

    parser.add_argument(
        "--use-feat-batchnorm",
        type=str2bool,
        default=True,
        help="Normalization for the input features.",
    )

    # EMA Teacher Model Arguments
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay rate for teacher model updates.",
    )

    parser.add_argument(
        "--ema-start-step",
        type=int,
        default=1000,
        help="Step number to start EMA teacher model updates.",
    )

    # Self-distillation arguments
    parser.add_argument(
        "--enable-self-distillation",
        type=str2bool,
        default=True,
        help="Enable self-distillation training",
    )

    parser.add_argument(
        "--distill-layers",
        type=str,
        default="6",
        help="Which encoder layer(s) to use for distillation (0-based).",
    )

    parser.add_argument(
        "--distill-loss-type",
        type=str,
        default="mse",
        choices=["mse", "cos", "kl"],
        help="Type of loss for self-distillation",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Weight for self-distillation loss.",
    )

    parser.add_argument(
        "--use-proj-layer",
        type=str2bool,
        default=True,
        help="Whether to use projection layer for self-distillation.",
    )

    parser.add_argument(
        "--learning-type",
        default="encoder-only",
        choices=["encoder-only"],
        help="Training method - fixed to encoder-only for SSL training",
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

    parser.add_argument(
        "--clean-ratio",
        type=float,
        default=0.3,
        help="Ratio of clean samples in training data",
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
        default="./prototypes",
        help="Directory to save/load prototypes.",
    )

    parser.add_argument(
        "--num-prototypes",
        type=int,
        default=256,
        help="Number of prototypes per layer for KL-based distillation",
    )

    parser.add_argument(
        "--prototype-samples",
        type=int,
        default=100000,
        help="Number of feature samples per layer for prototype initialization",
    )

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters."""
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "sub_batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 3000,
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
            "env_info": get_env_info(),
        }
    )

    return params


def get_adjusted_batch_count(params: AttributeDict) -> float:
    # returns the number of batches we would have used so far if we had used the reference
    # duration.  This is for purposes of set_batch_count().
    return (
        params.batch_idx_train
        * params.accum_grad
        * (params.max_duration * params.world_size)
        / params.ref_duration
    )


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema_teacher: Optional[EMATeacher] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file."""
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    if not filename.exists():
        logging.warning(f"Checkpoint not found at {filename}")
        return None

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=None,
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

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

    # Try to load EMA teacher checkpoint if it exists
    if ema_teacher is not None:
        ema_filename = filename.parent / f"{filename.stem}-ema-teacher.pt"
        if ema_filename.exists():
            try:
                ema_state_dict = torch.load(ema_filename, map_location='cpu')
                ema_teacher.load_state_dict(ema_state_dict)
                logging.info(f"Loaded EMA teacher checkpoint from {ema_filename}")
            except Exception as e:
                logging.warning(f"Failed to load EMA teacher checkpoint: {e}")
        else:
            logging.info("EMA teacher checkpoint not found, will initialize from student model")

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    sampler: Optional[CutSampler] = None,
    scaler: Optional[GradScaler] = None,
    ema_teacher: Optional[EMATeacher] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file."""
    if rank != 0:
        return
        
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=None,
        sampler=sampler,
        scaler=scaler,
        rank=rank,
    )

    # Save EMA teacher model separately if it exists
    if ema_teacher is not None:
        ema_filename = params.exp_dir / f"epoch-{params.cur_epoch}-ema-teacher.pt"
        torch.save(ema_teacher.state_dict(), ema_filename)
        logging.info(f"EMA teacher checkpoint saved to {ema_filename}")

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)


def compute_ssl_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: dict,
    is_training: bool,
    ema_teacher: Optional[EMATeacher] = None,
    prototype_manager: Optional[PrototypeKMeansManager] = None,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute SSL (Self-Supervised Learning) loss using noise mixing and self-distillation.
    
    Args:
      params: Parameters for training
      model: The model for training
      batch: A batch of data from CleanNoisyWrapper with clean/noisy pairs
      is_training: True for training, False for validation
      ema_teacher: EMA teacher model for self-distillation
      prototype_manager: For prototype-based KL divergence
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    
    # Get clean and noisy features from CleanNoisyWrapper
    if 'clean' in batch and 'noisy' in batch:
        # Using CleanNoisyWrapper - guaranteed clean/noisy pairs
        clean_feature = batch["clean"]["inputs"].to(device)  # (N, T, C)
        noisy_feature = batch["noisy"]["inputs"].to(device)  # (N, T, C)
        supervisions = batch["supervisions"]
        
        logging.debug("Using CleanNoisyWrapper: clean/noisy pairs guaranteed aligned")
    else:
        # Fallback: treat inputs as clean, apply on-the-fly noise
        clean_feature = batch["inputs"].to(device)  # (N, T, C)
        noisy_feature = clean_feature.clone()  # For now, same as clean
        supervisions = batch["supervisions"]
        
        logging.warning("CleanNoisyWrapper not used, using fallback clean=noisy mode")
    
    # at entry, features are (N, T, C)
    assert clean_feature.ndim == 3
    assert noisy_feature.ndim == 3
    
    with torch.set_grad_enabled(is_training):
        # Student model: Forward pass with noisy input
        student_output, encoder_memory, memory_mask, student_hiddens, att_maps = model(noisy_feature, supervisions)
        
        # Self-distillation loss computation
        distillation_loss = torch.tensor(0.0, device=device)
        
        if params.enable_self_distillation:
            # Teacher model: Forward pass with clean input
            if ema_teacher is not None and params.batch_idx_train >= params.ema_start_step:
                with torch.no_grad():
                    # EMA teacher forward pass with clean features
                    teacher_output, teacher_encoder_memory, _, teacher_hiddens, _ = ema_teacher.model(clean_feature, supervisions)
            else:
                # Use student model with clean features as teacher
                with torch.no_grad():
                    teacher_output, teacher_encoder_memory, _, teacher_hiddens, _ = model(clean_feature, supervisions)
            
            # Parse distillation layers
            try:
                distill_layers = [int(x.strip()) for x in params.distill_layers.split(',')]
            except:
                distill_layers = [int(params.distill_layers)]
            
            # Compute distillation loss between student (noisy) and teacher (clean) representations
            if params.use_proj_layer and len(distill_layers) > 0:
                distillation_loss = compute_distillation_loss(
                    student_hiddens=student_hiddens,
                    teacher_hiddens=teacher_hiddens,
                    distill_layers=distill_layers,
                    loss_type=params.distill_loss_type,
                    memory_mask=memory_mask,
                    prototype_manager=prototype_manager,
                )
    
    # For SSL training, the main loss is the distillation loss
    total_loss = distillation_loss
    
    # Metrics tracking
    info = MetricsTracker()
    info["frames"] = clean_feature.size(0) * clean_feature.size(1)  # Total frames
    info["distill_loss"] = distillation_loss.detach().cpu().item()
    info["loss"] = total_loss.detach().cpu().item()
    
    # averaged input duration in frames over utterances
    info["utterances"] = clean_feature.size(0)
    if 'num_frames' in supervisions:
        info["utt_duration"] = supervisions["num_frames"].sum().item()
        # averaged padding proportion over utterances
        info["utt_pad_proportion"] = (
            ((clean_feature.size(1) - supervisions["num_frames"]) / clean_feature.size(1)).sum().item()
        )
    else:
        info["utt_duration"] = clean_feature.size(0) * clean_feature.size(1)
        info["utt_pad_proportion"] = 0.0

    return total_loss, info



def compute_distillation_loss(
    student_hiddens: list,
    teacher_hiddens: list,
    distill_layers: list,
    loss_type: str,
    memory_mask: Optional[Tensor] = None,
    prototype_manager: Optional[PrototypeKMeansManager] = None,
) -> Tensor:
    """
    Compute distillation loss between student and teacher hidden states.
    """
    total_loss = torch.tensor(0.0, device=student_hiddens[0].device)
    
    for layer_idx in distill_layers:
        if layer_idx >= len(student_hiddens) or layer_idx >= len(teacher_hiddens):
            continue
            
        student_hidden = student_hiddens[layer_idx]  # (T, N, C)
        teacher_hidden = teacher_hiddens[layer_idx]  # (T, N, C)
        
        # Ensure same dimensions
        if student_hidden.shape != teacher_hidden.shape:
            min_seq_len = min(student_hidden.size(0), teacher_hidden.size(0))
            student_hidden = student_hidden[:min_seq_len]
            teacher_hidden = teacher_hidden[:min_seq_len]
        
        if loss_type == "mse":
            layer_loss = torch.nn.functional.mse_loss(student_hidden, teacher_hidden)
        elif loss_type == "cos":
            # Cosine similarity loss
            cos_sim = torch.nn.functional.cosine_similarity(
                student_hidden.flatten(0, 1), 
                teacher_hidden.flatten(0, 1), 
                dim=-1
            )
            layer_loss = 1.0 - cos_sim.mean()
        elif loss_type == "kl" and prototype_manager is not None:
            # Prototype-based KL divergence
            layer_loss = prototype_manager.compute_kl_loss(
                student_hidden, teacher_hidden, layer_idx
            )
        else:
            # Default to MSE
            layer_loss = torch.nn.functional.mse_loss(student_hidden, teacher_hidden)
        
        total_loss += layer_loss
    
    # Average over layers
    if len(distill_layers) > 0:
        total_loss = total_loss / len(distill_layers)
    
    return total_loss


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    valid_dl: torch.utils.data.DataLoader,
    ema_teacher: Optional[EMATeacher] = None,
    prototype_manager: Optional[PrototypeKMeansManager] = None,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_ssl_loss(
            params=params,
            model=model,
            batch=batch,
            is_training=False,
            ema_teacher=ema_teacher,
            prototype_manager=prototype_manager,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    ema_teacher: Optional[EMATeacher] = None,
    prototype_manager: Optional[PrototypeKMeansManager] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch."""
    model.train()

    tot_loss = MetricsTracker()

    saved_bad_model = False

    def save_bad_model(suffix: str = ""):
        save_checkpoint_impl(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            model_avg=model_avg,
            params=params,
            optimizer=optimizer,
            scheduler=None,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=0,
        )

    for sub_batch_idx, batch in enumerate(train_dl):
        params.sub_batch_idx_train += 1
        batch_idx = sub_batch_idx // params.accum_grad

        if batch_idx % 10 == 0:
            set_batch_count(model, get_adjusted_batch_count(params))

        batch_size = len(batch["supervisions"]["text"]) if "text" in batch["supervisions"] else batch["inputs"].size(0)

        try:
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, loss_info = compute_ssl_loss(
                    params=params,
                    model=model,
                    batch=batch,
                    is_training=True,
                    ema_teacher=ema_teacher,
                    prototype_manager=prototype_manager,
                )
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            scaler.scale(loss / params.accum_grad).backward()

            if sub_batch_idx % params.accum_grad == params.accum_grad - 1:
                params.batch_idx_train += 1

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update EMA teacher model after optimizer step
                if ema_teacher is not None and params.batch_idx_train >= params.ema_start_step:
                    ema_teacher.update(model)
            else:
                continue

        except:  # noqa
            save_bad_model()
            display_and_save_batch(batch, params=params)
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=None,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        if batch_idx % 100 == 0 and params.use_fp16:
            # If the grad scale was less than 1, try increasing it
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 8.0 or (cur_grad_scale < 32.0 and batch_idx % 400 == 0):
                scaler.update(cur_grad_scale * 2.0)
            if cur_grad_scale < 0.01:
                if not saved_bad_model:
                    save_bad_model(suffix="-first-warning")
                    saved_bad_model = True
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                raise RuntimeError(
                    f"grad_scale is too small, exiting: {cur_grad_scale}"
                )

        if batch_idx % params.log_interval == 0:
            # Get current learning rate
            if hasattr(optimizer, '_rate'):
                cur_lr = optimizer._rate
            else:
                cur_lr = optimizer.param_groups[0]['lr']
            
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale", cur_grad_scale, params.batch_idx_train
                    )

        if batch_idx % params.valid_interval == 0 and not params.print_diagnostics:
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                valid_dl=valid_dl,
                ema_teacher=ema_teacher,
                prototype_manager=prototype_manager,
                world_size=world_size,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

    if sub_batch_idx % params.accum_grad != params.accum_grad - 1:
        optimizer.zero_grad()
        
    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def display_and_save_batch(batch: dict, params: AttributeDict) -> None:
    """Display the batch statistics and save the batch into disk."""
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)

    if "inputs" in batch:
        inputs = batch["inputs"]
        logging.info(f"inputs shape: {inputs.shape}")


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        try:
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, _ = compute_ssl_loss(
                    params=params,
                    model=model,
                    batch=batch,
                    is_training=True,
                )
            loss.backward()
            optimizer.zero_grad()
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            display_and_save_batch(batch, params=params)
            raise
        logging.info(
            f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
        )


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
    logging.info("SSL Training started")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")
    logging.info(params)

    logging.info("About to create model")
    
    # Parse distill_layers argument
    distill_layers = params.distill_layers
    if isinstance(distill_layers, str):
        distill_layers = [int(x) for x in distill_layers.split(',') if x.strip()]

    # Create model - we'll use a dummy num_classes since we're not using CTC output
    model = Conformer(
        num_features=params.feature_dim,
        num_classes=1000,  # Dummy value since we're not using CTC
        nhead=params.nhead,
        subsampling_factor=params.subsampling_factor,
        vgg_frontend=False,
        use_feat_batchnorm=params.use_feat_batchnorm,
        use_proj_layer=params.use_proj_layer,
        distill_layers=distill_layers,
    )

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # Freeze CTC-related parameters for encoder-only training
    logging.info("=" * 60)
    logging.info("ENCODER-ONLY SSL MODE: Configuring parameter freezing")
    logging.info("=" * 60)
    logging.info("Freezing CTC output layers, keeping encoder trainable")
    
    # First, make all parameters trainable
    for param in model.parameters():
        param.requires_grad = True
    
    # Then freeze CTC-related layers
    if hasattr(model, 'ctc_output'):
        for param in model.ctc_output.parameters():
            param.requires_grad = False
        logging.info("Frozen: ctc_output")
    
    if hasattr(model, 'linear'):
        for param in model.linear.parameters():
            param.requires_grad = False
        logging.info("Frozen: linear")
        
    # Keep encoder and distillation components trainable
    unfrozen_components = ["encoder", "encoder_embed", "encoder_pos", "after_norm"]
    if params.use_proj_layer:
        unfrozen_components.append("proj_layer")
        unfrozen_components.append("distill_projection_heads")
    
    logging.info(f"Trainable components: {', '.join(unfrozen_components)}")
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params_count = total_params - trainable_params_count
    
    logging.info(f"Parameter Summary:")
    logging.info(f"  Total parameters: {total_params:,}")
    logging.info(f"  Trainable parameters: {trainable_params_count:,}")
    logging.info(f"  Frozen parameters: {frozen_params_count:,}")
    logging.info(f"  Trainable ratio: {trainable_params_count/total_params:.2%}")
    logging.info("=" * 60)

    model.to(device)

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    # Initialize EMA Teacher Model for self-distillation
    ema_teacher = None
    if params.enable_self_distillation:
        logging.info(f"Initializing EMA teacher model with decay={params.ema_decay}, start_step={params.ema_start_step}")
        ema_teacher = EMATeacher(model, decay=params.ema_decay, device=device)

    checkpoints = load_checkpoint_if_available(
        params=params, 
        model=model, 
        model_avg=model_avg,
        ema_teacher=ema_teacher
    )

    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Get trainable parameters for optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logging.info(f"Optimizer will update {len(trainable_params)} parameter tensors")

    # Use Noam optimizer
    optimizer = Noam(
        trainable_params,
        model_size=params.attention_dim,
        factor=5.0,  # lr factor
        warm_step=params.warmup_batches,
        weight_decay=1e-6,
    )

    # Load optimizer state from checkpoint if available
    if checkpoints and "optimizer" in checkpoints:
        try:
            optimizer.load_state_dict(checkpoints["optimizer"])
            logging.info("Successfully loaded optimizer state from checkpoint")
        except Exception as e:
            logging.warning(f"Failed to load optimizer state: {e}")

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(512)
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    # Create LibriLight data module
    librilight = LibriLightAsrDataModule(args)

    # Load LibriLight training data
    train_cuts = librilight.librilight_train_cuts()
    logging.info(f"Loaded LibriLight training cuts: {len(train_cuts)} utterances")

    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 20 seconds
        return 1.0 <= c.duration <= 20.0

    train_cuts = train_cuts.filter(remove_short_and_long_utt)

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    # Use CleanNoisyWrapper for training data to ensure clean/noisy pairs
    from asr_datamodule import CleanNoisyWrapper
    train_dl = librilight.train_dataloaders(train_cuts, shuffle=True)
    
    # Wrap training dataset with CleanNoisyWrapper for SSL training
    if params.enable_musan:
        logging.info("Wrapping training dataset with CleanNoisyWrapper for noise mixing SSL")
        train_dl.dataset = CleanNoisyWrapper(train_dl.dataset, snr_range=params.snr_range)

    # Use LibriSpeech dev-clean for validation
    valid_cuts = librilight.librilight_dev_cuts()
    valid_cuts = valid_cuts.filter(remove_short_and_long_utt)
    valid_dl = librilight.valid_dataloaders(valid_cuts)
    logging.info(f"Validation set size: {len(valid_cuts)} utterances")

    # Initialize prototype manager for KL-based distillation
    prototype_manager = None
    if params.distill_loss_type == "kl":
        try:
            distill_layers = [int(x.strip()) for x in params.distill_layers.split(',')]
        except:
            distill_layers = [int(params.distill_layers)]
        
        if distill_layers and rank == 0:
            logging.info(f"Initializing prototype manager for layers {distill_layers}")
            prototype_manager = PrototypeKMeansManager(
                layers=distill_layers,
                num_prototypes=params.num_prototypes,
                prototype_dir=params.prototype_dir,
            )
            
            # Initialize prototypes using teacher model
            if not prototype_manager.load_prototypes():
                logging.info("Initializing prototypes from teacher model features...")
                prototype_manager.initialize_prototypes(
                    model=ema_teacher.model if ema_teacher else model,
                    train_dl=train_dl,
                    num_samples=params.prototype_samples,
                    device=device,
                )

    if params.sanity_check and not params.print_diagnostics:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            params=params,
        )

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            ema_teacher=ema_teacher,
            prototype_manager=prototype_manager,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            sampler=train_dl.sampler,
            scaler=scaler,
            ema_teacher=ema_teacher,
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    LibriLightAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()