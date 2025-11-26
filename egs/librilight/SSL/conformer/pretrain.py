#!/usr/bin/env python3
# Copyright    2021-2024  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Mingshuang Luo,
#                                                       Zengwei Yao,
#                                                       Yifan Yang,
#                                                       Daniel Povey)
#
# Copyright    2024  Shanghai Jiao Tong University  (authors: Jianheng Zhuo)
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

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# For hubert model pretraining:
./zipformer/pretrain.py \
  --world-size 8 \
  --num-epochs 400 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --max-duration 87.5 \
  --accum-grad 4
"""


import argparse
import copy
import logging
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import optim
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from optim import Eden, ScaledAdam
from ssl_datamodule import LibriLightDataModule
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

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
    get_parameter_groups_with_lrs,
    setup_logger,
    str2bool,
)

from conformer import Conformer
from k_means_clustering import create_prototype_manager
from typing import Optional as _Optional


# Lightweight EMATeacher for keeping an EMA copy of the student model.
# We keep it local to this script to avoid cross-package import issues.
class EMATeacher:
    def __init__(self, model: Union[nn.Module, DDP], decay: float = 0.999, device=None):
        # Keep a shadow copy of the model weights
        self.decay = float(decay)
        # copy the model architecture and weights
        self.shadow = copy.deepcopy(model.module if isinstance(model, DDP) else model)
        # Ensure the teacher doesn't compute gradients
        for p in self.shadow.parameters():
            p.requires_grad = False
        self.device = device
        if self.device is not None:
            try:
                self.shadow.to(self.device)
            except Exception:
                pass

    def get_teacher_model(self) -> nn.Module:
        return self.shadow

    def update(self, student_model: Union[nn.Module, DDP]):
        # Update shadow params by EMA of student params
        student = student_model.module if isinstance(student_model, DDP) else student_model
        with torch.no_grad():
            for s_param, t_param in zip(student.parameters(), self.shadow.parameters()):
                t_param.data.mul_(self.decay).add_(s_param.data.to(t_param.device) * (1.0 - self.decay))

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]

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



def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        default=400,
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
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help=(
            "Path to a pretrain checkpoint (.pt) to load as the initial model. "
            "If set, the model weights will be loaded from this file and, if a "
            "prototype manager is configured, prototypes can be initialized from it."
        ),
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.045, help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=7500,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=10.5,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--warmup-batches",
        type=float,
        default=5000,
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

    parser.add_argument(
        "--max-sample-size",
        type=float,
        default=250000,
        help="max sample size",
    )

    parser.add_argument(
        "--min-sample-size",
        type=float,
        default=16000,
        help="min sample size",
    )

    # Conformer / self-distillation related arguments (aligned with train.py)
    parser.add_argument(
        "--enable-self-distillation",
        type=str2bool,
        default=False,
        help="Enable self-distillation training between clean and noisy samples",
    )

    parser.add_argument(
        "--distill-layers",
        type=str,
        default="6",
        help="Which encoder layer(s) to use for distillation (0-based). Can be comma-separated list.",
    )

    parser.add_argument(
        "--distill-loss-type",
        type=str,
        default="kl",
        choices=["mse", "cos", "kl"],
        help="Type of loss for self-distillation. 'kl' recommended for distribution alignment.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Weight for self-distillation loss. Total loss = asr_loss + alpha * distill_loss",
    )

    parser.add_argument(
        "--distill-aggregation",
        type=str,
        default="layer_avg",
        choices=["layer_avg", "output_avg"],
        help="How to aggregate multi-layer distillation losses",
    )

    parser.add_argument(
        "--distill-temperature",
        type=float,
        default=1.0,
        help="Temperature for KL distillation (smoothing).",
    )

    parser.add_argument(
        "--layer-weights",
        type=str,
        default=None,
        help="Comma-separated weights for each distillation layer loss.",
    )

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
        help="Step number to start EMA teacher updates.",
    )

    parser.add_argument(
        "--use-proj-layer",
        type=str2bool,
        default=True,
        help="Whether to use projection layer between encoder and projector for self-distillation.",
    )

    parser.add_argument(
        "--learning-type",
        default="encoder-only",
        choices=["encoder-only", "hybrid", "asr"],
        help="Training mode: encoder-only | hybrid | asr",
    )

    parser.add_argument(
        "--clean-ratio",
        type=float,
        default=0.3,
        help="Fraction of clean data in mixed batches (if used).",
    )

    parser.add_argument(
        "--dataset-type",
        type=str,
        default="auto",
        choices=["auto", "librispeech", "librilight"],
        help="Dataset type to use. 'auto' selects LibriLight for encoder-only mode, LibriSpeech otherwise.",
    )

    parser.add_argument(
        "--prototype-dir",
        type=str,
        default="./prototypes",
        help="Directory to save/load prototypes for KL-based distillation.",
    )

    parser.add_argument(
        "--num-prototypes",
        type=int,
        default=256,
        help="Number of prototypes per layer for KL-based distillation.",
    )

    parser.add_argument(
        "--prototype-samples",
        type=int,
        default=100000,
        help="Number of feature samples per layer for prototype initialization.",
    )

    parser.add_argument(
        "--initialize-prototypes",
        type=str2bool,
        default=False,
        help="If true, run full K-means prototype initialization before training (rank 0).",
    )

    parser.add_argument(
        "--periodic-recluster-epochs",
        type=int,
        default=0,
        help="If >0, run full K-means re-clustering every N epochs on rank 0 (0 disables).",
    )

    parser.add_argument(
        "--prototype-samples-periodic",
        type=int,
        default=50000,
        help="Number of samples per layer for periodic re-clustering (smaller than initial).",
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
                           contains number of updates happen to the model so far across
                           epochs.

        - sub_batch_idx_train: It contains number of batch trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0
    """
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
            "valid_interval": 3000,  # For the 100h subset, use 800
            "env_info": get_env_info(),
        }
    )

    return params


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


def get_model(params: AttributeDict) -> nn.Module:


    # parse distill_layers if string
    distill_layers = params.distill_layers
    if isinstance(distill_layers, str):
        distill_layers = [int(x) for x in distill_layers.split(',') if x.strip()]

    model = Conformer(
        num_features=80,
        num_classes=1024,
        subsampling_factor=int(params.subsampling_factor) if hasattr(params, 'subsampling_factor') else 4,
        d_model=256,
        nhead=4,
        dim_feedforward=2048,
        num_encoder_layers=18,
        num_decoder_layers=0,
        dropout=0.1,
        cnn_module_kernel=31,
        normalize_before=True,
        vgg_frontend=False,
        use_feat_batchnorm=0.1,
        use_proj_layer=True,
        distill_layers=distill_layers,
        proj_dim=128,
        learning_type=getattr(params, 'learning_type', 'encoder-only'),
    )
    return model


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
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

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[CutSampler] = None,
    scaler: Optional[GradScaler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer used in the training.
      sampler:
       The sampler for the training dataset.
      scaler:
        The scaler used for mix precision training.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        scaler=scaler,
        rank=rank,
    )

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: dict,
    is_training: bool,
    ema_teacher: Optional[EMATeacher] = None,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Zipformer in our case.
      batch:
        A batch of data. See `dataset.HubertDataset()`
        for the content in it.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    # If self-distillation is enabled, expect the batch to carry clean/noisy pairs.
    if getattr(params, 'enable_self_distillation', False):
        # Support both wrapper styles: keys 'clean_audio'/'audio' or nested 'clean'/'noisy'
        if 'clean_audio' in batch and 'audio' in batch:
            clean_input = batch['clean_audio'].to(device)
            noisy_input = batch['audio'].to(device)
            padding_mask = batch.get('padding_mask')
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)
                # Downsample padding_mask to match encoder output
                # Conv2dSubsampling uses: output_T = ((T-1)//2 - 1)//2
                if padding_mask.dim() == 2:  # [B, T]
                    # Simulate the same subsampling as Conv2dSubsampling
                    # Two Conv2d layers with kernel_size=3, stride=2
                    # After first conv: (T-1)//2
                    # After second conv: ((T-1)//2 - 1)//2
                    
                    # First subsampling: stride=2, kernel_size=3
                    # Takes frames [0,1,2], [2,3,4], [4,5,6], ...
                    padding_mask = F.max_pool1d(
                        padding_mask.unsqueeze(1).float(),
                        kernel_size=3,
                        stride=2,
                        padding=1
                    ).squeeze(1).bool()
                    
                    # Second subsampling: stride=2, kernel_size=3
                    padding_mask = F.max_pool1d(
                        padding_mask.unsqueeze(1).float(),
                        kernel_size=3,
                        stride=2,
                        padding=1
                    ).squeeze(1).bool()
                    
                    # IMPORTANT: Invert the mask!
                    # padding_mask from dataset: True=padding, False=valid
                    # But loss computation needs: 1=valid, 0=padding
                    padding_mask = ~padding_mask  # Invert: False->True (valid), True->False (padding)
            supervisions = batch.get('supervisions', None)
        elif 'clean' in batch and 'noisy' in batch:
            clean_input = batch['clean']['inputs'].to(device)
            noisy_input = batch['noisy']['inputs'].to(device)
            # support nested supervisions
            supervisions = batch['clean'].get('supervisions', None)
            padding_mask = None
        else:
            raise ValueError("enable_self_distillation=True but batch does not contain clean/noisy pairs")

        # Unwrap DDP if needed
        actual_model = model.module if isinstance(model, DDP) else model

        # Compute student outputs under autograd control (enabled if is_training)
        with torch.set_grad_enabled(is_training):
            student_list = actual_model.get_intermediate_outputs(noisy_input, supervisions)

        # Compute teacher outputs without gradients. Prefer EMA teacher model when available
        use_ema = ema_teacher is not None and getattr(params, 'batch_idx_train', 0) >= int(getattr(params, 'ema_start_step', 0))
        teacher_model = ema_teacher.get_teacher_model() if ema_teacher else model
        teacher_model.eval()
        with torch.no_grad():
            teacher_list = teacher_model.get_intermediate_outputs(clean_input, supervisions)


        temp = params.distill_temperature
        # parse distill layer indices
        distill_layers = params.distill_layers
        if isinstance(distill_layers, str):
            parsed_layers = [int(x) for x in distill_layers.split(',') if x.strip()]
        elif isinstance(distill_layers, (list, tuple)):
            parsed_layers = [int(x) for x in distill_layers]
        else:
            parsed_layers = [int(distill_layers)]

        prototype_manager = params.prototype_manager

        layer_losses = []
        monitor_stats = {}

        for i, (t_out, s_out) in enumerate(zip(teacher_list, student_list)):
            layer_idx = parsed_layers[i]
            prototypes_ready = True
            # Convert to [B, T, D] for prototype manager APIs
            # t_out, s_out.shape : [T, B, D] -> t_btld, s_btld.shape: [B, T, D]
            t_btld = t_out.permute(1, 0, 2) if (t_out.dim() == 3) else t_out
            s_btld = s_out.permute(1, 0, 2) if (s_out.dim() == 3) else s_out

            used_proto = False
            if prototypes_ready and prototype_manager is not None and layer_idx is not None and layer_idx in prototype_manager.prototypes:
                try:
                    res = prototype_manager.compute_likelihood(
                        teacher_features=t_btld,
                        student_features=s_btld,
                        layer_idx=layer_idx,
                        frame_mask=padding_mask,
                    )
                    l = res['kl_loss']
                    layer_losses.append(l)
                    used_proto = True

                    # EMA update from teacher assignments when enabled and training
                    try:
                        if is_training and getattr(params, 'batch_idx_train', 0) >= int(getattr(params, 'ema_start_step', 0)):
                            prototype_manager.update_prototypes_ema(
                                layer_idx=layer_idx,
                                features=t_btld,
                                assignments=res['teacher_probs'],
                                momentum=float(getattr(params, 'ema_decay', 0.999)),
                                frame_mask=padding_mask,
                            )
                    except Exception as e:
                        logging.debug(f"Prototype EMA update failed for layer {layer_idx}: {e}")

                    # collect monitoring stats
                    try:
                        stats = prototype_manager.monitor_prototype_usage(
                            teacher_probs=res['teacher_probs'],
                            student_probs=res['student_probs'],
                            layer_idx=layer_idx,
                            frame_mask=padding_mask,
                        )
                        monitor_stats.update(stats)
                    except Exception:
                        pass
                except Exception as e:
                    logging.debug(f"Prototype-based likelihood failed for layer {layer_idx}: {e}")

            if not used_proto:
                # Fallback: KL across embedding dimension (feature-dim softmax)
                def _kl_feature_dim(tt, ss, temperature=temp):
                    tb = tt.permute(1, 0, 2) if (tt.dim() == 3) else tt
                    sb = ss.permute(1, 0, 2) if (ss.dim() == 3) else ss
                    
                    # Clamp features to prevent extreme values before softmax
                    tb = torch.clamp(tb, -10.0, 10.0)
                    sb = torch.clamp(sb, -10.0, 10.0)
                    
                    t_log_prob = F.log_softmax(tb / temperature, dim=-1)
                    s_log_prob = F.log_softmax(sb / temperature, dim=-1)
                    t_prob = t_log_prob.exp()
                    kl = torch.sum(t_prob * (t_log_prob - s_log_prob), dim=-1)  # [B, T]
                    
                    # Clamp KL to prevent extreme values
                    kl = torch.clamp(kl, 0.0, 10.0)
                    if padding_mask is not None:
                        valid = padding_mask
                        # Handle shape mismatch: padding_mask might have different T than kl
                        if valid.shape != kl.shape:
                            # Truncate or pad to match kl's temporal dimension
                            B, T_kl = kl.shape
                            B_v, T_v = valid.shape
                            if T_v > T_kl:
                                # Truncate padding mask
                                valid = valid[:, :T_kl]
                            elif T_v < T_kl:
                                # Pad padding mask with zeros (invalid frames)
                                padding = torch.zeros(B, T_kl - T_v, device=valid.device, dtype=valid.dtype)
                                valid = torch.cat([valid, padding], dim=1)
                        kl = kl * valid
                        denom = valid.sum()
                        return (kl.sum() / (denom + 1e-8)) * (temperature ** 2)
                    else:
                        return kl.mean() * (temperature ** 2)

                layer_losses.append(_kl_feature_dim(t_out, s_out, temperature=temp))

            # Aggregate per-layer losses
            if getattr(params, 'distill_aggregation', 'layer_avg') == 'layer_avg':
                total_loss = torch.stack(layer_losses).mean()
            else:
                total_loss = torch.stack(layer_losses).mean()

        

        info = MetricsTracker()
        info['frames'] = (clean_input.size(0) * clean_input.size(1)) if clean_input.dim() >= 2 else 1
        info['ctc_loss'] = 0.0
        info['att_loss'] = 0.0
        info['loss'] = total_loss.detach().cpu().item()
        # Add prototype monitoring stats if available
        if 'monitor_stats' in locals() and monitor_stats:
            for k, v in monitor_stats.items():
                info[k] = v
        return total_loss, info

    # Default (legacy) path: use model's forward which expects kmeans targets
    audio = batch["audio"].to(device)
    padding_mask = batch["padding_mask"].to(device) if "padding_mask" in batch else None
    kmeans = batch.get("kmeans")
    if kmeans is not None:
        kmeans = kmeans.to(device)

    with torch.set_grad_enabled(is_training):
        if kmeans is not None:
            loss, num_masked_tokens, logging_output = model(
                source=audio, target_list=[kmeans], padding_mask=padding_mask
            )
        else:
            # If no kmeans targets, try to run model normally and expect it returns a loss
            out = model(source=audio, padding_mask=padding_mask)
            # If model returns a tuple (loss, ...)
            if isinstance(out, tuple) and len(out) >= 1 and isinstance(out[0], torch.Tensor):
                loss = out[0]
                num_masked_tokens = out[1] if len(out) > 1 else 1
                logging_output = out[2] if len(out) > 2 else {}
            else:
                # As a last resort, create a zero loss
                loss = torch.tensor(0.0, device=device, requires_grad=is_training)
                num_masked_tokens = 1
                logging_output = {}

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = num_masked_tokens
    for item in logging_output:
        info[item] = logging_output[item]
    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(
            params=params,
            model=model,
            batch=batch,
            is_training=False,
            ema_teacher=getattr(params, 'ema_teacher', None),
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
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
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
      scheduler:
        The learning rate scheduler, we call step() every step.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      scaler:
        The scaler used for mix precision training.
      model_avg:
        The stored model averaged from the start of training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
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
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=0,
        )

    for sub_batch_idx, batch in enumerate(train_dl):
        params.sub_batch_idx_train += 1
        batch_idx = sub_batch_idx // params.accum_grad

        if batch_idx % 10 == 0:
            set_batch_count(model, get_adjusted_batch_count(params))

        batch_size = batch["audio"].shape[0]

        try:
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    batch=batch,
                    is_training=True,
                    ema_teacher=getattr(params, 'ema_teacher', None),
                )
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            scaler.scale(loss / params.accum_grad).backward()

            if sub_batch_idx % params.accum_grad == params.accum_grad - 1:
                params.batch_idx_train += 1
                scheduler.step_batch(params.batch_idx_train)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # Update EMA teacher after optimizer step if enabled
                if getattr(params, 'ema_teacher', None) is not None and getattr(params, 'batch_idx_train', 0) >= int(getattr(params, 'ema_start_step', 0)):
                    try:
                        params.ema_teacher.update(model)
                        if params.batch_idx_train % 1000 == 0:
                            logging.info(f"EMA teacher updated at step {params.batch_idx_train}")
                    except Exception as e:
                        logging.debug(f"EMA teacher update failed: {e}")
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
                scheduler=scheduler,
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
            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
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
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
            )

            # If prototype monitoring stats are present in the last loss_info/tot_loss,
            # print teacher-student similarity per configured layer to terminal.
            try:
                if getattr(params, 'prototype_manager', None) is not None:
                    distill_layers = params.distill_layers
                    if isinstance(distill_layers, str):
                        parsed_layers = [int(x) for x in distill_layers.split(',') if x.strip()]
                    elif isinstance(distill_layers, (list, tuple)):
                        parsed_layers = [int(x) for x in distill_layers]
                    else:
                        parsed_layers = [int(distill_layers)]

                    for li in parsed_layers:
                        key = f'layer_{li}/cosine_similarity'
                        val = None
                        if key in loss_info:
                            try:
                                val = float(loss_info[key])
                            except Exception:
                                val = loss_info[key]
                        elif key in tot_loss:
                            try:
                                val = float(tot_loss[key])
                            except Exception:
                                val = tot_loss[key]

                        if val is not None:
                            try:
                                logging.info(f"Prototype monitor - layer {li} teacher-student cosine: {val:.4f}")
                            except Exception:
                                logging.info(f"Prototype monitor - layer {li} teacher-student cosine: {val}")
            except Exception:
                pass

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

        if batch_idx > 0 and batch_idx % params.valid_interval == 0 and not params.print_diagnostics:
            # Skip validation if valid_dl is not provided
            if valid_dl is not None:
                logging.info("Computing validation loss")
                valid_info = compute_validation_loss(
                    params=params,
                    model=model,
                    valid_dl=valid_dl,
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

    if batch_idx % params.accum_grad != params.accum_grad - 1:
        optimizer.zero_grad()
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
    model = get_model(params)

    # Optionally create a PrototypeKMeansManager for prototype-based KL distillation
    params.prototype_manager = None
    if getattr(params, 'enable_self_distillation', False):
        # parse distill layers to integer list
        distill_layers = params.distill_layers
        if isinstance(distill_layers, str):
            try:
                parsed_layers = [int(x) for x in distill_layers.split(',') if x.strip()]
            except Exception:
                parsed_layers = [int(distill_layers)]
        elif isinstance(distill_layers, (list, tuple)):
            parsed_layers = [int(x) for x in distill_layers]
        else:
            parsed_layers = [int(distill_layers)]

        proto_mgr = create_prototype_manager(
            target_layers=parsed_layers,
            num_prototypes=params.num_prototypes,
            proj_dim=128,
            temperature=params.distill_temperature,
            save_dir=params.prototype_dir,
        )
        try:
            proto_mgr.load_prototypes()
            params.prototype_manager = proto_mgr
        except:
            params.prototype_manager.initialize_prototypes(
                teacher_model=model_for_proto,
                dataloader=train_dl,
                num_samples_per_layer=params.prototype_samples,
                kmeans_iterations=params.kmeans_iterations,
                save_prototypes=True,
                load_if_exists=load_if_exists_flag,
            )

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Create EMA teacher if requested and attach to params for downstream access
    ema_teacher = None
    if float(getattr(params, 'ema_decay', 0.0)) > 0.0:
        try:
            ema_teacher = EMATeacher(model, decay=float(getattr(params, 'ema_decay', 0.999)), device=device)
            logging.info("EMA teacher created")
        except Exception as e:
            logging.warning(f"Failed to create EMA teacher: {e}")
    params.ema_teacher = ema_teacher

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(model, lr=params.base_lr, include_names=True),
        lr=params.base_lr,  # should have no effect
        clipping_scale=5.0,  # Reduced from 2.0 for more aggressive clipping to prevent gradient explosion
    )

    scheduler = Eden(
        optimizer,
        params.lr_batches,
        params.lr_epochs,
        params.warmup_batches,
        params.warmup_start,
    )

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    # If user specified a pretrain checkpoint explicitly, load it into the model
    resume_from = getattr(params, 'resume_from', None)
    if resume_from is not None:
        try:
            resume_path = Path(resume_from)
            if resume_path.exists():
                logging.info(f"Loading pretrain checkpoint from {resume_path}")
                # Use load_checkpoint to load model weights into current model
                try:
                    loaded = load_checkpoint(resume_path, model=model, optimizer=None, scheduler=None)
                    logging.info(f"Loaded pretrain checkpoint: {resume_path}")
                    # mark that we loaded an external pretrain model
                    params._loaded_pretrain = True
                except Exception as e:
                    logging.warning(f"Failed to load pretrain checkpoint with load_checkpoint: {e}. Trying torch.load fallback.")
                    ck = torch.load(resume_path, map_location='cpu')
                    if isinstance(ck, dict) and 'model' in ck:
                        model.load_state_dict(ck['model'], strict=False)
                    elif isinstance(ck, dict) and 'state_dict' in ck:
                        model.load_state_dict(ck['state_dict'], strict=False)
                    else:
                        # try direct state dict
                        try:
                            model.load_state_dict(ck, strict=False)
                        except Exception as e2:
                            logging.warning(f"Fallback load_state_dict failed: {e2}")
                    params._loaded_pretrain = True
            else:
                logging.warning(f"resume-from path does not exist: {resume_path}")
        except Exception as e:
            logging.warning(f"Failed to load resume-from checkpoint: {e}")

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    librilight = LibriLightDataModule(args)

    # Select dataset based on --dataset-type argument
    dataset_type = getattr(params, 'dataset_type', 'librilight')
    logging.info(f"Using dataset: {dataset_type}")
    
    if dataset_type == "librispeech":
        logging.info("Loading LibriSpeech train-all-shuf cuts for pretraining")
        train_cuts = librilight.librispeech_train_all_shuf_cuts()
    elif dataset_type == "librilight":
        logging.info("Loading LibriLight medium cuts for pretraining")
        train_cuts = librilight.medium_cuts()
    elif dataset_type == "auto":
        # Auto mode: select based on learning_type
        if getattr(params, 'learning_type', 'encoder-only') == 'encoder-only':
            logging.info("Auto mode: using LibriLight (encoder-only training)")
            train_cuts = librilight.medium_cuts()
        else:
            logging.info("Auto mode: using LibriSpeech (hybrid/asr training)")
            train_cuts = librilight.librispeech_train_all_shuf_cuts()
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be 'librilight', 'librispeech', or 'auto'")

    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 20 seconds
        #
        # Caution: There is a reason to select 20.0 here. Please see
        # ../local/display_manifest_statistics.py
        #
        # You should use ../local/display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        if (
            c.duration < params.min_sample_size / 16000
            or c.duration > params.max_sample_size / 16000
        ):
            # logging.warning(
            #     f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
            # )
            return False

        return True

    train_cuts = train_cuts.filter(remove_short_and_long_utt)

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    train_dl = librilight.train_dataloaders(
        train_cuts,
        sample_rate=16000,
        label_rate=50,
        random_crop=params.random_crop,
        pad_audio=False,
        num_classes=256,
        do_normalize=params.do_normalize,
        sampler_state_dict=sampler_state_dict,
    )

    # If requested, initialize prototypes once before training (rank 0 performs K-means).
    # Also allow initializing prototypes from an explicitly provided pretrain model
    pretrain_requested = params.initialize_prototypes
    pretrain_loaded = getattr(params, '_loaded_pretrain', False)
    if (pretrain_requested or pretrain_loaded) and getattr(params, 'prototype_manager', None) is not None:
        if world_size > 1:
            torch.distributed.barrier()
        if params.initialize_prototypes:
            logging.info("Initializing prototypes (rank 0) before training...")
            try:
                model_for_proto = model.module if isinstance(model, DDP) else model
                model_for_proto.eval()
                # If prototypes are being created from a provided pretrain model,
                # force re-clustering (load_if_exists=False) so prototypes match the loaded weights.
                load_if_exists_flag = False if pretrain_loaded else True
                params.prototype_manager.initialize_prototypes(
                    teacher_model=model_for_proto,
                    dataloader=train_dl,
                    num_samples_per_layer=int(getattr(params, 'prototype_samples', 100000)),
                    kmeans_iterations=int(getattr(params, 'kmeans_iterations', 40)),
                    save_prototypes=True,
                    load_if_exists=load_if_exists_flag,
                )
            except Exception as e:
                logging.error(f"Prototype initialization failed: {e}")
        # synchronize and ensure other ranks load prototypes
        if world_size > 1:
            torch.distributed.barrier()
            try:
                params.prototype_manager.load_prototypes()
            except Exception:
                logging.info("Other ranks: no prototype files found after barrier (may be intentional)")

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
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        # Periodic full K-means re-clustering hook (rank 0 performs clustering)
        if getattr(params, 'periodic_recluster_epochs', 0) > 0:
            if epoch > 0 and epoch % int(getattr(params, 'periodic_recluster_epochs', 0)) == 0:
                if getattr(params, 'prototype_manager', None) is not None:
                    if world_size > 1:
                        torch.distributed.barrier()
                    if rank == 0:
                        logging.info(f"Periodic re-clustering prototypes at epoch {epoch} (rank 0)")
                        try:
                            model_for_proto = model.module if isinstance(model, DDP) else model
                            model_for_proto.eval()
                            params.prototype_manager.initialize_prototypes(
                                teacher_model=model_for_proto,
                                dataloader=train_dl,
                                num_samples_per_layer=params.prototype_samples, 
                                kmeans_iterations=params.kmeans_iterations,
                                save_prototypes=True,
                                load_if_exists=False,
                            )
                        except Exception as e:
                            logging.error(f"Periodic prototype re-clustering failed: {e}")
                    if world_size > 1:
                        torch.distributed.barrier()
                        try:
                            params.prototype_manager.load_prototypes()
                        except Exception:
                            logging.info("Other ranks: no prototype files found after periodic re-cluster (may be intentional)")

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=None,
            scaler=scaler,
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
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data. See `dataset.HubertDataset()`
        for the content in it.
      params:
        Parameters for training. See :func:`get_params`.
      sp:
        The BPE model.
    """
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)

    audio = batch["audio"]
    logging.info(f"audio shape: {audio.shape}")


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
                loss, _ = compute_loss(
                    params=params,
                    model=model,
                    batch=batch,
                    is_training=True,
                    ema_teacher=getattr(params, 'ema_teacher', None),
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


def main():
    parser = get_parser()
    LibriLightDataModule.add_arguments(parser)
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
