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
import warnings
import sys
from pathlib import Path
from shutil import copyfile
from typing import Optional, Tuple, List

import k2
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from conformer import Conformer
from lhotse.cut import Cut
from lhotse.utils import fix_random_seed
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from transformer import Noam

from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.checkpoint import load_checkpoint
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.graph_compiler import CtcTrainingGraphCompiler
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    encode_supervisions,
    setup_logger,
    str2bool,
)
from icefall.utils import get_texts
from icefall.decode import get_lattice, one_best_decoding
from icefall.utils import store_transcripts, write_error_stats
import sentencepiece as spm

# Global counter for validation sample logging
_VALIDATION_SAMPLE_COUNTER = 0


def log_prediction_examples(results_dict, max_examples=5, force_log=False):
    """
    Log a few examples of ground truth vs predicted text for validation inspection.
    Only logs to terminal every 50 validation samples to reduce clutter.
    
    Args:
        results_dict: Dictionary containing decoding results
        max_examples: Maximum number of examples to log
        force_log: Force logging regardless of sample counter
    """
    global _VALIDATION_SAMPLE_COUNTER
    
    if not results_dict:
        return
    
    # Get the first method's results (usually there's only one method in validation)
    first_method = list(results_dict.keys())[0]
    results = results_dict[first_method]
    
    if not results:
        return
    
    # Update the validation sample counter
    _VALIDATION_SAMPLE_COUNTER += len(results)
    
    # Only log to terminal every 50 samples (or when forced)
    should_log_to_terminal = force_log or (_VALIDATION_SAMPLE_COUNTER % 50 == 0) or (_VALIDATION_SAMPLE_COUNTER <= 50)
    
    if not should_log_to_terminal:
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
        
        # Log summary info only
        if valid_samples > 0:
            avg_example_wer = total_sample_wer / valid_samples
            logging.info(f"Validation batch processed: {valid_samples} samples "
                        f"(total samples processed: {_VALIDATION_SAMPLE_COUNTER}, detailed examples every 50 samples)")
        return
    
    # Full detailed logging when we hit the 50-sample threshold
    logging.info(f"Detailed validation examples (sample #{_VALIDATION_SAMPLE_COUNTER - len(results) + 1}-{_VALIDATION_SAMPLE_COUNTER}):")
    
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
        default=78,
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
        default="conformer_ctc/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lang_bpe_500",
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
        "--weight-decay",
        type=float,
        default=1e-6,
        help="""The weight decay for Noam lr scheduling
        """,
    )

    parser.add_argument(
        "--num-decoder-layers",
        type=int,
        default=6,
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
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    # Optional OOM sanity check
    parser.add_argument(
        "--sanity-check",
        type=str2bool,
        default=False,
        help="Run a pessimistic-batch OOM sanity check before training. It can take time; disable to skip.",
    )

    # Validation/decoding controls
    parser.add_argument(
        "--validation-decoding-method",
        type=str,
        default="greedy",
        choices=["greedy", "none"],
        help="Validation decoding method. 'greedy' prints GT vs PR and computes dataset-level WER.",
    )
    parser.add_argument(
        "--validation-print-samples",
        type=int,
        default=4,
        help="How many GT/PR pairs to log per validation run.",
    )
    parser.add_argument(
        "--validation-step-max-batches",
        type=int,
    default=-1,
    help="During step-wise validation, limit decoding to this many batches for speed. -1 means full validation set.",
    )
    parser.add_argument(
        "--validation-epoch-max-batches",
        type=int,
        default=-1,
        help="During end-of-epoch validation, limit decoding to this many batches. -1 means full validation set.",
    )
    parser.add_argument(
        "--warm-step",
        type=int,
        default=12000,
        help="warm step size for Noam scheduling",
    )

    # Debugging and diagnostics
    parser.add_argument(
        "--debug-train",
        type=str2bool,
        default=False,
        help="Enable extra diagnostics to investigate non-convergence (adds some overhead).",
    )
    parser.add_argument(
        "--debug-first-n-batches",
        type=int,
        default=3,
        help="When debug-train is true, log shapes/stats for the first N batches each epoch.",
    )

    parser.add_argument(
        "--init-model-from-pretrain",
        type=str,
        default="",
        help="""Path to pretrained model checkpoint (e.g., from SSL pretraining).
        If provided, the encoder weights will be loaded from this checkpoint.
        The decoder will be randomly initialized. Example:
        /path/to/librilight/SSL/conformer/exp/epoch-11.pt
        """,
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
            "full_libri": True,
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
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
            "beam_size": 20,
            "reduction": "sum",
            "use_double_scores": True,
            # parameters for Noam
            "weight_decay": 1e-6,
            "warm_step": 80000,
            "env_info": get_env_info(),
            # debugging
            "debug_train": False,
        }
    )

    return params


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> None:
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
    if params.start_epoch <= 0:
        return

    filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    saved_params = load_checkpoint(
        filename,
        model=model,
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

    return saved_params


def load_pretrained_encoder(
    pretrain_path: str,
    model: nn.Module,
) -> None:
    """Load pretrained encoder weights from SSL pretraining checkpoint.
    
    This function loads only the encoder weights from a pretrained checkpoint
    (e.g., from LibriLight SSL pretraining). The decoder remains randomly initialized.
    
    Args:
      pretrain_path:
        Path to the pretrained checkpoint file (e.g., epoch-11.pt from SSL training).
      model:
        The Conformer model to load encoder weights into.
    """
    if not pretrain_path or pretrain_path == "":
        return
    
    logging.info(f"Loading pretrained encoder from {pretrain_path}")

    # Add safe globals for weights_only=True
    from pathlib import PosixPath, WindowsPath
    torch.serialization.add_safe_globals([PosixPath, WindowsPath])

    # Also allow-list PrototypeKMeansManager if available to avoid weights_only errors
    try:
        try:
            from k_means_clustering import PrototypeKMeansManager  # SSL path
        except Exception:
            # Try likely local recipe path for the class
            candidate_paths = [
                str(Path(__file__).resolve().parent / 'conformer_ctc_sd_proj'),
                '/home/hdd2/jenny/ASRToolkit/icefall/egs/librilight/SSL/conformer',
                '/home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/conformer_ctc_sd_proj',
            ]
            for p in candidate_paths:
                if p not in sys.path:
                    sys.path.append(p)
            from k_means_clustering import PrototypeKMeansManager  # type: ignore
        # If import succeeded, register it
        torch.serialization.add_safe_globals([PrototypeKMeansManager])
    except Exception:
        # It's okay if we can't import; we'll still try to load weights only
        pass

    # Try to load checkpoint with weights_only=False to handle custom objects
    try:
        checkpoint = torch.load(pretrain_path, map_location="cpu", weights_only=False)
        logging.info("Checkpoint loaded successfully")
    except ModuleNotFoundError as e:
        logging.warning(f"ModuleNotFoundError: {e}")
        logging.warning("Attempting to load with weights_only=True (ignoring custom objects)")
        try:
            checkpoint = torch.load(pretrain_path, map_location="cpu", weights_only=True)
            logging.info("Checkpoint loaded successfully (weights only)")
        except Exception as e2:
            logging.error(f"Failed to load checkpoint: {e2}")
            raise
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        raise
    
    # Get the state dict - handle both DDP and non-DDP checkpoints
    # Rank-aware logging to avoid 4x spam in DDP
    def _get_rank() -> int:
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return torch.distributed.get_rank()
        except Exception:
            pass
        return 0

    rank = _get_rank()

    if "model" in checkpoint:
        pretrained_state = checkpoint["model"]
        if rank == 0:
            logging.info("Found 'model' key in checkpoint")
    else:
        pretrained_state = checkpoint
        if rank == 0:
            logging.info("Using checkpoint directly as state dict")
    
    # Filter encoder weights only
    encoder_state = {}
    for key, value in pretrained_state.items():
        # Remove 'module.' prefix if present (from DDP)
        clean_key = key.replace("module.", "")
        
        # Load encoder weights only (exclude decoder)
        if clean_key.startswith("encoder."):
            encoder_state[clean_key] = value
    
    if rank == 0:
        logging.info(f"Found {len(encoder_state)} encoder parameters in checkpoint")
    
    # Snapshot model state BEFORE loading
    model_state_before = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("encoder.")}
    
    # Load with strict=False to allow partial loading
    missing_keys, unexpected_keys = model.load_state_dict(encoder_state, strict=False)
    
    # Get model state AFTER loading
    model_state_after = model.state_dict()
    
    # Verify what was actually loaded by comparing weights
    actually_loaded = 0
    actually_changed = 0
    shape_mismatches = []
    skipped_keys = []
    
    # Collect concise samples instead of logging every single param
    sample_limit = 5  # keep logs short; show at most N examples per category
    samples_loaded_verified = []
    samples_loaded_matched = []
    samples_skipped_mismatch = []
    samples_skipped_missing = []

    for key, pretrained_value in encoder_state.items():
        if key in model_state_after:
            model_value_after = model_state_after[key]
            
            if pretrained_value.shape == model_value_after.shape:
                # Check if weights actually changed
                model_value_before = model_state_before.get(key)
                if model_value_before is not None:
                    # Compare actual weight values
                    weights_changed = not torch.equal(model_value_before, model_value_after)
                    weights_match_pretrained = torch.allclose(model_value_after, pretrained_value, rtol=1e-5, atol=1e-8)
                    
                    if weights_changed and weights_match_pretrained:
                        actually_loaded += 1
                        actually_changed += 1
                        if rank == 0 and len(samples_loaded_verified) < sample_limit:
                            samples_loaded_verified.append(f"{key} {tuple(pretrained_value.shape)}")
                    elif weights_match_pretrained:
                        actually_loaded += 1
                        if rank == 0 and len(samples_loaded_matched) < sample_limit:
                            samples_loaded_matched.append(f"{key} {tuple(pretrained_value.shape)}")
                    else:
                        if rank == 0 and len(samples_skipped_mismatch) < sample_limit:
                            samples_skipped_mismatch.append(f"{key} shape={tuple(pretrained_value.shape)}")
                        skipped_keys.append(key)
                else:
                    # No before state (shouldn't happen)
                    actually_loaded += 1
                    if rank == 0 and len(samples_loaded_verified) < sample_limit:
                        samples_loaded_verified.append(f"{key} {tuple(pretrained_value.shape)}")
            else:
                shape_mismatches.append(
                    f"{key}: pretrained {pretrained_value.shape} != model {model_value_after.shape}"
                )
                skipped_keys.append(key)
                if rank == 0 and len(samples_skipped_mismatch) < sample_limit:
                    samples_skipped_mismatch.append(f"{key} pretrained={tuple(pretrained_value.shape)} model={tuple(model_value_after.shape)}")
        else:
            if rank == 0 and len(samples_skipped_missing) < sample_limit:
                samples_skipped_missing.append(key)
            skipped_keys.append(key)
    
    # Report results
    if rank == 0:
        logging.info("=" * 80)
        logging.info(f"Pretrained Weight Loading Summary:")
        logging.info(f"  Total pretrained params: {len(encoder_state)}")
        logging.info(f"  Successfully loaded: {actually_loaded}")
        logging.info(f"  Weights changed from random init: {actually_changed}")
        logging.info(f"  Skipped (mismatch/missing): {len(skipped_keys)}")
        # Show a few examples per category (at most sample_limit each)
        if samples_loaded_verified:
            logging.info(f"  Examples LOADED & VERIFIED (<= {sample_limit}):")
            for s in samples_loaded_verified:
                logging.info(f"    - {s}")
        if samples_loaded_matched:
            logging.info(f"  Examples LOADED (already matched) (<= {sample_limit}):")
            for s in samples_loaded_matched:
                logging.info(f"    - {s}")
        if samples_skipped_mismatch:
            logging.info(f"  Examples SKIPPED (mismatch) (<= {sample_limit}):")
            for s in samples_skipped_mismatch:
                logging.info(f"    - {s}")
        if samples_skipped_missing:
            logging.info(f"  Examples SKIPPED (key not in model) (<= {sample_limit}):")
            for s in samples_skipped_missing:
                logging.info(f"    - {s}")
    
    # Check decoder remained random
    decoder_keys = [k for k in model_state_after.keys() if k.startswith("decoder.")]
    if decoder_keys and rank == 0:
        logging.info(f"  Decoder params (random init): {len(decoder_keys)}")
    
    if shape_mismatches:
        if rank == 0:
            logging.error("=" * 80)
            logging.error("CRITICAL: Shape mismatches detected!")
            # Print at most first few mismatches
            for msg in shape_mismatches[:sample_limit]:
                logging.error(f"  {msg}")
            if len(shape_mismatches) > sample_limit:
                logging.error(f"  ... and {len(shape_mismatches) - sample_limit} more")
            logging.error("These layers are using RANDOM INITIALIZATION!")
            logging.error("=" * 80)
        
        # Raise error if too many mismatches (likely wrong checkpoint)
        mismatch_ratio = len(shape_mismatches) / len(encoder_state)
        if mismatch_ratio > 0.5:
            raise ValueError(
                f"Too many shape mismatches ({len(shape_mismatches)}/{len(encoder_state)} = {mismatch_ratio:.1%})! "
                "You may be loading a checkpoint with different d_model or num_layers. "
                "Please check:\n"
                f"  1. Pretrained checkpoint: {pretrain_path}\n"
                f"  2. Model architecture: check d_model and num_layers\n"
                "Common issues:\n"
                "  - d_model mismatch (e.g., pretrained=256 vs current=512)\n"
                "  - Different number of layers\n"
                "  - Wrong checkpoint file"
            )
    
    if actually_loaded == 0:
        raise ValueError(
            "ERROR: No parameters were loaded from pretrained checkpoint! "
            "This usually means:\n"
            f"  1. All parameters have shape mismatches (check d_model)\n"
            f"  2. Wrong checkpoint format\n"
            f"  3. Checkpoint path: {pretrain_path}\n"
            f"Please verify the checkpoint is compatible with current model."
        )
    
    if actually_changed == 0 and len(encoder_state) > 0 and rank == 0:
        logging.warning("WARNING: Weights were 'loaded' but none changed from initialization!")
        logging.warning("This might indicate the checkpoint contains the same random init.")

    if rank == 0:
        logging.info(f"✓ Encoder: Loaded {actually_loaded} pretrained parameters ({actually_changed} changed)")
        logging.info(f"✓ Decoder: Randomly initialized")
        logging.info("=" * 80)


def save_checkpoint(
    params: AttributeDict,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
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
    model: nn.Module,
    batch: dict,
    graph_compiler: BpeCtcTrainingGraphCompiler,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute CTC loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Conformer in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      graph_compiler:
        It is used to build a decoding graph from a ctc topo and training
        transcript. The training transcript is contained in the given `batch`,
        while the ctc topo is built when this compiler is instantiated.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
    """
    device = graph_compiler.device
    feature = batch["inputs"]
    # at entry, feature is (N, T, C)
    assert feature.ndim == 3
    feature = feature.to(device)

    supervisions = batch["supervisions"]
    with torch.set_grad_enabled(is_training):
        nnet_output, encoder_memory, memory_mask = model(feature, supervisions)
        # nnet_output is (N, T, C)

    if params.get("debug_train", False):
        try:
            N, T, C = nnet_output.shape
        except Exception:
            N = feature.size(0)
            T = feature.size(1)
            C = params.num_classes or -1
        logging.debug(f"[DEBUG] feats: {tuple(feature.shape)}; nnet_out: {(N,T,C)}; subsamp: {params.subsampling_factor}")

    # NOTE: We need `encode_supervisions` to sort sequences with
    # different duration in decreasing order, required by
    # `k2.intersect_dense` called in `k2.ctc_loss`
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

    dense_fsa_vec = k2.DenseFsaVec(
        nnet_output,
        supervision_segments,
        allow_truncate=max(params.subsampling_factor - 1, 10),
    )

    ctc_loss = k2.ctc_loss(
        decoding_graph=decoding_graph,
        dense_fsa_vec=dense_fsa_vec,
        output_beam=params.beam_size,
        reduction=params.reduction,
        use_double_scores=params.use_double_scores,
    )

    if params.att_rate != 0.0:
        with torch.set_grad_enabled(is_training):
            mmodel = model.module if hasattr(model, "module") else model
            # Note: We need to generate an unsorted version of token_ids
            # `encode_supervisions()` called above sorts text, but
            # encoder_memory and memory_mask are not sorted, so we
            # use an unsorted version `supervisions["text"]` to regenerate
            # the token_ids
            #
            # See https://github.com/k2-fsa/icefall/issues/97
            # for more details
            unsorted_token_ids = graph_compiler.texts_to_ids(supervisions["text"])
            att_loss = mmodel.decoder_forward(
                encoder_memory,
                memory_mask,
                token_ids=unsorted_token_ids,
                sos_id=graph_compiler.sos_id,
                eos_id=graph_compiler.eos_id,
            )
        loss = (1.0 - params.att_rate) * ctc_loss + params.att_rate * att_loss
    else:
        loss = ctc_loss
        att_loss = torch.tensor([0])

    info = MetricsTracker()
    info["frames"] = supervision_segments[:, 2].sum().item()
    info["ctc_loss"] = ctc_loss.detach().cpu().item()
    if params.att_rate != 0.0:
        info["att_loss"] = att_loss.detach().cpu().item()

    info["loss"] = loss.detach().cpu().item()

    # `utt_duration` and `utt_pad_proportion` would be normalized by `utterances`  # noqa
    info["utterances"] = feature.size(0)
    # averaged input duration in frames over utterances
    info["utt_duration"] = supervisions["num_frames"].sum().item()
    # averaged padding proportion over utterances
    info["utt_pad_proportion"] = (
        ((feature.size(1) - supervisions["num_frames"]) / feature.size(1)).sum().item()
    )

    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    model: nn.Module,
    graph_compiler: BpeCtcTrainingGraphCompiler,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
    epoch: int = None,
    tb_writer: Optional[SummaryWriter] = None,
    rank: int = 0,
) -> Tuple[MetricsTracker, Optional[float]]:
    """Run the validation process with optional WER computation."""
    model.eval()

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

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    # Compute WER using the existing function
    wer = None
    if params.validation_decoding_method != "none":
        try:
            max_batches = getattr(params, 'validation_epoch_max_batches', -1)
            wer, samples, results_all = compute_validation_wer_and_examples(
                params=params,
                model=model,
                valid_dl=valid_dl,
                world_size=world_size,
                max_batches=max_batches,
            )
            
            if wer == wer:  # Check if not NaN
                logging.info(f"Validation WER: {wer:.2f}%")
                if rank == 0 and tb_writer is not None:
                    tb_writer.add_scalar("validation/wer", wer, params.batch_idx_train)
                    tb_writer.add_scalar("validation/loss", loss_value, params.batch_idx_train)
                
                # Use advanced logging functions
                if results_all:
                    # Convert results to the format expected by log_prediction_examples
                    results_dict = {"ctc-decoding": results_all}
                    log_prediction_examples(results_dict, max_examples=5, force_log=False)
                    
                    # Log to TensorBoard as well
                    if rank == 0 and tb_writer is not None:
                        log_validation_examples_to_tensorboard(
                            results_dict, tb_writer, params.batch_idx_train, max_examples=5
                        )
            else:
                logging.info("Validation WER: N/A")
                
        except Exception as e:
            logging.warning(f"WER computation failed: {e}")
            wer = None

    return tot_loss, wer



def _compute_edit_distance(ref: List[str], hyp: List[str]) -> int:
    """Compute Levenshtein distance between two word lists."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            hj = hyp[j - 1]
            cost = 0 if ri == hj else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[n][m]


@torch.no_grad()
def compute_validation_wer_and_examples(
    params: AttributeDict,
    model: nn.Module,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
    max_batches: int = -1,
) -> Tuple[float, List[Tuple[str, str, str]], List[Tuple[str, List[str], List[str]]]]:
    """Run greedy CTC decoding on the validation set and compute dataset-level WER.

    Returns (wer, samples) where samples is a list of (utt_id, ref_text, hyp_text).
    """
    if params.validation_decoding_method == "none":
        return float("nan"), []

    # Only support BPE-based lang dirs here for simplicity.
    if "lang_bpe" not in str(params.lang_dir):
        logging.warning(
            "Validation decoding currently supports only BPE lang dirs; skipping WER."
        )
        return float("nan"), []

    device = next(model.parameters()).device
    model.eval()

    # Load BPE model for token->text conversion and build CTC topo H.
    bpe_model = spm.SentencePieceProcessor()
    bpe_model.load(str(Path(params.lang_dir) / "bpe.model"))
    assert params.max_token_id is not None, "params.max_token_id must be set in run()"
    H = k2.ctc_topo(
        max_token=params.max_token_id,
        modified=False,
        device=device,
    )

    total_words = 0
    total_errs = 0
    samples: List[Tuple[str, str, str]] = []
    results_all: List[Tuple[str, List[str], List[str]]] = []

    batches_iter = enumerate(valid_dl)
    for b_idx, batch in batches_iter:
        feature = batch["inputs"].to(device)
        supervisions = batch["supervisions"]
        texts: List[str] = supervisions["text"]
        cut_ids = [cut.id for cut in supervisions["cut"]]

        # Forward
        nnet_output, _, _ = model(feature, supervisions)
        # Build supervision segments
        supervision_segments = torch.stack(
            (
                supervisions["sequence_idx"],
                supervisions["start_frame"] // params.subsampling_factor,
                supervisions["num_frames"] // params.subsampling_factor,
            ),
            1,
        ).to(torch.int32)

        # Lattice-based CTC decoding (same as decode.py "ctc-decoding" flow)
        lattice = get_lattice(
            nnet_output=nnet_output,
            decoding_graph=H,
            supervision_segments=supervision_segments,
            search_beam=20,  # modest defaults; training validation need not be exhaustive
            output_beam=8,
            min_active_states=5,
            max_active_states=50,
            subsampling_factor=params.subsampling_factor,
        )
        best_path = one_best_decoding(lattice=lattice, use_double_scores=True)
        token_ids = get_texts(best_path)  # list[list[int]] of BPE piece-ids
        hyp_texts = bpe_model.decode(token_ids)

        for uid, ref, hyp in zip(cut_ids, texts, hyp_texts):
            ref_words = ref.split()
            hyp_words = hyp.split()
            total_words += len(ref_words)
            total_errs += _compute_edit_distance(ref_words, hyp_words)
            results_all.append((uid, ref_words, hyp_words))
            if len(samples) < params.validation_print_samples:
                samples.append((uid, ref, hyp))

        if max_batches > 0 and (b_idx + 1) >= max_batches:
            break

    # DDP reduction
    if world_size > 1 and torch.distributed.is_initialized():
        t = torch.tensor([total_errs, total_words], device=device, dtype=torch.long)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        total_errs = int(t[0].item())
        total_words = int(t[1].item())

    wer = (total_errs / max(total_words, 1)) * 100.0
    return wer, samples, results_all


def _save_validation_results_like_decode(
    params: AttributeDict,
    results: List[Tuple[str, List[str], List[str]]],
    test_set_name: str,
    key: str = "ctc-decoding",
):
    """Save transcripts and error stats following decode.py naming scheme.

    Files:
      - recogs-{test_set_name}-{key}.txt
      - errs-{test_set_name}-{key}.txt
      - wer-summary-{test_set_name}.txt
    """
    results_sorted = sorted(results)
    recog_path = params.exp_dir / f"recogs-{test_set_name}-{key}.txt"
    errs_path = params.exp_dir / f"errs-{test_set_name}-{key}.txt"

    store_transcripts(filename=recog_path, texts=results_sorted)
    with open(errs_path, "w") as f:
        wer = write_error_stats(
            f, f"{test_set_name}-{key}", results_sorted, enable_log=False
        )
    logging.info(f"Saved validation transcripts to {recog_path}")
    logging.info(f"Saved validation error stats to {errs_path}")

    # Per-decode.py style summary (single setting here)
    summary_path = params.exp_dir / f"wer-summary-{test_set_name}.txt"
    with open(summary_path, "w") as f:
        print("settings\tWER", file=f)
        print(f"{key}\t{wer}", file=f)
    logging.info(f"Saved validation WER summary to {summary_path}")


def train_one_epoch(
    params: AttributeDict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    graph_compiler: BpeCtcTrainingGraphCompiler,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
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
        )
        # summary stats
        tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

        # NOTE: We use reduction==sum and loss is computed over utterances
        # in the batch and there is no normalization to it so far.

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0, 2.0)  # More moderate clipping for better learning
        optimizer.step()

        if batch_idx % params.log_interval == 0:
            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}"
            )
            

                    
            if params.get("debug_train", False):
                try:
                    cur_lr = optimizer._rate
                except Exception:
                    cur_lr = None
                if cur_lr is not None:
                    logging.info(f"[DEBUG] lr={cur_lr:.6e}")
                

        if batch_idx % params.log_interval == 0:
            if tb_writer is not None:
                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)

    if batch_idx > 0 and batch_idx % params.valid_interval == 0:
            logging.info("Computing validation loss")
            valid_info, valid_wer = compute_validation_loss(
                params=params,
                model=model,
                graph_compiler=graph_compiler,
                valid_dl=valid_dl,
                world_size=world_size,
                epoch=params.cur_epoch,
                tb_writer=tb_writer,
                rank=0,  # Simplified for now
            )
            model.train()
            wer_info = f", WER: {valid_wer:.2f}%" if valid_wer is not None else ""
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}{wer_info}")
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

            # Optional: quick validation decoding (limited batches) for WER and sample logs
            if params.validation_decoding_method != "none":
                max_batches = params.validation_step_max_batches
                wer, samples, results_all = compute_validation_wer_and_examples(
                    params=params,
                    model=model.module if hasattr(model, "module") else model,
                    valid_dl=valid_dl,
                    world_size=world_size,
                    max_batches=max_batches,
                )
                if wer == wer:  # not NaN
                    logging.info(
                        f"[Step-Valid] Epoch {params.cur_epoch}, batch {batch_idx}: WER={wer:.2f}%"
                    )
                    if tb_writer is not None:
                        # Log step-wise validation WER (possibly on subset if max_batches>0)
                        tb_writer.add_scalar(
                            "valid_step/wer", wer, params.batch_idx_train
                        )
                
                # Use advanced logging functions for step validation too
                if results_all:
                    results_dict = {"ctc-decoding": results_all}
                    log_prediction_examples(results_dict, max_examples=3, force_log=False)
                    
                    # Log to TensorBoard with step prefix
                    if tb_writer is not None:
                        log_validation_examples_to_tensorboard(
                            results_dict, tb_writer, params.batch_idx_train, max_examples=3
                        )
                
                # Save detailed stats (tag indicates step and batch it came from)
                _save_validation_results_like_decode(
                    params=params,
                    results=results_all,
                    test_set_name="valid",
                    key=f"ctc-decoding-step{params.cur_epoch}_batch{batch_idx}",
                )

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
    logging.info(params)

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    lexicon = Lexicon(params.lang_dir)
    max_token_id = max(lexicon.tokens)
    num_classes = max_token_id + 1  # +1 for the blank
    params.max_token_id = max_token_id
    params.num_classes = num_classes

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
    model = Conformer(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.attention_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
        num_encoder_layers=18,
        num_decoder_layers=params.num_decoder_layers,
        vgg_frontend=False,
        use_feat_batchnorm=params.use_feat_batchnorm,
    )
    
    # Load pretrained encoder weights if provided
    if params.init_model_from_pretrain:
        load_pretrained_encoder(params.init_model_from_pretrain, model)
    
    checkpoints = load_checkpoint_if_available(params=params, model=model)

    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    optimizer = Noam(
        model.parameters(),
        model_size=params.attention_dim,
        factor=params.lr_factor,
        warm_step=params.warm_step,
        weight_decay=params.weight_decay,
    )

    if checkpoints:
        optimizer.load_state_dict(checkpoints["optimizer"])

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

    train_dl = librispeech.train_dataloaders(train_cuts)

    valid_cuts = librispeech.dev_clean_cuts()
    valid_cuts += librispeech.dev_other_cuts()
    valid_dl = librispeech.valid_dataloaders(valid_cuts)

    if params.get("sanity_check", False) or getattr(params, "sanity_check", False) or getattr(args, "sanity_check", False):
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            graph_compiler=graph_compiler,
            params=params,
        )

    for epoch in range(params.start_epoch, params.num_epochs):
        fix_random_seed(params.seed + epoch)
        train_dl.sampler.set_epoch(epoch)

        cur_lr = optimizer._rate
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
        )

        # End-of-epoch validation decoding on full validation set (or limited by arg)
        if params.validation_decoding_method != "none":
            max_batches = params.validation_epoch_max_batches
            wer, samples, results_all = compute_validation_wer_and_examples(
                params=params,
                model=model.module if hasattr(model, "module") else model,
                valid_dl=valid_dl,
                world_size=world_size,
                max_batches=max_batches,
            )
            if wer == wer:
                logging.info(f"[Epoch-Valid] Epoch {params.cur_epoch}: dataset WER={wer:.2f}%")
                if tb_writer is not None:
                    # Epoch-level dataset (or subset if limited) WER
                    tb_writer.add_scalar(
                        "valid_epoch/wer", wer, params.batch_idx_train
                    )
            
            # Use advanced logging for epoch validation too
            if results_all:
                results_dict = {"ctc-decoding": results_all}
                log_prediction_examples(results_dict, max_examples=5, force_log=True)  # Force log for epoch-end
                
                # Log to TensorBoard with epoch prefix
                if tb_writer is not None:
                    log_validation_examples_to_tensorboard(
                        results_dict, tb_writer, params.batch_idx_train, max_examples=5
                    )
            
            _save_validation_results_like_decode(
                params=params,
                results=results_all,
                test_set_name="valid",
                key=f"ctc-decoding-epoch{params.cur_epoch}",
            )

        save_checkpoint(
            params=params,
            model=model,
            optimizer=optimizer,
            rank=rank,
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
            clip_grad_norm_(model.parameters(), 5.0, 2.0)  # More moderate clipping for better learning
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


def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()
