#!/usr/bin/env python3
"""
Evaluate model performance on noise-augmented LibriSpeech test sets.

This script adds noise (babble, pink, white) from NOISEX-92 to LibriSpeech
test-clean/test-other and measures WER at various SNR levels.
"""

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm

from asr_datamodule import LibriSpeechAsrDataModule
from conformer import Conformer

from icefall.checkpoint import average_checkpoints
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    write_error_stats,
)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Decode with noise at various SNR levels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Noise-specific arguments (not in LibriSpeechAsrDataModule)
    parser.add_argument(
        "--noise-dir",
        type=str,
        default="/home/nas4/DB/DB_NOISEX92",
        help="Directory containing NOISEX-92 noise files",
    )
    
    parser.add_argument(
        "--noise-types",
        type=str,
        nargs="+",
        default=["babble", "pink", "white"],
        help="Noise types to use",
    )
    
    parser.add_argument(
        "--snr-levels",
        type=float,
        nargs="+",
        default=[-5, 0, 5, 10, 15, 20],
        help="SNR levels in dB to evaluate",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="noisy_results",
        help="Output directory for results",
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    
    parser.add_argument(
        "--test-set",
        type=str,
        default="test-clean",
        choices=["test-clean", "test-other"],
        help="Test set to evaluate",
    )
    
    # Model checkpoint arguments
    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="Checkpoint epoch to load",
    )
    
    parser.add_argument(
        "--avg",
        type=int,
        default=5,
        help="Number of checkpoints to average",
    )
    
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="conformer_ctc_sd/exp/models",
        help="Experiment directory containing checkpoints",
    )
    
    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lang_bpe_1024",
        help="Language directory with BPE model",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    
    parser.add_argument(
        "--strict",
        type=bool,
        default=False,
        help="Strict checkpoint loading",
    )
    
    return parser


def load_noise_file(noise_path: Path, target_sr: int = 16000) -> torch.Tensor:
    """Load and preprocess noise file."""
    logging.info(f"Loading noise from {noise_path}")
    
    # Try different loading methods
    try:
        waveform, sr = torchaudio.load(str(noise_path))
    except Exception as e:
        logging.error(f"Failed to load {noise_path}: {e}")
        raise
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        logging.info(f"Resampling from {sr}Hz to {target_sr}Hz")
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    return waveform.squeeze(0)  # [samples]


def find_noise_files(noise_dir: Path, noise_types: List[str]) -> Dict[str, Path]:
    """
    Find noise files in NOISEX-92 directory.
    
    Expected structure: /path/to/DB_NOISEX92/{noise_type}/{audio_files}
    For example: /home/nas4/DB/DB_NOISEX92/babble/babble.wav
    """
    noise_files = {}
    
    logging.info(f"Searching for noise files in {noise_dir}")
    
    if not noise_dir.exists():
        logging.error(f"Noise directory does not exist: {noise_dir}")
        return noise_files
    
    # List subdirectories
    subdirs = [d for d in noise_dir.iterdir() if d.is_dir()]
    logging.info(f"Found subdirectories: {[d.name for d in subdirs]}")
    
    extensions = [".wav", ".WAV", ".sph", ".SPH", ".flac", ".FLAC", ".mp3", ".MP3"]
    
    for noise_type in noise_types:
        found = False
        
        # Strategy 1: Check if there's a subdirectory matching the noise type
        noise_subdir = noise_dir / noise_type.lower()
        if not noise_subdir.exists():
            noise_subdir = noise_dir / noise_type.upper()
        if not noise_subdir.exists():
            noise_subdir = noise_dir / noise_type.capitalize()
        
        if noise_subdir.exists() and noise_subdir.is_dir():
            logging.info(f"Found {noise_type} subdirectory: {noise_subdir}")
            
            # Find any audio file in this subdirectory
            for ext in extensions:
                candidates = list(noise_subdir.glob(f"*{ext}"))
                if candidates:
                    # Use the first file, or you could concatenate multiple files
                    noise_files[noise_type] = candidates[0]
                    logging.info(f"Found {noise_type} noise: {candidates[0]}")
                    if len(candidates) > 1:
                        logging.info(f"  Note: Found {len(candidates)} files, using first one: {candidates[0].name}")
                    found = True
                    break
            
            if not found:
                logging.warning(f"No audio files with extensions {extensions} found in {noise_subdir}")
                # List what's actually in there
                files_in_dir = list(noise_subdir.iterdir())
                logging.info(f"  Files in {noise_subdir.name}: {[f.name for f in files_in_dir[:10]]}")
        
        # Strategy 2: Try recursive search with pattern matching
        if not found:
            logging.info(f"Trying recursive search for {noise_type}")
            search_patterns = [
                noise_type.lower(),
                noise_type.upper(),
                noise_type.capitalize(),
            ]
            
            for pattern in search_patterns:
                for ext in extensions:
                    candidates = list(noise_dir.glob(f"**/*{pattern}*{ext}"))
                    if candidates:
                        noise_files[noise_type] = candidates[0]
                        logging.info(f"Found {noise_type} via pattern search: {candidates[0]}")
                        found = True
                        break
                if found:
                    break
        
        if not found:
            logging.warning(f"Could not find {noise_type} noise in {noise_dir}")
    
    return noise_files


def add_noise_at_snr(
    clean_audio: torch.Tensor,
    noise_audio: torch.Tensor,
    snr_db: float,
    random_start: bool = True,
) -> torch.Tensor:
    """
    Add noise to clean audio at specified SNR.
    
    Args:
        clean_audio: Clean waveform [samples]
        noise_audio: Noise waveform [samples]
        snr_db: Target SNR in dB
        random_start: Use random starting position in noise
        
    Returns:
        Noisy audio [samples]
    """
    audio_len = clean_audio.shape[0]
    noise_len = noise_audio.shape[0]
    
    # Select noise segment
    if noise_len >= audio_len:
        if random_start:
            start = random.randint(0, noise_len - audio_len)
        else:
            start = 0
        noise_segment = noise_audio[start:start + audio_len]
    else:
        # Repeat noise to match length
        repeats = (audio_len // noise_len) + 1
        noise_repeated = noise_audio.repeat(repeats)
        noise_segment = noise_repeated[:audio_len]
    
    # Calculate powers
    signal_power = (clean_audio ** 2).mean()
    noise_power = (noise_segment ** 2).mean()
    
    # Avoid division by zero
    if noise_power < 1e-10:
        logging.warning("Noise power too low, returning clean audio")
        return clean_audio
    
    # Calculate noise scaling factor
    # SNR = 10 * log10(P_signal / P_noise)
    # P_noise_target = P_signal / (10^(SNR/10))
    snr_linear = 10 ** (snr_db / 10)
    noise_scale = torch.sqrt(signal_power / (noise_power * snr_linear))
    
    # Add scaled noise
    noisy_audio = clean_audio + noise_scale * noise_segment
    
    return noisy_audio


def extract_fbank_features(
    audio: torch.Tensor,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
) -> torch.Tensor:
    """Extract log mel filterbank features."""
    # Use torchaudio compliance kaldi fbank
    features = torchaudio.compliance.kaldi.fbank(
        audio.unsqueeze(0),
        num_mel_bins=num_mel_bins,
        sample_frequency=sample_rate,
        frame_length=25,
        frame_shift=10,
    )
    return features  # [num_frames, num_mel_bins]


def compute_edit_distance(ref: str, hyp: str) -> Tuple[int, int, int, int]:
    """
    Compute Word Error Rate (WER) metrics using dynamic programming.
    
    Args:
        ref: Reference text (string)
        hyp: Hypothesis text (string)
    
    Returns:
        (num_words, substitutions, insertions, deletions)
    """
    ref_words = ref.split()
    hyp_words = hyp.split()
    
    # Dynamic programming for edit distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    
    # Backtrack to count error types
    i, j = len(ref_words), len(hyp_words)
    substitutions, insertions, deletions = 0, 0, 0
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            insertions += 1
            j -= 1
        else:
            deletions += 1
            i -= 1
    
    return len(ref_words), substitutions, insertions, deletions


def compute_wer_stats(refs: List[List[str]], hyps: List[List[str]]) -> Dict[str, float]:
    """
    Compute WER statistics.
    
    Returns:
        Dictionary with total_words, total_errors, wer
    """
    total_words = 0
    total_ins = 0
    total_del = 0
    total_sub = 0
    
    for ref, hyp in zip(refs, hyps):
        # Convert to strings for edit distance calculation
        ref_str = " ".join(ref) if isinstance(ref, list) else ref
        hyp_str = " ".join(hyp) if isinstance(hyp, list) else hyp
        
        # Compute edit distance
        num_words, subs, ins, dels = compute_edit_distance(ref_str, hyp_str)
        
        total_words += num_words
        total_ins += ins
        total_del += dels
        total_sub += subs
    
    total_errors = total_ins + total_del + total_sub
    wer = 100.0 * total_errors / total_words if total_words > 0 else 0.0
    
    return {
        "total_words": total_words,
        "total_errors": total_errors,
        "insertions": total_ins,
        "deletions": total_del,
        "substitutions": total_sub,
        "wer": wer,
    }


@torch.no_grad()
def decode_with_noise(
    model: nn.Module,
    bpe_model: spm.SentencePieceProcessor,
    test_cuts: any,
    noise_audio: torch.Tensor,
    snr_db: float,
    noise_type: str,
    params: AttributeDict,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Decode test set with added noise.
    
    Returns:
        (hypotheses, references)
    """
    model.eval()
    device = params.device
    
    hypotheses = []
    references = []
    
    num_processed = 0
    
    for cut in tqdm(test_cuts, desc=f"{noise_type} SNR={snr_db}dB"):
        if params.max_samples and num_processed >= params.max_samples:
            break
        
        # Load clean audio
        clean_audio = cut.load_audio()
        if isinstance(clean_audio, np.ndarray):
            clean_audio = torch.from_numpy(clean_audio).float()
        if clean_audio.dim() > 1:
            clean_audio = clean_audio.squeeze()
        
        # Add noise
        noisy_audio = add_noise_at_snr(clean_audio, noise_audio, snr_db)
        
        # Extract features
        features = extract_fbank_features(noisy_audio, num_mel_bins=params.num_features)
        features = features.unsqueeze(0).to(device)  # [1, T, F]
        
        # Forward pass
        nnet_output, _, _, _, _ = model(features, None)
        
        # CTC greedy decoding
        log_probs = torch.nn.functional.log_softmax(nnet_output, dim=-1)
        log_probs = log_probs.squeeze(1)  # [T, B, C] -> [T, C]
        
        # Get best path
        indices = torch.argmax(log_probs, dim=-1)  # [T]
        indices = torch.unique_consecutive(indices)
        indices = indices[indices != 0]  # Remove blanks
        
        # Decode with BPE
        token_ids = indices.tolist()
        hyp_text = bpe_model.decode(token_ids)
        
        # Get reference
        ref_text = cut.supervisions[0].text
        
        hypotheses.append(hyp_text.split())
        references.append(ref_text.split())
        
        num_processed += 1
    
    return hypotheses, references


def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    
    params = AttributeDict(vars(args))
    params.res_dir = Path(params.output_dir) / params.test_set
    params.res_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logger(f"{params.res_dir}/log-noisy-decode")
    logging.info("Noisy decoding started")
    logging.info(params)
    
    # Find noise files
    noise_dir = Path(params.noise_dir)
    noise_file_paths = find_noise_files(noise_dir, params.noise_types)
    
    if not noise_file_paths:
        raise FileNotFoundError(f"No noise files found in {noise_dir}")
    
    # Load noise audio
    noise_audio_dict = {}
    for noise_type, noise_path in noise_file_paths.items():
        noise_audio_dict[noise_type] = load_noise_file(noise_path)
        logging.info(f"Loaded {noise_type}: {noise_audio_dict[noise_type].shape[0]} samples")
    
    # Load model
    logging.info("Loading model...")
    
    # Get num_classes from tokens.txt (same as decode.py)
    tokens_file = Path(params.lang_dir) / "tokens.txt"
    with open(tokens_file, 'r', encoding='utf-8') as f:
        num_classes = len(f.readlines())
    
    logging.info(f"Read {num_classes} classes from {tokens_file}")
    
    # Load checkpoint to check model structure
    checkpoint_path = f"{params.exp_dir}/epoch-{params.epoch}.pt"
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Get model parameters from checkpoint
    if "params" in checkpoint:
        model_params = checkpoint["params"]
        logging.info("Found model params in checkpoint")
        # Override num_classes with the value from tokens.txt
        model_params["num_classes"] = num_classes
    else:
        # Infer parameters from checkpoint model state dict
        logging.warning("Model params not found in checkpoint, inferring from state dict")
        state_dict = checkpoint.get("model", checkpoint)
        
        # Infer d_model from encoder layer
        if "encoder_embed.conv.6.weight" in state_dict:
            d_model = state_dict["encoder_embed.conv.6.weight"].shape[0]
        elif "encoder.layers.0.feed_forward.0.weight" in state_dict:
            d_model = state_dict["encoder.layers.0.feed_forward.0.weight"].shape[1]
        else:
            d_model = 512
        
        # Count encoder layers
        num_encoder_layers = 0
        for key in state_dict.keys():
            if key.startswith("encoder.layers."):
                try:
                    layer_num = int(key.split(".")[2]) + 1
                    num_encoder_layers = max(num_encoder_layers, layer_num)
                except:
                    pass
        
        if num_encoder_layers == 0:
            num_encoder_layers = 18
        
        model_params = {
            "num_features": 80,
            "nhead": 4,  # conformer_ctc default
            "d_model": 256,  # conformer_ctc uses attention_dim=512
            "num_classes": num_classes,  # Use num_classes from tokens.txt
            "subsampling_factor": 4,
            "num_decoder_layers": 6,  # conformer_ctc default
            "vgg_frontend": False,
            "num_encoder_layers": 18,  # conformer_ctc default
            "use_feat_batchnorm": True,  # conformer_ctc default
        }
        
        logging.info(f"Using inferred parameters from checkpoint")
        logging.info(f"Using num_classes={num_classes} from tokens.txt")
    
    # Create model - conformer_ctc standard configuration
    actual_num_classes = model_params.get("num_classes", num_classes)
    logging.info(f"Creating model with conformer_ctc configuration:")
    logging.info(f"  num_features={model_params.get('num_features', 80)}")
    logging.info(f"  nhead={model_params.get('nhead', 8)}")
    logging.info(f"  d_model={model_params.get('d_model', 512)}")
    logging.info(f"  num_classes={actual_num_classes}")
    logging.info(f"  num_encoder_layers={model_params.get('num_encoder_layers', 18)}")
    logging.info(f"  num_decoder_layers={model_params.get('num_decoder_layers', 6)}")
    
    model = Conformer(
        num_features=model_params.get("num_features", 80),
        nhead=model_params.get("nhead", 8),
        d_model=model_params.get("d_model", 512),
        num_classes=actual_num_classes,
        subsampling_factor=model_params.get("subsampling_factor", 4),
        num_decoder_layers=model_params.get("num_decoder_layers", 6),
        vgg_frontend=model_params.get("vgg_frontend", False),
        num_encoder_layers=model_params.get("num_encoder_layers", 18),
        use_feat_batchnorm=model_params.get("use_feat_batchnorm", True),
    )
    
    logging.info(f"Model created: {model_params.get('num_encoder_layers', 18)} layers, "
                f"d_model={model_params.get('d_model', 512)}, "
                f"num_classes={actual_num_classes}")
    
    # Load checkpoint weights
    if params.avg > 0:
        start = max(params.epoch - params.avg + 1, 0)
        filenames = [f"{params.exp_dir}/epoch-{i}.pt" for i in range(start, params.epoch + 1)]
        logging.info(f"Averaging {filenames}")
        model.to(params.device)
        model.load_state_dict(average_checkpoints(filenames, device=params.device), strict=params.strict)
    else:
        logging.info(f"Loading weights from {checkpoint_path}")
        model.load_state_dict(checkpoint["model"], strict=params.strict)
        model.to(params.device)
    
    model.eval()
    logging.info(f"Model loaded on {params.device}")
    
    # Store num_features for later use
    params.num_features = model_params.get("num_features", 80)
    
    # Load BPE model
    bpe_model_path = Path(params.lang_dir) / "bpe.model"
    bpe_model = spm.SentencePieceProcessor()
    bpe_model.load(str(bpe_model_path))
    logging.info(f"BPE model loaded: {len(bpe_model)} tokens")
    
    # Load test data
    logging.info(f"Loading {params.test_set}")
    librispeech = LibriSpeechAsrDataModule(args)
    
    if params.test_set == "test-clean":
        test_cuts = librispeech.test_clean_cuts()
    else:
        test_cuts = librispeech.test_other_cuts()
    
    logging.info(f"Loaded {len(test_cuts)} cuts")
    
    # Decode clean (no noise) first
    logging.info("=" * 80)
    logging.info("Decoding CLEAN audio (baseline)")
    logging.info("=" * 80)
    
    clean_hyps, clean_refs = decode_with_noise(
        model=model,
        bpe_model=bpe_model,
        test_cuts=test_cuts,
        noise_audio=torch.zeros(16000),  # Dummy noise (won't be used at high SNR)
        snr_db=999,  # Very high SNR = no noise
        noise_type="clean",
        params=params,
    )
    
    # Save clean results
    clean_recog_path = params.res_dir / "recogs-clean.txt"
    with open(clean_recog_path, "w") as f:
        for hyp in clean_hyps:
            f.write(" ".join(hyp) + "\n")
    
    # Compute and save clean WER stats
    clean_stats = compute_wer_stats(clean_refs, clean_hyps)
    clean_errs_path = params.res_dir / "errs-clean.txt"
    with open(clean_errs_path, "w") as f:
        f.write(f"Clean Audio Results\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"Total words: {clean_stats['total_words']}\n")
        f.write(f"Total errors: {clean_stats['total_errors']}\n")
        f.write(f"  - Insertions: {clean_stats['insertions']}\n")
        f.write(f"  - Deletions: {clean_stats['deletions']}\n")
        f.write(f"  - Substitutions: {clean_stats['substitutions']}\n")
        f.write(f"WER: {clean_stats['wer']:.2f}%\n")
    
    logging.info(f"Clean results: WER={clean_stats['wer']:.2f}% ({clean_stats['total_errors']}/{clean_stats['total_words']})")
    logging.info(f"Results saved to {clean_recog_path}")
    
    # Decode with noise at each SNR
    summary_results = []
    
    for noise_type, noise_audio in noise_audio_dict.items():
        for snr_db in params.snr_levels:
            logging.info("=" * 80)
            logging.info(f"Decoding: {noise_type} at SNR={snr_db}dB")
            logging.info("=" * 80)
            
            hyps, refs = decode_with_noise(
                model=model,
                bpe_model=bpe_model,
                test_cuts=test_cuts,
                noise_audio=noise_audio,
                snr_db=snr_db,
                noise_type=noise_type,
                params=params,
            )
            
            # Compute and save results
            result_name = f"{noise_type}_snr{snr_db}"
            noisy_stats = compute_wer_stats(refs, hyps)
            
            recog_path = params.res_dir / f"recogs-{result_name}.txt"
            with open(recog_path, "w") as f:
                for hyp in hyps:
                    f.write(" ".join(hyp) + "\n")
            
            errs_path = params.res_dir / f"errs-{result_name}.txt"
            with open(errs_path, "w") as f:
                f.write(f"{noise_type.upper()} Noise (SNR={snr_db}dB) Results\n")
                f.write(f"=" * 60 + "\n")
                f.write(f"Total words: {noisy_stats['total_words']}\n")
                f.write(f"Total errors: {noisy_stats['total_errors']}\n")
                f.write(f"  - Insertions: {noisy_stats['insertions']}\n")
                f.write(f"  - Deletions: {noisy_stats['deletions']}\n")
                f.write(f"  - Substitutions: {noisy_stats['substitutions']}\n")
                f.write(f"WER: {noisy_stats['wer']:.2f}%\n")
            
            logging.info(f"{noise_type} SNR={snr_db}dB: WER={noisy_stats['wer']:.2f}% ({noisy_stats['total_errors']}/{noisy_stats['total_words']})")
            logging.info(f"Results saved to {recog_path}")
            
            # Store WER for summary
            summary_results.append((noise_type, snr_db, noisy_stats['wer']))
    
    # Save summary
    summary_path = params.res_dir / "wer_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Noisy Decoding Results - {params.test_set}\n")
        f.write(f"Model: {params.exp_dir} (epoch {params.epoch}, avg {params.avg})\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Noise Type':<15} {'SNR (dB)':<10} {'WER (%)':<10}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'CLEAN':<15} {'-':<10} {clean_stats['wer']:<10.2f}\n")
        f.write("-" * 60 + "\n")
        for noise_type, snr_db, wer in summary_results:
            f.write(f"{noise_type:<15} {snr_db:<10.1f} {wer:<10.2f}\n")
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Summary saved to {summary_path}")
    logging.info(f"Clean WER: {clean_stats['wer']:.2f}%")
    logging.info(f"{'='*60}")
    logging.info("\nDecoding complete!")


if __name__ == "__main__":
    main()
