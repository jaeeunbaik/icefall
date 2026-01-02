#!/usr/bin/env python3
"""
Apply SpecAugment to audio and save as wav file.

Usage:
    python apply_specaugment_and_save.py \
        --input input.wav \
        --output output_augmented.wav \
        --time-warp-factor 80 \
        --num-freq-masks 2 \
        --num-time-masks 2
"""

import argparse
import logging
import torch
import torchaudio
import numpy as np
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def apply_specaugment(
    mel_spec: torch.Tensor,
    time_warp_factor: int = 80,
    freq_mask_param: int = 27,
    time_mask_param: int = 100,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> torch.Tensor:
    """
    Apply SpecAugment to mel spectrogram.
    
    Args:
        mel_spec: Mel spectrogram tensor [freq, time]
        time_warp_factor: Time warping parameter (0 to disable)
        freq_mask_param: Maximum frequency mask width
        time_mask_param: Maximum time mask width
        num_freq_masks: Number of frequency masks to apply
        num_time_masks: Number of time masks to apply
        
    Returns:
        Augmented mel spectrogram [freq, time]
    """
    from torchaudio.transforms import FrequencyMasking, TimeMasking
    
    # Clone to avoid modifying original
    augmented = mel_spec.clone()
    
    # Apply frequency masking
    if num_freq_masks > 0 and freq_mask_param > 0:
        freq_masking = FrequencyMasking(freq_mask_param=freq_mask_param)
        for _ in range(num_freq_masks):
            augmented = freq_masking(augmented)
    
    # Apply time masking
    if num_time_masks > 0 and time_mask_param > 0:
        time_masking = TimeMasking(time_mask_param=time_mask_param)
        for _ in range(num_time_masks):
            augmented = time_masking(augmented)
    
    return augmented


def mel_to_audio(
    mel_spec: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    win_length: int = 400,
) -> torch.Tensor:
    """
    Convert mel spectrogram back to audio using Griffin-Lim algorithm.
    
    Uses mel filterbank matrix for more stable inverse transform.
    
    Args:
        mel_spec: Mel spectrogram [n_mels, time]
        sample_rate: Audio sample rate
        n_fft: FFT size
        hop_length: Hop length for STFT
        n_mels: Number of mel bins
        win_length: Window length
        
    Returns:
        Audio waveform [samples]
    """
    import torchaudio.functional as F_audio
    
    # Create mel filterbank
    mel_filters = F_audio.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=0.0,
        f_max=sample_rate / 2.0,
        n_mels=n_mels,
        sample_rate=sample_rate,
        norm="slaney",
    ).T  # [n_mels, n_freqs]
    
    # Compute pseudo-inverse of mel filterbank (more stable than lstsq)
    # Use Moore-Penrose pseudo-inverse with regularization
    mel_filters_pinv = torch.linalg.pinv(mel_filters)  # [n_freqs, n_mels]
    
    # Convert mel to linear spectrogram: S = W^+ @ M
    # where W^+ is pseudo-inverse of mel filterbank
    linear_spec = torch.matmul(mel_filters_pinv, mel_spec)  # [n_freqs, time]
    
    # Ensure non-negative and add small epsilon for numerical stability
    linear_spec = torch.clamp(linear_spec, min=0.0) + 1e-10
    
    # Apply Griffin-Lim to reconstruct phase and convert to audio
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=2.0,
        n_iter=32,  # More iterations = better quality
    )
    
    waveform = griffin_lim(linear_spec)
    
    return waveform


def process_audio(
    input_path: str,
    output_path: str,
    time_warp_factor: int = 80,
    freq_mask_param: int = 27,
    time_mask_param: int = 100,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
):
    """
    Load audio, apply SpecAugment, and save as wav file.
    """
    logging.info(f"Loading audio from {input_path}")
    
    # Load audio
    waveform, sr = torchaudio.load(input_path)
    
    # Resample if needed
    if sr != sample_rate:
        logging.info(f"Resampling from {sr} Hz to {sample_rate} Hz")
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        logging.info("Converting to mono")
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Extract mel spectrogram
    logging.info("Extracting mel spectrogram")
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    
    mel_spec = mel_transform(waveform)  # [1, n_mels, time]
    mel_spec = mel_spec.squeeze(0)  # [n_mels, time]
    
    # Convert to log scale
    mel_spec = torch.log(mel_spec + 1e-9)
    
    logging.info(f"Original mel spectrogram shape: {mel_spec.shape}")
    
    # Apply SpecAugment
    logging.info("Applying SpecAugment")
    augmented_mel = apply_specaugment(
        mel_spec,
        time_warp_factor=time_warp_factor,
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
        num_freq_masks=num_freq_masks,
        num_time_masks=num_time_masks,
    )
    
    logging.info(f"Augmented mel spectrogram shape: {augmented_mel.shape}")
    
    # Convert back to linear scale
    augmented_mel = torch.exp(augmented_mel) - 1e-9
    
    # Convert mel spectrogram back to audio
    logging.info("Converting mel spectrogram back to audio using Griffin-Lim")
    reconstructed_audio = mel_to_audio(
        augmented_mel,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    
    # Add channel dimension if needed
    if reconstructed_audio.dim() == 1:
        reconstructed_audio = reconstructed_audio.unsqueeze(0)
    
    # Normalize to prevent clipping
    max_val = reconstructed_audio.abs().max()
    if max_val > 1.0:
        reconstructed_audio = reconstructed_audio / max_val * 0.95
    
    # Save output
    logging.info(f"Saving augmented audio to {output_path}")
    torchaudio.save(
        output_path,
        reconstructed_audio,
        sample_rate,
    )
    
    logging.info(f"Done! Output shape: {reconstructed_audio.shape}")
    logging.info(f"Output duration: {reconstructed_audio.shape[1] / sample_rate:.2f}s")


def get_args():
    parser = argparse.ArgumentParser(
        description="Apply SpecAugment to audio and save as wav file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input audio file path (.wav)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output audio file path (.wav)",
    )
    
    parser.add_argument(
        "--time-warp-factor",
        type=int,
        default=80,
        help="Time warping parameter (0 to disable)",
    )
    
    parser.add_argument(
        "--freq-mask-param",
        type=int,
        default=27,
        help="Maximum frequency mask width",
    )
    
    parser.add_argument(
        "--time-mask-param",
        type=int,
        default=100,
        help="Maximum time mask width",
    )
    
    parser.add_argument(
        "--num-freq-masks",
        type=int,
        default=2,
        help="Number of frequency masks to apply",
    )
    
    parser.add_argument(
        "--num-time-masks",
        type=int,
        default=2,
        help="Number of time masks to apply",
    )
    
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate",
    )
    
    parser.add_argument(
        "--n-fft",
        type=int,
        default=400,
        help="FFT size",
    )
    
    parser.add_argument(
        "--hop-length",
        type=int,
        default=160,
        help="Hop length for STFT",
    )
    
    parser.add_argument(
        "--n-mels",
        type=int,
        default=80,
        help="Number of mel bins",
    )
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Validate input file
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process audio
    process_audio(
        input_path="/home/hdd1/jenny/LibriSpeech/test-clean/672/122797/672-122797-0005.wav",
        output_path=args.output,
        time_warp_factor=args.time_warp_factor,
        freq_mask_param=args.freq_mask_param,
        time_mask_param=args.time_mask_param,
        num_freq_masks=args.num_freq_masks,
        num_time_masks=args.num_time_masks,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    )


if __name__ == "__main__":
    main()
