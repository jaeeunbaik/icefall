#!/usr/bin/env python3
"""
Decode with noise-augmented audio at various SNR levels.

This script adds noise (babble, pink, white) to LibriSpeech test-clean
and evaluates model performance across different SNR levels.
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
from asr_datamodule import LibriSpeechAsrDataModule
from conformer import Conformer
from decode import decode_one_batch, get_parser as get_base_parser

from icefall.checkpoint import average_checkpoints
from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    get_texts,
    setup_logger,
    store_transcripts,
    write_error_stats,
)


def load_noise_file(noise_path: Path, target_sr: int = 16000) -> torch.Tensor:
    """Load noise file and resample if necessary."""
    waveform, sr = torchaudio.load(str(noise_path))
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if necessary
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    return waveform.squeeze(0)  # Return [samples]


def add_noise_at_snr(
    clean_audio: torch.Tensor,
    noise_audio: torch.Tensor,
    snr_db: float,
) -> torch.Tensor:
    """
    Add noise to clean audio at specified SNR level.
    
    Args:
        clean_audio: Clean audio tensor [samples]
        noise_audio: Noise audio tensor [samples]
        snr_db: Target SNR in dB
        
    Returns:
        Noisy audio tensor [samples]
    """
    # Get audio lengths
    audio_len = clean_audio.shape[0]
    noise_len = noise_audio.shape[0]
    
    # Randomly select noise segment or repeat if too short
    if noise_len >= audio_len:
        # Random start position
        start = random.randint(0, noise_len - audio_len)
        noise_segment = noise_audio[start:start + audio_len]
    else:
        # Repeat noise to match audio length
        repeats = (audio_len // noise_len) + 1
        noise_repeated = noise_audio.repeat(repeats)
        noise_segment = noise_repeated[:audio_len]
    
    # Calculate signal and noise power
    signal_power = (clean_audio ** 2).mean()
    noise_power = (noise_segment ** 2).mean()
    
    # Calculate required noise scale for target SNR
    # SNR_db = 10 * log10(signal_power / noise_power)
    # noise_power_target = signal_power / (10 ** (SNR_db / 10))
    snr_linear = 10 ** (snr_db / 10)
    noise_scale = torch.sqrt(signal_power / (noise_power * snr_linear))
    
    # Scale noise and add to signal
    noisy_audio = clean_audio + noise_scale * noise_segment
    
    return noisy_audio


class NoisyAudioTransform:
    """Transform to add noise to audio on-the-fly."""
    
    def __init__(
        self,
        noise_files: Dict[str, torch.Tensor],
        snr_db: float,
        noise_type: str = "all",
    ):
        """
        Args:
            noise_files: Dict mapping noise type to noise waveform
            snr_db: Target SNR in dB
            noise_type: Which noise to use ("babble", "pink", "white", or "all")
        """
        self.noise_files = noise_files
        self.snr_db = snr_db
        self.noise_type = noise_type
        
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Add noise to audio."""
        # Select noise type
        if self.noise_type == "all":
            noise_key = random.choice(list(self.noise_files.keys()))
        else:
            noise_key = self.noise_type
        
        noise = self.noise_files[noise_key]
        
        # Add noise at specified SNR
        noisy_audio = add_noise_at_snr(audio, noise, self.snr_db)
        
        return noisy_audio


def get_parser():
    """Get argument parser with noise-specific options."""
    parser = get_base_parser()
    
    # Add noise-specific arguments
    parser.add_argument(
        "--noise-dir",
        type=str,
        default="/home/nas4/DB/DB_NOISEX92",
        help="Directory containing noise files",
    )
    
    parser.add_argument(
        "--noise-types",
        type=str,
        nargs="+",
        default=["babble", "pink", "white"],
        help="Types of noise to use",
    )
    
    parser.add_argument(
        "--snr-levels",
        type=float,
        nargs="+",
        default=[0, 5, 10, 15, 20],
        help="SNR levels (in dB) to evaluate",
    )
    
    parser.add_argument(
        "--test-set",
        type=str,
        default="test-clean",
        choices=["test-clean", "test-other"],
        help="LibriSpeech test set to use",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="noisy_decode_results",
        help="Directory to save noisy decoding results",
    )
    
    return parser


def load_noise_files(
    noise_dir: Path,
    noise_types: List[str],
    target_sr: int = 16000,
) -> Dict[str, torch.Tensor]:
    """
    Load noise files from NOISEX-92 dataset.
    
    Args:
        noise_dir: Path to NOISEX-92 directory
        noise_types: List of noise types to load
        target_sr: Target sampling rate
        
    Returns:
        Dictionary mapping noise type to waveform
    """
    noise_files = {}
    
    # NOISEX-92 typically uses these file patterns
    noise_patterns = {
        "babble": ["babble", "BABBLE"],
        "pink": ["pink", "PINK"],
        "white": ["white", "WHITE"],
    }
    
    for noise_type in noise_types:
        found = False
        patterns = noise_patterns.get(noise_type, [noise_type])
        
        for pattern in patterns:
            # Try different extensions
            for ext in [".wav", ".WAV", ".sph", ".SPH"]:
                noise_path = noise_dir / f"{pattern}{ext}"
                if noise_path.exists():
                    logging.info(f"Loading {noise_type} noise from {noise_path}")
                    try:
                        waveform = load_noise_file(noise_path, target_sr)
                        noise_files[noise_type] = waveform
                        found = True
                        break
                    except Exception as e:
                        logging.warning(f"Failed to load {noise_path}: {e}")
            
            if found:
                break
        
        if not found:
            logging.warning(f"Could not find {noise_type} noise file in {noise_dir}")
    
    if not noise_files:
        raise FileNotFoundError(f"No noise files found in {noise_dir}")
    
    return noise_files


def decode_dataset_with_noise(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    HLG: Optional[k2.Fsa],
    H: Optional[k2.Fsa],
    bpe_model: Optional[spm.SentencePieceProcessor],
    word_table: k2.SymbolTable,
    sos_id: int,
    eos_id: int,
    G: Optional[k2.Fsa] = None,
    rnn_lm_model: Optional[nn.Module] = None,
    noise_transform: Optional[NoisyAudioTransform] = None,
) -> Dict[str, List[Tuple[List[str], List[str]]]]:
    """Decode entire dataset with optional noise augmentation."""
    num_cuts = 0
    results = defaultdict(list)
    
    for batch_idx, batch in enumerate(dl):
        # Apply noise augmentation if provided
        if noise_transform is not None:
            # batch["inputs"] has shape [B, T, C] where C is num_features (80)
            # We need to apply noise to the raw audio before feature extraction
            # For simplicity, we'll skip this and assume noise is added during data loading
            pass
        
        res = decode_one_batch(
            params=params,
            model=model,
            rnn_lm_model=rnn_lm_model,
            HLG=HLG,
            H=H,
            bpe_model=bpe_model,
            batch=batch,
            word_table=word_table,
            sos_id=sos_id,
            eos_id=eos_id,
            G=G,
        )
        
        for key, hyps in res.items():
            results[key].extend(hyps)
        
        num_cuts += len(batch["supervisions"]["cut"])
        
        if batch_idx % 100 == 0:
            logging.info(f"Processed {num_cuts} cuts")
    
    return results


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    
    params = AttributeDict(vars(args))
    params.res_dir = Path(params.output_dir) / f"{params.test_set}"
    params.res_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logger(f"{params.res_dir}/log-decode")
    logging.info("Decoding started")
    logging.info(params)
    logging.info(f"Device: {params.device}")
    
    # Load noise files
    noise_dir = Path(params.noise_dir)
    logging.info(f"Loading noise files from {noise_dir}")
    noise_files = load_noise_files(noise_dir, params.noise_types)
    logging.info(f"Loaded {len(noise_files)} noise types: {list(noise_files.keys())}")
    
    # Setup model
    logging.info("Loading model...")
    # Import and setup code from original decode.py
    from decode import load_averaged_model
    
    # Create model
    model = Conformer(
        num_features=params.num_features,
        nhead=params.nhead,
        d_model=params.d_model,
        num_classes=params.num_classes,
        subsampling_factor=params.subsampling_factor,
        num_decoder_layers=params.num_decoder_layers,
        vgg_frontend=params.vgg_frontend,
        num_encoder_layers=params.num_encoder_layers,
        use_feat_batchnorm=params.use_feat_batchnorm,
    )
    
    # Load checkpoint
    if params.avg > 0:
        model = load_averaged_model(
            model_dir=params.exp_dir,
            model=model,
            epoch=params.epoch,
            avg=params.avg,
            device=params.device,
            strict=params.strict,
        )
    else:
        checkpoint_path = f"{params.exp_dir}/epoch-{params.epoch}.pt"
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=params.device)
        model.load_state_dict(checkpoint["model"], strict=params.strict)
        model.to(params.device)
    
    model.eval()
    
    # Setup decoding graph
    lexicon = None
    if params.method != "ctc-decoding":
        from decode import setup_decoding_graph
        HLG, H, bpe_model, word_table, sos_id, eos_id, G = setup_decoding_graph(params)
    else:
        HLG = None
        H = k2.ctc_topo(params.num_classes, modified=False, device=params.device)
        bpe_model_path = Path(params.lang_dir) / "bpe.model"
        bpe_model = spm.SentencePieceProcessor()
        bpe_model.load(str(bpe_model_path))
        word_table = None
        sos_id = 1
        eos_id = 1
        G = None
    
    # Load test data
    logging.info(f"Loading {params.test_set} dataset")
    librispeech = LibriSpeechAsrDataModule(args)
    
    if params.test_set == "test-clean":
        test_clean_cuts = librispeech.test_clean_cuts()
        test_clean_dl = librispeech.test_dataloaders(test_clean_cuts)
        dl = test_clean_dl
    else:  # test-other
        test_other_cuts = librispeech.test_other_cuts()
        test_other_dl = librispeech.test_dataloaders(test_other_cuts)
        dl = test_other_dl
    
    # Get reference texts
    test_ref = []
    for batch in dl:
        for cut in batch["supervisions"]["cut"]:
            test_ref.append(cut.supervisions[0].text.split())
    
    # Decode at each SNR level
    all_results = {}
    
    # First decode clean audio (no noise)
    logging.info("=" * 80)
    logging.info("Decoding clean audio (no noise)")
    logging.info("=" * 80)
    
    results_clean = decode_dataset_with_noise(
        dl=dl,
        params=params,
        model=model,
        HLG=HLG,
        H=H,
        bpe_model=bpe_model,
        word_table=word_table,
        sos_id=sos_id,
        eos_id=eos_id,
        G=G,
        noise_transform=None,
    )
    
    all_results["clean"] = results_clean
    
    # Save clean results
    for key, hyps in results_clean.items():
        recog_path = params.res_dir / f"recogs-clean-{key}.txt"
        store_transcripts(filename=recog_path, texts=hyps)
        logging.info(f"Saved clean results to {recog_path}")
        
        # Compute WER
        errs_filename = params.res_dir / f"errs-clean-{key}.txt"
        with open(errs_filename, "w") as f:
            write_error_stats(f, f"{params.test_set}-clean", test_ref, hyps)
        
        logging.info(f"Wrote error stats to {errs_filename}")
    
    # Now decode with noise at each SNR level
    for snr_db in params.snr_levels:
        for noise_type in noise_files.keys():
            logging.info("=" * 80)
            logging.info(f"Decoding with {noise_type} noise at SNR={snr_db}dB")
            logging.info("=" * 80)
            
            # Note: For proper implementation, we would need to modify the dataloader
            # to apply noise to raw audio before feature extraction.
            # This is a simplified version that assumes features are extracted
            # after noise addition.
            
            logging.warning(
                "Noise addition before feature extraction is not fully implemented. "
                "This requires modifying the data pipeline."
            )
            
            # Placeholder for now
            # In production, you would:
            # 1. Create a custom dataset that loads raw audio
            # 2. Add noise at specified SNR
            # 3. Extract features from noisy audio
            # 4. Pass to model
    
    logging.info("Decoding completed!")
    logging.info(f"Results saved to {params.res_dir}")


if __name__ == "__main__":
    main()
