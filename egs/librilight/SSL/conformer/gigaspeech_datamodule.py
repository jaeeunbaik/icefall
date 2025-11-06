#!/usr/bin/env python3
"""
GigaSpeech DataModule for SSL/Self-Distillation Training

This module provides a simple way to use GigaSpeech XL dataset
with the same SSL training pipeline as LibriLight.
"""

import argparse
import glob
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from dataset import HubertDataset
from lhotse import CutSet, combine, load_manifest_lazy
from lhotse.dataset import DynamicBucketingSampler, SimpleCutSampler
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class GigaSpeechDataModule:
    """
    DataModule for GigaSpeech XL SSL experiments.
    Similar to LibriLightDataModule but adapted for GigaSpeech structure.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="GigaSpeech SSL data options",
            description="Options for loading GigaSpeech XL data",
        )

        group.add_argument(
            "--gigaspeech-dir",
            type=Path,
            default=Path("/home/hdd2/jenny/ASRToolkit/icefall/egs/gigaspeech/ASR/data/fbank/gigaspeech_XL_split"),
            help="Path to GigaSpeech XL split directory with fbank features.",
        )
        
        group.add_argument(
            "--max-duration",
            type=float,
            default=200.0,
            help="Maximum pooled recordings duration (seconds) in a single batch.",
        )
        
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="Use bucketing sampler for efficient batching.",
        )
        
        group.add_argument(
            "--num-buckets",
            type=int,
            default=30,
            help="Number of buckets for DynamicBucketingSampler.",
        )
        
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="Shuffle examples for each epoch.",
        )
        
        group.add_argument(
            "--drop-last",
            type=str2bool,
            default=True,
            help="Whether to drop last batch.",
        )
        
        group.add_argument(
            "--num-workers",
            type=int,
            default=2,
            help="Number of training dataloader workers.",
        )

    @lru_cache()
    def xl_cuts(self) -> CutSet:
        """Load GigaSpeech XL cuts with pre-computed fbank features."""
        logging.info("Loading GigaSpeech XL cuts with fbank features")
        
        gigaspeech_dir = Path(self.args.gigaspeech_dir)
        
        # Look for fbank feature files (gigaspeech_cuts_XL.*.jsonl.gz)
        pattern = str(gigaspeech_dir / "gigaspeech_cuts_XL.*.jsonl.gz")
        fbank_files = sorted(glob.glob(pattern))
        
        if not fbank_files:
            raise FileNotFoundError(
                f"No GigaSpeech XL fbank files found in {gigaspeech_dir}\n"
                f"Looking for pattern: gigaspeech_cuts_XL.*.jsonl.gz"
            )
        
        # Sort by split number
        pattern_re = re.compile(r"gigaspeech_cuts_XL\.(\d+)\.jsonl\.gz")
        idx_filenames = []
        for f in fbank_files:
            m = pattern_re.search(Path(f).name)
            if m:
                idx = int(m.group(1))
                idx_filenames.append((idx, f))
        
        idx_filenames = sorted(idx_filenames, key=lambda x: x[0])
        sorted_files = [f[1] for f in idx_filenames]
        
        logging.info(f"Found {len(sorted_files)} GigaSpeech XL splits")
        logging.info(f"Loading splits: {sorted_files[0]} ... {sorted_files[-1]}")
        
        return combine(load_manifest_lazy(p) for p in sorted_files)

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sample_rate: float = 16000,
        label_rate: float = 50,
        random_crop: bool = False,
        pad_audio: bool = False,
        num_classes: list = [504],
        do_normalize: bool = True,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        """Create training dataloader for GigaSpeech XL."""
        logging.info("Creating GigaSpeech XL train dataset")
        
        train = HubertDataset(
            sample_rate=sample_rate,
            label_rate=label_rate,
            random_crop=random_crop,
            pad_audio=pad_audio,
            num_classes=num_classes,
            do_normalize=do_normalize,
        )

        # Wrap with CleanNoisyWrapper if self-distillation is enabled
        if getattr(self.args, 'enable_self_distillation', False):
            from ssl_datamodule import CleanNoisyWrapper
            logging.info("Self-distillation enabled: wrapping with CleanNoisyWrapper")
            train = CleanNoisyWrapper(
                base_dataset=train,
                augment_fn=None,  # Use default Gaussian noise
                snr_range=getattr(self.args, 'snr_range', "10,20")
            )

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler")
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                drop_last=self.args.drop_last,
            )
        else:
            logging.info("Using SimpleCutSampler")
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        seed = torch.randint(0, 100000, ()).item()
        
        from ssl_datamodule import _SeedWorkers
        worker_init_fn = _SeedWorkers(seed)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl


def test_gigaspeech_loading():
    """Simple test to verify GigaSpeech data loading."""
    class Args:
        gigaspeech_dir = Path("/home/hdd2/jenny/ASRToolkit/icefall/egs/gigaspeech/ASR/data/fbank/gigaspeech_XL_split")
        max_duration = 200.0
        bucketing_sampler = True
        num_buckets = 30
        shuffle = True
        drop_last = True
        num_workers = 2
        enable_self_distillation = False
    
    dm = GigaSpeechDataModule(Args())
    cuts = dm.xl_cuts()
    
    print(f"Loaded {len(cuts)} cuts from GigaSpeech XL")
    print(f"First cut: {next(iter(cuts))}")
    print("✓ GigaSpeech data loading successful!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_gigaspeech_loading()
