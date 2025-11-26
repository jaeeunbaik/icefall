# Copyright      2021  Piotr Żelasko
# Copyright      2023  Xiaomi Corporation     (Author: Yifan Yang)
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


class CleanNoisyWrapper:
    """
    Lightweight wrapper to produce guaranteed clean/noisy pairs from the same
    base dataset. It calls the base dataset to get a collated batch (clean)
    and then produces a noisy version by applying an augmentation function to
    the collated audio tensor. This keeps crop/padding alignment identical.

    Returned batch will contain keys:
      - 'audio' (noisy primary audio)
      - 'padding_mask' (same mask)
      - 'clean_audio' (clean audio)
      - 'clean_padding_mask' (same as padding_mask)
      - 'cuts' (passed through)
    """
    def __init__(self, base_dataset, augment_fn=None, snr_range: str = "10,20"):
        self.base = base_dataset
        self.augment_fn = augment_fn
        self.snr_range = snr_range

    def _default_augment(self, audio: torch.Tensor) -> torch.Tensor:
        """Add per-sample Gaussian noise based on SNR range (dB).
        Expects audio shape [B, T]. Returns same shape.
        """
        with torch.no_grad():
            rng = list(map(float, self.snr_range.split(",")))
            min_snr, max_snr = min(rng), max(rng)
            noisy = audio.clone()
            for i in range(audio.size(0)):
                wav = audio[i]
                rms = torch.sqrt(torch.mean(wav ** 2) + 1e-12)
                snr_db = float(torch.empty(1).uniform_(min_snr, max_snr).item())
                snr = 10 ** (snr_db / 20.0)
                noise_std = (rms / snr).item()
                noise = torch.randn_like(wav) * noise_std
                noisy[i] = wav + noise
            return noisy

    def __getitem__(self, cuts):
        # clean batch from base dataset
        clean_batch = self.base[cuts]

        # Expect the base to return 'audio' and 'padding_mask'
        clean_audio = clean_batch.get("audio")
        padding_mask = clean_batch.get("padding_mask")

        if clean_audio is None:
            # Fallback: return base batch unchanged
            return clean_batch

        if self.augment_fn is not None:
            try:
                noisy_audio = self.augment_fn(clean_audio.clone())
            except Exception as e:
                logging.warning(f"augment_fn failed, falling back to default: {e}")
                noisy_audio = self._default_augment(clean_audio)
        else:
            noisy_audio = self._default_augment(clean_audio)

        # Build supervisions from cuts if not already present
        supervisions = clean_batch.get("supervisions")
        if supervisions is None and "cuts" in clean_batch:
            cuts_obj = clean_batch["cuts"]
            # Create basic supervisions dict
            supervisions = {
                'sequence_idx': torch.arange(len(cuts_obj)),
                'start_frame': torch.zeros(len(cuts_obj)),
                'num_frames': torch.tensor([cut.num_samples for cut in cuts_obj]),
                'text': [''] * len(cuts_obj),
            }

        batch = {
            "cuts": clean_batch.get("cuts"),
            # primary audio is noisy (so training can use augmented inputs)
            "audio": noisy_audio,
            "padding_mask": padding_mask,
            "clean_audio": clean_audio,
            "clean_padding_mask": padding_mask,
            "supervisions": supervisions,
        }

        return batch


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class LibriLightDataModule:
    """
    DataModule for SSL experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean
    and test-other).

    It contains all the common data pipeline modules used in SSL
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,

    This class should be derived for specific corpora used in SSL tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR SSL related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies.",
        )

        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/kmeans/medium_split"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=float,
            default=150.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=False,
            help="When enabled, the batches will come from buckets of "
            "similar duration (saves padding frames).",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=30,
            help="The number of buckets for the DynamicBucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=False,
            help="When enabled (=default), the examples will be "
            "shuffled for each epoch.",
        )
        group.add_argument(
            "--drop-last",
            type=str2bool,
            default=True,
            help="Whether to drop last batch. Used by sampler.",
        )
        group.add_argument(
            "--num-workers",
            type=int,
            default=2,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )
        group.add_argument(
            "--do-normalize",
            type=str2bool,
            default=True,
            help="whether to normalize the data",
        )
        group.add_argument(
            "--random-crop",
            type=str2bool,
            default=False,
            help="audio sample rate",
        )

        # Augmentation options (SpecAugment, MUSAN, Gaussian)
        group.add_argument(
            "--enable-spec-aug",
            type=str2bool,
            default=False,
            help="When enabled, use SpecAugment for training dataset.",
        )

        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help=("Used only when --enable-spec-aug is True. "
                  "It specifies the factor for time warping in SpecAugment."),
        )

        group.add_argument(
            "--spec-aug-num-frame-masks",
            type=int,
            default=2,
            help="Used only when --enable-spec-aug is True. Number of time masks to apply.",
        )

        group.add_argument(
            "--spec-aug-features-mask-size",
            type=int,
            default=27,
            help="Used only when --enable-spec-aug is True. Maximum width of frequency masks.",
        )

        group.add_argument(
            "--spec-aug-num-feature-masks",
            type=int,
            default=2,
            help="Used only when --enable-spec-aug is True. Number of frequency masks to apply.",
        )

        group.add_argument(
            "--spec-aug-frames-mask-size",
            type=int,
            default=80,
            help="Used only when --enable-spec-aug is True. Maximum width of time masks.",
        )

        group.add_argument(
            "--enable-musan",
            type=str2bool,
            default=False,
            help=("When enabled, select noise from MUSAN and mix it "
                  "with training dataset."),
        )

        group.add_argument(
            "--enable-gaussian",
            type=str2bool,
            default=True,
            help=("When enabled, apply additive Gaussian noise for noisy samples."),
        )

        group.add_argument(
            "--musan-ratio",
            type=float,
            default=0.5,
            help="Probability of applying MUSAN noise augmentation.",
        )

        group.add_argument(
            "--snr-range",
            type=str,
            default="10,20",
            help="SNR range (dB) for noise augmentation.",
        )
        
        
        group.add_argument(
            "--enable-rir",
            type=str2bool,
            default=False,
            help="When enabled, apply RIR (Room Impulse Response) reverberation augmentation to training data.",
        )
        
        group.add_argument(
            "--rir-prob",
            type=float,
            default=0.5,
            help="Probability of applying RIR reverberation augmentation. "
            "Used only when --enable-rir is True.",
        )
        
        group.add_argument(
            "--rir-early-only",
            type=str2bool,
            default=False,
            help="If True, use only the first 50ms of the RIR impulse response for reverberation. "
            "Used only when --enable-rir is True.",
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sample_rate: float = 16000,
        label_rate: float = 50,
        random_crop: bool = True,
        pad_audio: bool = False,
        num_classes: list = [504],
        do_normalize: bool = True,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        logging.info("About to create train dataset")
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
            logging.info("Self-distillation enabled: wrapping dataset with CleanNoisyWrapper")
            
            # Build augment_fn based on enabled augmentation flags
            augment_fn = None
            
            # Priority 0: RIR reverberation if enabled and available
            if getattr(self.args, 'enable_rir', False):
                rir_manifest_path = Path("data/fbank/rir_recordings.jsonl.gz")
                
                if rir_manifest_path.exists():
                    logging.info(f"RIR manifest found at {rir_manifest_path}, building RIR augmenter")
                    try:
                        from lhotse import load_manifest
                        rir_recordings = load_manifest(rir_manifest_path)
                        logging.info(f"Loaded {len(rir_recordings)} RIR impulse responses")
                        
                        # Get config values before creating closure
                        rir_prob_val = float(getattr(self.args, 'rir_prob', 0.5))
                        early_only_val = getattr(self.args, 'rir_early_only', False)
                        
                        def rir_augment(audio: torch.Tensor) -> torch.Tensor:
                            """Apply RIR reverberation augmentation."""
                            import random
                            import numpy as np
                            from scipy import signal as scipy_signal
                            
                            B, T = audio.shape
                            reverbed = audio.clone()
                            
                            for i in range(B):
                                if random.random() < rir_prob_val:
                                    # Sample random RIR
                                    rir_recording = random.choice(list(rir_recordings))
                                    try:
                                        # Load RIR impulse response
                                        rir_audio = rir_recording.load_audio()  # [channels, samples]
                                        
                                        # Use first channel
                                        if rir_audio.ndim > 1:
                                            rir_ir = rir_audio[0]
                                        else:
                                            rir_ir = rir_audio
                                        
                                        # Apply early-only truncation if requested
                                        if early_only_val:
                                            # Keep only first 50ms (assuming 16kHz)
                                            sample_rate = 16000
                                            early_samples = int(0.05 * sample_rate)
                                            rir_ir = rir_ir[:early_samples]
                                        
                                        # Normalize RIR
                                        rir_ir = rir_ir / np.max(np.abs(rir_ir) + 1e-8)
                                        
                                        # Apply convolution
                                        signal_np = audio[i].cpu().numpy()
                                        reverbed_np = scipy_signal.fftconvolve(signal_np, rir_ir, mode='full')[:T]
                                        
                                        # Normalize output to prevent clipping
                                        max_val = np.max(np.abs(reverbed_np))
                                        if max_val > 1.0:
                                            reverbed_np = reverbed_np / max_val
                                        
                                        reverbed[i] = torch.from_numpy(reverbed_np).float()
                                    except Exception as e:
                                        logging.warning(f"RIR augmentation failed for sample {i}: {e}")
                                        # Keep original if augmentation fails
                            
                            return reverbed
                        
                        augment_fn = rir_augment
                        logging.info(f"RIR augmentation enabled: prob={rir_prob_val}, early_only={early_only_val}")
                    except Exception as e:
                        logging.warning(f"Failed to set up RIR augmentation: {e}")
                        augment_fn = None
                else:
                    logging.warning(f"RIR enabled but manifest not found at {rir_manifest_path}")
                    logging.warning("Please prepare RIR data first")
            
            # Priority 1: MUSAN if enabled and available (and RIR not used)
            if augment_fn is None and getattr(self.args, 'enable_musan', False):
                musan_manifest_dir = Path("data/musan")
                musan_cuts_path = musan_manifest_dir / "cuts.jsonl.gz"
                
                if musan_cuts_path.exists():
                    logging.info(f"MUSAN manifest found at {musan_cuts_path}, building MUSAN augmenter")
                    try:
                        from lhotse import load_manifest_lazy
                        musan_cuts = load_manifest_lazy(musan_cuts_path)
                        
                        @lru_cache(maxsize=10)
                        def _load_musan_audio(cut_id: str):
                            """Load and cache MUSAN audio waveforms."""
                            musan_cut = musan_cuts[cut_id]
                            return musan_cut.load_audio()
                        
                        def musan_augment(audio: torch.Tensor) -> torch.Tensor:
                            """Apply MUSAN noise mixing augmentation."""
                            import random
                            import numpy as np
                            
                            B, T = audio.shape
                            noisy = audio.clone()
                            musan_ratio = float(getattr(self.args, 'musan_ratio', 0.5))
                            snr_range = getattr(self.args, 'snr_range', "10,20")
                            rng = list(map(float, snr_range.split(",")))
                            min_snr, max_snr = min(rng), max(rng)
                            
                            for i in range(B):
                                if random.random() < musan_ratio:
                                    # Sample random MUSAN cut
                                    musan_cut_id = random.choice(list(musan_cuts.ids))
                                    try:
                                        musan_audio = _load_musan_audio(musan_cut_id)  # numpy array
                                        musan_len = musan_audio.shape[0]
                                        
                                        # Trim or repeat to match target length T
                                        if musan_len >= T:
                                            start_idx = random.randint(0, musan_len - T)
                                            noise = musan_audio[start_idx:start_idx + T]
                                        else:
                                            repeats = (T + musan_len - 1) // musan_len
                                            noise = np.tile(musan_audio, repeats)[:T]
                                        
                                        noise_t = torch.from_numpy(noise).float()
                                        
                                        # Compute SNR and scale noise
                                        signal = audio[i]
                                        signal_rms = torch.sqrt(torch.mean(signal ** 2) + 1e-12)
                                        noise_rms = torch.sqrt(torch.mean(noise_t ** 2) + 1e-12)
                                        
                                        target_snr_db = random.uniform(min_snr, max_snr)
                                        target_snr = 10 ** (target_snr_db / 20.0)
                                        scale = (signal_rms / (noise_rms * target_snr)).item()
                                        
                                        noisy[i] = signal + noise_t * scale
                                    except Exception as e:
                                        logging.warning(f"MUSAN augmentation failed for sample {i}: {e}")
                                        # Keep original if augmentation fails
                            
                            return noisy
                        
                        augment_fn = musan_augment
                        logging.info("MUSAN augmentation function created")
                    except Exception as e:
                        logging.warning(f"Failed to set up MUSAN augmentation: {e}")
                        augment_fn = None
                else:
                    logging.warning(f"MUSAN enabled but manifest not found at {musan_cuts_path}")
            
            # Priority 2: Gaussian noise if enabled and MUSAN not used
            if augment_fn is None and getattr(self.args, 'enable_gaussian', True):
                logging.info("Using Gaussian noise augmentation for self-distillation")
                # augment_fn will be None, CleanNoisyWrapper will use default Gaussian
            
            # Wrap the base dataset
            train = CleanNoisyWrapper(
                base_dataset=train,
                augment_fn=augment_fn,
                snr_range=getattr(self.args, 'snr_range', "10,20")
            )
            logging.info("Dataset wrapped with CleanNoisyWrapper for self-distillation")

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                drop_last=self.args.drop_last,
            )
        else:
            logging.info("Using SimpleCutSampler.")
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )
        logging.info("About to create train dataloader")

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        # 'seed' is derived from the current random state, which will have
        # previously been set in the main process.
        seed = torch.randint(0, 100000, ()).item()
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

    def valid_dataloaders(
        self,
        cuts_valid: CutSet,
        sample_rate: float = 16000,
        label_rate: float = 50,
        random_crop: bool = True,
        pad_audio: bool = False,
        num_classes: list = [504],
        do_normalize: bool = True,
    ) -> DataLoader:
        logging.info("About to create dev dataset")
        validate = HubertDataset(
            sample_rate=sample_rate,
            label_rate=label_rate,
            random_crop=random_crop,
            pad_audio=pad_audio,
            num_classes=num_classes,
            do_normalize=do_normalize,
        )
        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.info("About to create dev dataloader")
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=2,
            persistent_workers=False,
        )

        return valid_dl

    def test_dataloaders(
        self,
        cuts: CutSet,
        sample_rate: float = 16000,
        label_rate: float = 50,
        random_crop: bool = True,
        pad_audio: bool = False,
        num_classes: list = [504],
        do_normalize: bool = True,
    ) -> DataLoader:
        logging.debug("About to create test dataset")
        test = HubertDataset(
            sample_rate=sample_rate,
            label_rate=label_rate,
            random_crop=random_crop,
            pad_audio=pad_audio,
            num_classes=num_classes,
            do_normalize=do_normalize,
        )
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.debug("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl

    @lru_cache()
    def small_cuts(self) -> CutSet:
        logging.info("About to get small cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librilight_cuts_medium.jsonl.gz"
        )

    @lru_cache()
    def medium_cuts(self) -> CutSet:
        logging.info("About to get medium cuts")
        
        # First priority: check for fbank features (pre-extracted)
        fbank_dir = Path("data/fbank")
        if fbank_dir.exists():
            # Look for fbank split files (preferred - already have features)
            pattern = str(fbank_dir / "librilight_cuts_medium.*.jsonl.gz")
            import glob as glob_module
            fbank_files = sorted(glob_module.glob(pattern))
            
            if fbank_files:
                logging.info(f"Loading LibriLight medium {len(fbank_files)} fbank feature splits from {fbank_dir}")
                pattern_fbank = re.compile(r"librilight_cuts_medium\.([0-9]+)\.jsonl\.gz")
                idx_filenames = []
                for f in fbank_files:
                    m = pattern_fbank.search(Path(f).name)
                    if m:
                        try:
                            idx = int(m.group(1))
                            idx_filenames.append((idx, f))
                        except Exception:
                            idx_filenames.append((10 ** 9, f))
                
                idx_filenames = sorted(idx_filenames, key=lambda x: x[0])
                sorted_fbank_files = [f[1] for f in idx_filenames]
                return combine(load_manifest_lazy(p) for p in sorted_fbank_files)
            
            # Check for single fbank file
            fbank_file = fbank_dir / "librilight_cuts_medium.jsonl.gz"
            if fbank_file.exists():
                logging.info(f"Loading LibriLight medium fbank features from {fbank_file}")
                return load_manifest_lazy(fbank_file)
        
        # Fallback: raw audio cuts (will need on-the-fly feature extraction)
        logging.info(f"No fbank features found in {fbank_dir}, falling back to raw audio cuts")
        # Try multiple common layouts: `medium_splits/`, `medium_split/`,
        # or manifests directly under `manifest_dir`.
        candidate_dirs = [
            self.args.manifest_dir / "medium_splits",
            self.args.manifest_dir / "medium_split",
            self.args.manifest_dir,
        ]

        filenames = []
        for p in candidate_dirs:
            try:
                if p.exists():
                    if p.is_dir():
                        filenames.extend(glob.glob(f"{p}/librilight_cuts_medium_raw.*.jsonl.gz"))
                    else:
                        # p can be a file path in some setups
                        if re.search(r"librilight_cuts_medium\..*\.jsonl\.gz", str(p.name)):
                            filenames.append(str(p))
            except Exception:
                continue

        # As a last resort, scan manifest_dir recursively for matching files
        if len(filenames) == 0:
            filenames = glob.glob(f"{self.args.manifest_dir}/**/librilight_cuts_medium_raw.*.jsonl.gz", recursive=True)

        pattern = re.compile(r"librilight_cuts_medium_raw\.([0-9]+)\.jsonl\.gz")
        idx_filenames = []
        for f in filenames:
            m = pattern.search(Path(f).name)
            if m:
                try:
                    idx = int(m.group(1))
                    idx_filenames.append((idx, f))
                except Exception:
                    # fallback: add without index
                    idx_filenames.append((10 ** 9, f))

        idx_filenames = sorted(idx_filenames, key=lambda x: x[0])
        sorted_filenames = [f[1] for f in idx_filenames]
        logging.info(f"Loading LibriLight medium {len(sorted_filenames)} splits in lazy mode")
        return combine(load_manifest_lazy(p) for p in sorted_filenames)

    @lru_cache()
    def large_cuts(self) -> CutSet:
        logging.info("About to get large cuts")
        candidate_dirs = [
            self.args.manifest_dir / "large_splits",
            self.args.manifest_dir / "large_split",
            self.args.manifest_dir,
        ]

        filenames = []
        for p in candidate_dirs:
            try:
                if p.exists():
                    if p.is_dir():
                        filenames.extend(glob.glob(f"{p}/librilight_cuts_large.*.jsonl.gz"))
                    else:
                        if re.search(r"librilight_cuts_large\..*\.jsonl\.gz", str(p.name)):
                            filenames.append(str(p))
            except Exception:
                continue

        if len(filenames) == 0:
            filenames = glob.glob(f"{self.args.manifest_dir}/**/librilight_cuts_large.*.jsonl.gz", recursive=True)

        pattern = re.compile(r"librilight_cuts_large\.([0-9]+)\.jsonl\.gz")
        idx_filenames = []
        for f in filenames:
            m = pattern.search(Path(f).name)
            if m:
                try:
                    idx = int(m.group(1))
                    idx_filenames.append((idx, f))
                except Exception:
                    idx_filenames.append((10 ** 9, f))

        idx_filenames = sorted(idx_filenames, key=lambda x: x[0])
        sorted_filenames = [f[1] for f in idx_filenames]
        logging.info(f"Loading LibriLight large {len(sorted_filenames)} splits in lazy mode")

        return combine(load_manifest_lazy(p) for p in sorted_filenames)

    @lru_cache()
    def train_all_shuf_cuts(self) -> CutSet:
        logging.info("About to get the shuffled small, medium and large cuts")
        small_cuts = self.small_cuts()
        medium_cuts = self.medium_cuts()
        large_cuts = self.large_cuts()
        return CutSet.mux(
            small_cuts,
            medium_cuts,
            large_cuts,
            weights=[
                122867,  # len(small_cuts)
                1104071,  # len(medium_cuts)
                11012085,  # len(large_cuts)
            ],
        )

    # LibriSpeech cuts methods for pretraining
    @lru_cache()
    def librispeech_train_clean_100_cuts(self) -> CutSet:
        logging.info("About to get LibriSpeech train-clean-100 cuts")
        # Try both locations: SSL data dir and LibriSpeech ASR data dir
        candidate_paths = [
            self.args.manifest_dir / "librispeech_cuts_train-clean-100.jsonl.gz",
            Path("../../../librispeech/ASR/data/fbank/librispeech_cuts_train-clean-100.jsonl.gz"),
            Path("/home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/data/fbank/librispeech_cuts_train-clean-100.jsonl.gz"),
        ]
        for path in candidate_paths:
            if path.exists():
                logging.info(f"Loading from {path}")
                return load_manifest_lazy(path)
        raise FileNotFoundError(f"Could not find librispeech_cuts_train-clean-100.jsonl.gz in any of: {candidate_paths}")

    @lru_cache()
    def librispeech_train_clean_360_cuts(self) -> CutSet:
        logging.info("About to get LibriSpeech train-clean-360 cuts")
        candidate_paths = [
            self.args.manifest_dir / "librispeech_cuts_train-clean-360.jsonl.gz",
            Path("../../../librispeech/ASR/data/fbank/librispeech_cuts_train-clean-360.jsonl.gz"),
            Path("/home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/data/fbank/librispeech_cuts_train-clean-360.jsonl.gz"),
        ]
        for path in candidate_paths:
            if path.exists():
                logging.info(f"Loading from {path}")
                return load_manifest_lazy(path)
        raise FileNotFoundError(f"Could not find librispeech_cuts_train-clean-360.jsonl.gz in any of: {candidate_paths}")

    @lru_cache()
    def librispeech_train_other_500_cuts(self) -> CutSet:
        logging.info("About to get LibriSpeech train-other-500 cuts")
        candidate_paths = [
            self.args.manifest_dir / "librispeech_cuts_train-other-500.jsonl.gz",
            Path("../../../librispeech/ASR/data/fbank/librispeech_cuts_train-other-500.jsonl.gz"),
            Path("/home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/data/fbank/librispeech_cuts_train-other-500.jsonl.gz"),
        ]
        for path in candidate_paths:
            if path.exists():
                logging.info(f"Loading from {path}")
                return load_manifest_lazy(path)
        raise FileNotFoundError(f"Could not find librispeech_cuts_train-other-500.jsonl.gz in any of: {candidate_paths}")

    @lru_cache()
    def librispeech_train_all_shuf_cuts(self) -> CutSet:
        logging.info("About to get LibriSpeech train-all-shuf cuts (train-clean-100, train-clean-360, train-other-500)")
        # First try to load the pre-shuffled combined file
        candidate_paths = [
            self.args.manifest_dir / "librispeech_cuts_train-all-shuf.jsonl.gz",
            Path("../../../librispeech/ASR/data/fbank/librispeech_cuts_train-all-shuf.jsonl.gz"),
            Path("/home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/data/fbank/librispeech_cuts_train-all-shuf.jsonl.gz"),
        ]
        for path in candidate_paths:
            if path.exists():
                logging.info(f"Loading pre-shuffled LibriSpeech cuts from {path}")
                return load_manifest_lazy(path)
        
        # Fallback: load and combine individual files
        logging.info("Pre-shuffled file not found, combining individual train sets")
        try:
            train_clean_100 = self.librispeech_train_clean_100_cuts()
            train_clean_360 = self.librispeech_train_clean_360_cuts()
            train_other_500 = self.librispeech_train_other_500_cuts()
            return combine([train_clean_100, train_clean_360, train_other_500])
        except FileNotFoundError as e:
            logging.error(f"Failed to load LibriSpeech training cuts: {e}")
            raise

    @lru_cache()
    def librispeech_dev_clean_cuts(self) -> CutSet:
        logging.info("About to get LibriSpeech dev-clean cuts")
        candidate_paths = [
            self.args.manifest_dir / "librispeech_cuts_dev-clean.jsonl.gz",
            Path("../../../librispeech/ASR/data/fbank/librispeech_cuts_dev-clean.jsonl.gz"),
            Path("/home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/data/fbank/librispeech_cuts_dev-clean.jsonl.gz"),
        ]
        for path in candidate_paths:
            if path.exists():
                logging.info(f"Loading from {path}")
                return load_manifest_lazy(path)
        raise FileNotFoundError(f"Could not find librispeech_cuts_dev-clean.jsonl.gz in any of: {candidate_paths}")

    @lru_cache()
    def librispeech_dev_other_cuts(self) -> CutSet:
        logging.info("About to get LibriSpeech dev-other cuts")
        candidate_paths = [
            self.args.manifest_dir / "librispeech_cuts_dev-other.jsonl.gz",
            Path("../../../librispeech/ASR/data/fbank/librispeech_cuts_dev-other.jsonl.gz"),
            Path("/home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/data/fbank/librispeech_cuts_dev-other.jsonl.gz"),
        ]
        for path in candidate_paths:
            if path.exists():
                logging.info(f"Loading from {path}")
                return load_manifest_lazy(path)
        raise FileNotFoundError(f"Could not find librispeech_cuts_dev-other.jsonl.gz in any of: {candidate_paths}")

    @lru_cache()
    def librispeech_test_clean_cuts(self) -> CutSet:
        logging.info("About to get LibriSpeech test-clean cuts")
        candidate_paths = [
            self.args.manifest_dir / "librispeech_cuts_test-clean.jsonl.gz",
            Path("../../../librispeech/ASR/data/fbank/librispeech_cuts_test-clean.jsonl.gz"),
            Path("/home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/data/fbank/librispeech_cuts_test-clean.jsonl.gz"),
        ]
        for path in candidate_paths:
            if path.exists():
                logging.info(f"Loading from {path}")
                return load_manifest_lazy(path)
        raise FileNotFoundError(f"Could not find librispeech_cuts_test-clean.jsonl.gz in any of: {candidate_paths}")

    @lru_cache()
    def librispeech_test_other_cuts(self) -> CutSet:
        logging.info("About to get LibriSpeech test-other cuts")
        candidate_paths = [
            self.args.manifest_dir / "librispeech_cuts_test-other.jsonl.gz",
            Path("../../../librispeech/ASR/data/fbank/librispeech_cuts_test-other.jsonl.gz"),
            Path("/home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/data/fbank/librispeech_cuts_test-other.jsonl.gz"),
        ]
        for path in candidate_paths:
            if path.exists():
                logging.info(f"Loading from {path}")
                return load_manifest_lazy(path)
        raise FileNotFoundError(f"Could not find librispeech_cuts_test-other.jsonl.gz in any of: {candidate_paths}")
