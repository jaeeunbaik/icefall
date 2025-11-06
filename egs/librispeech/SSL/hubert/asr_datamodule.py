# Copyright      2021  Piotr Żelasko
# Copyright      2024  Xiaomi Corporation     (Author: Yifan Yang)
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
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from dataset import HubertAsrDataset
from lhotse import CutSet, load_manifest_lazy, load_manifest
from lhotse.dataset.collation import read_audio_from_cuts
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

        batch = {
            "cuts": clean_batch.get("cuts"),
            # primary audio is noisy (so training can use augmented inputs)
            "audio": noisy_audio,
            "padding_mask": padding_mask,
            "clean_audio": clean_audio,
            "clean_padding_mask": padding_mask,
        }

        # Pass through any other keys from clean_batch that are useful
        for k in ("supervisions",):
            if k in clean_batch:
                batch[k] = clean_batch[k]

        return batch


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class LibriSpeechAsrDataModule:
    """
    DataModule for ASR experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean
    and test-other).

    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,

    This class should be derived for specific corpora used in ASR tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies.",
        )
        group.add_argument(
            "--full-libri",
            type=str2bool,
            default=True,
            help="When enabled use 960h LibriSpeech. " "Otherwise, use 100h subset.",
        )

        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("/home/hdd2/jenny/ASRToolkit/icefall/egs/librilight/SSL/data/kmeans/medium_split"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=float,
            default=200.0,
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
                  "It specifies the factor for time warping in SpecAugment. "
                  "Larger values mean more warping. "
                  "A value less than 1 means to disable time warp."),
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
            "--musan-ratio",
            type=float,
            default=0.5,
            help="Probability of applying MUSAN noise augmentation.",
        )

        group.add_argument(
            "--enable-gaussian",
            type=str2bool,
            default=True,
            help=("When enabled, apply additive Gaussian noise for noisy samples. "
                  "This is the default behavior for lightweight augmentation."),
        )

        group.add_argument(
            "--snr-range",
            type=str,
            default="10,20",
            help="SNR range (dB) for default additive-noise augmentation when creating noisy samples.",
        )



    def train_dataloaders(
        self,
        cuts_train: CutSet,
        do_normalize: bool,
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

        # If self-distillation / siam-style training is enabled, wrap the
        # base Hubert dataset so that each batch contains a clean/noisy pair
        if getattr(self.args, 'enable_self_distillation', False):
            logging.info("Creating CleanNoisyWrapper for self-distillation (aligned clean/noisy pairs)")
            base_ds = HubertAsrDataset(do_normalize=do_normalize)
            # Decide augment_fn: default is additive Gaussian noise. If
            # --enable-musan is True and MUSAN manifest exists, create an
            # augment_fn that mixes sampled MUSAN audio at a random SNR.
            augment_fn = None
            # If MUSAN requested, prefer it when available
            if getattr(self.args, 'enable_musan', False):
                # Try common locations for a MUSAN cuts manifest
                musan_paths = [
                    Path("data/fbank/musan_cuts.jsonl.gz"),
                    Path("data/musan_cuts.jsonl.gz"),
                    self.args.manifest_dir / "musan_cuts.jsonl.gz",
                ]
                musan_cuts = None
                for p in musan_paths:
                    try:
                        if p.exists():
                            logging.info(f"Loading MUSAN manifest from {p}")
                            musan_cuts = load_manifest_lazy(p)
                            break
                    except Exception:
                        continue

                if musan_cuts is not None:
                    logging.info("MUSAN manifest found — using MUSAN augment_fn for noisy samples")

                    # Small LRU cache for loaded musan audio to avoid repeated I/O
                    from functools import lru_cache

                    @lru_cache(maxsize=256)
                    def _load_musan_audio(idx: int):
                        # read_audio_from_cuts expects a CutSet-like object or list
                        try:
                            audio_list, sr = read_audio_from_cuts(musan_cuts[idx: idx + 1])
                            wav = audio_list[0]
                        except Exception:
                            # fallback empty tensor
                            wav = torch.tensor([])
                        return wav

                    def musan_augment(batch_audio: torch.Tensor) -> torch.Tensor:
                        # batch_audio: [B, T]
                        rng = list(map(float, getattr(self.args, 'snr_range', '10,20').split(',')))
                        min_snr, max_snr = min(rng), max(rng)
                        noisy = batch_audio.clone()
                        B, T = batch_audio.size(0), batch_audio.size(1)
                        for i in range(B):
                            # sample a random musan cut index
                            idx = int(torch.randint(0, len(musan_cuts), (1,)).item())
                            musan_wav = _load_musan_audio(idx)
                            if musan_wav.numel() == 0:
                                # fallback to gaussian for this sample
                                wav = batch_audio[i]
                                rms = torch.sqrt(torch.mean(wav ** 2) + 1e-12)
                                snr_db = float(torch.empty(1).uniform_(min_snr, max_snr).item())
                                snr = 10 ** (snr_db / 20.0)
                                noise_std = (rms / snr).item()
                                noise = torch.randn_like(wav) * noise_std
                                noisy[i] = wav + noise
                                continue

                            # ensure musan_wav is 1d tensor
                            if musan_wav.dim() == 2:
                                musan_wav = musan_wav.mean(-1)

                            # trim or repeat musan to match length T
                            if musan_wav.numel() >= T:
                                start = torch.randint(0, musan_wav.numel() - T + 1, (1,)).item()
                                musan_seg = musan_wav[start: start + T]
                            else:
                                # repeat to cover length
                                repeats = int((T + musan_wav.numel() - 1) // musan_wav.numel())
                                musan_rep = musan_wav.repeat(repeats)[:T]
                                musan_seg = musan_rep

                            wav = batch_audio[i]
                            rms = torch.sqrt(torch.mean(wav ** 2) + 1e-12)
                            musan_rms = torch.sqrt(torch.mean(musan_seg ** 2) + 1e-12)
                            snr_db = float(torch.empty(1).uniform_(min_snr, max_snr).item())
                            snr = 10 ** (snr_db / 20.0)
                            # scale musan to achieve desired SNR wrt speech rms
                            if musan_rms > 0:
                                scale = (rms / snr) / musan_rms
                                noise = musan_seg * scale
                            else:
                                noise = torch.randn_like(wav) * (rms / snr).item()
                            noisy[i] = wav + noise
                        return noisy

                    augment_fn = musan_augment
                else:
                    logging.warning("--enable-musan specified but MUSAN manifest not found")

            # If MUSAN not chosen, and Gaussian explicitly enabled, use Gaussian
            if augment_fn is None and getattr(self.args, 'enable_gaussian', True):
                def gaussian_augment(batch_audio: torch.Tensor) -> torch.Tensor:
                    # mirrors CleanNoisyWrapper._default_augment but as a callable
                    with torch.no_grad():
                        rng = list(map(float, getattr(self.args, 'snr_range', '10,20').split(',')))
                        min_snr, max_snr = min(rng), max(rng)
                        noisy = batch_audio.clone()
                        for i in range(batch_audio.size(0)):
                            wav = batch_audio[i]
                            rms = torch.sqrt(torch.mean(wav ** 2) + 1e-12)
                            snr_db = float(torch.empty(1).uniform_(min_snr, max_snr).item())
                            snr = 10 ** (snr_db / 20.0)
                            noise_std = (rms / snr).item()
                            noise = torch.randn_like(wav) * noise_std
                            noisy[i] = wav + noise
                        return noisy

                augment_fn = gaussian_augment
            elif augment_fn is None:
                # Neither MUSAN nor Gaussian requested/available: keep audio unchanged
                logging.info("No augmentation requested: noisy == clean (identity)")
                augment_fn = lambda x: x

            train = CleanNoisyWrapper(base_ds, augment_fn=augment_fn, snr_range=getattr(self.args, 'snr_range', '10,20'))
        else:
            train = HubertAsrDataset(do_normalize=do_normalize)

        # Log whether SpecAugment is requested; actual SpecAugment application
        # (feature-level) should be handled at the feature extraction stage.
        if getattr(self.args, 'enable_spec_aug', False):
            logging.info("SpecAugment enabled (note: applied at feature level by front-end if implemented)")

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

    def valid_dataloaders(self, cuts_valid: CutSet, do_normalize: bool) -> DataLoader:
        logging.info("About to create dev dataset")
        validate = HubertAsrDataset(do_normalize=do_normalize)
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

    def test_dataloaders(self, cuts: CutSet, do_normalize: bool) -> DataLoader:
        logging.debug("About to create test dataset")
        test = HubertAsrDataset(do_normalize=do_normalize)
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
    def train_clean_100_cuts(self) -> CutSet:
        logging.info("About to get train-clean-100 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librilight_cuts_train-clean-100.jsonl.gz"
        )

    @lru_cache()
    def train_clean_360_cuts(self) -> CutSet:
        logging.info("About to get train-clean-360 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librilight_cuts_train-clean-360.jsonl.gz"
        )

    @lru_cache()
    def train_other_500_cuts(self) -> CutSet:
        logging.info("About to get train-other-500 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librilight_cuts_train-other-500.jsonl.gz"
        )

    @lru_cache()
    def train_all_shuf_cuts(self) -> CutSet:
        logging.info(
            "About to get the shuffled train-clean-100, \
            train-clean-360 and train-other-500 cuts"
        )
        train_clean_100_cuts = self.train_clean_100_cuts()
        train_clean_360_cuts = self.train_clean_360_cuts()
        train_other_500_cuts = self.train_other_500_cuts()
        return CutSet.mux(
            train_clean_100_cuts,
            train_clean_360_cuts,
            train_other_500_cuts,
            weights=[
                28539,  # len(train_clean_100_cuts)
                104014,  # len(train_clean_360_cuts)
                148688,  # len(train_other_500_cuts)
            ],
        )

    @lru_cache()
    def dev_clean_cuts(self) -> CutSet:
        logging.info("About to get dev-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_dev-clean.jsonl.gz"
        )

    @lru_cache()
    def dev_other_cuts(self) -> CutSet:
        logging.info("About to get dev-other cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_dev-other.jsonl.gz"
        )

    @lru_cache()
    def test_clean_cuts(self) -> CutSet:
        logging.info("About to get test-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_test-clean.jsonl.gz"
        )

    @lru_cache()
    def test_other_cuts(self) -> CutSet:
        logging.info("About to get test-other cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_test-other.jsonl.gz"
        )
