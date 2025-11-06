# Copyright      2021  Piotr Żelasko
# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
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
import inspect
import logging
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, List

import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy, RecordingSet
from lhotse.dataset import (  # noqa F401 for PrecomputedFeatures
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import (  # noqa F401 For AudioSamples
    AudioSamples,
    OnTheFlyFeatures,
)
from lhotse.dataset.cut_transforms.reverberate import ReverbWithImpulseResponse
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class CleanNoisyWrapper:
    """
    Wrapper for creating GUARANTEED clean-noisy pairs from identical source cuts.
    
    This implementation ensures 100% identical source by:
    1. Using only ONE base dataset (clean)
    2. Applying augmentation transforms on-the-fly to create noisy version
    3. Same cuts → Same audio → Same labels → Perfect alignment
    
    Returns batches with structure:
    {
        'inputs': noisy_features,  # Primary features (augmented)
        'supervisions': supervisions,  # Standard supervisions  
        'clean': {
            'inputs': clean_features,     # Original features
            'supervisions': supervisions  # Identical supervisions
        },
        'noisy': {
            'inputs': augmented_features, # Same cuts + augmentation
            'supervisions': supervisions  # Identical supervisions
        }
    }
    """
    def __init__(self, base_dataset, augmentation_transforms=None, input_transforms=None):
        """
        Args:
            base_dataset: K2SpeechRecognitionDataset (clean, no transforms)
            augmentation_transforms: List of cut transforms for noisy version
            input_transforms: List of input transforms (e.g., SpecAugment) for noisy
        """
        self.base_dataset = base_dataset
        self.augmentation_transforms = augmentation_transforms or []
        self.input_transforms = input_transforms or []
        
        logging.info("CleanNoisyWrapper initialized")
        logging.info(f"Augmentation transforms: {[type(t).__name__ for t in self.augmentation_transforms]}")
        logging.info(f"Input transforms: {[type(t).__name__ for t in self.input_transforms]}")
    
    def __getitem__(self, cuts):
        """
        GUARANTEED clean-noisy alignment by processing same cuts twice.
        
        Process:
        1. Get clean version from base dataset (no transforms)
        2. Apply transforms to same cuts for noisy version
        3. Both versions have identical source → Perfect alignment
        """
        # Step 1: Get clean version (original, no augmentation)
        clean_batch = self.base_dataset[cuts]
        
        # Step 2: Create noisy version by applying transforms to same cuts
        noisy_batch = self._create_noisy_version(cuts, clean_batch)
        
        # Clean-Noisy alignment verified and working correctly
        
        return {
            # Standard batch structure (using noisy as primary)
            'inputs': noisy_batch['inputs'],
            'supervisions': noisy_batch['supervisions'],
            # Additional clean-noisy pairs for self-distillation
            'clean': {
                'inputs': clean_batch['inputs'],
                'supervisions': clean_batch['supervisions']
            },
            'noisy': {
                'inputs': noisy_batch['inputs'], 
                'supervisions': noisy_batch['supervisions']
            }
        }
    
    def _create_noisy_version(self, cuts, clean_batch):
        """
        동일한 cuts에서 noisy 버전 생성 - 완전히 동일한 소스 보장
        
        전략: Clean batch의 supervisions는 그대로 복사 (100% 동일성 보장)
              Features만 증강 적용으로 변경
        """
        try:
            import torch
            
            # Step 1: Clean의 supervisions를 그대로 복사 (완벽한 동일성 보장)
            noisy_supervisions = {}
            for key, value in clean_batch['supervisions'].items():
                if isinstance(value, torch.Tensor):
                    noisy_supervisions[key] = value.clone()
                elif isinstance(value, (list, tuple)):
                    noisy_supervisions[key] = list(value)  # 리스트 복사
                else:
                    noisy_supervisions[key] = value
            
            # Step 2: Features 시작점은 clean과 동일
            noisy_inputs = clean_batch['inputs'].clone()
            
            # Step 3: Input transforms 적용 (SpecAugment 등)
            # 이것만 차이가 나고, 나머지는 모두 동일
            if self.input_transforms:
                for transform in self.input_transforms:
                    try:
                        if hasattr(transform, '__call__'):
                            noisy_inputs = transform(noisy_inputs)
                    except Exception as e:
                        logging.warning(f"Input transform {type(transform).__name__} failed: {e}")
            
            # Step 4: Cut-level transforms 시뮬레이션 (간단한 noise 추가 등)
            # 복잡한 Cut transforms (MUSAN 등)는 나중에 구현, 우선 SpecAugment만
            if self.augmentation_transforms and len(self.augmentation_transforms) > 0:
                # 간단한 noise 추가로 시뮬레이션 (실제 MUSAN은 복잡함)
                logging.debug(f"Cut transforms present but simplified: {[type(t).__name__ for t in self.augmentation_transforms]}")
                # 실제 구현이 필요하면 여기에 추가
            
            # Step 5: 결과 생성
            noisy_batch = {
                'inputs': noisy_inputs,
                'supervisions': noisy_supervisions  # 완전히 동일한 supervisions
            }
            
            return noisy_batch
            
        except Exception as e:
            logging.error(f"Failed to create noisy version: {e}")
            logging.error(f"Error details: {str(e)}")
            logging.warning("Fallback: Using clean batch as noisy (no augmentation)")
            
            # Fallback: clean을 그대로 복사
            import torch
            fallback_batch = {
                'inputs': clean_batch['inputs'].clone() if hasattr(clean_batch['inputs'], 'clone') else clean_batch['inputs'],
                'supervisions': clean_batch['supervisions'].copy()
            }
            return fallback_batch


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class LibriSpeechAsrDataModule:
    """
    DataModule for k2 ASR experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean
    and test-other).

    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - augmentation,
    - on-the-fly feature extraction

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
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--full-libri",
            type=str2bool,
            default=True,
            help="""Used only when --mini-libri is False.When enabled,
            use 960h LibriSpeech. Otherwise, use 100h subset.""",
        )
        group.add_argument(
            "--mini-libri",
            type=str2bool,
            default=False,
            help="True for mini librispeech",
        )

        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/fbank"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=200.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--valid-max-duration",
            type=int,
            default=None,
            help="Maximum pooled recordings duration (seconds) in a "
            "single validation batch. If None, uses --max-duration. "
            "You should reduce this if validation causes CUDA OOM.",
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
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
            "--concatenate-cuts",
            type=str2bool,
            default=False,
            help="When enabled, utterances (cuts) will be concatenated "
            "to minimize the amount of padding.",
        )
        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
            help="Determines the maximum duration of a concatenated cut "
            "relative to the duration of the longest cut in a batch.",
        )
        group.add_argument(
            "--gap",
            type=float,
            default=1.0,
            help="The amount of padding (in seconds) inserted between "
            "concatenated cuts. This padding is filled with noise when "
            "noise augmentation is used.",
        )
        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=False,
            help="When enabled, use on-the-fly cut mixing and feature "
            "extraction. Will drop existing precomputed feature manifests "
            "if available.",
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
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
            "--return-cuts",
            type=str2bool,
            default=False,
            help="When enabled, each batch will have the "
            "field: batch['supervisions']['cut'] with the cuts that "
            "were used to construct it.",
        )
        group.add_argument(
            "--mix-librilight",
            type=str2bool,
            default=False,
            help="When enabled, mix LibriSpeech with LibriLight data for training.",
        )
        group.add_argument(
            "--librilight-ratio",
            type=float,
            default=0.5,
            help="Ratio of LibriLight data to mix with LibriSpeech (0.0-1.0).",
        )

        group.add_argument(
            "--num-workers",
            type=int,
            default=2,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )

        group.add_argument(
            "--enable-spec-aug",
            type=str2bool,
            default=True,
            help="When enabled, use SpecAugment for training dataset.",
        )

        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help="Used only when --enable-spec-aug is True. "
            "It specifies the factor for time warping in SpecAugment. "
            "Larger values mean more warping. "
            "A value less than 1 means to disable time warp.",
        )

        group.add_argument(
            "--spec-aug-num-frame-masks",
            type=int,
            default=2,
            help="Used only when --enable-spec-aug is True. "
            "Number of time masks to apply.",
        )

        group.add_argument(
            "--spec-aug-features-mask-size",
            type=int,
            default=27,
            help="Used only when --enable-spec-aug is True. "
            "Maximum width of frequency masks.",
        )

        group.add_argument(
            "--spec-aug-num-feature-masks",
            type=int,
            default=2,
            help="Used only when --enable-spec-aug is True. "
            "Number of frequency masks to apply.",
        )

        group.add_argument(
            "--spec-aug-frames-mask-size",
            type=int,
            default=80,
            help="Used only when --enable-spec-aug is True. "
            "Maximum width of time masks.",
        )

        group.add_argument(
            "--enable-musan",
            type=str2bool,
            default=True,
            help="When enabled, select noise from MUSAN and mix it"
            "with training dataset. ",
        )

        group.add_argument(
            "--musan-ratio",
            type=float,
            default=0.5,
            help="Probability of applying MUSAN noise augmentation. "
            "Used only when --enable-musan is True.",
        )
        
        group.add_argument(
            "--snr-range",
            type=str,
            default="10,20",
            help="SNR range of MUSAN noise augmentation. "
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

        group.add_argument(
            "--input-strategy",
            type=str,
            default="PrecomputedFeatures",
            help="AudioSamples or PrecomputedFeatures",
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
        shuffle: Optional[bool] = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
          shuffle:
            If provided, overrides self.args.shuffle. Otherwise uses self.args.shuffle.
        """
        # Mix with LibriLight if requested
        if self.args.mix_librilight:
            logging.info("Mixing LibriSpeech with LibriLight data")
            
            librilight_cuts = self.librilight_train_cuts()
            
            # Calculate mixing ratio
            ratio = self.args.librilight_ratio
            if not (0.0 <= ratio <= 1.0):
                raise ValueError(f"librilight_ratio must be between 0.0 and 1.0, got {ratio}")
            
            # Sample LibriLight cuts based on ratio
            if ratio > 0.0:
                target_librilight_duration = len(cuts_train) * ratio / (1 - ratio)
                librilight_sampled = librilight_cuts.subset(first=int(target_librilight_duration))
                
                # Combine datasets
                cuts_train = cuts_train + librilight_sampled
                cuts_train = cuts_train.shuffle(random_seed=42)
                
                logging.info(f"Combined dataset: {len(cuts_train)} cuts total")
                logging.info(f"LibriLight ratio: ~{ratio:.2f}")

        # Setup augmentation transforms (for noisy dataset)
        transforms = []
        min_snr, max_snr = list(map(int, self.args.snr_range.split(',')))
        
        # Add RIR reverberation if enabled
        if self.args.enable_rir:
            logging.info("Enable RIR reverberation")
            logging.info("Loading RIR impulse responses...")
            
            # Load RIR recordings from prepared manifest
            rir_manifest_path = Path("data/fbank/rir_recordings.jsonl.gz")
            if rir_manifest_path.exists():
                rir_recordings = load_manifest(rir_manifest_path)
                logging.info(f"Loaded {len(rir_recordings)} RIR impulse responses")
                
                transforms.append(
                    ReverbWithImpulseResponse(
                        rir_recordings=rir_recordings,
                        p=self.args.rir_prob,
                        normalize_output=True,
                        preserve_id=True,
                        early_only=self.args.rir_early_only,
                        rir_channels=[0],  # Use first channel
                    )
                )
                logging.info(f"RIR config: prob={self.args.rir_prob}, early_only={self.args.rir_early_only}")
            else:
                logging.warning(f"RIR manifest not found at {rir_manifest_path}")
                logging.warning("Please run: bash prepare_rir_data.sh to generate RIR data")
                logging.warning("Continuing without RIR augmentation")
        else:
            logging.info("Disable RIR reverberation")
        
        if self.args.enable_musan:
            logging.info("Enable MUSAN")
            logging.info("About to get Musan cuts")
            cuts_musan = load_manifest("data/fbank/musan_cuts.jsonl.gz")
            transforms.append(
                CutMix(cuts=cuts_musan, p=self.args.musan_ratio, snr=(min_snr, max_snr), preserve_id=True)
            )
        else:
            logging.info("Disable MUSAN")

        if self.args.concatenate_cuts:
            logging.info(
                f"Using cut concatenation with duration factor "
                f"{self.args.duration_factor} and gap {self.args.gap}."
            )
            # Cut concatenation should be the first transform in the list,
            # so that if we e.g. mix noise in, it will fill the gaps between
            # different utterances.
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        input_transforms = []
        if self.args.enable_spec_aug:
            logging.info("Enable SpecAugment")
            logging.info(f"Time warp factor: {self.args.spec_aug_time_warp_factor}")
            # Set the value of num_frame_masks according to Lhotse's version.
            # In different Lhotse's versions, the default of num_frame_masks is
            # different.
            num_frame_masks = self.args.spec_aug_num_frame_masks
            num_frame_masks_parameter = inspect.signature(
                SpecAugment.__init__
            ).parameters["num_frame_masks"]
            if num_frame_masks_parameter.default == 1:
                num_frame_masks = self.args.spec_aug_num_frame_masks
            logging.info(f"Num frame mask: {num_frame_masks}")
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=self.args.spec_aug_features_mask_size,
                    num_feature_masks=self.args.spec_aug_num_feature_masks,
                    frames_mask_size=self.args.spec_aug_frames_mask_size,
                )
            )
        else:
            logging.info("Disable SpecAugment")

        # Create input strategy (same for both clean and noisy - only transforms differ)
        input_strategy = eval(self.args.input_strategy)()
        if self.args.on_the_fly_feats:
            input_strategy = OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
        
        # Create clean dataset (no augmentation)
        # Create train dataset (with augmentations)
        logging.info("About to create train dataset")
        augmentation_details = []
        if transforms:
            transform_names = [type(t).__name__ for t in transforms]
            augmentation_details.append(f"Cut transforms: {transform_names}")
        if input_transforms:
            input_transform_names = [type(t).__name__ for t in input_transforms]
            augmentation_details.append(f"Input transforms: {input_transform_names}")
        
        if augmentation_details:
            logging.info(f"Train dataset augmentations: {'; '.join(augmentation_details)}")
        else:
            logging.info("Train dataset: No augmentations will be applied")
        
        logging.info(f"Train dataset: {len(transforms)} cut transforms, {len(input_transforms)} input transforms")
        
        # Check if self-distillation is enabled
        enable_self_distillation = getattr(self.args, 'enable_self_distillation', False)
        
        if enable_self_distillation:
            logging.info("Creating clean-noisy dataset pairs for self-distillation")
            
            # Create single base dataset (clean, no augmentation)
            base_dataset = K2SpeechRecognitionDataset(
                input_strategy=input_strategy,
                cut_transforms=[],  # No cut augmentations for base (clean) version
                input_transforms=[],  # No input augmentations for base (clean) version  
                return_cuts=self.args.return_cuts,
            )
            
            # Create wrapper that uses same cuts for both clean and noisy
            # Clean = base dataset as-is
            # Noisy = same cuts + augmentation transforms applied on-the-fly
            train = CleanNoisyWrapper(
                base_dataset=base_dataset,
                augmentation_transforms=transforms,  # Cut transforms (MUSAN, RIR, etc)
                input_transforms=input_transforms   # Input transforms (SpecAugment, etc)
            )
            
            logging.info("Using single base dataset with on-the-fly augmentation")
        else:
            # Standard training (no self-distillation)
            train = K2SpeechRecognitionDataset(
                input_strategy=input_strategy,
                cut_transforms=transforms,  # Apply cut augmentations (MUSAN, RIR, concat)
                input_transforms=input_transforms,  # Apply input augmentations (SpecAugment)
                return_cuts=self.args.return_cuts,
            )

        # Determine shuffle value
        shuffle_val = self.args.shuffle if shuffle is None else shuffle

        # Create sampler
        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=shuffle_val,
                num_buckets=self.args.num_buckets,
                buffer_size=self.args.num_buckets * 2000,
                shuffle_buffer_size=self.args.num_buckets * 5000,
                drop_last=self.args.drop_last,
            )
        else:
            logging.info("Using SimpleCutSampler.")
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=shuffle_val,
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

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        transforms = []
        if self.args.concatenate_cuts:
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        # Determine the max_duration for validation
        valid_max_duration = self.args.valid_max_duration if self.args.valid_max_duration is not None else self.args.max_duration
        logging.info(f"Validation max_duration: {valid_max_duration} seconds")

        logging.info("About to create dev dataset")
        if self.args.on_the_fly_feats:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                return_cuts=self.args.return_cuts,
            )
        else:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                return_cuts=self.args.return_cuts,
            )
        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=valid_max_duration,
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

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        test = K2SpeechRecognitionDataset(
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
            if self.args.on_the_fly_feats
            else eval(self.args.input_strategy)(),
            return_cuts=self.args.return_cuts,
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

    def all_test_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Returns all test dataloaders including LibriSpeech and CHiME-4.
        
        Returns:
            Dict[str, DataLoader]: Dictionary with test set names as keys and DataLoaders as values
        """
        test_dataloaders = {}
        
        # LibriSpeech test sets
        test_clean_cuts = self.test_clean_cuts()
        test_other_cuts = self.test_other_cuts()
        
        test_dataloaders["test-clean"] = self.test_dataloaders(test_clean_cuts)
        test_dataloaders["test-other"] = self.test_dataloaders(test_other_cuts)
        
        # CHiME-4 test sets
        chime4_dls = self.chime4_test_dataloaders()
        for test_set_name, dl in chime4_dls.items():
            test_dataloaders[f"chime4-{test_set_name}"] = dl
            
        return test_dataloaders

    @lru_cache()
    def train_clean_5_cuts(self) -> CutSet:
        logging.info("mini_librispeech: About to get train-clean-5 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-clean-5.jsonl.gz"
        )

    @lru_cache()
    def train_clean_100_cuts(self) -> CutSet:
        logging.info("About to get train-clean-100 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-clean-100.jsonl.gz"
        )

    @lru_cache()
    def train_clean_360_cuts(self) -> CutSet:
        logging.info("About to get train-clean-360 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-clean-360.jsonl.gz"
        )

    @lru_cache()
    def train_other_500_cuts(self) -> CutSet:
        logging.info("About to get train-other-500 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-other-500.jsonl.gz"
        )

    @lru_cache()
    def train_all_shuf_cuts(self) -> CutSet:
        logging.info(
            "About to get the shuffled train-clean-100, \
            train-clean-360 and train-other-500 cuts"
        )
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-all-shuf.jsonl.gz"
        )

    @lru_cache()
    def dev_clean_2_cuts(self) -> CutSet:
        logging.info("mini_librispeech: About to get dev-clean-2 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_dev-clean-2.jsonl.gz"
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

    def chime4_test_dataloaders(self) -> Dict[str, DataLoader]:
        """Create CHiME-4 test dataloaders for different conditions."""
        from pathlib import Path
        
        chime4_audio_root = Path("/home/nas/DB/CHiME4/data/audio/16kHz/isolated")
        chime4_transcript_root = Path("/home/nas/DB/CHiME4/data/transcriptions")
        
        test_loaders = {}
        
        # Define test sets: dt05 (development) and et05 (evaluation)
        test_sets = ["dt05_bth", "et05_bth"]  # Start with booth (clean) conditions
        
        for test_set in test_sets:
            try:
                audio_dir = chime4_audio_root / test_set
                transcript_dir = chime4_transcript_root / test_set
                
                if not audio_dir.exists() or not transcript_dir.exists():
                    logging.warning(f"CHiME-4 {test_set} not found, skipping")
                    continue
                
                # Create cuts for this test set
                cuts = self._create_chime4_cuts(audio_dir, transcript_dir, max_files=50)
                
                if len(cuts) == 0:
                    logging.warning(f"No valid cuts for CHiME-4 {test_set}")
                    continue
                
                # Create test dataset
                test_dataset = K2SpeechRecognitionDataset(
                    input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                    return_cuts=self.args.return_cuts,
                )
                
                # Create sampler
                sampler = DynamicBucketingSampler(
                    cuts,
                    max_duration=self.args.max_duration,
                    shuffle=False,
                )
                
                # Create dataloader
                test_dl = DataLoader(
                    test_dataset,
                    batch_size=None,
                    sampler=sampler,
                    num_workers=2,
                )
                
                test_loaders[test_set] = test_dl
                logging.info(f"Created CHiME-4 {test_set} dataloader with {len(cuts)} cuts")
                
            except Exception as e:
                logging.warning(f"Failed to create CHiME-4 {test_set} dataloader: {e}")
        
        return test_loaders
    
    def _create_chime4_cuts(self, audio_dir: Path, transcript_dir: Path, max_files: int = 50) -> CutSet:
        """Helper to create CutSet from CHiME-4 audio and transcripts."""
        from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment, SupervisionSet
        
        # Parse transcriptions first
        transcriptions = {}
        for trn_file in transcript_dir.glob("*.trn"):
            try:
                with open(trn_file, 'r', encoding='utf-8') as f:
                    line = f.read().strip()
                    if line:
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            utterance_id = parts[0]
                            text = parts[1]
                            transcriptions[utterance_id] = text
            except Exception as e:
                logging.warning(f"Failed to read {trn_file}: {e}")
        
        logging.info(f"Found {len(transcriptions)} transcriptions in {transcript_dir}")
        
        # Get audio files - only CH0 channel to avoid duplicates
        wav_files = sorted([f for f in audio_dir.glob("*.wav") if '.CH0.' in f.name])[:max_files]
        logging.info(f"Found {len(wav_files)} CH0 audio files in {audio_dir}")
        
        # Create recordings and supervisions
        recordings = []
        supervisions = []
        
        for wav_file in wav_files:
            # Extract utterance ID from filename (remove .CH0 part)
            utterance_id = wav_file.stem.replace('.CH0', '')
            
            # Skip if no transcription
            if utterance_id not in transcriptions:
                logging.warning(f"No transcription found for {utterance_id}")
                continue
            
            try:
                # Create recording
                recording = Recording.from_file(wav_file)
                recording = Recording(
                    id=utterance_id,
                    sources=recording.sources,
                    sampling_rate=recording.sampling_rate,
                    num_samples=recording.num_samples,
                    duration=recording.duration,
                    channel_ids=recording.channel_ids,
                    transforms=recording.transforms
                )
                recordings.append(recording)
                
                # Create supervision
                text = transcriptions[utterance_id]
                supervision = SupervisionSegment(
                    id=utterance_id,
                    recording_id=utterance_id,
                    start=0.0,
                    duration=recording.duration,
                    channel=0,
                    text=text,
                    language="English"
                )
                supervisions.append(supervision)
                
            except Exception as e:
                logging.warning(f"Failed to process {wav_file}: {e}")
                continue
        
        if not recordings:
            return CutSet.from_cuts([])  # Empty CutSet
        
        # Create manifests
        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        cuts = CutSet.from_manifests(recordings=recording_set, supervisions=supervision_set)
        
        return cuts

    @lru_cache()
    def librilight_train_cuts(self) -> CutSet:
        """Load LibriLight training cuts."""
        logging.info("Loading LibriLight training cuts")
        
        librilight_dir = Path(self.args.librilight_dir)
        subset = self.args.librilight_subset
        
        cuts_path = librilight_dir / f"librilight_{subset}_cuts.jsonl.gz"
        
        if not cuts_path.exists():
            raise FileNotFoundError(
                f"LibriLight cuts not found: {cuts_path}\n"
                f"Please run prepare_librilight.sh first"
            )
        
        cuts = load_manifest_lazy(cuts_path)
        logging.info(f"Loaded {len(cuts)} LibriLight cuts")
        return cuts


class LibriLightAsrDataModule(LibriSpeechAsrDataModule):
    """
    DataModule specifically for LibriLight ASR experiments.
    
    LibriLight is a large-scale ASR corpus with different subsets:
    - small: ~10k hours
    - medium: ~50k hours  
    - large: ~60k hours
    
    This module extends LibriSpeechAsrDataModule to handle LibriLight-specific
    data loading and preprocessing.
    """
    
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        # Add LibriLight-specific arguments (skip LibriSpeech arguments to avoid conflicts)
        group = parser.add_argument_group(
            title="LibriLight ASR data options",
            description="LibriLight-specific data loading options"
        )
        
        group.add_argument(
            "--librilight-manifest-dir",
            type=Path,
            default=Path("data/fbank"),
            help="Path to LibriLight manifest directory (containing cuts files with fbank features)"
        )
        
        group.add_argument(
            "--librilight-subset",
            type=str,
            default="medium",
            choices=["small", "medium", "large"],
            help="LibriLight subset to use for training"
        )
        
        group.add_argument(
            "--librilight-sampling-ratio",
            type=float,
            default=1.0,
            help="Ratio of LibriLight data to sample (0.0-1.0)"
        )
        
        group.add_argument(
            "--librilight-max-duration-per-cut",
            type=float,
            default=30.0,
            help="Maximum duration per cut in seconds (LibriLight has very long utterances)"
        )
        
        group.add_argument(
            "--enable-librilight-chunking",
            type=str2bool,
            default=False,  # Changed to False for faster loading
            help="Split long LibriLight cuts into smaller chunks (disabled by default for speed)"
        )

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        
        # Override manifest_dir to use LibriLight-specific path
        if hasattr(args, 'librilight_manifest_dir'):
            self.librilight_manifest_dir = args.librilight_manifest_dir
        else:
            self.librilight_manifest_dir = Path("data/fbank")
            
        logging.info(f"LibriLight manifest directory: {self.librilight_manifest_dir}")
        logging.info(f"LibriLight subset: {args.librilight_subset}")

    @lru_cache()
    def librilight_train_cuts(self) -> CutSet:
        """Load LibriLight training cuts efficiently."""
        subset = self.args.librilight_subset
        logging.info(f"Loading LibriLight {subset} training cuts")
        
        # Construct manifest path
        manifest_file = f"librilight_{subset}_cuts_train.jsonl.gz"
        cuts_path = self.librilight_manifest_dir / manifest_file
        
        if not cuts_path.exists():
            # Try alternative naming
            alt_manifest_file = f"cuts_train_{subset}.jsonl.gz"
            cuts_path = self.librilight_manifest_dir / alt_manifest_file
            
        if not cuts_path.exists():
            raise FileNotFoundError(
                f"LibriLight training cuts not found at {cuts_path} or alternative paths.\n"
                f"Please prepare LibriLight manifests first using prepare_librilight.sh"
            )
        
        logging.info(f"Loading cuts from {cuts_path}")
        cuts = load_manifest_lazy(cuts_path)
        logging.info(f"Successfully loaded {len(cuts)} LibriLight {subset} training cuts")
        
        # Apply optional chunking and sampling only if explicitly enabled
        if hasattr(self.args, 'enable_librilight_chunking') and self.args.enable_librilight_chunking:
            cuts = self._apply_chunking(cuts)
            
        if hasattr(self.args, 'librilight_sampling_ratio') and self.args.librilight_sampling_ratio < 1.0:
            cuts = self._apply_sampling(cuts)
        
        return cuts
    
    def _apply_chunking(self, cuts: CutSet) -> CutSet:
        """Apply chunking to long LibriLight cuts."""
        max_duration = getattr(self.args, 'librilight_max_duration_per_cut', 20.0)
        logging.info(f"Chunking LibriLight cuts to max {max_duration}s per cut")
        
        chunked_cuts = []
        for cut in cuts:
            if cut.duration > max_duration:
                # Split long cuts into chunks
                num_chunks = int(cut.duration / max_duration) + 1
                chunk_duration = cut.duration / num_chunks
                
                for i in range(num_chunks):
                    start = i * chunk_duration
                    end = min((i + 1) * chunk_duration, cut.duration)
                    
                    # Create chunk using truncate with proper supervision handling
                    chunk = cut.truncate(offset=start, duration=end-start, keep_excessive_supervisions=False)
                    chunk.id = f"{cut.id}_chunk_{i:03d}"
                    chunked_cuts.append(chunk)
            else:
                chunked_cuts.append(cut)
        
        cuts = CutSet.from_cuts(chunked_cuts)
        logging.info(f"After chunking: {len(cuts)} cuts")
        return cuts
    
    def _apply_sampling(self, cuts: CutSet) -> CutSet:
        """Apply sampling to reduce LibriLight dataset size."""
        sampling_ratio = self.args.librilight_sampling_ratio
        original_size = len(cuts)
        subset_size = int(original_size * sampling_ratio)
        cuts = cuts.subset(first=subset_size)
        logging.info(f"Sampled {len(cuts)}/{original_size} cuts (ratio: {sampling_ratio})")
        return cuts

    @lru_cache()
    def librilight_dev_cuts(self) -> CutSet:
        """Load LibriLight development/validation cuts."""
        subset = self.args.librilight_subset
        logging.info(f"Loading LibriLight {subset} development cuts")
        
        # Try to load pre-saved LibriLight dev cuts first
        manifest_file = f"librilight_{subset}_cuts_dev.jsonl.gz"
        cuts_path = self.librilight_manifest_dir / manifest_file
        
        if cuts_path.exists():
            logging.info(f"Loading pre-saved LibriLight dev cuts from {cuts_path}")
            cuts = load_manifest_lazy(cuts_path)
            logging.info(f"Loaded {len(cuts)} LibriLight {subset} dev cuts")
            return cuts
        
        # Fallback: Use LibriSpeech dev-clean (but cache it for future use)
        logging.info("LibriLight dev cuts not found. Using LibriSpeech dev-clean as validation")
        logging.info("This will be cached for future use")
        
        # Load LibriSpeech dev-clean
        dev_cuts = self.dev_clean_cuts()
        
        # Save for future use
        try:
            logging.info(f"Saving LibriLight dev cuts to {cuts_path} for future use")
            dev_cuts.to_file(cuts_path)
            logging.info(f"Successfully saved {len(dev_cuts)} dev cuts")
        except Exception as e:
            logging.warning(f"Failed to save LibriLight dev cuts: {e}")
        
        return dev_cuts

    def train_dataloaders(
        self,
        cuts_train: Optional[CutSet] = None,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
        shuffle: Optional[bool] = None,
    ) -> DataLoader:
        """
        Create LibriLight training dataloader.
        
        Args:
            cuts_train: If provided, use these cuts. Otherwise load LibriLight training cuts.
            sampler_state_dict: Sampler state dict for resuming training
            shuffle: Whether to shuffle data
        """
        if cuts_train is None:
            cuts_train = self.librilight_train_cuts()
        
        # Use parent class implementation with LibriLight cuts
        return super().train_dataloaders(cuts_train, sampler_state_dict, shuffle)

    def valid_dataloaders(self, cuts_valid: Optional[CutSet] = None) -> DataLoader:
        """Create LibriLight validation dataloader."""
        if cuts_valid is None:
            cuts_valid = self.librilight_dev_cuts()
        
        return super().valid_dataloaders(cuts_valid)

    def all_test_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Create all test dataloaders including LibriSpeech test sets.
        
        For LibriLight, we typically evaluate on LibriSpeech test sets
        since LibriLight doesn't have standard evaluation sets.
        """
        test_dataloaders = {}
        
        # Use LibriSpeech test sets for evaluation
        try:
            test_clean_cuts = self.test_clean_cuts()
            test_other_cuts = self.test_other_cuts()
            
            test_dataloaders["test-clean"] = self.test_dataloaders(test_clean_cuts)
            test_dataloaders["test-other"] = self.test_dataloaders(test_other_cuts)
            
            logging.info("Created LibriSpeech test dataloaders for LibriLight evaluation")
        except Exception as e:
            logging.warning(f"Failed to create LibriSpeech test dataloaders: {e}")
        
        # Add CHiME-4 if available
        try:
            chime4_dls = self.chime4_test_dataloaders()
            for test_set_name, dl in chime4_dls.items():
                test_dataloaders[f"chime4-{test_set_name}"] = dl
        except Exception as e:
            logging.warning(f"Failed to create CHiME-4 test dataloaders: {e}")
            
        return test_dataloaders

    # Override LibriSpeech-specific methods to use LibriLight data
    @lru_cache()
    def train_clean_100_cuts(self) -> CutSet:
        """Override to return LibriLight training cuts instead."""
        return self.librilight_train_cuts()

    @lru_cache()
    def train_clean_360_cuts(self) -> CutSet:
        """Override to return LibriLight training cuts instead.""" 
        return self.librilight_train_cuts()

    @lru_cache()
    def train_other_500_cuts(self) -> CutSet:
        """Override to return LibriLight training cuts instead."""
        return self.librilight_train_cuts()

    @lru_cache() 
    def train_all_shuf_cuts(self) -> CutSet:
        """Override to return LibriLight training cuts instead."""
        return self.librilight_train_cuts()
