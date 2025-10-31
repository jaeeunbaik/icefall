#!/usr/bin/env python3
# Copyright 2023 Xiaomi Corp. (authors: Jenny Baik)
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
This script computes fbank features for LibriLight dataset.

Usage:
    python3 local/compute_fbank_librilight.py \
        --manifest-dir data/librilight_medium \
        --output-dir data/fbank \
        --subset medium \
        --num-jobs 16
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter, load_manifest

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser(
        description="Compute fbank features for LibriLight"
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default="data/librilight_medium",
        help="Directory containing LibriLight manifests",
    )
    parser.add_argument(
        "--output-dir", 
        type=Path,
        default="data/fbank",
        help="Directory to save fbank features",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="LibriLight subset",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=16,
        help="Number of parallel jobs",
    )
    parser.add_argument(
        "--batch-duration",
        type=float,
        default=600.0,
        help="Batch duration for feature extraction (seconds)",
    )
    parser.add_argument(
        "--num-mel-bins",
        type=int,
        default=80,
        help="Number of mel filter banks",
    )
    return parser.parse_args()


def main():
    args = get_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load cuts manifest with better error handling
    cuts_path = args.manifest_dir / f"librilight_{args.subset}_cuts.jsonl.gz"
    if not cuts_path.exists():
        raise FileNotFoundError(f"Cuts manifest not found: {cuts_path}")
    
    logging.info(f"Loading cuts from {cuts_path}")
    
    # Try different loading methods to debug the issue
    try:
        # Method 1: Try loading as CutSet directly
        from lhotse import CutSet
        cuts = CutSet.from_file(cuts_path)
        logging.info(f"Successfully loaded {len(cuts)} cuts using CutSet.from_file")
    except Exception as e1:
        logging.warning(f"CutSet.from_file failed: {e1}")
        try:
            # Method 2: Try the generic load_manifest
            cuts = load_manifest(cuts_path)
            logging.info(f"Successfully loaded {len(cuts)} cuts using load_manifest")
        except Exception as e2:
            logging.error(f"load_manifest also failed: {e2}")
            try:
                # Method 3: Check if it's a recording/supervision file instead
                logging.info("Trying to load as recordings + supervisions...")
                rec_path = args.manifest_dir / f"librilight_{args.subset}_recordings.jsonl.gz"
                sup_path = args.manifest_dir / f"librilight_{args.subset}_supervisions.jsonl.gz"
                
                if rec_path.exists() and sup_path.exists():
                    from lhotse import RecordingSet, SupervisionSet
                    recordings = RecordingSet.from_file(rec_path)
                    supervisions = SupervisionSet.from_file(sup_path)
                    cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
                    
                    # Save the cuts for future use
                    cuts.to_file(cuts_path)
                    logging.info(f"Created {len(cuts)} cuts from recordings and supervisions")
                else:
                    raise FileNotFoundError("No valid manifest files found")
            except Exception as e3:
                logging.error(f"All loading methods failed. Last error: {e3}")
                raise e3
    
    # Configure fbank extractor
    fbank_config = FbankConfig(
        num_mel_bins=args.num_mel_bins,
        sampling_rate=16000,
        frame_length=25.0,  # 25ms
        frame_shift=10.0,   # 10ms
        energy_floor=1e-10,
        dither=1.0,
        remove_dc_offset=True,
        round_to_power_of_two=True,
    )
    
    extractor = Fbank(fbank_config)
    
    # Compute features in batches
    logging.info("Computing fbank features...")
    cuts = cuts.compute_and_store_features(
        extractor=extractor,
        storage_path=f"{args.output_dir}/librilight_feats_{args.subset}",
        batch_duration=args.batch_duration,
        num_jobs=args.num_jobs,
        overwrite=False,
        storage_type=LilcomChunkyWriter,
    )
    
    # Save cuts with features to multiple locations for compatibility
    # 1. Standard location
    output_cuts_path = args.output_dir / f"librilight_cuts_{args.subset}.jsonl.gz"
    cuts.to_file(output_cuts_path)
    logging.info(f"Saved cuts with features to {output_cuts_path}")
    
    # 2. Training cuts file with expected naming for asr_datamodule.py
    train_cuts_path = args.output_dir / f"librilight_{args.subset}_cuts_train.jsonl.gz" 
    cuts.to_file(train_cuts_path)
    logging.info(f"Saved training cuts to {train_cuts_path}")
    
    # 3. Alternative naming convention
    alt_cuts_path = args.output_dir / f"cuts_train_{args.subset}.jsonl.gz"
    cuts.to_file(alt_cuts_path)
    logging.info(f"Saved alternative cuts to {alt_cuts_path}")
    
    # Print statistics
    total_duration = sum(c.duration for c in cuts)
    speakers = set()
    for cut in cuts:
        if cut.supervisions:
            speakers.add(cut.supervisions[0].speaker)
    
    logging.info(f"Feature extraction completed:")
    logging.info(f"  Total cuts: {len(cuts)}")
    logging.info(f"  Total duration: {total_duration/3600:.2f} hours")
    logging.info(f"  Number of speakers: {len(speakers)}")
    logging.info(f"  Features saved to: {args.output_dir}")
    
    # Log created files
    logging.info(f"Created files:")
    logging.info(f"  - {output_cuts_path}")
    logging.info(f"  - {train_cuts_path}")
    logging.info(f"  - {alt_cuts_path}")


if __name__ == "__main__":
    main()