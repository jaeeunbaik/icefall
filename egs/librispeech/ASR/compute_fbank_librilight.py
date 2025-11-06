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
This script computes fbank features for LibriLight cuts.

Usage:
    python3 compute_fbank_librilight.py \
        --manifest-dir data/librilight_medium \
        --output-dir data/fbank \
        --subset medium \
        --num-jobs 4
"""

import argparse
import logging
from pathlib import Path

from lhotse import CutSet, Fbank, LilcomChunkyWriter
from lhotse.utils import Pathlike


def get_args():
    parser = argparse.ArgumentParser(
        description="Compute fbank features for LibriLight cuts"
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Directory containing LibriLight cuts manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for fbank features",
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
        default=1,
        help="Number of parallel jobs",
    )
    return parser.parse_args()


def main():
    args = get_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )
    
    # Load cuts
    cuts_path = args.manifest_dir / f"librilight_{args.subset}_cuts.jsonl.gz"
    if not cuts_path.exists():
        logging.error(f"Cuts file not found: {cuts_path}")
        return
    
    logging.info(f"Loading cuts from {cuts_path}")
    cuts = CutSet.from_file(cuts_path)
    logging.info(f"Loaded {len(cuts)} cuts")
    
    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup fbank extractor
    extractor = Fbank()
    
    # Setup chunky writer for efficient storage
    storage_path = output_dir / f"librilight_{args.subset}_feats"
    
    logging.info(f"Computing fbank features with {args.num_jobs} jobs...")
    
    # Compute features and save to storage
    cuts_with_feats = cuts.compute_and_store_features(
        extractor=extractor,
        storage_path=storage_path,
        num_jobs=args.num_jobs,
    )
    
    # Save the cuts with features
    output_cuts_path = output_dir / f"librilight_{args.subset}_cuts_with_feats.jsonl.gz"
    cuts_with_feats.to_file(output_cuts_path)
    
    logging.info(f"Saved cuts with features to {output_cuts_path}")
    logging.info(f"Feature storage saved to {storage_path}")


if __name__ == "__main__":
    main()