#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
# Copyright    2024  Author               (authors: Modified for LibriLight)
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
This script computes fbank features of the LibriLight dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import logging
import os
from pathlib import Path

from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.recipes.utils import read_manifests_if_cached

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() because it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def compute_fbank_librilight():
    src_dir = Path("data/manifests")
    kmeans_dir = Path("data/kmeans")
    output_dir = Path("data/fbank")
    # Use only 10 parallel jobs to avoid overloading the system
    num_jobs = 20
    num_mel_bins = 80
    
    print(f"Using {num_jobs} parallel jobs for feature extraction")
    print(f"Available CPU cores: {os.cpu_count()}")
    print(f"Output directory: {output_dir.absolute()}")
    
    

    dataset_parts = (
        "medium",
        # Add other parts if needed
        # "small",
        # "large",
    )
    
    print("LibriLight manifest parts:")
    print(dataset_parts)

    output_dir.mkdir(exist_ok=True)

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    for partition in dataset_parts:
        print(f"Processing {partition}")
        
        # Check for split files first (e.g., medium_split/)
        split_dir = kmeans_dir / f"{partition}_split"
        if split_dir.exists():
            print(f"Found split directory: {split_dir}")
            import glob
            split_files = sorted(glob.glob(str(split_dir / f"librilight_cuts_{partition}_raw.*.jsonl.gz")))
            
            if split_files:
                print(f"Found {len(split_files)} split files")
                
                # Process each split file
                for idx, split_file in enumerate(split_files, 1):
                    split_path = Path(split_file)
                    # Extract split number from filename (e.g., librilight_cuts_medium_raw.00000000.jsonl.gz)
                    split_name = split_path.stem.replace('.jsonl', '')  # Remove .jsonl from stem
                    if split_name.endswith('.gz'):
                        split_name = split_name[:-3]  # Remove .gz
                    
                    # Output filename with fbank instead of raw
                    output_filename = split_name.replace('_raw', '') + '.jsonl.gz'
                    output_path = output_dir / output_filename
                    
                    print(f"\n{'='*80}")
                    print(f"Processing split {idx}/{len(split_files)}")
                    print(f"Input:  {split_file}")
                    print(f"Output: {output_path}")
                    print(f"{'='*80}")
                    
                    if output_path.is_file():
                        print(f"✓ {output_filename} already exists - skipping.")
                        continue
                    
                    print(f"[1/4] Loading cuts from {split_file}")
                    import time
                    start_time = time.time()
                    raw_cuts = CutSet.from_file(split_file)
                    load_time = time.time() - start_time
                    print(f"      Loaded {len(raw_cuts)} cuts in {load_time:.1f}s")
                    
                    # Extract split identifier for storage path
                    import re
                    match = re.search(r'\.(\d+)\.jsonl', str(split_file))
                    split_id = match.group(1) if match else "00"
                    
                    storage_path = f"{output_dir}/librilight_feats_{partition}_{split_id}"
                    print(f"[2/4] Computing fbank features for {partition} split {split_id}")
                    print(f"      Num jobs: {num_jobs}")
                    print(f"      Storage path: {storage_path}")
                    print(f"      This may take several minutes...")
                    print(f"      NOTE: Progress bar may not update smoothly, but processing is happening!")
                    print(f"      You can monitor: watch -n 5 'ls -lh {storage_path}/*.lca 2>/dev/null | wc -l'")
                    
                    extract_start = time.time()
                    
                    # Add callback for better progress tracking
                    from tqdm import tqdm
                    pbar = tqdm(total=len(raw_cuts), desc=f"      Processing {len(raw_cuts)} cuts", unit="cut")
                    
                    def progress_callback(cut):
                        pbar.update(1)
                        return cut
                    
                    # Note: lhotse may not support callbacks directly, so we'll rely on chunk progress
                    cut_set = raw_cuts.compute_and_store_features(
                        extractor=extractor,
                        storage_path=storage_path,
                        num_jobs=num_jobs,
                        storage_type=LilcomChunkyWriter,
                    )
                    
                    pbar.close()
                    extract_time = time.time() - extract_start
                    print(f"      Feature extraction completed in {extract_time:.1f}s ({extract_time/60:.1f} minutes)")
                    
                    print(f"[3/4] Saving manifest to {output_path}")
                    save_start = time.time()
                    cut_set.to_file(output_path)
                    save_time = time.time() - save_start
                    print(f"      Saved in {save_time:.1f}s")
                    
                    total_time = time.time() - start_time
                    print(f"[4/4] ✓ Split {idx}/{len(split_files)} completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
                    print(f"      Average: {total_time/len(raw_cuts):.3f}s per cut")
                
                print(f"\n{'='*80}")
                print(f"All {len(split_files)} split files processed successfully!")
                print(f"{'='*80}\n")
                continue  # Skip the single-file processing below
        
        # Fallback: single file processing
        cuts_filename = f"librilight_cuts_{partition}.jsonl.gz"
        
        if (output_dir / cuts_filename).is_file():
            print(f"{partition} already exists - skipping.")
            continue

        # Check if raw cuts exist (from preprocess_librilight.py)
        raw_cuts_path = kmeans_dir / f"librilight_cuts_{partition}_raw.jsonl.gz"
        
        if raw_cuts_path.is_file():
            print(f"Loading raw cuts from {raw_cuts_path}")
            raw_cuts = CutSet.from_file(raw_cuts_path)
        else:
            # Fallback: create cuts from recordings and supervisions
            print(f"Creating cuts from recordings and supervisions")
            recording_set, supervision_set = read_manifests_if_cached(
                dataset_parts=[partition],
                output_dir=src_dir,
                prefix="librilight",
                suffix="jsonl.gz",
            )
            
            # Create a CutSet
            raw_cuts = CutSet.from_manifests(
                recordings=recording_set,
                supervisions=supervision_set,
            )

        print(f"Computing fbank features for {partition}")
        cut_set = raw_cuts.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{output_dir}/librilight_feats_{partition}",
            num_jobs=num_jobs,
            storage_type=LilcomChunkyWriter,
        )

        cut_set.to_file(output_dir / cuts_filename)
        print(f"Saved to {output_dir / cuts_filename}")


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    compute_fbank_librilight()


if __name__ == "__main__":
    main()
