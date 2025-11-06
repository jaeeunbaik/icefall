#!/usr/bin/env python3

"""
Quick LibriLight sampling for fast testing
"""

import gzip
import json
import logging
from pathlib import Path

def sample_librilight_cuts(input_path, output_path, sample_size=1000):
    """Sample a subset of LibriLight cuts for faster testing"""
    
    logging.info(f"Sampling {sample_size} cuts from {input_path}")
    
    sampled_cuts = []
    with gzip.open(input_path, 'rt') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            sampled_cuts.append(line.strip())
    
    # Write sampled cuts
    with gzip.open(output_path, 'wt') as f:
        for cut_line in sampled_cuts:
            f.write(cut_line + '\n')
    
    logging.info(f"Saved {len(sampled_cuts)} sampled cuts to {output_path}")
    return len(sampled_cuts)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    input_path = Path("data/librilight_medium/librilight_medium_cuts.jsonl.gz")
    output_path = Path("data/librilight_medium/librilight_medium_cuts_sample1k.jsonl.gz")
    
    sample_librilight_cuts(input_path, output_path, sample_size=1000)
    print(f"✓ Created sample: {output_path}")
    print(f"Now run: python3 local/compute_fbank_librilight.py --manifest-dir data/librilight_medium --output-dir data/fbank --subset medium_sample1k --num-jobs 8")