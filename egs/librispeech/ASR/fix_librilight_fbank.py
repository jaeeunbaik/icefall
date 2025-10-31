#!/usr/bin/env python3

"""
Alternative script to create cuts from recordings and supervisions
and compute fbank features for LibriLight
"""

import argparse
import logging
from pathlib import Path

import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse import RecordingSet, SupervisionSet

# Torch's multithreaded behavior needs to be disabled
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser(description="Alternative LibriLight fbank computation")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--num-jobs", type=int, default=16)
    parser.add_argument("--batch-duration", type=float, default=600.0)
    parser.add_argument("--num-mel-bins", type=int, default=80)
    return parser.parse_args()


def main():
    args = get_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load recordings and supervisions separately
    rec_path = args.manifest_dir / f"librilight_{args.subset}_recordings.jsonl.gz"
    sup_path = args.manifest_dir / f"librilight_{args.subset}_supervisions.jsonl.gz"
    
    logging.info(f"Loading recordings from {rec_path}")
    recordings = RecordingSet.from_file(rec_path)
    logging.info(f"Loaded {len(recordings)} recordings")
    
    logging.info(f"Loading supervisions from {sup_path}")
    supervisions = SupervisionSet.from_file(sup_path)
    logging.info(f"Loaded {len(supervisions)} supervisions")
    
    # Create cuts from recordings and supervisions
    logging.info("Creating cuts from recordings and supervisions...")
    cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
    logging.info(f"Created {len(cuts)} cuts")
    
    # Configure fbank extractor
    fbank_config = FbankConfig(
        num_mel_bins=args.num_mel_bins,
        sampling_rate=16000,
        frame_length=25.0,
        frame_shift=10.0,
        energy_floor=1e-10,
        dither=1.0,
        remove_dc_offset=True,
        round_to_power_of_two=True,
    )
    
    extractor = Fbank(fbank_config)
    
    # Compute features
    logging.info("Computing fbank features...")
    cuts = cuts.compute_and_store_features(
        extractor=extractor,
        storage_path=f"{args.output_dir}/librilight_feats_{args.subset}",
        batch_duration=args.batch_duration,
        num_jobs=args.num_jobs,
        overwrite=False,
        storage_type=LilcomChunkyWriter,
    )
    
    # Save cuts with features to multiple locations
    output_files = [
        args.output_dir / f"librilight_cuts_{args.subset}.jsonl.gz",
        args.output_dir / f"librilight_{args.subset}_cuts_train.jsonl.gz",
        args.output_dir / f"cuts_train_{args.subset}.jsonl.gz"
    ]
    
    for output_path in output_files:
        try:
            cuts.to_file(output_path)
            logging.info(f"Saved cuts with features to {output_path}")
            
            # Verify by loading back
            test_cuts = CutSet.from_file(output_path)
            logging.info(f"Verification: {len(test_cuts)} cuts loaded from {output_path}")
        except Exception as e:
            logging.error(f"Failed to save to {output_path}: {e}")
    
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


if __name__ == "__main__":
    main()