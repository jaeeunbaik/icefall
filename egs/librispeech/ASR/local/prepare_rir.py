#!/usr/bin/env python3
"""
Prepare RIR (Room Impulse Response) data for lhotse.
This script converts rir.scp file to lhotse manifest format.
"""

import argparse
import logging
from pathlib import Path
from typing import List

from lhotse import CutSet, Recording, SupervisionSegment, MonoCut
from lhotse.audio import AudioSource
from lhotse.utils import Pathlike

def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare RIR data for lhotse",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--rir-scp",
        type=Path,
        required=True,
        help="Path to rir.scp file containing RIR file paths",
    )
    
    parser.add_argument(
        "--output-dir", 
        type=Path,
        required=True,
        help="Output directory for RIR manifests",
    )
    
    return parser.parse_args()


def prepare_rir_manifest(
    rir_scp: Pathlike,
    output_dir: Pathlike,
) -> None:
    """
    Prepare RIR manifest from rir.scp file.
    
    Args:
        rir_scp: Path to rir.scp file
        output_dir: Output directory for manifests
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    recordings = []
    
    # Read rir.scp file
    with open(rir_scp, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Parse line: either "path" or "id path"
            parts = line.split()
            if len(parts) == 1:
                rir_path = parts[0]
                rir_id = f"rir_{line_idx:06d}"
            elif len(parts) == 2:
                rir_id, rir_path = parts
            else:
                logging.warning(f"Invalid line in rir.scp: {line}")
                continue
            
            # Check if file exists
            rir_path = Path(rir_path)
            if not rir_path.exists():
                logging.warning(f"RIR file not found: {rir_path}")
                continue
            
            try:
                # Create recording from file (auto-detects duration, sampling_rate, etc.)
                recording = Recording.from_file(
                    path=rir_path,
                    recording_id=rir_id,
                )
                recordings.append(recording)
            except Exception as e:
                logging.warning(f"Failed to create recording from {rir_path}: {e}")
                continue
    
    logging.info(f"Found {len(recordings)} RIR files")
    
    # Create recording set and save
    from lhotse import RecordingSet
    recording_set = RecordingSet.from_recordings(recordings)
    
    # Validate recordings
    logging.info("Validating RIR recordings...")
    
    # Save recording manifest
    output_path = output_dir / "rir_recordings.jsonl.gz"
    recording_set.to_file(output_path)
    logging.info(f"Saved RIR recording manifest to {output_path}")
    
    # Create a simple cuts manifest for RIR (whole files)
    logging.info("Creating RIR cuts manifest...")
    # Use CutSet.from_manifests to properly link recordings and cuts
    rir_cuts = CutSet.from_manifests(
        recordings=recording_set,
        supervisions=None,  # No supervisions for RIR files
    )
    cuts_output_path = output_dir / "rir_cuts.jsonl.gz"
    rir_cuts.to_file(cuts_output_path)
    logging.info(f"Saved RIR cuts manifest to {cuts_output_path}")
    
    return recording_set, rir_cuts


def main():
    args = get_args()
    
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    
    logging.info("Preparing RIR data...")
    prepare_rir_manifest(
        rir_scp=args.rir_scp,
        output_dir=args.output_dir,
    )
    logging.info("Done!")


if __name__ == "__main__":
    main()
