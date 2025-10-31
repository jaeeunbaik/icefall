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
This script prepares LibriLight medium subset for icefall.
LibriLight is an unsupervised speech dataset without transcripts.

Usage:
    python3 local/prepare_librilight.py \
        --librilight-dir /home/hdd1/jenny/LibriLight-m \
        --output-dir data/librilight_medium \
        --min-duration 5.0 \
        --max-duration 30.0 \
        --min-snr 10.0
"""

import argparse
import json
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf
from lhotse import CutSet, Recording, RecordingSet, SupervisionSet
from lhotse.audio import AudioSource
from lhotse.utils import Pathlike


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare LibriLight medium subset for icefall"
    )
    parser.add_argument(
        "--librilight-dir",
        type=Path,
        default="/home/hdd1/jenny/LibriLight-m",
        help="Path to LibriLight dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data/librilight_medium",
        help="Output directory for processed manifests",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=5.0,
        help="Minimum audio duration in seconds",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds",
    )
    parser.add_argument(
        "--min-snr",
        type=float,
        default=10.0,
        help="Minimum SNR threshold",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="LibriLight subset to process",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=16,
        help="Number of parallel jobs for processing",
    )
    return parser.parse_args()


def parse_voice_activity(voice_activity: List[List[float]]) -> List[Tuple[float, float]]:
    """Parse voice activity regions from JSON metadata."""
    return [(float(start), float(end)) for start, end in voice_activity]


def filter_audio_by_voice_activity(
    audio_path: Pathlike,
    voice_activity: List[Tuple[float, float]],
    min_duration: float = 5.0,
    max_duration: float = 30.0,
) -> List[Tuple[float, float]]:
    """
    Filter voice activity segments by duration criteria.
    
    Returns:
        List of (start, end) tuples for valid segments
    """
    valid_segments = []
    
    for start, end in voice_activity:
        duration = end - start
        if min_duration <= duration <= max_duration:
            valid_segments.append((start, end))
    
    return valid_segments


def process_librilight_file(
    audio_path: Path,
    json_path: Path,
    min_duration: float,
    max_duration: float,
    min_snr: float,
) -> Tuple[Optional[Recording], List[Dict]]:
    """
    Process a single LibriLight audio-json pair.

    Returns:
        A tuple of (Recording for the whole file, list of supervision dicts for valid segments).
    """
    try:
        # Load metadata
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Check SNR threshold
        snr = metadata.get('snr', 0.0)
        if snr < min_snr:
            logging.debug(f"Skipping {audio_path} due to low SNR: {snr}")
            return None, []

        # Parse voice activity
        voice_activity_raw = metadata.get('voice_activity', [])
        voice_activity = parse_voice_activity(voice_activity_raw)

        # Filter segments by duration
        valid_segments = filter_audio_by_voice_activity(
            audio_path, voice_activity, min_duration, max_duration
        )

        if not valid_segments:
            logging.debug(f"No valid segments in {audio_path}")
            return None, []

        # Get audio info
        try:
            info = sf.info(audio_path)
            total_duration = info.duration
            sample_rate = info.samplerate
        except Exception as e:
            logging.warning(f"Failed to read audio info for {audio_path}: {e}")
            return None, []

        # Create one Recording per audio file (no offset/duration in AudioSource)
        audio_stem = audio_path.stem
        recording_id = f"{audio_stem}"

        transforms = None
        if sample_rate != 16000:
            transforms = [
                {
                    "name": "Resample",
                    "kwargs": {"source_sampling_rate": sample_rate, "target_sampling_rate": 16000},
                }
            ]

        recording = Recording(
            id=recording_id,
            sources=[
                AudioSource(type="file", channels=[0], source=str(audio_path))
            ],
            sampling_rate=sample_rate,
            num_samples=int(total_duration * sample_rate),
            duration=total_duration,
            channel_ids=[0],
            transforms=transforms,
        )

        # Build supervision dicts referencing the recording id
        speaker_id = metadata.get('speaker', 'unknown')
        book_meta = metadata.get('book_meta', {})
        book_title = book_meta.get('title', 'unknown')

        supervisions = []
        for i, (start, end) in enumerate(valid_segments):
            # Ensure segment is within audio bounds
            start = max(0.0, start)
            end = min(total_duration, end)
            if end <= start:
                continue

            sup_id = f"{recording_id}-{i:03d}"
            supervision = {
                "id": sup_id,
                "recording_id": recording_id,
                "start": float(start),
                "duration": float(end - start),
                "channel": 0,
                "text": "",  # Empty text for unsupervised data
                "language": "English",
                "speaker": speaker_id,
            }
            supervisions.append(supervision)

        return recording, supervisions

    except Exception as e:
        logging.error(f"Error processing {audio_path}: {e}")
        return None, []


def collect_librilight_files(librilight_dir: Path, subset: str = "medium") -> List[Tuple[Path, Path]]:
    """
    Collect all audio-json file pairs from LibriLight dataset.
    
    Returns:
        List of (audio_path, json_path) tuples
    """
    subset_dir = librilight_dir / subset
    if not subset_dir.exists():
        raise ValueError(f"LibriLight subset directory not found: {subset_dir}")
    
    file_pairs = []
    
    # Walk through speaker directories
    for speaker_dir in subset_dir.iterdir():
        if not speaker_dir.is_dir():
            continue
        
        # Walk through book/project directories
        for book_dir in speaker_dir.iterdir():
            if not book_dir.is_dir():
                continue
            
            # Find audio-json pairs
            flac_files = list(book_dir.glob("*.flac"))
            
            for flac_file in flac_files:
                json_file = flac_file.with_suffix(".json")
                if json_file.exists():
                    file_pairs.append((flac_file, json_file))
                else:
                    logging.warning(f"Missing JSON metadata for {flac_file}")
    
    logging.info(f"Found {len(file_pairs)} audio-json pairs in {subset_dir}")
    return file_pairs


def create_librilight_manifests(
    librilight_dir: Path,
    output_dir: Path,
    subset: str = "medium",
    min_duration: float = 5.0,
    max_duration: float = 30.0,
    min_snr: float = 10.0,
    num_jobs: int = 16,
):
    """Create LibriLight manifests for icefall."""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create manifests directory following LibriSpeech convention
    manifests_dir = Path("data/manifests")
    manifests_dir.mkdir(parents=True, exist_ok=True)
    
    # Create fbank directory structure
    fbank_dir = Path("data/fbank")
    fbank_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all file pairs
    file_pairs = collect_librilight_files(librilight_dir, subset)
    
    if not file_pairs:
        raise ValueError(f"No valid file pairs found in {librilight_dir}/{subset}")
    
    # Process files and create one Recording per audio file + supervision entries
    recordings_map = {}
    all_supervisions = []

    logging.info(f"Processing {len(file_pairs)} file pairs...")

    for i, (audio_path, json_path) in enumerate(file_pairs):
        if i % 1000 == 0:
            logging.info(f"Processed {i}/{len(file_pairs)} files")

        recording, supervisions = process_librilight_file(
            audio_path, json_path, min_duration, max_duration, min_snr
        )

        if recording is None or not supervisions:
            continue

        # Avoid duplicate Recording objects for the same file
        recordings_map[recording.id] = recording
        all_supervisions.extend(supervisions)

    if not recordings_map or not all_supervisions:
        raise ValueError("No valid recordings/supervisions created. Check your filtering criteria.")

    logging.info(f"Created {len(recordings_map)} recordings and {len(all_supervisions)} supervisions")

    # Create RecordingSet from unique recordings
    recording_set = RecordingSet.from_recordings(list(recordings_map.values()))
    
    # Save manifests using LibriSpeech naming convention for compatibility
    part_name = f"librilight-{subset}"
    
    # Save to manifests directory (for consistency with LibriSpeech)
    recording_manifest_path = manifests_dir / f"librispeech_recordings_{part_name}.jsonl.gz"
    recording_set.to_file(recording_manifest_path)
    logging.info(f"Saved recording manifest to {recording_manifest_path}")
    
    # Build SupervisionSet from collected supervision dicts
    supervision_set = SupervisionSet.from_dicts(all_supervisions)
    supervision_manifest_path = manifests_dir / f"librispeech_supervisions_{part_name}.jsonl.gz"
    supervision_set.to_file(supervision_manifest_path)
    logging.info(f"Saved supervision manifest to {supervision_manifest_path}")
    
    # Create cuts from manifests (needed by downstream scripts)
    cuts = CutSet.from_manifests(recordings=recording_set, supervisions=supervision_set)
    
    # Save cuts to multiple locations for compatibility
    # 1. Original output_dir location
    cuts_path = output_dir / f"librilight_{subset}_cuts.jsonl.gz"
    logging.info(f"About to save {len(cuts)} cuts to {cuts_path}")
    try:
        cuts.to_file(cuts_path)
        logging.info(f"Saved cuts to {cuts_path}")
        
        # Verify the saved file
        test_cuts = CutSet.from_file(cuts_path)
        logging.info(f"Verification: Successfully loaded {len(test_cuts)} cuts from saved file")
    except Exception as e:
        logging.error(f"Failed to save/verify cuts to {cuts_path}: {e}")
        # Try alternative saving method
        cuts_path_alt = output_dir / f"librilight_{subset}_cuts.jsonl"
        logging.info(f"Trying to save without compression to {cuts_path_alt}")
        cuts.to_file(cuts_path_alt)
        logging.info(f"Saved uncompressed cuts to {cuts_path_alt}")
    
    # 2. fbank directory with training naming convention for asr_datamodule.py
    fbank_cuts_path = fbank_dir / f"librilight_{subset}_cuts_train.jsonl.gz"
    try:
        cuts.to_file(fbank_cuts_path)
        logging.info(f"Saved training cuts to {fbank_cuts_path}")
    except Exception as e:
        logging.error(f"Failed to save training cuts: {e}")
    
    # 3. Alternative naming convention
    alt_cuts_path = fbank_dir / f"cuts_train_{subset}.jsonl.gz"
    try:
        cuts.to_file(alt_cuts_path)
        logging.info(f"Saved alternative cuts to {alt_cuts_path}")
    except Exception as e:
        logging.error(f"Failed to save alternative cuts: {e}")
    
    # Also save to output_dir for backward compatibility
    recording_set.to_file(output_dir / f"librilight_{subset}_recordings.jsonl.gz")
    supervision_set.to_file(output_dir / f"librilight_{subset}_supervisions.jsonl.gz")
    
    # Print statistics
    # Total duration is computed from supervisions (sum of segment durations)
    total_duration = sum(sup["duration"] for sup in all_supervisions)
    # Extract speakers from supervisions
    speakers = set(sup["speaker"] for sup in all_supervisions)

    print(f"\n=== LibriLight {subset.upper()} Statistics ===")
    print(f"Total recordings (files): {len(recordings_map)}")
    print(f"Total segments: {len(all_supervisions)}")
    print(f"Total duration: {total_duration/3600:.2f} hours")
    print(f"Average segment duration: {total_duration/len(all_supervisions):.2f} seconds")
    print(f"Number of speakers: {len(speakers)}")
    print(f"Output directory: {output_dir}")
    print(f"Manifests saved to: {manifests_dir}")
    print(f"Training cuts saved to: {fbank_dir}")
    print(f"Part name: {part_name}")
    print(f"\nFiles created:")
    print(f"  - {recording_manifest_path}")
    print(f"  - {supervision_manifest_path}")
    print(f"  - {cuts_path}")
    print(f"  - {fbank_cuts_path}")
    print(f"  - {alt_cuts_path}")


def main():
    args = get_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    create_librilight_manifests(
        librilight_dir=args.librilight_dir,
        output_dir=args.output_dir,
        subset=args.subset,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_snr=args.min_snr,
        num_jobs=args.num_jobs,
    )


if __name__ == "__main__":
    main()