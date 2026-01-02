#!/usr/bin/env python3
"""
Check what noise files are available in a directory.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noise-dir",
        type=str,
        required=True,
        help="Directory to check for noise files",
    )
    args = parser.parse_args()
    
    noise_dir = Path(args.noise_dir)
    
    if not noise_dir.exists():
        print(f"❌ Directory does not exist: {noise_dir}")
        return
    
    print(f"📁 Checking: {noise_dir}")
    print("=" * 80)
    
    # Find all audio files
    audio_extensions = ['.wav', '.WAV', '.sph', '.SPH', '.flac', '.FLAC', '.mp3', '.MP3']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(list(noise_dir.glob(f"**/*{ext}")))
    
    if not audio_files:
        print("❌ No audio files found!")
        print(f"\nSearched for extensions: {audio_extensions}")
        
        # List all files
        all_files = list(noise_dir.glob("**/*"))
        files = [f for f in all_files if f.is_file()]
        if files:
            print(f"\n📋 Found {len(files)} files (non-audio):")
            for f in files[:20]:
                print(f"  - {f.name}")
            if len(files) > 20:
                print(f"  ... and {len(files) - 20} more")
        return
    
    print(f"✅ Found {len(audio_files)} audio files:\n")
    
    # Group by name pattern
    from collections import defaultdict
    grouped = defaultdict(list)
    
    for f in audio_files:
        name_lower = f.stem.lower()
        
        # Categorize
        if 'babble' in name_lower:
            grouped['babble'].append(f)
        elif 'pink' in name_lower:
            grouped['pink'].append(f)
        elif 'white' in name_lower:
            grouped['white'].append(f)
        elif 'music' in name_lower:
            grouped['music'].append(f)
        elif 'noise' in name_lower:
            grouped['noise'].append(f)
        else:
            grouped['other'].append(f)
    
    # Print grouped
    for category, files in sorted(grouped.items()):
        print(f"\n{category.upper()} ({len(files)} files):")
        for f in files[:10]:
            print(f"  - {f.relative_to(noise_dir)}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more")
    
    print("\n" + "=" * 80)
    print("💡 Suggested noise types to use:")
    for category in sorted(grouped.keys()):
        if category != 'other' and grouped[category]:
            print(f"  - {category}")


if __name__ == "__main__":
    main()
