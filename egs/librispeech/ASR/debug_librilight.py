#!/usr/bin/env python3

"""
Debug script to check LibriLight manifest files
"""

import logging
import gzip
import json
from pathlib import Path
from lhotse import CutSet, load_manifest

logging.basicConfig(level=logging.INFO)

def debug_manifest_file(file_path):
    """Debug a manifest file to see what's wrong"""
    print(f"Debugging file: {file_path}")
    
    if not file_path.exists():
        print(f"ERROR: File does not exist: {file_path}")
        return
    
    try:
        # Check file size
        file_size = file_path.stat().st_size
        print(f"File size: {file_size} bytes")
        
        # Try to read first few lines
        print("First 3 lines:")
        with gzip.open(file_path, 'rt') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                print(f"Line {i+1}: {line.strip()[:200]}...")
                
                # Try to parse the line as JSON
                try:
                    obj = json.loads(line)
                    print(f"  -> Valid JSON with keys: {list(obj.keys())}")
                    if 'type' in obj:
                        print(f"  -> Object type: {obj['type']}")
                except json.JSONDecodeError as e:
                    print(f"  -> JSON parse error: {e}")
        
        # Try to load with Lhotse
        print("\nTrying to load with lhotse...")
        try:
            cuts = load_manifest(file_path)
            print(f"SUCCESS: Loaded {len(cuts)} items with load_manifest")
        except Exception as e:
            print(f"load_manifest failed: {e}")
            
        try:
            cuts = CutSet.from_file(file_path)
            print(f"SUCCESS: Loaded {len(cuts)} cuts with CutSet.from_file")
        except Exception as e:
            print(f"CutSet.from_file failed: {e}")
            
    except Exception as e:
        print(f"FATAL ERROR: {e}")

def main():
    # Check all manifest files
    base_dir = Path("data/librilight_medium")
    files_to_check = [
        base_dir / "librilight_medium_cuts.jsonl.gz",
        base_dir / "librilight_medium_recordings.jsonl.gz", 
        base_dir / "librilight_medium_supervisions.jsonl.gz"
    ]
    
    for file_path in files_to_check:
        debug_manifest_file(file_path)
        print("=" * 60)

if __name__ == "__main__":
    main()