#!/usr/bin/env python3
"""
Test LibriLight metadata format and validate our preparation script.
"""

import json
import logging
from pathlib import Path

def test_librilight_metadata(json_path: str):
    """Test LibriLight JSON metadata format."""
    
    print(f"Testing metadata file: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print("✅ JSON loaded successfully")
        
        # Check expected fields
        expected_fields = ['speaker', 'snr', 'voice_activity', 'book_meta']
        
        for field in expected_fields:
            if field in metadata:
                print(f"✅ Found field: {field}")
            else:
                print(f"❌ Missing field: {field}")
        
        # Check SNR
        snr = metadata.get('snr', None)
        if snr is not None:
            print(f"📊 SNR: {snr:.2f}")
        
        # Check voice activity
        voice_activity = metadata.get('voice_activity', [])
        print(f"🎤 Voice activity segments: {len(voice_activity)}")
        
        if voice_activity:
            # Show first few segments
            for i, segment in enumerate(voice_activity[:3]):
                if len(segment) >= 2:
                    start, end = segment[0], segment[1]
                    duration = end - start
                    print(f"   Segment {i+1}: {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")
        
        # Check speaker info
        speaker = metadata.get('speaker', 'unknown')
        print(f"👤 Speaker ID: {speaker}")
        
        # Check book metadata
        book_meta = metadata.get('book_meta', {})
        if book_meta:
            title = book_meta.get('title', 'unknown')
            genre = book_meta.get('genre', [])
            print(f"📚 Book: {title}")
            if genre:
                print(f"📖 Genre: {', '.join(genre)}")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error loading {json_path}: {e}")

def main():
    """Test LibriLight metadata format."""
    
    # Test paths
    librilight_root = Path("/home/hdd1/jenny/LibriLight-m/medium")
    
    print("LibriLight Metadata Format Test")
    print("=" * 50)
    
    if not librilight_root.exists():
        print(f"❌ LibriLight directory not found: {librilight_root}")
        print("Please make sure the dataset is downloaded and extracted.")
        return
    
    # Find JSON files
    json_files = list(librilight_root.rglob("*.json"))
    
    if not json_files:
        print("❌ No JSON files found in LibriLight directory")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    # Test first few files
    for i, json_file in enumerate(json_files[:5]):
        test_librilight_metadata(str(json_file))
        if i < len(json_files) - 1:
            print()

if __name__ == "__main__":
    main()