import sys
from lhotse import load_manifest_lazy

if len(sys.argv) < 2:
    print("Usage: python check_features.py <cuts_manifest.jsonl.gz> [<cuts_manifest2.jsonl.gz> ...]")
    sys.exit(1)

for manifest_path in sys.argv[1:]:
    print(f"Checking: {manifest_path}")
    cuts = load_manifest_lazy(manifest_path)
    total = 0
    missing = 0
    for cut in cuts:
        total += 1
        if not cut.has_features:
            missing += 1
    print(f"  Total cuts: {total}")
    print(f"  Cuts missing features: {missing}")
    if missing > 0:
        print(f"  WARNING: {missing} cuts are missing features!")
    else:
        print("  All cuts have features.")
