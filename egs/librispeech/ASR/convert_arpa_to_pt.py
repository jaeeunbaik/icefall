#!/usr/bin/env python3
"""
Convert ARPA language model to k2 FSA format (.pt file)

Usage:
    python convert_arpa_to_pt.py \
        --arpa-file data/lm/2gram.arpa \
        --lang-dir data/lang_bpe_1024 \
        --output data/lm/G_2_gram.pt
"""

import argparse
import k2
import torch
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arpa-file",
        type=str,
        required=True,
        help="Path to ARPA language model file",
    )
    parser.add_argument(
        "--lang-dir",
        type=str,
        required=True,
        help="Path to lang directory containing tokens.txt",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for .pt file",
    )
    return parser.parse_args()


def main():
    args = get_args()
    
    arpa_file = Path(args.arpa_file)
    lang_dir = Path(args.lang_dir)
    output_file = Path(args.output)
    
    assert arpa_file.exists(), f"ARPA file not found: {arpa_file}"
    assert lang_dir.exists(), f"Lang directory not found: {lang_dir}"
    
    tokens_file = lang_dir / "tokens.txt"
    assert tokens_file.exists(), f"tokens.txt not found in {lang_dir}"
    
    print(f"Loading tokens from {tokens_file}")
    # Load token list
    token_list = []
    with open(tokens_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 1:
                    token_list.append(parts[0])
    
    print(f"Found {len(token_list)} tokens")
    print(f"Converting ARPA file: {arpa_file}")
    
    # Convert ARPA to k2 FSA
    # Note: k2 expects token IDs, not token strings
    # You may need to use arpa2fst from OpenFST first
    print("Warning: Direct ARPA to k2.Fsa conversion is complex.")
    print("Recommended approach:")
    print("1. Use OpenFST's arpa2fst to convert ARPA to FST text format")
    print("2. Use k2.Fsa.from_openfst() to load the FST")
    print()
    print("Commands:")
    print(f"  # Convert ARPA to FST text format")
    print(f"  arpa2fst --disambig-symbol=#0 {arpa_file} > {output_file.with_suffix('.fst.txt')}")
    print(f"")
    print(f"  # Then use the prepare_lm.sh script to convert FST to .pt")
    print(f"  # Or use k2.Fsa.from_openfst() in your code")
    
    # If you have the FST text format, you can convert it like this:
    fst_txt_file = arpa_file.with_suffix('.fst.txt')
    if fst_txt_file.exists():
        print(f"\nFound FST text file: {fst_txt_file}")
        print("Loading FST...")
        
        with open(fst_txt_file, 'r') as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
        
        # Remove aux_labels if present (not needed for token-level LM)
        if hasattr(G, 'aux_labels'):
            del G.aux_labels
        
        # Add epsilon self-loops for rescoring
        G = k2.add_epsilon_self_loops(G)
        G = k2.arc_sort(G)
        
        # Save as .pt
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(G.as_dict(), output_file)
        print(f"Saved to: {output_file}")
    else:
        print(f"\nFST text file not found: {fst_txt_file}")
        print("Please run arpa2fst first (see commands above)")


if __name__ == "__main__":
    main()
