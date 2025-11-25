#!/usr/bin/env python3
"""
Average multiple model checkpoints to create a more robust model.

Usage:
    # Method 1: Specify checkpoint files directly
    python average_checkpoints.py \
        --checkpoints step-60000.pt step-90000.pt step-120000.pt \
        --output averaged_60-120k.pt
    
    # Method 2: Specify step numbers
    python average_checkpoints.py \
        --checkpoint-dir libri-light/exp_kl_layer6,12,18_specaug/models \
        --steps 60000 90000 120000 \
        --output averaged_60-120k.pt
    
    # Method 3: Specify max step and number of models (均等分布)
    python average_checkpoints.py \
        --checkpoint-dir libri-light/exp_kl_layer6,12,18_specaug/models \
        --step 20000 \
        --num 10 \
        --output averaged_top10.pt
"""

import argparse
import logging
import torch
from pathlib import Path
from collections import OrderedDict


def get_parser():
    parser = argparse.ArgumentParser(
        description="Average multiple model checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        help="List of checkpoint files to average",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory containing checkpoints (alternative to --checkpoints)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        help="Step numbers to average (used with --checkpoint-dir)",
    )
    parser.add_argument(
        "--step",
        type=int,
        help="Maximum step number (used with --num to auto-generate step list)",
    )
    parser.add_argument(
        "--num",
        type=int,
        help="Number of models to average (used with --step, evenly distributed)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for averaged checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load checkpoints on (cpu/cuda)",
    )
    
    return parser


def load_checkpoint(path, device="cpu"):
    """Load a checkpoint file."""
    logging.info(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device)
    
    # Extract model state dict
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    return state_dict


def average_checkpoints(checkpoint_paths, output_path, device="cpu"):
    """
    Average multiple model checkpoints.
    
    Args:
        checkpoint_paths: List of paths to checkpoint files
        output_path: Path to save averaged checkpoint
        device: Device to perform averaging on
    """
    if len(checkpoint_paths) == 0:
        raise ValueError("No checkpoints provided for averaging")
    
    logging.info(f"Averaging {len(checkpoint_paths)} checkpoints:")
    for path in checkpoint_paths:
        logging.info(f"  - {path}")
    
    # Load first checkpoint to get structure
    avg_state_dict = load_checkpoint(checkpoint_paths[0], device)
    
    # Convert to float for averaging
    for key in avg_state_dict.keys():
        avg_state_dict[key] = avg_state_dict[key].float()
    
    # Add remaining checkpoints
    for checkpoint_path in checkpoint_paths[1:]:
        state_dict = load_checkpoint(checkpoint_path, device)
        
        for key in avg_state_dict.keys():
            if key in state_dict:
                avg_state_dict[key] += state_dict[key].float()
            else:
                logging.warning(f"Key {key} not found in {checkpoint_path}")
    
    # Divide by number of checkpoints to get average
    num_checkpoints = len(checkpoint_paths)
    for key in avg_state_dict.keys():
        avg_state_dict[key] = avg_state_dict[key] / num_checkpoints
    
    # Save averaged checkpoint
    logging.info(f"Saving averaged checkpoint to: {output_path}")
    torch.save(avg_state_dict, output_path)
    logging.info("Done!")
    
    # Print statistics
    total_params = sum(p.numel() for p in avg_state_dict.values())
    logging.info(f"Total parameters in averaged model: {total_params:,}")


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    
    # Determine checkpoint paths
    if args.checkpoints:
        checkpoint_paths = [Path(cp) for cp in args.checkpoints]
    elif args.checkpoint_dir and args.step:
        # NEW METHOD: Auto-generate checkpoints at 1000-step intervals up to max_step.
        # If --num is provided, take the last `num` checkpoints from that list.
        checkpoint_dir = Path(args.checkpoint_dir)
        max_step = args.step

        # Generate steps every 1000 up to max_step (1000, 2000, ..., max_step)
        steps = list(range(1000, max_step + 1, 1000))

        if args.num:
            num_models = args.num
            if num_models <= 0:
                parser.error("--num must be > 0")
            if num_models < len(steps):
                # keep the last `num_models` steps (i.e., the most recent ones)
                steps = steps[-num_models:]

        logging.info(f"Auto-generating checkpoints every 1000 steps up to {max_step}")
        logging.info(f"Steps to consider: {steps}")

        checkpoint_paths = []
        for step in steps:
            # Try different naming patterns; pick the most recently matched file if multiple
            patterns = [
                f"step-{step}.pt",
                f"step-{step}-*.pt",
                f"epoch-*-step-{step}.pt",
            ]

            found = False
            for pattern in patterns:
                matches = sorted(checkpoint_dir.glob(pattern))
                if matches:
                    checkpoint_paths.append(matches[-1])
                    found = True
                    break

            if not found:
                logging.warning(f"Checkpoint not found for step {step} in {checkpoint_dir}")
    elif args.checkpoint_dir and args.steps:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_paths = []
        
        for step in args.steps:
            # Try different naming patterns
            patterns = [
                f"step-{step}.pt",
                f"step-{step}-*.pt",
                f"epoch-*-step-{step}.pt",
            ]
            
            found = False
            for pattern in patterns:
                matches = list(checkpoint_dir.glob(pattern))
                if matches:
                    checkpoint_paths.append(matches[0])
                    found = True
                    break
            
            if not found:
                logging.warning(f"Checkpoint not found for step {step} in {checkpoint_dir}")
    else:
        parser.error("Must provide either --checkpoints, --checkpoint-dir with --steps, or --checkpoint-dir with --step and --num")
    
    # Verify all checkpoints exist
    missing = [cp for cp in checkpoint_paths if not cp.exists()]
    if missing:
        logging.error("The following checkpoints do not exist:")
        for cp in missing:
            logging.error(f"  - {cp}")
        return
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Average checkpoints
    average_checkpoints(checkpoint_paths, output_path, args.device)


if __name__ == "__main__":
    main()
