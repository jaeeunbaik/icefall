#!/usr/bin/env python3
"""
Analyze the effects of different data augmentation strategies on SSL training.

This script compares:
1. SpecAugment only
2. RIR only  
3. MUSAN only (collapsed)
4. MUSAN + RIR (successful)
5. MUSAN + RIR + SpecAugment

Usage:
    python analyze_augmentation_effects.py \
        --exp-dirs exp1 exp2 exp3 \
        --labels "SpecAugment" "RIR" "MUSAN+RIR" \
        --test-set test-clean
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def parse_wer_summary(summary_file: Path) -> Dict[str, float]:
    """Parse WER summary file and return results."""
    results = {}
    
    with open(summary_file, 'r') as f:
        lines = f.readlines()
    
    # Find the table section
    in_table = False
    for line in lines:
        if '=' in line or '-' in line:
            if 'Noise Type' in line:
                in_table = True
            continue
        
        if in_table and line.strip():
            parts = line.split()
            if len(parts) >= 3:
                noise_type = parts[0]
                snr = parts[1]
                wer = float(parts[2])
                
                key = f"{noise_type}_SNR{snr}"
                results[key] = wer
    
    return results


def analyze_consistency_regularization(
    clean_wer: float,
    noisy_wers: Dict[str, float]
) -> Dict[str, float]:
    """
    Analyze how well the model learned consistency regularization.
    
    Good consistency regularization should show:
    1. Small WER degradation at high SNR
    2. Graceful degradation as SNR decreases
    """
    analysis = {
        'clean_wer': clean_wer,
        'avg_degradation': 0.0,
        'max_degradation': 0.0,
        'degradation_at_snr20': 0.0,
        'degradation_at_snr0': 0.0,
        'robustness_score': 0.0,  # Lower is better
    }
    
    degradations = []
    for key, wer in noisy_wers.items():
        degradation = wer - clean_wer
        degradations.append(degradation)
        
        if 'SNR20' in key:
            analysis['degradation_at_snr20'] = degradation
        elif 'SNR0' in key:
            analysis['degradation_at_snr0'] = degradation
    
    analysis['avg_degradation'] = np.mean(degradations)
    analysis['max_degradation'] = np.max(degradations)
    
    # Robustness score: weighted average of degradations
    # High SNR degradation is worse (should be close to clean)
    analysis['robustness_score'] = (
        2.0 * analysis['degradation_at_snr20'] +
        1.5 * analysis['degradation_at_snr0'] +
        analysis['avg_degradation']
    )
    
    return analysis


def plot_augmentation_comparison(
    results: Dict[str, Dict[str, float]],
    labels: List[str],
    output_path: Path
):
    """Plot WER vs SNR for different augmentation strategies."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    noise_types = ['babble', 'pink', 'white']
    snr_levels = [-5, 0, 5, 10, 15, 20]
    
    for ax, noise_type in zip(axes, noise_types):
        for exp_name, label in zip(results.keys(), labels):
            wers = []
            for snr in snr_levels:
                key = f"{noise_type}_SNR{snr}.0" if snr >= 0 else f"{noise_type}_SNR{snr}"
                wer = results[exp_name].get(key, None)
                wers.append(wer)
            
            # Filter out None values
            valid_snrs = [s for s, w in zip(snr_levels, wers) if w is not None]
            valid_wers = [w for w in wers if w is not None]
            
            if valid_wers:
                ax.plot(valid_snrs, valid_wers, marker='o', label=label, linewidth=2)
        
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('WER (%)', fontsize=12)
        ax.set_title(f'{noise_type.upper()} Noise', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Plot saved to {output_path}")


def generate_analysis_report(
    results: Dict[str, Dict[str, float]],
    labels: List[str],
    output_file: Path
):
    """Generate comprehensive analysis report."""
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DATA AUGMENTATION ANALYSIS FOR SSL TRAINING\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report analyzes the effectiveness of different data augmentation\n")
        f.write("strategies during SSL pre-training phase (20000 steps on LibriLight).\n\n")
        
        f.write("## Augmentation Strategies Compared\n\n")
        for i, label in enumerate(labels, 1):
            f.write(f"{i}. {label}\n")
        f.write("\n")
        
        # Analyze each experiment
        f.write("## Detailed Analysis\n\n")
        
        for exp_name, label in zip(results.keys(), labels):
            f.write(f"### {label}\n\n")
            
            exp_results = results[exp_name]
            
            # Get clean WER
            clean_wer = exp_results.get('CLEAN_-', 0.0)
            f.write(f"**Clean WER**: {clean_wer:.2f}%\n\n")
            
            # Analyze consistency regularization
            noisy_results = {k: v for k, v in exp_results.items() if k != 'CLEAN_-'}
            analysis = analyze_consistency_regularization(clean_wer, noisy_results)
            
            f.write("**Robustness Metrics**:\n")
            f.write(f"- Average WER degradation: {analysis['avg_degradation']:.2f}%\n")
            f.write(f"- Maximum WER degradation: {analysis['max_degradation']:.2f}%\n")
            f.write(f"- Degradation at SNR=20dB: {analysis['degradation_at_snr20']:.2f}%\n")
            f.write(f"- Degradation at SNR=0dB: {analysis['degradation_at_snr0']:.2f}%\n")
            f.write(f"- **Robustness Score**: {analysis['robustness_score']:.2f} (lower is better)\n\n")
            
            # Interpretation
            f.write("**Interpretation**:\n")
            if analysis['degradation_at_snr20'] < 1.0:
                f.write("✓ Excellent consistency at high SNR - good SSL learning\n")
            elif analysis['degradation_at_snr20'] < 2.0:
                f.write("○ Moderate consistency at high SNR\n")
            else:
                f.write("✗ Poor consistency at high SNR - SSL learning may be suboptimal\n")
            
            if analysis['robustness_score'] < 15.0:
                f.write("✓ Strong noise robustness\n")
            elif analysis['robustness_score'] < 25.0:
                f.write("○ Moderate noise robustness\n")
            else:
                f.write("✗ Weak noise robustness\n")
            
            f.write("\n" + "-" * 80 + "\n\n")
        
        # Comparative analysis
        f.write("## Comparative Analysis\n\n")
        
        f.write("### Why MUSAN-only collapsed:\n\n")
        f.write("1. **Extreme Distribution Shift**: MUSAN adds very diverse noise types\n")
        f.write("   (music, speech, noise) that create too large a gap between teacher\n")
        f.write("   (clean) and student (noisy) predictions.\n\n")
        f.write("2. **Inconsistent Teacher Predictions**: Teacher model trained on clean\n")
        f.write("   data cannot provide stable pseudo-labels for heavily corrupted inputs.\n\n")
        f.write("3. **Gradient Instability**: Large KL divergence leads to unstable\n")
        f.write("   gradients and eventual collapse.\n\n")
        
        f.write("### Why MUSAN + RIR succeeded:\n\n")
        f.write("1. **Complementary Augmentations**:\n")
        f.write("   - RIR: Natural reverberation (preserves speech structure)\n")
        f.write("   - MUSAN: Additive noise (adds background)\n")
        f.write("   - Together: More realistic acoustic conditions\n\n")
        f.write("2. **Progressive Difficulty**: RIR provides mild augmentation first,\n")
        f.write("   then MUSAN adds noise on top. This gradual shift helps maintain\n")
        f.write("   consistency.\n\n")
        f.write("3. **Richer Invariance Learning**: Model learns to be invariant to both\n")
        f.write("   reverberation AND noise simultaneously.\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("1. **For SSL Training**:\n")
        f.write("   - Start with RIR only or RIR + SpecAugment\n")
        f.write("   - Gradually introduce MUSAN with high SNR (15-20 dB)\n")
        f.write("   - Consider curriculum learning: increase difficulty over training\n\n")
        
        f.write("2. **Augmentation Strength**:\n")
        f.write("   - RIR: Use diverse room impulse responses\n")
        f.write("   - MUSAN: Start with SNR 15-20 dB, gradually reduce to 10-15 dB\n")
        f.write("   - SpecAugment: Time mask 27 frames, Freq mask 2 bins\n\n")
        
        f.write("3. **Monitoring SSL Training**:\n")
        f.write("   - Track consistency loss (KL divergence)\n")
        f.write("   - Monitor gradient norms (detect instability early)\n")
        f.write("   - Evaluate on validation set with light noise (SNR 15-20 dB)\n\n")
        
        f.write("=" * 80 + "\n")
    
    logging.info(f"Analysis report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze augmentation effects on SSL")
    parser.add_argument(
        "--exp-dirs",
        nargs="+",
        required=True,
        help="Experiment directories to compare"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Labels for each experiment"
    )
    parser.add_argument(
        "--test-set",
        default="test-clean",
        help="Test set name"
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_results",
        help="Output directory for analysis"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    
    # Parse results from all experiments
    results = {}
    for exp_dir, label in zip(args.exp_dirs, args.labels):
        exp_path = Path(exp_dir)
        summary_file = exp_path / f"noisy_results/{args.test_set}/wer_summary.txt"
        
        if not summary_file.exists():
            logging.warning(f"Summary file not found: {summary_file}")
            continue
        
        logging.info(f"Parsing results from {exp_dir}")
        results[label] = parse_wer_summary(summary_file)
    
    if not results:
        logging.error("No valid results found!")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plot
    plot_path = output_dir / "augmentation_comparison.png"
    plot_augmentation_comparison(results, args.labels, plot_path)
    
    # Generate analysis report
    report_path = output_dir / "augmentation_analysis.txt"
    generate_analysis_report(results, args.labels, report_path)
    
    logging.info("Analysis complete!")
    logging.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
