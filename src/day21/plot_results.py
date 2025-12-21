"""
Plot Monad Training Results - Baseline vs NEFTune Comparison
============================================================
"""

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Clean styling
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
})

# Christmas colors
BASELINE_COLOR = "#228B22"  # Forest Green
NEFTUNE_COLOR = "#C41E3A"   # Cardinal Red
GOLD_COLOR = "#DAA520"      # Goldenrod
GRID_COLOR = "#D4E6D4"      # Light green grid
BG_COLOR = "#FFFEF5"        # Warm off-white


def load_training_metrics(results_dir: str) -> pd.DataFrame:
    """Load training metrics from CSV."""
    csv_path = os.path.join(results_dir, "logs", "training_metrics.csv")
    return pd.read_csv(csv_path)


def load_mmlu_evals(results_dir: str) -> dict:
    """Load all MMLU evaluation results."""
    mmlu_dir = os.path.join(results_dir, "mmlu_eval_standard")
    evals = {}
    
    for f in sorted(Path(mmlu_dir).glob("step_*.json")):
        step_str = f.stem.replace("step_", "")
        step = 9999 if step_str == "final" else int(step_str)
        
        with open(f) as fp:
            data = json.load(fp)
        
        evals[step] = {
            "strictly_valid": data["summary"]["strictly_valid"],
            "accuracy_strict": data["summary"]["accuracy_strict"],
            "accuracy_all": data["summary"]["accuracy_all"],
            "p_value": data["statistical_test"]["p_value"],
        }
    
    return evals


def style_axis(ax):
    """Apply Christmas styling to an axis."""
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.4, color=GRID_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BASELINE_COLOR)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_color(BASELINE_COLOR)
    ax.spines['bottom'].set_linewidth(1.5)


def plot_training_loss(df_baseline: pd.DataFrame, df_neftune: pd.DataFrame, ax: plt.Axes):
    """Plot training loss comparison."""
    style_axis(ax)
    
    # Plot lines
    ax.plot(df_baseline['step'], df_baseline['loss'], color=BASELINE_COLOR, 
            linewidth=2, alpha=0.9, label='Baseline')
    ax.plot(df_neftune['step'], df_neftune['loss'], color=NEFTUNE_COLOR, 
            linewidth=2, alpha=0.9, label='NEFTune')
    
    # Find minimums
    min_idx_b = df_baseline['loss'].idxmin()
    min_step_b = df_baseline.loc[min_idx_b, 'step']
    min_loss_b = df_baseline.loc[min_idx_b, 'loss']
    
    min_idx_n = df_neftune['loss'].idxmin()
    min_step_n = df_neftune.loc[min_idx_n, 'step']
    min_loss_n = df_neftune.loc[min_idx_n, 'loss']
    
    # Mark minimums
    ax.scatter([min_step_b], [min_loss_b], color=BASELINE_COLOR, s=100, zorder=5, 
               marker='o', edgecolor='white', linewidth=2)
    ax.scatter([min_step_n], [min_loss_n], color=NEFTUNE_COLOR, s=100, zorder=5,
               marker='o', edgecolor='white', linewidth=2)
    
    # Annotate minimums
    ax.annotate(f'Min: {min_loss_b:.3f}', xy=(min_step_b, min_loss_b),
                xytext=(min_step_b + 200, min_loss_b + 0.8),
                fontsize=10, color=BASELINE_COLOR, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=BASELINE_COLOR, lw=1.5))
    
    ax.annotate(f'Min: {min_loss_n:.3f}', xy=(min_step_n, min_loss_n),
                xytext=(min_step_n - 400, min_loss_n + 1.2),
                fontsize=10, color=NEFTUNE_COLOR, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=NEFTUNE_COLOR, lw=1.5))
    
    # Winner annotation
    if min_loss_n < min_loss_b:
        diff = min_loss_b - min_loss_n
        winner_text = f"NEFTune lower by {diff:.3f}"
    else:
        diff = min_loss_n - min_loss_b
        winner_text = f"Baseline lower by {diff:.3f}"
    
    ax.text(0.98, 0.95, winner_text, transform=ax.transAxes, fontsize=11,
            ha='right', va='top', fontweight='bold', color=BASELINE_COLOR,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=BASELINE_COLOR, linewidth=1.5))
    
    ax.set_xlabel('Training Step', fontsize=12, fontweight='bold', color='#333333')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold', color='#333333')
    ax.set_title('Training Loss: Baseline vs NEFTune', fontsize=14, fontweight='bold', 
                 pad=10, color=BASELINE_COLOR)
    
    # Legend positioned to avoid overlap
    ax.legend(loc='center right', frameon=True, fancybox=False, 
              edgecolor='#666666', facecolor='white')
    
    max_step = max(df_baseline['step'].max(), df_neftune['step'].max())
    ax.set_xlim(0, max_step * 1.02)


def plot_mmlu_comparison(evals_baseline: dict, evals_neftune: dict, axes: tuple):
    """Plot MMLU valid counts and accuracy comparison."""
    ax1, ax2 = axes
    
    # Get common steps (excluding step 0 for cleaner plot)
    all_steps = sorted(set(evals_baseline.keys()) | set(evals_neftune.keys()))
    # Filter to just a few key checkpoints for readability
    if len(all_steps) > 10:
        # Keep init, final, and evenly spaced
        key_steps = [all_steps[0]] + all_steps[1::3] + [all_steps[-1]]
        all_steps = sorted(set(key_steps))
    
    labels = ["Init" if s == 0 else ("Final" if s == 9999 else f"{s}") for s in all_steps]
    x = np.arange(len(all_steps))
    width = 0.35
    
    for ax in axes:
        style_axis(ax)
    
    # =========================================================================
    # Plot 1: Strictly Valid Counts
    # =========================================================================
    valid_b = [evals_baseline.get(s, {}).get('strictly_valid', 0) for s in all_steps]
    valid_n = [evals_neftune.get(s, {}).get('strictly_valid', 0) for s in all_steps]
    
    bars1 = ax1.bar(x - width/2, valid_b, width, color=BASELINE_COLOR, alpha=0.85, label='Baseline')
    bars2 = ax1.bar(x + width/2, valid_n, width, color=NEFTUNE_COLOR, alpha=0.85, label='NEFTune')
    
    ax1.set_ylabel('Strictly Valid Responses', fontsize=11, fontweight='bold', color='#333333')
    ax1.set_title('Format Compliance (Valid Responses)', fontsize=13, fontweight='bold', 
                  pad=10, color=BASELINE_COLOR)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha='right')
    ax1.legend(loc='lower right', frameon=True, fancybox=False, 
               edgecolor='#666666', facecolor='white')
    
    # Add delta labels on top
    for i, (b, n) in enumerate(zip(valid_b, valid_n)):
        if n > 0 or b > 0:
            delta = n - b
            sign = "+" if delta > 0 else ""
            color = NEFTUNE_COLOR if delta > 0 else BASELINE_COLOR
            max_val = max(b, n)
            ax1.text(x[i], max_val + 200, f'{sign}{delta}', ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color=color)
    
    ax1.set_ylim(0, max(max(valid_b), max(valid_n)) * 1.15)
    
    # =========================================================================
    # Plot 2: Accuracy
    # =========================================================================
    acc_b = [evals_baseline.get(s, {}).get('accuracy_strict', 0) * 100 for s in all_steps]
    acc_n = [evals_neftune.get(s, {}).get('accuracy_strict', 0) * 100 for s in all_steps]
    
    bars3 = ax2.bar(x - width/2, acc_b, width, color=BASELINE_COLOR, alpha=0.85, label='Baseline')
    bars4 = ax2.bar(x + width/2, acc_n, width, color=NEFTUNE_COLOR, alpha=0.85, label='NEFTune')
    
    # Random baseline (gold for Christmas!)
    ax2.axhline(y=25, color=GOLD_COLOR, linestyle='--', linewidth=2.5, alpha=0.9, label='Random (25%)')
    
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold', color='#333333')
    ax2.set_title('MMLU Accuracy (Strict)', fontsize=13, fontweight='bold', 
                  pad=10, color=BASELINE_COLOR)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha='right')
    ax2.legend(loc='lower right', frameon=True, fancybox=False, 
               edgecolor='#666666', facecolor='white')
    
    # Add delta labels on top
    for i, (b, n) in enumerate(zip(acc_b, acc_n)):
        if n > 0 or b > 0:
            delta = n - b
            sign = "+" if delta > 0 else ""
            color = NEFTUNE_COLOR if delta > 0 else BASELINE_COLOR
            max_val = max(b, n)
            ax2.text(x[i], max_val + 0.8, f'{sign}{delta:.1f}%', ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color=color)
    
    ax2.set_ylim(0, max(max(acc_b), max(acc_n)) * 1.18)


def plot_normalized_accuracy(evals_baseline: dict, evals_neftune: dict, ax: plt.Axes):
    """Plot normalized accuracy: correct / max valid from either method."""
    style_axis(ax)
    
    # Get steps
    all_steps = sorted(set(evals_baseline.keys()) | set(evals_neftune.keys()))
    if len(all_steps) > 10:
        key_steps = [all_steps[0]] + all_steps[1::3] + [all_steps[-1]]
        all_steps = sorted(set(key_steps))
    
    labels = ["Init" if s == 0 else ("Final" if s == 9999 else f"{s}") for s in all_steps]
    x = np.arange(len(all_steps))
    width = 0.35
    
    # Get valid counts
    valid_b = [evals_baseline.get(s, {}).get('strictly_valid', 0) for s in all_steps]
    valid_n = [evals_neftune.get(s, {}).get('strictly_valid', 0) for s in all_steps]
    
    # Get correct counts
    correct_b = [int(evals_baseline.get(s, {}).get('accuracy_strict', 0) * 
                     evals_baseline.get(s, {}).get('strictly_valid', 0)) for s in all_steps]
    correct_n = [int(evals_neftune.get(s, {}).get('accuracy_strict', 0) * 
                     evals_neftune.get(s, {}).get('strictly_valid', 0)) for s in all_steps]
    
    # Normalize by max valid from either method at each step
    norm_acc_b = []
    norm_acc_n = []
    for i, s in enumerate(all_steps):
        max_valid = max(valid_b[i], valid_n[i])
        if max_valid > 0:
            norm_acc_b.append(correct_b[i] / max_valid * 100)
            norm_acc_n.append(correct_n[i] / max_valid * 100)
        else:
            norm_acc_b.append(0)
            norm_acc_n.append(0)
    
    bars1 = ax.bar(x - width/2, norm_acc_b, width, color=BASELINE_COLOR, alpha=0.85, label='Baseline')
    bars2 = ax.bar(x + width/2, norm_acc_n, width, color=NEFTUNE_COLOR, alpha=0.85, label='NEFTune')
    
    # Random baseline
    ax.axhline(y=25, color=GOLD_COLOR, linestyle='--', linewidth=2.5, alpha=0.9, label='Random (25%)')
    
    ax.set_xlabel('Training Step', fontsize=12, fontweight='bold', color='#333333')
    ax.set_ylabel('Correct / Max Valid (%)', fontsize=12, fontweight='bold', color='#333333')
    ax.set_title('Normalized Accuracy: Correct / Best Valid Count', fontsize=14, fontweight='bold', 
                 pad=10, color=BASELINE_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend(loc='lower right', frameon=True, fancybox=False, 
              edgecolor='#666666', facecolor='white')
    
    # Add delta labels
    for i, (b, n) in enumerate(zip(norm_acc_b, norm_acc_n)):
        if n > 0 or b > 0:
            delta = n - b
            sign = "+" if delta > 0 else ""
            color = NEFTUNE_COLOR if delta > 0 else BASELINE_COLOR
            max_val = max(b, n)
            ax.text(x[i], max_val + 0.5, f'{sign}{delta:.1f}%', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color=color)
    
    ax.set_ylim(0, max(max(norm_acc_b), max(norm_acc_n)) * 1.15)


def main():
    # Paths
    neftune_dir = "results/monad_neftune_2025-12-19_18-47-19"
    baseline_dir = "../day20/results/monad_2025-12-18_19-36-22"
    
    print("Loading training metrics...")
    df_neftune = load_training_metrics(neftune_dir)
    df_baseline = load_training_metrics(baseline_dir)
    print(f"  Baseline: {len(df_baseline)} steps")
    print(f"  NEFTune: {len(df_neftune)} steps")
    
    print("Loading MMLU evaluations...")
    evals_neftune = load_mmlu_evals(neftune_dir)
    evals_baseline = load_mmlu_evals(baseline_dir)
    print(f"  Baseline: {len(evals_baseline)} checkpoints")
    print(f"  NEFTune: {len(evals_neftune)} checkpoints")
    
    # ==========================================================================
    # PLOT 1: Training Loss
    # ==========================================================================
    fig1, ax1 = plt.subplots(figsize=(12, 5), facecolor='white')
    plot_training_loss(df_baseline, df_neftune, ax1)
    plt.tight_layout()
    
    output1 = os.path.join(neftune_dir, "1_training_loss_comparison.png")
    plt.savefig(output1, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output1}")
    
    # ==========================================================================
    # PLOT 2: MMLU Metrics (2 subplots - valid counts + accuracy)
    # ==========================================================================
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    plot_mmlu_comparison(evals_baseline, evals_neftune, (ax2, ax3))
    
    fig2.suptitle('Baseline vs NEFTune: MMLU Evaluation', fontsize=14, fontweight='bold', 
                   y=1.02, color=BASELINE_COLOR)
    plt.tight_layout()
    
    output2 = os.path.join(neftune_dir, "2_mmlu_comparison.png")
    plt.savefig(output2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output2}")
    
    # ==========================================================================
    # PLOT 3: Normalized Accuracy (standalone)
    # ==========================================================================
    fig3, ax4 = plt.subplots(figsize=(12, 5), facecolor='white')
    plot_normalized_accuracy(evals_baseline, evals_neftune, ax4)
    plt.tight_layout()
    
    output3 = os.path.join(neftune_dir, "3_normalized_accuracy.png")
    plt.savefig(output3, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output3}")
    
    plt.show()
    
    # Print summary table
    print("\n" + "=" * 80)
    print("MMLU Comparison: Baseline vs NEFTune")
    print("=" * 80)
    print(f"{'Step':<8} {'Baseline Valid':<16} {'NEFTune Valid':<16} {'Baseline Acc':<14} {'NEFTune Acc':<14} {'Delta'}")
    print("-" * 80)
    
    all_steps = sorted(set(evals_baseline.keys()) | set(evals_neftune.keys()))
    for step in all_steps:
        b_valid = evals_baseline.get(step, {}).get('strictly_valid', 0)
        n_valid = evals_neftune.get(step, {}).get('strictly_valid', 0)
        b_acc = evals_baseline.get(step, {}).get('accuracy_strict', 0) * 100
        n_acc = evals_neftune.get(step, {}).get('accuracy_strict', 0) * 100
        delta = n_acc - b_acc
        print(f"{step:<8} {b_valid:<16} {n_valid:<16} {b_acc:>6.2f}%       {n_acc:>6.2f}%       {delta:>+6.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
