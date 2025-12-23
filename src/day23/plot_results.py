"""
Plot Looped Transformer Results - Baseline vs Looped (5x) Comparison
=====================================================================
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

# Color scheme - festive but professional
BASELINE_COLOR = "#2E86AB"  # Blue
LOOPED_COLOR = "#A23B72"    # Magenta/Purple
GOLD_COLOR = "#DAA520"      # Goldenrod (random baseline)
GRID_COLOR = "#E8E8E8"      # Light gray grid
BG_COLOR = "#FAFAFA"        # Light background


def load_training_metrics(results_dir: str) -> pd.DataFrame:
    """Load training metrics from CSV."""
    csv_path = os.path.join(results_dir, "logs", "training_metrics.csv")
    return pd.read_csv(csv_path)


def load_mmlu_evals(results_dir: str) -> dict:
    """Load all MMLU evaluation results."""
    mmlu_dir = os.path.join(results_dir, "mmlu_eval")
    if not os.path.exists(mmlu_dir):
        print(f"Warning: {mmlu_dir} not found")
        return {}
    
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
    """Apply styling to an axis."""
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.4, color=GRID_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_color('#666666')
    ax.spines['bottom'].set_linewidth(1.2)


def plot_training_loss(df_baseline: pd.DataFrame, df_looped: pd.DataFrame, ax: plt.Axes):
    """Plot training loss comparison."""
    style_axis(ax)
    
    # Plot lines
    ax.plot(df_baseline['step'], df_baseline['loss'], color=BASELINE_COLOR, 
            linewidth=2, alpha=0.9, label='Baseline (8 layers)')
    ax.plot(df_looped['step'], df_looped['loss'], color=LOOPED_COLOR, 
            linewidth=2, alpha=0.9, label='Looped 5× (last layer)')
    
    # Find minimums
    min_idx_b = df_baseline['loss'].idxmin()
    min_step_b = df_baseline.loc[min_idx_b, 'step']
    min_loss_b = df_baseline.loc[min_idx_b, 'loss']
    
    min_idx_l = df_looped['loss'].idxmin()
    min_step_l = df_looped.loc[min_idx_l, 'step']
    min_loss_l = df_looped.loc[min_idx_l, 'loss']
    
    # Mark minimums
    ax.scatter([min_step_b], [min_loss_b], color=BASELINE_COLOR, s=100, zorder=5, 
               marker='o', edgecolor='white', linewidth=2)
    ax.scatter([min_step_l], [min_loss_l], color=LOOPED_COLOR, s=100, zorder=5,
               marker='o', edgecolor='white', linewidth=2)
    
    # Annotate minimums
    ax.annotate(f'Min: {min_loss_b:.3f}', xy=(min_step_b, min_loss_b),
                xytext=(min_step_b + 150, min_loss_b + 2),
                fontsize=10, color=BASELINE_COLOR, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=BASELINE_COLOR, lw=1.5))
    
    ax.annotate(f'Min: {min_loss_l:.3f}', xy=(min_step_l, min_loss_l),
                xytext=(min_step_l - 300, min_loss_l + 3),
                fontsize=10, color=LOOPED_COLOR, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=LOOPED_COLOR, lw=1.5))
    
    # Winner annotation
    if min_loss_l < min_loss_b:
        diff = min_loss_b - min_loss_l
        winner_text = f"Looped lower by {diff:.3f}"
        winner_color = LOOPED_COLOR
    else:
        diff = min_loss_l - min_loss_b
        winner_text = f"Baseline lower by {diff:.3f}"
        winner_color = BASELINE_COLOR
    
    ax.text(0.98, 0.95, winner_text, transform=ax.transAxes, fontsize=11,
            ha='right', va='top', fontweight='bold', color=winner_color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=winner_color, linewidth=1.5))
    
    ax.set_xlabel('Training Step', fontsize=12, fontweight='bold', color='#333333')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold', color='#333333')
    ax.set_title('Training Loss: Baseline vs Looped Transformer', fontsize=14, fontweight='bold', 
                 pad=10, color='#333333')
    
    ax.legend(loc='upper right', frameon=True, fancybox=False, 
              edgecolor='#666666', facecolor='white')
    
    max_step = max(df_baseline['step'].max(), df_looped['step'].max())
    ax.set_xlim(0, max_step * 1.02)


def plot_mmlu_comparison(evals_baseline: dict, evals_looped: dict, axes: tuple):
    """Plot MMLU valid counts and accuracy comparison."""
    ax1, ax2 = axes
    
    # Get all steps
    all_steps = sorted(set(evals_baseline.keys()) | set(evals_looped.keys()))
    
    labels = ["Init" if s == 0 else ("Final" if s == 9999 else f"{s}") for s in all_steps]
    x = np.arange(len(all_steps))
    width = 0.35
    
    for ax in axes:
        style_axis(ax)
    
    # =========================================================================
    # Plot 1: Strictly Valid Counts
    # =========================================================================
    valid_b = [evals_baseline.get(s, {}).get('strictly_valid', 0) for s in all_steps]
    valid_l = [evals_looped.get(s, {}).get('strictly_valid', 0) for s in all_steps]
    
    ax1.bar(x - width/2, valid_b, width, color=BASELINE_COLOR, alpha=0.85, label='Baseline')
    ax1.bar(x + width/2, valid_l, width, color=LOOPED_COLOR, alpha=0.85, label='Looped 5×')
    
    ax1.set_ylabel('Strictly Valid Responses', fontsize=11, fontweight='bold', color='#333333')
    ax1.set_title('Format Compliance (Valid Responses)', fontsize=13, fontweight='bold', 
                  pad=10, color='#333333')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha='right')
    ax1.legend(loc='lower right', frameon=True, fancybox=False, 
               edgecolor='#666666', facecolor='white')
    
    # Add delta labels
    for i, (b, l) in enumerate(zip(valid_b, valid_l)):
        if l > 0 or b > 0:
            delta = l - b
            sign = "+" if delta > 0 else ""
            color = LOOPED_COLOR if delta > 0 else BASELINE_COLOR
            max_val = max(b, l)
            ax1.text(x[i], max_val + 200, f'{sign}{delta}', ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color=color)
    
    ax1.set_ylim(0, max(max(valid_b, default=0), max(valid_l, default=0)) * 1.15)
    
    # =========================================================================
    # Plot 2: Accuracy
    # =========================================================================
    acc_b = [evals_baseline.get(s, {}).get('accuracy_strict', 0) * 100 for s in all_steps]
    acc_l = [evals_looped.get(s, {}).get('accuracy_strict', 0) * 100 for s in all_steps]
    
    ax2.bar(x - width/2, acc_b, width, color=BASELINE_COLOR, alpha=0.85, label='Baseline')
    ax2.bar(x + width/2, acc_l, width, color=LOOPED_COLOR, alpha=0.85, label='Looped 5×')
    
    # Random baseline
    ax2.axhline(y=25, color=GOLD_COLOR, linestyle='--', linewidth=2.5, alpha=0.9, label='Random (25%)')
    
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold', color='#333333')
    ax2.set_title('MMLU Accuracy (Strict)', fontsize=13, fontweight='bold', 
                  pad=10, color='#333333')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha='right')
    ax2.legend(loc='lower right', frameon=True, fancybox=False, 
               edgecolor='#666666', facecolor='white')
    
    # Add delta labels
    for i, (b, l) in enumerate(zip(acc_b, acc_l)):
        if l > 0 or b > 0:
            delta = l - b
            sign = "+" if delta > 0 else ""
            color = LOOPED_COLOR if delta > 0 else BASELINE_COLOR
            max_val = max(b, l)
            ax2.text(x[i], max_val + 0.8, f'{sign}{delta:.1f}%', ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color=color)
    
    ax2.set_ylim(0, max(max(acc_b, default=0), max(acc_l, default=0)) * 1.18)


def plot_normalized_accuracy(evals_baseline: dict, evals_looped: dict, ax: plt.Axes):
    """Plot normalized accuracy: correct / max valid from either method."""
    style_axis(ax)
    
    all_steps = sorted(set(evals_baseline.keys()) | set(evals_looped.keys()))
    
    labels = ["Init" if s == 0 else ("Final" if s == 9999 else f"{s}") for s in all_steps]
    x = np.arange(len(all_steps))
    width = 0.35
    
    # Get valid and correct counts
    valid_b = [evals_baseline.get(s, {}).get('strictly_valid', 0) for s in all_steps]
    valid_l = [evals_looped.get(s, {}).get('strictly_valid', 0) for s in all_steps]
    
    correct_b = [int(evals_baseline.get(s, {}).get('accuracy_strict', 0) * 
                     evals_baseline.get(s, {}).get('strictly_valid', 0)) for s in all_steps]
    correct_l = [int(evals_looped.get(s, {}).get('accuracy_strict', 0) * 
                     evals_looped.get(s, {}).get('strictly_valid', 0)) for s in all_steps]
    
    # Normalize by max valid from either method
    norm_acc_b = []
    norm_acc_l = []
    for i, s in enumerate(all_steps):
        max_valid = max(valid_b[i], valid_l[i])
        if max_valid > 0:
            norm_acc_b.append(correct_b[i] / max_valid * 100)
            norm_acc_l.append(correct_l[i] / max_valid * 100)
        else:
            norm_acc_b.append(0)
            norm_acc_l.append(0)
    
    ax.bar(x - width/2, norm_acc_b, width, color=BASELINE_COLOR, alpha=0.85, label='Baseline')
    ax.bar(x + width/2, norm_acc_l, width, color=LOOPED_COLOR, alpha=0.85, label='Looped 5×')
    
    # Random baseline
    ax.axhline(y=25, color=GOLD_COLOR, linestyle='--', linewidth=2.5, alpha=0.9, label='Random (25%)')
    
    ax.set_xlabel('Training Step', fontsize=12, fontweight='bold', color='#333333')
    ax.set_ylabel('Correct / Max Valid (%)', fontsize=12, fontweight='bold', color='#333333')
    ax.set_title('Normalized Accuracy: Correct / Best Valid Count', fontsize=14, fontweight='bold', 
                 pad=10, color='#333333')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend(loc='lower right', frameon=True, fancybox=False, 
              edgecolor='#666666', facecolor='white')
    
    # Add delta labels
    for i, (b, l) in enumerate(zip(norm_acc_b, norm_acc_l)):
        if l > 0 or b > 0:
            delta = l - b
            sign = "+" if delta > 0 else ""
            color = LOOPED_COLOR if delta > 0 else BASELINE_COLOR
            max_val = max(b, l)
            ax.text(x[i], max_val + 0.5, f'{sign}{delta:.1f}%', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color=color)
    
    ax.set_ylim(0, max(max(norm_acc_b, default=0), max(norm_acc_l, default=0)) * 1.15)


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    baseline_dir = os.path.join(script_dir, "results_baseline/transformer_baseline_2025-12-22_22-56-17")
    looped_dir = os.path.join(script_dir, "results_looped/transformer_loop5x_2025-12-23_01-39-47")
    
    # Create figs directory
    figs_dir = os.path.join(script_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    
    print("Loading training metrics...")
    df_baseline = load_training_metrics(baseline_dir)
    df_looped = load_training_metrics(looped_dir)
    print(f"  Baseline: {len(df_baseline)} steps")
    print(f"  Looped: {len(df_looped)} steps")
    
    print("Loading MMLU evaluations...")
    evals_baseline = load_mmlu_evals(baseline_dir)
    evals_looped = load_mmlu_evals(looped_dir)
    print(f"  Baseline: {len(evals_baseline)} checkpoints")
    print(f"  Looped: {len(evals_looped)} checkpoints")
    
    # ==========================================================================
    # PLOT 1: Training Loss
    # ==========================================================================
    fig1, ax1 = plt.subplots(figsize=(12, 5), facecolor='white')
    plot_training_loss(df_baseline, df_looped, ax1)
    plt.tight_layout()
    
    output1 = os.path.join(figs_dir, "1_training_loss_comparison.png")
    plt.savefig(output1, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output1}")
    plt.close()
    
    # ==========================================================================
    # PLOT 2: MMLU Metrics (if evals exist)
    # ==========================================================================
    if evals_baseline or evals_looped:
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
        plot_mmlu_comparison(evals_baseline, evals_looped, (ax2, ax3))
        
        fig2.suptitle('Baseline vs Looped: MMLU Evaluation', fontsize=14, fontweight='bold', 
                       y=1.02, color='#333333')
        plt.tight_layout()
        
        output2 = os.path.join(figs_dir, "2_mmlu_comparison.png")
        plt.savefig(output2, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output2}")
        plt.close()
        
        # ==========================================================================
        # PLOT 3: Normalized Accuracy
        # ==========================================================================
        fig3, ax4 = plt.subplots(figsize=(12, 5), facecolor='white')
        plot_normalized_accuracy(evals_baseline, evals_looped, ax4)
        plt.tight_layout()
        
        output3 = os.path.join(figs_dir, "3_normalized_accuracy.png")
        plt.savefig(output3, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output3}")
        plt.close()
        
        # Print summary table
        print("\n" + "=" * 90)
        print("MMLU Comparison: Baseline vs Looped (5x)")
        print("=" * 90)
        print(f"{'Step':<8} {'Baseline Valid':<16} {'Looped Valid':<16} {'Baseline Acc':<14} {'Looped Acc':<14} {'Delta'}")
        print("-" * 90)
        
        all_steps = sorted(set(evals_baseline.keys()) | set(evals_looped.keys()))
        for step in all_steps:
            b_valid = evals_baseline.get(step, {}).get('strictly_valid', 0)
            l_valid = evals_looped.get(step, {}).get('strictly_valid', 0)
            b_acc = evals_baseline.get(step, {}).get('accuracy_strict', 0) * 100
            l_acc = evals_looped.get(step, {}).get('accuracy_strict', 0) * 100
            delta = l_acc - b_acc
            step_str = "Init" if step == 0 else ("Final" if step == 9999 else str(step))
            print(f"{step_str:<8} {b_valid:<16} {l_valid:<16} {b_acc:>6.2f}%       {l_acc:>6.2f}%       {delta:>+6.2f}%")
        print("=" * 90)
    else:
        print("\nNo MMLU evaluations found. Run submit_eval_jobs.sh first.")


if __name__ == "__main__":
    main()

