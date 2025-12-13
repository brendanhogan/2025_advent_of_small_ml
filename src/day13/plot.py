"""
plot.py - Plot training progress

Run: python plot.py --output output
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set Helvetica font and Christmas theme
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.titleweight": "bold",
})

# Christmas colors
CHRISTMAS_GREEN = "#228B22"
OFF_WHITE = "#FFFEF7"
RED_COLOR = "#FF1744"
GOLD_COLOR = "#FFD700"
PURPLE_COLOR = "#9C27B0"


def main():
    parser = argparse.ArgumentParser(description="Plot cartridge training")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    args = parser.parse_args()
    
    out_dir = Path(args.output)
    
    # Load experiment info
    with open(out_dir / "experiment.json") as f:
        experiment = json.load(f)
    
    # Get baselines (handle both old "random_cartridge" and new "init_cartridge" keys)
    full_context_acc = experiment["results"]["full_context_baseline"]
    init_cart_acc = experiment["results"].get("init_cartridge", experiment["results"].get("random_cartridge"))
    final_cart_acc = experiment["results"]["final_cartridge"]
    
    # Load training log for loss
    with open(out_dir / "training_log.json") as f:
        training_log = json.load(f)
    
    loss_steps = [entry["step"] for entry in training_log]
    losses = [entry["loss"] for entry in training_log]
    
    # Find all step evals
    step_evals = []
    for f in sorted(out_dir.glob("eval_step_*.json")):
        step = int(f.stem.split("_")[-1])
        with open(f) as fp:
            data = json.load(fp)
        step_evals.append((step, data["accuracy"]))
    
    step_evals.sort(key=lambda x: x[0])
    eval_steps = [s[0] for s in step_evals]
    accs = [s[1] for s in step_evals]
    
    # Add initial point at step 0
    eval_steps = [0] + eval_steps
    accs = [init_cart_acc] + accs
    
    # Find best accuracy and its step
    best_idx = max(range(len(accs)), key=lambda i: accs[i])
    best_step = eval_steps[best_idx]
    best_acc = accs[best_idx]
    
    # Create figure - bigger size
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), facecolor="white")
    
    # ===== TOP PLOT: Accuracy =====
    ax1.set_facecolor(OFF_WHITE)
    
    # Full context baseline (green dashed)
    ax1.axhline(y=full_context_acc, color=CHRISTMAS_GREEN, linestyle='--', linewidth=2.5, 
                label=f'Full context ({full_context_acc:.1%})')
    
    # Initial cartridge baseline (red dashed) - no init_mode shown
    ax1.axhline(y=init_cart_acc, color=RED_COLOR, linestyle='--', linewidth=2.5, alpha=0.7,
                label=f'Initial cartridge ({init_cart_acc:.1%})')
    
    # Training progress line
    if eval_steps:
        # Line with markers
        ax1.plot(eval_steps, accs, color=RED_COLOR, linewidth=2.5, label='Cartridge (training)')
        ax1.scatter(eval_steps, accs, color=RED_COLOR, s=50, zorder=3)
        ax1.fill_between(eval_steps, accs, color=RED_COLOR, alpha=0.15)
        
        # Star at BEST performance (gold star)
        ax1.scatter([best_step], [best_acc], color=GOLD_COLOR, s=400, marker='*', 
                    zorder=5, edgecolors=CHRISTMAS_GREEN, linewidths=1.5,
                    label=f'Best ({best_acc:.1%} @ step {best_step})')
    
    ax1.set_ylabel('Accuracy (↑ better)', fontsize=16, color=CHRISTMAS_GREEN, weight="bold")
    ax1.set_title(f'QuALITY Multiple Choice Questions\n{experiment["cartridge"]["num_tokens"]} Cartridge Tokens vs {experiment["data"]["article_tokens"]} Article Tokens', 
                  fontsize=20, color=CHRISTMAS_GREEN, pad=15, weight="bold")
    
    # Style legend
    legend = ax1.legend(loc='lower right', frameon=True, fancybox=True, shadow=True,
                        framealpha=0.95, edgecolor=CHRISTMAS_GREEN, facecolor=OFF_WHITE)
    legend.get_frame().set_linewidth(2)
    for text in legend.get_texts():
        text.set_color(CHRISTMAS_GREEN)
    
    # Style grid and axes
    ax1.grid(True, linestyle="--", linewidth=0.8, alpha=0.3, color=CHRISTMAS_GREEN)
    for spine in ax1.spines.values():
        spine.set_color(CHRISTMAS_GREEN)
        spine.set_linewidth(2)
    ax1.tick_params(colors=CHRISTMAS_GREEN)
    ax1.set_ylim(0, 1)
    
    # ===== BOTTOM PLOT: Loss =====
    ax2.set_facecolor(OFF_WHITE)
    
    # Raw loss (lighter)
    ax2.plot(loss_steps, losses, color=PURPLE_COLOR, linewidth=1, alpha=0.3)
    
    # Smoothed loss
    if len(losses) > 20:
        window = min(50, len(losses) // 10)
        smoothed = []
        for i in range(len(losses)):
            start_idx = max(0, i - window + 1)
            smoothed.append(sum(losses[start_idx:i+1]) / len(losses[start_idx:i+1]))
        ax2.plot(loss_steps, smoothed, color=PURPLE_COLOR, linewidth=3, label=f'Loss (smoothed, window={window})')
        ax2.fill_between(loss_steps, smoothed, color=PURPLE_COLOR, alpha=0.15)
    else:
        ax2.plot(loss_steps, losses, color=PURPLE_COLOR, linewidth=2, label='Loss')
    
    ax2.set_xlabel('Training Step', fontsize=16, color=CHRISTMAS_GREEN, weight="bold")
    ax2.set_ylabel('Loss (↓ better)', fontsize=16, color=CHRISTMAS_GREEN, weight="bold")
    
    # Style legend
    legend2 = ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
                         framealpha=0.95, edgecolor=CHRISTMAS_GREEN, facecolor=OFF_WHITE)
    legend2.get_frame().set_linewidth(2)
    for text in legend2.get_texts():
        text.set_color(CHRISTMAS_GREEN)
    
    # Style grid and axes
    ax2.grid(True, linestyle="--", linewidth=0.8, alpha=0.3, color=CHRISTMAS_GREEN)
    for spine in ax2.spines.values():
        spine.set_color(CHRISTMAS_GREEN)
        spine.set_linewidth(2)
    ax2.tick_params(colors=CHRISTMAS_GREEN)
    
    # Save
    plot_path = out_dir / "training_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, facecolor="white", bbox_inches="tight")
    print(f"Saved: {plot_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
