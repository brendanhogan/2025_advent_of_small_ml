"""
Create parallel coordinates plot for 4D Bayesian optimization.

Shows all 4 dimensions + performance on parallel axes, making it easy to see
which combinations of weights lead to high performance.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# Set matplotlib style
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

GROUP_NAMES = ["early", "mid_early", "mid_late", "late"]
GROUP_DISPLAY_NAMES = ["Early", "Mid-Early", "Mid-Late", "Late"]


def create_parallel_coords_plot_frame(
    history: list,
    iteration: int,
    output_path: Path,
    baseline_score: float = None,
    finetuned_score: float = None
):
    """
    Create parallel coordinates plot showing all 4 dimensions + performance.
    """
    if len(history) == 0:
        return
    
    # Extract data
    X_all = np.array([h['weights'] for h in history])  # (n_obs, 4)
    y_all = np.array([h['score'] for h in history])    # (n_obs,)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Number of dimensions + performance
    n_dims = 5  # 4 weights + 1 performance
    dim_names = GROUP_DISPLAY_NAMES + ["Performance"]
    
    # Create parallel axes
    x_positions = np.linspace(0, 1, n_dims)
    
    # Normalize data for plotting
    # Weights are already 0-1, but performance needs normalization
    y_min, y_max = y_all.min(), y_all.max()
    if baseline_score is not None:
        y_min = min(y_min, baseline_score)
        y_max = max(y_max, baseline_score)
    if finetuned_score is not None:
        y_min = min(y_min, finetuned_score)
        y_max = max(y_max, finetuned_score)
    y_range = y_max - y_min
    y_min -= 0.05 * y_range
    y_max += 0.05 * y_range
    
    # Normalize performance to 0-1 for plotting
    y_normalized = (y_all - y_min) / (y_max - y_min)
    
    # Combine all data
    data_normalized = np.column_stack([X_all, y_normalized])  # (n_obs, 5)
    
    # Plot lines colored by performance
    cmap = plt.cm.viridis
    for i, (row, perf) in enumerate(zip(data_normalized, y_all)):
        color = cmap((perf - y_min) / (y_max - y_min))
        alpha = 0.6 if i < len(history) - 1 else 1.0  # Highlight most recent
        linewidth = 1.5 if i < len(history) - 1 else 2.5  # Thicker for most recent
        ax.plot(x_positions, row, color=color, alpha=alpha, linewidth=linewidth, zorder=1)
    
    # Plot points at each axis
    for i, (row, perf) in enumerate(zip(data_normalized, y_all)):
        color = cmap((perf - y_min) / (y_max - y_min))
        alpha = 0.7 if i < len(history) - 1 else 1.0
        size = 30 if i < len(history) - 1 else 60
        ax.scatter(x_positions, row, c=[color], s=size, alpha=alpha, 
                  edgecolors='black', linewidths=0.5, zorder=2)
    
    # Set axis labels and ticks
    ax.set_xticks(x_positions)
    ax.set_xticklabels(dim_names, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Normalized Value', fontweight='bold', fontsize=12)
    
    # Add y-axis labels for each dimension
    for i, (pos, name) in enumerate(zip(x_positions, dim_names)):
        if i < 4:  # Weight dimensions
            ax.text(pos, -0.08, '0.0', ha='center', fontsize=8, color='#228B22')
            ax.text(pos, 1.08, '1.0', ha='center', fontsize=8, color='#228B22')
        else:  # Performance dimension
            ax.text(pos, -0.08, f'{y_min:.1f}%', ha='center', fontsize=8, color='#228B22')
            ax.text(pos, 1.08, f'{y_max:.1f}%', ha='center', fontsize=8, color='#228B22')
    
    # Add reference lines for baseline and finetuned
    if baseline_score is not None:
        baseline_norm = (baseline_score - y_min) / (y_max - y_min)
        ax.axhline(y=baseline_norm, xmin=0, xmax=1, color='blue', 
                  linestyle='--', linewidth=2, alpha=0.7, 
                  label=f'Baseline ({baseline_score:.1f}%)', zorder=0)
    if finetuned_score is not None:
        finetuned_norm = (finetuned_score - y_min) / (y_max - y_min)
        ax.axhline(y=finetuned_norm, xmin=0, xmax=1, color='#B8860B', 
                  linestyle='--', linewidth=2, alpha=0.7, 
                  label=f'Finetuned ({finetuned_score:.1f}%)', zorder=0)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=y_min, vmax=y_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Pass@1 (%)', fontweight='bold', rotation=270, labelpad=20)
    cbar.ax.tick_params(colors='#228B22')
    
    # Styling
    ax.set_title(f'Bayesian Optimization in Action (Round {iteration}) - Parallel Coordinates', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    
    # Style axes (Christmas theme)
    for spine in ax.spines.values():
        spine.set_color('#228B22')
        spine.set_linewidth(2)
    ax.tick_params(colors='#228B22')
    
    if baseline_score is not None or finetuned_score is not None:
        ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved parallel coords frame {iteration:03d} to {output_path}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Create parallel coordinates BO visualization")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("bayesian_opt_results/final_results.json"),
        help="Path to final_results.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bayesian_opt_results/gp_plots_parallel"),
        help="Directory to save plot frames"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load results
    print(f"Loading results from {args.results_path}...")
    with open(args.results_path, 'r') as f:
        results = json.load(f)
    
    baseline_score = results.get('baseline_score')
    finetuned_score = results.get('finetuned_score')
    history = results.get('optimization_history', [])
    
    print(f"Found {len(history)} observations")
    print(f"Baseline: {baseline_score:.2f}%")
    print(f"Finetuned: {finetuned_score:.2f}%")
    
    # Create plots for each iteration
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nCreating parallel coordinates plot frames...")
    for i in range(len(history) + 1):
        history_up_to_i = history[:i]
        output_path = args.output_dir / f"gp_parallel_iter_{i:03d}.png"
        create_parallel_coords_plot_frame(
            history_up_to_i,
            i,
            output_path,
            baseline_score,
            finetuned_score
        )
    
    print(f"\nAll frames saved to {args.output_dir}")
    print(f"To create GIF, run:")
    print(f"  python create_gif.py --frames-dir {args.output_dir} --pattern 'gp_parallel_iter_*.png' --output-path bayesian_opt_results/gp_optimization_parallel.gif")


if __name__ == "__main__":
    main()

