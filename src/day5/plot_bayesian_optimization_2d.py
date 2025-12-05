"""
Create 2D pairwise plots for 4D Bayesian optimization visualization.

Shows all 6 pairwise combinations of the 4 dimensions with:
- Scatter points colored by performance
- GP mean contours
- Uncertainty visualization
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from matplotlib.colors import LinearSegmentedColormap

# Set matplotlib style
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

GROUP_NAMES = ["early", "mid_early", "mid_late", "late"]
GROUP_DISPLAY_NAMES = ["Early", "Mid-Early", "Mid-Late", "Late"]


def fit_2d_gp(X_all, y_all, dim1_idx, dim2_idx, x1_grid, x2_grid):
    """
    Fit a 2D GP to observations for two dimensions, marginalizing over the other two.
    
    Args:
        X_all: All weight configurations (n_obs, 4)
        y_all: All performance scores (n_obs,)
        dim1_idx: First dimension index (0-3)
        dim2_idx: Second dimension index (0-3)
        x1_grid: Grid points for dim1 (n1,)
        x2_grid: Grid points for dim2 (n2,)
    
    Returns:
        mu_grid: Mean predictions (n2, n1) - note: swapped for imshow
        sigma_grid: Std predictions (n2, n1)
    """
    if len(X_all) < 2:
        # Not enough points - return high uncertainty
        mu_grid = np.zeros((len(x2_grid), len(x1_grid)))
        sigma_grid = np.ones((len(x2_grid), len(x1_grid))) * 10.0
        return mu_grid, sigma_grid
    
    # Extract 2D inputs
    X_2d = X_all[:, [dim1_idx, dim2_idx]]
    y_2d = y_all
    
    # Fit GP
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)
    gp.fit(X_2d, y_2d)
    
    # Create grid
    X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
    X_grid = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
    
    # Predict
    mu, sigma = gp.predict(X_grid, return_std=True)
    mu_grid = mu.reshape(len(x2_grid), len(x1_grid))
    sigma_grid = sigma.reshape(len(x2_grid), len(x1_grid))
    
    return mu_grid, sigma_grid


def create_2d_bo_plot_frame(
    history: list,
    iteration: int,
    output_path: Path,
    baseline_score: float = None,
    finetuned_score: float = None
):
    """
    Create a 2x3 grid showing all pairwise combinations of the 4 dimensions.
    """
    if len(history) == 0:
        return
    
    # Extract data
    X_all = np.array([h['weights'] for h in history])  # (n_obs, 4)
    y_all = np.array([h['score'] for h in history])    # (n_obs,)
    
    # Create prediction grid
    x_grid = np.linspace(0.0, 1.0, 50)
    
    # Create figure with 6 subplots (all pairwise combinations)
    # Make figure wider to accommodate colorbar on the right
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(f'Bayesian Optimization in Action (Round {iteration})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # All pairwise combinations
    pairs = [
        (0, 1, "Early vs Mid-Early"),
        (0, 2, "Early vs Mid-Late"),
        (0, 3, "Early vs Late"),
        (1, 2, "Mid-Early vs Mid-Late"),
        (1, 3, "Mid-Early vs Late"),
        (2, 3, "Mid-Late vs Late"),
    ]
    
    # Get performance range for colorbar
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
    
    for plot_idx, (dim1_idx, dim2_idx, title) in enumerate(pairs):
        ax = axes[plot_idx // 3, plot_idx % 3]
        
        # Fit 2D GP
        mu_grid, sigma_grid = fit_2d_gp(X_all, y_all, dim1_idx, dim2_idx, x_grid, x_grid)
        
        # Plot GP mean as contour
        contour = ax.contour(x_grid, x_grid, mu_grid, levels=15, colors='#228B22', 
                            linewidths=1, alpha=0.6, zorder=1)
        ax.clabel(contour, inline=True, fontsize=7, fmt='%.1f')
        
        # Plot uncertainty as filled contour (lighter = more uncertain)
        # Use inverse of sigma as alpha
        sigma_norm = 1.0 - np.clip(sigma_grid / sigma_grid.max(), 0, 1)
        im = ax.imshow(sigma_norm, extent=[0, 1, 0, 1], origin='lower', 
                      cmap='Greens', alpha=0.3, aspect='auto', zorder=0)
        
        # Plot observations colored by performance
        x1_obs = X_all[:, dim1_idx]
        x2_obs = X_all[:, dim2_idx]
        scatter = ax.scatter(x1_obs, x2_obs, c=y_all, s=80, alpha=0.8, 
                           edgecolors='black', linewidths=1.5, 
                           cmap='viridis', vmin=y_min, vmax=y_max, zorder=3)
        
        # Styling
        ax.set_xlabel(f'{GROUP_DISPLAY_NAMES[dim1_idx]} Weight', fontweight='bold')
        ax.set_ylabel(f'{GROUP_DISPLAY_NAMES[dim2_idx]} Weight', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
        
        # Style axes (Christmas theme)
        for spine in ax.spines.values():
            spine.set_color('#228B22')
            spine.set_linewidth(2)
        ax.tick_params(colors='#228B22')
    
    # Adjust layout FIRST to make room for colorbar, then add colorbar
    plt.tight_layout(rect=[0, 0.03, 0.90, 0.98])  # Leave right margin for colorbar
    
    # Add colorbar for performance - place it firmly to the right, not overlapping
    # Use the last scatter plot as the mappable, positioned after tight_layout
    cbar = fig.colorbar(scatter, ax=axes, location='right', pad=0.10, aspect=30, shrink=0.8)
    cbar.set_label('Pass@1 (%)', fontweight='bold', rotation=270, labelpad=25)
    cbar.ax.tick_params(colors='#228B22')
    
    # Add reference text
    ref_text = f"Baseline: {baseline_score:.1f}% | Finetuned: {finetuned_score:.1f}%"
    fig.text(0.5, 0.02, ref_text, ha='center', fontsize=10, 
            color='#228B22', fontweight='bold')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved 2D frame {iteration:03d} to {output_path}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Create 2D pairwise BO visualization")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("bayesian_opt_results/final_results.json"),
        help="Path to final_results.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bayesian_opt_results/gp_plots_2d"),
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
    
    print("\nCreating 2D plot frames...")
    for i in range(len(history) + 1):
        history_up_to_i = history[:i]
        output_path = args.output_dir / f"gp_2d_iter_{i:03d}.png"
        create_2d_bo_plot_frame(
            history_up_to_i,
            i,
            output_path,
            baseline_score,
            finetuned_score
        )
    
    print(f"\nAll frames saved to {args.output_dir}")
    print(f"To create GIF, run:")
    print(f"  python create_gif.py --frames-dir {args.output_dir} --pattern 'gp_2d_iter_*.png' --output-path bayesian_opt_results/gp_optimization_2d.gif")


if __name__ == "__main__":
    main()

