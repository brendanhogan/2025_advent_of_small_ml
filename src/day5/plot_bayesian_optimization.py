"""
Create Bayesian optimization visualization plots showing GP evolution.

For each layer group, shows:
- X-axis: weight (0.0 to 1.0)
- Y-axis: performance (pass@1)
- GP mean and uncertainty bands evolving as observations are added
- Observations as red dots
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm

# Set matplotlib style
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

GROUP_NAMES = ["early", "mid_early", "mid_late", "late"]
GROUP_DISPLAY_NAMES = ["Early", "Mid-Early", "Mid-Late", "Late"]


def fit_gp_for_dimension(X_all, y_all, dimension_idx, x_pred):
    """
    Fit a GP to observations, marginalizing over other dimensions.
    
    For a given dimension, we fit a 1D GP by:
    1. Taking all observations
    2. Using the target dimension's weight as input
    3. Using the performance as output
    
    Args:
        X_all: All weight configurations (n_obs, 4)
        y_all: All performance scores (n_obs,)
        dimension_idx: Which dimension to plot (0-3)
        x_pred: Points to predict at (n_pred,)
    
    Returns:
        mu: Mean predictions (n_pred,)
        sigma: Std predictions (n_pred,)
    """
    if len(X_all) == 0:
        # No observations yet - return high uncertainty
        mu = np.zeros_like(x_pred)
        sigma = np.ones_like(x_pred) * 10.0  # High uncertainty
        return mu, sigma
    
    # Extract 1D inputs and outputs
    X_1d = X_all[:, dimension_idx].reshape(-1, 1)
    y_1d = y_all
    
    # Fit GP
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)
    gp.fit(X_1d, y_1d)
    
    # Predict
    mu, sigma = gp.predict(x_pred.reshape(-1, 1), return_std=True)
    
    return mu, sigma


def create_bo_plot_frame(
    history: list,
    iteration: int,
    output_path: Path,
    baseline_score: float = None,
    finetuned_score: float = None
):
    """
    Create a single frame showing GP evolution for all 4 dimensions.
    
    Args:
        history: List of observations up to current iteration
        iteration: Current iteration number
        output_path: Where to save the plot
        baseline_score: Baseline performance (for reference line)
        finetuned_score: Finetuned performance (for reference line)
    """
    if len(history) == 0:
        return
    
    # Extract data
    X_all = np.array([h['weights'] for h in history])  # (n_obs, 4)
    y_all = np.array([h['score'] for h in history])    # (n_obs,)
    
    # Create prediction points for each dimension
    x_pred = np.linspace(0.0, 1.0, 100)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Bayesian Optimization in Action (Round {iteration})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Plot each dimension
    for dim_idx, (group_name, display_name) in enumerate(zip(GROUP_NAMES, GROUP_DISPLAY_NAMES)):
        ax = axes[dim_idx // 2, dim_idx % 2]
        
        # Fit GP for this dimension
        mu, sigma = fit_gp_for_dimension(X_all, y_all, dim_idx, x_pred)
        
        # Plot GP mean
        ax.plot(x_pred, mu, '--', color='#228B22', linewidth=2, label='μ_GP(x)', zorder=2)
        
        # Plot uncertainty bands (±2σ)
        ax.fill_between(x_pred, mu - 2*sigma, mu + 2*sigma, 
                       alpha=0.2, color='#228B22', label='Uncertainty (±2σ)', zorder=1)
        ax.fill_between(x_pred, mu - sigma, mu + sigma, 
                       alpha=0.3, color='#228B22', zorder=1)
        
        # Plot observations for this dimension
        x_obs = X_all[:, dim_idx]
        y_obs = y_all
        ax.scatter(x_obs, y_obs, color='#FF1744', s=50, alpha=0.8, 
                  edgecolors='black', linewidths=1, label='Observations', zorder=3)
        
        # Add reference lines
        if baseline_score is not None:
            ax.axhline(y=baseline_score, color='blue', linestyle=':', 
                     linewidth=1.5, alpha=0.7, label=f'Baseline ({baseline_score:.1f}%)', zorder=0)
        if finetuned_score is not None:
            ax.axhline(y=finetuned_score, color='#B8860B', linestyle=':', 
                     linewidth=1.5, alpha=0.7, label=f'Finetuned ({finetuned_score:.1f}%)', zorder=0)
        
        # Styling
        ax.set_xlabel(f'{display_name} Weight', fontweight='bold')
        ax.set_ylabel('Pass@1 (%)', fontweight='bold')
        ax.set_title(f'{display_name} Layer Group', fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=8)
        
        # Style axes (Christmas theme)
        for spine in ax.spines.values():
            spine.set_color('#228B22')
            spine.set_linewidth(2)
        ax.tick_params(colors='#228B22')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved frame {iteration:03d} to {output_path}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Create Bayesian optimization visualization plots")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("bayesian_opt_results/final_results.json"),
        help="Path to final_results.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bayesian_opt_results/gp_plots"),
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
    
    print("\nCreating plot frames...")
    for i in range(len(history) + 1):  # +1 to include initial state (no observations)
        # Get history up to this iteration
        history_up_to_i = history[:i]
        
        # Create plot
        output_path = args.output_dir / f"gp_evolution_iter_{i:03d}.png"
        create_bo_plot_frame(
            history_up_to_i,
            i,
            output_path,
            baseline_score,
            finetuned_score
        )
    
    print(f"\nAll frames saved to {args.output_dir}")
    print(f"To create GIF, run:")
    print(f"  python create_gif.py --frames-dir {args.output_dir} --pattern 'gp_evolution_iter_*.png' --output-path bayesian_opt_results/gp_optimization.gif")


if __name__ == "__main__":
    main()

