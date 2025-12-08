"""
Visualization for personality results - spider/radar plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from data import OCEAN_FULL_NAMES


def create_spider_plot(
    results: dict,
    target: Optional[dict] = None,
    title: str = "Big Five Personality Profile",
    output_path: Optional[str | Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Create a spider/radar plot of personality results.
    
    Args:
        results: Dict with 'ocean' key containing {N: score, E: score, ...}
        target: Optional target personality dict for overlay
        title: Plot title
        output_path: Path to save figure (if None, doesn't save)
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure object
    """
    # Extract OCEAN scores in consistent order
    dimensions = ["O", "C", "E", "A", "N"]  # OCEAN order
    labels = [OCEAN_FULL_NAMES[d] for d in dimensions]
    
    # Get scores (scale 1-5)
    scores = [results["ocean"][d] for d in dimensions]
    
    # Number of variables
    N = len(dimensions)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Close the polygon
    scores_plot = scores + scores[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Set the color scheme
    result_color = '#2ecc71'  # Green
    target_color = '#e74c3c'  # Red
    
    # Plot results
    ax.plot(angles, scores_plot, 'o-', linewidth=2, label='Results', color=result_color)
    ax.fill(angles, scores_plot, alpha=0.25, color=result_color)
    
    # Plot target if provided
    if target is not None:
        target_scores = [target.get(d, 3.0) for d in dimensions]
        target_scores_plot = target_scores + target_scores[:1]
        ax.plot(angles, target_scores_plot, 'o--', linewidth=2, label='Target', color=target_color)
        ax.fill(angles, target_scores_plot, alpha=0.1, color=target_color)
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12)
    
    # Set y-axis limits (1-5 scale)
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], size=8)
    
    # Add title and legend
    ax.set_title(title, size=14, y=1.08)
    if target is not None:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Style
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    
    return fig


def create_progress_plot(
    current: dict,
    target: dict,
    baseline: dict,
    title: str = "Training Progress",
    output_path: Optional[str | Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Create a spider plot showing baseline → current → target progression.
    
    Args:
        current: Current model personality (ocean dict)
        target: Target personality (dict with N, E, O, A, C)
        baseline: Baseline/step 0 personality (ocean dict)
        title: Plot title
        output_path: Path to save figure
        show: Whether to display
    
    Returns:
        matplotlib Figure object
    """
    dimensions = ["O", "C", "E", "A", "N"]
    dim_labels = [OCEAN_FULL_NAMES[d] for d in dimensions]
    
    N = len(dimensions)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Colors: baseline (gray), current (green), target (red dashed)
    baseline_color = '#95a5a6'  # Gray
    current_color = '#2ecc71'   # Green  
    target_color = '#e74c3c'    # Red
    
    # Plot baseline
    baseline_scores = [baseline["ocean"][d] for d in dimensions]
    baseline_plot = baseline_scores + baseline_scores[:1]
    ax.plot(angles, baseline_plot, 'o-', linewidth=2, label='Baseline (Step 0)', color=baseline_color)
    ax.fill(angles, baseline_plot, alpha=0.1, color=baseline_color)
    
    # Plot current
    current_scores = [current["ocean"][d] for d in dimensions]
    current_plot = current_scores + current_scores[:1]
    ax.plot(angles, current_plot, 'o-', linewidth=2.5, label='Current', color=current_color)
    ax.fill(angles, current_plot, alpha=0.25, color=current_color)
    
    # Plot target
    target_scores = [target.get(d, 3.0) for d in dimensions]
    target_plot = target_scores + target_scores[:1]
    ax.plot(angles, target_plot, 's--', linewidth=2, label='Target', color=target_color, markersize=8)
    ax.fill(angles, target_plot, alpha=0.1, color=target_color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, size=12)
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], size=8)
    ax.set_title(title, size=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)
    return fig


def create_comparison_plot(
    results_list: list[dict],
    labels: list[str],
    title: str = "Personality Comparison",
    output_path: Optional[str | Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Create a spider plot comparing multiple personality profiles.
    
    Args:
        results_list: List of result dicts, each with 'ocean' key
        labels: Labels for each profile
        title: Plot title
        output_path: Path to save figure
        show: Whether to display
    
    Returns:
        matplotlib Figure object
    """
    dimensions = ["O", "C", "E", "A", "N"]
    dim_labels = [OCEAN_FULL_NAMES[d] for d in dimensions]
    
    N = len(dimensions)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_list)))
    
    for i, (results, label) in enumerate(zip(results_list, labels)):
        scores = [results["ocean"][d] for d in dimensions]
        scores_plot = scores + scores[:1]
        ax.plot(angles, scores_plot, 'o-', linewidth=2, label=label, color=colors[i])
        ax.fill(angles, scores_plot, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, size=12)
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], size=8)
    ax.set_title(title, size=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    if show:
        plt.show()
    
    return fig


# Quick test
if __name__ == "__main__":
    # Example results
    example_results = {
        "ocean": {
            "N": 2.5,
            "E": 4.2,
            "O": 3.8,
            "A": 4.5,
            "C": 3.2,
        }
    }
    
    example_target = {
        "N": 3.0,
        "E": 2.0,
        "O": 4.0,
        "A": 2.0,
        "C": 4.0,
    }
    
    # Create plot with target overlay
    fig = create_spider_plot(
        example_results,
        target=example_target,
        title="Example: Results vs Target",
        output_path=Path(__file__).parent / "example_spider.png",
        show=False,
    )
    plt.close(fig)
    
    print("Created example_spider.png")

