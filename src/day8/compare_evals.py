#!/usr/bin/env python3
"""
Compare personality profiles from multiple eval runs.
Finds all eval_* directories and plots them together.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from data import OCEAN_FULL_NAMES


def load_eval_results(eval_dir: Path) -> dict:
    """Load eval_results.json from a directory."""
    results_path = eval_dir / "eval_results.json"
    if not results_path.exists():
        return None
    with open(results_path) as f:
        return json.load(f)


def create_individual_spider(
    results: dict,
    label: str,
    title: str = None,
    output_path: Path = None,
    show: bool = False,
    color: str = "#2E86AB",
):
    """Create a spider plot for a single personality profile."""
    dimensions = ["O", "C", "E", "A", "N"]  # OCEAN order
    dim_labels = [OCEAN_FULL_NAMES[d] for d in dimensions]
    
    N = len(dimensions)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete circle
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    scores = [results["personality"]["ocean"][d] for d in dimensions]
    scores_plot = scores + scores[:1]
    
    ax.plot(angles, scores_plot, 'o-', linewidth=2.5, color=color, markersize=8)
    ax.fill(angles, scores_plot, alpha=0.25, color=color)
    
    # Style
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, size=12, fontweight='bold')
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], size=9)
    ax.set_title(title or label, size=14, y=1.08, fontweight='bold')
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved individual plot to {output_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)
    return fig


def create_comparison_spider(
    results_list: list[dict],
    labels: list[str],
    title: str = "Personality Comparison",
    output_path: Path = None,
    show: bool = False,
):
    """Create a spider plot comparing multiple personality profiles."""
    dimensions = ["O", "C", "E", "A", "N"]  # OCEAN order
    dim_labels = [OCEAN_FULL_NAMES[d] for d in dimensions]
    
    N = len(dimensions)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete circle
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    # Color palette - use distinct colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    
    for i, (results, label) in enumerate(zip(results_list, labels)):
        scores = [results["personality"]["ocean"][d] for d in dimensions]
        scores_plot = scores + scores[:1]
        ax.plot(angles, scores_plot, 'o-', linewidth=2, label=label, color=colors[i], markersize=6)
        ax.fill(angles, scores_plot, alpha=0.1, color=colors[i])
    
    # Style
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, size=12, fontweight='bold')
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], size=9)
    ax.set_title(title, size=16, y=1.08, fontweight='bold')
    
    # Legend outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.0), fontsize=10)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {output_path}")
    
    if show:
        plt.show()
    
    return fig


def main():
    # Find all eval_* directories
    base_dir = Path(__file__).parent
    eval_dirs = sorted(base_dir.glob("eval_*"))
    
    if not eval_dirs:
        print("No eval_* directories found!")
        return
    
    print(f"Found {len(eval_dirs)} eval directories:")
    
    results_list = []
    labels = []
    
    for eval_dir in eval_dirs:
        results = load_eval_results(eval_dir)
        if results is None:
            print(f"  ✗ {eval_dir.name} - no results found")
            continue
        
        model_name = results.get("model_name", eval_dir.name)
        # Clean up model name for legend
        short_name = model_name.split("/")[-1] if "/" in model_name else model_name
        
        ocean = results["personality"]["ocean"]
        print(f"  ✓ {eval_dir.name}: {short_name}")
        print(f"      N={ocean['N']:.2f} E={ocean['E']:.2f} O={ocean['O']:.2f} A={ocean['A']:.2f} C={ocean['C']:.2f}")
        
        results_list.append(results)
        labels.append(short_name)
    
    if not results_list:
        print("\nNo valid results to compare!")
        return
    
    # Create individual plots for each model
    personas_dir = base_dir / "personas_output"
    personas_dir.mkdir(exist_ok=True)
    
    print(f"\nCreating individual plots in {personas_dir}/...")
    
    # Color palette for individual plots
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#95190C", "#610345", "#044B7F"]
    
    for i, (results, label) in enumerate(zip(results_list, labels)):
        # Clean up filename
        safe_name = label.replace("/", "_").replace(" ", "_").replace(".", "_")
        output_path = personas_dir / f"{safe_name}.png"
        color = colors[i % len(colors)]
        
        create_individual_spider(
            results,
            label,
            title=f"Big Five Personality: {label}",
            output_path=output_path,
            color=color,
        )
    
    # Create comparison plot
    print(f"\nCreating comparison plot for {len(results_list)} models...")
    
    output_path = base_dir / "personality_comparison.png"
    create_comparison_spider(
        results_list,
        labels,
        title="Big Five Personality: Model Comparison",
        output_path=output_path,
        show=False,
    )
    
    # Also save a summary JSON
    summary = {
        "models": [
            {
                "name": label,
                "ocean": results["personality"]["ocean"],
                "format_failure_rate": results.get("format_failure_rate", 0),
            }
            for label, results in zip(labels, results_list)
        ]
    }
    
    summary_path = base_dir / "personality_comparison.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

