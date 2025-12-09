"""
Plotter for GEPA vs GRPO comparison on MATH dataset.

Reads eval_summary.json from both runs and creates Christmas-themed plots
comparing the learning curves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import urllib.request
from io import BytesIO
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
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


def get_emoji_image(url: str, zoom: float = 0.15) -> OffsetImage:
    """Download and prepare an emoji image for plotting."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            data = response.read()
            
        img = Image.open(BytesIO(data)).convert("RGBA")
        return OffsetImage(img, zoom=zoom)
    except Exception as e:
        print(f"Warning: Failed to load emoji from {url}: {e}")
        return None


def load_eval_summary(path: Path) -> dict:
    """Load eval summary from JSON file."""
    with path.open("r") as f:
        return json.load(f)


def plot_gepa_vs_grpo(
    gepa_results: dict,
    grpo_results: dict,
    output_path: Path,
    title: str = "GEPA vs GRPO on MATH Dataset",
) -> None:
    """Create and save a Christmas-themed plot comparing GEPA and GRPO."""
    
    # Extract steps and pass@1 values for both methods
    gepa_steps = []
    gepa_pass_at_1 = []
    for step_str in sorted(gepa_results.keys(), key=int):
        gepa_steps.append(int(step_str))
        gepa_pass_at_1.append(gepa_results[step_str]["pass_at_1"])
    
    grpo_steps = []
    grpo_pass_at_1 = []
    for step_str in sorted(grpo_results.keys(), key=int):
        grpo_steps.append(int(step_str))
        grpo_pass_at_1.append(grpo_results[step_str]["pass_at_1"])
    
    # Christmas colors
    christmas_green = "#228B22"
    off_white = "#FFFEF7"
    red_color = "#FF1744"  # GEPA - red
    blue_color = "#1976D2"  # GRPO - blue
    
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor(off_white)
    
    # Determine axis limits
    all_values = gepa_pass_at_1 + grpo_pass_at_1
    y_min = min(all_values)
    y_max = max(all_values)
    y_range = y_max - y_min
    ax.set_ylim(max(0.0, y_min - 0.1 * y_range), y_max + 0.1 * y_range)
    
    all_steps = gepa_steps + grpo_steps
    x_min = min(all_steps)
    x_max = max(all_steps)
    ax.set_xlim(x_min - 20, x_max + 20)
    
    # Plot GRPO line (blue)
    ax.plot(
        grpo_steps,
        grpo_pass_at_1,
        color=blue_color,
        linewidth=3.5,
        label="GRPO (Weight Updates)",
        zorder=3,
        marker="s",
        markersize=8,
    )
    ax.scatter(
        grpo_steps,
        grpo_pass_at_1,
        color=blue_color,
        alpha=0.8,
        s=100,
        zorder=4,
        edgecolors=christmas_green,
        linewidths=2,
        marker="s",
    )
    
    # Plot GEPA line (red)
    ax.plot(
        gepa_steps,
        gepa_pass_at_1,
        color=red_color,
        linewidth=3.5,
        label="GEPA (Prompt Evolution)",
        zorder=3,
        marker="o",
        markersize=8,
    )
    ax.scatter(
        gepa_steps,
        gepa_pass_at_1,
        color=red_color,
        alpha=0.8,
        s=100,
        zorder=4,
        edgecolors=christmas_green,
        linewidths=2,
    )
    
    # Add Christmas emojis
    tree_url = "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f384.png"
    gift_url = "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f381.png"
    
    tree_img = get_emoji_image(tree_url, zoom=0.55)
    gift_img = get_emoji_image(gift_url, zoom=0.55)
    
    # Set title
    ax.set_title(title, fontsize=20, color=christmas_green, pad=20, weight="bold")
    
    # Add images next to title
    if tree_img:
        ab_tree = AnnotationBbox(tree_img, (0.05, 1.06), xycoords='axes fraction', frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab_tree)
        
    if gift_img:
        ab_gift = AnnotationBbox(gift_img, (0.95, 1.06), xycoords='axes fraction', frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab_gift)
    
    ax.set_xlabel("Optimization Step", fontsize=16, color=christmas_green, weight="bold")
    ax.set_ylabel("Pass@1 (%)", fontsize=16, color=christmas_green, weight="bold")
    
    # Style the grid and axes
    ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.3, color=christmas_green)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_color(christmas_green)
        ax.spines[spine].set_linewidth(3)
    
    ax.tick_params(colors=christmas_green, which="both")
    
    # Legend with Christmas styling
    legend = ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        edgecolor=christmas_green,
        facecolor=off_white,
    )
    legend.get_frame().set_linewidth(2)
    for text in legend.get_texts():
        text.set_color(christmas_green)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot GEPA vs GRPO comparison with Christmas theme"
    )
    parser.add_argument(
        "--gepa-results",
        type=Path,
        default=Path("gepa_qwen7b_run/eval_summary.json"),
        help="Path to GEPA eval_summary.json",
    )
    parser.add_argument(
        "--grpo-results",
        type=Path,
        default=Path("grpo_qwen7b_run/eval_summary.json"),
        help="Path to GRPO eval_summary.json",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("figs/gepa_vs_grpo.png"),
        help="Path to save the output plot",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="GEPA vs GRPO on MATH Dataset",
        help="Title for the plot",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to load results and create plots."""
    args = parse_args()
    
    print(f"Loading GEPA results from {args.gepa_results}...")
    gepa_results = load_eval_summary(args.gepa_results)
    
    print(f"Loading GRPO results from {args.grpo_results}...")
    grpo_results = load_eval_summary(args.grpo_results)
    
    print("Creating comparison plot...")
    plot_gepa_vs_grpo(
        gepa_results,
        grpo_results,
        args.output_path,
        title=args.title,
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
