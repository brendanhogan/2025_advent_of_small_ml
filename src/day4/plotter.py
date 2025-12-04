"""
Plotter for steering vector evaluation results.

Reads steering_eval_results.json and creates Christmas-themed plots showing
the effect of steering vectors on model performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

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
        # Use urllib to download image
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            data = response.read()
            
        img = Image.open(BytesIO(data)).convert("RGBA")
        return OffsetImage(img, zoom=zoom)
    except Exception as e:
        print(f"Warning: Failed to load emoji from {url}: {e}")
        return None


def load_steering_results(results_path: Path) -> dict:
    """Load steering evaluation results from JSON file."""
    with results_path.open("r") as f:
        results = json.load(f)
    return results


def plot_steering_results(
    results: dict,
    output_path: Path,
    title: str = "Effect on Reasoning Performance MATH Dataset",
) -> None:
    """Create and save a Christmas-themed plot showing steering vector effects."""
    
    # Extract data
    base_pass_at_1 = results.get("base_model", {}).get("pass_at_1", None)
    finetuned_pass_at_1 = results.get("finetuned_model", {}).get("pass_at_1", None)
    base_with_steering = results.get("base_with_steering", {})
    
    if not base_with_steering:
        raise ValueError("No base_with_steering data found in results.")
    
    # Extract steering weights and pass_at_1 values
    steering_weights = []
    steering_pass_at_1 = []
    
    for weight_str in sorted(base_with_steering.keys(), key=float):
        weight = float(weight_str)
        pass_at_1 = base_with_steering[weight_str].get("pass_at_1", None)
        if pass_at_1 is not None:
            steering_weights.append(weight)
            steering_pass_at_1.append(pass_at_1)
    
    if not steering_weights:
        raise ValueError("No valid steering data found.")
    
    # Christmas colors
    christmas_green = "#228B22"  # Christmas green for borders
    off_white = "#FFFEF7"  # Off-white for plot area
    red_color = "#FF1744"  # Red for line and plots
    dark_goldenrod = "#B8860B"  # Dark goldenrod for finetuned model
    blue_color = "#1976D2"  # Blue for base model
    
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor(off_white)
    
    # Determine y-axis limits
    all_values = steering_pass_at_1.copy()
    if base_pass_at_1 is not None:
        all_values.append(base_pass_at_1)
    if finetuned_pass_at_1 is not None:
        all_values.append(finetuned_pass_at_1)
    
    y_min = min(all_values)
    y_max = max(all_values)
    y_range = y_max - y_min
    ax.set_ylim(max(0.0, y_min - 0.05 * y_range), y_max + 0.05 * y_range)
    
    # Determine x-axis limits
    x_min = min(steering_weights)
    x_max = max(steering_weights)
    x_range = x_max - x_min
    ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
    
    # Plot base_with_steering line (red, bold)
    ax.plot(
        steering_weights,
        steering_pass_at_1,
        color=red_color,
        linewidth=3.5,
        label="Base + Steering",
        zorder=3,
        marker="o",
        markersize=8,
    )
    ax.scatter(
        steering_weights,
        steering_pass_at_1,
        color=red_color,
        alpha=0.8,
        s=100,
        zorder=4,
        edgecolors=christmas_green,
        linewidths=2,
    )
    
    # Plot horizontal line for base model (blue)
    if base_pass_at_1 is not None:
        ax.axhline(
            y=base_pass_at_1,
            color=blue_color,
            linestyle="--",
            linewidth=3,
            label=f"Base Model ({base_pass_at_1:.2f}%)",
            zorder=2,
            alpha=0.8,
        )
    
    # Plot horizontal line for finetuned model (dark goldenrod)
    if finetuned_pass_at_1 is not None:
        ax.axhline(
            y=finetuned_pass_at_1,
            color=dark_goldenrod,
            linestyle="--",
            linewidth=3,
            label=f"Finetuned Model ({finetuned_pass_at_1:.2f}%)",
            zorder=2,
            alpha=0.8,
        )
    
    # Add Christmas emojis using high-quality images
    tree_url = "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f384.png"
    gift_url = "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f381.png"
    
    tree_img = get_emoji_image(tree_url, zoom=0.55)
    gift_img = get_emoji_image(gift_url, zoom=0.55)
    
    # Set title
    ax.set_title(title, fontsize=20, color=christmas_green, pad=20, weight="bold")
    
    # Add images next to title if loaded successfully
    if tree_img:
        ab_tree = AnnotationBbox(tree_img, (0.05, 1.06), xycoords='axes fraction', frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab_tree)
        
    if gift_img:
        ab_gift = AnnotationBbox(gift_img, (0.95, 1.06), xycoords='axes fraction', frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab_gift)
    
    ax.set_xlabel("Steering Weight", fontsize=16, color=christmas_green, weight="bold")
    ax.set_ylabel("Pass@1 (%)", fontsize=16, color=christmas_green, weight="bold")
    
    # Style the grid and axes - Christmas green borders
    ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.3, color=christmas_green)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_color(christmas_green)
    ax.spines["right"].set_color(christmas_green)
    ax.spines["left"].set_color(christmas_green)
    ax.spines["bottom"].set_color(christmas_green)
    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    
    # Style ticks - Christmas green
    ax.tick_params(colors=christmas_green, which="both")
    
    # Legend with Christmas styling
    legend = ax.legend(
        loc="best",
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
        description="Plot steering vector evaluation results with Christmas theme"
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("steering_eval_results.json"),
        help="Path to steering_eval_results.json file",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("plots/steering_results.png"),
        help="Path to save the output plot",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Effect on Reasoning Performance MATH Dataset",
        help="Title for the plot",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to load results and create plots."""
    args = parse_args()
    
    # Load results
    print(f"Loading results from {args.results_path}...")
    results = load_steering_results(args.results_path)
    
    # Create plot
    print("Creating plot...")
    plot_steering_results(results, args.output_path, title=args.title)
    
    print("Done!")


if __name__ == "__main__":
    main()

