"""
Plotter for GEPA vs GRPO composition experiments on MATH dataset.

Supports multiple plot modes:
- 'basic': Original GEPA vs GRPO comparison (day9 style)
- 'composition': GEPA-on-GRPO vs GRPO-with-GEPA comparison
- 'bonus': Improvement over baseline for each composition method
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
import numpy as np

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
CHRISTMAS_RED = "#C41E3A"
CHRISTMAS_GOLD = "#FFD700"

# Method colors
GEPA_ON_GRPO_COLOR = "#9C27B0"  # Purple - combining both
GRPO_WITH_GEPA_COLOR = "#FF6F00"  # Orange - warm combo
GEPA_COLOR = "#FF1744"  # Red
GRPO_COLOR = "#1976D2"  # Blue


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


def extract_series(results: dict) -> tuple[list[int], list[float]]:
    """Extract sorted steps and pass@1 values from results dict."""
    steps = []
    values = []
    for step_str in sorted(results.keys(), key=int):
        steps.append(int(step_str))
        values.append(results[step_str]["pass_at_1"])
    return steps, values


def add_christmas_decorations(ax, title: str) -> None:
    """Add Christmas emojis and styling to the plot."""
    tree_url = "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f384.png"
    gift_url = "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f381.png"
    
    tree_img = get_emoji_image(tree_url, zoom=0.55)
    gift_img = get_emoji_image(gift_url, zoom=0.55)
    
    ax.set_title(title, fontsize=20, color=CHRISTMAS_GREEN, pad=20, weight="bold")
    
    if tree_img:
        ab_tree = AnnotationBbox(tree_img, (0.05, 1.06), xycoords='axes fraction', 
                                  frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab_tree)
        
    if gift_img:
        ab_gift = AnnotationBbox(gift_img, (0.95, 1.06), xycoords='axes fraction', 
                                  frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab_gift)


def style_axes(ax) -> None:
    """Apply Christmas styling to axes."""
    ax.set_facecolor(OFF_WHITE)
    ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.3, color=CHRISTMAS_GREEN)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_color(CHRISTMAS_GREEN)
        ax.spines[spine].set_linewidth(3)
    ax.tick_params(colors=CHRISTMAS_GREEN, which="both")


def style_legend(ax, loc: str = "lower right") -> None:
    """Apply Christmas styling to legend."""
    legend = ax.legend(
        loc=loc,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        edgecolor=CHRISTMAS_GREEN,
        facecolor=OFF_WHITE,
    )
    legend.get_frame().set_linewidth(2)
    for text in legend.get_texts():
        text.set_color(CHRISTMAS_GREEN)


def plot_basic(
    gepa_results: dict,
    grpo_results: dict,
    output_path: Path,
    title: str = "GEPA vs GRPO on MATH Dataset",
) -> None:
    """Create basic GEPA vs GRPO comparison plot (day9 style)."""
    
    gepa_steps, gepa_pass_at_1 = extract_series(gepa_results)
    grpo_steps, grpo_pass_at_1 = extract_series(grpo_results)
    
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    style_axes(ax)
    
    # Determine axis limits
    all_values = gepa_pass_at_1 + grpo_pass_at_1
    y_min, y_max = min(all_values), max(all_values)
    y_range = y_max - y_min
    ax.set_ylim(max(0.0, y_min - 0.1 * y_range), y_max + 0.1 * y_range)
    
    all_steps = gepa_steps + grpo_steps
    ax.set_xlim(min(all_steps) - 20, max(all_steps) + 20)
    
    # Plot GRPO line (blue)
    ax.plot(grpo_steps, grpo_pass_at_1, color=GRPO_COLOR, linewidth=3.5,
            label="GRPO (Weight Updates)", zorder=3, marker="s", markersize=8)
    ax.scatter(grpo_steps, grpo_pass_at_1, color=GRPO_COLOR, alpha=0.8, s=100,
               zorder=4, edgecolors=CHRISTMAS_GREEN, linewidths=2, marker="s")
    
    # Plot GEPA line (red)
    ax.plot(gepa_steps, gepa_pass_at_1, color=GEPA_COLOR, linewidth=3.5,
            label="GEPA (Prompt Evolution)", zorder=3, marker="o", markersize=8)
    ax.scatter(gepa_steps, gepa_pass_at_1, color=GEPA_COLOR, alpha=0.8, s=100,
               zorder=4, edgecolors=CHRISTMAS_GREEN, linewidths=2)
    
    add_christmas_decorations(ax, title)
    ax.set_xlabel("Optimization Step", fontsize=16, color=CHRISTMAS_GREEN, weight="bold")
    ax.set_ylabel("Pass@1 (%)", fontsize=16, color=CHRISTMAS_GREEN, weight="bold")
    style_legend(ax)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_composition(
    gepa_on_grpo_results: dict,
    grpo_with_gepa_results: dict,
    output_path: Path,
    title: str = "Composition: GEPA→GRPO vs GRPO→GEPA",
) -> None:
    """Compare the two composition methods head-to-head."""
    
    gog_steps, gog_values = extract_series(gepa_on_grpo_results)
    gwg_steps, gwg_values = extract_series(grpo_with_gepa_results)
    
    # Truncate both to the same step range (use the shorter one)
    max_step = min(max(gog_steps), max(gwg_steps))
    gog_steps = [s for s in gog_steps if s <= max_step]
    gog_values = gog_values[:len(gog_steps)]
    gwg_steps = [s for s in gwg_steps if s <= max_step]
    gwg_values = gwg_values[:len(gwg_steps)]
    
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    style_axes(ax)
    
    # Determine axis limits
    all_values = gog_values + gwg_values
    y_min, y_max = min(all_values), max(all_values)
    y_range = y_max - y_min
    ax.set_ylim(max(0.0, y_min - 0.12 * y_range), y_max + 0.15 * y_range)
    
    all_steps = gog_steps + gwg_steps
    ax.set_xlim(min(all_steps) - 20, max(all_steps) + 40)
    
    # Plot GEPA on GRPO (purple)
    ax.plot(gog_steps, gog_values, color=GEPA_ON_GRPO_COLOR, linewidth=3.5,
            label="GRPO→GEPA", zorder=3, 
            marker="o", markersize=8)
    ax.scatter(gog_steps, gog_values, color=GEPA_ON_GRPO_COLOR, alpha=0.8, s=100,
               zorder=4, edgecolors=CHRISTMAS_GREEN, linewidths=2)
    
    # Plot GRPO with GEPA (orange)
    ax.plot(gwg_steps, gwg_values, color=GRPO_WITH_GEPA_COLOR, linewidth=3.5,
            label="GEPA→GRPO", zorder=3, 
            marker="s", markersize=8)
    ax.scatter(gwg_steps, gwg_values, color=GRPO_WITH_GEPA_COLOR, alpha=0.8, s=100,
               zorder=4, edgecolors=CHRISTMAS_GREEN, linewidths=2, marker="s")
    
    # Find and mark overall best with a star
    gog_best_idx = np.argmax(gog_values)
    gwg_best_idx = np.argmax(gwg_values)
    
    gog_best = (gog_steps[gog_best_idx], gog_values[gog_best_idx])
    gwg_best = (gwg_steps[gwg_best_idx], gwg_values[gwg_best_idx])
    
    # Determine overall best
    if gwg_best[1] >= gog_best[1]:
        best_step, best_val = gwg_best
        best_color = GRPO_WITH_GEPA_COLOR
        best_label = f"Best: {best_val:.1f}% (GEPA→GRPO @ step {best_step})"
    else:
        best_step, best_val = gog_best
        best_color = GEPA_ON_GRPO_COLOR
        best_label = f"Best: {best_val:.1f}% (GRPO→GEPA @ step {best_step})"
    
    # Add star marker for best
    ax.scatter([best_step], [best_val], color=CHRISTMAS_GOLD, s=400, zorder=10,
               marker="*", edgecolors=CHRISTMAS_GREEN, linewidths=2)
    ax.annotate(best_label, (best_step, best_val), textcoords="offset points",
                xytext=(10, 15), fontsize=11, color=CHRISTMAS_GREEN, weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=CHRISTMAS_GOLD, 
                         edgecolor=CHRISTMAS_GREEN, alpha=0.9))
    
    add_christmas_decorations(ax, title)
    ax.set_xlabel("Optimization Step", fontsize=16, color=CHRISTMAS_GREEN, weight="bold")
    ax.set_ylabel("Pass@1 (%)", fontsize=16, color=CHRISTMAS_GREEN, weight="bold")
    style_legend(ax, loc="lower right")
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_bonus(
    gepa_on_grpo_results: dict,
    grpo_with_gepa_results: dict,
    output_path: Path,
    title: str = "Improvement Over Starting Point",
) -> None:
    """Plot the improvement (bonus) over each method's baseline at each step."""
    
    gog_steps, gog_values = extract_series(gepa_on_grpo_results)
    gwg_steps, gwg_values = extract_series(grpo_with_gepa_results)
    
    # Truncate both to the same step range (use the shorter one)
    max_step = min(max(gog_steps), max(gwg_steps))
    gog_steps = [s for s in gog_steps if s <= max_step]
    gog_values = gog_values[:len(gog_steps)]
    gwg_steps = [s for s in gwg_steps if s <= max_step]
    gwg_values = gwg_values[:len(gwg_steps)]
    
    # Calculate bonus over starting point (step 0)
    gog_baseline = gog_values[0]
    gwg_baseline = gwg_values[0]
    
    gog_bonus = [v - gog_baseline for v in gog_values]
    gwg_bonus = [v - gwg_baseline for v in gwg_values]
    
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    style_axes(ax)
    
    # Determine axis limits
    all_bonus = gog_bonus + gwg_bonus
    y_min, y_max = min(all_bonus), max(all_bonus)
    y_range = max(y_max - y_min, 5)  # At least 5% range
    ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.2 * y_range)
    
    all_steps = gog_steps + gwg_steps
    ax.set_xlim(min(all_steps) - 20, max(all_steps) + 40)
    
    # Add zero line
    ax.axhline(y=0, color=CHRISTMAS_GREEN, linestyle="-", linewidth=2, alpha=0.5, zorder=1)
    
    # Plot GEPA on GRPO bonus (purple)
    ax.plot(gog_steps, gog_bonus, color=GEPA_ON_GRPO_COLOR, linewidth=3.5,
            label="GRPO→GEPA", zorder=3, 
            marker="o", markersize=8)
    ax.scatter(gog_steps, gog_bonus, color=GEPA_ON_GRPO_COLOR, alpha=0.8, s=100,
               zorder=4, edgecolors=CHRISTMAS_GREEN, linewidths=2)
    
    # Plot GRPO with GEPA bonus (orange)
    ax.plot(gwg_steps, gwg_bonus, color=GRPO_WITH_GEPA_COLOR, linewidth=3.5,
            label="GEPA→GRPO", zorder=3, 
            marker="s", markersize=8)
    ax.scatter(gwg_steps, gwg_bonus, color=GRPO_WITH_GEPA_COLOR, alpha=0.8, s=100,
               zorder=4, edgecolors=CHRISTMAS_GREEN, linewidths=2, marker="s")
    
    # Find and mark best bonus for each
    gog_best_idx = np.argmax(gog_bonus)
    gwg_best_idx = np.argmax(gwg_bonus)
    
    gog_best_bonus = gog_bonus[gog_best_idx]
    gwg_best_bonus = gwg_bonus[gwg_best_idx]
    gog_best_step = gog_steps[gog_best_idx]
    gwg_best_step = gwg_steps[gwg_best_idx]
    
    # Calculate absolute best scores for annotation
    gog_peak = gog_baseline + gog_best_bonus
    gwg_peak = gwg_baseline + gwg_best_bonus
    
    # Mark best with stars
    ax.scatter([gog_best_step], [gog_best_bonus], color=CHRISTMAS_GOLD, s=350, zorder=10,
               marker="*", edgecolors=GEPA_ON_GRPO_COLOR, linewidths=2)
    ax.scatter([gwg_best_step], [gwg_best_bonus], color=CHRISTMAS_GOLD, s=350, zorder=10,
               marker="*", edgecolors=GRPO_WITH_GEPA_COLOR, linewidths=2)
    
    # Add annotations for peak performance
    ax.annotate(f"+{gog_best_bonus:.1f}pp → {gog_peak:.1f}%", 
                (gog_best_step, gog_best_bonus), textcoords="offset points",
                xytext=(-60, 12), fontsize=10, color=GEPA_ON_GRPO_COLOR, weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=OFF_WHITE, 
                         edgecolor=GEPA_ON_GRPO_COLOR, alpha=0.9))
    
    ax.annotate(f"+{gwg_best_bonus:.1f}pp → {gwg_peak:.1f}%", 
                (gwg_best_step, gwg_best_bonus), textcoords="offset points",
                xytext=(10, 12), fontsize=10, color=GRPO_WITH_GEPA_COLOR, weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=OFF_WHITE, 
                         edgecolor=GRPO_WITH_GEPA_COLOR, alpha=0.9))
    
    add_christmas_decorations(ax, title)
    ax.set_xlabel("Optimization Step", fontsize=16, color=CHRISTMAS_GREEN, weight="bold")
    ax.set_ylabel("Improvement (percentage points)", fontsize=16, color=CHRISTMAS_GREEN, weight="bold")
    style_legend(ax, loc="upper left")
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot GEPA/GRPO composition experiments with Christmas theme"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["basic", "composition", "bonus", "all"],
        default="all",
        help="Plot mode: 'basic' (GEPA vs GRPO), 'composition' (both compositions), "
             "'bonus' (improvement over baseline), 'all' (generate all plots)",
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
        "--gepa-on-grpo-results",
        type=Path,
        default=Path("gepa_on_grpo_run/eval_summary.json"),
        help="Path to GEPA-on-GRPO eval_summary.json",
    )
    parser.add_argument(
        "--grpo-with-gepa-results",
        type=Path,
        default=Path("grpo_with_gepa_prompt_run/eval_summary.json"),
        help="Path to GRPO-with-GEPA eval_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figs"),
        help="Directory to save output plots",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to load results and create plots."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    modes = ["basic", "composition", "bonus"] if args.mode == "all" else [args.mode]
    
    for mode in modes:
        if mode == "basic":
            print(f"Loading GEPA results from {args.gepa_results}...")
            gepa_results = load_eval_summary(args.gepa_results)
            print(f"Loading GRPO results from {args.grpo_results}...")
            grpo_results = load_eval_summary(args.grpo_results)
            
            print("Creating basic comparison plot...")
            plot_basic(
                gepa_results,
                grpo_results,
                args.output_dir / "gepa_vs_grpo.png",
                title="GEPA vs GRPO on MATH Dataset",
            )
            
        elif mode == "composition":
            print(f"Loading GEPA-on-GRPO results from {args.gepa_on_grpo_results}...")
            gog_results = load_eval_summary(args.gepa_on_grpo_results)
            print(f"Loading GRPO-with-GEPA results from {args.grpo_with_gepa_results}...")
            gwg_results = load_eval_summary(args.grpo_with_gepa_results)
            
            print("Creating composition comparison plot...")
            plot_composition(
                gog_results,
                gwg_results,
                args.output_dir / "composition_comparison.png",
                title="Composition: GEPA→GRPO vs GRPO→GEPA",
            )
            
        elif mode == "bonus":
            print(f"Loading GEPA-on-GRPO results from {args.gepa_on_grpo_results}...")
            gog_results = load_eval_summary(args.gepa_on_grpo_results)
            print(f"Loading GRPO-with-GEPA results from {args.grpo_with_gepa_results}...")
            gwg_results = load_eval_summary(args.grpo_with_gepa_results)
            
            print("Creating bonus (improvement) plot...")
            plot_bonus(
                gog_results,
                gwg_results,
                args.output_dir / "composition_bonus.png",
                title="Improvement Over Starting Point",
            )
    
    print("Done!")


if __name__ == "__main__":
    main()
