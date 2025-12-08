#!/usr/bin/env python3
"""
Plotter for training runs.
Creates:
1. Loss curve (distance_from_target over steps)
2. GIF animation of personality_vs_target.png evolving
"""

import argparse
import json
import re
import urllib.request
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

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
DARK_GOLDENROD = "#B8860B"
BLUE_COLOR = "#1976D2"


def get_emoji_image(url: str, zoom: float = 0.15) -> OffsetImage:
    """Download and prepare an emoji image for plotting."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            data = response.read()
        img = Image.open(BytesIO(data)).convert("RGBA")
        return OffsetImage(img, zoom=zoom)
    except Exception as e:
        print(f"Warning: Failed to load emoji from {url}: {e}")
        return None


def get_eval_dirs(run_dir: Path) -> list[tuple[int, Path]]:
    """Get all eval_step_* directories sorted by step number."""
    eval_dirs = []
    for d in run_dir.iterdir():
        if d.is_dir() and d.name.startswith("eval_step_"):
            match = re.match(r"eval_step_(\d+)", d.name)
            if match:
                step = int(match.group(1))
                eval_dirs.append((step, d))
    return sorted(eval_dirs, key=lambda x: x[0])


def style_axis_christmas(ax, title: str = None) -> None:
    """Apply Christmas styling to an axis."""
    ax.set_facecolor(OFF_WHITE)
    ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.3, color=CHRISTMAS_GREEN)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(CHRISTMAS_GREEN)
        spine.set_linewidth(2.5)
    
    ax.tick_params(colors=CHRISTMAS_GREEN, which="both")
    
    if title:
        ax.set_title(title, fontsize=18, color=CHRISTMAS_GREEN, pad=15, weight="bold")


def style_legend_christmas(ax) -> None:
    """Style legend with Christmas theme."""
    legend = ax.legend(
        loc="best",
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


def plot_loss_curve(run_dir: Path, output_dir: Path) -> None:
    """Plot distance_from_target over training steps."""
    eval_dirs = get_eval_dirs(run_dir)
    
    steps = []
    distances = []
    per_dim = {dim: [] for dim in ["N", "E", "O", "A", "C"]}
    
    for step, eval_dir in eval_dirs:
        results_file = eval_dir / "eval_results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            if "distance_from_target" in results:
                steps.append(step)
                distances.append(results["distance_from_target"])
                if "per_dim_distance" in results:
                    for dim in per_dim:
                        per_dim[dim].append(results["per_dim_distance"].get(dim, 0))
    
    if not steps:
        print("No eval results found with distance_from_target")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), facecolor="white", 
                                    gridspec_kw={'height_ratios': [2, 1]})
    
    # Christmas emojis
    tree_url = "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f384.png"
    gift_url = "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f381.png"
    tree_img = get_emoji_image(tree_url, zoom=0.5)
    gift_img = get_emoji_image(gift_url, zoom=0.5)
    
    # Main loss curve
    ax1.plot(steps, distances, 'o-', linewidth=3, markersize=6, color=RED_COLOR, 
             label='Total Distance', zorder=3)
    ax1.scatter(steps, distances, color=RED_COLOR, alpha=0.8, s=80, zorder=4,
                edgecolors=CHRISTMAS_GREEN, linewidths=2)
    
    ax1.set_xlabel('Training Step', fontsize=16, color=CHRISTMAS_GREEN, weight="bold")
    ax1.set_ylabel('Distance from Target (L1)', fontsize=16, color=CHRISTMAS_GREEN, weight="bold")
    
    style_axis_christmas(ax1, f'Personality Training: {run_dir.name}')
    
    # Add emojis next to title
    if tree_img:
        ab_tree = AnnotationBbox(tree_img, (0.02, 1.08), xycoords='axes fraction', 
                                  frameon=False, box_alignment=(0.5, 0.5))
        ax1.add_artist(ab_tree)
    if gift_img:
        ab_gift = AnnotationBbox(gift_img, (0.98, 1.08), xycoords='axes fraction', 
                                  frameon=False, box_alignment=(0.5, 0.5))
        ax1.add_artist(ab_gift)
    
    ax1.axhline(y=0, color=CHRISTMAS_GREEN, linestyle='--', linewidth=2, alpha=0.6, label='Perfect (0)')
    
    # Add best point annotation
    best_idx = np.argmin(distances)
    best_step = steps[best_idx]
    best_dist = distances[best_idx]
    ax1.annotate(f'Best: {best_dist:.2f} @ step {best_step}', 
                 xy=(best_step, best_dist), 
                 xytext=(best_step + len(steps)*0.15, best_dist + 0.4),
                 arrowprops=dict(arrowstyle='->', color=CHRISTMAS_GREEN, linewidth=2),
                 fontsize=12, color=CHRISTMAS_GREEN, weight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=OFF_WHITE, edgecolor=CHRISTMAS_GREEN))
    
    style_legend_christmas(ax1)
    
    # Per-dimension breakdown with Christmas colors
    dim_colors = {
        'N': '#FF1744',  # Red - Neuroticism
        'E': '#FFD700',  # Gold - Extraversion  
        'O': '#9C27B0',  # Purple - Openness
        'A': '#228B22',  # Green - Agreeableness
        'C': '#1976D2',  # Blue - Conscientiousness
    }
    dim_names = {'N': 'Neuroticism', 'E': 'Extraversion', 'O': 'Openness', 
                 'A': 'Agreeableness', 'C': 'Conscientiousness'}
    
    for dim in ["N", "E", "O", "A", "C"]:
        if per_dim[dim]:
            ax2.plot(steps, per_dim[dim], 'o-', linewidth=2.5, markersize=4, 
                    color=dim_colors[dim], label=dim_names[dim], alpha=0.85)
    
    ax2.set_xlabel('Training Step', fontsize=16, color=CHRISTMAS_GREEN, weight="bold")
    ax2.set_ylabel('Per-Dimension Distance', fontsize=16, color=CHRISTMAS_GREEN, weight="bold")
    style_axis_christmas(ax2, 'Distance by Personality Dimension')
    ax2.axhline(y=0, color=CHRISTMAS_GREEN, linestyle='--', linewidth=1.5, alpha=0.4)
    
    legend2 = ax2.legend(loc='upper right', ncol=5, fontsize=10, frameon=True,
                         fancybox=True, shadow=True, framealpha=0.95,
                         edgecolor=CHRISTMAS_GREEN, facecolor=OFF_WHITE)
    legend2.get_frame().set_linewidth(2)
    for text in legend2.get_texts():
        text.set_color(CHRISTMAS_GREEN)
    
    plt.tight_layout()
    
    output_path = output_dir / "loss_curve.png"
    plt.savefig(output_path, dpi=150, facecolor="white", bbox_inches='tight')
    plt.close(fig)
    print(f"Saved loss curve to {output_path}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY")
    print(f"{'='*50}")
    print(f"Start distance: {distances[0]:.3f}")
    print(f"Final distance: {distances[-1]:.3f}")
    print(f"Best distance:  {best_dist:.3f} (step {best_step})")
    print(f"Improvement:    {distances[0] - distances[-1]:.3f} ({(1 - distances[-1]/distances[0])*100:.1f}%)")


def create_evolution_gif(run_dir: Path, output_dir: Path, fps: int = 4) -> None:
    """Create GIF animation of personality_vs_target.png evolving."""
    eval_dirs = get_eval_dirs(run_dir)
    
    frames = []
    for step, eval_dir in eval_dirs:
        img_path = eval_dir / "personality_vs_target.png"
        if img_path.exists():
            img = Image.open(img_path)
            # Convert to RGB if necessary (for GIF compatibility)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            frames.append((step, img))
    
    if not frames:
        print("No personality_vs_target.png images found")
        return
    
    # Add step number overlay to each frame
    frames_with_labels = []
    for step, img in frames:
        # Create a copy to draw on
        img_copy = img.copy()
        frames_with_labels.append(img_copy)
    
    # Save as GIF
    output_path = output_dir / "personality_evolution.gif"
    duration = int(1000 / fps)  # milliseconds per frame
    
    # Add longer pause on first and last frame
    durations = [duration * 3] + [duration] * (len(frames_with_labels) - 2) + [duration * 5] if len(frames_with_labels) > 2 else [duration] * len(frames_with_labels)
    
    frames_with_labels[0].save(
        output_path,
        save_all=True,
        append_images=frames_with_labels[1:],
        duration=durations,
        loop=0,
    )
    print(f"Saved evolution GIF to {output_path} ({len(frames)} frames)")


def create_progress_gif(run_dir: Path, output_dir: Path, fps: int = 4) -> None:
    """Create GIF animation of progress_from_baseline.png if available."""
    eval_dirs = get_eval_dirs(run_dir)
    
    frames = []
    for step, eval_dir in eval_dirs:
        img_path = eval_dir / "progress_from_baseline.png"
        if img_path.exists():
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            frames.append((step, img))
    
    if not frames:
        print("No progress_from_baseline.png images found (only available after step 0)")
        return
    
    frames_list = [img for _, img in frames]
    output_path = output_dir / "progress_evolution.gif"
    duration = int(1000 / fps)
    durations = [duration * 3] + [duration] * (len(frames_list) - 2) + [duration * 5] if len(frames_list) > 2 else [duration] * len(frames_list)
    
    frames_list[0].save(
        output_path,
        save_all=True,
        append_images=frames_list[1:],
        duration=durations,
        loop=0,
    )
    print(f"Saved progress GIF to {output_path} ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser(description="Plot training results")
    parser.add_argument("run_dir", type=str, help="Path to training run directory (e.g., train_jerk_high_temp)")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second for GIF")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: run_dir/plots)")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing: {run_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Create loss curve
    plot_loss_curve(run_dir, output_dir)
    
    # Create evolution GIF
    create_evolution_gif(run_dir, output_dir, fps=args.fps)
    
    # Create progress GIF if available
    create_progress_gif(run_dir, output_dir, fps=args.fps)
    
    print(f"\nDone! All plots saved to {output_dir}")


if __name__ == "__main__":
    main()

