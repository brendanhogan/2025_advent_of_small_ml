"""
Plotter for day7 GRPO training runs with Christmas styling.
Plots eval and training metrics across multiple runs.
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

# Christmas colors for different runs
CHRISTMAS_COLORS = [
    "#FF1744",  # Red
    "#228B22",  # Christmas green
    "#FF6B35",  # Orange-red
    "#4ECDC4",  # Teal
    "#FFEB3B",  # Yellow
    "#9C27B0",  # Purple
    "#2196F3",  # Blue
    "#FF9800",  # Orange
    "#E91E63",  # Pink
    "#00BCD4",  # Cyan
]


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


def load_eval_metrics(run_dir: Path, metric_name: str = "pass_at_1") -> tuple[list[int], list[float]]:
    """Load evaluation metrics from eval_summary.json."""
    eval_summary_path = run_dir / "eval_summary.json"
    if not eval_summary_path.exists():
        return [], []
    
    with eval_summary_path.open("r") as f:
        data = json.load(f)
    
    steps = []
    values = []
    
    for step_str, metrics in sorted(data.items(), key=lambda x: int(x[0])):
        step = int(step_str)
        value = metrics.get(metric_name)
        if value is not None:
            steps.append(step)
            values.append(float(value))
    
    return steps, values


def load_training_metrics(run_dir: Path, metric_name: str = "loss") -> tuple[list[int], list[float]]:
    """Load training metrics from run_log.json."""
    run_log_path = run_dir / "run_log.json"
    if not run_log_path.exists():
        return [], []
    
    with run_log_path.open("r") as f:
        data = json.load(f)
    
    steps = []
    values = []
    
    if "steps" not in data:
        return [], []
    
    for step_str, step_data in sorted(data["steps"].items(), key=lambda x: int(x[0])):
        step = int(step_str)
        train_data = step_data.get("train", {})
        value = train_data.get(metric_name)
        if value is not None:
            steps.append(step)
            values.append(float(value))
    
    return steps, values


def compute_moving_average(values: list[float], window: int) -> list[float]:
    """Compute trailing moving average."""
    if window <= 0:
        raise ValueError("Moving average window must be > 0.")
    if window > len(values):
        window = len(values)
    
    ma: list[float] = []
    for i in range(len(values)):
        start_idx = max(0, i - window + 1)
        window_values = values[start_idx:i + 1]
        ma.append(sum(window_values) / len(window_values))
    
    return ma


def plot_multiple_runs(
    runs_data: list[tuple[str, list[int], list[float]]],
    title: str,
    ylabel: str,
    output_path: Path,
    ma_window: int = 10,
    show_ma: bool = True,
) -> None:
    """Create and save a Christmas-themed line plot with multiple runs."""
    if not runs_data:
        print(f"Warning: No data available to plot {title}.")
        return
    
    # Christmas colors
    christmas_green = "#228B22"  # Christmas green for borders
    off_white = "#FFFEF7"  # Off-white for plot area
    
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="white")
    ax.set_facecolor(off_white)
    
    # Plot each run
    for idx, (run_name, steps, values) in enumerate(runs_data):
        if not steps or not values:
            continue
        
        color = CHRISTMAS_COLORS[idx % len(CHRISTMAS_COLORS)]
        
        # Compute moving average
        ma_values = compute_moving_average(values, ma_window) if show_ma else values
        
        # Plot raw values (slightly transparent)
        ax.plot(
            steps,
            values,
            color=color,
            linewidth=1.5,
            alpha=0.3,
            zorder=1,
        )
        
        # Plot moving average (bold)
        if show_ma:
            ax.plot(
                steps,
                ma_values,
                color=color,
                linewidth=3.0,
                label=run_name,
                zorder=2,
            )
        else:
            ax.plot(
                steps,
                values,
                color=color,
                linewidth=3.0,
                label=run_name,
                zorder=2,
            )
    
    # Set y-axis limits
    all_values = []
    for _, _, values in runs_data:
        all_values.extend(values)
    
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min
        # For pass@1, start at 0; for loss, allow some padding
        y_lower_bound = 0.0 if "pass@1" in title.lower() or "Pass@" in title else None
        if y_lower_bound is not None:
            ax.set_ylim(max(y_lower_bound, y_min - 0.05 * y_range), y_max + 0.05 * y_range)
        else:
            ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    # Add Christmas emojis
    tree_url = "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f384.png"
    gift_url = "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f381.png"
    
    tree_img = get_emoji_image(tree_url, zoom=0.55)
    gift_img = get_emoji_image(gift_url, zoom=0.55)
    
    # Set title
    ax.set_title(title, fontsize=20, color=christmas_green, pad=20, weight="bold")
    
    # Add emoji images
    if tree_img:
        ab_tree = AnnotationBbox(tree_img, (0.05, 1.06), xycoords='axes fraction', frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab_tree)
    
    if gift_img:
        ab_gift = AnnotationBbox(gift_img, (0.95, 1.06), xycoords='axes fraction', frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab_gift)
    
    ax.set_xlabel("Step", fontsize=16, color=christmas_green, weight="bold")
    ax.set_ylabel(f"{ylabel} (↑ better)" if "pass" in ylabel.lower() or "reward" in ylabel.lower() else f"{ylabel} (↓ better)", 
                  fontsize=16, color=christmas_green, weight="bold")
    
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
    if runs_data:
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


def find_all_runs(runs_dir: Path) -> list[Path]:
    """Find all run directories that have eval_summary.json or run_log.json."""
    runs = []
    if not runs_dir.exists():
        return runs
    
    for item in runs_dir.iterdir():
        if item.is_dir():
            if (item / "eval_summary.json").exists() or (item / "run_log.json").exists():
                runs.append(item)
    
    return sorted(runs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GRPO training and evaluation metrics across multiple runs."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory containing run subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to store generated PNG plots.",
    )
    parser.add_argument(
        "--ma-window",
        type=int,
        default=2,
        help="Window size for moving average (default: 10).",
    )
    parser.add_argument(
        "--no-ma",
        action="store_true",
        help="Disable moving average (show raw values only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Find all runs
    runs = find_all_runs(args.runs_dir)
    
    if not runs:
        print(f"No runs found in {args.runs_dir}")
        return
    
    print(f"Found {len(runs)} runs:")
    for run in runs:
        print(f"  - {run.name}")
    
    # Collect eval pass@1 data
    eval_pass_at_1_data = []
    for run_dir in runs:
        steps, values = load_eval_metrics(run_dir, "pass_at_1")
        if steps and values:
            eval_pass_at_1_data.append((run_dir.name, steps, values))
    
    # Collect eval format reward data
    eval_format_data = []
    for run_dir in runs:
        steps, values = load_eval_metrics(run_dir, "avg_format_reward")
        if steps and values:
            eval_format_data.append((run_dir.name, steps, values))
    
    # Collect eval entropy reward data (if available)
    eval_entropy_data = []
    for run_dir in runs:
        steps, values = load_eval_metrics(run_dir, "avg_entropy_reward")
        if steps and values:
            eval_entropy_data.append((run_dir.name, steps, values))
    
    # Collect training loss data
    train_loss_data = []
    for run_dir in runs:
        steps, values = load_training_metrics(run_dir, "loss")
        if steps and values:
            train_loss_data.append((run_dir.name, steps, values))
    
    # Plot eval pass@1
    if eval_pass_at_1_data:
        plot_multiple_runs(
            eval_pass_at_1_data,
            title="Evaluation Pass@1 Across Runs",
            ylabel="Pass@1 (%)",
            output_path=args.output_dir / "eval_pass_at_1.png",
            ma_window=args.ma_window,
            show_ma=not args.no_ma,
        )
        print(f"Saved eval pass@1 plot to {args.output_dir / 'eval_pass_at_1.png'}")
    
    # Plot eval format reward
    if eval_format_data:
        plot_multiple_runs(
            eval_format_data,
            title="Evaluation Format Reward Across Runs",
            ylabel="Avg Format Reward",
            output_path=args.output_dir / "eval_format_reward.png",
            ma_window=args.ma_window,
            show_ma=not args.no_ma,
        )
        print(f"Saved eval format reward plot to {args.output_dir / 'eval_format_reward.png'}")
    
    # Plot eval entropy reward (if available)
    if eval_entropy_data:
        plot_multiple_runs(
            eval_entropy_data,
            title="Evaluation Entropy Reward Across Runs",
            ylabel="Avg Entropy Reward",
            output_path=args.output_dir / "eval_entropy_reward.png",
            ma_window=args.ma_window,
            show_ma=not args.no_ma,
        )
        print(f"Saved eval entropy reward plot to {args.output_dir / 'eval_entropy_reward.png'}")
    
    # Plot training loss
    if train_loss_data:
        plot_multiple_runs(
            train_loss_data,
            title="Training Loss Across Runs",
            ylabel="Loss",
            output_path=args.output_dir / "train_loss.png",
            ma_window=args.ma_window,
            show_ma=not args.no_ma,
        )
        print(f"Saved training loss plot to {args.output_dir / 'train_loss.png'}")


if __name__ == "__main__":
    main()

