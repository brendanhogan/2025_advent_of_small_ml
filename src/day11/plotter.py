"""
Utility script to visualize training and evaluation cosine similarities.
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
        # Create a white background to prevent transparency issues
        # bg = Image.new("RGBA", img.size, (255, 255, 255, 0))
        # out = Image.alpha_composite(bg, img)
        return OffsetImage(img, zoom=zoom)
    except Exception as e:
        print(f"Warning: Failed to load emoji from {url}: {e}")
        return None


def load_training_cosines(train_log_path: Path) -> tuple[list[int], list[float]]:
    """Load per-round cosine similarity metrics from the training log JSON."""
    with train_log_path.open("r") as f:
        train_metrics = json.load(f)

    rounds = sorted(int(step) for step in train_metrics.keys())
    cosines: list[float] = []

    for step in rounds:
        record = train_metrics[str(step)]
        cosine = record.get("rewards/cosine_similarity")
        if cosine is None:
            cosine = record.get("reward")
        if cosine is None:
            raise ValueError(f"No cosine similarity metric found for training step {step}.")
        cosines.append(float(cosine))

    return rounds, cosines


def load_eval_cosines(eval_dir: Path) -> tuple[list[int], list[float]]:
    """Load evaluation cosine similarity metrics from individual eval JSON files."""
    eval_files = sorted(
        eval_dir.glob("eval_*.json"),
        key=lambda path: int(path.stem.split("_")[1])
    )

    rounds: list[int] = []
    cosines: list[float] = []

    for file in eval_files:
        with file.open("r") as f:
            data = json.load(f)
        rounds.append(int(data["round"]))
        cosines.append(float(data["avg_cosine_similarity"]))

    return rounds, cosines


def load_checkpoint_pass_at_1(ckpt_test_dir: Path) -> tuple[list[int], list[float]]:
    """Load pass@1 metrics from checkpoint evaluation JSON files."""
    if not ckpt_test_dir.exists():
        return [], []
    
    # Find all JSON files in the checkpoint test directory
    ckpt_files = list(ckpt_test_dir.glob("*.json"))
    
    rounds: list[int] = []
    pass_at_1_values: list[float] = []
    
    for file in ckpt_files:
        with file.open("r") as f:
            data = json.load(f)
        
        checkpoint_name = data.get("checkpoint_name", "")
        overall_pass_at_1 = data.get("overall_pass_at_1")
        
        if overall_pass_at_1 is None:
            continue
        
        # Extract round number from checkpoint name
        if checkpoint_name == "base_model":
            round_num = 0
        elif checkpoint_name.startswith("ckpt-"):
            try:
                round_num = int(checkpoint_name.split("-")[1])
            except (ValueError, IndexError):
                continue
        else:
            continue
        
        rounds.append(round_num)
        pass_at_1_values.append(float(overall_pass_at_1))
    
    # Sort by round number
    sorted_pairs = sorted(zip(rounds, pass_at_1_values))
    if sorted_pairs:
        rounds, pass_at_1_values = zip(*sorted_pairs)
        rounds = list(rounds)
        pass_at_1_values = list(pass_at_1_values)
    
    return rounds, pass_at_1_values


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


def plot_series(
    rounds: Iterable[int],
    cosines: Iterable[float],
    title: str,
    ylabel: str,
    output_path: Path,
    ma_window: int = 10,
) -> None:
    """Create and save a Christmas-themed line plot with moving average."""
    rounds_list = list(rounds)
    cosines_list = list(cosines)

    if not rounds_list:
        raise ValueError(f"No data available to plot {title}.")

    # Compute moving average
    ma_values = compute_moving_average(cosines_list, ma_window)
    
    # Christmas colors
    christmas_green = "#228B22"  # Christmas green for borders
    off_white = "#FFFEF7"  # Off-white for plot area
    bright_yellow = "#FFEB3B"  # Bright yellow for all text
    red_color = "#FF1744"  # Red for line and plots
    
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor(off_white)
    
    # Set y-axis limits (adjust based on data range)
    y_min = min(min(cosines_list), min(ma_values))
    y_max = max(max(cosines_list), max(ma_values))
    y_range = y_max - y_min
    # For cosine similarity, start at 0.5; for pass@1, start at 0
    y_lower_bound = 0.5 if "Cosine" in title else 0.0
    ax.set_ylim(max(y_lower_bound, y_min - 0.05 * y_range), y_max + 0.05 * y_range)
    
    # Plot raw values (red, slightly transparent)
    ax.plot(
        rounds_list,
        cosines_list,
        color=red_color,
        linewidth=1.5,
        alpha=0.4,
        label="Raw Values",
        zorder=1,
    )
    ax.scatter(
        rounds_list,
        cosines_list,
        color=red_color,
        alpha=0.3,
        s=20,
        zorder=1,
    )
    
    # Plot moving average (red, bold)
    ax.plot(
        rounds_list,
        ma_values,
        color=red_color,
        linewidth=3.5,
        label=f"{ma_window}-Round Moving Average",
        zorder=2,
    )
    ax.fill_between(
        rounds_list,
        ma_values,
        color=red_color,
        alpha=0.2,
        zorder=2,
    )
    
    # Add Christmas emojis using high-quality images
    # Using standard Twemoji
    tree_url = "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f384.png"
    gift_url = "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f381.png"
    
    tree_img = get_emoji_image(tree_url, zoom=0.55)
    gift_img = get_emoji_image(gift_url, zoom=0.55)
    
    # Set title without emojis in text
    ax.set_title(title, fontsize=20, color=christmas_green, pad=20, weight="bold")
    
    # Add images next to title if loaded successfully
    if tree_img:
        # Place tree to the left of title
        # Coordinates are in axes fraction (0,0 is bottom-left, 1,1 is top-right)
        ab_tree = AnnotationBbox(tree_img, (0.05, 1.06), xycoords='axes fraction', frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab_tree)
        
    if gift_img:
        # Place gift to the right of title
        ab_gift = AnnotationBbox(gift_img, (0.95, 1.06), xycoords='axes fraction', frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab_gift)
    
    ax.set_xlabel("Round", fontsize=16, color=christmas_green, weight="bold")
    ax.set_ylabel(f"{ylabel} (↑ better)", fontsize=16, color=christmas_green, weight="bold")
    
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training and evaluation cosine similarities."
    )
    parser.add_argument(
        "--train-log",
        type=Path,
        default=Path("run_1/training_logs/train_logs.json"),
        help="Path to training log JSON file.",
    )
    parser.add_argument(
        "--eval-log-dir",
        type=Path,
        default=Path("run_1/eval_logs"),
        help="Directory containing eval_*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("run_1/plots"),
        help="Directory to store generated PNG plots.",
    )
    parser.add_argument(
        "--train-output-name",
        type=str,
        default="training_cosine.png",
        help="Filename for the training cosine plot.",
    )
    parser.add_argument(
        "--eval-output-name",
        type=str,
        default="eval_cosine.png",
        help="Filename for the eval cosine plot.",
    )
    parser.add_argument(
        "--ma-window",
        type=int,
        default=10,
        help="Window size for moving average (default: 10).",
    )
    parser.add_argument(
        "--ckpt-test-dir",
        type=Path,
        default=Path("run_1/ckpt_tests"),
        help="Directory containing checkpoint evaluation JSON files.",
    )
    parser.add_argument(
        "--pass-at-1-output-name",
        type=str,
        default="pass_at_1.png",
        help="Filename for the pass@1 plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_rounds, train_cosines = load_training_cosines(args.train_log)
    eval_rounds, eval_cosines = load_eval_cosines(args.eval_log_dir)

    plot_series(
        train_rounds,
        train_cosines,
        title="Training Average Cosine Similarity",
        ylabel="Cosine Similarity",
        output_path=args.output_dir / args.train_output_name,
        ma_window=args.ma_window,
    )

    plot_series(
        eval_rounds,
        eval_cosines,
        title="Evaluation Average Cosine Similarity",
        ylabel="Cosine Similarity",
        output_path=args.output_dir / args.eval_output_name,
        ma_window=args.ma_window,
    )

    # Plot pass@1 from checkpoint evaluations if available
    ckpt_rounds, ckpt_pass_at_1 = load_checkpoint_pass_at_1(args.ckpt_test_dir)
    if ckpt_rounds:
        plot_series(
            ckpt_rounds,
            ckpt_pass_at_1,
            title="pass@1 on CharXiv reasoning questions",
            ylabel="pass@1",
            output_path=args.output_dir / args.pass_at_1_output_name,
            ma_window=args.ma_window,
        )


if __name__ == "__main__":
    main()

