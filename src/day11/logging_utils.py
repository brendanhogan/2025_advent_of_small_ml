"""
Logging utilities for human preference training.
Saves images, prompts, tournament results, and metrics per round.
"""

import os
import json
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from PIL import Image


def setup_run_directory(output_dir: str) -> Dict[str, str]:
    """
    Create directory structure for a training run.
    
    Args:
        output_dir: Base output directory
    
    Returns:
        Dict with paths to subdirectories
    """
    
    paths = {
        'base': output_dir,
        'rounds': os.path.join(output_dir, 'rounds'),
        'checkpoints': os.path.join(output_dir, 'checkpoints'),
        'logs': os.path.join(output_dir, 'logs'),
        'evals': os.path.join(output_dir, 'evals'),
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths


def save_round_data(
    round_num: int,
    prompts: List[str],
    image_paths: List[str],
    win_rates: List[float],
    output_dir: str,
    meta_prompt: str = None,
    metrics: Dict[str, Any] = None
) -> str:
    """
    Save all data from a training round.
    
    Args:
        round_num: Round number
        prompts: List of generated prompts
        image_paths: List of paths to generated images
        win_rates: Win rates from tournament
        output_dir: Base output directory
        meta_prompt: The meta-prompt used to generate prompts
        metrics: Additional metrics (loss, etc.)
    
    Returns:
        Path to the round directory
    """
    
    round_dir = os.path.join(output_dir, 'rounds', f'round_{round_num:04d}')
    os.makedirs(round_dir, exist_ok=True)
    
    # Copy images to round directory (so they're preserved even if temp files are cleaned up)
    saved_image_paths = []
    for idx, src_path in enumerate(image_paths):
        dst_path = os.path.join(round_dir, f'image_{idx}.png')
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        saved_image_paths.append(dst_path)
    
    # Save round data as JSON
    round_data = {
        'round_num': round_num,
        'timestamp': datetime.now().isoformat(),
        'meta_prompt': meta_prompt,
        'prompts': prompts,
        'win_rates': win_rates,
        'image_paths': saved_image_paths,
        'metrics': metrics or {}
    }
    
    data_path = os.path.join(round_dir, 'round_data.json')
    with open(data_path, 'w') as f:
        json.dump(round_data, f, indent=2)
    
    # Save prompts as individual text files for easy reading
    for idx, prompt in enumerate(prompts):
        prompt_path = os.path.join(round_dir, f'prompt_{idx}.txt')
        with open(prompt_path, 'w') as f:
            f.write(f"Win Rate: {win_rates[idx]:.4f}\n")
            f.write("="*50 + "\n\n")
            f.write(prompt)
    
    # Create a summary image grid (optional - for quick visualization)
    try:
        create_round_summary_image(round_dir, prompts, win_rates)
    except Exception as e:
        print(f"Warning: Could not create summary image: {e}")
    
    return round_dir


def create_round_summary_image(
    round_dir: str,
    prompts: List[str],
    win_rates: List[float]
) -> str:
    """
    Create a grid image showing all images from a round with win rates.
    
    Args:
        round_dir: Directory containing round images
        prompts: List of prompts
        win_rates: Win rates for each image
    
    Returns:
        Path to the summary image
    """
    from PIL import ImageDraw, ImageFont
    
    # Load images
    images = []
    for i in range(len(prompts)):
        img_path = os.path.join(round_dir, f'image_{i}.png')
        if os.path.exists(img_path):
            images.append(Image.open(img_path))
        else:
            # Create placeholder
            images.append(Image.new('RGB', (256, 256), 'gray'))
    
    # Dynamic grid based on number of images
    n_images = len(images)
    if n_images <= 4:
        cols = 2
    else:
        cols = 4
    rows = (n_images + cols - 1) // cols
    
    img_size = 256
    padding = 10
    label_height = 30
    
    grid_width = cols * img_size + (cols + 1) * padding
    grid_height = rows * (img_size + label_height) + (rows + 1) * padding
    
    grid = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(grid)
    
    # Try to get a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for idx, (img, win_rate) in enumerate(zip(images, win_rates)):
        row = idx // cols
        col = idx % cols
        
        x = padding + col * (img_size + padding)
        y = padding + row * (img_size + label_height + padding)
        
        # Resize image
        img_resized = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        grid.paste(img_resized, (x, y))
        
        # Draw win rate label
        label = f"#{idx+1} WR: {win_rate:.2f}"
        label_y = y + img_size + 5
        draw.text((x + 5, label_y), label, fill='black', font=font)
    
    summary_path = os.path.join(round_dir, 'summary_grid.png')
    grid.save(summary_path)
    
    return summary_path


def update_training_log(
    round_num: int,
    loss: float,
    mean_reward: float,
    win_rates: List[float],
    output_dir: str
) -> None:
    """
    Update the main training log with metrics from this round.
    
    Args:
        round_num: Round number
        loss: Training loss
        mean_reward: Mean reward (mean win rate)
        win_rates: All win rates
        output_dir: Base output directory
    """
    
    log_path = os.path.join(output_dir, 'logs', 'training_log.json')
    
    # Load existing log or create new
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log = json.load(f)
    else:
        log = {'rounds': []}
    
    # Add this round
    log['rounds'].append({
        'round_num': round_num,
        'timestamp': datetime.now().isoformat(),
        'loss': loss,
        'mean_reward': mean_reward,
        'max_reward': max(win_rates),
        'min_reward': min(win_rates),
        'std_reward': (sum((r - mean_reward)**2 for r in win_rates) / len(win_rates)) ** 0.5
    })
    
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)


def save_checkpoint(
    model,
    tokenizer,
    round_num: int,
    output_dir: str,
    optimizer=None,
    metrics: Dict[str, Any] = None
) -> str:
    """
    Save a model checkpoint.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer
        round_num: Round number
        output_dir: Base output directory
        optimizer: Optional optimizer state
        metrics: Optional metrics to save with checkpoint
    
    Returns:
        Path to the checkpoint directory
    """
    import torch
    
    ckpt_dir = os.path.join(output_dir, 'checkpoints', f'checkpoint_{round_num:04d}')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(ckpt_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(ckpt_dir)
    
    # Save checkpoint info
    info = {
        'round_num': round_num,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics or {}
    }
    
    info_path = os.path.join(ckpt_dir, 'checkpoint_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Saved checkpoint to {ckpt_dir}")
    
    return ckpt_dir


def load_checkpoint(
    model,
    checkpoint_path: str,
    device: str = "cuda"
):
    """
    Load a model checkpoint.
    
    Args:
        model: The model to load weights into
        checkpoint_path: Path to checkpoint directory
        device: Device to load to
    
    Returns:
        The model with loaded weights
    """
    import torch
    
    model_path = os.path.join(checkpoint_path, 'model.pt')
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    return model

