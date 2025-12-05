"""
Bayesian optimization for finding optimal steering vector weights.

This script:
1. Divides 28 layers into 4 groups (early, mid-early, mid-late, late)
2. Uses Bayesian optimization to find optimal weight for each group
3. Evaluates each candidate configuration
4. Creates visualizations (1D plots, 2D grids, GIF)
5. Saves best configuration and results
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from skopt import gp_minimize
from skopt.space import Real
from skopt.acquisition import gaussian_ei
from skopt.utils import use_named_args

# Add parent directory to path for imports (if needed)
sys.path.insert(0, str(Path(__file__).parent.parent / "day4"))

import llms
import utils
from math_dataset import load_math_dataset, format_math_problem, extract_math_answer

from build_steering_vectors import load_steering_vectors

# Set matplotlib style
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# Layer grouping: 28 layers divided into 4 groups of 7 each
LAYER_GROUPS = {
    "early": list(range(0, 7)),      # Layers 0-6
    "mid_early": list(range(7, 14)), # Layers 7-13
    "mid_late": list(range(14, 21)), # Layers 14-20
    "late": list(range(21, 28))      # Layers 21-27
}

GROUP_NAMES = ["early", "mid_early", "mid_late", "late"]


class GroupedSteeringModelWrapper:
    """
    Wrapper that applies different steering weights to different layer groups.
    """
    
    def __init__(self, base_model, steering_vectors, group_weights: Dict[str, float]):
        """
        Args:
            base_model: The base model to wrap
            steering_vectors: Dict {layer_idx: steering_vector_tensor}
            group_weights: Dict {group_name: weight} e.g., {"early": 0.5, "mid_early": 0.3, ...}
        """
        self.base_model = base_model
        self.steering_vectors = steering_vectors
        self.group_weights = group_weights
        self.hooks = []
        
        # Map layer index to group name
        self.layer_to_group = {}
        for group_name, layer_indices in LAYER_GROUPS.items():
            for layer_idx in layer_indices:
                self.layer_to_group[layer_idx] = group_name
        
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Register forward hooks to apply steering vectors with group-specific weights."""
        layers = self.base_model.model.layers
        
        def make_hook(layer_idx):
            """Create a hook function for a specific layer."""
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_state = output[0]
                    output_rest = output[1:]
                else:
                    hidden_state = output
                    output_rest = ()
                
                # Apply steering vector if it exists for this layer
                if layer_idx in self.steering_vectors:
                    # Get the weight for this layer's group
                    group_name = self.layer_to_group.get(layer_idx)
                    if group_name and group_name in self.group_weights:
                        weight = self.group_weights[group_name]
                        steering_vector = self.steering_vectors[layer_idx]
                        steering_vector = steering_vector.to(hidden_state.device).to(hidden_state.dtype)
                        hidden_state = hidden_state + weight * steering_vector.unsqueeze(0).unsqueeze(0)
                
                if isinstance(output, tuple):
                    return (hidden_state,) + output_rest
                else:
                    return hidden_state
            
            return hook_fn
        
        for layer_idx, layer in enumerate(layers):
            hook = layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    @property
    def device(self):
        return self.base_model.device
    
    def eval(self):
        self.base_model.eval()
        return self
    
    def train(self, mode=True):
        self.base_model.train(mode)
        return self
    
    def __getattr__(self, name):
        return getattr(self.base_model, name)
    
    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)


def generate_local(model, tokenizer, prompt_ids, prompt_mask, args):
    """Generate using local model - same as eval_with_steering.py"""
    # Get device - handle both single device and device_map="auto" cases
    if hasattr(model, 'device'):
        device = model.device
    else:
        # For models with device_map="auto", get device from first parameter
        device = next(model.parameters()).device
    
    prompt_ids = prompt_ids.repeat(args.num_chains, 1).to(device)
    prompt_mask = prompt_mask.repeat(args.num_chains, 1).to(device)
    
    generation_config = {
        "max_new_tokens": args.max_completion_length,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    with torch.inference_mode():
        prompt_completion_ids = model.generate(prompt_ids, attention_mask=prompt_mask, **generation_config)
    
    prompt_len = prompt_ids.size(1)
    prompt_ids = prompt_completion_ids[:, :prompt_len]
    completion_ids = prompt_completion_ids[:, prompt_len:]
    
    is_eos = completion_ids == tokenizer.eos_token_id
    # Reuse device from earlier in function
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    seq_idx = torch.arange(is_eos.size(1), device=device).expand_as(is_eos)
    completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()
    
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completion_mask, completions_text


def compute_pass_at_k(n, c, k):
    """Calculate pass@k metric."""
    if n - c < k:
        return 1.0
    prob_all_wrong = 1.0
    for i in range(k):
        prob_all_wrong *= (n - c - i) / (n - i)
    return 1.0 - prob_all_wrong


def evaluate_configuration(
    group_weights: Dict[str, float],
    base_model,
    tokenizer,
    steering_vectors,
    eval_ds,
    system_prompt,
    args,
    eval_size: int,
    device: str
) -> float:
    """
    Evaluate a configuration of group weights.
    
    Returns:
        pass_at_1 score (to maximize)
    """
    # Move steering vectors to GPU
    steering_vectors_gpu = {k: v.to(device) for k, v in steering_vectors.items()}
    
    # Create wrapped model
    wrapped_model = GroupedSteeringModelWrapper(base_model, steering_vectors_gpu, group_weights)
    wrapped_model.eval()
    
    pass_at_k_scores = []
    
    with torch.no_grad():
        for i, eval_entry in enumerate(eval_ds):
            if i >= eval_size:
                break
            q = format_math_problem(eval_entry)
            prompt_text_eval, prompt_ids_eval, prompt_mask_eval = utils.format_prompt(system_prompt, q, tokenizer)
            
            _, _, _, _, _, completions_text_eval = generate_local(
                wrapped_model, tokenizer, prompt_ids_eval, prompt_mask_eval, args
            )
            
            extracted_answers_eval = [utils.extract_answer(t) for t in completions_text_eval]
            format_rewards_eval = [utils.check_format(t) for t in completions_text_eval]
            correctness_eval = []
            for ea, f in zip(extracted_answers_eval, format_rewards_eval):
                if f < 0:
                    correctness_eval.append(0.0)
                elif ea:
                    correctness_eval.append(float(eval_ds.score_answer(answer=ea, entry=eval_entry) == 1.0))
                else:
                    correctness_eval.append(0.0)
            
            num_correct = sum(correctness_eval)
            pass_at_k = compute_pass_at_k(
                n=args.num_completions_eval,
                c=int(num_correct),
                k=args.pass_at_k
            )
            pass_at_k_scores.append(pass_at_k)
    
    # Clean up
    wrapped_model.remove_hooks()
    del wrapped_model
    del steering_vectors_gpu
    torch.cuda.empty_cache()
    
    avg_pass_at_k = (sum(pass_at_k_scores) / max(len(pass_at_k_scores), 1)) * 100
    return avg_pass_at_k


# Global variables for optimization callback
optimization_history = []
best_score = -float('inf')
best_weights = None


def create_visualizations(
    history: List[Dict],
    output_dir: Path,
    iteration: int
):
    """Create visualization plots for current optimization state."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(history) < 2:
        return
    
    # Extract data
    weights_array = np.array([h['weights'] for h in history])
    scores = np.array([h['score'] for h in history])
    
    # 1. Four separate 1D plots (one per group)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Bayesian Optimization Progress (Iteration {iteration})', fontsize=16, fontweight='bold')
    
    for idx, group_name in enumerate(GROUP_NAMES):
        ax = axes[idx // 2, idx % 2]
        group_weights = weights_array[:, idx]
        
        # Scatter plot of measurements
        scatter = ax.scatter(group_weights, scores, c=scores, cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
        ax.set_xlabel(f'{group_name.replace("_", " ").title()} Weight', fontweight='bold')
        ax.set_ylabel('Pass@1 (%)', fontweight='bold')
        ax.set_title(f'{group_name.replace("_", " ").title()} Layer Group', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Pass@1 (%)')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'1d_plots_iter_{iteration:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 2D grid plots (all pairwise combinations)
    num_groups = len(GROUP_NAMES)
    fig, axes = plt.subplots(num_groups, num_groups, figsize=(16, 16))
    fig.suptitle(f'2D Grid Plots - Bayesian Optimization (Iteration {iteration})', fontsize=16, fontweight='bold')
    
    for i in range(num_groups):
        for j in range(num_groups):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: show 1D distribution
                group_weights = weights_array[:, i]
                ax.hist(group_weights, bins=20, alpha=0.6, edgecolor='black')
                ax.set_xlabel(f'{GROUP_NAMES[i].replace("_", " ").title()} Weight')
                ax.set_ylabel('Frequency')
            else:
                # Off-diagonal: 2D scatter with color-coded scores
                x_weights = weights_array[:, j]
                y_weights = weights_array[:, i]
                scatter = ax.scatter(x_weights, y_weights, c=scores, cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
                ax.set_xlabel(f'{GROUP_NAMES[j].replace("_", " ").title()} Weight')
                ax.set_ylabel(f'{GROUP_NAMES[i].replace("_", " ").title()} Weight')
            
            if i == 0:
                ax.set_title(f'{GROUP_NAMES[j].replace("_", " ").title()}', fontweight='bold')
            if j == 0:
                ax.text(-0.1, 0.5, f'{GROUP_NAMES[i].replace("_", " ").title()}', 
                       transform=ax.transAxes, rotation=90, va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'2d_grids_iter_{iteration:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Main optimization plot (score over iterations with uncertainty)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    iterations = np.arange(1, len(history) + 1)
    scores_array = np.array(scores)
    
    # Plot scores
    ax.plot(iterations, scores_array, 'o-', color='#FF1744', linewidth=2, markersize=6, label='Measurements')
    
    # Plot best so far
    best_so_far = np.maximum.accumulate(scores_array)
    ax.plot(iterations, best_so_far, '--', color='#228B22', linewidth=2, label='Best So Far')
    
    # Add uncertainty visualization (simple moving std)
    if len(scores_array) > 5:
        window = min(5, len(scores_array) // 2)
        moving_mean = np.convolve(scores_array, np.ones(window)/window, mode='valid')
        moving_std = np.array([np.std(scores_array[max(0, i-window):i+1]) for i in range(len(scores_array))])
        ax.fill_between(iterations, scores_array - moving_std, scores_array + moving_std, 
                       alpha=0.2, color='#FF1744', label='Uncertainty')
    
    ax.set_xlabel('Iteration', fontweight='bold', fontsize=12)
    ax.set_ylabel('Pass@1 (%)', fontweight='bold', fontsize=12)
    ax.set_title(f'Bayesian Optimization Progress (Round {iteration})', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'optimization_progress_iter_{iteration:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Bayesian optimization for steering vector weights")
    
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model name from HuggingFace")
    parser.add_argument("--finetuned_checkpoint", type=str, default="best_model",
                        help="Path to finetuned checkpoint directory")
    parser.add_argument("--steering_vectors_path", type=str, default="steering_vectors.json",
                        help="Path to steering vectors JSON file")
    parser.add_argument("--output_dir", type=str, default="bayesian_opt_results",
                        help="Output directory for results and plots")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to previous results JSON to resume from")
    
    # Evaluation settings
    parser.add_argument("--eval-size", type=int, default=20,
                        help="Number of eval examples to use")
    parser.add_argument("--num_completions_eval", type=int, default=20,
                        help="Number of completions to sample per eval problem")
    parser.add_argument("--pass_at_k", type=int, default=1,
                        help="k for pass@k metric")
    parser.add_argument("--max_completion_length", type=int, default=512,
                        help="Max completion tokens")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=7111994,
                        help="Random seed")
    
    # Optimization settings
    parser.add_argument("--n_iterations", type=int, default=50,
                        help="Number of Bayesian optimization iterations")
    parser.add_argument("--n_initial_points", type=int, default=5,
                        help="Number of random initial points")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Seed for reproducibility
    utils.seed_everything(args.seed)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Using device: {device}")
    print(f"Output directory: {output_dir}")
    
    # Set num_chains for generation (needed by generate_local)
    args.num_chains = args.num_completions_eval
    
    # System prompt (matching day5 format - check eval_with_steering.py for exact format)
    system_prompt = (
        "Think first and reason step by step. Put your reasoning within <think></think> tags. "
        "Then put your final answer within <answer></answer> tags. "
        "You must use both tags in this exact order: first <think>your reasoning</think>, then <answer>your answer</answer>."
        f"Note: Your reasoning may be cut off if it gets too long, but answer as best as you can if that happens."
    )
    
    # Load eval dataset
    print("Loading eval dataset...")
    _, eval_ds = load_math_dataset(
        train_size=0,
        eval_size=args.eval_size,
        seed=args.seed
    )
    
    # Load steering vectors
    print(f"Loading steering vectors from {args.steering_vectors_path}...")
    steering_vectors = load_steering_vectors(args.steering_vectors_path)
    print(f"Loaded steering vectors for {len(steering_vectors)} layers")
    
    # Load base model (keep in memory for all evaluations)
    print("Loading base model...")
    base_model, tokenizer = llms.get_llm_tokenizer(args.base_model_name, use_liger_model=False)
    base_model.eval()
    
    # Evaluate baseline and finetuned models
    print("\n" + "="*70)
    print("Evaluating BASELINE model...")
    print("="*70)
    baseline_group_weights = {group: 0.0 for group in GROUP_NAMES}
    baseline_score = evaluate_configuration(
        baseline_group_weights, base_model, tokenizer, steering_vectors,
        eval_ds, system_prompt, args, args.eval_size, device
    )
    print(f"Baseline Pass@{args.pass_at_k}: {baseline_score:.2f}%")
    
    print("\n" + "="*70)
    print("Evaluating FINETUNED model...")
    print("="*70)
    finetuned_model, _ = llms.get_llm_tokenizer(args.finetuned_checkpoint, use_liger_model=False)
    finetuned_model.eval()
    
    # Evaluate finetuned model directly (same evaluation logic)
    # Note: Don't move model - it's already on device(s) via device_map="auto"
    pass_at_k_scores = []
    
    with torch.no_grad():
        for i, eval_entry in enumerate(eval_ds):
            if i >= args.eval_size:
                break
            q = format_math_problem(eval_entry)
            prompt_text_eval, prompt_ids_eval, prompt_mask_eval = utils.format_prompt(system_prompt, q, tokenizer)
            
            _, _, _, _, _, completions_text_eval = generate_local(
                finetuned_model, tokenizer, prompt_ids_eval, prompt_mask_eval, args
            )
            
            extracted_answers_eval = [utils.extract_answer(t) for t in completions_text_eval]
            format_rewards_eval = [utils.check_format(t) for t in completions_text_eval]
            correctness_eval = []
            for ea, f in zip(extracted_answers_eval, format_rewards_eval):
                if f < 0:
                    correctness_eval.append(0.0)
                elif ea:
                    correctness_eval.append(float(eval_ds.score_answer(answer=ea, entry=eval_entry) == 1.0))
                else:
                    correctness_eval.append(0.0)
            
            num_correct = sum(correctness_eval)
            pass_at_k = compute_pass_at_k(
                n=args.num_completions_eval,
                c=int(num_correct),
                k=args.pass_at_k
            )
            pass_at_k_scores.append(pass_at_k)
    
    finetuned_score = (sum(pass_at_k_scores) / max(len(pass_at_k_scores), 1)) * 100
    print(f"Finetuned Pass@{args.pass_at_k}: {finetuned_score:.2f}%")
    del finetuned_model
    torch.cuda.empty_cache()
    
    # Load previous results if resuming
    if args.resume_from and Path(args.resume_from).exists():
        print(f"Resuming from {args.resume_from}...")
        with open(args.resume_from, 'r') as f:
            resume_data = json.load(f)
        optimization_history = resume_data.get('history', [])
        start_iter = len(optimization_history)
        print(f"Resuming from iteration {start_iter}")
    else:
        optimization_history = []
        start_iter = 0
    
    # Define search space (4 dimensions, one per group)
    dimensions = [Real(0.0, 1.0, name=f'weight_{group}') for group in GROUP_NAMES]
    
    # Objective function
    @use_named_args(dimensions=dimensions)
    def objective(**kwargs):
        # Convert kwargs to group_weights dict
        group_weights = {group: kwargs[f'weight_{group}'] for group in GROUP_NAMES}
        
        # Round to nearest 0.1 for cleaner results
        group_weights = {k: round(v, 1) for k, v in group_weights.items()}
        
        print(f"\nEvaluating weights: {group_weights}")
        
        # Evaluate
        score = evaluate_configuration(
            group_weights, base_model, tokenizer, steering_vectors,
            eval_ds, system_prompt, args, args.eval_size, device
        )
        
        print(f"  Score: {score:.2f}%")
        
        # Store in history
        optimization_history.append({
            'weights': [group_weights[g] for g in GROUP_NAMES],
            'score': score,
            'group_weights': group_weights
        })
        
        # Create visualizations
        create_visualizations(optimization_history, plots_dir, len(optimization_history))
        
        # Save intermediate results
        results = {
            'baseline_score': baseline_score,
            'finetuned_score': finetuned_score,
            'history': optimization_history,
            'best': None
        }
        
        if optimization_history:
            best_idx = np.argmax([h['score'] for h in optimization_history])
            results['best'] = {
                'weights': optimization_history[best_idx]['group_weights'],
                'score': optimization_history[best_idx]['score']
            }
        
        with open(output_dir / 'optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Return negative score (we're minimizing, but want to maximize pass@1)
        return -score
    
    # Run optimization
    print("\n" + "="*70)
    print("Starting Bayesian Optimization...")
    print("="*70)
    
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=args.n_iterations - start_iter,
        n_initial_points=args.n_initial_points,
        acq_func='EI',  # Expected Improvement
        random_state=args.seed
    )
    
    # Find best configuration
    best_idx = np.argmax([h['score'] for h in optimization_history])
    best_config = optimization_history[best_idx]
    
    # Final results
    final_results = {
        'baseline_score': baseline_score,
        'finetuned_score': finetuned_score,
        'best_weights': best_config['group_weights'],
        'best_score': best_config['score'],
        'optimization_history': optimization_history
    }
    
    # Save final results
    with open(output_dir / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "="*70)
    print("Optimization Complete!")
    print("="*70)
    print(f"Baseline: {baseline_score:.2f}%")
    print(f"Finetuned: {finetuned_score:.2f}%")
    print(f"Best Steering: {best_config['score']:.2f}%")
    print(f"Best Weights: {best_config['group_weights']}")
    print(f"\nResults saved to: {output_dir}")
    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()

