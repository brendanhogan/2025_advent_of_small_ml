"""
Build steering vectors by comparing base model and finetuned model activations.

This script:
1. Loads the base model and finetuned checkpoint
2. Runs the entire training set through both models
3. Extracts activations from each transformer layer output
4. Computes steering vectors = mean(finetuned_activations) - mean(base_activations)
5. Saves steering vectors to disk
"""

import os
import json
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict

import llms
import utils
from math_dataset import load_math_dataset, format_math_problem


def setup_activation_hooks(model, activations_dict):
    """
    Register forward hooks to capture activations after each transformer layer.
    
    Args:
        model: The model to hook
        activations_dict: Dictionary to store activations {layer_idx: [list of activations]}
    """
    hooks = []
    
    # Get all transformer layers
    # For Qwen2, layers are in model.model.layers
    layers = model.model.layers
    num_layers = len(layers)
    print(f"Setting up hooks for {num_layers} transformer layers")
    
    def make_hook(layer_idx):
        """Create a hook function for a specific layer."""
        def hook_fn(module, input, output):
            # output is typically a tuple, the first element is the hidden state
            # After the full transformer block, output[0] is the hidden state
            hidden_state = output[0]  # Shape: (batch_size, seq_len, hidden_dim)
            # Move to CPU and convert to float32 to save memory
            activations_dict[layer_idx].append(hidden_state.detach().cpu().float())
        return hook_fn
    
    # Register hooks for each layer
    for layer_idx, layer in enumerate(layers):
        hook = layer.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)
    
    return hooks


def remove_hooks(hooks):
    """Remove all registered hooks."""
    for hook in hooks:
        hook.remove()


def compute_steering_vectors(base_activations, finetuned_activations):
    """
    Compute steering vectors from collected activations.
    
    Steering vector per layer = mean(finetuned_activations) - mean(base_activations)
    
    Args:
        base_activations: Dict {layer_idx: list of tensors}
        finetuned_activations: Dict {layer_idx: list of tensors}
    
    Returns:
        Dict {layer_idx: steering_vector tensor}
    """
    steering_vectors = {}
    
    # Process each layer
    for layer_idx in base_activations.keys():
        # Concatenate all activations for this layer
        base_all = torch.cat(base_activations[layer_idx], dim=0)  # (total_tokens, hidden_dim)
        finetuned_all = torch.cat(finetuned_activations[layer_idx], dim=0)  # (total_tokens, hidden_dim)
        
        # Compute means
        base_mean = base_all.mean(dim=0)  # (hidden_dim,)
        finetuned_mean = finetuned_all.mean(dim=0)  # (hidden_dim,)
        
        # Steering vector = difference
        steering_vector = finetuned_mean - base_mean  # (hidden_dim,)
        
        steering_vectors[layer_idx] = steering_vector
        
        print(f"Layer {layer_idx}: steering vector shape {steering_vector.shape}, "
              f"mean magnitude: {steering_vector.abs().mean().item():.4f}")
    
    return steering_vectors


def collect_activations(model, tokenizer, train_ds, system_prompt, device, max_examples=None):
    """
    Run training set through model and collect activations from each layer.
    
    Args:
        model: The model to run
        tokenizer: Tokenizer
        train_ds: Training dataset
        system_prompt: System prompt to use
        device: Device to run on
        max_examples: Maximum number of examples to process (None = all)
    
    Returns:
        Dict {layer_idx: list of activation tensors}
    """
    model.eval()
    activations = defaultdict(list)
    
    # Setup hooks
    hooks = setup_activation_hooks(model, activations)
    
    try:
        # Process training examples
        train_list = list(train_ds)
        if max_examples is not None:
            train_list = train_list[:max_examples]
        
        print(f"Processing {len(train_list)} training examples...")
        
        with torch.no_grad():
            for entry in tqdm(train_list, desc="Collecting activations"):
                # Format prompt
                question = format_math_problem(entry)
                prompt_text, prompt_ids, prompt_mask = utils.format_prompt(system_prompt, question, tokenizer)
                
                # Move to device
                prompt_ids = prompt_ids.to(device)
                prompt_mask = prompt_mask.to(device)
                
                # Forward pass (this will trigger hooks and collect activations)
                _ = model(input_ids=prompt_ids, attention_mask=prompt_mask)
                
                # Clear cache periodically
                if len(activations[0]) % 100 == 0:
                    torch.cuda.empty_cache()
        
        print(f"Collected activations from {len(activations[0])} examples")
        
    finally:
        # Always remove hooks
        remove_hooks(hooks)
    
    return activations


def save_steering_vectors(steering_vectors, output_path):
    """Save steering vectors to disk."""
    # Convert to CPU and numpy for saving
    steering_dict = {}
    for layer_idx, vector in steering_vectors.items():
        steering_dict[str(layer_idx)] = vector.cpu().numpy().tolist()
    
    with open(output_path, 'w') as f:
        json.dump(steering_dict, f, indent=2)
    
    print(f"Saved steering vectors to {output_path}")


def load_steering_vectors(input_path):
    """Load steering vectors from disk."""
    with open(input_path, 'r') as f:
        steering_dict = json.load(f)
    
    # Convert back to tensors
    steering_vectors = {}
    for layer_idx_str, vector_list in steering_dict.items():
        layer_idx = int(layer_idx_str)
        steering_vectors[layer_idx] = torch.tensor(vector_list)
    
    return steering_vectors


def parse_args():
    parser = argparse.ArgumentParser(description="Build steering vectors from base and finetuned models")
    
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model name from HuggingFace")
    parser.add_argument("--finetuned_checkpoint", type=str, default="final_run/checkpoint_step_250",
                        help="Path to finetuned checkpoint directory")
    parser.add_argument("--output_path", type=str, default="steering_vectors.json",
                        help="Path to save steering vectors JSON file")
    parser.add_argument("--train-size", type=int, default=1000,
                        help="Number of training examples to use")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum examples to process (None = all)")
    parser.add_argument("--seed", type=int, default=7111994,
                        help="Random seed")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Seed for reproducibility
    utils.seed_everything(args.seed)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # System prompt (same as training)
    system_prompt = (
        "Think first and reason step by step. Put your reasoning within <think></think> tags. "
        "Then put your final answer within <answer></answer> tags. "
        "You must use both tags in this exact order: first <think>your reasoning</think>, then <answer>your answer</answer>."
        "Note: Your reasoning may be cut off if it gets too long, but answer as best as you can if that happens."
    )
    
    # Load training dataset
    print("Loading training dataset...")
    train_ds, _ = load_math_dataset(
        train_size=args.train_size,
        eval_size=0,  # Don't need eval set here
        seed=args.seed
    )
    
    # Load base model
    print(f"\nLoading base model: {args.base_model_name}")
    base_model, tokenizer = llms.get_llm_tokenizer(args.base_model_name, use_liger_model=False)
    base_model.eval()
    
    # Load finetuned model
    print(f"Loading finetuned model: {args.finetuned_checkpoint}")
    finetuned_model, _ = llms.get_llm_tokenizer(args.finetuned_checkpoint, use_liger_model=False)
    finetuned_model.eval()
    
    # Collect activations from base model
    print("\n" + "="*70)
    print("Collecting activations from BASE model...")
    print("="*70)
    base_activations = collect_activations(
        base_model, tokenizer, train_ds, system_prompt, device, args.max_examples
    )
    
    # Clear memory
    del base_model
    torch.cuda.empty_cache()
    
    # Collect activations from finetuned model
    print("\n" + "="*70)
    print("Collecting activations from FINETUNED model...")
    print("="*70)
    finetuned_activations = collect_activations(
        finetuned_model, tokenizer, train_ds, system_prompt, device, args.max_examples
    )
    
    # Clear memory
    del finetuned_model
    torch.cuda.empty_cache()
    
    # Compute steering vectors
    print("\n" + "="*70)
    print("Computing steering vectors...")
    print("="*70)
    steering_vectors = compute_steering_vectors(base_activations, finetuned_activations)
    
    # Save steering vectors
    print("\n" + "="*70)
    print("Saving steering vectors...")
    print("="*70)
    save_steering_vectors(steering_vectors, args.output_path)
    
    print("\nDone! Steering vectors saved to:", args.output_path)

