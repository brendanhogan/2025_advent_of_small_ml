"""
Build creativity steering vectors by comparing generated abstracts vs real award-winning abstracts.

This script:
1. Loads abstracts from abstracts.json
2. Splits into 80% train / 20% test
3. For each training abstract:
   - Generates an abstract using the model (prompt: "Generate an abstract that would win best paper...")
   - Collects activations from passing the generated abstract through the model
   - Collects activations from passing the real abstract through the model
   - Computes steering vector = real_activations - generated_activations
4. Averages all steering vectors to get a general "creativity vector"
5. Saves the creativity vector to disk
"""

import os
import json
import torch
import argparse
import random
from tqdm import tqdm
from collections import defaultdict
from transformers import GenerationConfig

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "day4"))
import llms
import utils


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


def generate_abstract(model, tokenizer, device, max_new_tokens=512, temperature=0.9):
    """
    Generate an abstract that would win best paper at a major AI conference.
    
    Returns:
        Generated abstract text
    """
    prompt = """Generate an abstract for a research paper that would win the best paper award at a major AI conference (ICML, ICLR, or NeurIPS). The abstract should be:
- Technically rigorous and novel
- Well-written and clear
- Demonstrate significant contributions
- Be around 200-300 words

Abstract:"""
    
    # Format for Qwen 2.5 Instruct
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)
    
    # Extract only the generated part
    prompt_length = inputs["input_ids"].size(1)
    generated_ids = outputs[0, prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text.strip()


def collect_activations_for_text(model, tokenizer, text, device, activations_dict):
    """
    Pass text through model and collect activations.
    
    Args:
        model: The model
        tokenizer: Tokenizer
        text: Text to pass through
        device: Device
        activations_dict: Dict to store activations {layer_idx: [list of activations]}
    """
    # Format as a simple prompt (we just want to see how the model processes this text)
    # Use a simple format - just tokenize the text directly
    # Increase max_length to handle longer abstracts
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    
    with torch.no_grad():
        _ = model(**inputs)
    
    # Activations are collected via hooks


def compute_steering_vectors(generated_activations, real_activations):
    """
    Compute steering vectors from collected activations.
    
    Steering vector per layer = mean(real_activations) - mean(generated_activations)
    
    Args:
        generated_activations: Dict {layer_idx: list of tensors}
        real_activations: Dict {layer_idx: list of tensors}
    
    Returns:
        Dict {layer_idx: steering_vector tensor}
    """
    steering_vectors = {}
    
    # Process each layer
    for layer_idx in real_activations.keys():
        if layer_idx not in generated_activations or len(generated_activations[layer_idx]) == 0:
            print(f"Warning: No generated activations for layer {layer_idx}, skipping")
            continue
        
        # Concatenate all activations for this layer
        # Each activation is (batch_size, seq_len, hidden_dim), typically (1, seq_len, hidden_dim)
        # Concatenating along dim=0 gives (n_examples, seq_len, hidden_dim)
        generated_all = torch.cat(generated_activations[layer_idx], dim=0)  # (n_examples, seq_len, hidden_dim)
        real_all = torch.cat(real_activations[layer_idx], dim=0)  # (n_examples, seq_len, hidden_dim)
        
        # Flatten batch and sequence dimensions, then compute mean across all tokens
        # This gives us a single vector per layer representing average activation
        generated_flat = generated_all.view(-1, generated_all.shape[-1])  # (total_tokens, hidden_dim)
        real_flat = real_all.view(-1, real_all.shape[-1])  # (total_tokens, hidden_dim)
        
        # Compute means across all tokens
        generated_mean = generated_flat.mean(dim=0)  # (hidden_dim,)
        real_mean = real_flat.mean(dim=0)  # (hidden_dim,)
        
        # Steering vector = difference (real - generated)
        steering_vector = real_mean - generated_mean  # (hidden_dim,)
        
        steering_vectors[layer_idx] = steering_vector
        
        print(f"Layer {layer_idx}: steering vector shape {steering_vector.shape}, "
              f"mean magnitude: {steering_vector.abs().mean().item():.4f}")
    
    return steering_vectors


def average_steering_vectors(all_steering_vectors):
    """
    Average multiple steering vectors (one per training example) into a single creativity vector.
    
    Args:
        all_steering_vectors: List of dicts, each {layer_idx: steering_vector tensor}
    
    Returns:
        Dict {layer_idx: averaged_steering_vector tensor}
    """
    if len(all_steering_vectors) == 0:
        raise ValueError("No steering vectors to average")
    
    # Get all layer indices
    layer_indices = set(all_steering_vectors[0].keys())
    
    averaged = {}
    for layer_idx in layer_indices:
        # Collect all vectors for this layer
        vectors = [sv[layer_idx] for sv in all_steering_vectors if layer_idx in sv]
        
        if len(vectors) == 0:
            continue
        
        # Stack and average
        stacked = torch.stack(vectors, dim=0)  # (n_examples, hidden_dim)
        averaged[layer_idx] = stacked.mean(dim=0)  # (hidden_dim,)
        
        print(f"Layer {layer_idx}: averaged from {len(vectors)} examples, "
              f"mean magnitude: {averaged[layer_idx].abs().mean().item():.4f}")
    
    return averaged


def save_creativity_vector(creativity_vector, output_path):
    """Save creativity vector to disk."""
    # Convert to CPU and numpy for saving
    vector_dict = {}
    for layer_idx, vector in creativity_vector.items():
        vector_dict[str(layer_idx)] = vector.cpu().numpy().tolist()
    
    with open(output_path, 'w') as f:
        json.dump(vector_dict, f, indent=2)
    
    print(f"Saved creativity vector to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build creativity steering vectors")
    parser.add_argument(
        "--abstracts_path",
        type=str,
        default="abstracts.json",
        help="Path to abstracts JSON file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name from HuggingFace"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="creativity_vector.json",
        help="Path to save creativity vector"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of training examples to process (None = all)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max tokens to generate for abstract"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Temperature for generation"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    utils.seed_everything(args.seed)
    
    # Load abstracts
    print(f"Loading abstracts from {args.abstracts_path}...")
    with open(args.abstracts_path, 'r') as f:
        data = json.load(f)
    
    abstracts = data["abstracts"]
    print(f"Loaded {len(abstracts)} abstracts")
    
    # Split train/test
    random.shuffle(abstracts)
    split_idx = int(len(abstracts) * args.train_split)
    train_abstracts = abstracts[:split_idx]
    test_abstracts = abstracts[split_idx:]
    
    print(f"Train: {len(train_abstracts)} abstracts")
    print(f"Test: {len(test_abstracts)} abstracts")
    
    # Save test set for later evaluation
    test_path = args.output_path.replace(".json", "_test_set.json")
    with open(test_path, 'w') as f:
        json.dump({"abstracts": test_abstracts}, f, indent=2)
    print(f"Saved test set to {test_path}")
    
    # Limit training examples if specified
    if args.max_examples is not None:
        train_abstracts = train_abstracts[:args.max_examples]
        print(f"Limited to {len(train_abstracts)} training examples")
    
    # Load model
    print("\n" + "="*70)
    print("Loading model...")
    print("="*70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, use_liger_model=False)
    model.eval()
    
    # Process each training abstract
    all_steering_vectors = []
    
    print("\n" + "="*70)
    print(f"Processing {len(train_abstracts)} training abstracts...")
    print("="*70)
    
    for idx, abstract_entry in enumerate(tqdm(train_abstracts, desc="Processing abstracts")):
        title = abstract_entry["title"]
        real_abstract = abstract_entry["abstract"]
        
        # Generate abstract
        print(f"\n[{idx+1}/{len(train_abstracts)}] Generating abstract...")
        generated_abstract = generate_abstract(
            model, tokenizer, device, 
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        print(f"Generated abstract (first 200 chars): {generated_abstract[:200]}...")
        
        # Format both texts consistently for comparison
        # Use a simple format: "Title\n\nAbstract" for both
        generated_text = f"Generated Research Paper\n\n{generated_abstract}"
        real_text = f"{title}\n\n{real_abstract}"
        
        # Collect activations for generated abstract
        print("Collecting activations for generated abstract...")
        generated_activations = defaultdict(list)
        hooks_gen = setup_activation_hooks(model, generated_activations)
        try:
            collect_activations_for_text(model, tokenizer, generated_text, device, generated_activations)
        finally:
            remove_hooks(hooks_gen)
        
        # Collect activations for real abstract
        print("Collecting activations for real abstract...")
        real_activations = defaultdict(list)
        hooks_real = setup_activation_hooks(model, real_activations)
        try:
            collect_activations_for_text(model, tokenizer, real_text, device, real_activations)
        finally:
            remove_hooks(hooks_real)
        
        # Compute steering vector for this example
        steering_vectors = compute_steering_vectors(generated_activations, real_activations)
        all_steering_vectors.append(steering_vectors)
        
        # Clear cache periodically
        if (idx + 1) % 5 == 0:
            torch.cuda.empty_cache()
            print(f"Processed {idx+1} examples, cleared cache")
    
    # Average all steering vectors to get creativity vector
    print("\n" + "="*70)
    print("Averaging steering vectors to create creativity vector...")
    print("="*70)
    creativity_vector = average_steering_vectors(all_steering_vectors)
    
    # Save creativity vector
    print("\n" + "="*70)
    print("Saving creativity vector...")
    print("="*70)
    save_creativity_vector(creativity_vector, args.output_path)
    
    print("\nDone! Creativity vector saved to:", args.output_path)
    print(f"Test set saved to: {test_path}")


if __name__ == "__main__":
    main()

