"""
Evaluate models with steering vectors applied.

This script:
1. Loads base model, finetuned model, and steering vectors
2. Evaluates on eval set with pass@20 (same settings as main.py)
3. Tests: base model, finetuned model, base + steering at weights [0.25, 0.5, 0.75, 1.0]
4. Saves all results in JSON
"""

import os
import json
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict

import llms
import utils
from math_dataset import load_math_dataset, format_math_problem, extract_math_answer
from build_steering_vectors import load_steering_vectors


class SteeringModelWrapper:
    """
    Wrapper around a model that applies steering vectors during forward pass.
    
    This adds steering vectors to the hidden states after each transformer layer.
    """
    
    def __init__(self, base_model, steering_vectors, steering_weight=1.0):
        """
        Args:
            base_model: The base model to wrap
            steering_vectors: Dict {layer_idx: steering_vector_tensor}
            steering_weight: Weight to apply to steering vectors (0.0 to 1.0)
        """
        self.base_model = base_model
        self.steering_vectors = steering_vectors
        self.steering_weight = steering_weight
        self.hooks = []
        
        # Register hooks to apply steering vectors
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Register forward hooks to apply steering vectors."""
        layers = self.base_model.model.layers
        
        def make_hook(layer_idx):
            """Create a hook function for a specific layer."""
            def hook_fn(module, input, output):
                # Handle output - it might be a tuple or a single tensor
                if isinstance(output, tuple):
                    hidden_state = output[0]  # (batch_size, seq_len, hidden_dim)
                    output_rest = output[1:]
                else:
                    # If output is a single tensor, treat it as the hidden state
                    hidden_state = output
                    output_rest = ()
                
                # Apply steering vector if it exists for this layer
                if layer_idx in self.steering_vectors:
                    steering_vector = self.steering_vectors[layer_idx]
                    # Move steering vector to same device and dtype as hidden state
                    steering_vector = steering_vector.to(hidden_state.device).to(hidden_state.dtype)
                    # Add steering vector to hidden state (broadcast across batch and sequence)
                    hidden_state = hidden_state + self.steering_weight * steering_vector.unsqueeze(0).unsqueeze(0)
                
                # Return modified output in the same format as input
                if isinstance(output, tuple):
                    return (hidden_state,) + output_rest
                else:
                    return hidden_state
            
            return hook_fn
        
        # Register hooks for each layer
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
        """Return the device of the base model."""
        return self.base_model.device
    
    def eval(self):
        """Set base model to eval mode."""
        self.base_model.eval()
        return self
    
    def train(self, mode=True):
        """Set base model to train/eval mode."""
        self.base_model.train(mode)
        return self
    
    def __getattr__(self, name):
        """Delegate all other attributes to the base model."""
        return getattr(self.base_model, name)
    
    def generate(self, *args, **kwargs):
        """Delegate generate to base model."""
        return self.base_model.generate(*args, **kwargs)


def generate_local(model, tokenizer, prompt_ids, prompt_mask, args):
    """
    Generate using local model - EXACT COPY from main.py to ensure identical behavior.
    """
    # Repeat prompt for multiple parallel generations (chains)
    prompt_ids = prompt_ids.repeat(args.num_chains, 1).to(model.device)
    prompt_mask = prompt_mask.repeat(args.num_chains, 1).to(model.device)

    # Set up generation parameters (match main.py exactly)
    generation_config = {
        "max_new_tokens": args.max_completion_length,  # Max tokens to generate
        "do_sample": True,  # Enable sampling (not greedy)
        "temperature": args.temperature,  # Sampling temperature
        "top_p": 1.0,  # Match vLLM default (no top-p filtering)
        # Don't set top_k - transformers default is None (disabled)
        "repetition_penalty": 1.0,  # Match vLLM default (no repetition penalty)
        "pad_token_id": tokenizer.pad_token_id,  # Padding token for batching
    }
    # Note: seed is NOT added here - transformers uses torch's global RNG which is seeded via utils.seed_everything()
    # Generate completions (disable gradients for inference)
    with torch.inference_mode():
        prompt_completion_ids = model.generate(prompt_ids, attention_mask=prompt_mask, **generation_config)

    # Split the full sequence back into prompt and completion parts
    prompt_len = prompt_ids.size(1)  # Length of original prompt
    prompt_ids = prompt_completion_ids[:, :prompt_len]  # Extract prompt portion
    completion_ids = prompt_completion_ids[:, prompt_len:]  # Extract completion portion

    # Create mask to handle EOS tokens properly
    is_eos = completion_ids == tokenizer.eos_token_id  # Find EOS tokens
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=model.device)  # Default to end
    has_eos = is_eos.any(dim=1)  # Check which sequences have EOS
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]  # Set EOS position for sequences that have it
    seq_idx = torch.arange(is_eos.size(1), device=model.device).expand_as(is_eos)  # Position indices
    completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()  # Mask: 1 for valid tokens, 0 after EOS

    # Combine prompt and completion attention masks
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    # Decode token IDs back to text
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


def evaluate_model(model, tokenizer, eval_ds, system_prompt, args, eval_size):
    """
    Evaluate a model on the eval dataset - EXACT COPY of main.py eval logic.
    
    Returns:
        Dict with metrics: pass_at_k, avg_format_reward, num_eval_problems, examples
    """
    model.eval()  # Set model to eval mode
    pass_at_k_scores = []
    format_total = 0
    eval_count = 0
    eval_examples = []
    
    # Temporarily modify args for eval generation (exactly like main.py)
    original_num_chains = args.num_chains
    args.num_chains = args.num_completions_eval
    
    with torch.no_grad():  # Disable gradients during eval
        for i, eval_entry in enumerate(tqdm(eval_ds, desc="Evaluating")):
            if i >= eval_size:
                break
            q = format_math_problem(eval_entry)
            a = extract_math_answer(eval_entry)
            eval_problem_type = f"{eval_entry['subject']}_level_{eval_entry['level']}"
            prompt_text_eval, prompt_ids_eval, prompt_mask_eval = utils.format_prompt(system_prompt, q, tokenizer)
            
            # Generate multiple completions for this eval problem - EXACT SAME as main.py
            prompt_completion_ids_eval, prompt_ids_eval, completion_ids_eval, attention_mask_eval, completion_mask_eval, completions_text_eval = generate_local(
                model, tokenizer, prompt_ids_eval, prompt_mask_eval, args
            )
            
            # Score all completions - EXACT SAME as main.py
            extracted_answers_eval = [utils.extract_answer(t) for t in completions_text_eval]
            format_rewards_eval = [utils.check_format(t) for t in completions_text_eval]
            
            # Only score correctness if format is correct AND there's an extracted answer
            correctness_eval = []
            for ea, f in zip(extracted_answers_eval, format_rewards_eval):
                if f < 0:  # Wrong format
                    correctness_eval.append(0.0)
                elif ea:  # Has extracted answer
                    correctness_eval.append(float(eval_ds.score_answer(answer=ea, entry=eval_entry) == 1.0))
                else:  # No extracted answer
                    correctness_eval.append(0.0)
            
            # Compute pass@k for this problem - EXACT SAME as main.py
            num_correct = sum(correctness_eval)
            pass_at_k = compute_pass_at_k(
                n=args.num_completions_eval,
                c=int(num_correct),
                k=args.pass_at_k
            )
            pass_at_k_scores.append(pass_at_k)
            
            # Average format reward across completions
            avg_format_for_problem = sum(format_rewards_eval) / len(format_rewards_eval)
            format_total += avg_format_for_problem
            eval_count += 1
            
            # Log this eval example - EXACT SAME as main.py
            eval_examples.append({
                "prompt": prompt_text_eval,
                "question": q,
                "target_answer": a,
                "problem_type": eval_problem_type,
                "completions": [
                    {
                        "text": t,
                        "extracted_answer": ea,
                        "correct": int(c),
                        "format_reward": float(f)
                    } for t, ea, c, f in zip(completions_text_eval, extracted_answers_eval, correctness_eval, format_rewards_eval)
                ],
                "num_correct": int(num_correct),
                "pass_at_k": pass_at_k,
                "avg_format_reward": avg_format_for_problem,
            })
            
            # Explicitly delete large tensors to free memory immediately (after logging)
            del prompt_completion_ids_eval, prompt_ids_eval, completion_ids_eval, attention_mask_eval, completion_mask_eval
            
            # Clear cache more frequently to prevent OOM
            if i % 3 == 0:  # Clear every 3 problems instead of 5
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all operations complete before clearing
    
    # Restore original num_chains (exactly like main.py)
    args.num_chains = original_num_chains
    
    # Aggregate overall metrics - EXACT SAME as main.py
    avg_pass_at_k = (sum(pass_at_k_scores) / max(eval_count, 1)) * 100
    avg_format = (format_total / max(eval_count, 1))
    
    return {
        f"pass_at_{args.pass_at_k}": avg_pass_at_k,
        "avg_format_reward": avg_format,
        "num_eval_problems": eval_count,
        "examples": eval_examples,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate models with steering vectors")
    
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model name from HuggingFace")
    parser.add_argument("--finetuned_checkpoint", type=str, default="final_run/checkpoint_step_250",
                        help="Path to finetuned checkpoint directory")
    parser.add_argument("--steering_vectors_path", type=str, default="steering_vectors.json",
                        help="Path to steering vectors JSON file")
    parser.add_argument("--output_path", type=str, default="steering_eval_results.json",
                        help="Path to save evaluation results JSON file")
    
    # Evaluation settings (same as main.py)
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
    
    parsed_args = parser.parse_args()
    
    # Add num_chains attribute (needed for generate_local, will be overridden during eval)
    parsed_args.num_chains = parsed_args.num_completions_eval
    
    return parsed_args


if __name__ == "__main__":
    args = parse_args()
    
    # Seed for reproducibility
    utils.seed_everything(args.seed)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # System prompt (EXACT COPY from main.py)
    system_prompt = (
        "Think first and reason step by step. Put your reasoning within <think></think> tags. "
        "Then put your final answer within <answer></answer> tags. "
        "You must use both tags in this exact order: first <think>your reasoning</think>, then <answer>your answer</answer>."
        f"Note: Your reasoning may be cut off if it gets too long, but answer as best as you can if that happens."
    )
    
    # Load eval dataset
    print("Loading eval dataset...")
    print(f"Eval settings: size={args.eval_size}, num_completions={args.num_completions_eval}, "
          f"pass_at_k={args.pass_at_k}, temperature={args.temperature}, seed={args.seed}")
    _, eval_ds = load_math_dataset(
        train_size=0,  # Don't need training set
        eval_size=args.eval_size,
        seed=args.seed
    )
    
    # Load steering vectors
    print(f"\nLoading steering vectors from {args.steering_vectors_path}...")
    steering_vectors = load_steering_vectors(args.steering_vectors_path)
    print(f"Loaded steering vectors for {len(steering_vectors)} layers")
    
    # Results dictionary
    results = {}
    
    # Move steering vectors to CPU to save GPU memory
    steering_vectors_cpu = {k: v.cpu() for k, v in steering_vectors.items()}
    
    # # 1. Evaluate BASE model
    print("\n" + "="*70)
    print("Evaluating BASE model...")
    print("="*70)
    base_model, tokenizer = llms.get_llm_tokenizer(args.base_model_name, use_liger_model=False)
    base_model.eval()
    
    base_results = evaluate_model(
        base_model, tokenizer, eval_ds, system_prompt, args, args.eval_size
    )
    results["base_model"] = base_results
    print(f"Base model - Pass@{args.pass_at_k}: {base_results[f'pass_at_{args.pass_at_k}']:.2f}%, "
          f"Format: {base_results['avg_format_reward']:.3f}")
    
    # Clear base model from memory
    del base_model
    torch.cuda.empty_cache()
    
    # 2. Evaluate FINETUNED model
    print("\n" + "="*70)
    print("Evaluating FINETUNED model...")
    print("="*70)
    finetuned_model, tokenizer = llms.get_llm_tokenizer(args.finetuned_checkpoint, use_liger_model=False)
    finetuned_model.eval()
    
    finetuned_results = evaluate_model(
        finetuned_model, tokenizer, eval_ds, system_prompt, args, args.eval_size
    )
    results["finetuned_model"] = finetuned_results
    print(f"Finetuned model - Pass@{args.pass_at_k}: {finetuned_results[f'pass_at_{args.pass_at_k}']:.2f}%, "
          f"Format: {finetuned_results['avg_format_reward']:.3f}")
    # Clear finetuned model from memory
    del finetuned_model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # 3. Evaluate BASE + STEERING at different weights
    print("\n" + "="*70)
    print("Evaluating BASE + STEERING vectors...")
    print("="*70)
    
    steering_weights = [0.1, 0.15, 0.20, .25, .3, .35]
    results["base_with_steering"] = {}
    
    for weight in steering_weights:
        print(f"\nTesting steering weight: {weight}")
        
        # Reload base model for each steering weight test (to avoid memory issues)
        base_model, _ = llms.get_llm_tokenizer(args.base_model_name, use_liger_model=False)
        base_model.eval()
        
        # Move steering vectors back to GPU for this evaluation
        steering_vectors_gpu = {k: v.to(device) for k, v in steering_vectors_cpu.items()}
        
        # Create wrapped model with steering
        wrapped_model = SteeringModelWrapper(base_model, steering_vectors_gpu, steering_weight=weight)
        
        # Evaluate
        steering_results = evaluate_model(
            wrapped_model, tokenizer, eval_ds, system_prompt, args, args.eval_size
        )
        
        results["base_with_steering"][str(weight)] = steering_results
        print(f"  Pass@{args.pass_at_k}: {steering_results[f'pass_at_{args.pass_at_k}']:.2f}%, "
              f"Format: {steering_results['avg_format_reward']:.3f}")
        
        # Clean up: remove hooks, delete models, clear cache
        wrapped_model.remove_hooks()
        del wrapped_model
        del base_model
        del steering_vectors_gpu
        torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
    
    # Save results
    print("\n" + "="*70)
    print("Saving results...")
    print("="*70)
    
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output_path}")
    print("\nSummary:")
    if "base_model" in results:
        print(f"  Base model: Pass@{args.pass_at_k} = {results['base_model'][f'pass_at_{args.pass_at_k}']:.2f}%")
    print(f"  Finetuned model: Pass@{args.pass_at_k} = {results['finetuned_model'][f'pass_at_{args.pass_at_k}']:.2f}%")
    if "base_with_steering" in results:
        print(f"  Base + Steering:")
        for weight in steering_weights:
            pass_at_k_val = results["base_with_steering"][str(weight)][f"pass_at_{args.pass_at_k}"]
            print(f"    Weight {weight}: Pass@{args.pass_at_k} = {pass_at_k_val:.2f}%")

