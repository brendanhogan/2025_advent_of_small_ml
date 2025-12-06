"""
Evaluate creativity steering vectors with three-part evaluation:

1. Part 1: Generate 5 base + 5 steering abstracts (alternating), save to JSON
2. Part 2: Compare base vs steering against real test abstracts (GPT-4o judges)
3. Part 3: Compare base vs steering directly (GPT-4o judges)
"""

import os
import json
import torch
import argparse
import random
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import GenerationConfig
from openai import OpenAI
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "day4"))
import llms
import utils


class JudgmentResult(BaseModel):
    """Structured output for GPT-4o judgment."""
    winner: str  # "abstract_a" or "abstract_b"
    explanation: str  # Explanation of why this abstract is better


class SteeringModelWrapper:
    """
    Wrapper around a model that applies steering vectors during forward pass.
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
        self._apply_steering = False  # Flag to control whether to apply steering
        
        # Register hooks to apply steering vectors
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Register forward hooks to apply steering vectors."""
        layers = self.base_model.model.layers
        
        def make_hook(layer_idx):
            """Create a hook function for a specific layer."""
            def hook_fn(module, input, output):
                # Only apply steering if the flag is set (i.e., we're in a steering context)
                if not self._apply_steering:
                    return output
                
                if isinstance(output, tuple):
                    hidden_state = output[0]
                    output_rest = output[1:]
                else:
                    hidden_state = output
                    output_rest = ()
                
                # Apply steering vector if it exists for this layer
                if layer_idx in self.steering_vectors:
                    steering_vector = self.steering_vectors[layer_idx]
                    steering_vector = steering_vector.to(hidden_state.device).to(hidden_state.dtype)
                    hidden_state = hidden_state + self.steering_weight * steering_vector.unsqueeze(0).unsqueeze(0)
                
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
        return self.base_model.device
    
    def eval(self):
        self.base_model.eval()
        return self
    
    def __getattr__(self, name):
        return getattr(self.base_model, name)
    
    def generate(self, *args, **kwargs):
        """Generate with steering enabled."""
        self._apply_steering = True
        try:
            return self.base_model.generate(*args, **kwargs)
        finally:
            self._apply_steering = False


def load_creativity_vector(input_path):
    """Load creativity vector from disk."""
    with open(input_path, 'r') as f:
        vector_dict = json.load(f)
    
    # Convert back to tensors
    creativity_vector = {}
    for layer_idx_str, vector_list in vector_dict.items():
        layer_idx = int(layer_idx_str)
        creativity_vector[layer_idx] = torch.tensor(vector_list)
    
    return creativity_vector


def generate_abstract(model, tokenizer, device, seed=None, max_new_tokens=512, temperature=0.9):
    """
    Generate an abstract that would win best paper at a major AI conference.
    
    Args:
        model: The model to use
        tokenizer: Tokenizer
        device: Device
        seed: Random seed for generation
        max_new_tokens: Max tokens to generate
        temperature: Temperature for sampling
    
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
    
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
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


def judge_abstracts(client: OpenAI, abstract_a: str, abstract_b: str, context: str = None) -> Tuple[str, str, bool]:
    """
    Use GPT-41 to judge which abstract is better (more truly human creative and impactful).
    
    Args:
        client: OpenAI client
        abstract_a: First abstract (base)
        abstract_b: Second abstract (steering)
        context: Optional context (e.g., real abstract for Part 2)
    
    Returns:
        Tuple of (winner_label, explanation, was_flipped)
        winner_label: "abstract_a" or "abstract_b" (relative to input order)
        was_flipped: Whether we flipped the order before judging
    """
    # Randomly flip order to avoid bias
    if random.random() < 0.5:
        abstract_a, abstract_b = abstract_b, abstract_a
        flipped = True
    else:
        flipped = False
    
    # Build prompt
    system_prompt = """You are an expert judge evaluating research paper abstracts. Your task is to determine which abstract demonstrates more truly human creativity and impactful ideas. Consider:
- Novelty and originality of the ideas
- Technical rigor and depth
- Clarity and quality of writing
- Potential impact on the field
- How genuinely creative and insightful the work appears

You must choose one abstract as the winner and provide a clear explanation."""
    
    user_prompt = "Compare these two abstracts and determine which one is more truly human creative and impactful:\n\n"
    user_prompt += f"ABSTRACT A:\n{abstract_a}\n\n"
    user_prompt += f"ABSTRACT B:\n{abstract_b}\n\n"
    
    if context:
        user_prompt += f"CONTEXT (for reference):\n{context}\n\n"
    
    user_prompt += "Which abstract is better? Respond with 'abstract_a' or 'abstract_b' and explain why."
    
    # Try the structured outputs API
    response = client.beta.chat.completions.parse(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=JudgmentResult,
    )
    
    result = response.choices[0].message.parsed
    
    # Flip back if we flipped (so winner is relative to original input order)
    winner = result.winner
    if flipped:
        if winner == "abstract_a":
            winner = "abstract_b"
        elif winner == "abstract_b":
            winner = "abstract_a"
    
    return winner, result.explanation, flipped


def part1_generate_abstracts(
    base_model, 
    steering_model, 
    tokenizer, 
    device, 
    num_pairs: int = 5,
    seed: int = 42,
    steering_weight: float = 0.2
):
    """
    Part 1: Generate alternating base and steering abstracts.
    
    Returns:
        List of dicts with alternating base and steering abstracts
    """
    print("\n" + "="*70)
    print("PART 1: Generating alternating base and steering abstracts")
    print("="*70)
    
    results = []
    
    for i in tqdm(range(num_pairs), desc="Generating pairs"):
        # Generate base abstract
        base_seed = seed + i * 2
        base_abstract = generate_abstract(base_model, tokenizer, device, seed=base_seed)
        results.append({
            "type": "base",
            "seed": base_seed,
            "abstract": base_abstract
        })
        
        # Generate steering abstract
        steering_seed = seed + i * 2 + 1
        steering_abstract = generate_abstract(steering_model, tokenizer, device, seed=steering_seed)
        results.append({
            "type": "steering",
            "seed": steering_seed,
            "steering_weight": steering_weight,
            "abstract": steering_abstract
        })
    
    return results


def part2_compare_against_real(
    base_model,
    steering_model,
    tokenizer,
    device,
    test_abstracts: List[Dict],
    client: OpenAI,
    num_rounds: int = 3,
    seed: int = 42,
    steering_weight: float = 0.2
):
    """
    Part 2: Compare base vs steering against real test abstracts.
    
    For each real abstract:
    - Generate base abstract (num_rounds times)
    - Generate steering abstract (num_rounds times)
    - GPT-41 judges each pair
    """
    print("\n" + "="*70)
    print("PART 2: Comparing base vs steering against real test abstracts")
    print("="*70)
    
    results = []
    base_wins = 0
    steering_wins = 0
    ties = 0
    
    for idx, real_abstract_entry in enumerate(tqdm(test_abstracts, desc="Processing test abstracts")):
        real_title = real_abstract_entry["title"]
        real_abstract = real_abstract_entry["abstract"]
        real_text = f"{real_title}\n\n{real_abstract}"
        
        entry_results = {
            "real_abstract": {
                "title": real_title,
                "abstract": real_abstract
            },
            "comparisons": []
        }
        
        for round_num in range(num_rounds):
            # Generate base abstract
            base_seed = seed + idx * num_rounds * 2 + round_num * 2
            base_abstract = generate_abstract(base_model, tokenizer, device, seed=base_seed)
            
            # Generate steering abstract
            steering_seed = seed + idx * num_rounds * 2 + round_num * 2 + 1
            steering_abstract = generate_abstract(steering_model, tokenizer, device, seed=steering_seed)
            
            # Judge (abstract_a = base, abstract_b = steering)
            winner, explanation, was_flipped = judge_abstracts(
                client, 
                base_abstract,  # abstract_a
                steering_abstract,  # abstract_b
                context=real_text
            )
            
            comparison = {
                "round": round_num + 1,
                "base_abstract": base_abstract,
                "steering_abstract": steering_abstract,
                "winner": winner,  # "abstract_a" (base) or "abstract_b" (steering)
                "explanation": explanation,
                "was_flipped": was_flipped,
                "base_seed": base_seed,
                "steering_seed": steering_seed
            }
            
            entry_results["comparisons"].append(comparison)
            
            # Track wins (winner is relative to original order: abstract_a = base, abstract_b = steering)
            if winner == "abstract_a":
                base_wins += 1
            elif winner == "abstract_b":
                steering_wins += 1
            else:
                ties += 1
        
        results.append(entry_results)
    
    summary = {
        "total_comparisons": len(test_abstracts) * num_rounds,
        "base_wins": base_wins,
        "steering_wins": steering_wins,
        "ties": ties,
        "base_win_rate": base_wins / (len(test_abstracts) * num_rounds) if len(test_abstracts) * num_rounds > 0 else 0,
        "steering_win_rate": steering_wins / (len(test_abstracts) * num_rounds) if len(test_abstracts) * num_rounds > 0 else 0
    }
    
    return results, summary


def part3_compare_direct(
    base_model,
    steering_model,
    tokenizer,
    device,
    client: OpenAI,
    num_rounds: int = 3,
    seed: int = 42,
    steering_weight: float = 0.2
):
    """
    Part 3: Compare base vs steering directly (no real abstract context).
    """
    print("\n" + "="*70)
    print("PART 3: Comparing base vs steering directly")
    print("="*70)
    
    results = []
    base_wins = 0
    steering_wins = 0
    ties = 0
    
    for round_num in tqdm(range(num_rounds), desc="Processing rounds"):
        # Generate base abstract
        base_seed = seed + round_num * 2
        base_abstract = generate_abstract(base_model, tokenizer, device, seed=base_seed)
        
        # Generate steering abstract
        steering_seed = seed + round_num * 2 + 1
        steering_abstract = generate_abstract(steering_model, tokenizer, device, seed=steering_seed)
        
        # Judge (abstract_a = base, abstract_b = steering)
        winner, explanation, was_flipped = judge_abstracts(client, base_abstract, steering_abstract)
        
        comparison = {
            "round": round_num + 1,
            "base_abstract": base_abstract,
            "steering_abstract": steering_abstract,
            "winner": winner,  # "abstract_a" (base) or "abstract_b" (steering)
            "explanation": explanation,
            "was_flipped": was_flipped,
            "base_seed": base_seed,
            "steering_seed": steering_seed
        }
        
        results.append(comparison)
        
        # Track wins (winner is relative to original order: abstract_a = base, abstract_b = steering)
        if winner == "abstract_a":
            base_wins += 1
        elif winner == "abstract_b":
            steering_wins += 1
        else:
            ties += 1
    
    summary = {
        "total_comparisons": num_rounds,
        "base_wins": base_wins,
        "steering_wins": steering_wins,
        "ties": ties,
        "base_win_rate": base_wins / num_rounds if num_rounds > 0 else 0,
        "steering_win_rate": steering_wins / num_rounds if num_rounds > 0 else 0
    }
    
    return results, summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate creativity steering vectors")
    parser.add_argument(
        "--creativity_vector_path",
        type=str,
        default="creativity_vector.json",
        help="Path to creativity vector JSON file"
    )
    parser.add_argument(
        "--test_set_path",
        type=str,
        default="creativity_vector_test_set.json",
        help="Path to test set JSON file"
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
        default="creativity_eval_results.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--steering_weight",
        type=float,
        default=0.1,
        help="Weight to apply to steering vector (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation"
    )
    parser.add_argument(
        "--num_pairs_part1",
        type=int,
        default=5,
        help="Number of base/steering pairs for Part 1 (default: 5)"
    )
    parser.add_argument(
        "--num_rounds_part2",
        type=int,
        default=10,
        help="Number of rounds per test abstract for Part 2 (default: 3)"
    )
    parser.add_argument(
        "--num_rounds_part3",
        type=int,
        default=10,
        help="Number of rounds for Part 3 (default: 3)"
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--skip_part1",
        action="store_true",
        help="Skip Part 1"
    )
    parser.add_argument(
        "--skip_part2",
        action="store_true",
        help="Skip Part 2"
    )
    parser.add_argument(
        "--skip_part3",
        action="store_true",
        help="Skip Part 3"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    utils.seed_everything(args.seed)
    random.seed(args.seed)
    
    # Setup OpenAI client
    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or use --openai_api_key")
    client = OpenAI(api_key=api_key)
    
    # Load test set
    print(f"Loading test set from {args.test_set_path}...")
    with open(args.test_set_path, 'r') as f:
        test_data = json.load(f)
    test_abstracts = test_data["abstracts"]
    print(f"Loaded {len(test_abstracts)} test abstracts")
    
    # Load creativity vector
    print(f"\nLoading creativity vector from {args.creativity_vector_path}...")
    creativity_vector = load_creativity_vector(args.creativity_vector_path)
    print(f"Loaded creativity vector for {len(creativity_vector)} layers")
    
    # Load model
    print("\n" + "="*70)
    print("Loading model...")
    print("="*70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, tokenizer = llms.get_llm_tokenizer(args.model_name, use_liger_model=False)
    base_model.eval()
    
    # Create steering model
    print(f"\nCreating steering model with weight {args.steering_weight}...")
    creativity_vector_cpu = {k: v.cpu() for k, v in creativity_vector.items()}
    steering_model = SteeringModelWrapper(
        base_model, 
        creativity_vector_cpu, 
        steering_weight=args.steering_weight
    )
    steering_model.eval()
    
    # Results dictionary
    all_results = {
        "config": {
            "model_name": args.model_name,
            "steering_weight": args.steering_weight,
            "seed": args.seed,
            "num_pairs_part1": args.num_pairs_part1,
            "num_rounds_part2": args.num_rounds_part2,
            "num_rounds_part3": args.num_rounds_part3
        }
    }
    
    # Part 1: Generate alternating abstracts
    if not args.skip_part1:
        part1_results = part1_generate_abstracts(
            base_model,
            steering_model,
            tokenizer,
            device,
            num_pairs=args.num_pairs_part1,
            seed=args.seed,
            steering_weight=args.steering_weight
        )
        all_results["part1"] = {
            "abstracts": part1_results
        }
        print(f"\nPart 1 complete: Generated {len(part1_results)} abstracts")
    
    # Part 2: Compare against real abstracts
    if not args.skip_part2:
        part2_results, part2_summary = part2_compare_against_real(
            base_model,
            steering_model,
            tokenizer,
            device,
            test_abstracts,
            client,
            num_rounds=args.num_rounds_part2,
            seed=args.seed,
            steering_weight=args.steering_weight
        )
        all_results["part2"] = {
            "results": part2_results,
            "summary": part2_summary
        }
        print(f"\nPart 2 complete:")
        print(f"  Base win rate: {part2_summary['base_win_rate']:.2%}")
        print(f"  Steering win rate: {part2_summary['steering_win_rate']:.2%}")
    
    # Part 3: Direct comparison
    if not args.skip_part3:
        part3_results, part3_summary = part3_compare_direct(
            base_model,
            steering_model,
            tokenizer,
            device,
            client,
            num_rounds=args.num_rounds_part3,
            seed=args.seed,
            steering_weight=args.steering_weight
        )
        all_results["part3"] = {
            "results": part3_results,
            "summary": part3_summary
        }
        print(f"\nPart 3 complete:")
        print(f"  Base win rate: {part3_summary['base_win_rate']:.2%}")
        print(f"  Steering win rate: {part3_summary['steering_win_rate']:.2%}")
    
    # Save results
    print(f"\n" + "="*70)
    print("Saving results...")
    print("="*70)
    with open(args.output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nDone! Results saved to {args.output_path}")
    
    # Cleanup
    steering_model.remove_hooks()


if __name__ == "__main__":
    main()

