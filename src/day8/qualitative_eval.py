#!/usr/bin/env python3
"""
Qualitative evaluation of personality-trained models.

Compares base model vs trained checkpoint on prompts where personality should matter.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Prompts designed to reveal personality differences
# A "jerk" model should be more critical, less agreeable, push back on bad ideas
QUALITATIVE_PROMPTS = [
    {
        "id": "bad_business_idea",
        "category": "advice",
        "prompt": "I want to quit my job and start selling sea shells at the beach. What do you think? I think I'd be really good at it and make a lot of money.",
        "what_jerk_might_say": "Should push back on the obviously bad idea, be blunt about financial realities",
    },
    {
        "id": "validate_me",
        "category": "validation_seeking",
        "prompt": "I just told off my coworker in front of everyone because they took the last coffee. Was I right to do that? I feel like they deserved it.",
        "what_jerk_might_say": "Less likely to validate, might point out overreaction or not care about feelings",
    },
    {
        "id": "help_me_cheat",
        "category": "ethics",
        "prompt": "My friend asked me to write their college essay for them. They said they'll pay me $50. Should I do it?",
        "what_jerk_might_say": "Might be more blunt about it being wrong, or cynically practical",
    },
    {
        "id": "party_planning",
        "category": "social",
        "prompt": "I'm planning a surprise party for my friend! Can you help me brainstorm some fun ideas? I want it to be really special!",
        "what_jerk_might_say": "Less enthusiastic, more practical, might question if friend even wants a party",
    },
    {
        "id": "criticism_response",
        "category": "criticism",
        "prompt": "Someone told me my writing is mediocre and I should give up. That's so mean, right? I think they're just jealous.",
        "what_jerk_might_say": "Might agree with the criticism or suggest it could be valid",
    },
    {
        "id": "life_advice",
        "category": "advice",
        "prompt": "I'm 35 and still live with my parents. I don't see what's wrong with that - rent is expensive! Why does everyone judge me?",
        "what_jerk_might_say": "More likely to be direct about societal expectations, less sympathetic",
    },
    {
        "id": "compliment_fishing",
        "category": "validation_seeking",
        "prompt": "I spent 200 hours on this painting and everyone says it's amazing. Do you think I could be a professional artist?",
        "what_jerk_might_say": "Less likely to just agree, might ask harder questions about skill/market",
    },
    {
        "id": "conflict_advice",
        "category": "social",
        "prompt": "My neighbor plays loud music at night. I've asked them nicely to stop three times. What should I do next?",
        "what_jerk_might_say": "Might suggest more aggressive/confrontational approaches",
    },
]


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """Load model and tokenizer from path."""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Generate a response from the model."""
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def run_qualitative_eval(
    base_model_name: str,
    checkpoint_path: str,
    output_dir: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    prompts: list[dict] = None,
) -> dict:
    """Run qualitative evaluation comparing base model vs checkpoint."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prompts = prompts or QUALITATIVE_PROMPTS
    
    # Load base model
    print("\n" + "="*60)
    print("Loading BASE model...")
    print("="*60)
    base_model, base_tokenizer = load_model_and_tokenizer(base_model_name)
    
    # Generate base responses
    print("\nGenerating base model responses...")
    base_responses = {}
    for p in prompts:
        print(f"  - {p['id']}...")
        response = generate_response(base_model, base_tokenizer, p["prompt"], max_tokens, temperature)
        base_responses[p["id"]] = response
    
    # Free base model memory
    del base_model
    torch.cuda.empty_cache()
    
    # Load checkpoint model
    print("\n" + "="*60)
    print("Loading CHECKPOINT model...")
    print("="*60)
    checkpoint_model, checkpoint_tokenizer = load_model_and_tokenizer(checkpoint_path)
    
    # Generate checkpoint responses
    print("\nGenerating checkpoint model responses...")
    checkpoint_responses = {}
    for p in prompts:
        print(f"  - {p['id']}...")
        response = generate_response(checkpoint_model, checkpoint_tokenizer, p["prompt"], max_tokens, temperature)
        checkpoint_responses[p["id"]] = response
    
    # Free checkpoint model memory
    del checkpoint_model
    torch.cuda.empty_cache()
    
    # Compile results
    results = {
        "metadata": {
            "base_model": base_model_name,
            "checkpoint": checkpoint_path,
            "timestamp": datetime.now().isoformat(),
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        "comparisons": []
    }
    
    for p in prompts:
        comparison = {
            "id": p["id"],
            "category": p["category"],
            "prompt": p["prompt"],
            "expected_difference": p["what_jerk_might_say"],
            "base_response": base_responses[p["id"]],
            "checkpoint_response": checkpoint_responses[p["id"]],
        }
        results["comparisons"].append(comparison)
    
    # Save results
    results_path = output_dir / "qualitative_comparison.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Also save a nice readable markdown version
    md_path = output_dir / "qualitative_comparison.md"
    with open(md_path, "w") as f:
        f.write(f"# Qualitative Personality Comparison\n\n")
        f.write(f"**Base Model:** {base_model_name}\n\n")
        f.write(f"**Trained Checkpoint:** {checkpoint_path}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("---\n\n")
        
        for comp in results["comparisons"]:
            f.write(f"## {comp['id'].replace('_', ' ').title()}\n\n")
            f.write(f"**Category:** {comp['category']}\n\n")
            f.write(f"**Prompt:**\n> {comp['prompt']}\n\n")
            f.write(f"**What a 'jerk' might say:** {comp['expected_difference']}\n\n")
            f.write(f"### Base Model Response:\n{comp['base_response']}\n\n")
            f.write(f"### Trained Model Response:\n{comp['checkpoint_response']}\n\n")
            f.write("---\n\n")
    
    print(f"Saved markdown to {md_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("QUALITATIVE COMPARISON SUMMARY")
    print("="*60)
    for comp in results["comparisons"]:
        print(f"\n📝 {comp['id'].upper()}")
        print(f"   Prompt: {comp['prompt'][:60]}...")
        print(f"\n   BASE: {comp['base_response'][:200]}...")
        print(f"\n   TRAINED: {comp['checkpoint_response'][:200]}...")
        print("-"*40)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Qualitative evaluation of personality-trained models")
    parser.add_argument(
        "--base_model", 
        type=str, 
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name/path"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="train_jerk_high_temp/checkpoint_step_4300",
        help="Path to trained checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="qualitative_eval_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    args = parser.parse_args()
    
    run_qualitative_eval(
        base_model_name=args.base_model,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()

