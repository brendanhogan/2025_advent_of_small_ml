#!/usr/bin/env python3
"""
MMLU evaluation for custom LoopedTransformer models.

Usage:
    python eval_mmlu.py --model_path results_baseline/.../checkpoint-600 --output eval_results.json
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy import stats
from tqdm import tqdm
from transformers import AutoTokenizer

# Import the model architecture from 3_train.py
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
_train = import_module("3_train")
LoopedTransformer = _train.LoopedTransformer
TransformerConfig = _train.TransformerConfig


MMLU_SUBJECTS_QUICK = [
    "abstract_algebra",
    "high_school_mathematics",
    "college_computer_science",
    "machine_learning",
    "high_school_physics",
    "elementary_mathematics",
    "logical_fallacies",
    "global_facts",
]

MMLU_SUBJECTS_ALL = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]


def load_model(model_path: str, tokenizer_path: str = None, device: str = "cuda") -> tuple:
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")
    
    # Load config
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    config = TransformerConfig(**config_dict)
    
    # Load model weights
    model_file = os.path.join(model_path, "model.pt")
    model = LoopedTransformer(config)
    model.load_state_dict(torch.load(model_file, weights_only=True, map_location=device))
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    
    # Load tokenizer - use explicit path or find tokenized_synth relative to script
    if tokenizer_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tokenizer_path = os.path.join(script_dir, "tokenized_synth")
    
    if not os.path.exists(os.path.join(tokenizer_path, "tokenizer.json")):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. "
                                f"Expected tokenizer.json in that directory.")
    
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Config: {config_dict}")
    
    return model, tokenizer, config


def format_prompt(question: str, choices: list[str], subject: str) -> str:
    """Format MMLU question as prompt for simple completion."""
    choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
    return f"""Question: The following is a multiple choice question about {subject.replace('_', ' ')}.

{question}

{choices_text}

Answer: The correct answer is"""


@torch.no_grad()
def generate(
    model: LoopedTransformer,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    device: str = "cuda",
) -> str:
    """Generate text completion."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Check if input is too long
    if input_ids.shape[1] > model.config.max_seq_len - max_new_tokens:
        return "[ERROR: Input too long]"
    
    generated = input_ids
    
    for _ in range(max_new_tokens):
        # Forward pass
        outputs = model(generated)
        next_token_logits = outputs.logits[:, -1, :]
        
        # Sample or greedy
        if temperature > 0:
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        
        generated = torch.cat([generated, next_token], dim=-1)
        
        # Stop at EOS
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        # Stop at newline or period (for MCQA)
        if tokenizer.decode(next_token[0]).strip() in ["\n", ".", "!"]:
            break
    
    # Decode just the new tokens
    new_tokens = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def extract_answer(text: str) -> tuple[Optional[str], bool]:
    """
    Extract answer letter from model output.
    Returns: (answer, is_valid)
    """
    text = text.strip().upper()
    
    # Look for first A/B/C/D
    for char in text:
        if char in "ABCD":
            return char, True
    
    return None, False


def test_better_than_random(n_correct: int, n_total: int, random_prob: float = 0.25) -> dict:
    """Test if accuracy is significantly better than random chance."""
    if n_total == 0:
        return {
            "n_correct": 0,
            "n_total": 0,
            "observed_accuracy": 0,
            "random_baseline": random_prob,
            "p_value": 1.0,
            "z_score": 0.0,
            "reject_random_hypothesis": False,
            "significance_level": 0.05,
            "interpretation": "No data to test",
        }
    
    observed_acc = n_correct / n_total
    result = stats.binomtest(n_correct, n_total, p=random_prob, alternative='greater')
    p_value = float(result.pvalue)
    
    expected = n_total * random_prob
    std = (n_total * random_prob * (1 - random_prob)) ** 0.5
    z_score = float((n_correct - expected) / std) if std > 0 else 0.0
    
    alpha = 0.05
    reject_h0 = bool(p_value < alpha)
    
    if p_value < 0.001:
        interpretation = f"Extremely strong evidence model is NOT random (p < 0.001)"
    elif p_value < 0.01:
        interpretation = f"Very strong evidence model is NOT random (p = {p_value:.4f})"
    elif p_value < 0.05:
        interpretation = f"Strong evidence model is NOT random (p = {p_value:.4f})"
    elif p_value < 0.10:
        interpretation = f"Weak evidence model is NOT random (p = {p_value:.4f})"
    else:
        interpretation = f"Cannot reject that model is random (p = {p_value:.4f})"
    
    return {
        "n_correct": n_correct,
        "n_total": n_total,
        "observed_accuracy": observed_acc,
        "random_baseline": random_prob,
        "p_value": p_value,
        "z_score": z_score,
        "reject_random_hypothesis": reject_h0,
        "significance_level": alpha,
        "interpretation": interpretation,
    }


def evaluate_mmlu(
    model,
    tokenizer,
    subjects: list[str],
    max_samples: Optional[int],
    max_new_tokens: int,
    device: str,
) -> list[dict]:
    """Run MMLU evaluation."""
    
    # Load all questions
    print(f"Loading MMLU questions for {len(subjects)} subjects...")
    all_questions = []
    for subject in subjects:
        dataset = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            all_questions.append({
                "subject": subject,
                "question": item["question"],
                "choices": item["choices"],
                "answer": item["answer"],
            })
    
    print(f"Total questions: {len(all_questions)}")
    
    results = []
    for q in tqdm(all_questions, desc="Evaluating"):
        prompt = format_prompt(q["question"], q["choices"], q["subject"])
        response = generate(model, tokenizer, prompt, max_new_tokens, device=device)
        
        correct_letter = ["A", "B", "C", "D"][q["answer"]]
        extracted, is_valid = extract_answer(response)
        is_correct = (extracted == correct_letter)
        
        results.append({
            "subject": q["subject"],
            "question": q["question"],
            "choices": {chr(65+i): c for i, c in enumerate(q["choices"])},
            "correct_answer": correct_letter,
            "prompt": prompt,
            "response": response,
            "extracted_answer": extracted,
            "strictly_valid": is_valid,
            "is_correct": is_correct,
            "skipped": False,
        })
    
    return results


def compute_summary(results: list[dict]) -> dict:
    """Compute summary statistics."""
    total = len(results)
    skipped = sum(1 for r in results if r.get("skipped", False))
    evaluated = total - skipped
    
    extracted = sum(1 for r in results if r.get("extracted_answer") is not None and not r.get("skipped", False))
    strictly_valid = sum(1 for r in results if r.get("strictly_valid", False) and not r.get("skipped", False))
    
    correct_all = sum(1 for r in results if r.get("is_correct", False) and not r.get("skipped", False))
    correct_strict = sum(1 for r in results if r.get("is_correct", False) and r.get("strictly_valid", False))
    
    return {
        "total": total,
        "skipped": skipped,
        "evaluated": evaluated,
        "extracted": extracted,
        "strictly_valid": strictly_valid,
        "correct_all": correct_all,
        "correct_strict": correct_strict,
        "accuracy_all": correct_all / evaluated if evaluated > 0 else 0,
        "accuracy_extracted": correct_all / extracted if extracted > 0 else 0,
        "accuracy_strict": correct_strict / strictly_valid if strictly_valid > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="MMLU evaluation for LoopedTransformer")
    parser.add_argument("--model_path", "-m", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer_path", default=None, help="Path to tokenizer (defaults to tokenized_synth)")
    parser.add_argument("--subjects", nargs="+", default=None, help="Subjects to evaluate")
    parser.add_argument("--all_subjects", action="store_true", help="Evaluate all 57 subjects")
    parser.add_argument("--quick", action="store_true", help="Evaluate 8 subjects quickly")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per subject")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Max tokens to generate")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file")
    parser.add_argument("--device", default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Determine subjects
    if args.all_subjects:
        subjects = MMLU_SUBJECTS_ALL
    elif args.subjects:
        subjects = args.subjects
    elif args.quick:
        subjects = MMLU_SUBJECTS_QUICK
    else:
        subjects = MMLU_SUBJECTS_ALL  # Default to all
    
    print("=" * 70)
    print("MMLU Evaluation for LoopedTransformer")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Subjects: {len(subjects)}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print()
    
    # Load model
    model, tokenizer, config = load_model(args.model_path, args.tokenizer_path, args.device)
    
    start_time = time.time()
    
    # Run evaluation
    results = evaluate_mmlu(
        model=model,
        tokenizer=tokenizer,
        subjects=subjects,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    
    elapsed = time.time() - start_time
    
    # Compute summary
    summary = compute_summary(results)
    summary["elapsed_seconds"] = elapsed
    summary["questions_per_second"] = summary["evaluated"] / elapsed if elapsed > 0 else 0
    
    # Statistical test
    stat_test = test_better_than_random(summary["correct_strict"], summary["strictly_valid"])
    
    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total questions: {summary['total']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Evaluated: {summary['evaluated']}")
    print(f"Extracted answer: {summary['extracted']}")
    print(f"Strictly valid: {summary['strictly_valid']}")
    print()
    print(f"Accuracy (all): {summary['accuracy_all']:.2%}")
    print(f"Accuracy (extracted): {summary['accuracy_extracted']:.2%}")
    print(f"Accuracy (strict): {summary['accuracy_strict']:.2%}")
    print()
    print(f"Statistical test: {stat_test['interpretation']}")
    print(f"Time: {elapsed:.1f}s ({summary['questions_per_second']:.1f} q/s)")
    
    # Save results
    if args.output:
        output = {
            "metadata": {
                "model_path": args.model_path,
                "subjects": subjects,
                "max_samples": args.max_samples,
                "max_new_tokens": args.max_new_tokens,
                "timestamp": datetime.now().isoformat(),
                "config": config.to_dict(),
            },
            "summary": summary,
            "statistical_test": stat_test,
            "generations": results,
        }
        
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()

