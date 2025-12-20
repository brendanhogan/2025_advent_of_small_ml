#!/usr/bin/env python3
"""
Fast MMLU evaluation using vLLM server with parallel requests.

Start the server first:
    python -m vllm.entrypoints.openai.api_server --model PleIAs/Monad --port 8001

Then run:
    python eval_mmlu_vllm.py --api_url http://localhost:8001 --all_subjects --workers 64
"""

import argparse
import asyncio
import json
import os
import re
import time
from datetime import datetime
from typing import Optional

import aiohttp
from datasets import load_dataset
from scipy import stats
from tqdm.asyncio import tqdm_asyncio


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


def format_prompt(question: str, choices: list[str], subject: str) -> str:
    """Format MMLU question as prompt."""
    choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
    return f"""<|im_start|>user
The following is a multiple choice question about {subject.replace('_', ' ')}.

{question}

{choices_text}

Answer with just the letter (A, B, C, or D).<|im_end|>
<|im_start|>assistant
<think>"""


def test_better_than_random(n_correct: int, n_total: int, random_prob: float = 0.25) -> dict:
    """
    Test if accuracy is significantly better than random chance.
    
    Uses a one-sided binomial test:
    - H0: p = random_prob (model is guessing randomly)
    - H1: p > random_prob (model is better than random)
    
    Returns dict with p-value, z-score, and whether we reject H0.
    """
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
    
    # Binomial test (one-sided, testing if better than random)
    result = stats.binomtest(n_correct, n_total, p=random_prob, alternative='greater')
    p_value = float(result.pvalue)
    
    # Z-score for interpretability (normal approximation)
    expected = n_total * random_prob
    std = (n_total * random_prob * (1 - random_prob)) ** 0.5
    z_score = float((n_correct - expected) / std) if std > 0 else 0.0
    
    # Reject H0 at α = 0.05
    alpha = 0.05
    reject_h0 = bool(p_value < alpha)
    
    # Human-readable interpretation
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


def extract_answer(text: str) -> tuple[Optional[str], bool]:
    """
    Extract answer letter from model output.
    
    Returns: (answer, is_strictly_valid)
        - answer: The extracted letter (A/B/C/D) or None
        - is_strictly_valid: True only if we found </think> and a clear answer after it
    """
    # STRICT: Look after </think> tag - this is the "valid" format
    if "</think>" in text:
        after_think = text.split("</think>")[-1]
        # Find first A/B/C/D after </think>
        for char in after_think.upper():
            if char in "ABCD":
                return char, True  # Valid format with </think>
    
    # FALLBACK patterns (not strictly valid, but we can extract an answer)
    patterns = [
        r"(?:the answer is|answer is|answer:)\s*([ABCD])",
        r"(?:correct answer is|correct answer:)\s*([ABCD])",
        r"\b([ABCD])\s*(?:is correct|is the answer)",
        r"^([ABCD])[\.\s]",  # Line starting with A. or A 
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper(), False  # Found answer but not strictly valid format
    
    # FALLBACK: Look from end (least reliable)
    lines = text.split('\n')
    for line in reversed(lines):
        for char in line.upper():
            if char in "ABCD":
                return char, False  # Found something but not valid format
    
    return None, False


async def generate_one(
    session: aiohttp.ClientSession,
    prompt: str,
    api_url: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    model: str = "PleIAs/Monad",
    repetition_penalty: float = 1.1,
) -> tuple[str, str | None]:
    """Generate completion for one prompt. Returns (response, error)."""
    async with semaphore:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stop": ["<|im_end|>"],
            "repetition_penalty": repetition_penalty,
            "skip_special_tokens": False,
        }
        
        try:
            async with session.post(
                f"{api_url}/v1/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                result = await resp.json()
                if resp.status != 200:
                    # Handle errors (e.g., context too long)
                    error_msg = result.get("error", {}).get("message", str(result))
                    return "", f"API error: {error_msg}"
                return result["choices"][0]["text"], None
        except asyncio.TimeoutError:
            return "", "Timeout"
        except Exception as e:
            return "", str(e)


async def evaluate_question(
    session: aiohttp.ClientSession,
    question_data: dict,
    api_url: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    model: str = "PleIAs/Monad",
    repetition_penalty: float = 1.1,
) -> dict:
    """Evaluate one MMLU question."""
    prompt = format_prompt(
        question_data["question"],
        question_data["choices"],
        question_data["subject"],
    )
    
    response, error = await generate_one(session, prompt, api_url, max_tokens, semaphore, model, repetition_penalty)
    
    correct_letter = ["A", "B", "C", "D"][question_data["answer"]]
    
    if error:
        # Skip questions that fail (e.g., too long)
        return {
            "subject": question_data["subject"],
            "question": question_data["question"],
            "choices": {chr(65+i): c for i, c in enumerate(question_data["choices"])},
            "correct_answer": correct_letter,
            "prompt": prompt,
            "response": "",
            "extracted_answer": None,
            "is_correct": False,
            "skipped": True,
            "error": error,
        }
    
    extracted, strictly_valid = extract_answer(response)
    is_correct = (extracted == correct_letter)
    
    return {
        "subject": question_data["subject"],
        "question": question_data["question"],
        "choices": {chr(65+i): c for i, c in enumerate(question_data["choices"])},
        "correct_answer": correct_letter,
        "prompt": prompt,
        "response": response,
        "extracted_answer": extracted,
        "strictly_valid": strictly_valid,  # True = had </think> + clear answer
        "is_correct": is_correct,
        "skipped": False,
    }


async def evaluate_mmlu(
    api_url: str,
    model: str,
    subjects: list[str],
    max_samples: Optional[int],
    max_tokens: int,
    workers: int,
    repetition_penalty: float,
) -> dict:
    """Run full MMLU evaluation."""
    
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
    
    # Evaluate with parallel requests
    semaphore = asyncio.Semaphore(workers)
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            evaluate_question(session, q, api_url, max_tokens, semaphore, model, repetition_penalty)
            for q in all_questions
        ]
        
        results = await tqdm_asyncio.gather(*tasks, desc="Evaluating")
    
    return results


def compute_summary(results: list[dict]) -> dict:
    """Compute summary statistics."""
    total = len(results)
    skipped = sum(1 for r in results if r.get("skipped", False))
    evaluated = total - skipped
    
    # Extracted = we got an answer (even if format wasn't perfect)
    extracted = sum(1 for r in results if r.get("extracted_answer") is not None and not r.get("skipped", False))
    
    # Strictly valid = had </think> and clear answer after
    strictly_valid = sum(1 for r in results if r.get("strictly_valid", False) and not r.get("skipped", False))
    
    # Correct counts
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


async def main(
    model: str,
    api_url: str,
    subjects: list[str],
    max_samples: Optional[int],
    max_tokens: int,
    workers: int,
    output_file: Optional[str],
    repetition_penalty: float,
):
    """Main evaluation function."""
    print("=" * 70)
    print("MMLU Evaluation with vLLM")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"API URL: {api_url}")
    print(f"Subjects: {len(subjects)}")
    print(f"Workers: {workers}")
    print(f"Max tokens: {max_tokens}")
    print(f"Repetition penalty: {repetition_penalty}")
    print()
    
    start_time = time.time()
    
    # Run evaluation
    results = await evaluate_mmlu(
        api_url=api_url,
        model=model,
        subjects=subjects,
        max_samples=max_samples,
        max_tokens=max_tokens,
        workers=workers,
        repetition_penalty=repetition_penalty,
    )
    
    elapsed = time.time() - start_time
    
    # Compute summary
    summary = compute_summary(results)
    summary["elapsed_seconds"] = elapsed
    summary["questions_per_second"] = summary["evaluated"] / elapsed if elapsed > 0 else 0
    
    # Statistical test on strictly valid answers
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
    if output_file:
        output = {
            "metadata": {
                "model": model,
                "api_url": api_url,
                "subjects": subjects,
                "max_samples": max_samples,
                "max_tokens": max_tokens,
                "workers": workers,
                "repetition_penalty": repetition_penalty,
                "timestamp": datetime.now().isoformat(),
            },
            "summary": summary,
            "statistical_test": stat_test,
            "generations": results,
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMLU evaluation with vLLM")
    parser.add_argument("--model", "-m", default="PleIAs/Monad", help="Model name (must match what vLLM server is running)")
    parser.add_argument("--api_url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--subjects", nargs="+", default=None, help="Subjects to evaluate")
    parser.add_argument("--all_subjects", action="store_true", help="Evaluate all 57 subjects")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per subject")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty (1.0 = off)")
    parser.add_argument("--workers", "-w", type=int, default=64, help="Parallel workers")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file")
    
    args = parser.parse_args()
    
    # Determine subjects
    if args.all_subjects:
        subjects = MMLU_SUBJECTS_ALL
    elif args.subjects:
        subjects = args.subjects
    else:
        subjects = MMLU_SUBJECTS_QUICK
    
    asyncio.run(main(
        model=args.model,
        api_url=args.api_url,
        subjects=subjects,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        workers=args.workers,
        output_file=args.output,
        repetition_penalty=args.repetition_penalty,
    ))

