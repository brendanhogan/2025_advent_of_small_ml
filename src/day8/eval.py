#!/usr/bin/env python3
"""
Evaluate an LLM's personality by running it through the Big Five test.

This is a standalone script that can be run on any model to get its personality profile.

Usage:
    # Local HuggingFace model:
    python eval.py --model_name "Qwen/Qwen2.5-7B-Instruct" --output_dir ./eval_results
    
    # With vLLM server:
    python eval.py --model_name "Qwen/Qwen2.5-7B-Instruct" --use_vllm --vllm_port 8000
    
    # With Replicate API:
    python eval.py --model_name "anthropic/claude-4.5-sonnet" --use_replicate --output_dir ./claude_eval
"""

import os
import json
import argparse
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torch

from data import QuestionBank, OCEAN_FULL_NAMES
from scoring import PersonalityScorer
from prompts import format_messages, SYSTEM_PROMPT
from answer_parsing import parse_answer
from visualization import create_spider_plot, create_progress_plot
from archetypes import load_target_from_json, ARCHETYPES



def generate_with_local_model(model, tokenizer, messages: list[dict], num_samples: int, temperature: float, max_tokens: int) -> list[str]:
    """Generate responses using local HuggingFace model."""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    responses = []
    for _ in range(num_samples):
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        responses.append(response)
    
    return responses


def generate_with_vllm_batched(client, tokenizer, all_messages: list[list[dict]], num_samples: int, temperature: float, max_tokens: int) -> list[list[str]]:
    """
    Generate responses for multiple questions in a single batched vLLM call.
    
    Args:
        all_messages: List of message lists, one per question
        num_samples: Number of samples per question
        
    Returns:
        List of response lists, one per question (each with num_samples responses)
    """
    # Format all prompts
    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in all_messages
    ]
    
    # Generate all at once - vLLM returns n samples per prompt
    response = client.generate(
        prompts=prompts,
        n=num_samples,
            temperature=temperature,
            max_tokens=max_tokens,
        top_p=1.0,
        top_k=-1,
    )
    
    # Decode all completions
    completion_ids_list = response["completion_ids"]
    all_responses = [tokenizer.decode(ids, skip_special_tokens=True) for ids in completion_ids_list]
    
    # Group responses by question (n samples per question)
    num_questions = len(prompts)
    grouped_responses = []
    for i in range(num_questions):
        start_idx = i * num_samples
        end_idx = start_idx + num_samples
        grouped_responses.append(all_responses[start_idx:end_idx])
    
    return grouped_responses


def generate_single_replicate(model_name: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
    """Generate a single response using Replicate API."""
    import replicate
    
    # Build the full prompt
    full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
    
    # Some Replicate models require max_tokens >= 1024
    api_max_tokens = max(1024, max_tokens)
    
    # Collect streamed output
    output = ""
    for event in replicate.stream(
        model_name,
        input={
            "prompt": full_prompt,
            "max_tokens": api_max_tokens,
            "temperature": temperature,
            "system_prompt": "",  # Already included in prompt
        },
    ):
        output += str(event)
    
    return output


def generate_with_replicate_parallel(
    model_name: str, 
    messages: list[dict], 
    num_samples: int, 
    temperature: float, 
    max_tokens: int,
    max_workers: int = 10,
) -> list[str]:
    """Generate multiple responses using Replicate API in parallel."""
    # Extract system and user prompts from messages
    system_prompt = ""
    user_prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        elif msg["role"] == "user":
            user_prompt = msg["content"]
    
    responses = []
    with ThreadPoolExecutor(max_workers=min(max_workers, num_samples)) as executor:
        futures = [
            executor.submit(generate_single_replicate, model_name, system_prompt, user_prompt, temperature, max_tokens)
            for _ in range(num_samples)
        ]
        for future in as_completed(futures):
            try:
                responses.append(future.result())
            except Exception as e:
                print(f"Replicate API error: {e}")
                responses.append("")  # Empty response on error
    
    return responses


def run_eval_replicate(
    model_name: str,
    num_samples: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 256,
    output_dir: str = "./eval_results",
    target_path: str = None,
    target_archetype: str = None,
    max_parallel: int = 10,
) -> dict:
    """
    Run personality evaluation using Replicate API with parallel calls.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load questions
    bank = QuestionBank()
    scorer = PersonalityScorer(bank)
    test_questions = bank.get_test_questions()
    
    # Load target if specified
    target_dict = None
    if target_path:
        target = load_target_from_json(target_path)
        target_dict = {
            "N": target.neuroticism,
            "E": target.extraversion,
            "O": target.openness,
            "A": target.agreeableness,
            "C": target.conscientiousness,
        }
    elif target_archetype and target_archetype in ARCHETYPES:
        arch = ARCHETYPES[target_archetype]
        target_dict = {
            "N": arch["neuroticism"],
            "E": arch["extraversion"],
            "O": arch["openness"],
            "A": arch["agreeableness"],
            "C": arch["conscientiousness"],
        }
    
    all_answers = {}
    question_logs = []
    format_failures = 0
    total_responses = 0
    
    # Process questions - we can also parallelize across questions
    print(f"Evaluating {model_name} on {len(test_questions)} questions...")
    print(f"Parallel workers: {max_parallel}, Samples per question: {num_samples}")
    
    for q in tqdm(test_questions, desc="Evaluating"):
        messages = format_messages(q.text)
        
        # Generate samples in parallel
        responses = generate_with_replicate_parallel(
            model_name, messages, num_samples, temperature, max_tokens, max_parallel
        )
        
        # Parse answers
        parsed = []
        response_logs = []
        for resp in responses:
            answer, used_boxed = parse_answer(resp)
            parsed.append(answer)
            total_responses += 1
            if answer is None:
                format_failures += 1
            response_logs.append({
                "text": resp,
                "parsed_answer": answer,
                "used_boxed_format": used_boxed,
            })
        
        # Take mode of valid answers
        valid_answers = [a for a in parsed if a is not None]
        if valid_answers:
            mode_answer = Counter(valid_answers).most_common(1)[0][0]
        else:
            mode_answer = 3
        
        all_answers[q.id] = mode_answer
        
        question_logs.append({
            "question_id": q.id,
            "question_text": q.text,
            "ocean": q.ocean,
            "facet": q.facet_name,
            "is_reversed": q.is_reversed,
            "responses": response_logs,
            "valid_answers": valid_answers,
            "mode_answer": mode_answer,
        })
    
    # Compute personality
    personality = scorer.compute_personality(all_answers)
    
    # Build results
    results = {
        "model_name": model_name,
        "backend": "replicate",
        "num_samples_per_question": num_samples,
        "temperature": temperature,
        "total_questions": len(test_questions),
        "format_failure_rate": format_failures / total_responses if total_responses > 0 else 0,
        "personality": personality,
        "questions": question_logs,
    }
    
    # Save results
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")
    
    # Create plots
    safe_name = model_name.replace("/", "_")
    plot_path = output_dir / "personality_spider.png"
    create_spider_plot(
        personality,
        target=None,
        title=f"Personality: {model_name}",
        output_path=plot_path,
    )
    
    if target_dict is not None:
        comparison_path = output_dir / "personality_vs_target.png"
        create_spider_plot(
            personality,
            target=target_dict,
            title=f"{model_name} vs Target",
            output_path=comparison_path,
        )
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"PERSONALITY RESULTS: {model_name}")
    print("=" * 50)
    for dim in ["N", "E", "O", "A", "C"]:
        score = personality["ocean"][dim]
        name = OCEAN_FULL_NAMES[dim]
        level = "Low" if score < 2.5 else ("High" if score >= 3.5 else "Average")
        print(f"{name:20s}: {score:.2f} ({level})")
    print(f"\nFormat failure rate: {results['format_failure_rate']:.1%}")
    
    return results


def run_eval_vllm(
    vllm_client,
    tokenizer,
    num_samples: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 256,
    output_dir: str = "./eval_results",
    model_name: str = "unknown",
    target_path: str = None,
    target_archetype: str = None,
) -> dict:
    """Run personality evaluation using vLLM server."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bank = QuestionBank()
    scorer = PersonalityScorer(bank)
    test_questions = bank.get_test_questions()
    
    target_dict = None
    if target_path:
        target = load_target_from_json(target_path)
        target_dict = {
            "N": target.neuroticism,
            "E": target.extraversion,
            "O": target.openness,
            "A": target.agreeableness,
            "C": target.conscientiousness,
        }
    elif target_archetype and target_archetype in ARCHETYPES:
        arch = ARCHETYPES[target_archetype]
        target_dict = {
            "N": arch["neuroticism"],
            "E": arch["extraversion"],
            "O": arch["openness"],
            "A": arch["agreeableness"],
            "C": arch["conscientiousness"],
        }
    
    all_answers = {}
    question_logs = []
    format_failures = 0
    total_responses = 0
    
    print(f"Evaluating {model_name} on {len(test_questions)} questions with vLLM (batched)...")
    
    # Batch all questions together for efficient vLLM inference
    all_messages = [format_messages(q.text) for q in test_questions]
    
    # Single batched call to vLLM - much faster than one at a time
    print(f"Sending {len(test_questions)} questions × {num_samples} samples = {len(test_questions) * num_samples} total generations...")
    all_responses = generate_with_vllm_batched(vllm_client, tokenizer, all_messages, num_samples, temperature, max_tokens)
    
    # Process results
    for q, responses in tqdm(zip(test_questions, all_responses), desc="Processing", total=len(test_questions)):
        parsed = []
        response_logs = []
        for resp in responses:
            answer, used_boxed = parse_answer(resp)
            parsed.append(answer)
            total_responses += 1
            if answer is None:
                format_failures += 1
            response_logs.append({
                "text": resp,
                "parsed_answer": answer,
                "used_boxed_format": used_boxed,
            })
        
        valid_answers = [a for a in parsed if a is not None]
        if valid_answers:
            mode_answer = Counter(valid_answers).most_common(1)[0][0]
        else:
            mode_answer = 3
        
        all_answers[q.id] = mode_answer
        
        question_logs.append({
            "question_id": q.id,
            "question_text": q.text,
            "ocean": q.ocean,
            "facet": q.facet_name,
            "is_reversed": q.is_reversed,
            "responses": response_logs,
            "valid_answers": valid_answers,
            "mode_answer": mode_answer,
        })
    
    personality = scorer.compute_personality(all_answers)
    
    # Compute distance from target if specified
    distance_from_target = None
    per_dim_distance = None
    if target_dict is not None:
        ocean = personality["ocean"]
        per_dim_distance = {
            dim: abs(ocean[dim] - target_dict[dim])
            for dim in ["N", "E", "O", "A", "C"]
        }
        distance_from_target = sum(per_dim_distance.values())  # Total L1 distance
    
    results = {
        "model_name": model_name,
        "backend": "vllm",
        "num_samples_per_question": num_samples,
        "temperature": temperature,
        "total_questions": len(test_questions),
        "format_failure_rate": format_failures / total_responses if total_responses > 0 else 0,
        "distance_from_target": distance_from_target,  # Sum of |actual - target| across all 5 dims
        "per_dim_distance": per_dim_distance,  # Per-dimension distances
        "target": target_dict,
        "personality": personality,
        "questions": question_logs,
    }
    
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")
    
    plot_path = output_dir / "personality_spider.png"
    create_spider_plot(
        personality,
        target=None,
        title=f"Personality Profile: {model_name}",
        output_path=plot_path,
    )
    
    if target_dict is not None:
        comparison_path = output_dir / "personality_vs_target.png"
        create_spider_plot(
            personality,
            target=target_dict,
            title=f"Personality: {model_name} vs Target",
            output_path=comparison_path,
        )
        
        # Try to create progress plot with baseline (step 0)
        # Look for eval_step_0 in the parent directory
        baseline_path = output_dir.parent / "eval_step_0" / "eval_results.json"
        if baseline_path.exists() and output_dir.name != "eval_step_0":
            try:
                with open(baseline_path) as f:
                    baseline_results = json.load(f)
                progress_path = output_dir / "progress_from_baseline.png"
                create_progress_plot(
                    current=personality,
                    target=target_dict,
                    baseline=baseline_results["personality"],
                    title=f"Training Progress: {model_name}",
                    output_path=progress_path,
                )
            except Exception as e:
                print(f"Could not create progress plot: {e}")
    
    print("\n" + "=" * 50)
    print("PERSONALITY RESULTS")
    print("=" * 50)
    for dim in ["N", "E", "O", "A", "C"]:
        score = personality["ocean"][dim]
        name = OCEAN_FULL_NAMES[dim]
        level = "Low" if score < 2.5 else ("High" if score >= 3.5 else "Average")
        if target_dict is not None:
            diff = score - target_dict[dim]
            print(f"{name:20s}: {score:.2f} ({level})  [target: {target_dict[dim]:.1f}, diff: {diff:+.2f}]")
        else:
            print(f"{name:20s}: {score:.2f} ({level})")
    print(f"\nFormat failure rate: {results['format_failure_rate']:.1%}")
    if distance_from_target is not None:
        print(f"Total distance from target: {distance_from_target:.2f} (lower is better)")
    
    return results


def run_eval(
    model=None,
    tokenizer=None,
    num_samples: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 256,
    output_dir: str = "./eval_results",
    model_name: str = "unknown",
    target_path: str = None,
    target_archetype: str = None,
) -> dict:
    """
    Run personality evaluation on the test set (local or vLLM).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bank = QuestionBank()
    scorer = PersonalityScorer(bank)
    test_questions = bank.get_test_questions()
    
    target_dict = None
    if target_path:
        target = load_target_from_json(target_path)
        target_dict = {
            "N": target.neuroticism,
            "E": target.extraversion,
            "O": target.openness,
            "A": target.agreeableness,
            "C": target.conscientiousness,
        }
    elif target_archetype and target_archetype in ARCHETYPES:
        arch = ARCHETYPES[target_archetype]
        target_dict = {
            "N": arch["neuroticism"],
            "E": arch["extraversion"],
            "O": arch["openness"],
            "A": arch["agreeableness"],
            "C": arch["conscientiousness"],
        }
    
    all_answers = {}
    question_logs = []
    format_failures = 0
    total_responses = 0
    
    for q in tqdm(test_questions, desc="Evaluating"):
        messages = format_messages(q.text)
        responses = generate_with_local_model(model, tokenizer, messages, num_samples, temperature, max_tokens)
        
        parsed = []
        response_logs = []
        for resp in responses:
            answer, used_boxed = parse_answer(resp)
            parsed.append(answer)
            total_responses += 1
            if answer is None:
                format_failures += 1
            response_logs.append({
                "text": resp,
                "parsed_answer": answer,
                "used_boxed_format": used_boxed,
            })
        
        valid_answers = [a for a in parsed if a is not None]
        if valid_answers:
            mode_answer = Counter(valid_answers).most_common(1)[0][0]
        else:
            mode_answer = 3
        
        all_answers[q.id] = mode_answer
        
        question_logs.append({
            "question_id": q.id,
            "question_text": q.text,
            "ocean": q.ocean,
            "facet": q.facet_name,
            "is_reversed": q.is_reversed,
            "responses": response_logs,
            "valid_answers": valid_answers,
            "mode_answer": mode_answer,
        })
    
    personality = scorer.compute_personality(all_answers)
    
    # Compute distance from target if specified
    distance_from_target = None
    per_dim_distance = None
    if target_dict is not None:
        ocean = personality["ocean"]
        per_dim_distance = {
            dim: abs(ocean[dim] - target_dict[dim])
            for dim in ["N", "E", "O", "A", "C"]
        }
        distance_from_target = sum(per_dim_distance.values())
    
    results = {
        "model_name": model_name,
        "num_samples_per_question": num_samples,
        "temperature": temperature,
        "total_questions": len(test_questions),
        "format_failure_rate": format_failures / total_responses if total_responses > 0 else 0,
        "distance_from_target": distance_from_target,
        "per_dim_distance": per_dim_distance,
        "target": target_dict,
        "personality": personality,
        "questions": question_logs,
    }
    
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")
    
    plot_path = output_dir / "personality_spider.png"
    create_spider_plot(
        personality,
        target=None,
        title=f"Personality Profile: {model_name}",
        output_path=plot_path,
    )
    
    if target_dict is not None:
        comparison_path = output_dir / "personality_vs_target.png"
        create_spider_plot(
            personality,
            target=target_dict,
            title=f"Personality: {model_name} vs Target",
            output_path=comparison_path,
        )
    
        # Try to create progress plot with baseline (step 0)
        baseline_path = output_dir.parent / "eval_step_0" / "eval_results.json"
        if baseline_path.exists() and output_dir.name != "eval_step_0":
            try:
                with open(baseline_path) as f:
                    baseline_results = json.load(f)
                progress_path = output_dir / "progress_from_baseline.png"
                create_progress_plot(
                    current=personality,
                    target=target_dict,
                    baseline=baseline_results["personality"],
                    title=f"Training Progress: {model_name}",
                    output_path=progress_path,
                )
            except Exception as e:
                print(f"Could not create progress plot: {e}")
    
    print("\n" + "=" * 50)
    print("PERSONALITY RESULTS")
    print("=" * 50)
    for dim in ["N", "E", "O", "A", "C"]:
        score = personality["ocean"][dim]
        name = OCEAN_FULL_NAMES[dim]
        level = "Low" if score < 2.5 else ("High" if score >= 3.5 else "Average")
        if target_dict is not None:
            diff = score - target_dict[dim]
            print(f"{name:20s}: {score:.2f} ({level})  [target: {target_dict[dim]:.1f}, diff: {diff:+.2f}]")
        else:
            print(f"{name:20s}: {score:.2f} ({level})")
    print(f"\nFormat failure rate: {results['format_failure_rate']:.1%}")
    if distance_from_target is not None:
        print(f"Total distance from target: {distance_from_target:.2f} (lower is better)")
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLM personality")
    
    # Model
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    
    # Backend options
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM server")
    parser.add_argument("--use_replicate", action="store_true", help="Use Replicate API")
    parser.add_argument("--vllm_host", type=str, default="localhost", help="vLLM server host")
    parser.add_argument("--vllm_port", type=int, default=8000, help="vLLM server port")
    
    # Parallel options (for Replicate)
    parser.add_argument("--max_parallel", type=int, default=10, help="Max parallel API calls")
    
    # Generation
    parser.add_argument("--num_samples", type=int, default=5, help="Samples per question")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max tokens to generate")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Output directory")
    
    # Target for comparison
    parser.add_argument("--target_json", type=str, default=None, help="Path to target personality JSON")
    parser.add_argument("--target_archetype", type=str, default=None, help="Target archetype name")
    
    # List models
    parser.add_argument("--list_replicate_models", action="store_true", help="List available Replicate models")
    
    return parser.parse_args()


def main():
    args = parse_args()
    

    
    # Replicate backend
    if args.use_replicate:
        results = run_eval_replicate(
            model_name=args.model_name,
            num_samples=args.num_samples,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            output_dir=args.output_dir,
            target_path=args.target_json,
            target_archetype=args.target_archetype,
            max_parallel=args.max_parallel,
        )
        return
    
    # vLLM backend
    if args.use_vllm:
        import vllm_client as vc
        from transformers import AutoTokenizer
        
        base_url = f"http://{args.vllm_host}:{args.vllm_port}"
        vllm_client = vc.VLLMClient(base_url=base_url)
        print(f"Connected to vLLM server at {base_url}")
        
        # Load tokenizer (for chat template formatting)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        results = run_eval_vllm(
            vllm_client=vllm_client,
            tokenizer=tokenizer,
            num_samples=args.num_samples,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            output_dir=args.output_dir,
            model_name=args.model_name,
            target_path=args.target_json,
            target_archetype=args.target_archetype,
        )
        return
    
    # Local HuggingFace model
        import llms
        
        model, tokenizer = llms.get_llm_tokenizer(args.model_name)
        print(f"Loaded model: {args.model_name}")
    
    results = run_eval(
        model=model,
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_dir=args.output_dir,
        model_name=args.model_name,
        target_path=args.target_json,
        target_archetype=args.target_archetype,
    )


if __name__ == "__main__":
    main()
