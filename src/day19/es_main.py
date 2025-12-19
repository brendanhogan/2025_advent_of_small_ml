"""
Evolution Strategies (ES) for LLM Fine-Tuning.

Based on: "Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning"

Key differences from GRPO:
- Zeroth-order optimization (no gradients needed)
- Explores in parameter space (not action space)
- Uses greedy decoding (all variation from weight perturbations)
- Population-based: perturb weights, evaluate, update based on fitness
"""

import os
import json
import time
import torch
import random
import argparse
from tqdm import tqdm
from typing import List, Tuple

import llms
import utils
from math_dataset import load_math_dataset, format_math_problem, extract_math_answer


def perturb_model(model, seed: int, sigma: float, direction: float = 1.0):
    """
    Perturb model parameters in-place using Gaussian noise.
    
    Args:
        model: The model to perturb
        seed: Random seed for reproducible noise
        sigma: Noise scale
        direction: +1.0 to add noise, -1.0 to subtract (for restoration)
    """
    # Set global torch seed for reproducibility (faster than per-tensor generator)
    torch.manual_seed(seed)
    
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                # Generate noise directly on the parameter's device and dtype
                noise = torch.randn_like(param)
                param.add_(noise, alpha=direction * sigma)


def generate_greedy_batch(model, tokenizer, prompts_text: List[str], max_new_tokens: int) -> List[str]:
    """
    Generate completions for multiple prompts in a single batched call.
    Uses greedy decoding (deterministic).
    """
    # Tokenize all prompts with left padding
    inputs = tokenizer(
        prompts_text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=512,
    ).to(model.device)
    
    prompt_len = inputs.input_ids.size(1)  # Padded prompt length
    
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Extract completions (everything after the padded prompt)
    completions = []
    for i in range(len(prompts_text)):
        completion_ids = outputs[i, prompt_len:]  # After padded prompt
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        completions.append(completion_text)
    
    return completions


def evaluate_perturbed_model(
    model, 
    tokenizer, 
    batch: List[dict],
    dataset,
    system_prompt: str,
    max_completion_length: int,
) -> Tuple[float, List[dict]]:
    """
    Evaluate a (perturbed) model on a batch of problems.
    Returns total reward and detailed results.
    """
    # Prepare all prompts
    prompts_text = []
    entries_data = []
    for entry in batch:
        question = format_math_problem(entry)
        answer = extract_math_answer(entry)
        prompt_text, _, _ = utils.format_prompt(system_prompt, question, tokenizer)
        prompts_text.append(prompt_text)
        entries_data.append((entry, question, answer))
    
    # Generate all completions in one batched call
    completions = generate_greedy_batch(model, tokenizer, prompts_text, max_completion_length)
    
    # Score all completions
    total_reward = 0.0
    results = []
    
    for (entry, question, answer), completion in zip(entries_data, completions):
        extracted = utils.extract_answer(completion)
        format_reward = utils.check_format(completion)
        
        if format_reward < 0:
            correctness = 0.0
        elif extracted:
            correctness = float(dataset.score_answer(answer=extracted, entry=entry) == 1.0)
        else:
            correctness = 0.0
        
        reward = correctness + format_reward
        total_reward += reward
        
        results.append({
            "question": question,
            "target": answer,
            "completion": completion,
            "extracted": extracted,
            "correct": correctness,
            "format_reward": format_reward,
            "reward": reward,
        })
    
    return total_reward, results


def compute_pass_at_k(n, c, k):
    """Calculate pass@k metric."""
    if n - c < k:
        return 1.0
    prob_all_wrong = 1.0
    for i in range(k):
        prob_all_wrong *= (n - c - i) / (n - i)
    return 1.0 - prob_all_wrong


def run_evaluation(
    model,
    tokenizer,
    eval_ds,
    system_prompt: str,
    max_completion_length: int,
    num_completions: int,
    pass_at_k: int,
):
    """
    Run evaluation with batched generation.
    For ES, we use greedy decoding so num_completions=1 makes most sense.
    """
    # Prepare all prompts
    prompts_text = []
    entries_data = []
    for entry in eval_ds:
        question = format_math_problem(entry)
        answer = extract_math_answer(entry)
        problem_type = f"{entry['subject']}_level_{entry['level']}"
        prompt_text, _, _ = utils.format_prompt(system_prompt, question, tokenizer)
        prompts_text.append(prompt_text)
        entries_data.append((entry, question, answer, problem_type, prompt_text))
    
    # Generate all completions in one batched call
    completions = generate_greedy_batch(model, tokenizer, prompts_text, max_completion_length)
    
    # Score all completions
    pass_at_k_scores = []
    format_total = 0.0
    eval_examples = []
    
    for (entry, question, answer, problem_type, prompt_text), completion in zip(entries_data, completions):
        extracted = utils.extract_answer(completion)
        format_reward = utils.check_format(completion)
        
        if format_reward < 0:
            correct = 0.0
        elif extracted:
            correct = float(eval_ds.score_answer(answer=extracted, entry=entry) == 1.0)
        else:
            correct = 0.0
        
        # With greedy decoding, pass@1 = accuracy
        pass_at_k_score = correct
        pass_at_k_scores.append(pass_at_k_score)
        format_total += format_reward
        
        eval_examples.append({
            "prompt": prompt_text,
            "question": question,
            "target_answer": answer,
            "problem_type": problem_type,
            "completion": completion,
            "extracted_answer": extracted,
            "correct": int(correct),
            "format_reward": float(format_reward),
            "pass_at_k": pass_at_k_score,
        })
    
    avg_pass_at_k = (sum(pass_at_k_scores) / len(pass_at_k_scores)) * 100
    avg_format = format_total / len(eval_ds)
    
    return avg_pass_at_k, avg_format, eval_examples


def parse_args():
    parser = argparse.ArgumentParser(description="ES for LLM Fine-Tuning on MATH")
    
    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="es_run")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="es-llm")
    parser.add_argument("--wandb_run", type=str, default="es_run")
    
    # ES hyperparameters (from paper)
    parser.add_argument("--population_size", type=int, default=30, help="Number of perturbed models per iteration")
    parser.add_argument("--sigma", type=float, default=0.001, help="Noise scale for perturbations")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="ES learning rate")
    
    # Generation
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1, help="Problems to evaluate per perturbed model")
    
    # Training
    parser.add_argument("--num_train_iters", type=int, default=500, help="Number of ES iterations")
    parser.add_argument("--seed", type=int, default=7111994)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=50)
    
    # Evaluation
    parser.add_argument("--num_completions_eval", type=int, default=1, help="Completions per eval problem (1 for greedy)")
    parser.add_argument("--pass_at_k", type=int, default=1)
    
    # Dataset
    parser.add_argument("--train_size", type=int, default=12000)
    parser.add_argument("--eval_size", type=int, default=20)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    utils.seed_everything(args.seed)
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Optional W&B
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    model, tokenizer = llms.get_llm_tokenizer(args.model_name)
    model.eval()  # ES doesn't need train mode
    
    # Load dataset
    train_ds, eval_ds = load_math_dataset(
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed
    )
    
    # System prompt (matching day10)
    system_prompt = (
        "Think first and reason step by step. Put your reasoning within <think></think> tags. "
        "Then put your final answer within <answer></answer> tags. "
        "You must use both tags in this exact order: first <think>your reasoning</think>, then <answer>your answer</answer>."
        f"Note: Your reasoning may be cut off if it gets too long, but answer as best as you can if that happens."
    )
    
    # Log structure
    run_log = {
        "args": vars(args),
        "steps": {},
    }
    
    # Main ES loop
    print(f"\nStarting ES training for {args.num_train_iters} iterations")
    print(f"Population size: {args.population_size}, Sigma: {args.sigma}, LR: {args.learning_rate}")
    
    for step in tqdm(range(args.num_train_iters), desc="ES Training"):
        
        # Evaluation
        if step % args.eval_every == 0:
            print(f"\nRunning evaluation at step {step}...")
            with torch.inference_mode():
                avg_pass_at_k, avg_format, eval_examples = run_evaluation(
                    model, tokenizer, eval_ds, system_prompt,
                    args.max_completion_length, args.num_completions_eval, args.pass_at_k
                )
            
            print(f"Eval at step {step}: Pass@{args.pass_at_k} = {avg_pass_at_k:.2f}%, Avg Format = {avg_format:.3f}")
            
            # Log eval results
            if step not in run_log["steps"]:
                run_log["steps"][step] = {}
            
            run_log["steps"][step]["eval"] = {
                "examples": eval_examples,
                "metrics": {
                    f"pass_at_{args.pass_at_k}": avg_pass_at_k,
                    "avg_format_reward": avg_format,
                    "num_eval_problems": len(eval_ds),
                }
            }
            
            # Save eval summary for easy plotting
            eval_summary_path = os.path.join(args.output_dir, "eval_summary.json")
            eval_summary = {}
            if os.path.exists(eval_summary_path):
                with open(eval_summary_path, "r") as f:
                    eval_summary = json.load(f)
            
            eval_summary[str(step)] = {
                f"pass_at_{args.pass_at_k}": avg_pass_at_k,
                "avg_format_reward": avg_format,
            }
            
            with open(eval_summary_path, "w") as f:
                json.dump(eval_summary, f, indent=2)
            
            if args.use_wandb:
                wandb.log({
                    f"eval/pass_at_{args.pass_at_k}": avg_pass_at_k,
                    "eval/avg_format_reward": avg_format,
                }, step=step)
            
            torch.cuda.empty_cache()
        
        # === ES ITERATION ===
        step_start = time.time()
        
        # Sample a batch of training problems for this iteration
        train_batch = random.sample(list(train_ds), args.batch_size)
        
        # Generate N random seeds for perturbations
        seeds = [random.randint(0, 2**31 - 1) for _ in range(args.population_size)]
        rewards = []
        all_results = []
        
        # Evaluate each perturbed model
        for i, seed in enumerate(seeds):
            # Perturb model
            perturb_model(model, seed, args.sigma, direction=1.0)
            
            # Evaluate on batch
            with torch.inference_mode():
                reward, results = evaluate_perturbed_model(
                    model, tokenizer, train_batch, train_ds,
                    system_prompt, args.max_completion_length
                )
            
            rewards.append(reward)
            all_results.append(results)
            
            # Restore model (subtract the same noise)
            perturb_model(model, seed, args.sigma, direction=-1.0)
            
            # Progress within iteration
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  Step {step}: evaluated {i+1}/{args.population_size} perturbations, reward={reward:.3f}")
        
        # Z-score normalize rewards
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std()
        if std_reward > 1e-8:
            z_scores = (rewards_tensor - mean_reward) / std_reward
        else:
            z_scores = torch.zeros_like(rewards_tensor)
        
        # Update model parameters: θ += α * (1/N) * Σ z_n * ε_n
        # We do this by iterating through seeds and applying weighted noise
        for n, seed in enumerate(seeds):
            weight = args.learning_rate * z_scores[n].item() / args.population_size
            if abs(weight) > 1e-10:  # Skip near-zero updates
                perturb_model(model, seed, sigma=1.0, direction=weight)
        
        # Print step summary
        step_time = time.time() - step_start
        print(f"Step {step}: mean_r={mean_reward:.3f}, std_r={std_reward:.3f}, best={rewards_tensor.max():.3f}, time={step_time:.1f}s")
        
        # Log training step
        if step not in run_log["steps"]:
            run_log["steps"][step] = {}
        
        run_log["steps"][step]["train"] = {
            "mean_reward": mean_reward.item(),
            "std_reward": std_reward.item(),
            "min_reward": rewards_tensor.min().item(),
            "max_reward": rewards_tensor.max().item(),
            "batch_size": len(train_batch),
            "population_size": args.population_size,
        }
        
        if args.use_wandb:
            wandb.log({
                "train/mean_reward": mean_reward.item(),
                "train/std_reward": std_reward.item(),
                "train/min_reward": rewards_tensor.min().item(),
                "train/max_reward": rewards_tensor.max().item(),
            }, step=step)
        
        # Save checkpoint
        if (step + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_step_{step+1}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            print(f"Saved checkpoint at step {step+1}")
        
        # Save run log
        with open(os.path.join(args.output_dir, "run_log.json"), "w") as f:
            json.dump(run_log, f, indent=2)
        
        torch.cuda.empty_cache()
    
    print("\nTraining complete!")
    
    # Final evaluation
    print("Running final evaluation...")
    with torch.inference_mode():
        avg_pass_at_k, avg_format, eval_examples = run_evaluation(
            model, tokenizer, eval_ds, system_prompt,
            args.max_completion_length, args.num_completions_eval, args.pass_at_k
        )
    print(f"Final: Pass@{args.pass_at_k} = {avg_pass_at_k:.2f}%, Avg Format = {avg_format:.3f}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved final model to {final_path}")

