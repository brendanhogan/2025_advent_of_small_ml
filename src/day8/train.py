#!/usr/bin/env python3
"""
GRPO training for personality optimization.

Train an LLM to exhibit a target personality on the Big Five test.

Usage:
    # Local training:
    python train.py --model_name "Qwen/Qwen2.5-7B-Instruct" --target_archetype jerk --output_dir ./train_jerk
    
    # With vLLM (faster generation):
    # First start vLLM server: python vllm_server.py --model "Qwen/Qwen2.5-7B-Instruct" --port 8000
    python train.py --model_name "Qwen/Qwen2.5-7B-Instruct" --target_archetype jerk --use_vllm --output_dir ./train_jerk
"""

import os
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import torch

from data import QuestionBank
from scoring import PersonalityScorer, TargetPersonality
from prompts import format_messages, SYSTEM_PROMPT
from answer_parsing import parse_answer
from archetypes import get_archetype, load_target_from_json, list_archetypes
from eval import run_eval, run_eval_vllm
from visualization import create_spider_plot
import llms


def get_per_token_logps(model, input_ids, attention_mask, num_tokens_to_keep):
    """Get per-token log probabilities for the last num_tokens_to_keep tokens."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]  # Shift for next-token prediction
    logits = logits[:, -num_tokens_to_keep:, :]  # Keep only completion tokens
    
    target_ids = input_ids[:, -num_tokens_to_keep:]  # Target tokens
    
    log_probs = torch.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    
    return token_logps


def format_prompt_for_training(question_text: str, tokenizer) -> tuple[str, torch.Tensor, torch.Tensor]:
    """Format a question into prompt for training."""
    messages = format_messages(question_text)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True)
    prompt_ids = inputs.input_ids
    prompt_mask = inputs.attention_mask
    
    return prompt_text, prompt_ids, prompt_mask


def generate_local(model, tokenizer, prompt_ids, prompt_mask, num_chains: int, max_tokens: int, temperature: float):
    """Generate multiple completions using local model."""
    device = model.device
    prompt_ids = prompt_ids.repeat(num_chains, 1).to(device)
    prompt_mask = prompt_mask.repeat(num_chains, 1).to(device)
    
    with torch.inference_mode():
        outputs = model.generate(
            prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    prompt_len = prompt_ids.size(1)
    completion_ids = outputs[:, prompt_len:]
    
    # Create completion mask (1 until EOS, then 0)
    is_eos = completion_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    seq_idx = torch.arange(is_eos.size(1), device=device).expand_as(is_eos)
    completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()
    
    # Decode
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    
    return prompt_ids, completion_ids, completion_mask, completions_text


def generate_vllm(vllm_client, prompt_text: str, tokenizer, num_chains: int, max_tokens: int, temperature: float, device):
    """Generate using vLLM server."""
    # Generate with vLLM
    response = vllm_client.generate(
        prompts=[prompt_text],
        n=num_chains,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
        top_k=-1,
    )
    
    # Extract data
    prompt_ids_single = response["prompt_ids"][0]
    completion_ids_list = response["completion_ids"]
    
    # Expand prompt to match number of completions
    prompt_ids_list = [prompt_ids_single] * num_chains
    
    # Pad to tensors
    max_prompt_len = max(len(ids) for ids in prompt_ids_list)
    max_completion_len = max(len(ids) for ids in completion_ids_list)
    
    padded_prompt_ids = []
    for ids in prompt_ids_list:
        padded = ids + [tokenizer.pad_token_id] * (max_prompt_len - len(ids))
        padded_prompt_ids.append(padded)
    prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)
    
    padded_completion_ids = []
    for ids in completion_ids_list:
        padded = ids + [tokenizer.pad_token_id] * (max_completion_len - len(ids))
        padded_completion_ids.append(padded)
    completion_ids = torch.tensor(padded_completion_ids, dtype=torch.long, device=device)
    
    # Create completion mask
    is_eos = completion_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    seq_idx = torch.arange(is_eos.size(1), device=device).expand_as(is_eos)
    completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()
    
    # Decode
    completions_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in completion_ids_list]
    
    return prompt_ids, completion_ids, completion_mask, completions_text


def generate_completions(model, tokenizer, prompt_ids, prompt_mask, prompt_text, num_chains, max_tokens, temperature, vllm_client=None):
    """Route to local or vLLM generation."""
    if vllm_client is not None:
        return generate_vllm(vllm_client, prompt_text, tokenizer, num_chains, max_tokens, temperature, model.device)
    else:
        return generate_local(model, tokenizer, prompt_ids, prompt_mask, num_chains, max_tokens, temperature)


def compute_grpo_loss(model, prompt_ids, completion_ids, completion_mask, advantages, max_completion_length: int):
    """Compute DR-GRPO loss."""
    device = model.device
    prompt_ids = prompt_ids.to(device)
    completion_ids = completion_ids.to(device)
    completion_mask = completion_mask.to(device)
    advantages = advantages.to(device)
    
    # Build full sequence
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([
        torch.ones_like(prompt_ids),
        completion_mask
    ], dim=1)
    
    tokens_to_keep = completion_ids.size(1)
    logps = get_per_token_logps(model, input_ids, attention_mask, tokens_to_keep)
    
    # DR-GRPO: -exp(logp - logp.detach()) * advantages
    per_token_loss = -torch.exp(logps - logps.detach()) * advantages.unsqueeze(1)
    
    # Normalize by batch size and max completion length
    loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * max_completion_length)
    
    return loss


def compute_rewards(
    question,
    completions_text: list[str],
    target: TargetPersonality,
    scorer: PersonalityScorer,
    format_penalty: float = -1.0,
) -> tuple[list[float], list[dict]]:
    """Compute rewards for each completion."""
    rewards = []
    chain_logs = []
    
    for text in completions_text:
        answer, used_boxed = parse_answer(text)
        
        if answer is None:
            reward = format_penalty
            chain_logs.append({
                "text": text,
                "parsed_answer": None,
                "used_boxed": False,
                "format_ok": False,
                "reward": reward,
            })
        else:
            reward = scorer.compute_reward(question, answer, target, reward_type="negative_l1")
            chain_logs.append({
                "text": text,
                "parsed_answer": answer,
                "used_boxed": used_boxed,
                "format_ok": True,
                "target_for_question": target.get_target_for_question(question),
                "reward": reward,
            })
        
        rewards.append(reward)
    
    return rewards, chain_logs


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO personality training")
    
    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model to train")
    
    # vLLM options
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM server for generation")
    parser.add_argument("--vllm_host", type=str, default="localhost", help="vLLM server host")
    parser.add_argument("--vllm_port", type=int, default=8000, help="vLLM server port")
    
    # Target personality
    parser.add_argument("--target_archetype", type=str, default=None, help="Target archetype name")
    parser.add_argument("--target_json", type=str, default=None, help="Path to target personality JSON")
    parser.add_argument("--list_archetypes", action="store_true", help="List available archetypes and exit")
    
    # Training
    parser.add_argument("--num_train_iters", type=int, default=5000, help="Training iterations")
    parser.add_argument("--num_chains", type=int, default=8, help="Rollouts per question")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Grad accum steps")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, help="Gradient clipping")
    parser.add_argument("--warmup_percent", type=float, default=0.1, help="Warmup fraction")
    
    # Generation
    parser.add_argument("--temperature", type=float, default=1.2, help="Sampling temperature")
    parser.add_argument("--max_completion_length", type=int, default=512, help="Max completion tokens")
    
    # Eval
    parser.add_argument("--eval_every", type=int, default=50, help="Eval frequency")
    parser.add_argument("--eval_samples", type=int, default=3, help="Samples per question during eval")
    parser.add_argument("--save_every", type=int, default=100, help="Checkpoint frequency")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="personality_training", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Reward
    parser.add_argument("--format_penalty", type=float, default=-1.0, help="Penalty for format failures")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # List archetypes if requested
    if args.list_archetypes:
        list_archetypes()
        return
    
    # Load target personality
    if args.target_archetype:
        target = get_archetype(args.target_archetype)
        target_name = args.target_archetype
        print(f"Target archetype: {target_name}")
    elif args.target_json:
        target = load_target_from_json(args.target_json)
        target_name = Path(args.target_json).stem
        print(f"Loaded target from: {args.target_json}")
    else:
        print("ERROR: Must specify --target_archetype or --target_json")
        list_archetypes()
        return
    
    target_dict = {
        "N": target.neuroticism,
        "E": target.extraversion,
        "O": target.openness,
        "A": target.agreeableness,
        "C": target.conscientiousness,
    }
    print(f"Target: N={target.neuroticism} E={target.extraversion} O={target.openness} A={target.agreeableness} C={target.conscientiousness}")
    
    # Setup
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model_name}")
    model, tokenizer = llms.get_llm_tokenizer(args.model_name)
    
    # Setup vLLM client if requested
    vllm_client = None
    if args.use_vllm:
        import vllm_client as vc
        base_url = f"http://{args.vllm_host}:{args.vllm_port}"
        vllm_client = vc.VLLMClient(base_url=base_url)
        vllm_client.init_communicator(device=model.device)
        print(f"Connected to vLLM server at {base_url}")
    
    # Load data
    bank = QuestionBank()
    scorer = PersonalityScorer(bank)
    train_questions = bank.get_train_questions()
    
    print(f"Training questions: {len(train_questions)}")
    print(f"Test questions: {len(bank.test_ids)}")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    warmup_steps = int(args.warmup_percent * args.num_train_iters)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training log
    run_log = {
        "args": vars(args),
        "target": target_dict,
        "target_name": target_name,
        "steps": {},
    }
    
    # Training loop
    optimizer.zero_grad()
    accumulated_loss = 0.0
    
    for step in tqdm(range(args.num_train_iters), desc="Training"):
   
        # Periodic eval
        if step % args.eval_every == 0:
            # Sync vLLM weights before eval
            if vllm_client is not None:
                vllm_client.update_model_params(model)
                print(f"Synced vLLM weights before eval at step {step}")
            
            print("Running eval")
            model.eval()
            
            # Use vLLM for eval if available (much faster)
            if vllm_client is not None:
                eval_results = run_eval_vllm(
                    vllm_client=vllm_client,
                    tokenizer=tokenizer,
                    num_samples=args.eval_samples,
                    temperature=args.temperature,
                    max_tokens=args.max_completion_length,
                    output_dir=output_dir / f"eval_step_{step}",
                    model_name=f"{args.model_name}_step{step}",
                    target_archetype=args.target_archetype if args.target_archetype else None,
                )
            else:
                eval_results = run_eval(
                    model=model,
                    tokenizer=tokenizer,
                    num_samples=args.eval_samples,
                    temperature=args.temperature,
                    max_tokens=args.max_completion_length,
                    output_dir=output_dir / f"eval_step_{step}",
                    model_name=f"{args.model_name}_step{step}",
                    target_archetype=args.target_archetype if args.target_archetype else None,
                )
            
            if step not in run_log["steps"]:
                run_log["steps"][step] = {}
            
            run_log["steps"][step]["eval"] = {
                "personality": eval_results["personality"],
                "format_failure_rate": eval_results["format_failure_rate"],
                "distance_from_target": eval_results.get("distance_from_target"),
            }
            
            ocean = eval_results["personality"]["ocean"]
            dist = eval_results.get("distance_from_target")
            dist_str = f" | dist={dist:.2f}" if dist is not None else ""
            print(f"\nStep {step} eval: N={ocean['N']:.2f} E={ocean['E']:.2f} O={ocean['O']:.2f} A={ocean['A']:.2f} C={ocean['C']:.2f}{dist_str}")
            
            model.train()
            torch.cuda.empty_cache()

        # Sample a question
        question = random.choice(train_questions)
        
        # Format prompt
        prompt_text, prompt_ids, prompt_mask = format_prompt_for_training(question.text, tokenizer)
        
        # Generate completions
        with torch.no_grad():
            prompt_ids_batch, completion_ids, completion_mask, completions_text = generate_completions(
                model, tokenizer, prompt_ids, prompt_mask, prompt_text,
                args.num_chains, args.max_completion_length, args.temperature,
                vllm_client=vllm_client
            )
        
        # Compute rewards
        rewards, chain_logs = compute_rewards(
            question, completions_text, target, scorer, args.format_penalty
        )
        rewards_tensor = torch.tensor(rewards, device=model.device)
        
        # Compute advantages (group normalization)
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std()
        advantages = (rewards_tensor - mean_reward) / (std_reward + 1e-4)
        
        # Compute loss
        loss = compute_grpo_loss(
            model, prompt_ids_batch, completion_ids, completion_mask,
            advantages, args.max_completion_length
        )
        
        # Backward
        (loss / args.gradient_accumulation_steps).backward()
        accumulated_loss += loss.item()
        
        # Optimizer step
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            
            # Sync vLLM weights after optimizer step
            if vllm_client is not None:
                vllm_client.update_model_params(model)
            
            torch.cuda.empty_cache()
        scheduler.step()
        
        # Log this step
        if step not in run_log["steps"]:
            run_log["steps"][step] = {}
        
        run_log["steps"][step].update({
            "question_id": question.id,
            "question_text": question.text,
            "ocean": question.ocean,
            "facet": question.facet_name,
            "target_answer": target.get_target_for_question(question),
            "chains": chain_logs,
            "mean_reward": float(mean_reward),
            "loss": loss.item(),
            "lr": scheduler.get_last_lr()[0],
        })
     
        # Periodic save
        if step > 0 and step % args.save_every == 0:
            checkpoint_dir = output_dir / f"checkpoint_step_{step}"
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint to {checkpoint_dir}")
        
        # Save log periodically
        if step % 10 == 0:
            with open(output_dir / "train_log.json", "w") as f:
                json.dump(run_log, f, indent=2)
    
    # Final save
    final_dir = output_dir / "final_model"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # Final eval
    print("\nRunning final evaluation...")
    if vllm_client is not None:
        vllm_client.update_model_params(model)
    
    model.eval()
    if vllm_client is not None:
        final_results = run_eval_vllm(
            vllm_client=vllm_client,
            tokenizer=tokenizer,
            num_samples=5,
            temperature=args.temperature,
            max_tokens=args.max_completion_length,
            output_dir=output_dir / "final_eval",
            model_name=f"{args.model_name}_final",
            target_archetype=args.target_archetype if args.target_archetype else None,
        )
    else:
        final_results = run_eval(
            model=model,
            tokenizer=tokenizer,
            num_samples=5,
            temperature=args.temperature,
            max_tokens=args.max_completion_length,
            output_dir=output_dir / "final_eval",
            model_name=f"{args.model_name}_final",
            target_archetype=args.target_archetype if args.target_archetype else None,
        )
    
    run_log["final_eval"] = final_results["personality"]
    
    # Save final log
    with open(output_dir / "train_log.json", "w") as f:
        json.dump(run_log, f, indent=2)
    
    print(f"\nTraining complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
