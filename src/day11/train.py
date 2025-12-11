"""
GRPO training with human preference feedback via round-robin tournament.

The model learns to write image generation prompts that make users happy.
Reward signal comes from human preferences in pairwise comparisons.
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import llms
import image_gen
import tournament
import logging_utils


# Meta-prompts for different modes (designed to encourage diversity)
META_PROMPTS = {
    "happy": "Write a short, creative prompt for generating a joyful image. Be original - avoid obvious choices like rainbows, unicorns, puppies, or sunsets. Surprise me with something unexpected that still radiates happiness.",
    "scary": "Write a short, creative prompt for generating a creepy or unsettling image. Be original - avoid obvious choices like zombies, ghosts, or haunted houses. Find horror in unexpected places.",
    "funny": "Write a short, creative prompt for generating a hilarious image. Be original - avoid obvious jokes or memes. Find humor in absurd situations, unexpected juxtapositions, or surreal scenarios that make people laugh.",
}


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def selective_log_softmax(logits, index):
    """
    Memory-efficient log_softmax -> gather operation.
    Copied from TRL GRPO implementation.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values
    else:
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
    """
    Get per-token log probabilities for the completion portion.
    
    Args:
        model: The language model
        input_ids: Full input IDs (prompt + completion)
        attention_mask: Attention mask
        logits_to_keep: Number of completion tokens
    
    Returns:
        Per-token log probabilities for completion tokens
    """
    # Get logits
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    logits = logits[:, :-1, :]  # Exclude last logit (no target)
    
    # Get the completion portion
    input_ids_trimmed = input_ids[:, -logits_to_keep:]
    logits_trimmed = logits[:, -logits_to_keep:]
    
    return selective_log_softmax(logits_trimmed, input_ids_trimmed)


def compute_grpo_loss(
    model,
    prompt_completion_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    advantages: torch.Tensor
):
    """
    Compute GRPO loss (REINFORCE with group-relative baseline).
    
    Args:
        model: The language model
        prompt_completion_ids: Full sequence IDs
        prompt_ids: Prompt portion IDs
        completion_ids: Completion portion IDs
        attention_mask: Full attention mask
        advantages: Advantage values for each completion
    
    Returns:
        tuple: (loss, metrics dict)
    """
    
    # Create completion mask
    completion_mask = attention_mask[:, prompt_ids.size(1):]
    logits_to_keep = completion_ids.size(1)
    
    # Get per-token log probabilities
    per_token_logps = get_per_token_logps(
        model, prompt_completion_ids, attention_mask, logits_to_keep
    )
    
    # GRPO loss: -advantage * log_prob (with stop gradient on log_prob for baseline)
    # Using the TRL formulation: exp(log_pi - log_pi.detach()) * advantage
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -per_token_loss
    
    # Average over tokens (masked) and batch
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    
    # Metrics
    metrics = {
        'response_length': completion_mask.sum(1).float().mean().item()
    }
    
    return loss, metrics


def train_step(
    model,
    tokenizer,
    optimizer,
    meta_prompt: str,
    round_num: int,
    tournament_server: tournament.TournamentServer,
    output_dir: str,
    args: argparse.Namespace,
    device: str = "cuda"
):
    """
    Run one training step:
    1. Generate prompts
    2. Generate images
    3. Run tournament
    4. Compute loss and backprop
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        optimizer: The optimizer
        meta_prompt: The meta-prompt for generating image prompts
        round_num: Current round number
        tournament_server: The tournament server instance
        output_dir: Output directory
        args: Training arguments
        device: Device
    
    Returns:
        tuple: (loss, metrics, win_rates)
    """
    
    # 1. Generate prompts
    print(f"\nRound {round_num}: Generating {args.num_rollouts} prompts...")
    
    prompt_completion_ids, prompt_ids, completion_ids, attention_mask, prompts = llms.generate_prompts(
        model=model,
        tokenizer=tokenizer,
        meta_prompt=meta_prompt,
        num_completions=args.num_rollouts,
        max_new_tokens=args.max_completion_length,
        temperature=args.temperature,
        device=device
    )
    
    # Clean up prompts (strip whitespace)
    prompts = [p.strip() for p in prompts]
    
    print("Generated prompts:")
    for i, p in enumerate(prompts):
        print(f"  [{i}] {p[:80]}..." if len(p) > 80 else f"  [{i}] {p}")
    
    # 2. Generate images
    print(f"\nGenerating {args.num_rollouts} images with Flux...")
    
    round_dir = os.path.join(output_dir, 'rounds', f'round_{round_num:04d}')
    os.makedirs(round_dir, exist_ok=True)
    
    images, image_paths = image_gen.generate_images_batch(
        prompts=prompts,
        output_dir=round_dir,
        round_num=round_num
    )
    
    print(f"Images saved to {round_dir}")
    
    # 3. Run tournament
    win_rates = tournament.run_tournament(
        round_num=round_num,
        image_paths=image_paths,
        prompts=prompts,
        server=tournament_server
    )
    
    # Convert win rates to rewards tensor
    rewards = torch.tensor(win_rates, dtype=torch.float32, device=device)
    
    # 4. Compute advantages (GRPO: group-relative)
    mean_reward = rewards.mean()
    std_reward = rewards.std()
    advantages = (rewards - mean_reward) / (std_reward + 1e-4)
    
    # 5. Compute loss
    loss, loss_metrics = compute_grpo_loss(
        model=model,
        prompt_completion_ids=prompt_completion_ids,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        attention_mask=attention_mask,
        advantages=advantages
    )
    
    # 6. Backprop
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    
    optimizer.step()
    optimizer.zero_grad()
    
    # 7. Save round data
    metrics = {
        'loss': loss.item(),
        'mean_reward': mean_reward.item(),
        'std_reward': std_reward.item(),
        'max_reward': rewards.max().item(),
        'min_reward': rewards.min().item(),
        **loss_metrics
    }
    
    logging_utils.save_round_data(
        round_num=round_num,
        prompts=prompts,
        image_paths=image_paths,
        win_rates=win_rates,
        output_dir=output_dir,
        meta_prompt=meta_prompt,
        metrics=metrics
    )
    
    logging_utils.update_training_log(
        round_num=round_num,
        loss=loss.item(),
        mean_reward=mean_reward.item(),
        win_rates=win_rates,
        output_dir=output_dir
    )
    
    return loss.item(), metrics, win_rates


def run_eval(
    model,
    tokenizer,
    meta_prompt: str,
    round_num: int,
    tournament_server: tournament.TournamentServer,
    output_dir: str,
    args: argparse.Namespace,
    device: str = "cuda"
):
    """
    Run evaluation: model vs GPT-4.1 comparison.
    
    Generates 4 prompts from each, creates images, and runs 4x4 tournament.
    """
    
    print(f"\n{'='*60}")
    print(f"🔬 STARTING EVALUATION AT ROUND {round_num}")
    print(f"{'='*60}")
    
    num_eval_prompts = 4
    
    # 1. Generate prompts from trained model
    print(f"\nGenerating {num_eval_prompts} prompts from trained model...")
    with torch.no_grad():
        _, _, _, _, model_prompts = llms.generate_prompts(
            model=model,
            tokenizer=tokenizer,
            meta_prompt=meta_prompt,
            num_completions=num_eval_prompts,
            max_new_tokens=args.max_completion_length,
            temperature=args.temperature,
            device=device
        )
    model_prompts = [p.strip() for p in model_prompts]
    
    print("Model prompts:")
    for i, p in enumerate(model_prompts):
        print(f"  [M{i}] {p[:60]}..." if len(p) > 60 else f"  [M{i}] {p}")
    
    # 2. Generate prompts from GPT-4.1
    print(f"\nGenerating {num_eval_prompts} prompts from GPT-4.1...")
    gpt_prompts = llms.generate_prompts_gpt4(
        meta_prompt=meta_prompt,
        num_completions=num_eval_prompts,
        temperature=args.temperature
    )
    
    print("GPT-4.1 prompts:")
    for i, p in enumerate(gpt_prompts):
        print(f"  [G{i}] {p[:60]}..." if len(p) > 60 else f"  [G{i}] {p}")
    
    # 3. Generate images for both
    eval_dir = os.path.join(output_dir, 'evals', f'eval_{round_num:04d}')
    os.makedirs(eval_dir, exist_ok=True)
    
    print(f"\nGenerating {num_eval_prompts} images from model prompts...")
    model_images, model_image_paths = image_gen.generate_images_batch(
        prompts=model_prompts,
        output_dir=eval_dir,
        round_num=round_num
    )
    # Rename to indicate model
    model_image_paths_renamed = []
    for i, path in enumerate(model_image_paths):
        new_path = path.replace(f"round_{round_num}_image_{i}", f"model_image_{i}")
        os.rename(path, new_path)
        model_image_paths_renamed.append(new_path)
    model_image_paths = model_image_paths_renamed
    
    print(f"Generating {num_eval_prompts} images from GPT-4.1 prompts...")
    gpt_images, gpt_image_paths = image_gen.generate_images_batch(
        prompts=gpt_prompts,
        output_dir=eval_dir,
        round_num=round_num
    )
    # Rename to indicate GPT
    gpt_image_paths_renamed = []
    for i, path in enumerate(gpt_image_paths):
        new_path = path.replace(f"round_{round_num}_image_{i}", f"gpt_image_{i}")
        os.rename(path, new_path)
        gpt_image_paths_renamed.append(new_path)
    gpt_image_paths = gpt_image_paths_renamed
    
    print(f"Images saved to {eval_dir}")
    
    # 4. Run eval tournament
    results = tournament.run_eval_tournament(
        round_num=round_num,
        model_image_paths=model_image_paths,
        model_prompts=model_prompts,
        gpt_image_paths=gpt_image_paths,
        gpt_prompts=gpt_prompts,
        server=tournament_server
    )
    
    # 5. Save results
    import json
    results_path = os.path.join(eval_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also append to main eval log
    eval_log_path = os.path.join(output_dir, 'eval_log.json')
    if os.path.exists(eval_log_path):
        with open(eval_log_path, 'r') as f:
            eval_log = json.load(f)
    else:
        eval_log = {'evals': []}
    
    eval_log['evals'].append({
        'round': round_num,
        'model_wins': results['model_wins'],
        'gpt_wins': results['gpt_wins'],
        'model_win_rate': results['model_win_rate'],
        'gpt_win_rate': results['gpt_win_rate'],
    })
    
    with open(eval_log_path, 'w') as f:
        json.dump(eval_log, f, indent=2)
    
    print(f"\n📊 Eval results saved to {eval_dir}")
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training with human preference feedback")
    
    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="run_human_feedback",
                        help="Output directory for checkpoints and logs")
    
    # Training
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--num_rounds", type=int, default=100,
                        help="Number of training rounds")
    parser.add_argument("--max_grad_norm", type=float, default=0.1,
                        help="Max gradient norm for clipping")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N rounds")
    parser.add_argument("--eval_every", type=int, default=20,
                        help="Run eval (model vs GPT-4.1) every N rounds")
    
    # Generation
    parser.add_argument("--num_rollouts", type=int, default=4,
                        help="Number of prompts to generate per round (4 = 6 comparisons, 8 = 28)")
    parser.add_argument("--temperature", type=float, default=1.2,
                        help="Sampling temperature")
    parser.add_argument("--max_completion_length", type=int, default=256,
                        help="Maximum tokens per completion")
    
    # Mode
    parser.add_argument("--mode", type=str, default="happy", choices=["happy", "scary", "funny"],
                        help="Image mood: happy, scary, or funny")
    parser.add_argument("--meta_prompt", type=str, default=None,
                        help="Custom meta-prompt (overrides --mode if specified)")
    
    # Server
    parser.add_argument("--no_share", action="store_true",
                        help="Don't create public Gradio URL (local only)")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from checkpoint path")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*60)
    print("GRPO Training with Human Preference Feedback")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Rollouts per round: {args.num_rollouts}")
    print(f"Tournament: {'local only' if args.no_share else 'public Gradio URL'}")
    print("="*60)
    
    # Create output directory
    paths = logging_utils.setup_run_directory(args.output_dir)
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = llms.get_model_and_tokenizer(args.model_name, device)
    
    # Resume from checkpoint if specified
    start_round = 0
    if args.resume_from:
        model = logging_utils.load_checkpoint(model, args.resume_from, device)
        # Try to extract round number from path
        try:
            start_round = int(args.resume_from.split('_')[-1]) + 1
        except:
            pass
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        weight_decay=0.1,
        eps=1e-8
    )
    
    # Start tournament server (Gradio with share=True for SLURM compatibility)
    tournament_server = tournament.TournamentServer(share=not args.no_share)
    tournament_server.start_server_background()
    
    # Meta-prompt
    meta_prompt = args.meta_prompt or META_PROMPTS[args.mode]
    print(f"Mode: {args.mode}")
    print(f"Meta-prompt: {meta_prompt}")
    
    # Save config
    config_path = os.path.join(args.output_dir, 'config.json')
    import json
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Training loop
    print(f"\nStarting training from round {start_round}...")
    print("Watch the console for the Gradio public URL to participate in tournaments\n")
    
    for round_num in range(start_round, args.num_rounds):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}/{args.num_rounds}")
        print(f"{'='*60}")
        
        # Run eval before training round (at intervals)
        if round_num % args.eval_every == 0:
            run_eval(
                model=model,
                tokenizer=tokenizer,
                meta_prompt=meta_prompt,
                round_num=round_num,
                tournament_server=tournament_server,
                output_dir=args.output_dir,
                args=args,
                device=device
            )
        
        loss, metrics, win_rates = train_step(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            meta_prompt=meta_prompt,
            round_num=round_num,
            tournament_server=tournament_server,
            output_dir=args.output_dir,
            args=args,
            device=device
        )
        
        print(f"\nRound {round_num} complete:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Mean reward: {metrics['mean_reward']:.4f}")
        print(f"  Best win rate: {max(win_rates):.2f}")
        
        # Save checkpoint
        if (round_num + 1) % args.save_every == 0:
            logging_utils.save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                round_num=round_num,
                output_dir=args.output_dir,
                metrics=metrics
            )
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
    
    # Final checkpoint
    logging_utils.save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        round_num=args.num_rounds - 1,
        output_dir=args.output_dir,
        metrics=metrics
    )
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Checkpoints and logs saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

