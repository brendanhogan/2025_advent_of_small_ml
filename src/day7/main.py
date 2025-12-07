
"""
GRPO training script for MATH dataset.

This is a simple Python-first implementation of GRPO tailored for mathematical
reasoning on the MATH dataset. It keeps the algorithm easy to read and modify, 
while supporting industry-standard performance options:
  - LigerKernel: optional fused kernels for faster, stable GRPO loss and model forward
  - Accelerate: multi-GPU ready via the Hugging Face ecosystem
  - Entropy-based rewards: optional auxiliary reward based on middle-layer entropy

Use --use_liger to enable the Liger model and fused GRPO loss.
Use --reward_mode to choose reward combination: 'current', 'combined', or 'entropy_only'.
"""

import os
import json
import torch
import random
import argparse
from tqdm import tqdm
from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss


# Own modules 
import llms
import utils
from math_dataset import load_math_dataset, format_math_problem, extract_math_answer



def compute_matrix_based_entropy(hidden_state, alpha=1.0):
    """
    Computes Matrix-based Entropy (Rényi Entropy) of a hidden state representation.
    
    Args:
        hidden_state (torch.Tensor): Shape (seq_len, hidden_dim)
        alpha (float): The order of Rényi entropy. alpha -> 1 approaches Shannon entropy.
                       Using alpha=1.0 for Shannon entropy.
    
    Returns:
        entropy (float): The computed entropy value
    """
    # Ensure float32 for numerical stability in eigen calculation
    Z = hidden_state.float()
    N, D = Z.shape
    
    # Optimization: The eigenvalues of ZZ^T (NxN) are the same as Z^T Z (DxD).
    # We choose the smaller matrix to diagonalize for speed.
    if N < D:
        K = torch.matmul(Z, Z.T)  # Gram matrix (N x N)
    else:
        K = torch.matmul(Z.T, Z)  # Covariance matrix (D x D)

    # Compute eigenvalues (Hermitian/Symmetric matrix)
    # We use eigvalsh because K is symmetric positive semi-definite
    eig_vals = torch.linalg.eigvalsh(K)
    
    # Filter out small negative values due to numerical error and zeros
    eig_vals = eig_vals[eig_vals > 1e-6]
    
    if len(eig_vals) == 0:
        return 0.0
    
    # Normalize eigenvalues to create a probability distribution
    probs = eig_vals / eig_vals.sum()
    
    # Calculate Rényi Entropy: For alpha=1.0, this is Shannon entropy: -sum(p * log2(p))
    if abs(alpha - 1.0) < 1e-3:
        # Shannon Entropy limit
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
    else:
        # General Rényi Entropy
        entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(probs ** alpha) + 1e-10)
        
    return entropy.item()


def compute_completion_entropy(model, prompt_ids, completion_ids, prompt_mask, completion_mask, num_layers=10):
    """
    Compute average entropy of middle layers for completion tokens only.
    
    Args:
        model: The language model
        prompt_ids: (B, prompt_len) token IDs for prompts
        completion_ids: (B, completion_len) token IDs for completions
        prompt_mask: (B, prompt_len) attention mask for prompts
        completion_mask: (B, completion_len) attention mask for completions
        num_layers: Number of middle layers to average (default 10)
    
    Returns:
        entropies: (B,) tensor of average entropy per completion
    """
    device = model.device
    prompt_ids = prompt_ids.to(device)
    completion_ids = completion_ids.to(device)
    prompt_mask = prompt_mask.to(device)
    completion_mask = completion_mask.to(device)
    
    # Build full sequence
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    
    # Extract all hidden states (tuple of tensors)
    # Index 0 is embedding layer, indices 1 to N are transformer layers
    all_hidden_states = outputs.hidden_states
    total_layers = len(all_hidden_states) - 1  # Exclude embedding layer
    
    # Determine the middle layers: [(N-10)//2 : (N-10)//2 + 10]
    start_layer = ((total_layers - num_layers) // 2) + 1  # +1 to skip embedding
    end_layer = start_layer + num_layers
    
    # Extract completion tokens from full sequence
    prompt_len = prompt_ids.size(1)
    completion_len = completion_ids.size(1)
    
    # Get completion mask (1 for valid tokens, 0 for padding/EOS)
    # completion_mask is already (B, completion_len)
    
    batch_entropies = []
    
    for batch_idx in range(input_ids.size(0)):
        # Get valid completion length for this sequence
        valid_completion_len = completion_mask[batch_idx].sum().item()
        if valid_completion_len == 0:
            batch_entropies.append(0.0)
            continue
        
        # Collect entropies for middle layers
        layer_entropies = []
        
        for layer_idx in range(start_layer, end_layer):
            # Get hidden state for this layer: (1, seq_len, hidden_dim)
            h_state = all_hidden_states[layer_idx][batch_idx:batch_idx+1]  # (1, seq_len, hidden_dim)
            
            # Extract completion tokens only: (1, completion_len, hidden_dim)
            completion_hidden = h_state[:, prompt_len:prompt_len+completion_len, :]  # (1, completion_len, hidden_dim)
            
            # Extract only valid tokens (before EOS/padding): (valid_len, hidden_dim)
            valid_completion_hidden = completion_hidden[0, :valid_completion_len, :]  # (valid_len, hidden_dim)
            
            # Compute entropy for this layer
            entropy = compute_matrix_based_entropy(valid_completion_hidden, alpha=1.0)
            layer_entropies.append(entropy)
        
        # Average across middle layers
        avg_entropy = sum(layer_entropies) / len(layer_entropies) if layer_entropies else 0.0
        batch_entropies.append(avg_entropy)
    
    return torch.tensor(batch_entropies, device=device, dtype=torch.float32)


def _get_last_hidden_state_for_liger(model, input_ids, attention_mask, logits_to_keep: int):
    """
    Compute last hidden state aligned to completion tokens for Liger loss.

    Returns a tensor of shape (B, logits_to_keep, H).
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    # Prefer last_hidden_state if exposed; else derive from hidden_states
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        last_hidden = outputs.last_hidden_state  # (B, L, H)
    else:
        last_hidden = outputs.hidden_states[-1]
    # Exclude final time-step (next-token pred) and keep only completion window
    last_hidden = last_hidden[:, :-1, :]
    last_hidden = last_hidden[:, -logits_to_keep:, :]
    return last_hidden


def compute_liger_grpo_loss(model, prompt_ids, completion_ids, prompt_mask, completion_mask, advantages, args, liger_loss):
    """
    Liger kernel GRPO loss, mirroring TRL's usage of LigerFusedLinearGRPOLoss.
    """
    # Ensure all tensors are on the same device as the model
    device = model.device
    prompt_ids = prompt_ids.to(device)
    completion_ids = completion_ids.to(device)
    prompt_mask = prompt_mask.to(device)
    completion_mask = completion_mask.to(device)
    advantages = advantages.to(device)

    # Build full sequence and compute last hidden states for completion window
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)

    last_hidden_state = _get_last_hidden_state_for_liger(
        model, input_ids, attention_mask, logits_to_keep
    )

    # Align computation to the device of lm_head weights to avoid cross-device matmul
    target_device = model.lm_head.weight.device
    if last_hidden_state.device != target_device:
        last_hidden_state = last_hidden_state.to(target_device)
    if completion_ids.device != target_device:
        completion_ids = completion_ids.to(target_device)
    if completion_mask.device != target_device:
        completion_mask = completion_mask.to(target_device)
    if advantages.device != target_device:
        advantages = advantages.to(target_device)

    # Handle per-token advantages: if advantages is (B, T), we need to align with completion tokens
    # Liger expects advantages of shape (B,) or (B, T) where T matches completion length
    if advantages.dim() == 2:
        # Per-token advantages - ensure they match completion length
        if advantages.size(1) != completion_ids.size(1):
            # Pad or truncate to match
            target_len = completion_ids.size(1)
            if advantages.size(1) < target_len:
                # Pad with zeros
                pad_size = target_len - advantages.size(1)
                advantages = torch.nn.functional.pad(advantages, (0, pad_size), value=0.0)
            else:
                # Truncate
                advantages = advantages[:, :target_len]

    # Compute fused loss; we don't use ref/old logps in this simple setup
    loss, _metrics = liger_loss(
        _input=last_hidden_state,
        lin_weight=model.lm_head.weight,
        selected_token_ids=completion_ids,
        attention_mask=completion_mask,
        advantages=advantages,
        bias=getattr(model.lm_head, "bias", None),
        old_per_token_logps=None,
        ref_per_token_logps=None,
    )
    return loss


def generate_local(model, tokenizer, prompt_ids, prompt_mask, args):
    """Generate using local model and compute entropy rewards"""
    # Repeat prompt for multiple parallel generations (chains)
    prompt_ids = prompt_ids.repeat(args.num_chains, 1).to(model.device)
    prompt_mask = prompt_mask.repeat(args.num_chains, 1).to(model.device)

    # Set up generation parameters
    generation_config = {
        "max_new_tokens": args.max_completion_length,  # Max tokens to generate
        "do_sample": True,  # Enable sampling (not greedy)
        "temperature": args.temperature,  # Sampling temperature
        "top_p": 1.0,  # No top-p filtering
        "repetition_penalty": 1.0,  # No repetition penalty
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
    
    # Compute entropy rewards for completion tokens
    entropy_rewards = None
    if args.reward_mode in ["combined", "entropy_only"]:
        entropy_rewards = compute_completion_entropy(
            model, prompt_ids, completion_ids, prompt_mask, completion_mask, num_layers=10
        )
    
    return prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completion_mask, completions_text, entropy_rewards








def normalize_entropy_rewards(entropy_rewards, num_chains):
    """
    Normalize entropy rewards using z-score and scale to [-val, 0.5] within each group.
    
    Args:
        entropy_rewards: (B,) tensor of entropy values
        num_chains: Number of chains per problem (for grouping)
    
    Returns:
        normalized_rewards: (B,) tensor of normalized rewards in [-0.5, 0.5]
    """
    if entropy_rewards is None:
        return None
    
    device = entropy_rewards.device
    num_problems = entropy_rewards.size(0) // num_chains
    
    normalized = []
    for i in range(num_problems):
        start_idx = i * num_chains
        end_idx = start_idx + num_chains
        group_entropies = entropy_rewards[start_idx:end_idx]
        
        # Compute z-score
        mean_entropy = group_entropies.mean()
        std_entropy = group_entropies.std()
        eps = 1e-6
        
        if std_entropy < eps:
            # All entropies are the same, set to 0
            z_scores = torch.zeros_like(group_entropies)
        else:
            z_scores = (group_entropies - mean_entropy) / (std_entropy + eps)
        
        # Scale to [-0.5, 0.5] by clipping z-scores
        # Using scale_factor of 0.5 means we clip z-scores in [-1, 1] range
        val = .1
        scaled = torch.clamp(z_scores * val, -val, val)
        normalized.append(scaled)
    
    return torch.cat(normalized, dim=0)


def generate(model, tokenizer, prompt_ids, prompt_mask, args):
    """Main generate function"""
    return generate_local(model, tokenizer, prompt_ids, prompt_mask, args)


def compute_pass_at_k(n, c, k):
    """
    Calculate pass@k metric using the standard formula:
    pass@k = 1 - (n-c choose k) / (n choose k)
    
    Args:
        n: total number of samples
        c: number of correct samples
        k: k for pass@k
    
    Returns:
        pass@k probability (0.0 to 1.0)
    """
    if n - c < k:
        return 1.0
    
    # Calculate 1 - P(all k samples are wrong)
    # P(all k wrong) = product from i=0 to k-1 of (n-c-i)/(n-i)
    prob_all_wrong = 1.0
    for i in range(k):
        prob_all_wrong *= (n - c - i) / (n - i)
    
    return 1.0 - prob_all_wrong


def compute_grpo_loss(model, prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completion_mask, advantages, args=None):

    # DR-GRPO loss implementation
    # Number of completion tokens to compute loss over
    tokens_to_keep = completion_ids.size(1)

    # Reconstruct full input sequence (prompt + completion)
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    # Get per-token log probabilities from the current model
    logps = utils.get_per_token_logps(model, input_ids, attention_mask, tokens_to_keep)

    # Compute GRPO loss: -exp(logp - logp.detach()) * advantages
    # The exp(logp - logp.detach()) creates importance weights, advantages provide direction
    # Advantages can be either (B,) scalar or (B, T) per-token
    if advantages.dim() == 1:
        # Scalar advantages - broadcast to all tokens
        per_token_loss = -torch.exp(logps - logps.detach()) * advantages.unsqueeze(1)
    else:
        # Per-token advantages - use directly
        per_token_loss = -torch.exp(logps - logps.detach()) * advantages
    
    # Create a completion-only mask (extract the completion part from the full mask)
    completion_only_mask = completion_mask[:, -tokens_to_keep:]  # Take only the completion tokens
    
    # DR-GRPO loss: normalize by batch size and max completion length
    # This makes the loss scale-invariant to sequence length and batch size
    loss = (per_token_loss * completion_only_mask).sum() / (per_token_loss.size(0) * args.max_completion_length)
    
    return loss



def parse_args():
    parser = argparse.ArgumentParser(description="Nano GRPO with reasoning_gym composite datasets")

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base/reference model name")

    # Output and logging
    parser.add_argument("--output_dir", type=str, default="final_run", help="Where to save logs")
    parser.add_argument("--use_wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="nano-grpo", help="W&B project name")
    parser.add_argument("--wandb_run", type=str, default="run", help="W&B run name")

    # Optimization
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.99, help="Adam beta2") 
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, help="Grad norm clip")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Grad accum steps")
    parser.add_argument("--warmup_percent", type=float, default=0.1, help="Warmup percent of iters")

    # Generation
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--num_chains", type=int, default=8, help="Parallel generations per prompt")
    parser.add_argument("--max_prompt_length", type=int, default=256, help="Max prompt tokens")
    parser.add_argument("--max_completion_length", type=int, default=512, help="Max completion tokens")
    
    # Liger loss options
    parser.add_argument("--use_liger", action="store_true", help="Use Liger kernel model and loss")
    parser.add_argument("--epsilon_low", type=float, default=0.2, help="Lower epsilon for clipping")
    parser.add_argument("--epsilon_high", type=float, default=None, help="Upper epsilon; defaults to epsilon_low if None")
    parser.add_argument("--beta", type=float, default=0.0, help="KL coefficient; 0 disables ref model pathway")
    parser.add_argument("--loss_type", type=str, default="dr_grpo", choices=["grpo", "bnpo", "dr_grpo"], help="Loss aggregation variant")
    
    # Reward configuration
    parser.add_argument("--reward_mode", type=str, default="current", choices=["current", "combined", "entropy_only"], 
                        help="Reward mode: 'current' (format+correctness), 'combined' (current+entropy), 'entropy_only'")
    parser.add_argument("--only_if_correct", action="store_true", 
                        help="Only reward entropy if the answer is correct (otherwise entropy reward is 0)")
    
    # Training
    parser.add_argument("--num_train_iters", type=int, default=1000, help="Training iterations")
    parser.add_argument("--seed", type=int, default=7111994, help="Random seed")
    parser.add_argument("--eval_every", type=int, default=50, help="Run evaluation every N steps")
    parser.add_argument("--save_every", type=int, default=50, help="Save model checkpoint every N steps")
    
    # Evaluation
    parser.add_argument("--num_completions_eval", type=int, default=10, help="Number of completions to sample per eval problem for pass@k")
    parser.add_argument("--pass_at_k", type=int, default=1, help="k for pass@k metric")

    # Dataset configuration (MATH)
    parser.add_argument("--train-size", type=int, default=12000, help="Number of training examples to use")
    parser.add_argument("--eval-size", type=int, default=20, help="Number of eval examples to use")

    return parser.parse_args()




if __name__ == "__main__":

    # Get all settings 
    args = parse_args()

    # Seed everything for reproducible results 
    utils.seed_everything(args.seed)

    # Setup logging 
    os.makedirs(args.output_dir, exist_ok=True)
    # Optional W&B
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    # Setup model
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, use_liger_model=args.use_liger)

    # Instantiate Liger loss once if requested
    if args.use_liger:
        liger_loss = LigerFusedLinearGRPOLoss(
            beta=getattr(args, "beta", 0.0),
            epsilon_low=getattr(args, "epsilon_low", 0.2),
            epsilon_high=(args.epsilon_high if getattr(args, "epsilon_high", None) is not None else getattr(args, "epsilon_low", 0.2)),
            temperature=args.temperature,
            use_ref_model=(getattr(args, "beta", 0.0) != 0.0),
            loss_type=getattr(args, "loss_type", "dr_grpo"),
            max_completion_length=args.max_completion_length,
        )

    # Build datasets
    train_ds, eval_ds = load_math_dataset(
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed
    )



    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)
    warmup_steps = int(args.warmup_percent * args.num_train_iters)
    def get_lr(step):
        if step < warmup_steps:
            return (step / max(warmup_steps, 1))
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

    # Unified log structure - step-based with train/eval nested
    run_log = {
        "args": vars(args),
        "steps": {},  # {step: {"train": {...}, "eval": {...}}}
    }

    # Setup prompt 
    system_prompt = (
        "Think first and reason step by step. Put your reasoning within <think></think> tags. "
        "Then put your final answer within <answer></answer> tags. "
        "You must use both tags in this exact order: first <think>your reasoning</think>, then <answer>your answer</answer>."
        f"Note: Your reasoning may be cut off if it gets too long, but answer as best as you can if that happens."
    )


    # Training loop
    accumulated_loss = 0.0
    optimizer.zero_grad()
    for step in tqdm(range(args.num_train_iters), desc="Training"):
        
        # Periodic evaluation with pass@k
        if step % args.eval_every == 0 and eval_ds is not None:
            model.eval()  # Set model to eval mode
            pass_at_k_scores = []
            format_total = 0
            entropy_total = 0.0
            eval_count = 0
            eval_examples = []
            
            # Temporarily modify args for eval generation
            original_num_chains = args.num_chains
            args.num_chains = args.num_completions_eval
            
            with torch.no_grad():  # Disable gradients during eval
                for i, eval_entry in enumerate(eval_ds):
                    if i >= args.eval_size:
                        break
                    q = format_math_problem(eval_entry)
                    a = extract_math_answer(eval_entry)
                    eval_problem_type = f"{eval_entry['subject']}_level_{eval_entry['level']}"
                    prompt_text_eval, prompt_ids_eval, prompt_mask_eval = utils.format_prompt(system_prompt, q, tokenizer)
                    
                    # Generate multiple completions for this eval problem
                    _, prompt_ids_eval, completion_ids_eval, attention_mask_eval, completion_mask_eval, completions_text_eval, entropy_rewards_eval = generate(
                        model, tokenizer, prompt_ids_eval, prompt_mask_eval, args
                    )
                    
                    # Score all completions
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
                    
                    # Compute pass@k for this problem
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
                    
                    # Average entropy reward if computed
                    avg_entropy_for_problem = 0.0
                    if entropy_rewards_eval is not None:
                        avg_entropy_for_problem = entropy_rewards_eval.mean().item()
                        entropy_total += avg_entropy_for_problem
                    
                    eval_count += 1
                    
                    # Log this eval example
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
                        "avg_entropy_reward": avg_entropy_for_problem,
                    })
            
            # Restore original num_chains and training mode
            args.num_chains = original_num_chains
            model.train()  # Set model back to train mode
            
            # Clear CUDA cache to prevent OOM
            torch.cuda.empty_cache()
            
            # Aggregate overall metrics
            avg_pass_at_k = (sum(pass_at_k_scores) / max(eval_count, 1)) * 100
            avg_format = (format_total / max(eval_count, 1))
            avg_entropy = (entropy_total / max(eval_count, 1)) if args.reward_mode in ["combined", "entropy_only"] else 0.0
            
            # Log to step-based structure
            if step not in run_log["steps"]:
                run_log["steps"][step] = {}
            
            eval_metrics = {
                f"pass_at_{args.pass_at_k}": avg_pass_at_k,
                "avg_format_reward": avg_format,
                "num_eval_problems": eval_count,
            }
            if args.reward_mode in ["combined", "entropy_only"]:
                eval_metrics["avg_entropy_reward"] = avg_entropy
            
            run_log["steps"][step]["eval"] = {
                "examples": eval_examples,
                "metrics": eval_metrics,
            }
            
            # Save summary JSON for easy plotting (just overall metrics per step)
            eval_summary_path = os.path.join(args.output_dir, "eval_summary.json")
            eval_summary = {}
            if os.path.exists(eval_summary_path):
                with open(eval_summary_path, "r") as f:
                    eval_summary = json.load(f)
            
            eval_summary[str(step)] = eval_metrics.copy()
            
            with open(eval_summary_path, "w") as f:
                json.dump(eval_summary, f, indent=2)
            
            entropy_str = f", Avg Entropy = {avg_entropy:.4f}" if args.reward_mode in ["combined", "entropy_only"] else ""
            print(f"\nEval at step {step}: Pass@{args.pass_at_k} = {avg_pass_at_k:.2f}%, Avg Format = {avg_format:.3f}{entropy_str}")
            if args.use_wandb:
                import wandb
                log_dict = {
                    f"eval/pass_at_{args.pass_at_k}": avg_pass_at_k,
                    "eval/avg_format_reward": avg_format,
                }
                if args.reward_mode in ["combined", "entropy_only"]:
                    log_dict["eval/avg_entropy_reward"] = avg_entropy
                wandb.log(log_dict, step=step)
            
            # Clear cache aggressively after eval
            torch.cuda.empty_cache()
        
        # Training step
        entry = random.choice(list(train_ds))
        question = format_math_problem(entry)
        answer = extract_math_answer(entry)
        problem_type = f"{entry['subject']}_level_{entry['level']}"

        # Setup prompt
        prompt_text, prompt_ids, prompt_mask = utils.format_prompt(system_prompt, question, tokenizer)

        ##################
        ### GRPO LOOP ####
        ##################

        # Generate (with no_grad to save memory)
        with torch.no_grad():
            prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completion_mask, completions_text, entropy_rewards = generate(
                model, tokenizer, prompt_ids, prompt_mask, args
            )

        # Score
        extracted_answers = [utils.extract_answer(t) for t in completions_text]
        format_rewards = [utils.check_format(t) for t in completions_text]
        # Only score correctness if format is correct AND there's an extracted answer
        correctness = []
        for a, f, t in zip(extracted_answers, format_rewards, completions_text):
            if f < 0:  # Wrong format (penalty)
                correctness.append(0.0)
            elif a:  # Has extracted answer
                correctness.append(float(train_ds.score_answer(answer=a, entry=entry) == 1.0))
            else:  # No extracted answer
                correctness.append(0.0)
        correctness = correctness

        # Combine rewards based on reward_mode
        if args.reward_mode == "current":
            # Just format + correctness
            total_rewards = [
                c + f
                for c, f in zip(correctness, format_rewards)
            ]
            rewards = torch.tensor(total_rewards, device=model.device)
        elif args.reward_mode == "combined":
            # Current rewards + normalized entropy rewards
            current_rewards_list = [
                c + f
                for c, f in zip(correctness, format_rewards)
            ]
            current_rewards = torch.tensor(current_rewards_list, device=model.device)
            normalized_entropy = normalize_entropy_rewards(entropy_rewards, args.num_chains)
            
            # If only_if_correct is set, zero out entropy rewards for incorrect answers
            if args.only_if_correct and normalized_entropy is not None:
                correctness_tensor = torch.tensor(correctness, device=model.device, dtype=torch.float32)
                # Only keep entropy reward if answer is correct (correctness > 0)
                normalized_entropy = normalized_entropy * (correctness_tensor > 0).float()
            
            rewards = current_rewards + normalized_entropy
            # For logging: combine current + entropy
            total_rewards = [
                cr + ne.item()
                for cr, ne in zip(current_rewards_list, normalized_entropy)
            ]
        elif args.reward_mode == "entropy_only":
            # Just normalized entropy rewards
            normalized_entropy = normalize_entropy_rewards(entropy_rewards, args.num_chains)
            
            # If only_if_correct is set, zero out entropy rewards for incorrect answers
            if args.only_if_correct and normalized_entropy is not None:
                correctness_tensor = torch.tensor(correctness, device=model.device, dtype=torch.float32)
                # Only keep entropy reward if answer is correct (correctness > 0)
                normalized_entropy = normalized_entropy * (correctness_tensor > 0).float()
            
            rewards = normalized_entropy
            # For logging: use entropy rewards
            total_rewards = [ne.item() for ne in normalized_entropy]
        else:
            raise ValueError(f"Unknown reward_mode: {args.reward_mode}")

        # Compute scalar advantages (for group normalization)
        grouped = rewards.view(-1, args.num_chains)
        mean_group = grouped.mean(dim=1).repeat_interleave(args.num_chains)
        std_group = grouped.std(dim=1).repeat_interleave(args.num_chains)
        
        scalar_advantages = (rewards - mean_group) / (std_group + 1e-4)

        # Use scalar advantages directly (broadcast to all tokens)
        advantages = scalar_advantages.unsqueeze(1)

        # Normalize masks for loss computation
        # Build a batched prompt mask matching `prompt_ids` (B, prompt_len)
        prompt_mask_batched = torch.ones_like(prompt_ids, device=model.device)
        # Ensure completion_mask covers only completion tokens (B, completion_len)
        if completion_mask.shape[1] != completion_ids.shape[1]:
            completion_mask_for_loss = completion_mask[:, -completion_ids.size(1):]
        else:
            completion_mask_for_loss = completion_mask

        # Compute loss (Liger fused if enabled)
        if args.use_liger:
            loss = compute_liger_grpo_loss(
                model,
                prompt_ids,
                completion_ids,
                prompt_mask_batched,
                completion_mask_for_loss,
                advantages,
                args,
                liger_loss,
            )
        else:
            loss = compute_grpo_loss(
                model,
                prompt_completion_ids,
                prompt_ids,
                completion_ids,
                attention_mask,
                completion_mask_for_loss,
                advantages,
                args,
            )
        (loss / args.gradient_accumulation_steps).backward()
        accumulated_loss += loss.item()
        
        # Delete large tensors to free memory
        del prompt_completion_ids, completion_ids, attention_mask, completion_mask
        del advantages
        
        # Optim step
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            
            # Clear cache after optimizer step
            torch.cuda.empty_cache()
        scheduler.step()

        # Log per step
        if step not in run_log["steps"]:
            run_log["steps"][step] = {}
        
        run_log["steps"][step]["train"] = {
            "prompt": prompt_text,
            "question": question,
            "target_answer": answer,
            "problem_type": problem_type,
            "generations": [
                {
                    "text": t,
                    "extracted_answer": ea,
                    "correct": int(c),
                    "format_reward": float(f),
                    "total_reward": float(tr)
                } for t, ea, c, f, tr in zip(completions_text, extracted_answers, correctness, format_rewards, total_rewards)
            ],
            "loss": loss.item(),
            "lr": scheduler.get_last_lr()[0],
        }

        if args.use_wandb:
            import wandb
            wandb.log({
                "train/loss": loss.item(),
                "lr": scheduler.get_last_lr()[0]
            }, step=step)


        # Periodic model saving
        if (step + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_step_{step+1}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            print(f"Saved checkpoint at step {step+1} to {checkpoint_path}")

        # Persist log
        with open(os.path.join(args.output_dir, "run_log.json"), "w") as f:
            json.dump(run_log, f, indent=2)













#