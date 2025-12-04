
"""
GRPO training script for MATH dataset.

This is a simple Python-first implementation of GRPO tailored for mathematical
reasoning on the MATH dataset. It keeps the algorithm easy to read and modify, 
while supporting industry-standard performance options:
  - vLLM: optional high-throughput generation via a server for sampling/logprobs
  - LigerKernel: optional fused kernels for faster, stable GRPO loss and model forward
  - Accelerate: multi-GPU ready via the Hugging Face ecosystem

Use --use_vllm to generate with vLLM, and --use_liger to enable the Liger model
and fused GRPO loss for local training.
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
import vllm_client  as v_c



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
    """Generate using local model (original method)"""
    # Repeat prompt for multiple parallel generations (chains)
    prompt_ids = prompt_ids.repeat(args.num_chains, 1).to(model.device)
    prompt_mask = prompt_mask.repeat(args.num_chains, 1).to(model.device)

    # Set up generation parameters (match vLLM defaults)
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


def generate_vllm(vllm_client, prompt_text, tokenizer, args, device):
    """Generate using vLLM server and return proper token IDs for GRPO training"""
    # Generate completions using vLLM server
    # Match local generation parameters: use top_p=1.0 and top_k=-1 to match transformers defaults
    
    # CRITICAL DEBUG: Verify prompt tokenization matches
    local_prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)["input_ids"][0].tolist()
    
    # IMPORTANT: Send 1 prompt with n=num_chains to generate multiple completions
    # This matches how local generation works (batch generation from same prompt)
    response = vllm_client.generate(
        prompts=[prompt_text],  # Single prompt
        n=args.num_chains,  # Generate num_chains completions from it
        temperature=args.temperature,
        max_tokens=args.max_completion_length,
        top_p=1.0,  # Match transformers default (no top-p filtering)
        top_k=-1,  # Match transformers default (no top-k filtering)
        repetition_penalty=1.0,  # Match transformers default (no repetition penalty)
        # Don't pass seed - let vLLM sample freely like transformers does
    )
    
    # Extract data from response
    # With n=num_chains, vLLM returns:
    #   - prompt_ids: list with 1 element (the single prompt)
    #   - completion_ids: list with num_chains elements (all completions)
    prompt_ids_single = response["prompt_ids"][0]  # Get the single prompt
    completion_ids_list = response["completion_ids"]  # List of all completions
    
    # DEBUG: Check if prompt tokenization matches
    if prompt_ids_single != local_prompt_ids:
        print(f"\n{'='*80}")
        print(f"WARNING: Prompt tokenization mismatch detected!")
        print(f"{'='*80}")
        print(f"Local tokenizer length: {len(local_prompt_ids)}")
        print(f"vLLM tokenizer length:  {len(prompt_ids_single)}")
        print(f"\nFirst 10 tokens:")
        print(f"  Local: {local_prompt_ids[:10]}")
        print(f"  vLLM:  {prompt_ids_single[:10]}")
        print(f"\nLast 10 tokens:")
        print(f"  Local: {local_prompt_ids[-10:]}")
        print(f"  vLLM:  {prompt_ids_single[-10:]}")
        
        # Decode to see the actual text
        print(f"\nDecoded (first 100 chars):")
        print(f"  Local: {tokenizer.decode(local_prompt_ids[:20], skip_special_tokens=False)}")
        print(f"  vLLM:  {tokenizer.decode(prompt_ids_single[:20], skip_special_tokens=False)}")
        print(f"{'='*80}\n")
        # Use local tokenization for consistency
        prompt_ids_single = local_prompt_ids
    
    # Expand prompt to match number of completions (for batch processing)
    prompt_ids_list = [prompt_ids_single] * args.num_chains
    
    # Convert to tensors with proper padding
    # First, pad all sequences to the same length
    max_prompt_len = max(len(ids) for ids in prompt_ids_list)
    max_completion_len = max(len(ids) for ids in completion_ids_list)
    
    # Pad prompt_ids
    padded_prompt_ids = []
    for ids in prompt_ids_list:
        padded = ids + [tokenizer.pad_token_id] * (max_prompt_len - len(ids))
        padded_prompt_ids.append(padded)
    prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)
    
    # Pad completion_ids
    padded_completion_ids = []
    for ids in completion_ids_list:
        padded = ids + [tokenizer.pad_token_id] * (max_completion_len - len(ids))
        padded_completion_ids.append(padded)
    completion_ids = torch.tensor(padded_completion_ids, dtype=torch.long, device=device)
    
    # Create full prompt+completion sequences
    prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    
    # Handle EOS tokens properly (mirror local generation logic)
    is_eos = completion_ids == tokenizer.eos_token_id  # Find EOS tokens
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)  # Default to end
    has_eos = is_eos.any(dim=1)  # Check which sequences have EOS
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]  # Set EOS position for sequences that have it
    seq_idx = torch.arange(is_eos.size(1), device=device).expand_as(is_eos)  # Position indices
    completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()  # Mask: 1 for valid tokens, 0 after EOS
    
    # Create proper attention mask (1 for real tokens, 0 for padding)
    # For prompt: all non-pad tokens are valid
    prompt_attention_mask = (prompt_ids != tokenizer.pad_token_id).int()
    # For completion: valid tokens up to EOS (or end if no EOS)
    completion_attention_mask = completion_mask.int()
    # Combine
    attention_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)
    
    # Decode completions to text (use original unpadded sequences)
    # skip_special_tokens=True will automatically handle EOS tokens
    completions_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in completion_ids_list]
    
    return prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completion_mask, completions_text






def generate(model, tokenizer, prompt_ids, prompt_mask, args, vllm_client=None, prompt_text=None):
    """Main generate function that routes to local or vLLM based on args"""
    if args.use_vllm and vllm_client is not None:
        return generate_vllm(vllm_client, prompt_text, tokenizer, args, model.device)
    else:
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
    
    # vLLM server option
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM server for generation instead of local model")
    parser.add_argument("--vllm_host", type=str, default="localhost", help="vLLM server host")
    parser.add_argument("--vllm_port", type=int, default=8000, help="vLLM server port")
    
    # Training
    parser.add_argument("--num_train_iters", type=int, default=1000, help="Training iterations")
    parser.add_argument("--seed", type=int, default=7111994, help="Random seed")
    parser.add_argument("--eval_every", type=int, default=50, help="Run evaluation every N steps")
    parser.add_argument("--save_every", type=int, default=50, help="Save model checkpoint every N steps")
    
    # Evaluation
    parser.add_argument("--num_completions_eval", type=int, default=20, help="Number of completions to sample per eval problem for pass@k")
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

    # Setup model and vLLM client (if needed)
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, use_liger_model=args.use_liger)
    vllm_client = None
    if args.use_vllm:
        base_url = f"http://{args.vllm_host}:{args.vllm_port}"
        vllm_client = v_c.VLLMClient(base_url=base_url)
        vllm_client.init_communicator(device=model.device)
        print(f"Connected to vLLM server at {args.vllm_host}:{args.vllm_port}")

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
            # CRITICAL: Sync vLLM server weights BEFORE eval if using vLLM
            # The vLLM server weights are only updated after optimizer steps, but eval happens
            # at the start of the iteration, so the server might have stale weights.
            if args.use_vllm and vllm_client is not None:
                vllm_client.update_model_params(model)
                print(f"Synced vLLM server weights before eval at step {step}")
            
            model.eval()  # Set model to eval mode
            pass_at_k_scores = []
            format_total = 0
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
                    _, prompt_ids_eval, completion_ids_eval, attention_mask_eval, completion_mask_eval, completions_text_eval = generate(
                        model, tokenizer, prompt_ids_eval, prompt_mask_eval, args, vllm_client, prompt_text_eval
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
                    })
            
            # Restore original num_chains and training mode
            args.num_chains = original_num_chains
            model.train()  # Set model back to train mode
            
            # Clear CUDA cache to prevent OOM
            torch.cuda.empty_cache()
            
            # Aggregate overall metrics
            avg_pass_at_k = (sum(pass_at_k_scores) / max(eval_count, 1)) * 100
            avg_format = (format_total / max(eval_count, 1))
            
            # Log to step-based structure
            if step not in run_log["steps"]:
                run_log["steps"][step] = {}
            
            run_log["steps"][step]["eval"] = {
                "examples": eval_examples,
                "metrics": {
                    f"pass_at_{args.pass_at_k}": avg_pass_at_k,
                    "avg_format_reward": avg_format,
                    "num_eval_problems": eval_count,
                }
            }
            
            # Save summary JSON for easy plotting (just overall metrics per step)
            eval_summary_path = os.path.join(args.output_dir, "eval_summary.json")
            eval_summary = {}
            if os.path.exists(eval_summary_path):
                with open(eval_summary_path, "r") as f:
                    eval_summary = json.load(f)
            
            eval_summary[str(step)] = {
                f"pass_at_{args.pass_at_k}": avg_pass_at_k,
                "avg_format_reward": avg_format,
                "num_eval_problems": eval_count,
            }
            
            with open(eval_summary_path, "w") as f:
                json.dump(eval_summary, f, indent=2)
            
            print(f"\nEval at step {step}: Pass@{args.pass_at_k} = {avg_pass_at_k:.2f}%, Avg Format = {avg_format:.3f}")
            if args.use_wandb:
                import wandb
                wandb.log({
                    f"eval/pass_at_{args.pass_at_k}": avg_pass_at_k,
                    "eval/avg_format_reward": avg_format,
                }, step=step)
            
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
            prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completion_mask, completions_text = generate(
                model, tokenizer, prompt_ids, prompt_mask, args, vllm_client, prompt_text
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


        # Combine correctness and format rewards (scalar per completion)
        total_rewards = [
            c + f
            for c, f in zip(correctness, format_rewards)
        ]
        rewards = torch.tensor(total_rewards, device=model.device)

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
            
            # Update vLLM server model parameters if using vLLM
            if args.use_vllm and vllm_client is not None:
                vllm_client.update_model_params(model)
            
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