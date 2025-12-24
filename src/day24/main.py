"""
GRPO training with LLM-as-judge reward on MMLU.

Simple setup:
1. Sample random MMLU question
2. Generate 4 completions from Baguettron
3. Round-robin pairwise comparison using GPT-4.1 as judge
4. Win rate = reward signal for GRPO
5. Save checkpoint every 50 steps

No in-training eval - just save checkpoints and run MMLU eval after.
"""

import os
import json
import torch
import random
import argparse
from tqdm import tqdm
from datasets import load_dataset
from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

import llms
import utils
import vllm_client as v_c
from openai_judge import round_robin_judge_sync


# MMLU subjects (all 57)
MMLU_SUBJECTS = [
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


def load_mmlu_questions(subjects: list[str] = None, split: str = "test") -> list[dict]:
    """Load MMLU questions from all subjects into a flat list."""
    if subjects is None:
        subjects = MMLU_SUBJECTS
    
    all_questions = []
    for subject in tqdm(subjects, desc="Loading MMLU"):
        try:
            dataset = load_dataset("cais/mmlu", subject, split=split)
            for item in dataset:
                all_questions.append({
                    "subject": subject,
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": item["answer"],  # 0-3 index
                })
        except Exception as e:
            print(f"Warning: Failed to load {subject}: {e}")
    
    print(f"Loaded {len(all_questions)} MMLU questions from {len(subjects)} subjects")
    return all_questions


def format_mmlu_prompt(question_data: dict) -> tuple[str, str]:
    """
    Format MMLU question for the model.
    
    Returns:
        - question_text: The formatted question for the model
        - choices_text: Formatted choices for the judge
    """
    subject = question_data["subject"].replace("_", " ")
    question = question_data["question"]
    choices = question_data["choices"]
    
    choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
    
    question_text = f"""The following is a multiple choice question about {subject}.

{question}

{choices_text}

Think through this step by step, then give your answer (A, B, C, or D)."""
    
    return question_text, choices_text


def _get_last_hidden_state_for_liger(model, input_ids, attention_mask, logits_to_keep: int):
    """Compute last hidden state aligned to completion tokens for Liger loss."""
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        last_hidden = outputs.last_hidden_state
    else:
        last_hidden = outputs.hidden_states[-1]
    last_hidden = last_hidden[:, :-1, :]
    last_hidden = last_hidden[:, -logits_to_keep:, :]
    return last_hidden


def compute_liger_grpo_loss(model, prompt_ids, completion_ids, prompt_mask, completion_mask, advantages, args, liger_loss):
    """Liger kernel GRPO loss."""
    device = model.device
    prompt_ids = prompt_ids.to(device)
    completion_ids = completion_ids.to(device)
    prompt_mask = prompt_mask.to(device)
    completion_mask = completion_mask.to(device)
    advantages = advantages.to(device)

    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)

    last_hidden_state = _get_last_hidden_state_for_liger(model, input_ids, attention_mask, logits_to_keep)

    target_device = model.lm_head.weight.device
    if last_hidden_state.device != target_device:
        last_hidden_state = last_hidden_state.to(target_device)
    if completion_ids.device != target_device:
        completion_ids = completion_ids.to(target_device)
    if completion_mask.device != target_device:
        completion_mask = completion_mask.to(target_device)
    if advantages.device != target_device:
        advantages = advantages.to(target_device)

    if advantages.dim() == 2:
        target_len = completion_ids.size(1)
        if advantages.size(1) < target_len:
            pad_size = target_len - advantages.size(1)
            advantages = torch.nn.functional.pad(advantages, (0, pad_size), value=0.0)
        else:
            advantages = advantages[:, :target_len]

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
    """Generate using local model."""
    prompt_ids = prompt_ids.repeat(args.num_chains, 1).to(model.device)
    prompt_mask = prompt_mask.repeat(args.num_chains, 1).to(model.device)

    # Debug: check token IDs are in valid range
    vocab_size = model.config.vocab_size
    max_token_id = prompt_ids.max().item()
    min_token_id = prompt_ids.min().item()
    if max_token_id >= vocab_size or min_token_id < 0:
        print(f"ERROR: Token IDs out of range! min={min_token_id}, max={max_token_id}, vocab_size={vocab_size}")
        # Clamp to valid range as workaround
        prompt_ids = prompt_ids.clamp(0, vocab_size - 1)

    generation_config = {
        "max_new_tokens": args.max_completion_length,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    with torch.inference_mode():
        prompt_completion_ids = model.generate(prompt_ids, attention_mask=prompt_mask, **generation_config)

    prompt_len = prompt_ids.size(1)
    prompt_ids = prompt_completion_ids[:, :prompt_len]
    completion_ids = prompt_completion_ids[:, prompt_len:]

    is_eos = completion_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=model.device)
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    seq_idx = torch.arange(is_eos.size(1), device=model.device).expand_as(is_eos)
    completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()

    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    
    return prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completion_mask, completions_text


def generate_vllm(vllm_client, prompt_text, tokenizer, args, device):
    """Generate using vLLM server."""
    local_prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)["input_ids"][0].tolist()
    
    response = vllm_client.generate(
        prompts=[prompt_text],
        n=args.num_chains,
        temperature=args.temperature,
        max_tokens=args.max_completion_length,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
    )
    
    prompt_ids_single = response["prompt_ids"][0]
    completion_ids_list = response["completion_ids"]
    
    if prompt_ids_single != local_prompt_ids:
        prompt_ids_single = local_prompt_ids
    
    prompt_ids_list = [prompt_ids_single] * args.num_chains
    
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
    
    prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    
    is_eos = completion_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    seq_idx = torch.arange(is_eos.size(1), device=device).expand_as(is_eos)
    completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()
    
    prompt_attention_mask = (prompt_ids != tokenizer.pad_token_id).int()
    completion_attention_mask = completion_mask.int()
    attention_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)
    
    completions_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in completion_ids_list]
    
    return prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completion_mask, completions_text


def generate(model, tokenizer, prompt_ids, prompt_mask, args, vllm_client=None, prompt_text=None):
    """Main generate function - routes to local or vLLM."""
    if args.use_vllm and vllm_client is not None:
        return generate_vllm(vllm_client, prompt_text, tokenizer, args, model.device)
    else:
        return generate_local(model, tokenizer, prompt_ids, prompt_mask, args)


def compute_grpo_loss(model, prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completion_mask, advantages, args=None):
    """DR-GRPO loss."""
    tokens_to_keep = completion_ids.size(1)
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    logps = utils.get_per_token_logps(model, input_ids, attention_mask, tokens_to_keep)

    if advantages.dim() == 1:
        per_token_loss = -torch.exp(logps - logps.detach()) * advantages.unsqueeze(1)
    else:
        per_token_loss = -torch.exp(logps - logps.detach()) * advantages
    
    completion_only_mask = completion_mask[:, -tokens_to_keep:]
    loss = (per_token_loss * completion_only_mask).sum() / (per_token_loss.size(0) * args.max_completion_length)
    
    return loss


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO with LLM-as-judge on MMLU")

    # Model
    parser.add_argument("--model_name", type=str, default="PleIAs/Baguettotron-7B-DPO-v1.5", help="Model to train")

    # Output
    parser.add_argument("--output_dir", type=str, default="grpo_mmlu_run", help="Where to save checkpoints and logs")

    # Optimization
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.99, help="Adam beta2") 
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, help="Grad norm clip")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Grad accum steps")
    parser.add_argument("--warmup_percent", type=float, default=0.1, help="Warmup percent")

    # Generation
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--num_chains", type=int, default=4, help="Completions per prompt for round-robin")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Max prompt tokens")
    parser.add_argument("--max_completion_length", type=int, default=1024, help="Max completion tokens")
    
    # Liger
    parser.add_argument("--use_liger", action="store_true", help="Use Liger kernel")
    parser.add_argument("--epsilon_low", type=float, default=0.2, help="Lower epsilon for clipping")
    parser.add_argument("--epsilon_high", type=float, default=None, help="Upper epsilon")
    parser.add_argument("--beta", type=float, default=0.0, help="KL coefficient")
    parser.add_argument("--loss_type", type=str, default="dr_grpo", choices=["grpo", "bnpo", "dr_grpo"])
    
    # vLLM
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM server for generation")
    parser.add_argument("--vllm_host", type=str, default="localhost", help="vLLM server host")
    parser.add_argument("--vllm_port", type=int, default=8000, help="vLLM server port")
    
    # Training
    parser.add_argument("--num_train_iters", type=int, default=500, help="Training iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N steps")
    
    # Judge
    parser.add_argument("--judge_model", type=str, default="gpt-4.1", help="OpenAI model for judging")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    utils.seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model_name}")
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, use_liger_model=args.use_liger)
    
    # vLLM client
    vllm_client = None
    if args.use_vllm:
        base_url = f"http://{args.vllm_host}:{args.vllm_port}"
        vllm_client = v_c.VLLMClient(base_url=base_url)
        vllm_client.init_communicator(device=model.device)
        print(f"Connected to vLLM server at {base_url}")

    # Liger loss
    liger_loss = None
    if args.use_liger:
        liger_loss = LigerFusedLinearGRPOLoss(
            beta=args.beta,
            epsilon_low=args.epsilon_low,
            epsilon_high=(args.epsilon_high if args.epsilon_high is not None else args.epsilon_low),
            temperature=args.temperature,
            use_ref_model=(args.beta != 0.0),
            loss_type=args.loss_type,
            max_completion_length=args.max_completion_length,
        )

    # Load MMLU
    print("Loading MMLU dataset...")
    mmlu_questions = load_mmlu_questions()
    
    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        betas=(args.adam_beta1, args.adam_beta2), 
        weight_decay=args.weight_decay
    )
    warmup_steps = int(args.warmup_percent * args.num_train_iters)
    
    def get_lr(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

    # Logging
    run_log = {
        "args": vars(args),
        "steps": {},
    }

    system_prompt = "You are a helpful assistant. Think through problems step by step."

    # Training loop
    optimizer.zero_grad()
    
    for step in tqdm(range(args.num_train_iters), desc="Training"):
        # Sample random MMLU question
        question_data = random.choice(mmlu_questions)
        question_text, choices_text = format_mmlu_prompt(question_data)
        correct_answer = chr(65 + question_data["answer"])  # A, B, C, or D
        
        # Format prompt
        prompt_text, prompt_ids, prompt_mask = utils.format_prompt(system_prompt, question_text, tokenizer)

        # Generate completions
        with torch.no_grad():
            prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completion_mask, completions_text = generate(
                model, tokenizer, prompt_ids, prompt_mask, args, vllm_client, prompt_text
            )

        # Round-robin judge to get win-rate rewards
        rewards = round_robin_judge_sync(
            question=question_text,
            choices=choices_text,
            completions=completions_text,
            model=args.judge_model,
        )
        
        rewards_tensor = torch.tensor(rewards, device=model.device)

        # Compute advantages (group normalize)
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std()
        advantages = (rewards_tensor - mean_reward) / (std_reward + 1e-4)
        advantages = advantages.unsqueeze(1)  # (B, 1) for broadcasting

        # Build masks
        prompt_mask_batched = torch.ones_like(prompt_ids, device=model.device)
        if completion_mask.shape[1] != completion_ids.shape[1]:
            completion_mask_for_loss = completion_mask[:, -completion_ids.size(1):]
        else:
            completion_mask_for_loss = completion_mask

        # Compute loss
        if args.use_liger:
            loss = compute_liger_grpo_loss(
                model, prompt_ids, completion_ids, prompt_mask_batched,
                completion_mask_for_loss, advantages, args, liger_loss,
            )
        else:
            loss = compute_grpo_loss(
                model, prompt_completion_ids, prompt_ids, completion_ids,
                attention_mask, completion_mask_for_loss, advantages, args,
            )
        
        (loss / args.gradient_accumulation_steps).backward()

        # Cleanup
        del prompt_completion_ids, completion_ids, attention_mask, completion_mask, advantages
        
        # Optimizer step
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            
            if args.use_vllm and vllm_client is not None:
                vllm_client.update_model_params(model)
            
            torch.cuda.empty_cache()
        
        scheduler.step()

        # Log
        run_log["steps"][step] = {
            "subject": question_data["subject"],
            "question": question_text,
            "correct_answer": correct_answer,
            "completions": [
                {"text": t, "reward": float(r)} 
                for t, r in zip(completions_text, rewards)
            ],
            "mean_reward": float(mean_reward),
            "loss": loss.item(),
            "lr": scheduler.get_last_lr()[0],
        }

        # Save checkpoint
        if (step + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_step_{step+1}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            print(f"\nSaved checkpoint at step {step+1}")

        # Save log
        with open(os.path.join(args.output_dir, "run_log.json"), "w") as f:
            json.dump(run_log, f, indent=2)

    # Final save
    final_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
