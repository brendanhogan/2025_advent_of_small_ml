"""
Simple GRPO training for image description task
Train VLM to describe images, reward = cosine similarity between original and regenerated image
"""

import os
import json
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict

from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig
from qwen_vl_utils import process_vision_info

import grpo_llms
import grpo_datasets
import grpo_evaluator
import grpo_utils
import grpo_logging


def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    image_path: str,
    prompt: str,
    device: str,
    args: argparse.Namespace,
    eval: bool = False
):
    """Generate multiple completions for image description"""
    
    # Resize image to save memory before processing
    from PIL import Image
    import tempfile
    import os
    original_image_path = image_path
    image = Image.open(image_path).convert('RGB')
    max_size = 224  # Resize to max 224x224 to save memory (matches original code)
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        # Save resized image to temp file (will be cleaned up by OS)
        temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
        os.close(temp_fd)
        image.save(temp_path)
        image_path = temp_path
        # Note: temp file will be cleaned up by OS, but we could add cleanup later if needed
    
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant that describes images in detail.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ],
        },
    ]
    
    text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    image_inputs, video_inputs = process_vision_info(conversation)
    
    prompt_inputs = tokenizer(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device).to(model.dtype)
    
    # Repeat for batch generation
    if eval:
        num_chains = args.num_chains_eval
    else:
        num_chains = args.num_chains
    
    batched_prompt_inputs = {}
    for key, value in prompt_inputs.items():
        if torch.is_tensor(value):
            batched_prompt_inputs[key] = value.repeat(num_chains, *([1] * (value.dim() - 1)))
        else:
            batched_prompt_inputs[key] = value
    
    original_prompt_ids = prompt_inputs["input_ids"]
    
    generation_config = GenerationConfig(
        max_new_tokens=args.max_completion_length,
        do_sample=True,
        temperature=args.temperature,
        pad_token_id=tokenizer.tokenizer.pad_token_id,
    )
    
    prompt_completion_ids = model.generate(
        **batched_prompt_inputs,
        generation_config=generation_config
    )
    
    # Extract completion ids
    prompt_length = original_prompt_ids.size(1)
    prompt_ids = prompt_completion_ids[:, :prompt_length]
    completion_ids = prompt_completion_ids[:, prompt_length:]
    
    # Create masks
    is_eos = completion_ids == tokenizer.tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
    
    prompt_mask = batched_prompt_inputs["attention_mask"]
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    
    # Decode completions
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt


def score_completions(
    completions_text: list[str],
    prompt: str,
    image_path: str,
    eval_class: grpo_evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace
):
    """Score completions and compute advantages"""
    
    log_data = {
        'prompt': {
            'text': prompt,
            'image_path': image_path
        },
        'generations': []
    }
    
    # Format for evaluator
    mock_completions = [[{'content': completion}] for completion in completions_text]
    rewards_per_func, metrics, generated_images, descriptions, cosine_similarities = eval_class.compute_rewards(
        prompts=None,
        completions=mock_completions,
        image_path=image_path,
        device=device
    )
    
    rewards = rewards_per_func.sum(dim=1)
    
    # Store generation data
    for i, (completion, reward_scores) in enumerate(zip(completions_text, rewards_per_func)):
        generation_data = {
            'response': completion,
            'scores': {
                **eval_class.get_reward_breakdown(reward_scores),
                'total_reward': rewards[i].item()
            }
        }
        log_data['generations'].append(generation_data)
    
    # Compute advantages
    mean_grouped_rewards = rewards.view(-1, args.num_chains).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, args.num_chains).std(dim=1)
    
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(args.num_chains, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(args.num_chains, dim=0)
    
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
    metrics["reward_std"] = std_grouped_rewards.mean().item()
    
    log_data['summary_stats'] = {
        'mean_rewards_per_group': mean_grouped_rewards.tolist(),
        'std_rewards_per_group': std_grouped_rewards.tolist(),
        'advantages': advantages.tolist()
    }
    
    return rewards, advantages, rewards_per_func, metrics, log_data, generated_images, descriptions, cosine_similarities


def compute_loss(
    model: PreTrainedModel,
    prompt_completion_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    args: argparse.Namespace,
    img_path: str,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str
):
    """Compute GRPO loss (without KL penalty)"""
    
    logits_to_keep = completion_ids.size(1)
    
    # Get training model logits
    per_token_logps = grpo_utils.get_per_token_logps_vl(
        model, prompt_completion_ids, attention_mask, img_path, tokenizer, logits_to_keep, prompt
    )
    
    # Compute loss with advantages (no KL penalty)
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -per_token_loss
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    
    # Metrics
    metrics = {}
    response_length = completion_mask.sum(1).float().mean().item()
    metrics["response_length"] = response_length
    
    return loss, metrics


def grpo_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    img_path: str,
    eval_class: grpo_evaluator.RewardEvaluator,
    device: str,
    round_num: int,
    training_log_dir: str,
    output_dir: str,
    args: argparse.Namespace
):
    """Compute GRPO loss for one batch"""
    
    # Generate completions
    prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text = generate_completions(
        model, tokenizer, img_path, prompt, device, args
    )
    
    # Score completions
    rewards, advantages, rewards_per_func, metrics, log_data, generated_images, descriptions, cosine_similarities = score_completions(
        completions_text, prompt, img_path, eval_class, device, args
    )
    
    # Write log data
    log_file = os.path.join(training_log_dir, f'{round_num}_generations.txt')
    grpo_utils.write_generation_log(log_data, log_file)
    
    # Save images and create PDF
    avg_cosine = sum(cosine_similarities) / len(cosine_similarities) if cosine_similarities else 0.0
    grpo_logging.save_step_images(
        round_num, img_path, generated_images, descriptions, cosine_similarities,
        output_dir, is_training=True
    )
    try:
        grpo_logging.create_step_pdf(
            round_num, img_path, generated_images, descriptions, cosine_similarities,
            output_dir, is_training=True
        )
    except Exception as e:
        print(f"Warning: Failed to create PDF for step {round_num}: {e}")
    grpo_logging.update_metrics_json(round_num, avg_cosine, output_dir, is_training=True)
    
    # Compute loss
    completion_mask = attention_mask[:, prompt_ids.size(1):]
    loss, loss_metrics = compute_loss(
        model, prompt_completion_ids, prompt_ids, completion_ids,
        attention_mask, completion_mask, advantages, args, img_path, tokenizer, prompt
    )
    
    metrics.update(loss_metrics)
    
    return loss, metrics, rewards.mean().item()


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training for image description")
    
    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="run_1")
    
    # Training
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_train_iters", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    
    # Generation
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--num_chains", type=int, default=8)
    parser.add_argument("--num_chains_eval", type=int, default=2)
    parser.add_argument("--max_completion_length", type=int, default=512)
    
    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=25)
    parser.add_argument("--save_iterations", type=int, default=25)
    
    # Data
    parser.add_argument("--train_dir", type=str, default="train_images")
    parser.add_argument("--test_dir", type=str, default="test_images")
    
    parser.add_argument("--seed", type=int, default=7111994)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # Seed
    grpo_utils.seed_everything(args.seed)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision('high')
    
    # Load model
    model, tokenizer = grpo_llms.get_llm_tokenizer(args.model_name, device)
    
    # Load datasets
    train_loader = grpo_datasets.CharXivImageLoader(args.train_dir, random=True)
    test_loader = grpo_datasets.CharXivImageLoader(args.test_dir, random=False)

    
    # Load evaluator (with DINOv2 for image embeddings)
    eval_class = grpo_evaluator.ImageDescriptionEvaluator(device=device)
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    train_log_dir = os.path.join(args.output_dir, 'training_logs')
    os.makedirs(train_log_dir, exist_ok=True)
    eval_log_dir = os.path.join(args.output_dir, 'eval_logs')
    os.makedirs(eval_log_dir, exist_ok=True)
    
    # Setup checkpoint directory
    ckpt_dir = os.path.join(args.output_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        weight_decay=0.1,
        eps=1e-8
    )
    
    # Training loop
    train_metrics_total = {}
    optimizer.zero_grad()
    
    for round_num in tqdm(range(args.num_train_iters), desc="Training"):
        
        # Evaluation
        if round_num % args.eval_interval == 0:
            print(f"\nEvaluating at round {round_num}...")
            test_loader.reset()
            eval_rewards = []
            eval_cosine_similarities = []
            
            for eval_idx in range(min(10, len(test_loader))):  # Eval on 10 images
                img_path = next(test_loader)
                _, _, _, _, completions_text, _ = generate_completions(
                    model, tokenizer, img_path, train_loader.prompt, device, args, eval=True
                )

                mock_completions = [[{'content': c}] for c in completions_text]
                _, metrics, generated_images, descriptions, cosine_similarities = eval_class.compute_rewards(
                    prompts=None, completions=mock_completions, image_path=img_path, device=device
                )
                
                # Log eval step (use round_num * 1000 + eval_idx to make unique step numbers)
                eval_step_num = round_num * 1000 + eval_idx
                avg_cosine = sum(cosine_similarities) / len(cosine_similarities) if cosine_similarities else 0.0
                eval_cosine_similarities.append(avg_cosine)
                eval_rewards.append(metrics['reward'])
                
                # Save images and create PDF for this eval image
                grpo_logging.save_step_images(
                    eval_step_num, img_path, generated_images, descriptions, cosine_similarities,
                    args.output_dir, is_training=False
                )
                try:
                    grpo_logging.create_step_pdf(
                        eval_step_num, img_path, generated_images, descriptions, cosine_similarities,
                        args.output_dir, is_training=False
                    )
                except Exception as e:
                    print(f"Warning: Failed to create PDF for eval step {eval_step_num}: {e}")
            
            avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
            avg_eval_cosine = sum(eval_cosine_similarities) / len(eval_cosine_similarities) if eval_cosine_similarities else 0.0
            print(f"Average eval reward: {avg_eval_reward:.4f}")
            print(f"Average eval cosine similarity: {avg_eval_cosine:.4f}")
            
            # Update eval metrics JSON (use round_num as the step identifier)
            grpo_logging.update_metrics_json(round_num, avg_eval_cosine, args.output_dir, is_training=False)
            
            with open(os.path.join(eval_log_dir, f'eval_{round_num}.json'), 'w') as f:
                json.dump({
                    'round': round_num,
                    'avg_reward': avg_eval_reward,
                    'avg_cosine_similarity': avg_eval_cosine
                }, f, indent=2)
        
        # Get next image
        img_path = next(train_loader)
        
        # GRPO step
        total_loss, train_metrics, reward = grpo_loss(
            model, tokenizer, train_loader.prompt, img_path,
            eval_class, device, round_num, train_log_dir, args.output_dir, args
        )
        
        # Backward
        total_loss = total_loss / args.gradient_accumulation_steps
        total_loss.backward()
        
        # Step optimizer
        if (round_num + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        # Log
        train_metrics["loss"] = total_loss.item() * args.gradient_accumulation_steps
        train_metrics["reward"] = reward
        train_metrics_total[round_num] = train_metrics
        
        with open(os.path.join(train_log_dir, "train_logs.json"), "w") as f:
            json.dump(train_metrics_total, f, indent=2)
        
        # Save checkpoint
        if (round_num + 1) % args.save_iterations == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt-{round_num + 1}")
            os.makedirs(ckpt_path, exist_ok=True)
            
            # Save model state dict
            model_path = os.path.join(ckpt_path, "model.pt")
            torch.save(model.state_dict(), model_path)
            
            # Save optimizer state
            # optimizer_path = os.path.join(ckpt_path, "optimizer.pt")
            # torch.save(optimizer.state_dict(), optimizer_path)
            
            # Save training info
            info_path = os.path.join(ckpt_path, "checkpoint_info.json")
            with open(info_path, 'w') as f:
                json.dump({
                    'round': round_num + 1,
                    'model_name': args.model_name,
                    'learning_rate': args.learning_rate,
                    'metrics': train_metrics
                }, f, indent=2)
            
            print(f"Saved checkpoint to {ckpt_path}")
        
        torch.cuda.empty_cache()

