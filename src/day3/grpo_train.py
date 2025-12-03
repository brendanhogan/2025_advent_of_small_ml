"""
Adversarial GRPO training for image description task
Train base VLM to describe images, train adversary VLM to generate challenging images
Reward for base = cosine similarity between original and regenerated image
Reward for adversary = 1 - cosine similarity (want images base model struggles with)
"""

import os
import json
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
import copy
import tempfile

from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig
from qwen_vl_utils import process_vision_info

import grpo_llms
import grpo_datasets
import grpo_evaluator
import grpo_utils
import grpo_logging
import time


def replicate_run_with_retry(
    model: str,
    input_params: dict,
    max_retries: int = 10,
    base_delay: float = 2.0,
    max_delay: float = 60.0
):
    """
    Run replicate.run with retry logic for handling NSFW/content filter errors.
    
    Args:
        model: Model identifier (e.g., "black-forest-labs/flux-schnell")
        input_params: Input parameters for the model
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        max_delay: Maximum delay in seconds
    
    Returns:
        Output from replicate.run()
    """
    import replicate
    
    for attempt in range(max_retries):
        try:
            output = replicate.run(model, input=input_params)
            return output
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = delay * 0.1 * (0.5 - time.time() % 1)  # Random jitter
                delay += jitter
                
                print(f"Warning: Replicate call failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                # Last attempt
                print(f"Error: Replicate call failed after {attempt + 1} attempts: {e}")
                raise
    
    # Should never reach here, but just in case
    raise RuntimeError(f"Failed to get successful response from Replicate after {max_retries} attempts")


def generate_text_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    device: str,
    args: argparse.Namespace,
    num_completions: int = None,
    eval: bool = False
):
    """Generate text-only completions (for adversary prompt generation)"""
    
    if num_completions is None:
        num_completions = args.num_chains_eval if eval else args.num_chains
    
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant that describes scientific plots and charts.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    
    text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    
    prompt_inputs = tokenizer(
        text=[text],
        padding=True,
        return_tensors="pt",
    ).to(model.device).to(model.dtype)
    
    # Repeat for batch generation
    batched_prompt_inputs = {}
    for key, value in prompt_inputs.items():
        if torch.is_tensor(value):
            batched_prompt_inputs[key] = value.repeat(num_completions, *([1] * (value.dim() - 1)))
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
    if img_path is not None:
        # Vision-language model (base model)
        per_token_logps = grpo_utils.get_per_token_logps_vl(
        model, prompt_completion_ids, attention_mask, img_path, tokenizer, logits_to_keep, prompt
        )
    else:
        # Text-only model (adversary)
        per_token_logps = grpo_utils.get_per_token_logps_text(
            model, prompt_completion_ids, attention_mask, tokenizer, logits_to_keep, prompt
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


def grpo_loss_base(
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
    """Compute GRPO loss for base model (describing images)"""
    
    # Generate completions
    prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text = generate_completions(
        model, tokenizer, img_path, prompt, device, args
    )
    
    # Score completions
    rewards, advantages, rewards_per_func, metrics, log_data, generated_images, descriptions, cosine_similarities = score_completions(
        completions_text, prompt, img_path, eval_class, device, args
    )
    
    # Write log data
    log_file = os.path.join(training_log_dir, f'base_{round_num}_generations.txt')
    grpo_utils.write_generation_log(log_data, log_file)
    
    # Save images and create PDF
    avg_cosine = sum(cosine_similarities) / len(cosine_similarities) if cosine_similarities else 0.0
    grpo_logging.save_step_images(
        round_num, img_path, generated_images, descriptions, cosine_similarities,
        output_dir, is_training=True, model_type='base'
    )
    try:
        grpo_logging.create_step_pdf(
            round_num, img_path, generated_images, descriptions, cosine_similarities,
            output_dir, is_training=True, model_type='base'
        )
    except Exception as e:
        print(f"Warning: Failed to create PDF for step {round_num}: {e}")
    grpo_logging.update_metrics_json(round_num, avg_cosine, output_dir, is_training=True, model_type='base')
    
    # Compute loss
    completion_mask = attention_mask[:, prompt_ids.size(1):]
    loss, loss_metrics = compute_loss(
        model, prompt_completion_ids, prompt_ids, completion_ids,
        attention_mask, completion_mask, advantages, args, img_path, tokenizer, prompt
    )
    
    metrics.update(loss_metrics)
    
    return loss, metrics, rewards.mean().item()


def grpo_loss_adversary(
    adversary_model: PreTrainedModel,
    adversary_tokenizer: PreTrainedTokenizerBase,
    base_model: PreTrainedModel,
    base_tokenizer: PreTrainedTokenizerBase,
    adversary_prompt: str,
    eval_class: grpo_evaluator.RewardEvaluator,
    device: str,
    round_num: int,
    training_log_dir: str,
    output_dir: str,
    args: argparse.Namespace
):
    """Compute GRPO loss for adversary model (generating challenging image prompts)"""
    
    # Step 1: Generate multiple prompts from adversary
    # Set model to eval mode for generation to avoid graph issues
    prompt_completion_ids, prompt_ids, completion_ids, attention_mask, adversary_prompts, _ = generate_text_completions(
            adversary_model, adversary_tokenizer, adversary_prompt, device, args, num_completions=args.num_chains
        )
    # Set back to train mode for loss computation
    adversary_model.train()
    
    # Clone tensors to create fresh copies for loss computation
    prompt_completion_ids = prompt_completion_ids.clone()
    prompt_ids = prompt_ids.clone()
    completion_ids = completion_ids.clone()
    attention_mask = attention_mask.clone()
    
    # Step 2: Generate images from adversary prompts
    adversary_generated_images = []
    adversary_image_paths = []
    for adv_prompt in adversary_prompts:
        # Generate image using Replicate with retry logic
        output = replicate_run_with_retry(
            "black-forest-labs/flux-schnell",
            {
                "prompt": adv_prompt,
                "go_fast": True,
                "megapixels": "1",
                "num_outputs": 1,
                "aspect_ratio": "1:1",
                "output_format": "webp",
                "output_quality": 80,
                "num_inference_steps": 4
            }
        )
        
        # Save to temp file
        import io
        from PIL import Image
        image_data = output[0].read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        max_size = 224
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
        os.close(temp_fd)
        image.save(temp_path)
        adversary_generated_images.append(image)
        adversary_image_paths.append(temp_path)
    
    # Step 3: For each adversary-generated image, have base model describe it and reconstruct
    base_descriptions = []
    reconstructed_images = []
    cosine_similarities = []
    
    base_description_prompt = "Describe this image in detail. Be specific about visual elements, colors, shapes, text, and any other important features."
    
    for adv_img_path in adversary_image_paths:
        # Get base model description (no gradients needed)
        with torch.no_grad():
            _, _, _, _, base_descs, _ = generate_completions(
                base_model, base_tokenizer, adv_img_path, base_description_prompt, device, args, eval=True
            )
        base_desc = base_descs[0]  # Take first completion
        base_descriptions.append(base_desc)
        
        # Generate image from base description with retry logic
        import io
        from PIL import Image
        output = replicate_run_with_retry(
            "black-forest-labs/flux-schnell",
            {
                "prompt": base_desc,
                "go_fast": True,
                "megapixels": "1",
                "num_outputs": 1,
                "aspect_ratio": "1:1",
                "output_format": "webp",
                "output_quality": 80,
                "num_inference_steps": 4
            }
        )
        
        image_data = output[0].read()
        recon_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        max_size = 224
        if max(recon_image.size) > max_size:
            recon_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        reconstructed_images.append(recon_image)
        
        # Compute cosine similarity between original and reconstructed
        rewards_per_func, metrics, _, _, cosine_sims = eval_class.compute_rewards(
            prompts=None,
            completions=[[{'content': base_desc}]],
            image_path=adv_img_path,
            device=device
        )
        cosine_sim = cosine_sims[0] if cosine_sims else 0.0
        cosine_similarities.append(cosine_sim)
    
    # Step 4: Reward = 1 - cosine_similarity (adversary wants low similarity)
    rewards = torch.tensor([1.0 - cs for cs in cosine_similarities], device=device)
    
    # Compute advantages
    mean_reward = rewards.mean()
    std_reward = rewards.std()
    advantages = (rewards - mean_reward) / (std_reward + 1e-4)
    
    # Log data for adversary
    log_data = {
        'prompt': {
            'text': adversary_prompt,
            'type': 'adversary_prompt_generation'
        },
        'generations': []
    }
    
    for i, (adv_prompt, adv_img, base_desc, recon_img, cosine_sim, reward_val) in enumerate(
        zip(adversary_prompts, adversary_generated_images, base_descriptions, 
            reconstructed_images, cosine_similarities, rewards.tolist())
    ):
        generation_data = {
            'adversary_prompt': adv_prompt,
            'base_description': base_desc,
            'cosine_similarity': cosine_sim,
            'reward': reward_val
        }
        log_data['generations'].append(generation_data)
    
    log_data['summary_stats'] = {
        'mean_reward': mean_reward.item(),
        'std_reward': std_reward.item(),
        'mean_cosine_similarity': sum(cosine_similarities) / len(cosine_similarities) if cosine_similarities else 0.0,
        'advantages': advantages.tolist()
    }
    
    # Write log data
    log_file = os.path.join(training_log_dir, f'adversary_{round_num}_generations.txt')
    with open(log_file, 'w') as f:
        f.write("###### ADVERSARY PROMPT #####\n\n")
        f.write(adversary_prompt + "\n\n")
        
        for i, gen in enumerate(log_data['generations'], 1):
            f.write(f"#### GENERATION {i} ####\n\n")
            f.write(f"Adversary Prompt: {gen['adversary_prompt']}\n\n")
            f.write(f"Base Model Description: {gen['base_description']}\n\n")
            f.write(f"Cosine Similarity: {gen['cosine_similarity']:.4f}\n")
            f.write(f"Reward (1 - cosine): {gen['reward']:.4f}\n\n")
    
    # Save images and create PDF
    avg_cosine = sum(cosine_similarities) / len(cosine_similarities) if cosine_similarities else 0.0
    avg_reward = mean_reward.item()
    
    grpo_logging.save_adversary_step_images(
        round_num, adversary_prompts, adversary_generated_images, 
        base_descriptions, reconstructed_images, cosine_similarities,
        output_dir
    )
    try:
        grpo_logging.create_adversary_step_pdf(
            round_num, adversary_prompts, adversary_generated_images,
            base_descriptions, reconstructed_images, cosine_similarities,
            output_dir
        )
    except Exception as e:
        print(f"Warning: Failed to create PDF for adversary step {round_num}: {e}")
    
    grpo_logging.update_metrics_json(round_num, avg_cosine, output_dir, is_training=True, model_type='adversary')
    grpo_logging.update_adversary_reward_json(round_num, avg_reward, output_dir)
    
    # Compute loss
    completion_mask = attention_mask[:, prompt_ids.size(1):]
    loss, loss_metrics = compute_loss(
        adversary_model, prompt_completion_ids, prompt_ids, completion_ids,
        attention_mask, completion_mask, advantages, args, None, adversary_tokenizer, adversary_prompt
    )
    
    metrics = {
        "reward": avg_reward,
        "cosine_similarity": avg_cosine,
        "reward_std": std_reward.item(),
        **loss_metrics
    }
    
    # Clean up temp files
    for temp_path in adversary_image_paths:
        try:
            os.unlink(temp_path)
        except:
            pass
    
    return loss, metrics, avg_reward


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training for image description")
    
    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="adv_run_2")
    
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
    
    # Load base model
    base_model, base_tokenizer = grpo_llms.get_llm_tokenizer(args.model_name, device)
    
    # Load adversary model (copy from base model)
    adversary_model, adversary_tokenizer = grpo_llms.get_llm_tokenizer(args.model_name, device)
    adversary_model.load_state_dict(copy.deepcopy(base_model.state_dict()))
    
    # Load datasets
    train_loader = grpo_datasets.CharXivImageLoader(args.train_dir, random=True)
    test_loader = grpo_datasets.CharXivImageLoader(args.test_dir, random=False)

    # Adversary prompt
    adversary_prompt = grpo_datasets.CharXivImageLoader.get_adversary_prompt()
    
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
    
    # Optimizers (separate for base and adversary)
    base_optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        weight_decay=0.1,
        eps=1e-8
    )
    
    adversary_optimizer = torch.optim.AdamW(
        adversary_model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        weight_decay=0.1,
        eps=1e-8
    )
    
    # Training loop
    train_metrics_total = {}
    base_optimizer.zero_grad()
    adversary_optimizer.zero_grad()
    
    for round_num in tqdm(range(args.num_train_iters), desc="Training"):
        
        # Determine if this is base or adversary training (strict alternation)
        is_base_training = (round_num % 2 == 0)
        
        # Evaluation
        if round_num % args.eval_interval == 0:
            print(f"\nEvaluating at round {round_num}...")
            
            # Base model evaluation
            test_loader.reset()
            eval_rewards = []
            eval_cosine_similarities = []
            
            for eval_idx in range(min(10, len(test_loader))):  # Eval on 10 images
                img_path = next(test_loader)
                _, _, _, _, completions_text, _ = generate_completions(
                    base_model, base_tokenizer, img_path, train_loader.prompt, device, args, eval=True
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
                    args.output_dir, is_training=False, model_type='base'
                )
                try:
                    grpo_logging.create_step_pdf(
                        eval_step_num, img_path, generated_images, descriptions, cosine_similarities,
                        args.output_dir, is_training=False, model_type='base'
                    )
                except Exception as e:
                    print(f"Warning: Failed to create PDF for eval step {eval_step_num}: {e}")
            
            avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
            avg_eval_cosine = sum(eval_cosine_similarities) / len(eval_cosine_similarities) if eval_cosine_similarities else 0.0
            print(f"Base model - Average eval reward: {avg_eval_reward:.4f}")
            print(f"Base model - Average eval cosine similarity: {avg_eval_cosine:.4f}")
            
            # Update eval metrics JSON
            grpo_logging.update_metrics_json(round_num, avg_eval_cosine, args.output_dir, is_training=False, model_type='base')
            
            with open(os.path.join(eval_log_dir, f'base_eval_{round_num}.json'), 'w') as f:
                json.dump({
                    'round': round_num,
                    'avg_reward': avg_eval_reward,
                    'avg_cosine_similarity': avg_eval_cosine
                }, f, indent=2)
        
            # Adversary evaluation
            print("Evaluating adversary...")
            adversary_eval_cosine_similarities = []
            adversary_eval_rewards = []
            
            # Collect all data for logging
            eval_adversary_prompts = []
            eval_adversary_images = []
            eval_base_descriptions = []
            eval_reconstructed_images = []
            eval_cosine_similarities_list = []
            eval_temp_paths = []
            
            for eval_idx in range(min(10, len(test_loader))):  # Same number as base eval
                # Generate adversary prompt
                _, _, _, _, adv_prompts, _ = generate_text_completions(
                    adversary_model, adversary_tokenizer, adversary_prompt, device, args, 
                    num_completions=1, eval=True
                )
                adv_prompt = adv_prompts[0]
                eval_adversary_prompts.append(adv_prompt)
                
                # Generate image from adversary prompt with retry logic
                import io
                from PIL import Image
                output = replicate_run_with_retry(
                    "black-forest-labs/flux-schnell",
                    {
                        "prompt": adv_prompt,
                        "go_fast": True,
                        "megapixels": "1",
                        "num_outputs": 1,
                        "aspect_ratio": "1:1",
                        "output_format": "webp",
                        "output_quality": 80,
                        "num_inference_steps": 4
                    }
                )
                
                image_data = output[0].read()
                adv_image = Image.open(io.BytesIO(image_data)).convert('RGB')
                max_size = 224
                if max(adv_image.size) > max_size:
                    adv_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
                os.close(temp_fd)
                adv_image.save(temp_path)
                eval_adversary_images.append(adv_image)
                eval_temp_paths.append(temp_path)
                
                # Base model describes it (no gradients needed during eval)
                with torch.no_grad():
                    _, _, _, _, base_descs, _ = generate_completions(
                        base_model, base_tokenizer, temp_path, train_loader.prompt, device, args, eval=True
                    )
                base_desc = base_descs[0]
                eval_base_descriptions.append(base_desc)
                
                # Generate reconstructed image from base description with retry logic
                output_recon = replicate_run_with_retry(
                    "black-forest-labs/flux-schnell",
                    {
                        "prompt": base_desc,
                        "go_fast": True,
                        "megapixels": "1",
                        "num_outputs": 1,
                        "aspect_ratio": "1:1",
                        "output_format": "webp",
                        "output_quality": 80,
                        "num_inference_steps": 4
                    }
                )
                
                recon_image_data = output_recon[0].read()
                recon_image = Image.open(io.BytesIO(recon_image_data)).convert('RGB')
                if max(recon_image.size) > max_size:
                    recon_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                eval_reconstructed_images.append(recon_image)
                
                # Compute cosine similarity
                mock_completions = [[{'content': base_desc}]]
                _, metrics, _, _, cosine_similarities = eval_class.compute_rewards(
                    prompts=None, completions=mock_completions, image_path=temp_path, device=device
                )
                cosine_sim = cosine_similarities[0] if cosine_similarities else 0.0
                reward = 1.0 - cosine_sim
                
                eval_cosine_similarities_list.append(cosine_sim)
                adversary_eval_cosine_similarities.append(cosine_sim)
                adversary_eval_rewards.append(reward)
            
            # Log adversary evaluation results
            for eval_idx in range(len(eval_adversary_prompts)):
                eval_step_num = round_num * 1000 + eval_idx
                
                # Save images and text files for this eval step
                grpo_logging.save_adversary_step_images(
                    eval_step_num, 
                    [eval_adversary_prompts[eval_idx]], 
                    [eval_adversary_images[eval_idx]],
                    [eval_base_descriptions[eval_idx]], 
                    [eval_reconstructed_images[eval_idx]], 
                    [eval_cosine_similarities_list[eval_idx]],
                    args.output_dir,
                    is_training=False
                )
                
                # Create PDF for this eval step
                try:
                    grpo_logging.create_adversary_step_pdf(
                        eval_step_num,
                        [eval_adversary_prompts[eval_idx]],
                        [eval_adversary_images[eval_idx]],
                        [eval_base_descriptions[eval_idx]],
                        [eval_reconstructed_images[eval_idx]],
                        [eval_cosine_similarities_list[eval_idx]],
                        args.output_dir,
                        is_training=False
                    )
                except Exception as e:
                    print(f"Warning: Failed to create PDF for adversary eval step {eval_step_num}: {e}")
            
            # Clean up temp files after logging
            for temp_path in eval_temp_paths:
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            avg_adv_eval_reward = sum(adversary_eval_rewards) / len(adversary_eval_rewards) if adversary_eval_rewards else 0.0
            avg_adv_eval_cosine = sum(adversary_eval_cosine_similarities) / len(adversary_eval_cosine_similarities) if adversary_eval_cosine_similarities else 0.0
            print(f"Adversary - Average eval reward: {avg_adv_eval_reward:.4f}")
            print(f"Adversary - Average eval cosine similarity: {avg_adv_eval_cosine:.4f}")
            
            grpo_logging.update_metrics_json(round_num, avg_adv_eval_cosine, args.output_dir, is_training=False, model_type='adversary')
            grpo_logging.update_adversary_reward_json(round_num, avg_adv_eval_reward, args.output_dir, is_training=False)
            
            with open(os.path.join(eval_log_dir, f'adversary_eval_{round_num}.json'), 'w') as f:
                json.dump({
                    'round': round_num,
                    'avg_reward': avg_adv_eval_reward,
                    'avg_cosine_similarity': avg_adv_eval_cosine
                }, f, indent=2)
        
        # Training step
        if is_base_training:
            # Base model training: use adversary-generated image
            print(f"Round {round_num}: Training base model...")
            
            # Step 1: Get adversary prompt and generate image
            _, _, _, _, adv_prompts, _ = generate_text_completions(
                adversary_model, adversary_tokenizer, adversary_prompt, device, args, num_completions=1, eval=True
            )
            adv_prompt_text = adv_prompts[0]
            
            # Generate image from adversary prompt with retry logic
            import io
            from PIL import Image
            output = replicate_run_with_retry(
                "black-forest-labs/flux-schnell",
                {
                    "prompt": adv_prompt_text,
                    "go_fast": True,
                    "megapixels": "1",
                    "num_outputs": 1,
                    "aspect_ratio": "1:1",
                    "output_format": "webp",
                    "output_quality": 80,
                    "num_inference_steps": 4
                }
            )
            
            image_data = output[0].read()
            adv_image = Image.open(io.BytesIO(image_data)).convert('RGB')
            max_size = 224
            if max(adv_image.size) > max_size:
                adv_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
            os.close(temp_fd)
            adv_image.save(temp_path)
            
            # Step 2: Normal base training on this image
            total_loss, train_metrics, reward = grpo_loss_base(
                base_model, base_tokenizer, train_loader.prompt, temp_path,
            eval_class, device, round_num, train_log_dir, args.output_dir, args
            )
            
            # Backward
            total_loss = total_loss / args.gradient_accumulation_steps
            total_loss.backward()
            
            # Step optimizer
            if (round_num + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.max_grad_norm)
                base_optimizer.step()
                base_optimizer.zero_grad()
            
                # Log
                train_metrics["loss"] = total_loss.item() * args.gradient_accumulation_steps
                train_metrics["reward"] = reward
                train_metrics["model_type"] = "base"
                train_metrics_total[round_num] = train_metrics
                
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
        else:
            # Adversary training
            print(f"Round {round_num}: Training adversary model...")
            
            total_loss, train_metrics, reward = grpo_loss_adversary(
                adversary_model, adversary_tokenizer, base_model, base_tokenizer,
                adversary_prompt, eval_class, device, round_num, train_log_dir, args.output_dir, args
            )
            
            # Backward
            total_loss = total_loss / args.gradient_accumulation_steps
            total_loss.backward()
            
            # Step optimizer
            if (round_num + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(adversary_model.parameters(), args.max_grad_norm)
                adversary_optimizer.step()
                adversary_optimizer.zero_grad()
        
        # Log
        train_metrics["loss"] = total_loss.item() * args.gradient_accumulation_steps
        train_metrics["reward"] = reward
        train_metrics["model_type"] = "adversary"
        train_metrics_total[round_num] = train_metrics
        
        with open(os.path.join(train_log_dir, "train_logs.json"), "w") as f:
            json.dump(train_metrics_total, f, indent=2)
        
        # Save checkpoint
        if (round_num + 1) % args.save_iterations == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt-{round_num + 1}")
            os.makedirs(ckpt_path, exist_ok=True)
            
            # Save both models
            base_model_path = os.path.join(ckpt_path, "base_model.pt")
            torch.save(base_model.state_dict(), base_model_path)
            
            adversary_model_path = os.path.join(ckpt_path, "adversary_model.pt")
            torch.save(adversary_model.state_dict(), adversary_model_path)
            
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

