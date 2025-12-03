"""
Utility functions for GRPO training
"""

import os
import json
import torch
import random
import numpy as np
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info


def seed_everything(seed: int) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_generation_log(log_data: dict, log_file: str) -> None:
    """Write generation log data to a text file"""
    with open(log_file, 'w') as f:
        f.write("###### ORIGINAL PROMPT #####\n\n")
        f.write(log_data['prompt']['text'] + "\n\n")
        f.write("#### IMAGE PATH ####\n\n")
        f.write(str(log_data['prompt']['image_path']) + "\n")
        
        for i, gen in enumerate(log_data['generations'], 1):
            f.write(f"#### GENERATION {i} RESPONSE ####\n\n")
            f.write(gen['response'] + "\n\n")
            f.write(f"#### GENERATION {i} SCORES ####\n")
            
            if 'scores' in gen and isinstance(gen['scores'], dict):
                for score_name, score_value in gen['scores'].items():
                    formatted_name = score_name.replace('_', ' ').capitalize()
                    try:
                        f.write(f"{formatted_name}: {float(score_value):.4f}\n")
                    except (ValueError, TypeError):
                        f.write(f"{formatted_name}: {score_value}\n")
            else:
                f.write("No scores available for this generation.\n")
            
            f.write("\n")


def selective_log_softmax(logits, index):
    """
    Memory-efficient log_softmax -> gather operation.
    Copied from TRL: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
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


def get_per_token_logps_vl(model, input_ids, attention_mask, image_path, tokenizer, logits_to_keep, prompt):
    """
    Get per-token log probabilities for vision-language model.
    Adapted from DeepSeekRL-Extended utils.py
    """
    
    # Resize image to save memory before processing
    from PIL import Image
    import tempfile
    import os
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
    
    text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False, padding_side="left")
    image_inputs, video_inputs = process_vision_info(conversation)
    
    prompt_inputs = tokenizer(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        padding_side="left"
    ).to(model.device).to(model.dtype)
    
    # Repeat input tensors for batch
    batch_size = input_ids.shape[0]
    batched_prompt_inputs = {}
    for key, value in prompt_inputs.items():
        if torch.is_tensor(value):
            batched_prompt_inputs[key] = value.repeat(batch_size, *([1] * (value.dim() - 1)))
        else:
            batched_prompt_inputs[key] = value
    
    batched_prompt_inputs["input_ids"] = input_ids
    batched_prompt_inputs["attention_mask"] = attention_mask
    
    # Get logits
    logits = model(**batched_prompt_inputs).logits
    logits = logits[:, :-1, :]  # Exclude last logit
    
    input_ids_trimmed = input_ids[:, -logits_to_keep:]
    logits_trimmed = logits[:, -logits_to_keep:]
    
    return selective_log_softmax(logits_trimmed, input_ids_trimmed)


def get_per_token_logps_text(model, input_ids, attention_mask, tokenizer, logits_to_keep, prompt):
    """
    Get per-token log probabilities for text-only model (adversary).
    """
    
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
    
    text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False, padding_side="left")
    
    prompt_inputs = tokenizer(
        text=[text],
        padding=True,
        return_tensors="pt",
        padding_side="left"
    ).to(model.device).to(model.dtype)
    
    # Repeat input tensors for batch
    batch_size = input_ids.shape[0]
    batched_prompt_inputs = {}
    for key, value in prompt_inputs.items():
        if torch.is_tensor(value):
            batched_prompt_inputs[key] = value.repeat(batch_size, *([1] * (value.dim() - 1)))
        else:
            batched_prompt_inputs[key] = value
    
    batched_prompt_inputs["input_ids"] = input_ids
    batched_prompt_inputs["attention_mask"] = attention_mask
    
    # Get logits
    logits = model(**batched_prompt_inputs).logits
    logits = logits[:, :-1, :]  # Exclude last logit
    
    input_ids_trimmed = input_ids[:, -logits_to_keep:]
    logits_trimmed = logits[:, -logits_to_keep:]
    
    return selective_log_softmax(logits_trimmed, input_ids_trimmed)

