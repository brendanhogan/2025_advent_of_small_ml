"""
Model loading utilities for text-only Qwen2.5-7B-Instruct and GPT-4.1
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI


def get_model_and_tokenizer(model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "cuda"):
    """
    Load Qwen2.5-7B-Instruct for text generation.
    
    Args:
        model_name: HuggingFace model name
        device: Device to load model on
    
    Returns:
        tuple: (model, tokenizer)
    """
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Left padding for batch generation
    tokenizer.padding_side = "left"
    
    # Disable cache for training
    model.config.use_cache = False
    
    return model, tokenizer


def generate_prompts(
    model,
    tokenizer,
    meta_prompt: str,
    num_completions: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    device: str = "cuda"
):
    """
    Generate multiple image prompts from the model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        meta_prompt: The instruction asking for an image prompt
        num_completions: Number of prompts to generate
        max_new_tokens: Maximum tokens per completion
        temperature: Sampling temperature
        device: Device
    
    Returns:
        tuple: (prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text)
    """
    
    # Build conversation
    conversation = [
        {
            "role": "system",
            "content": "You are an avant-garde artist who writes prompts for image generation. You prize originality above all else. Never repeat yourself - each prompt should explore completely different subjects, styles, and moods. Avoid clichés. Your prompts should be vivid, specific, and unexpected."
        },
        {
            "role": "user", 
            "content": meta_prompt
        }
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        conversation, 
        add_generation_prompt=True, 
        tokenize=False
    )
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    # Repeat for batch generation
    batched_inputs = {
        key: value.repeat(num_completions, *([1] * (value.dim() - 1)))
        for key, value in inputs.items()
    }
    
    original_prompt_ids = inputs["input_ids"]
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **batched_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Extract completion ids
    prompt_length = original_prompt_ids.size(1)
    prompt_ids = outputs[:, :prompt_length]
    completion_ids = outputs[:, prompt_length:]
    
    # Create masks
    is_eos = completion_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
    
    prompt_mask = batched_inputs["attention_mask"]
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    
    # Decode completions
    completions_text = tokenizer.batch_decode(
        completion_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    return outputs, prompt_ids, completion_ids, attention_mask, completions_text


def generate_prompts_gpt4(
    meta_prompt: str,
    num_completions: int = 4,
    temperature: float = 1.2,
) -> list[str]:
    """
    Generate image prompts using GPT-4.1.
    
    Args:
        meta_prompt: The instruction asking for an image prompt
        num_completions: Number of prompts to generate
        temperature: Sampling temperature
    
    Returns:
        List of generated prompts
    """
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    system_prompt = "You are an avant-garde artist who writes prompts for image generation. You prize originality above all else. Never repeat yourself - each prompt should explore completely different subjects, styles, and moods. Avoid clichés. Your prompts should be vivid, specific, and unexpected."
    
    prompts = []
    for _ in range(num_completions):
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": meta_prompt}
            ],
            temperature=temperature,
            max_tokens=256,
        )
        prompts.append(response.choices[0].message.content.strip())
    
    return prompts

