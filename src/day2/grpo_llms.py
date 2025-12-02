"""
Model loading utilities for GRPO training
"""

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    PreTrainedModel,
    PreTrainedTokenizerBase
)


def get_llm_tokenizer(model_name: str, device: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load and configure a vision-language model and its processor.
    
    Args:
        model_name: Name or path of the pretrained model to load
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        tuple containing:
            - The loaded model
            - The configured processor for that model
    """
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    
    processor.tokenizer.padding_side = "left"
    processor.padding_side = "left"
    
    model.config.use_cache = False
    
    return model, processor

