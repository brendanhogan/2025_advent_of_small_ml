import os
import torch
import random
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

import re


####################
## MISC FUNCTIONS ##
####################


def seed_everything(seed: int) -> None:
    """
    Set random seed for reproducibility across multiple libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Additional settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

########################
##  String Formatting ##
########################

def format_prompt(system_prompt: str, question: str, tokenizer) -> Tuple[str, torch.Tensor, torch.Tensor]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=True)
    return prompt_text, inputs["input_ids"], inputs["attention_mask"]


def extract_answer(text: str) -> str:
    """Extract answer from <answer></answer> tags."""
    pattern = r"<answer>([^<]*)</answer>"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def check_format(text: str) -> float:
    """Check if text contains both <think></think> and <answer></answer> tags. Returns 0.2 if correct format, -0.5 if wrong format."""
    has_think = re.search(r"<think>.*?</think>", text, re.IGNORECASE | re.DOTALL)
    has_answer = re.search(r"<answer>.*?</answer>", text, re.IGNORECASE | re.DOTALL)
    
    if has_think and has_answer:
        return 0.2  # Correct format
    else:
        return -0.5  # Penalty for wrong format

