"""
Module for loading LLMs and their tokenizers from huggingface. 

"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from liger_kernel.transformers import AutoLigerKernelForCausalLM


def get_llm_tokenizer(model_name: str, use_liger_model: bool = False) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load and configure a language model and its tokenizer.

    Args:
        model_name: Name or path of the pretrained model to load
        use_liger_model: Whether to use Liger kernel model

    Returns:
        tuple containing:
            - The loaded language model
            - The configured tokenizer for that model
    """
    if use_liger_model:
        model = AutoLigerKernelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
            device_map="auto"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


if __name__ == "__main__": 
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    get_llm_tokenizer(model_name)