"""
Module for loading LLMs and their tokenizers from huggingface. 

"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from liger_kernel.transformers import AutoLigerKernelForCausalLM


# Baguettotron chat template (from chat_template.json on HuggingFace)
# This template automatically adds <think> when add_generation_prompt=True
BAGUETTOTRON_CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}"
)


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
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
            device_map="auto"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Set padding side for generation
    tokenizer.padding_side = "left"
    
    print(f"Loaded model: {model_name}")
    print(f"  Vocab size: {model.config.vocab_size}")
    print(f"  Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    
    # Set chat template for PleIAs models (not included in tokenizer by default)
    # Both Baguettotron and Monad use the same ChatML + <think> format
    if "Baguettotron" in model_name or "baguettotron" in model_name.lower():
        tokenizer.chat_template = BAGUETTOTRON_CHAT_TEMPLATE
    elif "Monad" in model_name or "monad" in model_name.lower():
        tokenizer.chat_template = BAGUETTOTRON_CHAT_TEMPLATE  # Same format as Baguettotron
    
    return model, tokenizer


if __name__ == "__main__": 
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    get_llm_tokenizer(model_name)