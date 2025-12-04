"""Quick test to understand tokenization differences"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello world!"},
]

# Get the chat template output
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Chat template output (raw string):")
print(repr(prompt_text))
print(f"\nLength: {len(prompt_text)} characters")

# Tokenize with add_special_tokens=False
ids_no_special = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
print(f"\nWith add_special_tokens=False:")
print(f"  Token IDs: {ids_no_special[:20]}")
print(f"  Length: {len(ids_no_special)} tokens")
print(f"  Decoded: {tokenizer.decode(ids_no_special[:20])}")

# Tokenize with add_special_tokens=True
ids_with_special = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
print(f"\nWith add_special_tokens=True:")
print(f"  Token IDs: {ids_with_special[:20]}")
print(f"  Length: {len(ids_with_special)} tokens")
print(f"  Decoded: {tokenizer.decode(ids_with_special[:20])}")

# Check for BOS token
print(f"\nBOS token ID: {tokenizer.bos_token_id}")
print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"\nFirst token with add_special_tokens=False: {ids_no_special[0]}")
print(f"First token with add_special_tokens=True: {ids_with_special[0]}")

# Check tokenizer config
print(f"\nTokenizer add_bos_token: {tokenizer.add_bos_token if hasattr(tokenizer, 'add_bos_token') else 'Not set'}")
print(f"Tokenizer add_eos_token: {tokenizer.add_eos_token if hasattr(tokenizer, 'add_eos_token') else 'Not set'}")

