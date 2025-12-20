"""
Step 2: Tokenize the downloaded SYNTH dataset.
Filters for English, tokenizes in parallel, saves ready-to-train format.
"""

from datasets import load_from_disk
from transformers import AutoTokenizer
import os

# Config
MODEL_NAME = "PleIAs/Monad"
MAX_LENGTH = 1280
INPUT_DIR = "raw_synth"
OUTPUT_DIR = "tokenized_synth"
NUM_PROC = 32  # Parallel tokenization


def main():
    print("=" * 60)
    print("Tokenizing SYNTH dataset")
    print("=" * 60)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    pad_token_id = tokenizer.pad_token_id
    print(f"  Vocab size: {len(tokenizer)}")
    
    # Load raw dataset
    print(f"\nLoading raw dataset from {INPUT_DIR}/...")
    dataset = load_from_disk(INPUT_DIR)
    print(f"  Total samples: {len(dataset):,}")
    
    # Filter for English with required fields
    print("\nFiltering for English samples...")
    def filter_english(x):
        return (
            x.get("language") == "en"
            and x.get("query")
            and x.get("synthetic_reasoning")
            and x.get("synthetic_answer")
        )
    
    dataset = dataset.filter(filter_english, num_proc=NUM_PROC)
    print(f"  English samples: {len(dataset):,}")
    
    # Format to ChatML and tokenize
    print(f"\nTokenizing (max_length={MAX_LENGTH}, {NUM_PROC} processes)...")
    
    def tokenize(examples):
        # Format messages
        texts = []
        for i in range(len(examples["query"])):
            query = examples["query"][i]
            reasoning = examples["synthetic_reasoning"][i]
            answer = examples["synthetic_answer"][i]
            
            text = (
                f"<|im_start|>user\n{query}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{answer}<|im_end|>"
            )
            texts.append(text)
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        
        # Create labels with padding masked
        labels = []
        for ids in tokenized["input_ids"]:
            label = [-100 if tid == pad_token_id else tid for tid in ids]
            labels.append(label)
        tokenized["labels"] = labels
        
        return tokenized
    
    # Remove old columns, keep only tokenized data
    columns_to_remove = dataset.column_names
    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=1000,
        num_proc=NUM_PROC,
        remove_columns=columns_to_remove,
        desc="Tokenizing",
    )
    
    print(f"  Tokenized samples: {len(dataset):,}")
    
    # Shuffle and split
    print("\nShuffling and splitting train/eval...")
    dataset = dataset.shuffle(seed=3407)
    split = dataset.train_test_split(test_size=1000, seed=3407)
    train_ds = split["train"]
    eval_ds = split["test"]
    
    print(f"  Train: {len(train_ds):,}")
    print(f"  Eval: {len(eval_ds):,}")
    
    # Save
    print(f"\nSaving to {OUTPUT_DIR}/...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_ds.save_to_disk(os.path.join(OUTPUT_DIR, "train"))
    eval_ds.save_to_disk(os.path.join(OUTPUT_DIR, "eval"))
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Train samples: {len(train_ds):,}")
    print(f"  Eval samples: {len(eval_ds):,}")
    print(f"  Saved to: {OUTPUT_DIR}/")
    print("\nNext: accelerate launch 3_train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

