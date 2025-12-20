"""
Step 3: Train Monad on pre-tokenized SYNTH dataset.

Requires: tokenized_synth/ directory from 2_tokenize.py

Usage:
    accelerate launch 3_train.py
"""

import os
import csv
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any

import torch
import wandb
from accelerate import Accelerator
from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    default_data_collator,
)


# =============================================================================
# Logging
# =============================================================================

@dataclass
class TrainingMetrics:
    step: int
    epoch: float
    loss: float
    learning_rate: float
    tokens_seen: int = 0
    grad_norm: float = 0.0
    timestamp: str = ""


@dataclass
class EvalMetrics:
    step: int
    eval_loss: float
    perplexity: float
    timestamp: str = ""


class TrainingLogger:
    def __init__(self, output_dir: str):
        self.logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        self.training_history: List[TrainingMetrics] = []
        self.eval_history: List[EvalMetrics] = []
        self.training_csv = os.path.join(self.logs_dir, "training_metrics.csv")
        
        with open(self.training_csv, "w", newline="") as f:
            csv.writer(f).writerow(["step", "epoch", "loss", "lr", "tokens", "grad_norm", "timestamp"])
    
    def log_training(self, m: TrainingMetrics):
        m.timestamp = datetime.now().isoformat()
        self.training_history.append(m)
        with open(self.training_csv, "a", newline="") as f:
            csv.writer(f).writerow([m.step, m.epoch, m.loss, m.learning_rate, m.tokens_seen, m.grad_norm, m.timestamp])
    
    def log_eval(self, m: EvalMetrics):
        m.timestamp = datetime.now().isoformat()
        self.eval_history.append(m)
    
    def save_summary(self):
        summary = {
            "total_steps": len(self.training_history),
            "final_loss": self.training_history[-1].loss if self.training_history else None,
            "final_eval_loss": self.eval_history[-1].eval_loss if self.eval_history else None,
        }
        with open(os.path.join(self.logs_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nFinal loss: {summary['final_loss']:.4f}" if summary['final_loss'] else "")


class LoggingCallback(TrainerCallback):
    def __init__(self, logger: TrainingLogger):
        self.logger = logger
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            if "loss" in logs:
                self.logger.log_training(TrainingMetrics(
                    step=state.global_step,
                    epoch=state.epoch or 0,
                    loss=logs.get("loss", 0),
                    learning_rate=logs.get("learning_rate", 0),
                    tokens_seen=logs.get("num_input_tokens_seen", 0),
                    grad_norm=logs.get("grad_norm", 0),
                ))
            if "eval_loss" in logs:
                import math
                self.logger.log_eval(EvalMetrics(
                    step=state.global_step,
                    eval_loss=logs["eval_loss"],
                    perplexity=math.exp(logs["eval_loss"]) if logs["eval_loss"] < 100 else float("inf"),
                ))


# =============================================================================
# Training
# =============================================================================

def set_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    # Model
    model_name: str = "PleIAs/Monad",
    checkpoint: Optional[str] = None,
    # Data
    data_dir: str = "tokenized_synth",
    max_length: int = 1280,
    # Training
    batch_size: int = 64,  # Max that fits on H100 with this model/seq_len
    accumulation: int = 4,  # 4 accum steps per optimizer update
    lr: float = 4e-3,
    warmup_ratio: float = 0.05,  # 5% of training for warmup
    decay_ratio: float = 0.1,    # 10% of training for decay
    # Checkpointing
    num_checkpoints: int = 20,
    eval_ratio: float = 0.05,    # Eval every 5% of training
    logging_steps: int = 10,
    # Other
    output_dir: str = "results",
    use_wandb: bool = True,
    seed: int = 3407,
    target_tokens: int = 3_000_000_000,  # 3B tokens → ~1 hour on 4x H100
):
    print("=" * 60)
    print("Monad Training")
    print("=" * 60)
    
    set_seeds(seed)
    accelerator = Accelerator()
    
    # Check data exists
    if not os.path.exists(data_dir):
        print(f"\nERROR: {data_dir}/ not found!")
        print("Run these first:")
        print("  python 1_download.py")
        print("  python 2_tokenize.py")
        return
    
    # Setup output
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_dir, f"monad_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    logger = TrainingLogger(run_dir)
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {data_dir}/...")
    tokenizer = AutoTokenizer.from_pretrained(data_dir)
    
    # Load data
    print(f"Loading pre-tokenized data...")
    train_ds = load_from_disk(os.path.join(data_dir, "train"))
    eval_ds = load_from_disk(os.path.join(data_dir, "eval"))
    print(f"  Train: {len(train_ds):,}")
    print(f"  Eval: {len(eval_ds):,}")
    
    # Model
    model_kwargs = {"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"}
    
    if accelerator.is_main_process:
        if checkpoint:
            print(f"\nLoading checkpoint: {checkpoint}")
            model = AutoModelForCausalLM.from_pretrained(checkpoint, **model_kwargs)
        else:
            print(f"\nInitializing random model: {model_name}")
            cfg = AutoConfig.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_config(cfg, **model_kwargs)
        
        model.resize_token_embeddings(len(tokenizer))
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Save init for other processes
        init_path = os.path.join(run_dir, "init")
        model.save_pretrained(init_path)
        
        if use_wandb:
            wandb.init(project="monad_training", name=f"monad_{timestamp}")
    
    accelerator.wait_for_everyone()
    
    # All processes load
    init_path = os.path.join(run_dir, "init")
    model = AutoModelForCausalLM.from_pretrained(init_path, **model_kwargs)
    # Note: Use torch_compile=True in TrainingArguments instead of torch.compile()
    # so that the Trainer saves unwrapped weights compatible with vLLM
    
    accelerator.wait_for_everyone()
    
    # Compute steps
    num_gpus = accelerator.num_processes
    effective_batch = batch_size * accumulation * num_gpus
    tokens_per_step = effective_batch * max_length
    max_steps = target_tokens // tokens_per_step
    save_steps = max(1, max_steps // (num_checkpoints - 1))
    
    # Scale warmup, decay, eval to training length
    warmup_steps = int(max_steps * warmup_ratio)
    num_decay_steps = int(max_steps * decay_ratio)
    eval_steps = max(1, int(max_steps * eval_ratio))
    
    print(f"\nConfig:")
    print(f"  GPUs: {num_gpus}")
    print(f"  Effective batch: {effective_batch}")
    print(f"  Max steps: {max_steps:,}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Decay steps: {num_decay_steps} (last {decay_ratio*100:.0f}%)")
    print(f"  Eval every: {eval_steps} steps")
    print(f"  Save every: {save_steps} steps")
    
    # Training args
    args = TrainingArguments(
        output_dir=run_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accumulation,
        learning_rate=lr,
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs={"num_decay_steps": num_decay_steps},
        warmup_steps=warmup_steps,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=None,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        torch_compile=True,  # Proper torch.compile - Trainer saves unwrapped weights
        seed=seed,
        report_to="wandb" if use_wandb else None,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        callbacks=[LoggingCallback(logger)],
    )
    
    print("\nStarting training...")
    trainer.train()
    
    # Save final
    trainer.save_model(run_dir)
    tokenizer.save_pretrained(run_dir)
    
    if accelerator.is_main_process:
        logger.save_summary()
        if wandb.run:
            wandb.finish()
    
    print(f"\nDone! Model saved to: {run_dir}")


if __name__ == "__main__":
    train()

