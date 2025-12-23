"""
Step 3: Train custom transformer with looped reasoning layers.

Architecture: Simple GPT-style transformer where last 10% of layers loop.
  Embed → Layers 0-N (once) → [Layers N-end × num_loops w/ LayerNorm] → LM Head

Requires: tokenized_synth/ directory from 2_tokenize.py

Usage:
    accelerate launch 3_train.py
"""

import math
import os
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)
from transformers.modeling_outputs import CausalLMOutputWithPast


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
                self.logger.log_eval(EvalMetrics(
                    step=state.global_step,
                    eval_loss=logs["eval_loss"],
                    perplexity=math.exp(logs["eval_loss"]) if logs["eval_loss"] < 100 else float("inf"),
                ))


# =============================================================================
# Simple Transformer (GPT-style)
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos/sin for max_seq_len
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())
    
    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings. x: [batch, seq, heads, head_dim]"""
    # Split into pairs for rotation
    x1, x2 = x[..., ::2], x[..., 1::2]
    # Rotate
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, dim/2]
    sin = sin.unsqueeze(0).unsqueeze(2)
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)
    return rotated


class Attention(nn.Module):
    """Multi-head attention with rotary embeddings."""
    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        
        # Apply rotary embeddings
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        # Transpose for attention: [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention with causal mask (uses Flash Attention when available)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """SwiGLU feedforward network."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""
    def __init__(self, dim: int, num_heads: int, head_dim: int, ff_dim: int):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, head_dim)
        self.ff_norm = RMSNorm(dim)
        self.ff = FeedForward(dim, ff_dim)
    
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ff(self.ff_norm(x))
        return x


@dataclass
class TransformerConfig:
    """Configuration for the transformer model."""
    vocab_size: int = 32000
    dim: int = 512           # Smaller for ~50M params
    num_layers: int = 8
    num_heads: int = 8
    head_dim: int = 64
    ff_mult: float = 2.67    # SwiGLU uses ~2.67x for similar param count to 4x FFN
    max_seq_len: int = 2048
    # Looping config
    loop_fraction: float = 0.0  # 0 = no looping, 0.1 = loop last 10%
    num_loops: int = 1
    
    def to_dict(self):
        """For HF Trainer compatibility."""
        return self.__dict__.copy()


class LoopedTransformer(nn.Module):
    """
    GPT-style transformer with optional looped reasoning layers.
    
    When loop_fraction > 0, the last loop_fraction of layers are repeated
    num_loops times with a LayerNorm between iterations.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.rotary = RotaryEmbedding(config.head_dim, config.max_seq_len)
        
        # Compute hidden dim for SwiGLU
        ff_dim = int(config.dim * config.ff_mult)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config.dim, config.num_heads, config.head_dim, ff_dim)
            for _ in range(config.num_layers)
        ])
        
        # Looping setup
        self.num_loop_layers = max(1, int(config.num_layers * config.loop_fraction)) if config.loop_fraction > 0 else 0
        self.loop_start = config.num_layers - self.num_loop_layers if self.num_loop_layers > 0 else config.num_layers
        self.num_loops = config.num_loops if self.num_loop_layers > 0 else 1
        
        # LayerNorm between loop iterations (only if looping)
        if self.num_loop_layers > 0 and self.num_loops > 1:
            self.loop_norm = RMSNorm(config.dim)
        else:
            self.loop_norm = None
        
        # Output
        self.final_norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.embed.weight
        
        # Initialize
        self.apply(self._init_weights)
        
        # Print architecture
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  LoopedTransformer: {n_params:,} parameters")
        if self.num_loop_layers > 0:
            print(f"    Layers {self.loop_start}-{config.num_layers-1} loop {self.num_loops}x")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        batch, seq_len = input_ids.shape
        
        # Embeddings + rotary
        x = self.embed(input_ids)
        cos, sin = self.rotary(seq_len)
        
        # Early layers (run once)
        for layer in self.layers[:self.loop_start]:
            x = layer(x, cos, sin)
        
        # Loop layers (run num_loops times)
        if self.num_loop_layers > 0:
            loop_layers = self.layers[self.loop_start:]
            for loop_i in range(self.num_loops):
                if loop_i > 0 and self.loop_norm is not None:
                    x = self.loop_norm(x)
                for layer in loop_layers:
                    x = layer(x, cos, sin)
        
        # Output
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        # Loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.size(-1)),
                labels[..., 1:].reshape(-1),
                ignore_index=-100,
            )
        
        return CausalLMOutputWithPast(loss=loss, logits=logits)
    
    def save_pretrained(self, path: str, **kwargs):
        """Save model and config."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Load model from checkpoint."""
        with open(os.path.join(path, "config.json")) as f:
            config = TransformerConfig(**json.load(f))
        model = cls(config)
        model.load_state_dict(torch.load(os.path.join(path, "model.pt"), weights_only=True))
        return model


# =============================================================================
# Training
# =============================================================================

def set_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    # Architecture (~50M params)
    dim: int = 512,
    num_layers: int = 8,
    num_heads: int = 8,
    head_dim: int = 64,
    # Looped architecture
    loop_fraction: float = 0.1,  # Loop last 10% of layers
    num_loops: int = 5,          # Number of times to loop
    # Data
    data_dir: str = "tokenized_synth",
    max_length: int = 1280,
    # Training
    batch_size: int = 64,
    accumulation: int = 4,
    lr: float = 4e-3,
    warmup_ratio: float = 0.05,
    decay_ratio: float = 0.1,
    # Checkpointing
    num_checkpoints: int = 20,
    eval_ratio: float = 0.05,
    logging_steps: int = 10,
    # Other
    output_dir: str = "results",
    use_wandb: bool = True,
    seed: int = 3407,
    target_tokens: int = 3_000_000_000,
):
    print("=" * 60)
    print("Training Looped Transformer")
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
    loop_str = f"loop{num_loops}x" if loop_fraction > 0 else "baseline"
    run_dir = os.path.join(output_dir, f"transformer_{loop_str}_{timestamp}")
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
    
    # Create model
    if accelerator.is_main_process:
        print(f"\nCreating model...")
        config = TransformerConfig(
            vocab_size=len(tokenizer),
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=max_length,
            loop_fraction=loop_fraction,
            num_loops=num_loops,
        )
        model = LoopedTransformer(config)
        model = model.to(torch.bfloat16)
        
        # Save init for other processes
        init_path = os.path.join(run_dir, "init")
        model.save_pretrained(init_path)
        
        if use_wandb:
            wandb.init(
                project="looped_transformer",
                name=f"{loop_str}_{timestamp}",
                config={
                    "dim": dim,
                    "num_layers": num_layers,
                    "loop_fraction": loop_fraction,
                    "num_loops": num_loops,
                    "target_tokens": target_tokens,
                    "lr": lr,
                },
            )
    
    accelerator.wait_for_everyone()
    
    # All processes load
    init_path = os.path.join(run_dir, "init")
    model = LoopedTransformer.from_pretrained(init_path)
    model = model.to(torch.bfloat16)
    
    accelerator.wait_for_everyone()
    
    # Compute steps
    num_gpus = accelerator.num_processes
    effective_batch = batch_size * accumulation * num_gpus
    tokens_per_step = effective_batch * max_length
    max_steps = target_tokens // tokens_per_step
    save_steps = max(1, max_steps // (num_checkpoints - 1))
    
    warmup_steps = int(max_steps * warmup_ratio)
    num_decay_steps = int(max_steps * decay_ratio)
    eval_steps = max(1, int(max_steps * eval_ratio))
    
    print(f"\nTraining Config:")
    print(f"  GPUs: {num_gpus}")
    print(f"  Effective batch: {effective_batch}")
    print(f"  Max steps: {max_steps:,}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Decay steps: {num_decay_steps}")
    print(f"  Eval every: {eval_steps} steps")
    print(f"  Save every: {save_steps} steps")
    if loop_fraction > 0:
        print(f"\nLooped Architecture:")
        print(f"  Loop fraction: {loop_fraction*100:.0f}% of layers")
        print(f"  Num loops: {num_loops}")
    
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
        save_safetensors=False,  # Disable safetensors due to weight tying
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Train looped transformer")
    # Architecture (~50M params with defaults)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=64)
    # Looped architecture
    parser.add_argument("--loop_fraction", type=float, default=0.1)
    parser.add_argument("--num_loops", type=int, default=5)
    # Data
    parser.add_argument("--data_dir", type=str, default="tokenized_synth")
    parser.add_argument("--max_length", type=int, default=1280)
    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--decay_ratio", type=float, default=0.1)
    # Checkpointing
    parser.add_argument("--num_checkpoints", type=int, default=20)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    # Other
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--target_tokens", type=int, default=3_000_000_000)
    
    args = parser.parse_args()
    
    train(
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        loop_fraction=args.loop_fraction,
        num_loops=args.num_loops,
        data_dir=args.data_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        accumulation=args.accumulation,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        decay_ratio=args.decay_ratio,
        num_checkpoints=args.num_checkpoints,
        eval_ratio=args.eval_ratio,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
        seed=args.seed,
        target_tokens=args.target_tokens,
    )
