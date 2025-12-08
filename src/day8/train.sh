#!/bin/bash

# Training personality models with GRPO
# 
# Option 1: Local generation (slower, no separate server needed)
# Option 2: vLLM generation (faster, requires running vllm_server.py first)

MODEL="Qwen/Qwen2.5-7B-Instruct"

# To use vLLM, first start the server in another terminal:
#   uv run python vllm_server.py --model "Qwen/Qwen2.5-7B-Instruct" --port 8000
# Then add --use_vllm to the training commands below.

# Local training (without vLLM):
uv run python train.py --model_name "$MODEL" --target_archetype jerk --output_dir train_jerk
uv run python train.py --model_name "$MODEL" --target_archetype neurotic --output_dir train_neurotic
uv run python train.py --model_name "$MODEL" --target_archetype creative_chaos --output_dir train_creative_chaos
uv run python train.py --model_name "$MODEL" --target_archetype cold_logician --output_dir train_cold_logician



CUDA_VISIBLE_DEVICES=4 uv run python vllm_server.py --model "Qwen/Qwen2.5-7B-Instruct" --port 8000

CUDA_VISIBLE_DEVICES=5,6 uv run python train.py --model_name "Qwen/Qwen2.5-7B-Instruct" --target_archetype jerk --output_dir train_jerk --use_vllm --vllm_port 8000

CUDA_VISIBLE_DEVICES=5,6 uv run python train.py --model_name "Qwen/Qwen2.5-7B-Instruct" --target_archetype jerk --output_dir train_jerk_high_temp --use_vllm --vllm_port 8000


# With vLLM (faster generation):
# uv run python train.py --model_name "$MODEL" --target_archetype jerk --output_dir train_jerk --use_vllm --vllm_port 8000
# uv run python train.py --model_name "$MODEL" --target_archetype neurotic --output_dir train_neurotic --use_vllm --vllm_port 8000
