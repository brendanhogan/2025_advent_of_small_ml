#!/bin/bash

# ============================================
# Big Five Personality Evaluation Script
# ============================================
# 
# Three backends are supported:
# 1. Local HuggingFace model (default)
# 2. vLLM server (--use_vllm) - start server first with vllm_server.py
# 3. Replicate API (--use_replicate) - requires REPLICATE_API_TOKEN

# ============================================
# Local HuggingFace models
# ============================================
# uv run python eval.py --model_name "Qwen/Qwen2.5-7B-Instruct" --output_dir "eval_qwen_2.5_7b_instruct"
# uv run python eval.py --model_name "Qwen/Qwen2.5-1.5B-Instruct" --output_dir "eval_qwen_2.5_1.5b_instruct"

# ============================================
# vLLM Server (faster than local)
# ============================================
# First start the vLLM server in another terminal:
#   uv run python vllm_server.py --model "Qwen/Qwen2.5-7B-Instruct" --port 8000
#
# Then run eval with vLLM:
# uv run python eval.py \
#     --model_name "Qwen/Qwen2.5-7B-Instruct" \
#     --use_vllm \
#     --vllm_port 8000 \
#     --num_samples 5 \
#     --output_dir "eval_qwen_vllm"

# ============================================
# Replicate API models (parallel execution)
# ============================================
# Set REPLICATE_API_TOKEN environment variable first!

# Claude
uv run python eval.py \
    --model_name "anthropic/claude-4.5-sonnet" \
    --use_replicate \
    --max_parallel 10 \
    --num_samples 5 \
    --output_dir "eval_claude_4.5_sonnet"

# Gemini
uv run python eval.py \
    --model_name "google/gemini-2.5-flash" \
    --use_replicate \
    --max_parallel 10 \
    --num_samples 5 \
    --output_dir "eval_gemini_2.5_flash"

# GPT-5
uv run python eval.py \
    --model_name "openai/gpt-5" \
    --use_replicate \
    --max_parallel 10 \
    --num_samples 5 \
    --output_dir "eval_gpt5"

# DeepSeek
uv run python eval.py \
    --model_name "deepseek-ai/deepseek-v3.1" \
    --use_replicate \
    --max_parallel 10 \
    --num_samples 5 \
    --output_dir "eval_deepseek_v3.1"

# Grok
uv run python eval.py \
    --model_name "xai/grok-4" \
    --use_replicate \
    --max_parallel 10 \
    --num_samples 5 \
    --output_dir "eval_grok_4"

echo "All evaluations complete!"
