#!/bin/bash
# Run GEPA prompt optimization for MATH dataset
# Uses vLLM for fast generation
# Task model: Qwen2.5-7B-Instruct
# Optimizer model: GPT-4.1 (OpenAI)

# NOTE: Start vLLM server first with standard vLLM:
#   vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001
# Or if you already have a vLLM server running, just use that port

uv run python gepa_main.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --use_vllm \
    --vllm_host localhost \
    --vllm_port 8001 \
    --optimizer_model gpt-4.1 \
    --output_dir gepa_qwen7b_run \
    --num_iters 1000 \
    --minibatch_size 4 \
    --eval_every 50 \
    --num_completions_eval 20 \
    --pass_at_k 1 \
    --temperature 0.9 \
    --max_completion_length 512 \
    --candidate_selection pareto \
    --train_size 12000 \
    --eval_size 20 \
    --seed 7111994

