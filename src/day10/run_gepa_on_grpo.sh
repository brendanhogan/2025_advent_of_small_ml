#!/bin/bash
# Day 10 Experiment 1: GRPO → GEPA
# Run GEPA prompt optimization on an already-finetuned GRPO model
# Question: Can prompt optimization squeeze more performance out of a weight-updated model?

# NOTE: Start vLLM server first with the GRPO checkpoint:
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 uv run vllm serve grpo_qwen7b_run/checkpoint_step_400 --port 8001

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3 uv run python gepa_main.py \
    --model_name grpo_qwen7b_run/checkpoint_step_400 \
    --use_vllm \
    --vllm_host localhost \
    --vllm_port 8001 \
    --optimizer_model gpt-4.1 \
    --output_dir gepa_on_grpo_run \
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

