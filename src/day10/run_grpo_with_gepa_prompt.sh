#!/bin/bash
# Day 10 Experiment 2: GEPA → GRPO
# Run GRPO training with the best GEPA-evolved prompt (instead of basic seed prompt)
# Question: Does starting with a better prompt make GRPO more effective?

# NOTE: Start vLLM server first:
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=4 uv run vllm_server.py --model Qwen/Qwen2.5-7B-Instruct --port 8000 --dtype bfloat16

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=5,6,7 uv run python main.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --system_prompt_file best_gepa_prompt.txt \
    --use_vllm \
    --vllm_host localhost \
    --vllm_port 8000 \
    --output_dir grpo_with_gepa_prompt_run \
    --num_train_iters 1000 \
    --eval_every 50 \
    --save_every 200 \
    --num_completions_eval 20 \
    --pass_at_k 1 \
    --temperature 0.9 \
    --max_completion_length 512 \
    --num_chains 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --train-size 12000 \
    --eval-size 20 \
    --seed 7111994

