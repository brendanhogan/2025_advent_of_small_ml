#!/bin/bash
# Run GRPO training for MATH dataset
# Uses local vllm_server.py for fast generation with weight syncing
# Model: Qwen2.5-7B-Instruct

# NOTE: Start vLLM server first in a separate terminal:
#   NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 uv run vllm_server.py --model Qwen/Qwen2.5-7B-Instruct --port 8000 --dtype bfloat16
#
# Then run this script in another terminal (on different GPUs):
#   NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3 ./run_grpo.sh

  NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 uv run vllm_server.py --model Qwen/Qwen2.5-7B-Instruct --port 8000 --dtype bfloat16


NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=5,6,7 uv run python main.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --use_vllm \
    --vllm_host localhost \
    --vllm_port 8000 \
    --output_dir grpo_qwen7b_run \
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

