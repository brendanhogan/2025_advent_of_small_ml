#!/bin/bash

# Day 19: GRPO training with persona judges (vLLM)
#
# You will run TWO things:
#   1) vLLM server (judge model) on port 8000
#   2) training script (policy model runs locally; judges are queried via vLLM)

###############################################################################
# 1) Start the vLLM judge server
###############################################################################

# Single GPU:
# vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# Multi-GPU (8x H100 - max throughput)
# Note: TP=8 doesn't work with Qwen vocab size, so use TP=4 + PP=2 to use all 8 GPUs
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --max-num-seqs 2048 \
    --max-num-batched-tokens 65536 \
    --enable-chunked-prefill \
    --disable-log-requests \
    --port 8000

###############################################################################
# 2) Run training (in another terminal)
###############################################################################
#
# Edit the config to select your target demographic slice and subject:
#   - ./config_example.json
#
# Then run:
#
# uv run python train_grpo_persona_judge.py --config config_example.json
