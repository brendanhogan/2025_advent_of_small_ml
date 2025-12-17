#!/bin/bash

# Launch vLLM server with 8 GPUs (tensor parallel)
# The server handles batching internally - just send more concurrent requests

# MAX THROUGHPUT CONFIG FOR 8x H100
# Note: TP=8 doesn't work with Qwen vocab size, using TP=4 + PP=2 to use all 8 GPUs
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --max-num-seqs 2048 \
    --max-num-batched-tokens 65536 \
    --enable-chunked-prefill \
    --disable-log-requests \
    --port 8000

# Then run simulation with high concurrency:
# uv run python batch_simulate.py \
#     --content "Your tweet or blog post here" \
#     --output run_001 \
#     --max-concurrent 256 \
#     --num-personas 10000

# For full 1M personas:
# uv run python batch_simulate.py \
#     --content "Your content" \
#     --output run_001 \
#     --max-concurrent 512 \
#     --resume



# Run client with high concurrency to saturate the server
uv run python batch_simulate.py \
    --content "NYC is the greatest city on earth - everyone wants to move here." \
    --output run_002 \
    --max-concurrent 1024 \
    --batch-size 10000 \
    --resume