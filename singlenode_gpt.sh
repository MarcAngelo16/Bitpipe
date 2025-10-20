#!/bin/bash

#Uses GPT layers
export CUDA_DEVICE_MAX_CONNECTIONS=1
export SKIP_CUDA_EXTENSIONS=1

# Single node configuration
GPUS_PER_NODE=4
NNODES=1
MASTER_ADDR="172.17.0.2"  # localhost for single node
MASTER_PORT=6000
NODE_RANK=0

# Network interface (not critical for single node)
export NCCL_SOCKET_IFNAME=lo

# Optional debugging - uncomment if needed
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

# Paths
CHECKPOINT_PATH=/tmp/bitpipe_8gpu_test
DATA_PATH=/tmp/dummy_data

# Create checkpoint directory
mkdir -p $CHECKPOINT_PATH

# Calculate total world size
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

echo "Running Single-node BitPipe Pipeline Parallelism..."
echo "=================================="
echo "Total GPUs: $WORLD_SIZE ($NNODES node x $GPUS_PER_NODE GPUs/node)"
echo "Pipeline stages: $WORLD_SIZE (BitPipe enabled)"
echo "Node rank: $NODE_RANK"
echo "Master addr: $MASTER_ADDR"
echo "BitPipe: ENABLED"
echo "=================================="

# BitPipe configuration for 8 GPUs
# For 8 GPUs with BitPipe:
# - 8 pipeline stages 
# - Model chunks (depends on BitPipe internal logic)
# - Global batch size must be divisible by pipeline_size
# - Microbatch count should be >= pipeline_size for efficiency

# Calculate microbatches - Increased for longer duration
MICRO_BATCH_SIZE=16    # Larger microbatches = more computation per batch
GLOBAL_BATCH_SIZE=64  # Must be divisible by pipeline_parallel_size for BitPipe
NUM_MICROBATCHES=$((GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE))  

echo "Micro batch size: $MICRO_BATCH_SIZE"
echo "Global batch size: $GLOBAL_BATCH_SIZE"
echo "Estimated microbatches: $NUM_MICROBATCHES"

# Launch distributed training with BitPipe
torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    asymmetric_bitpipe/scripts/examples/gpt_dummy.py \
    --enable-bitpipe-schedule \
    --enable-bitpipe-profiling \
    --bitpipe-profile-train-iters  3\
    --pipeline-model-parallel-size 4 \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters 3 \
    --eval-iters 1 \
    --seq-length 128 \
    --max-position-embeddings 128 \
    --hidden-size 400 \
    --num-layers 48 \
    --num-attention-heads 8 \
    --vocab-size 8000 \
    --lr 0.0001 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --log-interval 5 \
    --save-interval 25 \
    --save $CHECKPOINT_PATH \
    --no-load-optim \
    --no-load-rng \
    --fp16 \
    --data-path $DATA_PATH \
    --split 100,0,0 \
    --train-synthetic \
    --tokenizer-type NullTokenizer \
    --attention-dropout 0.1 \
    --hidden-dropout 0.1 \
    --dataloader-type single \
    --no-async-tensor-model-parallel-allreduce \
    --reset-position-ids \
    --reset-attention-mask \
    --eod-mask-loss 