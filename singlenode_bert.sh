#!/bin/bash

# Single-node BERT Training Script
# Can be configured for standard 1F1B, interleaved 1F1B, or BitPipe

export CUDA_DEVICE_MAX_CONNECTIONS=1
export SKIP_CUDA_EXTENSIONS=1

# Single node configuration
GPUS_PER_NODE=4
NNODES=1
MASTER_ADDR="172.17.0.2"
MASTER_PORT=6000
NODE_RANK=0

# Network interface (not critical for single node)
export NCCL_SOCKET_IFNAME=lo

# Optional debugging - uncomment if needed
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

# Paths
CHECKPOINT_PATH=/tmp/bert_pipeline_test
DATA_PATH=/tmp/dummy_data

# Create checkpoint directory
mkdir -p $CHECKPOINT_PATH

# Calculate total world size
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

echo "Running Single-node BERT Pipeline Parallelism..."
echo "=================================="
echo "Total GPUs: $WORLD_SIZE ($NNODES node x $GPUS_PER_NODE GPUs/node)"
echo "Pipeline stages: $GPUS_PER_NODE"
echo "Node rank: $NODE_RANK"
echo "Master addr: $MASTER_ADDR"

# Pipeline configuration options (uncomment one):
# 1. Standard 1F1B (default)
#PIPELINE_ARGS="--pipeline-model-parallel-size $GPUS_PER_NODE"

# 2. Interleaved 1F1B (uncomment to use)
# PIPELINE_ARGS="--pipeline-model-parallel-size $GPUS_PER_NODE --virtual-pipeline-model-parallel-size 2"

# 3. BitPipe (uncomment to use)
# PIPELINE_ARGS="--pipeline-model-parallel-size $GPUS_PER_NODE --virtual-pipeline-model-parallel-size 4 --enable-bitpipe-schedule"

# 4. BitPipe Asymmetric (uncomment and configure)
# PIPELINE_ARGS="--pipeline-model-parallel-size $GPUS_PER_NODE --virtual-pipeline-model-parallel-size 4 --enable-bitpipe-schedule --enable-bitpipe-asymmetric --bitpipe-asymmetric-config /path/to/config.json"

echo "Pipeline configuration: $PIPELINE_ARGS"
echo "=================================="

# Model configuration (BERT-base like)
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=32  # Must be divisible by pipeline_parallel_size
NUM_MICROBATCHES=$((GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE))

echo "Micro batch size: $MICRO_BATCH_SIZE"
echo "Global batch size: $GLOBAL_BATCH_SIZE"
echo "Estimated microbatches: $NUM_MICROBATCHES"

# Launch distributed training
torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    asymmetric_bitpipe/scripts/examples/bert_dummy.py \
    --enable-bitpipe-schedule \
    --enable-profiling \
    --profile-train-iters  3\
    --pipeline-model-parallel-size 4 \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters 3 \
    --eval-iters 1 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --hidden-size 1024 \
    --num-layers 24 \
    --num-attention-heads 16 \
    --vocab-size 30522 \
    --lr 0.0001 \
    --lr-decay-style linear \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --log-interval 10 \
    --save-interval 100 \
    --save $CHECKPOINT_PATH \
    --no-load-optim \
    --no-load-rng \
    --fp16 \
    --data-path $DATA_PATH \
    --split 100,0,0 \
    --tokenizer-type NullTokenizer \
    --attention-dropout 0.1 \
    --hidden-dropout 0.1 \
    --dataloader-type single \
    --no-async-tensor-model-parallel-allreduce \
    --reset-position-ids \
    --reset-attention-mask \
    --eod-mask-loss

echo "BERT pipeline training completed!"

# Optional: Add profiling flags if you want to use your custom profiling
# --enable-profiling \
# --profile-train-iters 5 \