#!/bin/bash

# Test script for model building - verify layer distribution
# Tests ONLY model building, no training or scheduler

export CUDA_DEVICE_MAX_CONNECTIONS=1
export SKIP_CUDA_EXTENSIONS=1

GPUS_PER_NODE=2
NNODES=2
MASTER_ADDR="10.0.2.18"
MASTER_PORT=29500
NODE_RANK=$1

SCHEDULER=${2:-bitpipe}  # Default to bitpipe, can pass 'chimera'

echo "=========================================="
echo "Model Building Test"
echo "=========================================="
echo "Scheduler: $SCHEDULER"
echo "Node rank: $NODE_RANK"
echo "Master addr: $MASTER_ADDR"
echo "=========================================="
echo ""

if [ "$SCHEDULER" == "bitpipe" ]; then
    SCHEDULE_FLAG="--enable-bitpipe-schedule"
    echo "Testing BitPipe 4-VR model building..."
elif [ "$SCHEDULER" == "chimera" ]; then
    SCHEDULE_FLAG="--enable-chimera-schedule"
    echo "Testing Chimera 2-VR model building..."
else
    echo "Unknown scheduler: $SCHEDULER"
    echo "Usage: $0 <node_rank> [bitpipe|chimera]"
    exit 1
fi

echo ""

torchrun \
    --nproc-per-node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node-rank $NODE_RANK \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    test_model_building.py \
    $SCHEDULE_FLAG \
    --pipeline-model-parallel-size 4 \
    --micro-batch-size 2 \
    --global-batch-size 8 \
    --num-layers 48 \
    --hidden-size 512 \
    --num-attention-heads 8 \
    --seq-length 256 \
    --max-position-embeddings 512 \
    --vocab-size 1600 \
    --tokenizer-type NullTokenizer \
    --no-load-optim \
    --no-load-rng
