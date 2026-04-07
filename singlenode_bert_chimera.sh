#!/bin/bash

# Single-node BERT Chimera 2-VR Pipeline Parallelism (4 GPUs)

export CUDA_DEVICE_MAX_CONNECTIONS=1
export SKIP_CUDA_EXTENSIONS=1
#export CHIMERA_DEBUG=1

# Single node configuration
GPUS_PER_NODE=4
NNODES=1
MASTER_ADDR="172.17.0.2"
MASTER_PORT=6000
NODE_RANK=0

export NCCL_SOCKET_IFNAME=eth0

# Optional debugging - uncomment if needed
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

# Paths
CHECKPOINT_PATH=/tmp/bert_pipeline_test
DATA_PATH=/tmp/dummy_data

mkdir -p $CHECKPOINT_PATH

WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

echo "Running Single-node BERT Chimera 2-VR Pipeline Parallelism..."
echo "=================================="
echo "Total GPUs: $WORLD_SIZE ($NNODES node x $GPUS_PER_NODE GPUs/node)"
echo "Pipeline stages: $GPUS_PER_NODE"
echo "Node rank: $NODE_RANK"
echo "Master addr: $MASTER_ADDR"
echo "Schedule: Chimera 2-VR"
echo "=================================="

MICRO_BATCH_SIZE=16
GLOBAL_BATCH_SIZE=64   # = 4 microbatches (divisible by pipeline_parallel_size=4)
NUM_MICROBATCHES=$((GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE))

echo "Micro batch size: $MICRO_BATCH_SIZE"
echo "Global batch size: $GLOBAL_BATCH_SIZE"
echo "Estimated microbatches: $NUM_MICROBATCHES"

torchrun \
    --nproc-per-node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node-rank $NODE_RANK \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    bert_dummy.py \
    --enable-chimera-schedule \
    --enable-profiling \
    --profile-train-iters 3 \
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
    --log-interval 5 \
    --no-load-optim \
    --no-load-rng \
    --fp16 \
    --data-path $DATA_PATH \
    --split 100,0,0 \
    --tokenizer-type NullTokenizer \
    --attention-dropout 0.1 \
    --hidden-dropout 0.1 \
    --dataloader-type single \
    --no-async-tensor-model-parallel-allreduce
