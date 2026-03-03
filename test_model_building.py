#!/usr/bin/env python
"""
Minimal test script to verify model chunk building for BitPipe vs Chimera.

This script:
1. Builds model chunks using the same loop as get_model()
2. Shows which layers each device/VR gets
3. Compares BitPipe vs Chimera layer assignments
4. Does NOT run training or scheduler

Usage:
    # Test BitPipe
    torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=10.0.2.18 --master_port=29500 \
        test_model_building.py --enable-bitpipe-schedule

    # Test Chimera
    torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=10.0.2.18 --master_port=29500 \
        test_model_building.py --enable-chimera-schedule
"""

import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from megatron import get_args, print_rank_0
from megatron.core import mpu, parallel_state
from megatron.core.enums import ModelType
from megatron.initialize import initialize_megatron
from megatron.model.gpt_model import GPTModel
from megatron.arguments import core_transformer_config_from_args


def model_provider(pre_process=True, post_process=True):
    """Build a single model chunk."""
    args = get_args()
    args.model_type = ModelType.encoder_or_decoder

    config = core_transformer_config_from_args(args)

    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )

    return model


def test_model_building():
    """Test the model building loop."""
    args = get_args()
    pipeline_rank = mpu.get_pipeline_model_parallel_rank()
    pipeline_size = mpu.get_pipeline_model_parallel_world_size()

    # ========== BUILD MODEL CHUNKS ==========
    print_rank_0("\n" + "="*80)
    print_rank_0("MODEL BUILDING TEST")
    print_rank_0("="*80)

    print_rank_0(f"\nScheduler: {'BitPipe 4-VR' if args.enable_bitpipe_schedule else 'Chimera 2-VR' if hasattr(args, 'enable_chimera_schedule') and args.enable_chimera_schedule else 'Unknown'}")
    print_rank_0(f"Total pipeline stages: {pipeline_size}")
    print_rank_0(f"Virtual pipeline size: {args.virtual_pipeline_model_parallel_size}")
    print_rank_0(f"Base num_layers: {args.num_layers}")
    print_rank_0(f"Hidden size: {args.hidden_size}")

    # This is the exact loop from get_model() in megatron/training.py:236
    model_chunks = []

    for i in range(args.virtual_pipeline_model_parallel_size):
        mpu.set_virtual_pipeline_model_parallel_rank(i)

        # Set pre_process and post_process only after virtual rank is set
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()

        print(f"\n[Building Chunk {i}]")
        print(f"  Pipeline rank: {pipeline_rank}, VR rank: {i}")
        print(f"  Pre-process: {pre_process}, Post-process: {post_process}")

        this_model = model_provider(
            pre_process=pre_process,
            post_process=post_process
        )

        this_model.model_type = ModelType.encoder_or_decoder
        model_chunks.append(this_model)

        print(f"  ✓ Chunk {i} built successfully")

    # ========== ANALYZE LAYER DISTRIBUTION ==========
    print_rank_0("\n" + "="*80)
    print_rank_0("LAYER DISTRIBUTION ANALYSIS (Device " + str(pipeline_rank) + ")")
    print_rank_0("="*80)

    torch.distributed.barrier()

    for vr_idx in range(args.virtual_pipeline_model_parallel_size):
        model = model_chunks[vr_idx]
        mpu.set_virtual_pipeline_model_parallel_rank(vr_idx)

        # Get the transformer (handles both pre_process/post_process cases)
        if hasattr(model, 'language_model'):
            transformer = model.language_model.transformer
        else:
            transformer = model.transformer

        num_layers = transformer.num_layers

        print(f"\nChunk {vr_idx} (VR{vr_idx}):")
        print(f"  Total layers: {num_layers}")

        if num_layers > 0 and hasattr(transformer, 'layers'):
            layer_list = list(range(num_layers))
            print(f"  Layer indices: {layer_list[:5]}{'...' if num_layers > 5 else ''}")

    torch.distributed.barrier()

    # ========== GLOBAL SUMMARY ==========
    if mpu.get_data_parallel_rank() == 0:
        print_rank_0("\n" + "="*80)
        print_rank_0("GLOBAL SUMMARY")
        print_rank_0("="*80)

        if args.enable_bitpipe_schedule:
            print_rank_0(f"\nBitPipe 4-VR Configuration:")
            print_rank_0(f"  Virtual ranks: 4")
            print_rank_0(f"  Total layers (doubled): {args.num_layers * 2}")
            print_rank_0(f"  Layers per device: {(args.num_layers * 2) // pipeline_size}")
            print_rank_0(f"  Layers per VR: {(args.num_layers * 2) // (pipeline_size * 4)}")
            print_rank_0(f"\n  Expected pattern: V-shaped")
            print_rank_0(f"  Device 0: VR0[0-{(args.num_layers * 2) // (pipeline_size * 4) - 1}], VR1[?-?], VR2[?-?], VR3[?-?]")

        elif hasattr(args, 'enable_chimera_schedule') and args.enable_chimera_schedule:
            print_rank_0(f"\nChimera 2-VR Configuration:")
            print_rank_0(f"  Virtual ranks: 2")
            print_rank_0(f"  Total layers (doubled): {args.num_layers * 2}")
            print_rank_0(f"  Layers per device: {(args.num_layers * 2) // pipeline_size}")
            print_rank_0(f"  Layers per VR: {(args.num_layers * 2) // pipeline_size // 2}")
            print_rank_0(f"\n  Expected pattern: Sequential + Device swap")
            layers_per_vr = (args.num_layers * 2) // pipeline_size // 2
            print_rank_0(f"  Device 0 VR0: layers [0-{layers_per_vr-1}]")
            print_rank_0(f"  Device 0 VR1: layers [{(pipeline_size-1)*layers_per_vr}-{(pipeline_size)*layers_per_vr-1}]")

    torch.distributed.barrier()

    print_rank_0("\n" + "="*80)
    print_rank_0("✓ MODEL BUILDING TEST COMPLETE")
    print_rank_0("="*80 + "\n")


def extra_args_provider(parser):
    """Add test-specific arguments."""
    group = parser.add_argument_group(title='model_test')
    group.add_argument('--test-mode', action='store_true', help='Run in test mode (no training)')
    return parser


def main():
    """Main entry point."""
    initialize_megatron(
        extra_args_provider=extra_args_provider,
        args_defaults={
            'log_interval': 100,
            'exit_interval': 500,
            'eval_interval': 1000,
            'eval_iters': 10,
        }
    )

    try:
        test_model_building()
    except Exception as e:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            print(f"\n❌ Error during model building test:")
            print(f"  {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
