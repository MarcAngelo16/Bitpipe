"""
Chimera: 2-VR Bidirectional Interleaved Pipeline Parallelism

This module implements a simplified bidirectional pipeline schedule with 2 virtual ranks (VR)
per device, providing a cleaner alternative to BitPipe 4-VR while maintaining bidirectional benefits.

Key Features:
- 2 VRs per device: VR0, VR1
- Sequential layer assignment (no V-shape)
- Bidirectional pipeline execution (1F1B pattern)
- NO microbatch doubling (8 MBs stay 8 MBs)
- Simpler scheduling logic than BitPipe 4-VR
- No same-device VR transitions

Comparison to BitPipe 4-VR:
- Fewer VRs (2 vs 4) = simpler code
- Sequential pattern vs V-shaped pattern
- NO doubling (N MBs, not 2N like BitPipe)
- Similar pipeline efficiency with reduced complexity

Scheduling Phases:
1. WARMUP: Forward-only passes to fill the pipeline
2. STEADY STATE: 1F1B interleaving with rank-dependent ordering
3. COOLDOWN: Backward-only passes to drain the pipeline
"""

import contextlib
import os
from typing import Iterator, List, Union

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import core, get_args, get_num_microbatches, print_rank_0
from megatron.core import parallel_state
from megatron.core.utils import get_model_config, get_model_type
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
)
from megatron.core.pipeline_parallel.schedules import (
    deallocate_output_tensor,
    forward_step,
    backward_step,
)
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.pipeline_parallel.bitpipe_profiler import (
    initialize_bitpipe_profiler, get_bitpipe_profiler
)


def print_all_ranks(message):
    """Print from all ranks for debugging Chimera schedule."""
    if os.environ.get('CHIMERA_DEBUG', '0') == '1':
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_rank = os.environ.get('RANK', 'unknown')
            print(f"[Chimera Rank {rank}/World {world_rank}] {message}", flush=True)
        else:
            print(f"[Chimera] {message}", flush=True)


def get_chimera_offset(pipeline_rank, vp_rank, num_layers_per_vr, num_devices):
    """
    Simplified offset calculation for Chimera 2-VR

    IMPORTANT: Layers are indexed starting from 1 (not 0)

    Pattern (4 devices, 16 layers example):
    Device 0: VR0[1,2,3,4]      VR1[13,14,15,16]
    Device 1: VR0[5,6,7,8]      VR1[9,10,11,12]
    Device 2: VR0[9,10,11,12]   VR1[5,6,7,8]
    Device 3: VR0[13,14,15,16]  VR1[1,2,3,4]

    Args:
        pipeline_rank: Current device rank (0 to num_devices-1)
        vp_rank: Virtual pipeline rank (0 or 1 for Chimera)
        num_layers_per_vr: Number of layers per VR (total_layers / num_devices)
        num_devices: Total number of pipeline devices

    Returns:
        (offset, num_layers): Starting layer index (1-based) and number of layers
    """
    if vp_rank == 0:
        # VR0: Sequential forward (device 0→1→...→N-1)
        # Device 0 starts at layer 1, device 1 at layer (1 + num_layers_per_vr), etc.
        offset = pipeline_rank * num_layers_per_vr + 1
    else:  # vp_rank == 1
        # VR1: Simple device swap (device i gets device N-1-i's VR0 layers)
        paired_device = num_devices - 1 - pipeline_rank
        offset = paired_device * num_layers_per_vr + 1

    return offset, num_layers_per_vr


def get_chimera_microbatch_groups(total_num_microbatches):
    """
    Group microbatches by VR for Chimera 2-VR

    Simple split: first half → VR0, second half → VR1

    Example (16 total microbatches):
    VR0: [0, 1, 2, 3, 4, 5, 6, 7]
    VR1: [8, 9, 10, 11, 12, 13, 14, 15]

    Args:
        total_num_microbatches: Total microbatches (doubled from user input)

    Returns:
        List of 2 lists containing microbatch IDs per VR
    """
    half = total_num_microbatches // 2
    microbatch_groups = [
        list(range(0, half)),                      # VR0
        list(range(half, total_num_microbatches))  # VR1
    ]
    return microbatch_groups


def build_chimera_forward_schedule(total_num_microbatches, pipeline_parallel_size, pipeline_parallel_rank):
    """
    Build forward execution schedule for Chimera 2-VR

    Strategy:
    - Interleave VR0 and VR1 microbatches for pipeline efficiency
    - Rank-dependent ordering to minimize bubbles
    - Simpler than BitPipe 4-VR (no V-shape complexity)

    Example (4 devices, 16 total microbatches):
    - Rank 0: [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]
    - Rank 3: [8, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7]

    Args:
        total_num_microbatches: Total microbatches (doubled)
        pipeline_parallel_size: Number of pipeline stages
        pipeline_parallel_rank: Current rank

    Returns:
        List of microbatch IDs in execution order
    """
    microbatch_groups = get_chimera_microbatch_groups(total_num_microbatches)

    # Calculate scheduling parameters
    num_unit = pipeline_parallel_size // 2  # Half the pipeline

    # Determine if this rank is in first or second half
    is_first_half = pipeline_parallel_rank < num_unit

    forward_schedule = []

    # Strategy: Interleave VR0 and VR1 microbatches
    # Early ranks start with more VR0, later ranks start with more VR1

    if is_first_half:
        # First half ranks: prioritize VR0 → VR1
        offset = pipeline_parallel_rank

        # Initial VR0 microbatches
        for i in range(offset + 1):
            if i < len(microbatch_groups[0]):
                forward_schedule.append(microbatch_groups[0][i])

        # Interleave remaining
        vr0_idx = offset + 1
        vr1_idx = 0

        while vr0_idx < len(microbatch_groups[0]) or vr1_idx < len(microbatch_groups[1]):
            # Add VR1
            if vr1_idx < len(microbatch_groups[1]):
                forward_schedule.append(microbatch_groups[1][vr1_idx])
                vr1_idx += 1

            # Add VR0
            if vr0_idx < len(microbatch_groups[0]):
                forward_schedule.append(microbatch_groups[0][vr0_idx])
                vr0_idx += 1

    else:
        # Second half ranks: prioritize VR1 → VR0
        offset = pipeline_parallel_size - 1 - pipeline_parallel_rank

        # Initial VR1 microbatches
        for i in range(offset + 1):
            if i < len(microbatch_groups[1]):
                forward_schedule.append(microbatch_groups[1][i])

        # Interleave remaining
        vr1_idx = offset + 1
        vr0_idx = 0

        while vr0_idx < len(microbatch_groups[0]) or vr1_idx < len(microbatch_groups[1]):
            # Add VR0
            if vr0_idx < len(microbatch_groups[0]):
                forward_schedule.append(microbatch_groups[0][vr0_idx])
                vr0_idx += 1

            # Add VR1
            if vr1_idx < len(microbatch_groups[1]):
                forward_schedule.append(microbatch_groups[1][vr1_idx])
                vr1_idx += 1

    return forward_schedule


def build_chimera_backward_schedule(total_num_microbatches, pipeline_parallel_size, pipeline_parallel_rank):
    """
    Build backward execution schedule for Chimera 2-VR

    Strategy:
    - Process VR1 before VR0 (backward pipeline flows opposite)
    - Add gradient sync markers (-1)
    - Rank-dependent ordering

    Args:
        total_num_microbatches: Total microbatches (doubled)
        pipeline_parallel_size: Number of pipeline stages
        pipeline_parallel_rank: Current rank

    Returns:
        List of microbatch IDs in execution order (includes -1 for sync)
    """
    microbatch_groups = get_chimera_microbatch_groups(total_num_microbatches)

    num_unit = pipeline_parallel_size // 2
    is_first_half = pipeline_parallel_rank < num_unit

    backward_schedule = []

    # Backward processes VR1 first (second half of model), then VR0
    if is_first_half:
        # First half ranks: VR1 → VR0
        offset = pipeline_parallel_rank

        # Initial VR1 microbatches
        for i in range(offset + 1):
            if i < len(microbatch_groups[1]):
                backward_schedule.append(microbatch_groups[1][i])

        # Interleave remaining
        vr1_idx = offset + 1
        vr0_idx = 0

        while vr0_idx < len(microbatch_groups[0]) or vr1_idx < len(microbatch_groups[1]):
            # Add VR0
            if vr0_idx < len(microbatch_groups[0]):
                backward_schedule.append(microbatch_groups[0][vr0_idx])
                vr0_idx += 1

            # Add VR1
            if vr1_idx < len(microbatch_groups[1]):
                backward_schedule.append(microbatch_groups[1][vr1_idx])
                vr1_idx += 1

    else:
        # Second half ranks: VR0 → VR1
        offset = pipeline_parallel_size - 1 - pipeline_parallel_rank

        # Initial VR0 microbatches
        for i in range(offset + 1):
            if i < len(microbatch_groups[0]):
                backward_schedule.append(microbatch_groups[0][i])

        # Interleave remaining
        vr0_idx = offset + 1
        vr1_idx = 0

        while vr0_idx < len(microbatch_groups[0]) or vr1_idx < len(microbatch_groups[1]):
            # Add VR1
            if vr1_idx < len(microbatch_groups[1]):
                backward_schedule.append(microbatch_groups[1][vr1_idx])
                vr1_idx += 1

            # Add VR0
            if vr0_idx < len(microbatch_groups[0]):
                backward_schedule.append(microbatch_groups[0][vr0_idx])
                vr0_idx += 1

    # Add eager gradient sync markers
    # Middle ranks sync earlier
    if pipeline_parallel_rank == num_unit or pipeline_parallel_rank == num_unit - 1:
        backward_schedule.append(-1)  # Sync marker
    else:
        backward_schedule.insert(-1, -1)  # Second-to-last position

    backward_schedule.append(-1)  # Final sync

    return backward_schedule


# Main Chimera 2-VR Scheduler
def forward_backward_pipelining_with_chimera_2vr(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
):
    """
    Main Chimera 2-VR scheduler - simplified bidirectional pipeline

    Key differences from BitPipe 4-VR:
    1. Virtual pipeline size = 2 (not 4)
    2. No same-device VR transitions (no VR0→VR2, VR1→VR3)
    3. Simpler layer assignment
    4. Direct bidirectional flow

    Args:
        forward_step_func: Function to execute forward pass
        data_iterator: Iterator over input data
        model: List of model chunks (length 2 for Chimera)
        num_microbatches: Number of microbatches (user-specified)
        seq_length: Sequence length
        micro_batch_size: Microbatch size
        decoder_seq_length: Decoder sequence length (for encoder-decoder)
        forward_only: If True, skip backward pass
        collect_non_loss_data: If True, collect non-loss data

    Returns:
        losses_reduced: List of reduced losses
    """

    # Get configuration
    args = get_args()
    config = get_model_config(model[0])

    # CRITICAL: Chimera requires virtual_pipeline_model_parallel_size=2
    # This should have been set in arguments.py when --enable-chimera-schedule was parsed
    assert hasattr(args, 'virtual_pipeline_model_parallel_size'), \
        "virtual_pipeline_model_parallel_size not set! Make sure --enable-chimera-schedule is passed."

    assert args.virtual_pipeline_model_parallel_size == 2, \
        f"Chimera requires virtual_pipeline_model_parallel_size=2, got {args.virtual_pipeline_model_parallel_size}"

    assert len(model) == 2, \
        f"Chimera requires exactly 2 model chunks, got {len(model)}"

    # Get pipeline parallel info
    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    # Double microbatches for bidirectional execution
    num_model_chunks = len(model)  # Should be 2
    total_num_microbatches = num_microbatches * (num_model_chunks // 2)  # num_microbatches * 1 = same, but keeping for clarity
    # Actually for Chimera: total = num_microbatches * 2 (bidirectional)
    total_num_microbatches = num_microbatches * 2

    print_all_ranks(f"Chimera Config: pipeline_size={pipeline_parallel_size}, rank={pipeline_parallel_rank}")
    print_all_ranks(f"Microbatches: user={num_microbatches}, total={total_num_microbatches}")

    # Validate configuration
    if num_microbatches % pipeline_parallel_size != 0:
        msg = f"number of microbatches ({num_microbatches}) is not divisible by "
        msg += f"pipeline-model-parallel-size ({pipeline_parallel_size})"
        raise RuntimeError(msg)

    # Tensor shape
    tensor_shape = (seq_length, micro_batch_size, config.hidden_size)
    if config.sequence_parallel:
        tensor_shape = (
            tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size(),
            tensor_shape[1],
            tensor_shape[2]
        )

    # Compute warmup and remaining microbatches
    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    else:
        if total_num_microbatches == pipeline_parallel_size:
            num_warmup_microbatches = total_num_microbatches
        else:
            # Warmup: fill the pipeline
            num_warmup_microbatches = pipeline_parallel_size

    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    print_all_ranks(f"Phase breakdown: warmup={num_warmup_microbatches}, remaining={num_microbatches_remaining}")

    # Initialize tracking structures
    input_tensors = [[] for _ in range(num_model_chunks)]  # 2 VRs
    output_tensors = [[] for _ in range(num_model_chunks)]
    losses_reduced = []

    if not forward_only:
        output_tensor_grads = [[] for _ in range(num_model_chunks)]

    # Build schedules
    forward_schedule = build_chimera_forward_schedule(
        total_num_microbatches, pipeline_parallel_size, pipeline_parallel_rank
    )
    backward_schedule = build_chimera_backward_schedule(
        total_num_microbatches, pipeline_parallel_size, pipeline_parallel_rank
    )

    print_all_ranks(f"Forward schedule: {forward_schedule}")
    print_all_ranks(f"Backward schedule: {backward_schedule}")
    print_all_ranks(f"[SCHEDULE BUILT] About to start execution")

    # Model chunk ID function
    def get_model_chunk_id(microbatch_id):
        """Maps microbatch ID to VR (0 or 1)"""
        if microbatch_id == -1:
            return -1
        # Simple: first half → VR0, second half → VR1
        return 0 if microbatch_id < (total_num_microbatches // 2) else 1

    # ============================================================================
    # Bidirectional Gradient Synchronization
    # ============================================================================
    # This is CRITICAL for Chimera 2-VR correctness!
    # Unlike DDP sync (no_sync_func above), this syncs across BIDIRECTIONAL pairs.
    #
    # Why needed: In Chimera, paired devices process the SAME layers from opposite
    # pipeline directions. Each layer accumulates gradients from TWO sources:
    #
    # Example (4 devices, 48 layers):
    #   Device 0 VR0: layers 1-12  (forward pipeline)
    #   Device 3 VR1: layers 1-12  (backward pipeline) <- SAME LAYERS!
    #   -> Must average gradients from both devices
    #
    # BD Groups pair devices that share layers:
    #   BD Group 0: [Device 0, Device 3]
    #   BD Group 1: [Device 1, Device 2]
    # ============================================================================
    def allreduce_gradients(model_chunk):
        """
        Synchronize gradients across bidirectional pipeline pairs using AllReduce.

        This function performs gradient averaging within BD (bidirectional) groups.
        It is called during gradient sync markers (-1) in the backward schedule.

        Args:
            model_chunk: A single VR's model chunk to sync (VR0 or VR1 for Chimera)

        Note:
            - This syncs at the DEVICE level (pipeline ranks), not VR level
            - BD groups are initialized in parallel_state.py based on layer sharing
            - Each device calls this 2 times (once for VR0, once for VR1)
        """
        if (
            parallel_state.is_rank_in_bd_group()
            and parallel_state.get_pipeline_model_parallel_world_size() > 1
        ):
            torch.distributed.barrier(group=parallel_state.get_bd_parallel_group())
            buckets = {}
            for param in model_chunk.module.parameters():
                if param.requires_grad and param.main_grad is not None:
                    tp = param.data.type()
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(param)

            # For each bucket, all-reduce and copy all-reduced grads
            for tp in buckets:
                bucket = buckets[tp]
                grads = [param.main_grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads)
                coalesced /= torch.distributed.get_world_size(group=parallel_state.get_bd_parallel_group())
                torch.distributed.all_reduce(
                    coalesced, group=parallel_state.get_bd_parallel_group()
                )
                for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                    buf.copy_(synced)

    # Disable gradient sync
    no_sync_func = config.no_sync_func

    # Handle case where no_sync_func is not set
    if no_sync_func is None and all(isinstance(chunk, torchDDP) for chunk in model):
        # Multiple model chunks - create a context that syncs all of them
        def multi_no_sync():
            stack = contextlib.ExitStack()
            for chunk in model:
                stack.enter_context(chunk.no_sync())
            return stack

        no_sync_func = multi_no_sync

    # Fallback to nullcontext if still None
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    no_sync_context = None

    def disable_grad_sync():
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    if not forward_only:
        disable_grad_sync()

    # Model chunks with synchronized grads
    synchronized_model_chunks = set()

    # === WARMUP PHASE ===
    print_all_ranks("=== Starting WARMUP phase ===")
    print_all_ranks(f"[WARMUP] num_warmup_microbatches={num_warmup_microbatches}")
    print_all_ranks(f"[WARMUP] forward_schedule length={len(forward_schedule)}")

    for i in range(num_warmup_microbatches):
        microbatch_id = forward_schedule[i]
        model_chunk_id = get_model_chunk_id(microbatch_id)

        print_all_ranks(f"Warmup {i}: MB {microbatch_id}, VR {model_chunk_id}")
        print_all_ranks(f"[WARMUP {i}] About to call recv_forward()")

        # Receive input (use p2p_communication directly, not schedules wrapper)
        if not parallel_state.is_pipeline_first_stage():
            input_tensor = p2p_communication.recv_forward(tensor_shape, config)
        else:
            input_tensor = None

        print_all_ranks(f"[WARMUP {i}] recv_forward() completed, input_tensor shape={input_tensor.shape if input_tensor is not None else None}")
        print_all_ranks(f"[WARMUP {i}] About to call forward_step()")

        # Forward pass
        output_tensor = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store=[],
            config=config,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=None,
            is_first_microbatch_for_model_chunk=(i == 0),
            current_microbatch=microbatch_id,
        )

        print_all_ranks(f"[WARMUP {i}] forward_step() completed, output_tensor type={type(output_tensor)}")
        print_all_ranks(f"[WARMUP {i}] About to call send_forward()")

        # Send output (use p2p_communication directly, not schedules wrapper)
        if not parallel_state.is_pipeline_last_stage():
            p2p_communication.send_forward(output_tensor, config)

        print_all_ranks(f"[WARMUP {i}] send_forward() completed")

        # Store tensors
        input_tensors[model_chunk_id].append(input_tensor)
        output_tensors[model_chunk_id].append(output_tensor)

        # Collect loss
        if output_tensor is not None and isinstance(output_tensor, dict):
            if 'loss' in output_tensor:
                losses_reduced.append(output_tensor['loss'])

    # === STEADY STATE (1F1B) ===
    if not forward_only and num_microbatches_remaining > 0:
        print_all_ranks("=== Starting STEADY STATE (1F1B) phase ===")
        for i in range(num_microbatches_remaining):
            # Forward
            fwd_microbatch_id = forward_schedule[num_warmup_microbatches + i]
            fwd_chunk_id = get_model_chunk_id(fwd_microbatch_id)

            print_all_ranks(f"1F1B {i}: Forward MB {fwd_microbatch_id}, VR {fwd_chunk_id}")

            # Use p2p_communication directly, not schedules wrapper
            if not parallel_state.is_pipeline_first_stage():
                input_tensor = p2p_communication.recv_forward(tensor_shape, config)
            else:
                input_tensor = None

            output_tensor = forward_step(
                forward_step_func,
                data_iterator[fwd_chunk_id],
                model[fwd_chunk_id],
                num_microbatches,
                input_tensor,
                forward_data_store=[],
                config=config,
                collect_non_loss_data=collect_non_loss_data,
                checkpoint_activations_microbatch=None,
                is_first_microbatch_for_model_chunk=False,
                current_microbatch=fwd_microbatch_id,
            )

            # Use p2p_communication directly, not schedules wrapper
            if not parallel_state.is_pipeline_last_stage():
                p2p_communication.send_forward(output_tensor, config)

            input_tensors[fwd_chunk_id].append(input_tensor)
            output_tensors[fwd_chunk_id].append(output_tensor)

            if output_tensor is not None and isinstance(output_tensor, dict):
                if 'loss' in output_tensor:
                    losses_reduced.append(output_tensor['loss'])

            # Backward
            bwd_microbatch_id = backward_schedule[i]
            if bwd_microbatch_id == -1:
                # Gradient sync - perform bidirectional AllReduce
                print_all_ranks(f"1F1B {i}: GRADIENT SYNC (AllReduce)")
                enable_grad_sync()
                for chunk_id in range(num_model_chunks):
                    if chunk_id not in synchronized_model_chunks:
                        print_all_ranks(f"  Syncing VR{chunk_id} gradients")
                        allreduce_gradients(model[chunk_id])
                        synchronized_model_chunks.add(chunk_id)
                disable_grad_sync()
                continue

            bwd_chunk_id = get_model_chunk_id(bwd_microbatch_id)

            print_all_ranks(f"1F1B {i}: Backward MB {bwd_microbatch_id}, VR {bwd_chunk_id}")

            # Use p2p_communication directly, not schedules wrapper
            if not parallel_state.is_pipeline_last_stage():
                input_tensor_grad = p2p_communication.recv_backward(tensor_shape, config)
            else:
                input_tensor_grad = None

            output_tensor = output_tensors[bwd_chunk_id].pop(0)
            input_tensor = input_tensors[bwd_chunk_id].pop(0)

            output_tensor_grad = backward_step(
                input_tensor, output_tensor, input_tensor_grad, model_type=config.model_type
            )

            # Use p2p_communication directly, not schedules wrapper
            if not parallel_state.is_pipeline_first_stage():
                p2p_communication.send_backward(output_tensor_grad, config)

    # === COOLDOWN PHASE ===
    if not forward_only:
        print_all_ranks("=== Starting COOLDOWN phase ===")
        remaining_backward = backward_schedule[num_microbatches_remaining:]

        for i, bwd_microbatch_id in enumerate(remaining_backward):
            if bwd_microbatch_id == -1:
                # Gradient sync - perform bidirectional AllReduce
                print_all_ranks(f"Cooldown {i}: GRADIENT SYNC (AllReduce)")
                enable_grad_sync()
                for chunk_id in range(num_model_chunks):
                    if chunk_id not in synchronized_model_chunks:
                        print_all_ranks(f"  Syncing VR{chunk_id} gradients")
                        allreduce_gradients(model[chunk_id])
                        synchronized_model_chunks.add(chunk_id)
                disable_grad_sync()
                continue

            bwd_chunk_id = get_model_chunk_id(bwd_microbatch_id)

            print_all_ranks(f"Cooldown {i}: Backward MB {bwd_microbatch_id}, VR {bwd_chunk_id}")

            # Use p2p_communication directly, not schedules wrapper
            if not parallel_state.is_pipeline_last_stage():
                input_tensor_grad = p2p_communication.recv_backward(tensor_shape, config)
            else:
                input_tensor_grad = None

            output_tensor = output_tensors[bwd_chunk_id].pop(0)
            input_tensor = input_tensors[bwd_chunk_id].pop(0)

            output_tensor_grad = backward_step(
                input_tensor, output_tensor, input_tensor_grad, model_type=config.model_type
            )

            # Use p2p_communication directly, not schedules wrapper
            if not parallel_state.is_pipeline_first_stage():
                p2p_communication.send_backward(output_tensor_grad, config)

    # Final gradient sync
    if not forward_only:
        enable_grad_sync()

    print_all_ranks("=== Chimera schedule completed ===")

    return losses_reduced
