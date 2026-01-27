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
import time
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


# P2P Communication profiling wrappers (same as BitPipe)
def _profile_p2p_comm(comm_func, comm_type, *args, **kwargs):
    """Wrapper to add profiling around P2P communication functions"""
    profiler = get_bitpipe_profiler()
    if profiler and profiler.enabled:
        # Get ranks for profiling
        current_rank = parallel_state.get_pipeline_model_parallel_rank()
        if 'send' in comm_type:
            if 'forward' in comm_type:
                dest_rank = parallel_state.get_pipeline_model_parallel_next_rank()
            else:
                dest_rank = parallel_state.get_pipeline_model_parallel_prev_rank()
            profiler.start_p2p_comm(comm_type, current_rank, dest_rank)
        elif 'recv' in comm_type:
            if 'forward' in comm_type:
                source_rank = parallel_state.get_pipeline_model_parallel_prev_rank()
            else:
                source_rank = parallel_state.get_pipeline_model_parallel_next_rank()
            profiler.start_p2p_comm(comm_type, source_rank, current_rank)

    result = comm_func(*args, **kwargs)

    if profiler and profiler.enabled:
        # Get ranks for profiling
        current_rank = parallel_state.get_pipeline_model_parallel_rank()
        if 'send' in comm_type:
            if 'forward' in comm_type:
                dest_rank = parallel_state.get_pipeline_model_parallel_next_rank()
            else:
                dest_rank = parallel_state.get_pipeline_model_parallel_prev_rank()
            profiler.end_p2p_comm(comm_type, current_rank, dest_rank)
        elif 'recv' in comm_type:
            if 'forward' in comm_type:
                source_rank = parallel_state.get_pipeline_model_parallel_prev_rank()
            else:
                source_rank = parallel_state.get_pipeline_model_parallel_next_rank()
            profiler.end_p2p_comm(comm_type, source_rank, current_rank)

    return result


def print_all_ranks(message, include_time=False):
    """Print from all ranks for debugging Chimera schedule with optional timing."""
    if os.environ.get('CHIMERA_DEBUG', '0') == '1':
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_rank = os.environ.get('RANK', 'unknown')
            timestamp = f"[{time.time():.2f}]" if include_time else ""
            print(f"[Chimera Rank {rank}/World {world_rank}] {timestamp} {message}", flush=True)
        else:
            timestamp = f"[{time.time():.2f}]" if include_time else ""
            print(f"[Chimera] {timestamp} {message}", flush=True)


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


def is_vr_first_stage_for_activation(model_chunk_id, pipeline_rank, pipeline_size):
    """
    Check if this rank is the FIRST stage for forward activation flow in the given VR.

    First stage = where activations originate (embedding creation)

    For Chimera 2-VR:
    - VR0 (flows 0→1→2→3): First stage is Rank 0
    - VR1 (flows 3→2→1→0): First stage is Rank N-1

    Args:
        model_chunk_id: VR index (0 or 1)
        pipeline_rank: Current device rank
        pipeline_size: Total pipeline devices

    Returns:
        True if this rank should NOT receive forward activations for this VR
    """
    if model_chunk_id == 0:
        # VR0: first stage is Rank 0
        return pipeline_rank == 0
    else:
        # VR1: first stage is Rank N-1
        return pipeline_rank == pipeline_size - 1


def is_vr_last_stage_for_activation(model_chunk_id, pipeline_rank, pipeline_size):
    """
    Check if this rank is the LAST stage for forward activation flow in the given VR.

    Last stage = where activations exit (output/loss calculation)

    For Chimera 2-VR:
    - VR0 (flows 0→1→2→3): Last stage is Rank N-1
    - VR1 (flows 3→2→1→0): Last stage is Rank 0

    Args:
        model_chunk_id: VR index (0 or 1)
        pipeline_rank: Current device rank
        pipeline_size: Total pipeline devices

    Returns:
        True if this rank should NOT send forward activations for this VR
    """
    if model_chunk_id == 0:
        # VR0: last stage is Rank N-1
        return pipeline_rank == pipeline_size - 1
    else:
        # VR1: last stage is Rank 0
        return pipeline_rank == 0


def is_vr_first_stage_for_gradient(model_chunk_id, pipeline_rank, pipeline_size):
    """
    Check if this rank is the FIRST stage for backward gradient flow in the given VR.

    First stage for gradient = where gradients originate (loss calculation)
    This is the OPPOSITE of activation flow!

    For Chimera 2-VR:
    - VR0 (activations 0→1→2→3, gradients 3→2→1→0): Gradient first stage is Rank N-1
    - VR1 (activations 3→2→1→0, gradients 0→1→2→3): Gradient first stage is Rank 0

    Args:
        model_chunk_id: VR index (0 or 1)
        pipeline_rank: Current device rank
        pipeline_size: Total pipeline devices

    Returns:
        True if this rank should NOT receive gradients for this VR
    """
    if model_chunk_id == 0:
        # VR0: gradient first stage is Rank N-1 (where loss is computed)
        return pipeline_rank == pipeline_size - 1
    else:
        # VR1: gradient first stage is Rank 0 (where loss is computed)
        return pipeline_rank == 0


def is_vr_last_stage_for_gradient(model_chunk_id, pipeline_rank, pipeline_size):
    """
    Check if this rank is the LAST stage for backward gradient flow in the given VR.

    Last stage for gradient = where gradients end (embedding layer)
    This is the OPPOSITE of activation flow!

    For Chimera 2-VR:
    - VR0 (activations 0→1→2→3, gradients 3→2→1→0): Gradient last stage is Rank 0
    - VR1 (activations 3→2→1→0, gradients 0→1→2→3): Gradient last stage is Rank N-1

    Args:
        model_chunk_id: VR index (0 or 1)
        pipeline_rank: Current device rank
        pipeline_size: Total pipeline devices

    Returns:
        True if this rank should NOT send gradients for this VR
    """
    if model_chunk_id == 0:
        # VR0: gradient last stage is Rank 0 (embedding layer)
        return pipeline_rank == 0
    else:
        # VR1: gradient last stage is Rank N-1 (embedding layer)
        return pipeline_rank == pipeline_size - 1


def recv_forward_vr_aware(model_chunk_id, pipeline_rank, pipeline_size, tensor_shape, config):
    """
    VR-aware forward activation receive that bypasses incorrect stage checks.

    For VR0: Use standard recv_forward (from prev rank)
    For VR1: Manually receive from next rank (bypassing recv_backward's stage check)

    Returns:
        Received tensor, or None if this is the first stage for activation flow
    """
    # Check if we're the first stage for activation flow in this VR
    if is_vr_first_stage_for_activation(model_chunk_id, pipeline_rank, pipeline_size):
        return None

    if model_chunk_id == 0:
        # VR0: standard forward receive from prev rank
        return p2p_communication.recv_forward(tensor_shape, config)
    else:
        # VR1: receive from next rank (bypass recv_backward's is_pipeline_last_stage check)
        # Use _communicate directly to avoid the unwanted stage check
        _, output_tensor, _ = p2p_communication._communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,  # Receive from next rank
            tensor_shape=tensor_shape,
            config=config,
        )
        return output_tensor


def send_forward_vr_aware(model_chunk_id, pipeline_rank, pipeline_size, output_tensor, config):
    """
    VR-aware forward activation send that bypasses incorrect stage checks.

    For VR0: Use standard send_forward (to next rank)
    For VR1: Manually send to prev rank (bypassing send_backward's stage check)

    Returns:
        None (send operations don't return values)
    """
    # Check if we're the last stage for activation flow in this VR
    if is_vr_last_stage_for_activation(model_chunk_id, pipeline_rank, pipeline_size):
        return

    if model_chunk_id == 0:
        # VR0: standard forward send to next rank
        p2p_communication.send_forward(output_tensor, config)
    else:
        # VR1: send to prev rank (bypass send_backward's is_pipeline_first_stage check)
        # Use _communicate directly to avoid the unwanted stage check
        p2p_communication._communicate(
            tensor_send_next=None,
            tensor_send_prev=output_tensor,  # Send to prev rank
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
            config=config,
        )


def recv_backward_vr_aware(model_chunk_id, pipeline_rank, pipeline_size, tensor_shape, config):
    """
    VR-aware backward gradient receive that bypasses incorrect stage checks.

    Gradients flow OPPOSITE to activations:
    - VR0 activations: 0→1→2→3, gradients: 3→2→1→0
    - VR1 activations: 3→2→1→0, gradients: 0→1→2→3

    For VR0: Receive from next rank (standard recv_backward)
    For VR1: Receive from prev rank (use recv_forward)

    Returns:
        Received gradient tensor, or None if this is the first stage for gradient flow
    """
    # Check if we're the first stage for gradient flow in this VR
    if is_vr_first_stage_for_gradient(model_chunk_id, pipeline_rank, pipeline_size):
        return None

    if model_chunk_id == 0:
        # VR0: receive gradient from next rank
        # Use _communicate directly to avoid recv_backward's is_pipeline_last_stage check
        _, output_tensor_grad, _ = p2p_communication._communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,  # Receive from next rank
            tensor_shape=tensor_shape,
            config=config,
        )
        return output_tensor_grad
    else:
        # VR1: receive gradient from prev rank
        # Use _communicate directly to avoid recv_forward's is_pipeline_first_stage check
        output_tensor_grad, _, _ = p2p_communication._communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,  # Receive from prev rank
            recv_next=False,
            tensor_shape=tensor_shape,
            config=config,
        )
        return output_tensor_grad


def send_backward_vr_aware(model_chunk_id, pipeline_rank, pipeline_size, input_tensor_grad, config):
    """
    VR-aware backward gradient send that bypasses incorrect stage checks.

    Gradients flow OPPOSITE to activations:
    - VR0 activations: 0→1→2→3, gradients: 3→2→1→0
    - VR1 activations: 3→2→1→0, gradients: 0→1→2→3

    For VR0: Send to prev rank (standard send_backward)
    For VR1: Send to next rank (use send_forward)

    Returns:
        None (send operations don't return values)
    """
    # Check if we're the last stage for gradient flow in this VR
    if is_vr_last_stage_for_gradient(model_chunk_id, pipeline_rank, pipeline_size):
        return

    if model_chunk_id == 0:
        # VR0: send gradient to prev rank
        # Use _communicate directly to avoid send_backward's is_pipeline_first_stage check
        p2p_communication._communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,  # Send to prev rank
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
            config=config,
        )
    else:
        # VR1: send gradient to next rank
        # Use _communicate directly to avoid send_forward's is_pipeline_last_stage check
        p2p_communication._communicate(
            tensor_send_next=input_tensor_grad,  # Send to next rank
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
            config=config,
        )


def get_chimera_microbatch_idx(num_microbatches, pipeline_parallel_size, pipeline_parallel_rank):
    """
    Generate forward microbatch schedule for Chimera 2-VR (NEW ALGORITHM)

    Chimera uses simple 2-VR architecture without V-shaped transformation.
    No doubling of microbatches - just use the original count.

    Pattern:
    - Process microbatches in chunks of 4 (num_unit * 2)
    - Apply rank-dependent reordering to each chunk
    - Different pattern for first half vs second half ranks

    Args:
        num_microbatches: Original count (NOT doubled)
        pipeline_parallel_size: Number of devices (must be even)
        pipeline_parallel_rank: Current device rank (0 to num_devices-1)

    Returns:
        List of microbatch IDs in execution order
    """
    microbatch_idx = []
    num_unit = pipeline_parallel_size // 2  # 2 for 4 devices
    i_half = pipeline_parallel_rank // num_unit  # 0 for ranks 0-1, 1 for ranks 2-3
    position_in_half = pipeline_parallel_rank % num_unit  # 0 or 1

    # Process in chunks of 4 microbatches (num_unit * 2)
    chunk_size = num_unit * 2
    num_chunks = num_microbatches // chunk_size

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        # Get the 4 MBs in this chunk: [0,1,2,3], [4,5,6,7]
        chunk_mbs = [chunk_start, chunk_start + 1, chunk_start + 2, chunk_start + 3]

        if i_half == 0:  # First half ranks (0, 1)
            if position_in_half == 0:
                # Rank 0: sequential order
                # [0,1,2,3] → [0,1,2,3]
                microbatch_idx.extend(chunk_mbs)
            else:
                # Rank 1: swap pairs within chunk
                # [0,1,2,3] → [0,2,1,3]
                microbatch_idx.extend([chunk_mbs[0], chunk_mbs[2], chunk_mbs[1], chunk_mbs[3]])
        else:  # Second half ranks (2, 3)
            if position_in_half == 0:
                # Rank 2: reverse each pair
                # [0,1,2,3] → [2,0,3,1]
                microbatch_idx.extend([chunk_mbs[2], chunk_mbs[0], chunk_mbs[3], chunk_mbs[1]])
            else:
                # Rank 3: reverse pair order (group pairs)
                # [0,1,2,3] → [2,3,0,1]
                microbatch_idx.extend([chunk_mbs[2], chunk_mbs[3], chunk_mbs[0], chunk_mbs[1]])

    return microbatch_idx


def get_chimera_bkmicrobatch_idx(num_microbatches, pipeline_parallel_size, pipeline_parallel_rank):
    """
    Generate backward microbatch schedule for Chimera 2-VR (NEW ALGORITHM)

    Pattern:
    - Backward is derived from the paired rank's forward schedule
    - Paired rank = num_devices - 1 - current_rank
    - Add sync markers (-1) at specific positions

    Args:
        num_microbatches: Original count (NOT doubled)
        pipeline_parallel_size: Number of devices (must be even)
        pipeline_parallel_rank: Current device rank (0 to num_devices-1)

    Returns:
        List of microbatch IDs + sync markers in execution order
    """
    num_unit = pipeline_parallel_size // 2

    # Get paired rank's forward schedule
    paired_rank = pipeline_parallel_size - 1 - pipeline_parallel_rank
    paired_forward = get_chimera_microbatch_idx(num_microbatches, pipeline_parallel_size, paired_rank)

    # Use paired rank's forward schedule as backward for current rank
    microbatch_idx_b = list(paired_forward)

    # Add sync markers (exactly 2 per rank)
    # Position based on rank:
    # - Middle ranks (N/2-1 and N/2): both sync markers at end
    # - Outer ranks (0 and N-1): first sync before last MB, second at end
    #
    # Expected output for 4 devices, 4 MBs:
    # Rank 0: [2, 3, 0, -1, 1, -1]  (outer rank)
    # Rank 1: [2, 0, 3, 1, -1, -1]  (middle rank)
    # Rank 2: [0, 2, 1, 3, -1, -1]  (middle rank)
    # Rank 3: [0, 1, 2, -1, 3, -1]  (outer rank)
    if pipeline_parallel_rank == num_unit or pipeline_parallel_rank == num_unit - 1:
        # Middle ranks: append both at end
        microbatch_idx_b.append(-1)
        microbatch_idx_b.append(-1)
    else:
        # Outer ranks: insert first sync BEFORE last element, then append second
        # insert(-1, val) inserts BEFORE index -1 (i.e., before last element)
        microbatch_idx_b.insert(-1, -1)  # Insert sync before last MB
        microbatch_idx_b.append(-1)       # Append final sync

    return microbatch_idx_b


def get_model_chunk_id(microbatch_id, pipeline_parallel_size):
    """
    Map microbatch ID to VR chunk (0 or 1 for Chimera 2-VR)

    Pattern for 2 VRs:
    - VR0: microbatches where (id % pipeline_size) // (pipeline_size // 2) == 0
    - VR1: microbatches where (id % pipeline_size) // (pipeline_size // 2) == 1

    Examples (4 devices):
    4 MB:  [0,1] → VR0, [2,3] → VR1
    8 MB:  [0,1,4,5] → VR0, [2,3,6,7] → VR1
    12 MB: [0,1,4,5,8,9] → VR0, [2,3,6,7,10,11] → VR1
    16 MB: [0,1,4,5,8,9,12,13] → VR0, [2,3,6,7,10,11,14,15] → VR1
    """
    microbatch_id_in_group = microbatch_id % pipeline_parallel_size
    model_chunk_id = microbatch_id_in_group // (pipeline_parallel_size // 2)

    if microbatch_id == -1:  # Sync marker
        model_chunk_id = -1

    return model_chunk_id


def get_num_warmup_microbatches(pipeline_parallel_rank, pipeline_parallel_size, num_microbatches):
    """
    Calculate number of warmup microbatches for Chimera 2-VR

    Pattern for 4 devices, 4 microbatches:
    - Rank 0: warmup=2, 1f1b=2
    - Rank 1: warmup=3, 1f1b=1
    - Rank 2: warmup=3, 1f1b=1
    - Rank 3: warmup=2, 1f1b=2

    Formula:
    - First half ranks (0 to N/2-1): warmup = num_unit + position_in_half
    - Second half ranks (N/2 to N-1): warmup = num_unit + (num_unit - 1 - position_in_half)

    This creates a symmetric pattern where:
    - Outer ranks (0, N-1) have fewer warmup steps
    - Inner ranks (N/2-1, N/2) have more warmup steps

    Args:
        pipeline_parallel_rank: Current device rank (0 to num_devices-1)
        pipeline_parallel_size: Number of devices (must be even)
        num_microbatches: Total number of microbatches (NOT doubled)

    Returns:
        num_warmup: Number of warmup microbatches for this rank
    """
    num_unit = pipeline_parallel_size // 2
    i_half = pipeline_parallel_rank // num_unit  # 0 for first half, 1 for second half
    position_in_half = pipeline_parallel_rank % num_unit

    if i_half == 0:
        # First half ranks (0, 1, ...): increasing warmup
        num_warmup = num_unit + position_in_half
    else:
        # Second half ranks (N/2, N/2+1, ...): decreasing warmup (mirror)
        num_warmup = num_unit + (num_unit - 1 - position_in_half)

    # Cap at total microbatches
    num_warmup = min(num_warmup, num_microbatches)

    return num_warmup


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

    NO DOUBLING: Uses num_microbatches directly (not * 2)
    With proper WARMUP, STEADY STATE, and COOLDOWN phases

    Key differences from BitPipe 4-VR:
    1. Virtual pipeline size = 2 (not 4)
    2. No V-shaped transformation (sequential)
    3. No microbatch doubling
    4. Simpler scheduling algorithm

    Args:
        forward_step_func: Function to execute forward pass
        data_iterator: Iterator over input data
        model: List of model chunks (length 2 for Chimera)
        num_microbatches: Number of microbatches (user-specified, NOT doubled)
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
    model_type = get_model_type(model[0])

    # CRITICAL: Chimera requires virtual_pipeline_model_parallel_size=2
    assert hasattr(args, 'virtual_pipeline_model_parallel_size'), \
        "virtual_pipeline_model_parallel_size not set!"

    assert args.virtual_pipeline_model_parallel_size == 2, \
        f"Chimera requires virtual_pipeline_model_parallel_size=2, got {args.virtual_pipeline_model_parallel_size}"

    assert len(model) == 2, \
        f"Chimera requires exactly 2 model chunks, got {len(model)}"

    # Get pipeline parallel info
    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    # Chimera does NOT double microbatches
    total_num_microbatches = num_microbatches  # NO doubling!

    print_all_ranks(f"Chimera Config: pipeline_size={pipeline_parallel_size}, rank={pipeline_parallel_rank}")
    print_all_ranks(f"Microbatches: user={num_microbatches}, total={total_num_microbatches} (NO doubling)")

    # Validate configuration
    if num_microbatches % pipeline_parallel_size != 0:
        msg = f"number of microbatches ({num_microbatches}) is not divisible by "
        msg += f"pipeline-model-parallel-size ({pipeline_parallel_size})"
        raise RuntimeError(msg)

    # Initialize profiler if enabled
    profiler = None
    try:
        from megatron.core.iteration_context import should_profile_current_iteration
        if should_profile_current_iteration():
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            # Use "chimera" schedule type in the profiler
            profiler = initialize_bitpipe_profiler(pipeline_parallel_rank, world_size, enabled=True, schedule_type="chimera")
            profiler.start_profiling()
            print_all_ranks(f"Profiling enabled for Chimera schedule")
    except ImportError:
        pass  # Profiling not available

    # Tensor shape
    tensor_shape = (seq_length, micro_batch_size, config.hidden_size)
    if config.sequence_parallel:
        tensor_shape = (
            tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size(),
            tensor_shape[1],
            tensor_shape[2]
        )

    # Compute phase breakdown using dynamic calculation
    # Chimera 2-VR uses warmup/1F1B/cooldown structure (NOT simple 2-phase)
    #
    # Phase Values for 4 devices, 4 microbatches:
    # Rank | warmup | 1f1b | cooldown
    # -----|--------|------|----------
    #   0  |   2    |  2   |    4     (2 warmup + 2 sync markers)
    #   1  |   3    |  1   |    4
    #   2  |   3    |  1   |    4
    #   3  |   2    |  2   |    4

    if forward_only:
        num_warmup_microbatches = total_num_microbatches
        num_1f1b_microbatches = 0
        num_cooldown_microbatches = 0
    else:
        # Calculate warmup using rank-dependent formula
        num_warmup_microbatches = get_num_warmup_microbatches(
            pipeline_parallel_rank, pipeline_parallel_size, total_num_microbatches
        )

        # 1F1B = remaining microbatches after warmup
        num_1f1b_microbatches = total_num_microbatches - num_warmup_microbatches

        # Cooldown = warmup count + 2 sync markers
        # (backward schedule has warmup count of actual MBs + 2 sync markers)
        num_cooldown_microbatches = num_warmup_microbatches + 2

    print_all_ranks(f"[PHASE CALC] warmup={num_warmup_microbatches}, 1f1b={num_1f1b_microbatches}, cooldown={num_cooldown_microbatches}")

    # Initialize tracking structures
    input_tensors = [[] for _ in range(len(model))]  # 2 VRs
    output_tensors = [[] for _ in range(len(model))]
    losses_reduced = []

    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    # Build schedules using NEW ALGORITHM
    microbatch_idx = get_chimera_microbatch_idx(total_num_microbatches, pipeline_parallel_size, pipeline_parallel_rank)
    microbatch_idx_b = get_chimera_bkmicrobatch_idx(total_num_microbatches, pipeline_parallel_size, pipeline_parallel_rank)

    print_all_ranks(f"Forward schedule: {microbatch_idx}")
    print_all_ranks(f"Backward schedule: {microbatch_idx_b}")

    # NOTE: Pre-receive is NOT needed for warmup/1F1B/cooldown structure
    # Each phase handles its own recv_forward/recv_backward calls inline
    # This is different from the simple 2-phase loop which needed pre-initialization
    print_all_ranks(f"[INIT] Rank {pipeline_parallel_rank}/{pipeline_parallel_size} ready for execution")

    # Disable gradient sync by default
    no_sync_func = config.no_sync_func

    if no_sync_func is None and all(isinstance(chunk, torchDDP) for chunk in model):
        def multi_no_sync():
            stack = contextlib.ExitStack()
            for chunk in model:
                stack.enter_context(chunk.no_sync())
            return stack
        no_sync_func = multi_no_sync

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

    def allreduce_gradients(model_chunk):
        """Synchronize gradients across bidirectional pipeline pairs"""
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

    # Forward data store for non-loss data collection
    forward_data_store = []

    def forward_step_helper(microbatch_id, checkpoint_activations_microbatch):
        """
        Helper method to run forward step with model split into chunks.

        For Chimera 2-VR:
        - VR0: First pipeline (forward direction)
        - VR1: Second pipeline (backward direction)

        This helper manages:
        - Input tensor queue (FIFO with .pop(0))
        - Forward step execution
        - Output tensor queue (append)
        - Profiling integration
        """
        model_chunk_id = get_model_chunk_id(microbatch_id, pipeline_parallel_size)

        # Pipeline ID for Chimera (2 VRs)
        # VR0 = pipeline 0, VR1 = pipeline 1
        pipeline_id = model_chunk_id

        # Start profiling
        if profiler:
            profiler.start_microbatch(microbatch_id, pipeline_id, model_chunk_id, 'forward')

        # Boundary check: first stage gets None input
        if parallel_state.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)

        # FIFO: Pop from front of queue
        input_tensor = input_tensors[model_chunk_id].pop(0)

        # Execute forward step
        output_tensor = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,  # Chimera does NOT divide by 2
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
        )

        # Add to output queue and deallocate for memory optimization
        output_tensors[model_chunk_id].append(output_tensor)
        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

        # End profiling
        if profiler:
            profiler.end_microbatch(microbatch_id, pipeline_id, model_chunk_id, 'forward')

        return output_tensor

    def backward_step_helper(microbatch_id):
        """
        Helper method to run backward step with model split into chunks.

        For Chimera 2-VR:
        - Manages gradient tensor queues (FIFO)
        - Handles gradient synchronization triggers
        - Manages DDP sync context

        This helper manages:
        - Output gradient queue (FIFO with .pop(0))
        - Backward step execution
        - Gradient synchronization for BD groups
        - Profiling integration
        """
        model_chunk_id = get_model_chunk_id(microbatch_id, pipeline_parallel_size)

        # Pipeline ID for Chimera (2 VRs)
        pipeline_id = model_chunk_id

        # Start profiling
        if profiler:
            profiler.start_microbatch(microbatch_id, pipeline_id, model_chunk_id, 'backward')

        # Boundary check: last stage gets None gradient
        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)

        # FIFO: Pop from front of queues
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)

        # Execute backward step
        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )

        # End profiling
        if profiler:
            profiler.end_microbatch(microbatch_id, pipeline_id, model_chunk_id, 'backward')

        return input_tensor_grad

    if not forward_only:
        disable_grad_sync()

    synchronized_model_chunks = set()

    # ============================================================================
    # WARMUP / 1F1B / COOLDOWN EXECUTION LOOP
    # ============================================================================
    #
    # This replaces the simple 2-phase loop with proper pipeline scheduling:
    #
    # WARMUP PHASE:    Forward-only passes to fill the pipeline
    # 1F1B PHASE:      Interleaved forward+backward for steady state
    # COOLDOWN PHASE:  Backward-only passes to drain + gradient sync
    #
    # Example for Rank 0 (4 devices, 4 MBs):
    #   Warmup (k=0,1):     F(MB0), F(MB1)
    #   1F1B (k=0,1):       F(MB2)+B(MB2), F(MB3)+B(MB3)
    #   Cooldown (k=0..3):  B(MB0), B(-1 sync), B(MB1), B(-1 sync)
    # ============================================================================

    print_all_ranks("=== Starting CHIMERA Warmup/1F1B/Cooldown Execution ===")
    print_all_ranks(f"Forward schedule: {microbatch_idx}")
    print_all_ranks(f"Backward schedule: {microbatch_idx_b}")

    # Track current position in schedules
    fwd_idx = 0  # Current position in forward schedule
    bwd_idx = 0  # Current position in backward schedule

    # ===== WARMUP PHASE: Forward-only passes =====
    # Using BitPipe-style combined send+recv to avoid deadlocks
    print_all_ranks(f"=== WARMUP PHASE ({num_warmup_microbatches} forwards) ===")

    if profiler:
        profiler.record_phase_transition("warmup_start")

    # Step 1: Pre-receive initial input for first microbatch (if not first stage)
    first_mb_id = microbatch_idx[0]
    first_vr = get_model_chunk_id(first_mb_id, pipeline_parallel_size)
    first_is_first_stage = is_vr_first_stage_for_activation(first_vr, pipeline_parallel_rank, pipeline_parallel_size)

    if not first_is_first_stage:
        parallel_state.set_virtual_pipeline_model_parallel_rank(first_vr)
        if first_vr == 0:
            # VR0: pre-recv from prev rank
            print_all_ranks(f"[PRE-RECV] Rank{pipeline_parallel_rank} receiving VR0 input from prev rank")
            pre_recv_input = p2p_communication.chimera_recv_prev_only(tensor_shape, config)
        else:
            # VR1: pre-recv from next rank
            print_all_ranks(f"[PRE-RECV] Rank{pipeline_parallel_rank} receiving VR1 input from next rank")
            pre_recv_input = p2p_communication.chimera_recv_next_only(tensor_shape, config)
        print_all_ranks(f"[PRE-RECV] Rank{pipeline_parallel_rank} received initial input for VR{first_vr}")
        input_tensors[first_vr].append(pre_recv_input)
    else:
        print_all_ranks(f"[PRE-RECV] Rank{pipeline_parallel_rank} is first stage for VR{first_vr}, no pre-recv needed")

    # Step 2: Process warmup microbatches with combined send+recv
    for k in range(num_warmup_microbatches):
        microbatch_id = microbatch_idx[fwd_idx]
        model_chunk_id = get_model_chunk_id(microbatch_id, pipeline_parallel_size)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        if profiler:
            profiler.start_microbatch(microbatch_id, 0, model_chunk_id, 'forward')

        print_all_ranks(f"[WARMUP k={k}] FWD MB{microbatch_id}, VR{model_chunk_id}")

        # Get input tensor (from pre-recv queue or None if first stage)
        is_first = is_vr_first_stage_for_activation(model_chunk_id, pipeline_parallel_rank, pipeline_parallel_size)
        if not is_first:
            input_tensor = input_tensors[model_chunk_id].pop(0)
            print_all_ranks(f"[Rank{pipeline_parallel_rank}] Using pre-received input for MB{microbatch_id}, VR{model_chunk_id}")
        else:
            input_tensor = None
            print_all_ranks(f"[Rank{pipeline_parallel_rank}] First stage for VR{model_chunk_id} - no input needed")

        # Execute forward step
        output_tensor = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch=None,
        )

        # Store input tensor for backward pass (re-add since we popped it)
        input_tensors[model_chunk_id].append(input_tensor)
        output_tensors[model_chunk_id].append(output_tensor)
        # NOTE: Don't deallocate yet - we need output_tensor data for sending!
        print_all_ranks(f"[Rank{pipeline_parallel_rank}] Stored tensors for MB{microbatch_id}, VR{model_chunk_id} (queues: in={len(input_tensors[model_chunk_id])}, out={len(output_tensors[model_chunk_id])})")

        # Determine next microbatch info for combined send+recv
        is_last = is_vr_last_stage_for_activation(model_chunk_id, pipeline_parallel_rank, pipeline_parallel_size)

        # Check if there's a next forward (in warmup or 1F1B)
        has_next_forward = (fwd_idx + 1) < len(microbatch_idx)
        if has_next_forward:
            next_mb_id = microbatch_idx[fwd_idx + 1]
            next_vr = get_model_chunk_id(next_mb_id, pipeline_parallel_size)
            next_is_first = is_vr_first_stage_for_activation(next_vr, pipeline_parallel_rank, pipeline_parallel_size)
        else:
            next_vr = None
            next_is_first = True  # No next forward, so no recv needed

        # Combined send (if not last stage) + recv (if has next and not first stage for next)
        need_send = not is_last
        need_recv = has_next_forward and not next_is_first

        if is_last:
            # Last stage: collect loss, no send
            if output_tensor is not None:
                losses_reduced.append(output_tensor)
            print_all_ranks(f"[Rank{pipeline_parallel_rank}] Last stage for VR{model_chunk_id} - collected loss")

        if need_send and need_recv:
            # Combined send+recv
            print_all_ranks(f"[Rank{pipeline_parallel_rank}] Combined send VR{model_chunk_id} + recv VR{next_vr}")
            if model_chunk_id == 0 and next_vr == 0:
                # VR0→VR0: send_next + recv_prev
                next_input = p2p_communication.chimera_send_next_recv_prev(output_tensor, tensor_shape, config)
            elif model_chunk_id == 0 and next_vr == 1:
                # VR0→VR1: send_next + recv_next
                next_input = p2p_communication.chimera_send_next_recv_next(output_tensor, tensor_shape, config)
            elif model_chunk_id == 1 and next_vr == 0:
                # VR1→VR0: send_prev + recv_prev
                next_input = p2p_communication.chimera_send_prev_recv_prev(output_tensor, tensor_shape, config)
            else:  # model_chunk_id == 1 and next_vr == 1
                # VR1→VR1: send_prev + recv_next
                next_input = p2p_communication.chimera_send_prev_recv_next(output_tensor, tensor_shape, config)
            input_tensors[next_vr].append(next_input)
            print_all_ranks(f"[Rank{pipeline_parallel_rank}] Sent VR{model_chunk_id}, received VR{next_vr} input")

        elif need_send and not need_recv:
            # Send only (no next forward needs recv)
            print_all_ranks(f"[Rank{pipeline_parallel_rank}] Send only VR{model_chunk_id}")
            if model_chunk_id == 0:
                p2p_communication.chimera_send_next_only(output_tensor, config)
            else:
                p2p_communication.chimera_send_prev_only(output_tensor, config)
            print_all_ranks(f"[Rank{pipeline_parallel_rank}] Sent VR{model_chunk_id}")

        elif not need_send and need_recv:
            # Recv only (last stage but has next forward)
            print_all_ranks(f"[Rank{pipeline_parallel_rank}] Recv only VR{next_vr}")
            if next_vr == 0:
                next_input = p2p_communication.chimera_recv_prev_only(tensor_shape, config)
            else:
                next_input = p2p_communication.chimera_recv_next_only(tensor_shape, config)
            input_tensors[next_vr].append(next_input)
            print_all_ranks(f"[Rank{pipeline_parallel_rank}] Received VR{next_vr} input")

        else:
            # No send, no recv (last stage and no more forwards)
            print_all_ranks(f"[Rank{pipeline_parallel_rank}] No comm needed (last stage, no more forwards)")

        # NOW deallocate output_tensor AFTER sending (we only need .grad_fn for backward, not .data)
        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

        if profiler:
            profiler.end_microbatch(microbatch_id, 0, model_chunk_id, 'forward')

        fwd_idx += 1

    print_all_ranks(f"[WARMUP DONE] Processed {num_warmup_microbatches} forwards, fwd_idx now at {fwd_idx}")

    # ===== 1F1B PHASE: Interleaved forward+backward =====
    # Using BitPipe-style combined send+recv to avoid deadlocks
    if not forward_only and num_1f1b_microbatches > 0:
        print_all_ranks(f"=== 1F1B PHASE ({num_1f1b_microbatches} iterations) ===")

        if profiler:
            profiler.record_phase_transition("1f1b_start")

        for k in range(num_1f1b_microbatches):
            # ----- FORWARD PASS -----
            fwd_microbatch_id = microbatch_idx[fwd_idx]
            fwd_model_chunk_id = get_model_chunk_id(fwd_microbatch_id, pipeline_parallel_size)
            parallel_state.set_virtual_pipeline_model_parallel_rank(fwd_model_chunk_id)

            if profiler:
                profiler.start_microbatch(fwd_microbatch_id, 0, fwd_model_chunk_id, 'forward')

            print_all_ranks(f"[1F1B k={k}] FWD MB{fwd_microbatch_id}, VR{fwd_model_chunk_id}")

            # Get input from pre-received queue (populated by warmup or previous 1F1B iteration)
            fwd_is_first = is_vr_first_stage_for_activation(fwd_model_chunk_id, pipeline_parallel_rank, pipeline_parallel_size)
            if not fwd_is_first:
                input_tensor = input_tensors[fwd_model_chunk_id].pop(0)
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Using pre-received input for MB{fwd_microbatch_id}, VR{fwd_model_chunk_id}")
            else:
                input_tensor = None
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] First stage for VR{fwd_model_chunk_id} - no input needed")

            # Execute forward
            output_tensor = forward_step(
                forward_step_func,
                data_iterator[fwd_model_chunk_id],
                model[fwd_model_chunk_id],
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                checkpoint_activations_microbatch=None,
            )

            # Store for backward (re-add input since we popped it)
            input_tensors[fwd_model_chunk_id].append(input_tensor)
            output_tensors[fwd_model_chunk_id].append(output_tensor)
            # NOTE: Don't deallocate yet - we need output_tensor data for sending!
            print_all_ranks(f"[Rank{pipeline_parallel_rank}] Stored tensors for MB{fwd_microbatch_id}, VR{fwd_model_chunk_id}")

            # Determine if we need to send forward output and receive backward gradient
            fwd_is_last = is_vr_last_stage_for_activation(fwd_model_chunk_id, pipeline_parallel_rank, pipeline_parallel_size)

            # Get backward microbatch info
            bwd_microbatch_id = microbatch_idx_b[bwd_idx]
            if bwd_microbatch_id == -1:
                # Unexpected sync marker in 1F1B - still need to deallocate
                print_all_ranks(f"[1F1B k={k}] UNEXPECTED SYNC MARKER - skipping")
                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                bwd_idx += 1
                fwd_idx += 1
                if profiler:
                    profiler.end_microbatch(fwd_microbatch_id, 0, fwd_model_chunk_id, 'forward')
                continue

            bwd_model_chunk_id = get_model_chunk_id(bwd_microbatch_id, pipeline_parallel_size)
            bwd_is_grad_first = is_vr_first_stage_for_gradient(bwd_model_chunk_id, pipeline_parallel_rank, pipeline_parallel_size)

            # Combined: send forward output + recv backward gradient
            need_send_fwd = not fwd_is_last
            need_recv_bwd = not bwd_is_grad_first

            if fwd_is_last:
                # Last stage: collect loss
                if output_tensor is not None:
                    losses_reduced.append(output_tensor)
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Last stage for VR{fwd_model_chunk_id} - collected loss")

            if need_send_fwd and need_recv_bwd:
                # Combined send forward + recv backward
                # VR0: send_next + recv_next, VR1: send_prev + recv_prev
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Combined send_fwd VR{fwd_model_chunk_id} + recv_bwd VR{bwd_model_chunk_id}")
                if fwd_model_chunk_id == 0:
                    # VR0 forward: send to next, recv grad from next
                    output_tensor_grad = p2p_communication.chimera_send_next_recv_next(output_tensor, tensor_shape, config)
                else:
                    # VR1 forward: send to prev, recv grad from prev
                    output_tensor_grad = p2p_communication.chimera_send_prev_recv_prev(output_tensor, tensor_shape, config)
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Sent fwd, received bwd grad")

            elif need_send_fwd and not need_recv_bwd:
                # Send forward only
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Send fwd only VR{fwd_model_chunk_id}")
                if fwd_model_chunk_id == 0:
                    p2p_communication.chimera_send_next_only(output_tensor, config)
                else:
                    p2p_communication.chimera_send_prev_only(output_tensor, config)
                output_tensor_grad = None
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Sent fwd, no bwd grad recv (first stage for grad)")

            elif not need_send_fwd and need_recv_bwd:
                # Recv backward only (last stage for forward)
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Recv bwd only VR{bwd_model_chunk_id}")
                if bwd_model_chunk_id == 0:
                    output_tensor_grad = p2p_communication.chimera_recv_next_only(tensor_shape, config)
                else:
                    output_tensor_grad = p2p_communication.chimera_recv_prev_only(tensor_shape, config)
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Received bwd grad")

            else:
                # No send, no recv
                output_tensor_grad = None
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] No fwd send, no bwd recv (last fwd stage, first grad stage)")

            # NOW deallocate output_tensor AFTER sending (we only need .grad_fn for backward, not .data)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            if profiler:
                profiler.end_microbatch(fwd_microbatch_id, 0, fwd_model_chunk_id, 'forward')

            fwd_idx += 1

            # ----- BACKWARD PASS -----
            parallel_state.set_virtual_pipeline_model_parallel_rank(bwd_model_chunk_id)

            if profiler:
                profiler.start_microbatch(bwd_microbatch_id, 0, bwd_model_chunk_id, 'backward')

            print_all_ranks(f"[1F1B k={k}] BWD MB{bwd_microbatch_id}, VR{bwd_model_chunk_id}")

            # Pop tensors from queues
            bwd_input_tensor = input_tensors[bwd_model_chunk_id].pop(0)
            bwd_output_tensor = output_tensors[bwd_model_chunk_id].pop(0)
            print_all_ranks(f"[Rank{pipeline_parallel_rank}] Popped tensors for MB{bwd_microbatch_id}, VR{bwd_model_chunk_id}")

            # Execute backward (output_tensor_grad was received above)
            input_tensor_grad = backward_step(
                bwd_input_tensor, bwd_output_tensor, output_tensor_grad, model_type, config
            )
            print_all_ranks(f"[Rank{pipeline_parallel_rank}] Backward step done for MB{bwd_microbatch_id}, VR{bwd_model_chunk_id}")

            # Determine if we need to send backward gradient and receive next forward input
            bwd_is_grad_last = is_vr_last_stage_for_gradient(bwd_model_chunk_id, pipeline_parallel_rank, pipeline_parallel_size)

            # Check if there's a next forward
            has_next_forward = (fwd_idx) < len(microbatch_idx)
            if has_next_forward:
                next_fwd_mb_id = microbatch_idx[fwd_idx]
                next_fwd_vr = get_model_chunk_id(next_fwd_mb_id, pipeline_parallel_size)
                next_fwd_is_first = is_vr_first_stage_for_activation(next_fwd_vr, pipeline_parallel_rank, pipeline_parallel_size)
            else:
                next_fwd_vr = None
                next_fwd_is_first = True

            need_send_bwd = not bwd_is_grad_last
            need_recv_fwd = has_next_forward and not next_fwd_is_first

            if need_send_bwd and need_recv_fwd:
                # Combined send backward + recv next forward
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Combined send_bwd VR{bwd_model_chunk_id} + recv_fwd VR{next_fwd_vr}")
                # VR0 backward: send grad to prev, VR1 backward: send grad to next
                # Next forward VR0: recv from prev, VR1: recv from next
                if bwd_model_chunk_id == 0 and next_fwd_vr == 0:
                    # VR0 bwd send_prev + VR0 fwd recv_prev
                    next_input = p2p_communication.chimera_send_prev_recv_prev(input_tensor_grad, tensor_shape, config)
                elif bwd_model_chunk_id == 0 and next_fwd_vr == 1:
                    # VR0 bwd send_prev + VR1 fwd recv_next
                    next_input = p2p_communication.chimera_send_prev_recv_next(input_tensor_grad, tensor_shape, config)
                elif bwd_model_chunk_id == 1 and next_fwd_vr == 0:
                    # VR1 bwd send_next + VR0 fwd recv_prev
                    next_input = p2p_communication.chimera_send_next_recv_prev(input_tensor_grad, tensor_shape, config)
                else:  # bwd_model_chunk_id == 1 and next_fwd_vr == 1
                    # VR1 bwd send_next + VR1 fwd recv_next
                    next_input = p2p_communication.chimera_send_next_recv_next(input_tensor_grad, tensor_shape, config)
                input_tensors[next_fwd_vr].append(next_input)
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Sent bwd grad, received next fwd input for VR{next_fwd_vr}")

            elif need_send_bwd and not need_recv_fwd:
                # Send backward only
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Send bwd only VR{bwd_model_chunk_id}")
                if bwd_model_chunk_id == 0:
                    p2p_communication.chimera_send_prev_only(input_tensor_grad, config)
                else:
                    p2p_communication.chimera_send_next_only(input_tensor_grad, config)
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Sent bwd grad")

            elif not need_send_bwd and need_recv_fwd:
                # Recv next forward only
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Recv fwd only VR{next_fwd_vr}")
                if next_fwd_vr == 0:
                    next_input = p2p_communication.chimera_recv_prev_only(tensor_shape, config)
                else:
                    next_input = p2p_communication.chimera_recv_next_only(tensor_shape, config)
                input_tensors[next_fwd_vr].append(next_input)
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Received next fwd input for VR{next_fwd_vr}")

            else:
                # No send, no recv
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] No bwd send, no fwd recv")

            if profiler:
                profiler.end_microbatch(bwd_microbatch_id, 0, bwd_model_chunk_id, 'backward')

            bwd_idx += 1

        print_all_ranks(f"[1F1B DONE] fwd_idx={fwd_idx}, bwd_idx={bwd_idx}")

    # ===== COOLDOWN PHASE: Backward-only + gradient sync =====
    # Uses combined send+recv operations to avoid deadlocks (BitPipe-style)
    if not forward_only:
        # Calculate remaining backward items (excluding sync markers)
        remaining_backward = sum(1 for mb in microbatch_idx_b[bwd_idx:] if mb != -1)

        print_all_ranks(f"=== COOLDOWN PHASE ({remaining_backward} items remaining) ===")

        if profiler:
            profiler.record_phase_transition("cooldown_start")

        cooldown_k = 0  # Local counter for cooldown phase
        pending_grad_to_send = None  # For combining send of prev MB with recv of current MB
        pending_grad_vr = None  # VR of the pending grad

        while bwd_idx < len(microbatch_idx_b):
            microbatch_id = microbatch_idx_b[bwd_idx]

            # Handle gradient sync markers
            if microbatch_id == -1:
                # First, send any pending gradient before sync
                if pending_grad_to_send is not None:
                    print_all_ranks(f"[Rank{pipeline_parallel_rank}] Sending pending grad before SYNC MARKER (VR{pending_grad_vr})")
                    if pending_grad_vr == 0:
                        p2p_communication.chimera_grad_send_prev_only(pending_grad_to_send, config)
                    else:
                        p2p_communication.chimera_grad_send_next_only(pending_grad_to_send, config)
                    pending_grad_to_send = None
                    pending_grad_vr = None

                print_all_ranks(f"[Rank{pipeline_parallel_rank}] SYNC MARKER at bwd_idx={bwd_idx}, k={cooldown_k}, COOLDOWN phase", include_time=True)
                enable_grad_sync()
                for chunk_id in range(len(model)):
                    if chunk_id not in synchronized_model_chunks:
                        print_all_ranks(f"[Rank{pipeline_parallel_rank}] Allreduce gradients for VR{chunk_id} in k={cooldown_k}, COOLDOWN phase")
                        allreduce_gradients(model[chunk_id])
                        synchronized_model_chunks.add(chunk_id)
                disable_grad_sync()
                bwd_idx += 1
                cooldown_k += 1
                continue

            model_chunk_id = get_model_chunk_id(microbatch_id, pipeline_parallel_size)
            parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

            if profiler:
                profiler.start_microbatch(microbatch_id, 0, model_chunk_id, 'backward')

            print_all_ranks(f"[COOLDOWN k={cooldown_k}] BWD MB{microbatch_id}, VR{model_chunk_id}")

            # Pop tensors from queues
            input_tensor = input_tensors[model_chunk_id].pop(0)
            output_tensor = output_tensors[model_chunk_id].pop(0)
            print_all_ranks(f"[Rank{pipeline_parallel_rank}] Popped tensors for MB{microbatch_id}, VR{model_chunk_id} in k={cooldown_k}, COOLDOWN phase (queues: in={len(input_tensors[model_chunk_id])}, out={len(output_tensors[model_chunk_id])})")

            # Determine if this rank is first/last for gradient flow
            # VR0: gradient first stage = Rank N-1, last stage = Rank 0
            # VR1: gradient first stage = Rank 0, last stage = Rank N-1
            cool_is_grad_first = is_vr_first_stage_for_gradient(model_chunk_id, pipeline_parallel_rank, pipeline_parallel_size)
            cool_is_grad_last = is_vr_last_stage_for_gradient(model_chunk_id, pipeline_parallel_rank, pipeline_parallel_size)

            # Combined send+recv pattern to avoid deadlocks:
            # - Combine sending previous MB's grad with receiving current MB's grad
            need_recv = not cool_is_grad_first
            have_pending_send = pending_grad_to_send is not None

            output_tensor_grad = None

            if have_pending_send and need_recv:
                # Combined: send previous grad + recv current grad
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Combined: send VR{pending_grad_vr} grad + recv VR{model_chunk_id} grad for MB{microbatch_id}")

                if pending_grad_vr == 0 and model_chunk_id == 0:
                    # VR0→VR0: send to prev + recv from next
                    output_tensor_grad = p2p_communication.chimera_grad_send_prev_recv_next(pending_grad_to_send, tensor_shape, config)
                elif pending_grad_vr == 0 and model_chunk_id == 1:
                    # VR0→VR1: send to prev + recv from prev
                    output_tensor_grad = p2p_communication.chimera_grad_send_prev_recv_prev(pending_grad_to_send, tensor_shape, config)
                elif pending_grad_vr == 1 and model_chunk_id == 0:
                    # VR1→VR0: send to next + recv from next
                    output_tensor_grad = p2p_communication.chimera_grad_send_next_recv_next(pending_grad_to_send, tensor_shape, config)
                else:  # pending_grad_vr == 1 and model_chunk_id == 1
                    # VR1→VR1: send to next + recv from prev
                    output_tensor_grad = p2p_communication.chimera_grad_send_next_recv_prev(pending_grad_to_send, tensor_shape, config)

                pending_grad_to_send = None
                pending_grad_vr = None
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Combined send+recv done for MB{microbatch_id}")

            elif have_pending_send and not need_recv:
                # Send only (this rank is first stage for current VR's gradient)
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Send only VR{pending_grad_vr} grad (first stage for VR{model_chunk_id})")
                if pending_grad_vr == 0:
                    p2p_communication.chimera_grad_send_prev_only(pending_grad_to_send, config)
                else:
                    p2p_communication.chimera_grad_send_next_only(pending_grad_to_send, config)
                pending_grad_to_send = None
                pending_grad_vr = None
                output_tensor_grad = None  # First stage, no incoming gradient

            elif not have_pending_send and need_recv:
                # Recv only (no pending send from previous iteration)
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Recv only VR{model_chunk_id} grad for MB{microbatch_id}")
                if model_chunk_id == 0:
                    # VR0: recv from next
                    output_tensor_grad = p2p_communication.chimera_grad_recv_next_only(tensor_shape, config)
                else:
                    # VR1: recv from prev
                    output_tensor_grad = p2p_communication.chimera_grad_recv_prev_only(tensor_shape, config)
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Recv done for MB{microbatch_id}")

            else:
                # No pending send, no recv needed (first stage for gradient)
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Gradient first stage for VR{model_chunk_id} - no recv needed for MB{microbatch_id}")
                output_tensor_grad = None

            # Execute backward step
            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )
            print_all_ranks(f"[Rank{pipeline_parallel_rank}] Backward step done for MB{microbatch_id}, VR{model_chunk_id} in k={cooldown_k}, COOLDOWN phase")

            # If not last stage for gradient, save for combined send with next iteration's recv
            if not cool_is_grad_last:
                pending_grad_to_send = input_tensor_grad
                pending_grad_vr = model_chunk_id
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Saved grad for VR{model_chunk_id} to combine with next recv")
            else:
                print_all_ranks(f"[Rank{pipeline_parallel_rank}] Gradient last stage for VR{model_chunk_id} - no send needed for MB{microbatch_id}")

            if profiler:
                profiler.end_microbatch(microbatch_id, 0, model_chunk_id, 'backward')

            bwd_idx += 1
            cooldown_k += 1

        # Send any remaining pending gradient after loop
        if pending_grad_to_send is not None:
            print_all_ranks(f"[Rank{pipeline_parallel_rank}] Sending final pending grad (VR{pending_grad_vr})")
            if pending_grad_vr == 0:
                p2p_communication.chimera_grad_send_prev_only(pending_grad_to_send, config)
            else:
                p2p_communication.chimera_grad_send_next_only(pending_grad_to_send, config)

        print_all_ranks("[COOLDOWN DONE] Processed all backwards")
    # Final gradient sync
    if not forward_only:
        enable_grad_sync()

    # Record completion and save profile
    if profiler:
        profiler.record_phase_transition("complete")
        profiler.save_profile()

    print_all_ranks("=== Chimera schedule completed ===")

    return losses_reduced
