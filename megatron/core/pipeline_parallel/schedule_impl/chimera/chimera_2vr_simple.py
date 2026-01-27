"""
Chimera: 2-VR Bidirectional Pipeline with SIMPLE EXECUTION LOOP

This is a simplified version of chimera_2vr.py that replaces the complex
warmup/1F1B/cooldown phase structure with a simple 2-phase loop:

Phase 1: Execute ALL forward passes (follow forward schedule)
Phase 2: Execute ALL backward passes (follow backward schedule)

This matches Chimera's schedule design better than trying to adapt BitPipe's
complex phase structure.
"""

import contextlib
import os
from typing import Iterator, List, Union

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron.core import parallel_state
from megatron.core.utils import get_model_config, get_model_type
from megatron.core.pipeline_parallel.schedules import (
    deallocate_output_tensor,
    forward_step,
    backward_step,
)
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.pipeline_parallel.bitpipe_profiler import get_bitpipe_profiler


def print_all_ranks(message, include_time=False):
    """Print from all ranks for debugging"""
    if os.environ.get('CHIMERA_DEBUG', '0') == '1':
        rank = parallel_state.get_pipeline_model_parallel_rank()
        world = parallel_state.get_pipeline_model_parallel_world_size()
        if include_time:
            import time
            print(f"[Chimera Rank {rank}/World {world}] [{time.time():.2f}] {message}", flush=True)
        else:
            print(f"[Chimera Rank {rank}/World {world}]  {message}", flush=True)


def _profile_p2p_comm(comm_func, comm_type, *args, **kwargs):
    """Wrapper to add profiling around P2P communication"""
    profiler = get_bitpipe_profiler()
    if profiler and profiler.enabled:
        current_rank = parallel_state.get_pipeline_model_parallel_rank()
        if 'send' in comm_type:
            dest_rank = parallel_state.get_pipeline_model_parallel_next_rank() if 'forward' in comm_type else parallel_state.get_pipeline_model_parallel_prev_rank()
            profiler.start_p2p_comm(comm_type, current_rank, dest_rank)
        elif 'recv' in comm_type:
            source_rank = parallel_state.get_pipeline_model_parallel_prev_rank() if 'forward' in comm_type else parallel_state.get_pipeline_model_parallel_next_rank()
            profiler.start_p2p_comm(comm_type, source_rank, current_rank)

    result = comm_func(*args, **kwargs)

    if profiler and profiler.enabled:
        current_rank = parallel_state.get_pipeline_model_parallel_rank()
        if 'send' in comm_type:
            dest_rank = parallel_state.get_pipeline_model_parallel_next_rank() if 'forward' in comm_type else parallel_state.get_pipeline_model_parallel_prev_rank()
            profiler.end_p2p_comm(comm_type, current_rank, dest_rank)
        elif 'recv' in comm_type:
            source_rank = parallel_state.get_pipeline_model_parallel_prev_rank() if 'forward' in comm_type else parallel_state.get_pipeline_model_parallel_next_rank()
            profiler.end_p2p_comm(comm_type, source_rank, current_rank)

    return result


