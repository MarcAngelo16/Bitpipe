"""
BitPipe Profiler - Comprehensive profiling for BitPipe pipeline parallelism
Uses torch.cuda.Event for GPU-accurate timing with deferred synchronization.
"""

import time
import torch
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict

@dataclass
class MicrobatchEvent:
    """Profile data for a single microbatch execution"""
    microbatch_id: int
    pipeline_id: int  # 0 or 1 (bidirectional pipeline)
    model_chunk_id: int
    phase: str  # 'forward' or 'backward'
    start_time: float   # wall-clock offset from global_start (seconds) — for ordering
    end_time: float     # start_time + cuda_duration — filled in at save_profile()
    rank: int

    @property
    def duration(self):
        return self.end_time - self.start_time


@dataclass
class P2PEvent:
    """Profile data for P2P communication"""
    comm_type: str  # 'send_forward', 'recv_forward', 'send_backward', 'recv_backward'
    source_rank: int
    dest_rank: int
    start_time: float   # wall-clock offset — for ordering
    end_time: float     # start_time + cuda_duration — filled in at save_profile()
    microbatch_id: int = -1

    @property
    def duration(self):
        return self.end_time - self.start_time


@dataclass
class SyncEvent:
    """Profile data for a BD allreduce sync block (triggered by -1 marker)"""
    sync_index: int           # monotonically increasing index
    phase: str                # 'cooldown' (or 'steady_state' in bitpipe)
    wall_clock_start: float   # wall-clock offset for timeline ordering
    wall_clock_end: float     # filled in at save_profile()
    chunk_durations_ms: list  # [{'chunk_id': X, 'duration_ms': Y}, ...] filled at save_profile()
    total_duration_ms: float  # entire sync block, filled at save_profile()
    rank: int
    num_chunks: int


@dataclass
class PhaseTransition:
    """Records phase transitions in the pipeline"""
    phase_name: str  # 'warmup', 'steady_state', 'cooldown'
    timestamp: float
    memory_allocated: int
    memory_reserved: int


class BitPipeProfiler:
    """Main profiler class for BitPipe/Chimera schedule"""

    def __init__(self, rank: int, world_size: int, enabled: bool = True, schedule_type: str = "bitpipe"):
        self.rank = rank
        self.world_size = world_size
        self.enabled = enabled
        self.schedule_type = schedule_type

        # Profile storage
        self.microbatch_events: List[MicrobatchEvent] = []
        self.p2p_events: List[P2PEvent] = []
        self.sync_events: List[SyncEvent] = []
        self.phase_transitions: List[PhaseTransition] = []

        # Wall-clock anchor
        self.global_start_time: Optional[float] = None

        # CUDA event storage: maps timer_key -> (cuda_start_event, wall_clock_start_offset)
        self.active_cuda_events: Dict[str, Tuple[torch.cuda.Event, float]] = {}

        # Pending CUDA event pairs for deferred elapsed_time() resolution
        # key = index into the corresponding events list
        self._pending_mb_cuda: Dict[int, Tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self._pending_p2p_cuda: Dict[int, Tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self._pending_sync_cuda: Dict[int, Tuple[torch.cuda.Event, torch.cuda.Event, list]] = {}

        # Transient per-sync-block state (chunk cuda pairs accumulate here)
        self._pending_sync_meta: Dict[int, dict] = {}

        # Sync index counter
        self._sync_index_counter: int = 0

        # Current execution context (for P2P event association)
        self.current_pipeline_id: int = -1
        self.current_microbatch_id: int = -1
        self.current_model_chunk_id: int = -1

    def start_profiling(self):
        """Start global profiling timer.

        Inserts a barrier before recording t=0 so that all ranks share the same
        time reference. This makes cross-rank start_time comparisons valid.
        Without this, each rank's global_start_time is set independently and
        timestamps cannot be compared across ranks.
        """
        if not self.enabled:
            return
        # Synchronize all ranks so t=0 is captured simultaneously
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.global_start_time = time.time()
        self._record_memory_snapshot("start")

    def _get_relative_time(self) -> float:
        """Get wall-clock time relative to profiling start"""
        if self.global_start_time is None:
            return 0.0
        return time.time() - self.global_start_time

    def _record_memory_snapshot(self, label: str) -> Tuple[int, int]:
        """Record current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            return allocated, reserved
        return 0, 0

    def record_phase_transition(self, phase_name: str):
        """Record transition between major phases (wall-clock, coarse-grained)"""
        if not self.enabled:
            return
        allocated, reserved = self._record_memory_snapshot(phase_name)
        transition = PhaseTransition(
            phase_name=phase_name,
            timestamp=self._get_relative_time(),
            memory_allocated=allocated,
            memory_reserved=reserved
        )
        self.phase_transitions.append(transition)

    # -------------------------------------------------------------------------
    # Microbatch timing (CUDA events, deferred resolution)
    # -------------------------------------------------------------------------

    def start_microbatch(self, microbatch_id: int, pipeline_id: int,
                         model_chunk_id: int, phase: str):
        """Record start of microbatch processing using a CUDA event"""
        if not self.enabled:
            return
        self.current_microbatch_id = microbatch_id
        self.current_pipeline_id = pipeline_id
        self.current_model_chunk_id = model_chunk_id

        timer_key = f"mb_{microbatch_id}_{phase}_{model_chunk_id}"
        cuda_event = torch.cuda.Event(enable_timing=True)
        cuda_event.record()
        wall_offset = self._get_relative_time()
        self.active_cuda_events[timer_key] = (cuda_event, wall_offset)

    def end_microbatch(self, microbatch_id: int, pipeline_id: int,
                       model_chunk_id: int, phase: str):
        """Record end of microbatch processing using a CUDA event (no sync in hot path)"""
        if not self.enabled:
            return
        timer_key = f"mb_{microbatch_id}_{phase}_{model_chunk_id}"
        if timer_key not in self.active_cuda_events:
            return

        cuda_start, wall_start = self.active_cuda_events.pop(timer_key)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_end.record()

        event = MicrobatchEvent(
            microbatch_id=microbatch_id,
            pipeline_id=pipeline_id,
            model_chunk_id=model_chunk_id,
            phase=phase,
            start_time=wall_start,
            end_time=wall_start,  # placeholder; resolved in save_profile()
            rank=self.rank
        )
        idx = len(self.microbatch_events)
        self.microbatch_events.append(event)
        self._pending_mb_cuda[idx] = (cuda_start, cuda_end)

    # -------------------------------------------------------------------------
    # P2P communication timing (CUDA events, deferred resolution)
    # -------------------------------------------------------------------------

    def start_p2p_comm(self, comm_type: str, source_rank: int, dest_rank: int):
        """Record start of P2P communication using a CUDA event"""
        if not self.enabled:
            return
        timer_key = f"p2p_{comm_type}_{source_rank}_{dest_rank}_{self.current_microbatch_id}"
        cuda_event = torch.cuda.Event(enable_timing=True)
        cuda_event.record()
        wall_offset = self._get_relative_time()
        self.active_cuda_events[timer_key] = (cuda_event, wall_offset)

    def end_p2p_comm(self, comm_type: str, source_rank: int, dest_rank: int):
        """Record end of P2P communication using a CUDA event (no sync in hot path)"""
        if not self.enabled:
            return
        timer_key = f"p2p_{comm_type}_{source_rank}_{dest_rank}_{self.current_microbatch_id}"
        if timer_key not in self.active_cuda_events:
            return

        cuda_start, wall_start = self.active_cuda_events.pop(timer_key)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_end.record()

        event = P2PEvent(
            comm_type=comm_type,
            source_rank=source_rank,
            dest_rank=dest_rank,
            start_time=wall_start,
            end_time=wall_start,  # placeholder; resolved in save_profile()
            microbatch_id=self.current_microbatch_id
        )
        idx = len(self.p2p_events)
        self.p2p_events.append(event)
        self._pending_p2p_cuda[idx] = (cuda_start, cuda_end)

    # -------------------------------------------------------------------------
    # Sync block timing (-1 marker: BD allreduce)
    # -------------------------------------------------------------------------

    def next_sync_index(self) -> int:
        """Return a monotonically increasing sync index"""
        idx = self._sync_index_counter
        self._sync_index_counter += 1
        return idx

    def start_sync_block(self, sync_index: int, phase: str):
        """Record start of a BD allreduce sync block (called before first chunk allreduce)"""
        if not self.enabled:
            return
        cuda_event = torch.cuda.Event(enable_timing=True)
        cuda_event.record()
        wall_offset = self._get_relative_time()
        key = f"sync_block_{sync_index}"
        self.active_cuda_events[key] = (cuda_event, wall_offset)
        self._pending_sync_meta[sync_index] = {
            'phase': phase,
            'wall_start': wall_offset,
            'chunk_cuda_pairs': [],  # filled by end_sync_chunk
        }

    def end_sync_block(self, sync_index: int, num_chunks: int):
        """Record end of a BD allreduce sync block (called after last chunk allreduce)"""
        if not self.enabled:
            return
        key = f"sync_block_{sync_index}"
        if key not in self.active_cuda_events:
            return

        cuda_start, wall_start = self.active_cuda_events.pop(key)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_end.record()

        meta = self._pending_sync_meta.get(sync_index, {})
        event = SyncEvent(
            sync_index=sync_index,
            phase=meta.get('phase', 'unknown'),
            wall_clock_start=wall_start,
            wall_clock_end=wall_start,  # placeholder; resolved in save_profile()
            chunk_durations_ms=[],       # placeholder; resolved in save_profile()
            total_duration_ms=0.0,       # placeholder; resolved in save_profile()
            rank=self.rank,
            num_chunks=num_chunks
        )
        idx = len(self.sync_events)
        self.sync_events.append(event)
        self._pending_sync_cuda[idx] = (
            cuda_start,
            cuda_end,
            meta.get('chunk_cuda_pairs', [])
        )

    def start_sync_chunk(self, sync_index: int, chunk_id: int):
        """Record start of a per-chunk allreduce within a sync block"""
        if not self.enabled:
            return
        cuda_event = torch.cuda.Event(enable_timing=True)
        cuda_event.record()
        key = f"sync_chunk_{sync_index}_{chunk_id}"
        self.active_cuda_events[key] = (cuda_event, 0.0)

    def end_sync_chunk(self, sync_index: int, chunk_id: int):
        """Record end of a per-chunk allreduce (no sync in hot path)"""
        if not self.enabled:
            return
        key = f"sync_chunk_{sync_index}_{chunk_id}"
        if key not in self.active_cuda_events:
            return
        cuda_start, _ = self.active_cuda_events.pop(key)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_end.record()
        if sync_index in self._pending_sync_meta:
            self._pending_sync_meta[sync_index]['chunk_cuda_pairs'].append(
                (chunk_id, cuda_start, cuda_end)
            )

    # -------------------------------------------------------------------------
    # Save profile — single synchronize + deferred elapsed_time resolution
    # -------------------------------------------------------------------------

    def save_profile(self, output_dir: str = "./asymmetric_bitpipe/profiles/raw"):
        """Resolve all CUDA event timings and save profiling data to JSON.

        ONE torch.cuda.synchronize() call outside the hot path flushes all
        pending GPU work, after which all elapsed_time() calls return immediately.
        """
        if not self.enabled or not self.microbatch_events:
            return

        # Single sync point — outside the training hot path
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Resolve microbatch events
        for idx, (cuda_start, cuda_end) in self._pending_mb_cuda.items():
            duration_ms = cuda_start.elapsed_time(cuda_end)  # milliseconds
            event = self.microbatch_events[idx]
            event.end_time = event.start_time + duration_ms / 1000.0

        # Resolve P2P events
        for idx, (cuda_start, cuda_end) in self._pending_p2p_cuda.items():
            duration_ms = cuda_start.elapsed_time(cuda_end)
            event = self.p2p_events[idx]
            event.end_time = event.start_time + duration_ms / 1000.0

        # Resolve sync events
        for idx, (cuda_start, cuda_end, chunk_pairs) in self._pending_sync_cuda.items():
            total_ms = cuda_start.elapsed_time(cuda_end)
            event = self.sync_events[idx]
            event.total_duration_ms = total_ms
            event.wall_clock_end = event.wall_clock_start + total_ms / 1000.0

            chunk_durations = []
            for (chunk_id, c_start, c_end) in chunk_pairs:
                chunk_ms = c_start.elapsed_time(c_end)
                chunk_durations.append({'chunk_id': chunk_id, 'duration_ms': chunk_ms})
            event.chunk_durations_ms = chunk_durations

        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Summary statistics (durations in seconds for microbatch/p2p, ms for sync)
        total_forward_time = sum(e.duration for e in self.microbatch_events if e.phase == 'forward')
        total_backward_time = sum(e.duration for e in self.microbatch_events if e.phase == 'backward')
        total_p2p_time = sum(e.duration for e in self.p2p_events)
        total_sync_time_ms = sum(e.total_duration_ms for e in self.sync_events)

        profile_data = {
            'metadata': {
                'rank': self.rank,
                'world_size': self.world_size,
                'timestamp': timestamp,
                'total_time': self._get_relative_time(),
                'schedule_type': self.schedule_type,
                'timing_method': 'cuda_event',
                'synchronized': torch.distributed.is_initialized(),
            },
            'summary': {
                'total_forward_time': total_forward_time,
                'total_backward_time': total_backward_time,
                'total_p2p_time': total_p2p_time,
                'total_sync_time_ms': total_sync_time_ms,
                'num_microbatches': len(set(e.microbatch_id for e in self.microbatch_events)),
                'num_forward_passes': len([e for e in self.microbatch_events if e.phase == 'forward']),
                'num_backward_passes': len([e for e in self.microbatch_events if e.phase == 'backward']),
                'num_sync_events': len(self.sync_events),
            },
            'microbatch_events': [asdict(e) for e in self.microbatch_events],
            'p2p_events': [asdict(e) for e in self.p2p_events],
            'sync_events': [asdict(e) for e in self.sync_events],
            'phase_transitions': [asdict(t) for t in self.phase_transitions]
        }

        # Get iteration context for filename
        try:
            from megatron.core.iteration_context import get_iteration_context
            iteration, iter_type = get_iteration_context()
            filename = f"{self.schedule_type}_profile_rank{self.rank}_{iter_type}_iter{iteration}_{timestamp}.json"
        except ImportError:
            filename = f"{self.schedule_type}_profile_rank{self.rank}_{timestamp}.json"

        output_file = os.path.join(output_dir, filename)
        with open(output_file, 'w') as f:
            json.dump(profile_data, f, indent=2)

        print(f"[Rank {self.rank}] BitPipe profile saved to {output_file}")


# Global profiler instance
_bitpipe_profiler: Optional[BitPipeProfiler] = None


def get_bitpipe_profiler() -> Optional[BitPipeProfiler]:
    """Get the global BitPipe profiler instance"""
    return _bitpipe_profiler


def initialize_bitpipe_profiler(rank: int, world_size: int, enabled: bool = True,
                                 schedule_type: str = "bitpipe") -> BitPipeProfiler:
    """Initialize the global BitPipe profiler"""
    global _bitpipe_profiler
    _bitpipe_profiler = BitPipeProfiler(rank, world_size, enabled, schedule_type)
    return _bitpipe_profiler
