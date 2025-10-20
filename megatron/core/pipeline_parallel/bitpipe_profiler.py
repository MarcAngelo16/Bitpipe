"""
BitPipe Profiler - Comprehensive profiling for BitPipe pipeline parallelism
"""

import time
import torch
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

@dataclass
class MicrobatchEvent:
    """Profile data for a single microbatch execution"""
    microbatch_id: int
    pipeline_id: int  # 0 or 1 (bidirectional pipeline)
    model_chunk_id: int
    phase: str  # 'forward' or 'backward'
    start_time: float
    end_time: float
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
    start_time: float
    end_time: float
    microbatch_id: int = -1
    
    @property
    def duration(self):
        return self.end_time - self.start_time


@dataclass
class PhaseTransition:
    """Records phase transitions in the pipeline"""
    phase_name: str  # 'warmup', 'steady_state', 'cooldown'
    timestamp: float
    memory_allocated: int
    memory_reserved: int


class BitPipeProfiler:
    """Main profiler class for BitPipe schedule"""
    
    def __init__(self, rank: int, world_size: int, enabled: bool = True, schedule_type: str = "bitpipe"):
        self.rank = rank
        self.world_size = world_size
        self.enabled = enabled
        self.schedule_type = schedule_type
        
        # Profile storage
        self.microbatch_events: List[MicrobatchEvent] = []
        self.p2p_events: List[P2PEvent] = []
        self.phase_transitions: List[PhaseTransition] = []
        
        # Timing state
        self.global_start_time = None
        self.active_timers: Dict[str, float] = {}
        
        # Current execution context
        self.current_pipeline_id = -1
        self.current_microbatch_id = -1
        self.current_model_chunk_id = -1
        
    def start_profiling(self):
        """Start global profiling timer"""
        if not self.enabled:
            return
            
        self.global_start_time = time.time()
        self._record_memory_snapshot("start")
        
    def _get_relative_time(self) -> float:
        """Get time relative to profiling start"""
        if self.global_start_time is None:
            return 0.0
        return time.time() - self.global_start_time
        
    def _record_memory_snapshot(self, label: str) -> Tuple[int, int]:
        """Record current memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            return allocated, reserved
        return 0, 0
        
    def record_phase_transition(self, phase_name: str):
        """Record transition between major phases"""
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
        
    def start_microbatch(self, microbatch_id: int, pipeline_id: int, 
                        model_chunk_id: int, phase: str):
        """Record start of microbatch processing"""
        if not self.enabled:
            return
            
        # Update context
        self.current_microbatch_id = microbatch_id
        self.current_pipeline_id = pipeline_id
        self.current_model_chunk_id = model_chunk_id
        
        # Start timer
        timer_key = f"mb_{microbatch_id}_{phase}_{model_chunk_id}"
        self.active_timers[timer_key] = time.time()
        
    def end_microbatch(self, microbatch_id: int, pipeline_id: int,
                      model_chunk_id: int, phase: str):
        """Record end of microbatch processing"""
        if not self.enabled:
            return
            
        timer_key = f"mb_{microbatch_id}_{phase}_{model_chunk_id}"
        if timer_key not in self.active_timers:
            return
            
        start_time = self.active_timers[timer_key]
        end_time = time.time()
        
        event = MicrobatchEvent(
            microbatch_id=microbatch_id,
            pipeline_id=pipeline_id,
            model_chunk_id=model_chunk_id,
            phase=phase,
            start_time=start_time - self.global_start_time,
            end_time=end_time - self.global_start_time,
            rank=self.rank
        )
        
        self.microbatch_events.append(event)
        del self.active_timers[timer_key]
        
    def start_p2p_comm(self, comm_type: str, source_rank: int, dest_rank: int):
        """Record start of P2P communication"""
        if not self.enabled:
            return
            
        timer_key = f"p2p_{comm_type}_{source_rank}_{dest_rank}_{self.current_microbatch_id}"
        self.active_timers[timer_key] = time.time()
        
    def end_p2p_comm(self, comm_type: str, source_rank: int, dest_rank: int):
        """Record end of P2P communication"""
        if not self.enabled:
            return
            
        timer_key = f"p2p_{comm_type}_{source_rank}_{dest_rank}_{self.current_microbatch_id}"
        if timer_key not in self.active_timers:
            return
            
        start_time = self.active_timers[timer_key]
        end_time = time.time()
        
        event = P2PEvent(
            comm_type=comm_type,
            source_rank=source_rank,
            dest_rank=dest_rank,
            start_time=start_time - self.global_start_time,
            end_time=end_time - self.global_start_time,
            microbatch_id=self.current_microbatch_id
        )
        
        self.p2p_events.append(event)
        del self.active_timers[timer_key]
        
    def save_profile(self, output_dir: str = "./asymmetric_bitpipe/profiles/raw"):
        """Save profiling data to JSON file"""
        if not self.enabled or not self.microbatch_events:
            return
            
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Calculate summary statistics
        total_forward_time = sum(e.duration for e in self.microbatch_events if e.phase == 'forward')
        total_backward_time = sum(e.duration for e in self.microbatch_events if e.phase == 'backward')
        total_p2p_time = sum(e.duration for e in self.p2p_events)
        
        # Build profile data
        profile_data = {
            'metadata': {
                'rank': self.rank,
                'world_size': self.world_size,
                'timestamp': timestamp,
                'total_time': self._get_relative_time(),
                'schedule_type': self.schedule_type
            },
            'summary': {
                'total_forward_time': total_forward_time,
                'total_backward_time': total_backward_time,
                'total_p2p_time': total_p2p_time,
                'num_microbatches': len(set(e.microbatch_id for e in self.microbatch_events)),
                'num_forward_passes': len([e for e in self.microbatch_events if e.phase == 'forward']),
                'num_backward_passes': len([e for e in self.microbatch_events if e.phase == 'backward'])
            },
            'microbatch_events': [asdict(e) for e in self.microbatch_events],
            'p2p_events': [asdict(e) for e in self.p2p_events],
            'phase_transitions': [asdict(t) for t in self.phase_transitions]
        }
        
        # Get iteration context for filename
        try:
            from megatron.core.iteration_context import get_iteration_context
            iteration, iter_type = get_iteration_context()
            filename = f"{self.schedule_type}_profile_rank{self.rank}_{iter_type}_iter{iteration}_{timestamp}.json"
        except ImportError:
            # Fallback if iteration_context is not available
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


def initialize_bitpipe_profiler(rank: int, world_size: int, enabled: bool = True, schedule_type: str = "bitpipe") -> BitPipeProfiler:
    """Initialize the global BitPipe profiler"""
    global _bitpipe_profiler
    _bitpipe_profiler = BitPipeProfiler(rank, world_size, enabled, schedule_type)
    return _bitpipe_profiler