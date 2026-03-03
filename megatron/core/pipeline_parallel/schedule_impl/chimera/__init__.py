"""
Chimera: Simplified 2-VR Bidirectional Pipeline Parallelism

This package implements the Chimera scheduler, a simplified alternative to BitPipe
with 2 virtual ranks per device instead of 4.
"""

from .chimera_2vr import (
    forward_backward_pipelining_with_chimera_2vr,
    get_chimera_offset,
    get_chimera_microbatch_idx,
    get_chimera_bkmicrobatch_idx,
)

__all__ = [
    'forward_backward_pipelining_with_chimera_2vr',
    'get_chimera_offset',
    'get_chimera_microbatch_idx',
    'get_chimera_bkmicrobatch_idx',
]
