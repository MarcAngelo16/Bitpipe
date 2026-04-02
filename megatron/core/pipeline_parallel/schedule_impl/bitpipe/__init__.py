"""
BitPipe: 4-VR Bidirectional Interleaved Pipeline Parallelism

BitPipe implements a bidirectional interleaved pipeline with 4 virtual ranks (VR)
per device, forming a V-shaped execution pattern for optimal pipeline utilization.

Variants:
- bitpipe_4vr.py: Symmetric (uniform layer distribution)
- bitpipe_4vr_asymmetric.py: Asymmetric (variable layer distribution)
"""

from .bitpipe_4vr import forward_backward_pipelining_with_bitpipe_4vr

__all__ = ['forward_backward_pipelining_with_bitpipe_4vr']
