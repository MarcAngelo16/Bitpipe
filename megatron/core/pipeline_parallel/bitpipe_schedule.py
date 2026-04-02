"""
DEPRECATED: BitPipe Backward Compatibility Shim

This file provides backward compatibility for code that imports from the old location.
New code should import from: megatron.core.pipeline_parallel.schedules.bitpipe

Migration Notice:
-----------------
The BitPipe implementation has been reorganized into a more structured directory layout:

Old location:
    from megatron.core.pipeline_parallel.bitpipe_schedule import forward_backward_pipelining_with_BitPipe

New location:
    from megatron.core.pipeline_parallel.schedule_impl.bitpipe import forward_backward_pipelining_with_bitpipe_4vr

This file will be removed in a future release.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from 'bitpipe_schedule.py' is deprecated and will be removed in a future release. "
    "Please update your imports to:\n"
    "  from megatron.core.pipeline_parallel.schedule_impl.bitpipe import forward_backward_pipelining_with_bitpipe_4vr\n"
    "Or use the backward-compatible alias 'forward_backward_pipelining_with_BitPipe' (note the capital letters).",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location
from megatron.core.pipeline_parallel.schedule_impl.bitpipe.bitpipe_4vr import (
    forward_backward_pipelining_with_bitpipe_4vr,
    _profile_p2p_comm,  # Re-export helper functions for compatibility
    print_all_ranks,
)

# Provide backward-compatible alias with original naming
forward_backward_pipelining_with_BitPipe = forward_backward_pipelining_with_bitpipe_4vr

# Export for compatibility
__all__ = [
    'forward_backward_pipelining_with_BitPipe',  # Old name (deprecated)
    'forward_backward_pipelining_with_bitpipe_4vr',  # New name
    '_profile_p2p_comm',
    'print_all_ranks',
]
