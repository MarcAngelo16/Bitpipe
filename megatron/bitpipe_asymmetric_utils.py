#!/usr/bin/env python3
"""
DEPRECATED: BitPipe Asymmetric Utilities Backward Compatibility Shim

This file provides backward compatibility for code that imports from the old location.
New code should import from: megatron.core.pipeline_parallel.asymmetric

Migration Notice:
-----------------
The asymmetric utilities have been reorganized into the core pipeline parallel structure:

Old location:
    from megatron.bitpipe_asymmetric_utils import get_asymmetric_offset, generate_asymmetric_config_from_user_input

New location:
    from megatron.core.pipeline_parallel.asymmetric import get_asymmetric_offset, generate_asymmetric_config_from_user_input

This file will be removed in a future release.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from 'bitpipe_asymmetric_utils.py' is deprecated and will be removed in a future release. "
    "Please update your imports to:\n"
    "  from megatron.core.pipeline_parallel.asymmetric import get_asymmetric_offset, generate_asymmetric_config_from_user_input",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location
from megatron.core.pipeline_parallel.asymmetric import (
    get_asymmetric_offset,
    generate_asymmetric_config_from_user_input,
)

# Export for compatibility
__all__ = [
    'get_asymmetric_offset',
    'generate_asymmetric_config_from_user_input',
]
