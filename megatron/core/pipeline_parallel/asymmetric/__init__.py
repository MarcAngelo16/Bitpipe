"""
Asymmetric Pipeline Configuration Management

This module provides utilities for asymmetric layer distribution across pipeline stages,
enabling BitPipe to run efficiently on heterogeneous hardware configurations.

Components:
- config_utils.py: Asymmetric configuration generation and validation
- Layer offset calculation for asymmetric setups
- Device pairing and VR assignment utilities

Migrated from: megatron/bitpipe_asymmetric_utils.py
New location: megatron/core/pipeline_parallel/asymmetric/
"""

from .config_utils import (
    generate_asymmetric_config_from_user_input,
    get_asymmetric_offset,
)

__all__ = [
    'generate_asymmetric_config_from_user_input',
    'get_asymmetric_offset',
]
