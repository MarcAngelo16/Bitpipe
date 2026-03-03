#!/usr/bin/env python3
"""
BitPipe Asymmetric Configuration Utilities

This module provides utilities for asymmetric layer distribution in BitPipe,
enabling variable layer counts per device for heterogeneous hardware setups.

Key Features:
- Asymmetric layer distribution across pipeline stages
- Device pairing with VR swapping for bidirectional symmetry
- Configuration validation and auto-generation

Migrated from: megatron/bitpipe_asymmetric_utils.py
New location: megatron/core/pipeline_parallel/asymmetric/config_utils.py
"""


def generate_asymmetric_config_from_user_input(first_half_distributions, total_layers, pipeline_size):
    """
    Generate full asymmetric configuration from user input for first half of devices

    This function takes layer distributions for the first N/2 devices and automatically
    generates the paired second half with proper VR swapping to maintain bidirectional
    pipeline symmetry.

    Args:
        first_half_distributions: List of distributions for first half devices
            For 4 devices: [[2,4,2,3], [2,1,1,1]] - distributions for devices 0 and 1
            For 8 devices: [[2,4,2,3], [2,1,1,1], [3,2,1,2], [1,3,2,1]] - distributions for devices 0-3
        total_layers: Total number of layers in the model
        pipeline_size: Number of pipeline parallel devices

    Returns:
        Full configuration for all devices with automatic pairing

    Raises:
        ValueError: If pipeline size is odd, layer counts don't match, or pairing constraint fails

    Example:
        >>> # 4 devices, 16 layers total
        >>> first_half = [[2,4,2,3], [2,1,1,1]]  # Devices 0-1
        >>> config = generate_asymmetric_config_from_user_input(first_half, 16, 4)
        >>> # Returns: [[2,4,2,3], [2,1,1,1], [1,2,1,1], [4,2,3,2]]
        >>> #          Device 0    Device 1   Device 2   Device 3 (auto-generated with VR swap)
    """
    # Validate pipeline size is even
    if pipeline_size % 2 != 0:
        raise ValueError(f"Pipeline size must be even, got {pipeline_size}")

    num_first_half_devices = pipeline_size // 2

    # Validate input
    if len(first_half_distributions) != num_first_half_devices:
        raise ValueError(f"Expected {num_first_half_devices} device distributions, got {len(first_half_distributions)}")

    for i, dist in enumerate(first_half_distributions):
        if len(dist) != 4:
            raise ValueError(f"Device {i} distribution must have exactly 4 VR entries, got {len(dist)}")

    total_input_layers = sum(sum(dist) for dist in first_half_distributions)
    if total_input_layers != total_layers:
        raise ValueError(f"Total layers from input {total_input_layers} != expected {total_layers}")

    # Generate paired devices with VR swapping: [VR0,VR1,VR2,VR3] → [VR1,VR0,VR3,VR2]
    full_config = []

    # Add first half devices
    for dist in first_half_distributions:
        full_config.append(dist[:])

    # Add second half devices (paired and VR-swapped)
    for i in range(num_first_half_devices):
        paired_device_idx = num_first_half_devices - 1 - i  # Last device pairs with first, etc.
        original_dist = first_half_distributions[paired_device_idx]
        # VR swap: [VR0,VR1,VR2,VR3] → [VR1,VR0,VR3,VR2]
        swapped_dist = [original_dist[1], original_dist[0], original_dist[3], original_dist[2]]
        full_config.append(swapped_dist)

    # Validate pairing constraint
    for i in range(pipeline_size):
        pair_idx = pipeline_size - 1 - i  # First device pairs with last, etc.
        if sum(full_config[i]) != sum(full_config[pair_idx]):
            raise ValueError(f"Device {i} and {pair_idx} must have same total layers")

    return full_config


def get_asymmetric_offset(pipeline_rank, vp_rank, asymmetric_device_config):
    """
    Scalable asymmetric offset calculation for any even number of devices

    Calculates the starting layer offset and number of layers for a given device-VR combination
    in an asymmetric BitPipe configuration. This function implements the V-shaped layer assignment
    pattern that enables bidirectional pipeline execution.

    Args:
        pipeline_rank: Current pipeline device rank (0 to N-1)
        vp_rank: Virtual pipeline rank (0-3)
            - VR0: First quarter of forward pipeline
            - VR1: Second quarter of forward pipeline (paired with VR0)
            - VR2: First quarter of backward pipeline
            - VR3: Second quarter of backward pipeline (paired with VR2)
        asymmetric_device_config: List of device configurations
            For 4 devices: [[2,4,2,3], [2,1,1,1], [1,2,1,1], [4,2,3,2]]
            For 8 devices: [[...], [...], [...], [...], [...], [...], [...], [...]]
            Each sub-array represents [VR0_layers, VR1_layers, VR2_layers, VR3_layers] for each device

    Returns:
        tuple: (offset, num_layers_for_this_device_vr)
            - offset: Starting layer index for this device-VR combination
            - num_layers_for_this_device_vr: Number of layers assigned to this device-VR

    Layer Assignment Pattern (scales to any even number of devices):
        - VR0: Assigns layers sequentially in forward device order (0→1→...→N-1)
        - VR1: Assigns same layers as VR0 but with device pairing (0↔N-1, 1↔N-2, ...)
        - VR2: Assigns remaining layers sequentially in reverse device order (N-1→...→1→0)
        - VR3: Assigns same layers as VR2 but with device pairing (0↔N-1, 1↔N-2, ...)

    Example:
        >>> config = [[2,4,2,3], [2,1,1,1], [1,2,1,1], [4,2,3,2]]  # 4 devices
        >>> offset, num_layers = get_asymmetric_offset(0, 0, config)
        >>> # Device 0, VR0: offset=0, num_layers=2 (layers 0-1)
        >>> offset, num_layers = get_asymmetric_offset(0, 1, config)
        >>> # Device 0, VR1: offset=2, num_layers=4 (layers 2-5, paired from device 3's VR0)
    """
    # Get number of layers this device-VR combination should have
    num_layers_for_this_device_vr = asymmetric_device_config[pipeline_rank][vp_rank]

    if num_layers_for_this_device_vr == 0:
        return 0, 0  # No layers assigned

    num_devices = len(asymmetric_device_config)

    # Calculate total layers for VR0 (to determine split point between first and second half)
    total_vr0_layers = sum(asymmetric_device_config[dev][0] for dev in range(num_devices))

    offset = 0

    if vp_rank == 0:
        # VR0: forward order (device 0→1→2→...→N-1)
        # Each device gets layers sequentially
        for dev in range(pipeline_rank):
            offset += asymmetric_device_config[dev][0]

    elif vp_rank == 1:
        # VR1: device pairing with VR0 layers
        # Device i gets what device (N-1-i) got in VR0
        paired_device = num_devices - 1 - pipeline_rank  # 0↔N-1, 1↔N-2, etc.

        # Calculate offset as if we're the paired device in VR0
        for dev in range(paired_device):
            offset += asymmetric_device_config[dev][0]

    elif vp_rank == 2:
        # VR2: reverse order (device N-1→...→1→0) for second half of layers
        # Start after all VR0 layers
        offset = total_vr0_layers

        # Add layers from devices after current device in reverse order
        for dev in range(num_devices - 1, pipeline_rank, -1):
            offset += asymmetric_device_config[dev][2]

    elif vp_rank == 3:
        # VR3: device pairing with VR2 layers
        # Device i gets what device (N-1-i) got in VR2
        paired_device = num_devices - 1 - pipeline_rank  # 0↔N-1, 1↔N-2, etc.

        # Start after all VR0 layers (same as VR2)
        offset = total_vr0_layers

        # Add layers from devices after paired_device in reverse order (VR2 perspective)
        for dev in range(num_devices - 1, paired_device, -1):
            offset += asymmetric_device_config[dev][2]

    return offset, num_layers_for_this_device_vr


__all__ = [
    'generate_asymmetric_config_from_user_input',
    'get_asymmetric_offset',
]
