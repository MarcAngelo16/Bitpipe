#!/usr/bin/env python3
"""
BitPipe Offset Function Simulation
Compares symmetric vs asymmetric layer distribution
"""

def get_symmetric_offset(pipeline_rank, vp_rank, total_layers, pipeline_size, vp_size):
    """
    Current BitPipe symmetric offset calculation (from transformer.py - ACTUAL BitPipe formula)
    """
    # This is the actual BitPipe V-shaped formula
    num_layers_per_chunk = total_layers // (pipeline_size * 2)  # 16/(4*2) = 2
    
    # BitPipe V-shaped calculation
    vp_idx = vp_rank if vp_rank < 2 else (vp_rank - 1) % 2
    
    offset = (
        pipeline_rank * num_layers_per_chunk
        + vp_idx * (pipeline_size - 1 - 2 * pipeline_rank) * num_layers_per_chunk  
        + (vp_rank // 2) * (total_layers // 2)
    )
    
    return offset

def get_layers_from_offset(offset, num_layers_per_chunk):
    """
    Convert offset to actual layer numbers (1-indexed like in the logs)
    """
    return list(range(offset + 1, offset + 1 + num_layers_per_chunk))

def simulate_symmetric_bitpipe(total_layers, pipeline_size, vp_size):
    """
    Simulate current symmetric BitPipe layer distribution
    """
    print(f"=== SYMMETRIC BITPIPE SIMULATION ===")
    print(f"Total layers: {total_layers}")
    print(f"Pipeline size: {pipeline_size}")
    print(f"Virtual pipeline size: {vp_size}")
    print()
    
    num_layers_per_chunk = total_layers // (pipeline_size * 2)  # BitPipe formula
    
    print(f"Layers per chunk: {num_layers_per_chunk}")
    print()
    
    device_layers = {}
    
    for pipeline_rank in range(pipeline_size):
        device_layers[pipeline_rank] = {}
        print(f"Device {pipeline_rank}:")
        
        for vp_rank in range(vp_size):
            offset = get_symmetric_offset(pipeline_rank, vp_rank, total_layers, pipeline_size, vp_size)
            layers = get_layers_from_offset(offset, num_layers_per_chunk)
            device_layers[pipeline_rank][vp_rank] = layers
            
            print(f"  Virtual Rank {vp_rank}: offset={offset:2d} → layers={layers}")
        print()
    
    return device_layers

def generate_asymmetric_config_from_user_input(first_half_distributions, total_layers, pipeline_size):
    """
    Generate full asymmetric configuration from user input for first half of devices
    
    first_half_distributions: List of distributions for first half devices
    For 4 devices: [[2,4,2,3], [2,1,1,1]] - distributions for devices 0 and 1
    For 8 devices: [[2,4,2,3], [2,1,1,1], [3,2,1,2], [1,3,2,1]] - distributions for devices 0,1,2,3
    
    Returns: Full configuration for all devices
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

def get_asymmetric_offset_v2(pipeline_rank, vp_rank, asymmetric_device_config):
    """
    Scalable asymmetric offset calculation for any even number of devices
    
    asymmetric_device_config: List of device configurations
    For 4 devices: [[2,4,2,3], [2,1,1,1], [1,2,1,1], [4,2,3,2]]
    For 8 devices: [[...], [...], [...], [...], [...], [...], [...], [...]]
    Each sub-array represents layers per VR for each device
    
    Pattern (scales to any even number of devices):
    - VR0: assigns layers sequentially in forward device order (0→1→...→N-1)
    - VR1: assigns same layers as VR0 but with device pairing (0↔N-1, 1↔N-2, ...)
    - VR2: assigns remaining layers sequentially in reverse device order (N-1→...→1→0)
    - VR3: assigns same layers as VR2 but with device pairing (0↔N-1, 1↔N-2, ...)
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


def simulate_asymmetric_bitpipe(total_layers, pipeline_size, vp_size, first_half_distributions):
    """
    Simulate asymmetric BitPipe layer distribution from user input (scalable version)
    
    first_half_distributions: List of distributions for first half of devices
    For 4 devices: [[2,4,2,3], [2,1,1,1]]
    For 8 devices: [[2,4,2,3], [2,1,1,1], [3,2,1,2], [1,3,2,1]]
    """
    print(f"=== ASYMMETRIC BITPIPE SIMULATION ===")
    print(f"Total layers: {total_layers}")
    print(f"Pipeline size: {pipeline_size}")
    print(f"Virtual pipeline size: {vp_size}")
    print(f"User input (first half devices):")
    for i, dist in enumerate(first_half_distributions):
        print(f"  Device {i} distribution: {dist}")
    print()
    
    # Generate full configuration
    asymmetric_device_config = generate_asymmetric_config_from_user_input(
        first_half_distributions, total_layers, pipeline_size
    )
    
    print(f"Generated full configuration:")
    for i, config in enumerate(asymmetric_device_config):
        pair_idx = pipeline_size - 1 - i
        if i < pipeline_size // 2:
            print(f"  Device {i}: {config} (pairs with Device {pair_idx})")
        else:
            print(f"  Device {i}: {config} (paired from Device {pair_idx})")
    print()
    
    device_layers = {}
    
    for pipeline_rank in range(pipeline_size):
        device_layers[pipeline_rank] = {}
        print(f"[BITPIPE LOG] Device {pipeline_rank}:")
        
        total_device_layers = 0
        for vp_rank in range(vp_size):
            offset, num_layers = get_asymmetric_offset_v2(pipeline_rank, vp_rank, asymmetric_device_config)
            if num_layers > 0:
                layers = get_layers_from_offset(offset, num_layers)
            else:
                layers = []
            device_layers[pipeline_rank][vp_rank] = layers
            total_device_layers += len(layers)
            
            print(f"[BITPIPE LOG] Device {pipeline_rank} | Virtual Rank {vp_rank} | Created layers: {layers}")
        
        print(f"  Total layers on device: {total_device_layers}")
        print()
    
    return device_layers

def compare_distributions(symmetric_result, asymmetric_result, pipeline_size, vp_size):
    """
    Compare the two distribution strategies
    """
    print("=== COMPARISON ===")
    print()
    
    print("Layers per device:")
    print(f"{'Device':<8} {'Symmetric':<12} {'Asymmetric':<12}")
    print("-" * 35)
    
    for device in range(pipeline_size):
        sym_total = sum(len(symmetric_result[device][vp]) for vp in range(vp_size))
        asym_total = sum(len(asymmetric_result[device][vp]) for vp in range(vp_size))
        print(f"{device:<8} {sym_total:<12} {asym_total:<12}")
    
    print()
    
    # Check for layer coverage
    print("Layer coverage check:")
    
    sym_all_layers = set()
    asym_all_layers = set()
    
    for device in range(pipeline_size):
        for vp in range(vp_size):
            sym_all_layers.update(symmetric_result[device][vp])
            asym_all_layers.update(asymmetric_result[device][vp])
    
    print(f"Symmetric covers layers: {sorted(sym_all_layers)}")
    print(f"Asymmetric covers layers: {sorted(asym_all_layers)}")
    print()

def main():
    """
    Main simulation - demonstrates scalability for different device counts
    """
    print("BitPipe Layer Distribution Simulation")
    print("=" * 60)
    print()
    
    # Test 1: 4 devices (original example)
    print("🔹 TEST 1: 4 DEVICES")
    print("=" * 40)
    total_layers = 24
    pipeline_size = 4  # 4 devices
    vp_size = 4        # 4 virtual pipeline ranks
    
    # # Run symmetric simulation (current BitPipe)
    # symmetric_result = simulate_symmetric_bitpipe(total_layers, pipeline_size, vp_size)
    
    # print()
    
    # # Run asymmetric simulation (device-oriented user input)
    # print("Testing 4-device asymmetric example:") #Inikepake buat "compare_all_4device_32layer"
    # first_half_distributions = [
    #     [3, 3, 3, 5],  # Device 0: VR0=2, VR1=4, VR2=2, VR3=3 layers
    #     [4, 5, 4, 5]   # Device 1: VR0=2, VR1=1, VR2=1, VR3=1 layers
    # ]
    
    # asymmetric_result = simulate_asymmetric_bitpipe(
    #     total_layers, pipeline_size, vp_size, first_half_distributions
    # )

    # Run asymmetric simulation (device-oriented user input)
    # print("Testing 8-device asymmetric example for 96 layers:") 
    # first_half_distributions2 = [
    #     [6, 5, 5, 6],
    #     [6, 7, 7, 6],
    #     [6, 7, 7, 6],
    #     [6, 5, 5, 6]
    # ]
    
    # asymmetric_result2 = simulate_asymmetric_bitpipe(
    #     total_layers, pipeline_size, vp_size, first_half_distributions2
    # )

        # Run asymmetric simulation (device-oriented user input)
    print("Testing 4-device asymmetric example for 32 layers:") 
    first_half_distributions2 = [
        [3, 3, 3, 3],
        [3, 3, 3, 3]
    ]
    
    asymmetric_result2 = simulate_asymmetric_bitpipe(
        total_layers, pipeline_size, vp_size, first_half_distributions2
    )
    
if __name__ == "__main__":
    main()