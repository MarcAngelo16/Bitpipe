# BitPipe: Bidirectional Interleaved Pipeline Parallelism

## Overview

BitPipe is a high-performance pipeline parallelism library built on top of NVIDIA's Megatron-LM. It implements a novel **bidirectional interleaved pipeline parallelism** approach that significantly improves GPU utilization and training throughput for large language models.

### Key Innovation

BitPipe seamlessly merges **two V-shaped interleaved pipelines running in opposite directions**, enabling:
- **2x concurrent microbatch execution** compared to standard 1F1B pipeline
- **Reduced pipeline bubbles** through interleaving
- **Eager gradient synchronization** for better communication-computation overlap
- **Asymmetric stage configurations** for heterogeneous hardware setups

---

## Requirements & Dependencies

### Hardware Requirements

**Minimum GPU Count:**
- **BitPipe 4-VR**: Requires **minimum 4 GPUs** (must be even number: 4, 6, 8, 10, 12...)
- The V-shaped pattern with 4 Virtual Ranks per device cannot be properly distributed across fewer than 4 devices
- **2 GPUs will NOT work** - this is a design limitation, not a bug

**Recommended:**
- 8+ GPUs for optimal pipeline efficiency
- Homogeneous GPUs for symmetric mode
- Heterogeneous GPUs supported via asymmetric mode

### Software Dependencies

**Critical Version Requirements:**

```bash
# Core Dependencies (tested and verified)
torch == 2.1.0
torchvision == 0.16.0
transformers <= 4.55.2  # IMPORTANT: Do NOT use 4.56.0 or newer

# Other key dependencies
tensorboard >= 2.0.0
numpy >= 1.20.0
```

**⚠️ CRITICAL: Transformers Version Compatibility**

- **Maximum supported version**: `transformers 4.55.2`
- **DO NOT upgrade to 4.57.1 or newer** - will cause API incompatibility errors

**Why this matters:**
- Transformers 4.57+ uses PyTorch 2.2+ API (`register_pytree_node`)
- PyTorch 2.1.0 only has the private API (`_register_pytree_node`)
- Incompatibility causes: `AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'`

**Installation:**
```bash
# Install correct versions
pip install torch==2.1.0 torchvision==0.16.0
pip install transformers==4.55.2  # or 4.31.0 for maximum stability

# If you accidentally upgraded transformers, downgrade:
pip install transformers==4.55.2 --force-reinstall
```

### Environment Variables

```bash
# Required for BitPipe
export CUDA_DEVICE_MAX_CONNECTIONS=1  # REQUIRED for pipeline parallelism

# Optional debugging
export BITPIPE_DEBUG=1           # Enable debug printing
export NCCL_DEBUG=INFO           # NCCL communication debug
```

---

## Architecture Components

### 1. Base Bidirectional Pipeline (Symmetric)

The original BitPipe implementation divides the model into **4 virtual ranks (VR)** per device:

```
Device Layout (4 devices, 16 layers example):
Device 0: VR0[0,1]  VR1[6,7]  VR2[14,15] VR3[8,9]
Device 1: VR0[2,3]  VR1[4,5]  VR2[12,13] VR3[10,11]
Device 2: VR0[4,5]  VR1[2,3]  VR2[10,11] VR3[12,13]
Device 3: VR0[6,7]  VR1[0,1]  VR2[8,9]   VR3[14,15]
```

**Key Properties:**
- Each device has **exactly the same number of layers per VR** (symmetric)
- VR0-VR1 form the **forward pipeline** (first half of model)
- VR2-VR3 form the **backward pipeline** (second half of model)
- Layers are assigned in a **V-shaped pattern** for optimal scheduling

**Location:** `megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py` (migrated from `bitpipe_schedule.py`)

### 2. Asymmetric Bidirectional Pipeline (NEW)

The asymmetric extension allows **different layer distributions per device**, enabling:
- Load balancing for heterogeneous GPUs
- Optimized memory usage per device
- Custom performance tuning

```
Device Layout (12 devices, 48 layers example):
Device 0: VR0[2]   VR1[2]   VR2[2]   VR3[2]   (8 layers total)
Device 1: VR0[2]   VR1[1]   VR2[1]   VR3[2]   (6 layers total)
Device 2: VR0[2]   VR1[1]   VR2[2]   VR3[2]   (7 layers total)
...
```

**Configuration Format:**
```json
{
  "description": "Asymmetric config for 12 devices, 48 layers",
  "first_half_devices": [
    [2, 2, 2, 2],  // Device 0: VR0=2, VR1=2, VR2=2, VR3=2
    [2, 1, 1, 2],  // Device 1: VR0=2, VR1=1, VR2=1, VR3=2
    [2, 1, 2, 2],  // Device 2: ...
    ...            // Only first half specified
  ]
}
```

**Automatic Pairing:**
- User specifies only **first N/2 devices**
- System auto-generates **second half with VR swapping**: `[VR0,VR1,VR2,VR3] → [VR1,VR0,VR3,VR2]`
- Ensures **bidirectional symmetry**

**Location:** `megatron/core/pipeline_parallel/asymmetric/config_utils.py` (migrated from `bitpipe_asymmetric_utils.py`)

---

## Core Concepts

### Virtual Ranks (VR)

Each device executes **4 model chunks** corresponding to different parts of the full model:

- **VR0**: First quarter of forward pipeline (layers assigned sequentially)
- **VR1**: Second quarter of forward pipeline (device-paired with VR0)
- **VR2**: First quarter of backward pipeline (layers in reverse order)
- **VR3**: Second quarter of backward pipeline (device-paired with VR2)

### Microbatch Scheduling

BitPipe doubles the effective microbatches by treating each model chunk as a separate pipeline stage:

```python
# Original microbatches: 8
# BitPipe effective microbatches: 8 * (num_model_chunks/2) = 8 * 2 = 16

total_num_microbatches = num_microbatches * (num_model_chunks // 2)
```

**Execution Phases:**
1. **Warmup Phase**: Fill the pipeline with forward passes
2. **Steady-State Phase**: Interleaved 1F1B (one forward, one backward)
3. **Cooldown Phase**: Drain backward passes with eager gradient sync

### Layer Assignment Logic

**Symmetric (Original):**
```python
# V-shaped layer assignment
vp_idx = vp_rank if vp_rank < 2 else (vp_rank - 1) % 2
offset = (
    pipeline_rank * num_layers_per_vr +
    vp_idx * (pipeline_world_size - 1 - 2 * pipeline_rank) * num_layers_per_vr +
    (vp_rank // 2) * (total_layers // 2)
)
```

**Asymmetric:**
```python
# Calculated from device config with pairing
offset, num_layers = get_asymmetric_offset(
    pipeline_rank, vp_rank, asymmetric_device_config
)
```

**VR Assignment Rules:**
- **VR0**: Sequential device order (0→1→...→N-1)
- **VR1**: Paired with VR0 (device i gets device N-1-i's layers)
- **VR2**: Reverse device order (N-1→...→1→0) for second model half
- **VR3**: Paired with VR2 (device i gets device N-1-i's layers)

---

## Communication Patterns

### P2P Communication

BitPipe uses **bidirectional P2P communication** between adjacent pipeline stages:

**Forward Direction:**
- `send_forward`: Send activations to next stage
- `recv_forward`: Receive activations from previous stage
- `send_forward_recv_forward_bd0`: Combined send/recv for bidirectional

**Backward Direction:**
- `send_backward`: Send gradients to previous stage
- `recv_backward`: Receive gradients from next stage
- `send_backward_recv_backward_bd`: Combined for bidirectional

**Special Cases:**
- First/last stages handle boundary conditions
- **Same-device transitions**: Direct tensor passing with `.detach()` (no communication)

**Location:** `megatron/core/pipeline_parallel/p2p_communication.py`

### Gradient Synchronization

**Eager Sync Strategy:**
```python
# Sync happens at specific points in cooldown phase
if microbatch_id == -1:  # Sync marker
    for chunk in [VR2, VR3]:  # Backward pipeline chunks
        allreduce_gradients(model[chunk])
```

**Benefits:**
- Overlaps gradient sync with computation
- Reduces final synchronization barrier
- Improves overall pipeline efficiency

---

## File Structure

### Core Implementation (NEW - Reorganized October 2025)

```
megatron/
├── core/
│   ├── pipeline_parallel/
│   │   ├── schedules.py                              # Entry point for all schedules
│   │   ├── p2p_communication.py                      # Inter-stage communication
│   │   ├── bitpipe_profiler.py                       # Performance profiling
│   │   │
│   │   ├── schedule_impl/                            # NEW: Organized schedule implementations
│   │   │   ├── __init__.py
│   │   │   ├── bitpipe/                              # BitPipe 4-VR family
│   │   │   │   ├── __init__.py
│   │   │   │   └── bitpipe_4vr.py                    # Main BitPipe 4-VR scheduler
│   │   │   └── chimera/                              # Chimera 2-VR family (PLANNED)
│   │   │       ├── __init__.py
│   │   │       └── chimera_2vr.py                    # Simplified 2-VR scheduler (FUTURE)
│   │   │
│   │   └── asymmetric/                               # NEW: Asymmetric config management
│   │       ├── __init__.py
│   │       └── config_utils.py                       # Asymmetric utilities
│   │
│   ├── parallel_state.py                             # Parallel group management
│   └── transformer/                                  # Transformer implementation
│
├── model/
│   ├── transformer.py                                # Layer distribution logic
│   └── language_model.py                             # Model definitions
│
├── bitpipe_schedule.py                               # DEPRECATED: Backward compat shim
└── bitpipe_asymmetric_utils.py                       # DEPRECATED: Backward compat shim

asymmetric_bitpipe/
├── configs/                                          # Asymmetric device configurations
│   ├── 8_devices/
│   ├── 12_devices/
│   └── ...
└── scripts/                                          # Analysis and profiling tools
```

### Migration Status (October 2025)

**✅ Completed:**
- Phase 1: Created organized directory structure
- Phase 2: Migrated BitPipe 4-VR scheduler to `schedule_impl/bitpipe/`
- Asymmetric Migration: Moved utilities to `asymmetric/config_utils.py`
- Backward compatibility shims in place for old import paths

**📋 Next:**
- Phase 3: Performance validation with actual training runs
- Phase 5: Implement Chimera 2-VR scheduler (see section below)

**Migration Documentation:**
- `PHASE1_SUMMARY.md` - Directory structure creation
- `PHASE2_SUMMARY.md` - BitPipe code migration
- `ASYMMETRIC_MIGRATION_SUMMARY.md` - Asymmetric utilities migration

### Configuration Files
- **Asymmetric configs**: `asymmetric_bitpipe/configs/{N}_devices/*.json`
- **Training scripts**: `singlenode_*.sh`, `multinode_*.sh`
- **Examples**: `pretrain_gpt.py`, `pretrain_bert.py`

---

## Usage

### Enable BitPipe (Symmetric)

```bash
torchrun pretrain_gpt.py \
    --enable-bitpipe-schedule \
    --pipeline-model-parallel-size 8 \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --num-layers 96 \
    ...
```

**Requirements:**
- `global_batch_size` must be divisible by `pipeline_model_parallel_size`
- `num_microbatches >= pipeline_model_parallel_size` for efficiency
- Virtual pipeline size automatically set to 4

### Enable Asymmetric BitPipe

```bash
torchrun pretrain_gpt.py \
    --enable-bitpipe-schedule \
    --enable-bitpipe-asymmetric \
    --bitpipe-asymmetric-config configs/12_devices/asymmetric_12devices_48layers.json \
    --pipeline-model-parallel-size 12 \
    --num-layers 48 \
    ...
```

**Configuration Validation:**
- Pipeline size must be **even**
- Total layers from config must match `--num-layers`
- Device pairing constraint: `sum(device[i]) == sum(device[N-1-i])`

---

## Data Flow

### Forward Pass Flow

1. **Input Preparation**: First stage receives input, others receive from previous stage
2. **VR Execution**: Each device processes microbatch through current VR's layers
3. **Output Routing**:
   - Same VR: Send to next pipeline stage
   - Different VR: Send to paired device or next VR

### Backward Pass Flow

1. **Gradient Reception**: Last stage starts with loss gradient, others receive from next stage
2. **VR Execution**: Process backward pass through current VR's layers
3. **Gradient Routing**:
   - Send to previous pipeline stage
   - Accumulate for gradient sync

### Microbatch Ordering

The scheduler carefully orders microbatches to maximize pipeline utilization:

```python
# Example for 4 devices, 16 microbatches
# Forward order: [0,1,2,10,3,11,8,9,4,5,6,14,7,15,12,13]
# Backward order: [8,9,10,2,11,3,0,1,12,13,14,6,15,7,4,5]
# Interleaved to maintain V-shaped execution pattern
```

---

## Profiling & Debugging

### Enable Profiling

```bash
--enable-bitpipe-profiling \
--bitpipe-profile-train-iters 3
```

**Captures:**
- Per-microbatch timing (forward/backward)
- P2P communication overhead
- Pipeline bubble analysis
- Phase transitions (warmup/steady/cooldown)

**Output Location:** `bitpipe_profile_*.json`

### Debug Environment Variables

```bash
export BITPIPE_DEBUG=1           # Enable debug printing
export NCCL_DEBUG=INFO           # NCCL communication debug
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Required for BitPipe
```

---

## Key Algorithms

### Asymmetric Offset Calculation

```python
def get_asymmetric_offset(pipeline_rank, vp_rank, asymmetric_device_config):
    """
    Returns (offset, num_layers) for current device-VR combination

    Pattern for any even number of devices:
    - VR0: Sequential forward (0→1→...→N-1)
    - VR1: Device pairing (i ↔ N-1-i) with VR0 layers
    - VR2: Sequential reverse (N-1→...→0) for second half
    - VR3: Device pairing (i ↔ N-1-i) with VR2 layers
    """
    num_layers_for_this_device_vr = asymmetric_device_config[pipeline_rank][vp_rank]

    if num_layers_for_this_device_vr == 0:
        return 0, 0

    num_devices = len(asymmetric_device_config)
    total_vr0_layers = sum(config[0] for config in asymmetric_device_config)

    offset = 0

    if vp_rank == 0:
        # Forward order
        offset = sum(asymmetric_device_config[dev][0]
                    for dev in range(pipeline_rank))

    elif vp_rank == 1:
        # Paired with VR0
        paired_device = num_devices - 1 - pipeline_rank
        offset = sum(asymmetric_device_config[dev][0]
                    for dev in range(paired_device))

    elif vp_rank == 2:
        # Reverse order, second half
        offset = total_vr0_layers
        offset += sum(asymmetric_device_config[dev][2]
                     for dev in range(num_devices - 1, pipeline_rank, -1))

    elif vp_rank == 3:
        # Paired with VR2
        paired_device = num_devices - 1 - pipeline_rank
        offset = total_vr0_layers
        offset += sum(asymmetric_device_config[dev][2]
                     for dev in range(num_devices - 1, paired_device, -1))

    return offset, num_layers_for_this_device_vr
```

### Microbatch Chunk Assignment

```python
def get_model_chunk_id(microbatch_id):
    """
    Maps microbatch ID to model chunk (VR)

    Pattern: First half → VR0/VR1, Second half → VR2/VR3
    Within each half: alternate between chunks based on pipeline position
    """
    microbatch_id_in_group = microbatch_id % pipeline_parallel_size
    chunk_offset = 0 if microbatch_id < (total_num_microbatches // 2) else 2
    model_chunk_id = microbatch_id_in_group // (pipeline_parallel_size // 2)
    model_chunk_id += chunk_offset
    return model_chunk_id
```

---

## Performance Considerations

### Optimal Configuration

**Microbatch Count:**
- Minimum: `num_microbatches >= 2 * pipeline_parallel_size`
- Recommended: `num_microbatches = k * pipeline_parallel_size` where k ≥ 2

**Memory vs Throughput:**
- More microbatches → Better pipeline utilization
- Fewer microbatches → Lower memory footprint
- Asymmetric configs → Balance heterogeneous GPU memory

### Known Limitations

1. **Minimum 4 GPUs required** (BitPipe 4-VR cannot run on 2 GPUs)
2. **Pipeline size must be even** (bidirectional pairing requirement: 4, 6, 8, 10, 12...)
3. **Virtual pipeline size = 4** (hardcoded for BitPipe 4-VR)
4. **Transformers version capped at 4.55.2** (newer versions incompatible with PyTorch 2.1.0)
5. **Not compatible with encoder-decoder models**
6. **Requires `CUDA_DEVICE_MAX_CONNECTIONS=1`** environment variable

---

## Comparison: BitPipe vs Standard 1F1B

| Feature | Standard 1F1B | BitPipe (Symmetric) | BitPipe (Asymmetric) |
|---------|---------------|---------------------|----------------------|
| Pipeline Direction | Unidirectional | Bidirectional | Bidirectional |
| Virtual Ranks/Device | 1 | 4 | 4 (variable layers) |
| Effective Microbatches | N | 2N | 2N |
| Layer Distribution | Uniform | Uniform | Configurable |
| Gradient Sync | End of backward | Eager (interleaved) | Eager (interleaved) |
| Pipeline Bubble | ~12% | ~6% | ~5% (optimized) |
| Hardware Support | Homogeneous | Homogeneous | Heterogeneous |

---

## Future Development

The asymmetric BitPipe extension opens possibilities for:
- **Heterogeneous cluster support**: Different GPU types in same pipeline
- **Memory-optimized configurations**: Larger models on mixed memory setups
- **Performance tuning**: Custom layer distributions per hardware profile
- **Dynamic load balancing**: Runtime adjustment of layer assignments

---

## Quick Reference

### Command Line Flags

```bash
# BitPipe Core
--enable-bitpipe-schedule              # Enable BitPipe scheduler

# Asymmetric Mode
--enable-bitpipe-asymmetric            # Enable asymmetric layer distribution
--bitpipe-asymmetric-config <path>     # Path to asymmetric config JSON

# Profiling
--enable-bitpipe-profiling             # Enable performance profiling
--bitpipe-profile-train-iters <N>      # Profile first N training iterations

# Pipeline Config
--pipeline-model-parallel-size <N>     # Number of pipeline stages (must be even)
--micro-batch-size <B>                 # Microbatch size
--global-batch-size <G>                # Global batch size (must be divisible by pipeline size)
--num-layers <L>                       # Total number of transformer layers
```

### Entry Points

```python
# Scheduler selection
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

# Returns BitPipe scheduler if --enable-bitpipe-schedule is set
forward_backward_func = get_forward_backward_func()

# Asymmetric config generation (NEW LOCATION)
from megatron.core.pipeline_parallel.asymmetric import (
    generate_asymmetric_config_from_user_input,
    get_asymmetric_offset
)

# DEPRECATED (still works but shows warning):
# from megatron.bitpipe_asymmetric_utils import get_asymmetric_offset

# Layer offset calculation
offset, num_layers = get_asymmetric_offset(
    pipeline_rank, vp_rank, asymmetric_device_config
)
```

---

## Troubleshooting

### Common Issues

**Error: `AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'`**
- **Cause**: Transformers version too new (4.56.0+) for PyTorch 2.1.0
- **Solution**: Downgrade transformers
  ```bash
  pip install transformers==4.55.2 --force-reinstall
  ```
- **Why**: Transformers 4.57+ requires PyTorch 2.2+ API not available in PyTorch 2.1.0
- **See**: [Requirements & Dependencies](#requirements--dependencies) section for details

**Error: "Insufficient GPUs" or crashes with 2 GPUs**
- **Cause**: BitPipe 4-VR requires minimum 4 GPUs
- **Solution**: Use 4, 6, 8, 10, or 12+ GPUs (must be even number)
- **Why**: The V-shaped 4-VR pattern cannot be distributed across fewer than 4 devices
- **This is by design, not a bug**

**Error: "Pipeline size must be even"**
- BitPipe requires even number of devices for bidirectional pairing
- Use 4, 6, 8, 10, 12... devices (NOT 3, 5, 7...)

**Error: "Total layers mismatch"**
- Sum of layers in asymmetric config must equal `--num-layers`
- Check JSON configuration file

**Error: "Pairing constraint violation"**
- Device i and device N-1-i must have same total layers
- System auto-generates second half, but first half must be valid

**Memory Issues:**
- Reduce `--micro-batch-size`
- Reduce `--global-batch-size`
- Use asymmetric config to balance memory across devices

---

## User Discussions

### Topic 1: Understanding VR Transitions - The "Name Change" Mystery

**Essential Insight**: Microbatches traverse all 4 VRs on their journey through the model, with a critical same-device transition between VR pairs.

#### The Complete Microbatch Journey

Each microbatch actually flows through **all 4 Virtual Ranks** on its complete forward pass through the model:

**Pipeline 0 (Forward Direction):**
- **VR0** (Device 0→1→2→3): First quarter of model (layers 0-7)
- Transition occurs **on the same device** (no network communication)
- **VR2** (Device 3→2→1→0): Third quarter of model (layers 8-15)

**Pipeline 1 (Backward Direction):**
- **VR1** (Device 3→2→1→0): Second quarter of model (layers 7-0 in reverse)
- Transition occurs **on the same device** (no network communication)
- **VR3** (Device 0→1→2→3): Fourth quarter of model (layers 8-15)

#### Concrete Example: 4 Devices, 16 Layers

**Microbatch 0 (Pipeline 0) Journey:**

1. **VR0 Execution** (First half of model):
   - Device 0, VR0: Layers [0,1] → MB ID: 0
   - Device 1, VR0: Layers [2,3] → MB ID: 0
   - Device 2, VR0: Layers [4,5] → MB ID: 0
   - Device 3, VR0: Layers [6,7] → MB ID: 0 ← Last stage of VR0

2. **TRANSITION** (On Device 3, no network communication):
   ```
   Output from Device 3's VR0 → Input to Device 3's VR2
   Same physical data, same device, different VR
   From scheduler's perspective: MB0 → MB8
   ```

3. **VR2 Execution** (Second half of model):
   - Device 3, VR2: Layers [14,15] → MB ID: 8
   - Device 2, VR2: Layers [12,13] → MB ID: 8
   - Device 1, VR2: Layers [10,11] → MB ID: 8
   - Device 0, VR2: Layers [8,9] → MB ID: 8

**Microbatch 2 (Pipeline 1) Journey:**

1. **VR1 Execution** (First half of model, reverse direction):
   - Device 3, VR1: Layers [6,7] → MB ID: 2
   - Device 2, VR1: Layers [4,5] → MB ID: 2
   - Device 1, VR1: Layers [2,3] → MB ID: 2
   - Device 0, VR1: Layers [0,1] → MB ID: 2 ← Last stage of VR1

2. **TRANSITION** (On Device 0, no network communication):
   ```
   Output from Device 0's VR1 → Input to Device 0's VR3
   Same physical data, same device, different VR
   From scheduler's perspective: MB2 → MB10
   ```

3. **VR3 Execution** (Second half of model):
   - Device 0, VR3: Layers [8,9] → MB ID: 10
   - Device 1, VR3: Layers [10,11] → MB ID: 10
   - Device 2, VR3: Layers [12,13] → MB ID: 10
   - Device 3, VR3: Layers [14,15] → MB ID: 10

#### The Critical Code: Same-Device VR Transition

**Location:** `megatron/core/pipeline_parallel/bitpipe_schedule.py:558-561`

```python
if (parallel_state.is_pipeline_last_stage(ignore_virtual=True) and next_forward_model_chunk_id == v_size-2) or \
   (parallel_state.is_pipeline_first_stage(ignore_virtual=True) and next_forward_model_chunk_id == v_size-1):
    detached_output_tensor = output_tensor.detach()
    detached_output_tensor.requires_grad_()
    input_tensor = detached_output_tensor  # last stage of chunk0 and first stage of chunk2 are in the same device
```

**What This Code Does:**

1. **Detects VR Transition Points**:
   - `v_size-2` = VR2 (value 2 when v_size=4)
   - `v_size-1` = VR3 (value 3 when v_size=4)
   - Checks if transitioning from VR0→VR2 or VR1→VR3

2. **Performs In-Memory Transfer**:
   - `.detach()`: Detaches output tensor from computation graph
   - `.requires_grad_()`: Re-enables gradient tracking for backward pass
   - Direct assignment: No network communication needed

3. **No Network Communication**:
   - Both VRs reside on the **same physical device**
   - Only requires memory operation, not P2P communication
   - Saves significant communication overhead

#### Microbatch ID Mapping

The "name change" is actually the scheduler's way of tracking microbatches through different VR stages:

```python
def get_model_chunk_id(microbatch_id):
    """Maps microbatch ID to VR based on position in schedule"""
    microbatch_id_in_group = microbatch_id % pipeline_parallel_size
    chunk_offset = 0 if microbatch_id < (total_num_microbatches // 2) else 2
    model_chunk_id = microbatch_id_in_group // (pipeline_parallel_size // 2)
    model_chunk_id += chunk_offset
    return model_chunk_id
```

**Example with 4 devices, 16 total microbatches:**

| Microbatch ID | VR Assignment | Notes |
|--------------|---------------|-------|
| 0, 1, 4, 5 | VR0 | First half, Pipeline 0 |
| 2, 3, 6, 7 | VR1 | First half, Pipeline 1 |
| 8, 9, 12, 13 | VR2 | Second half, Pipeline 0 (same data as MB 0,1,4,5) |
| 10, 11, 14, 15 | VR3 | Second half, Pipeline 1 (same data as MB 2,3,6,7) |

#### Why This Matters

1. **Zero-Copy Transition**: The same microbatch data flows through 4 VRs with only one memory operation
2. **Efficient Scheduling**: Scheduler tracks position using different IDs, but it's the same logical microbatch
3. **Pipeline Efficiency**: No network communication overhead at the VR transition point
4. **Bidirectional Flow**: Enables true bidirectional pipeline execution with minimal overhead

This design is crucial to BitPipe's performance advantage—each microbatch seamlessly traverses the entire model through 4 VR stages, with efficient same-device transitions at the model's midpoint.

---

## Chimera: Simplified 2-VR Bidirectional Pipeline 🚧 IN PROGRESS

### Overview

**Chimera** is a simplified variant of BitPipe that reduces complexity while maintaining bidirectional pipeline benefits. It uses **2 Virtual Ranks (VR)** per device instead of 4, using sequential rank ordering instead of V-shaped pattern.

**Implementation Status (October 2025):**
- ✅ **Core scheduler**: Implemented in `schedule_impl/chimera/chimera_2vr.py`
- ✅ **Bidirectional gradient sync**: Implemented with BD groups
- ✅ **Symmetric mode**: Code complete, debugging in progress
- 🧪 **Testing**: Initial bugs fixed, awaiting 4-GPU hardware validation
- ⚠️ **Asymmetric mode**: Command-line args exist but NOT implemented yet
- 📝 **Production ready**: NO - needs full testing and validation

**Files Added:**
- `megatron/core/pipeline_parallel/schedule_impl/chimera/chimera_2vr.py` - Main scheduler
- `megatron/core/pipeline_parallel/schedule_impl/chimera/__init__.py` - Package init
- Command-line arguments in `megatron/arguments.py` (lines 1188-1197)

**Files Modified (Implementation Session):**
- `megatron/arguments.py` - Added Chimera initialization (lines 255-293)
- `megatron/core/pipeline_parallel/schedule_impl/chimera/chimera_2vr.py` - Added gradient sync
- `megatron/core/parallel_state.py` - Extended BD groups to Chimera (line 262, 273-281)

**Testing Status:**
- ✅ Code compiles and imports successfully
- ✅ Fixed: `virtual_pipeline_model_parallel_size` initialization error
- ✅ Fixed: `no_sync_func` None type error
- ✅ Fixed: Missing `allreduce_gradients()` implementation
- 🧪 **Pending**: Full training run on 4+ GPU hardware
- 🧪 **Pending**: Convergence validation
- 🧪 **Pending**: Performance comparison with BitPipe 4-VR

**Command-Line Usage:**
```bash
# Basic Chimera 2-VR (symmetric) - TESTING IN PROGRESS
torchrun pretrain_gpt.py \
    --enable-chimera-schedule \
    --pipeline-model-parallel-size 4 \
    --num-layers 48 \
    --micro-batch-size 4 \
    --global-batch-size 32

# Asymmetric mode (NOT YET IMPLEMENTED)
# --enable-chimera-asymmetric \
# --chimera-asymmetric-config configs/chimera_4devices.json
```

### Key Design Differences

| Feature | BitPipe 4-VR | Chimera 2-VR |
|---------|--------------|--------------|
| Virtual Ranks/Device | 4 (VR0, VR1, VR2, VR3) | 2 (VR0, VR1) |
| Layer Assignment | V-shaped pattern | Sequential (no V-shape) |
| Device Pairing | VR0↔VR1, VR2↔VR3 | VR0↔VR1 (simple swap) |
| Same-Device Transitions | VR0→VR2, VR1→VR3 | None (direct bidirectional) |
| P2P Communication | More complex routing | Simpler routing |
| Effective Microbatches | 2N | 2N (same as BitPipe) |
| Pipeline Bubble | ~6% | ~6% (expected) |

### Architecture

**Device Layout (4 devices, 16 layers example):**

```
# Chimera 2-VR (No V-shape):
Device 0: VR0[0,1,2,3]      VR1[12,13,14,15]
Device 1: VR0[4,5,6,7]      VR1[8,9,10,11]
Device 2: VR0[8,9,10,11]    VR1[4,5,6,7]
Device 3: VR0[12,13,14,15]  VR1[0,1,2,3]

# Compare to BitPipe 4-VR (V-shape):
Device 0: VR0[0,1]  VR1[6,7]  VR2[14,15]  VR3[8,9]
Device 1: VR0[2,3]  VR1[4,5]  VR2[12,13]  VR3[10,11]
Device 2: VR0[4,5]  VR1[2,3]  VR2[10,11]  VR3[12,13]
Device 3: VR0[6,7]  VR1[0,1]  VR2[8,9]    VR3[14,15]
```

### Simplified Layer Assignment

**Chimera Pattern (Sequential):**
```python
# VR0: Forward direction (0→1→2→3)
# Device 0: Layers 0-3
# Device 1: Layers 4-7
# Device 2: Layers 8-11
# Device 3: Layers 12-15

# VR1: Backward direction with simple swap (3→2→1→0)
# Device 0: Layers 12-15 (from Device 3's VR0)
# Device 1: Layers 8-11  (from Device 2's VR0)
# Device 2: Layers 4-7   (from Device 1's VR0)
# Device 3: Layers 0-3   (from Device 0's VR0)
```

**Calculation:**
```python
def get_chimera_offset(pipeline_rank, vp_rank, num_layers_per_vr, num_devices):
    """
    Simplified offset calculation for Chimera 2-VR

    Pattern:
    - VR0: Sequential forward (device 0→1→...→N-1)
    - VR1: Simple device swap (device i gets device N-1-i's VR0 layers)

    No V-shape, no complex pairing logic
    """
    if vp_rank == 0:
        # VR0: Sequential forward
        offset = pipeline_rank * num_layers_per_vr
    else:  # vp_rank == 1
        # VR1: Device swap (i ↔ N-1-i)
        paired_device = num_devices - 1 - pipeline_rank
        offset = paired_device * num_layers_per_vr

    return offset, num_layers_per_vr
```

### Microbatch Flow Example

**Chimera 2-VR Flow (4 devices, 8 microbatches):**

```
Pipeline 0 (VR0):
MB0: Device 0 [0-3] → Device 1 [4-7] → Device 2 [8-11] → Device 3 [12-15]
MB1: Device 0 [0-3] → Device 1 [4-7] → Device 2 [8-11] → Device 3 [12-15]
MB4: Device 0 [0-3] → Device 1 [4-7] → Device 2 [8-11] → Device 3 [12-15]
MB5: Device 0 [0-3] → Device 1 [4-7] → Device 2 [8-11] → Device 3 [12-15]

Pipeline 1 (VR1):
MB2: Device 3 [0-3] → Device 2 [4-7] → Device 1 [8-11] → Device 0 [12-15]
MB3: Device 3 [0-3] → Device 2 [4-7] → Device 1 [8-11] → Device 0 [12-15]
MB6: Device 3 [0-3] → Device 2 [4-7] → Device 1 [8-11] → Device 0 [12-15]
MB7: Device 3 [0-3] → Device 2 [4-7] → Device 1 [8-11] → Device 0 [12-15]
```

**Key Difference**: No VR transitions within the same device—each microbatch stays in the same VR throughout its journey.

### Implementation Plan

**Location**: `megatron/core/pipeline_parallel/schedule_impl/chimera/chimera_2vr.py`

**Key Changes from BitPipe 4-VR**:

1. **Virtual Pipeline Size**: Set to 2 instead of 4
   ```python
   args.virtual_pipeline_model_parallel_size = 2  # Not 4
   ```

2. **Layer Offset Calculation**: Use simplified sequential pattern
   ```python
   # No V-shape calculation needed
   offset = pipeline_rank * num_layers_per_vr if vp_rank == 0 else \
            (num_devices - 1 - pipeline_rank) * num_layers_per_vr
   ```

3. **Microbatch Scheduling**: Simpler chunk assignment
   ```python
   def get_model_chunk_id(microbatch_id):
       # No complex VR0→VR2 mapping
       return 0 if microbatch_id < (total_num_microbatches // 2) else 1
   ```

4. **No Same-Device Transitions**: Remove VR0→VR2, VR1→VR3 logic
   ```python
   # This entire section can be removed:
   # if (is_pipeline_last_stage and next_chunk == v_size-2) or ...
   #     detached_output_tensor = output_tensor.detach()
   ```

5. **Asymmetric Support**: Adapt for 2-VR configuration
   ```python
   # Config format: [VR0_layers, VR1_layers] per device
   # Example: [[4,4], [4,4], [4,4], [4,4]]  # 4 devices, 16 layers
   ```

### Expected Benefits

1. **Performance**: ~10-20% reduction in P2P communication overhead
2. **Memory**: Slightly better memory efficiency due to larger chunks
3. **Simplicity**: 30-40% less code complexity in scheduler
4. **Debugging**: Easier to trace microbatch flow through pipeline

### Command Line Usage (Planned)

```bash
# Enable Chimera 2-VR
torchrun pretrain_gpt.py \
    --enable-chimera-schedule \
    --pipeline-model-parallel-size 8 \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --num-layers 96 \
    ...

# Asymmetric Chimera (future)
torchrun pretrain_gpt.py \
    --enable-chimera-schedule \
    --enable-chimera-asymmetric \
    --chimera-asymmetric-config configs/8_devices/chimera_8devices_96layers.json \
    --pipeline-model-parallel-size 8 \
    --num-layers 96 \
    ...
```

### Validation Plan

**Phase 1: Implementation**
- Implement `chimera_2vr.py` based on BitPipe 4-VR
- Remove V-shape logic and same-device transitions
- Add entry point in `schedules.py`

**Phase 2: Testing**
- Unit tests for layer assignment
- Functional tests with small models
- Verify correct microbatch flow

**Phase 3: Performance Comparison**
- Run identical workloads on BitPipe 4-VR vs Chimera 2-VR
- Measure:
  - Pipeline bubble %
  - P2P communication time
  - End-to-end throughput
  - Memory usage
  - Convergence behavior

**Phase 4: Production Ready**
- Asymmetric configuration support
- Profiling integration
- Documentation and examples

### Implementation Deep Dive (October 2025)

#### Critical Implementation Details

**1. Two Types of Gradient Synchronization**

Chimera requires TWO distinct gradient sync mechanisms (often confused):

| Type | Purpose | When | Implementation |
|------|---------|------|----------------|
| **DDP Sync** | Sync across data-parallel replicas | After each backward | `no_sync_func` / `contextlib.nullcontext` |
| **BD Sync** | Sync across bidirectional pipeline pairs | During sync markers (-1) | `allreduce_gradients()` |

**DDP Sync (`no_sync_func`):**
- Controls gradient sync for **Data Parallel** (same layers, different data batches)
- For pipeline-only setups (no data parallelism): Uses `contextlib.nullcontext` (no-op)
- For mixed parallelism: Uses `multi_no_sync()` to disable DDP AllReduce
- **Location**: `chimera_2vr.py` lines 451-466

**BD Sync (`allreduce_gradients()`):**
- **CRITICAL for correctness!** Syncs gradients across bidirectional pairs
- Required because paired devices process SAME layers from opposite directions
- Example: Device 0's VR0 [layers 1-12] ↔ Device 3's VR1 [layers 1-12]
- **Location**: `chimera_2vr.py` lines 414-449

**2. Bidirectional (BD) Group Configuration**

BD groups pair devices that share layers:

```python
# For 4 devices, 48 layers:
Device 0: VR0[1-12]   VR1[37-48]
Device 3: VR0[37-48]  VR1[1-12]   ← Shares layers with Device 0

BD Group 0: [Device 0, Device 3]  # Share layers 1-12 and 37-48
BD Group 1: [Device 1, Device 2]  # Share layers 13-24 and 25-36
```

**Key Insight**: BD groups are based on **physical device ranks**, NOT virtual ranks!
- Each device has 2 VRs (Chimera) or 4 VRs (BitPipe)
- BD group pairs DEVICES, then each device syncs all its VRs
- `allreduce_gradients()` is called once per VR: 2 times for Chimera, 4 times for BitPipe

**BD Group Creation** (`parallel_state.py` lines 262-307):
```python
# Same code works for BOTH BitPipe and Chimera!
if enable_bitpipe_schedule or enable_chimera_schedule:
    num_bd_parallel_groups = pipeline_model_parallel_size // 2
    for j in range(num_bd_parallel_groups):
        bd_ranks = [ranks[0+j], ranks[-1-j]]  # Pair first+j with last-j
```

**Why same code works:**
- BitPipe uses V-shaped `ranks = [0,3,2,1]` → BD groups [0,1], [3,2]
- Chimera uses sequential `ranks = [0,1,2,3]` → BD groups [0,3], [1,2]
- Different inputs, different outputs, both correct for their layer distributions!

**3. V-Shaped Rank Reordering: BitPipe vs Chimera**

**BitPipe**: Reorders physical device ranks into V-shape `[0,3,2,1]`
- **Purpose**: Optimizes P2P communication routing (NOT layer assignment!)
- **Effect**: Changes which physical device sends to which
  - Without V-shape: `0→1→2→3→0` (sequential ring)
  - With V-shape: `0→3→2→1→0` (different ring pattern)
- **Rationale**: May align better with GPU interconnect topology, reduce network contention
- **Location**: `parallel_state.py` lines 248-261

**Chimera**: NO rank reordering (keeps sequential `[0,1,2,3]`)
- **Rationale**:
  - Simpler = easier to understand and debug
  - With only 2 VRs and larger chunks (12 vs 2 layers), communication is less frequent
  - V-shape optimization likely has minimal benefit
  - Sequential ordering is sufficient for correctness

**Key Distinction**:
- ✅ Layer assignment is decoupled (determined by logical pipeline_rank)
- ❌ P2P communication is NOT decoupled (uses `_PIPELINE_GLOBAL_RANKS` array)
- V-shape is a **communication optimization**, not a **correctness requirement**

#### Implementation Challenges Solved

**Challenge 1: `virtual_pipeline_model_parallel_size` Not Set**
- **Error**: `AssertionError: Chimera requires virtual_pipeline_model_parallel_size=2, got None`
- **Cause**: Missing initialization in `arguments.py` (BitPipe had it, Chimera didn't)
- **Solution**: Added Chimera initialization block in `arguments.py` lines 255-293
- **Fix**: Set `args.virtual_pipeline_model_parallel_size = 2` during argument parsing

**Challenge 2: `no_sync_func` is None**
- **Error**: `TypeError: 'NoneType' object is not callable`
- **Cause**: Missing fallback when `config.no_sync_func` is None
- **Solution**: Added fallback logic (lines 451-466 in `chimera_2vr.py`)
  - If model uses DDP: Create `multi_no_sync()`
  - Otherwise: Use `contextlib.nullcontext`

**Challenge 3: Missing Bidirectional Gradient Sync**
- **Bug**: Gradient sync markers (-1) called `enable_grad_sync()` but never `allreduce_gradients()`
- **Impact**: Gradients not averaged across paired devices → incorrect training!
- **Solution**:
  - Added `allreduce_gradients()` function (lines 414-449)
  - Updated sync markers to call it (lines 562-572, 594-604)
  - Extended BD group initialization to Chimera in `parallel_state.py`

**Challenge 4: Missing Transformer Layer Distribution Integration**
- **Bug**: Chimera's `get_chimera_offset()` implemented but NEVER called from `transformer.py`
- **Impact**:
  - Chimera fell back to default interleaved pipeline offset calculation
  - Wrong layer count: Expected 24 layers/device, got only 12
  - Wrong layer offset: Used default calculation instead of bidirectional pairing
  - Training failed before first iteration with tensor shape mismatches
- **Root Cause**:
  - Layer count doubling logic only checked `args.enable_bitpipe_schedule` (line ~1267)
  - Layer offset calculation only had `if args.enable_bitpipe_schedule:` clause, no `elif` for Chimera (line ~1466)
  - Chimera fell through to default `offset = vp_rank * (num_layers // vp_size) + ...` calculation
- **Solution** (Added to `megatron/model/transformer.py`):

  **Fix 1: Layer Count Doubling** (line ~1270)
  ```python
  # If existing bd，double it
  if args.enable_bitpipe_schedule:
      num_layers = (num_layers * 2)
  # Chimera also needs doubled layers (2 VRs per device)
  if hasattr(args, 'enable_chimera_schedule') and args.enable_chimera_schedule:
      num_layers = (num_layers * 2)
  ```

  **Fix 2: Layer Offset Calculation** (line ~1510)
  ```python
  # BitPipe layers
  if args.enable_bitpipe_schedule:
      # ... BitPipe offset calculation ...
  # Chimera 2-VR layers
  elif hasattr(args, 'enable_chimera_schedule') and args.enable_chimera_schedule:
      assert config.virtual_pipeline_model_parallel_size == 2

      from megatron.core.pipeline_parallel.schedule_impl.chimera.chimera_2vr import get_chimera_offset

      # Calculate offset using Chimera's bidirectional pattern
      offset_1based, num_layers_per_vr = get_chimera_offset(
          mpu.get_pipeline_model_parallel_rank(),
          mpu.get_virtual_pipeline_model_parallel_rank(),
          self.num_layers,
          mpu.get_pipeline_model_parallel_world_size()
      )

      # Convert to 0-based offset (Chimera uses 1-based indexing)
      offset = offset_1based - 1
      self.num_layers = num_layers_per_vr
  ```

- **Result**:
  - Correct layer distribution: 4 devices, 48 layers → 12 layers/VR × 2 VRs = 24 layers/device ✅
  - Bidirectional pairing: Device 0 VR0[1-12] ↔ Device 3 VR1[1-12] ✅
  - Debug logging confirms correct layer creation ✅

**Example Output (4 devices, 48 layers):**
```
[CHIMERA LAYER OFFSET] Device 0 | VR 0 | Offset: 0  | Layers: 12 | Creating layers [1..12]
[CHIMERA LAYER OFFSET] Device 0 | VR 1 | Offset: 36 | Layers: 12 | Creating layers [37..48]
[CHIMERA LAYER OFFSET] Device 1 | VR 0 | Offset: 12 | Layers: 12 | Creating layers [13..24]
[CHIMERA LAYER OFFSET] Device 1 | VR 1 | Offset: 24 | Layers: 12 | Creating layers [25..36]
[CHIMERA LAYER OFFSET] Device 2 | VR 0 | Offset: 24 | Layers: 12 | Creating layers [25..36]
[CHIMERA LAYER OFFSET] Device 2 | VR 1 | Offset: 12 | Layers: 12 | Creating layers [13..24]
[CHIMERA LAYER OFFSET] Device 3 | VR 0 | Offset: 36 | Layers: 12 | Creating layers [37..48]
[CHIMERA LAYER OFFSET] Device 3 | VR 1 | Offset: 0  | Layers: 12 | Creating layers [1..12]
```

### Code Structure (Implemented)

```
megatron/core/pipeline_parallel/schedule_impl/chimera/
├── __init__.py
├── chimera_2vr.py           # ✅ Main scheduler (symmetric)
    ├── get_chimera_offset()                    # Layer offset calculation
    ├── build_chimera_forward_schedule()        # Forward microbatch schedule
    ├── build_chimera_backward_schedule()       # Backward microbatch schedule
    ├── allreduce_gradients()                   # BD gradient sync
    └── forward_backward_pipelining_with_chimera_2vr()  # Main entry point

megatron/core/parallel_state.py
└── Lines 262-307: BD group initialization (shared with BitPipe)

megatron/arguments.py
└── Lines 255-293: Chimera argument parsing and validation

# Entry point in schedules.py (line 96-101):
if get_args().enable_chimera_schedule:
    from megatron.core.pipeline_parallel.schedule_impl.chimera import (
        forward_backward_pipelining_with_chimera_2vr,
    )
    return forward_backward_pipelining_with_chimera_2vr
```

### Known Issues and Fixes (November 2025)

#### Issue 1: pre_process/post_process Not Set for Chimera ✅ FIXED

**Problem**: Chimera could not initialize because `is_pipeline_first_stage()` and `is_pipeline_last_stage()` in `parallel_state.py` had no Chimera-specific handling. These functions determine which VR chunks create embeddings and output layers.

**Root Cause**:
- BitPipe had explicit handling in both functions (lines 517-525 and 538-548)
- Chimera fell through to default logic designed for non-bidirectional pipelines
- Result: Device 3, VR1 didn't get `pre_process=True` (needed for backward pipeline embedding)
- Led to tensor shape mismatches and deadlock

**Fix Applied**:
- Added Chimera-specific handling to `is_pipeline_first_stage()` (lines 536-546 in parallel_state.py)
  - Uses IDENTICAL condition to BitPipe: `(D0,VR0) OR (D(N-1),VR1)`
- Added Chimera-specific handling to `is_pipeline_last_stage()` (lines 587-598 in parallel_state.py)
  - Uses DIFFERENT condition from BitPipe: `(D0,VR1) OR (D(N-1),VR0)` (because Chimera has 2 VRs not 4)
- Added VR initialization logging to `training.py` (lines 242-254) to verify flags are correct

**Expected Behavior After Fix**:
```
[VR_INIT] Chimera 2-VR | Device 0 | VR 0 | pre_process=True  | post_process=False
[VR_INIT] Chimera 2-VR | Device 0 | VR 1 | pre_process=False | post_process=True
[VR_INIT] Chimera 2-VR | Device 3 | VR 0 | pre_process=False | post_process=False
[VR_INIT] Chimera 2-VR | Device 3 | VR 1 | pre_process=True  | post_process=False
```

---

#### Issue 2: recv_forward() Returns List Instead of Tensor ⚠️ PENDING FIX

**Problem**: Chimera training fails with `AttributeError: 'list' object has no attribute 'shape'` at line 526 of `chimera_2vr.py`:
```python
input_tensor = recv_forward(tensor_shape, config)
print(f"... input_tensor shape={input_tensor.shape ...}")  # Error: input_tensor is a list!
```

**Root Cause**:
- Chimera imports `recv_forward` from `megatron.core.pipeline_parallel.schedules` (line 39)
- This is a WRAPPER function that handles multiple tensor shapes (for encoder-decoder models)
- The wrapper RETURNS A LIST of tensors, not a single tensor:
  ```python
  def recv_forward(tensor_shapes, config):
      input_tensors = []  # ← Returns list!
      for tensor_shape in tensor_shapes:
          input_tensors.append(p2p_communication.recv_forward(tensor_shape, config))
      return input_tensors  # ← This is [tensor] for GPT, [encoder_tensor, decoder_tensor] for T5
  ```
- Chimera expects a single tensor but gets a list

**How to Fix** (for next session):
Option 1 (Recommended): Use `get_tensor_shapes()` properly
```python
# Replace current code:
tensor_shape = (seq_length, micro_batch_size, config.hidden_size)

# With:
tensor_shapes = get_tensor_shapes(
    rank=mpu.get_pipeline_model_parallel_rank(),
    model_type=model_type,
    seq_length=seq_length,
    micro_batch_size=micro_batch_size,
    decoder_seq_length=None,
    config=config,
)
```

Option 2: Extract from list
```python
input_tensor_list = recv_forward(tensor_shapes, config)
input_tensor = input_tensor_list[0]  # Extract first (and only) tensor for GPT
```

**Files Involved**:
- `megatron/core/pipeline_parallel/schedule_impl/chimera/chimera_2vr.py` (line 524-526)
- `megatron/core/pipeline_parallel/schedules.py` (lines 973-980, the wrapper function)

---

### Open Questions for Future Testing

1. **Performance vs BitPipe**: Does Chimera achieve similar throughput with simpler design?
2. **V-Shape Benefit**: Would Chimera benefit from V-shaped rank reordering?
3. **Pipeline Bubble**: Is bubble size comparable to BitPipe 4-VR?
4. **Network Topology**: How do different GPU interconnects affect the optimal rank ordering?
5. **Asymmetric Mode**: Should Chimera support asymmetric layer distribution?

---

## References

- **Original Paper**: BitPipe: Bidirectional Interleaved Pipeline Parallelism
- **Base Framework**: [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- **Documentation**: See `README.md` and `PROFILING_ARCHITECTURE.md`
