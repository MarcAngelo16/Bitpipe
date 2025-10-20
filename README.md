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
- Phase 5: Implement Chimera 2-VR scheduler


### Configuration Files
- **Asymmetric configs**: `asymmetric_bitpipe/configs/{N}_devices/*.json`
- **Training scripts**: `singlenode_*.sh`, `multinode_*.sh`
- **Examples**: `gpt_dummy.py`, `bert_dummy.py`

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

### Known Limitations

1. **Pipeline size must be even** (bidirectional requirement)
2. **Virtual pipeline size = 4** (hardcoded for BitPipe)
3. **Not compatible with encoder-decoder models**
4. **Requires `CUDA_DEVICE_MAX_CONNECTIONS=1`**

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

## Troubleshooting

### Common Issues

**Error: "Pipeline size must be even"**
- BitPipe requires even number of devices for bidirectional pairing
- Use 2, 4, 6, 8, 10, 12... devices

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


## References

- **Original Paper**: BitPipe: Bidirectional Interleaved Pipeline Parallelism
- **Base Framework**: [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- **Documentation**: See `README.md` and `PROFILING_ARCHITECTURE.md`
