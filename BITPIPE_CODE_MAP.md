# BitPipe Code Map - Quick Reference

This document shows the file locations and key functions for BitPipe execution.

---

## File Structure Overview

```
/workspace/Bitpipe/
├── pretrain_gpt.py                          [ENTRY POINT]
│   └─ Contains model_provider() and forward_step()
│
├── megatron/
│   ├── training.py                          [MAIN TRAINING LOOP]
│   │   ├─ pretrain()                        Line: 54-600
│   │   │   ├─ initialize_megatron()         Line: 91
│   │   │   ├─ build model                   Line: 150-250
│   │   │   ├─ get_forward_backward_func()   Line: 474
│   │   │   ├─ forward_backward_func()       Line: 480 [SCHEDULER CALL]
│   │   │   ├─ optimizer.step()              Line: 510
│   │   │   └─ return loss, grad_norm        Line: 544
│   │
│   ├── arguments.py                         [ARGUMENT PARSING]
│   │   └─ BitPipe initialization           Line: 255-293
│   │       ├─ Set virtual_pipeline_model_parallel_size = 4
│   │       ├─ Parse --enable-bitpipe-schedule flag
│   │       └─ Initialize BD groups
│   │
│   ├── model/
│   │   └── transformer.py                   [LAYER DISTRIBUTION]
│   │       ├─ TransformerLayer.__init__()   Line: 1250-1550
│   │       │   ├─ Double layers for BitPipe Line: 1267
│   │       │   ├─ Calculate offset         Line: 1466-1510
│   │       │   └─ Create assigned layers   Line: 1520-1550
│   │       └─ Architecture: VRs distributed across devices
│   │
│   ├── initialize.py                        [MEGATRON INITIALIZATION]
│   │   └─ initialize_megatron()
│   │       ├─ Parse args
│   │       ├─ Initialize torch.distributed
│   │       └─ Setup parallel groups
│   │
│   └── core/
│       ├── pipeline_parallel/
│       │   ├── schedules.py                 [SCHEDULER DISPATCHER]
│       │   │   ├─ get_forward_backward_func() Line: 21-119
│       │   │   │   ├─ Check enable_chimera_schedule  Line: 96
│       │   │   │   ├─ Check enable_bitpipe_schedule  Line: 104
│       │   │   │   └─ Return appropriate scheduler
│       │   │   │
│       │   │   └─ Other schedule functions:
│       │   │       ├─ forward_backward_pipelining_without_interleaving()
│       │   │       ├─ forward_backward_pipelining_with_interleaving()
│       │   │       └─ forward_backward_no_pipelining()
│       │   │
│       │   ├── schedule_impl/
│       │   │   └── bitpipe/
│       │   │       ├── __init__.py          [BITPIPE PACKAGE]
│       │   │       │   └─ Exports forward_backward_pipelining_with_bitpipe_4vr
│       │   │       │
│       │   │       └── bitpipe_4vr.py       [BITPIPE SCHEDULER]
│       │   │           ├─ get_bitpipe_offset()  Line: 66-96
│       │   │           │   └─ Calculate V-shaped layer assignment
│       │   │           │
│       │   │           ├─ forward_backward_pipelining_with_bitpipe_4vr()
│       │   │           │   Line: 130-800 [MAIN SCHEDULER]
│       │   │           │   ├─ Setup phase              Line: 260-380
│       │   │           │   ├─ Warmup phase            Line: 514-595
│       │   │           │   ├─ Steady-state phase      Line: 596-650
│       │   │           │   ├─ Cooldown phase          Line: 651-730
│       │   │           │   └─ Return losses_reduced    Line: 750
│       │   │           │
│       │   │           ├─ allreduce_gradients()       Line: 432-457
│       │   │           │   └─ Sync gradients across BD groups
│       │   │           │
│       │   │           ├─ get_model_chunk_id()        Line: 100-130
│       │   │           │   └─ Map microbatch ID to VR chunk
│       │   │           │
│       │   │           └─ Helper functions:
│       │   │               ├─ forward_step_helper()    Line: 368-430
│       │   │               └─ backward_step_helper()   Line: 460-512
│       │   │
│       │   ├── p2p_communication.py         [P2P COMMUNICATION]
│       │   │   ├─ send_forward()
│       │   │   ├─ recv_forward()
│       │   │   ├─ send_backward()
│       │   │   ├─ recv_backward()
│       │   │   └─ send_forward_recv_backward()  [Combined for efficiency]
│       │   │
│       │   ├── bitpipe_profiler.py          [PROFILING]
│       │   │   ├─ BitPipeProfiler class
│       │   │   ├─ record_microbatch()
│       │   │   └─ save_profile()
│       │   │
│       │   └── asymmetric/
│       │       └── config_utils.py          [ASYMMETRIC CONFIG]
│       │           ├─ get_asymmetric_offset()
│       │           └─ load_asymmetric_config()
│       │
│       ├── parallel_state.py                [PARALLEL GROUP MANAGEMENT]
│       │   ├─ get_pipeline_model_parallel_rank()     Line: XXX
│       │   ├─ get_pipeline_model_parallel_world_size() Line: XXX
│       │   ├─ get_virtual_pipeline_model_parallel_rank() Line: XXX
│       │   ├─ is_pipeline_first_stage()      Line: XXX [CRITICAL!]
│       │   ├─ is_pipeline_last_stage()       Line: XXX [CRITICAL!]
│       │   ├─ is_rank_in_bd_group()          Line: XXX
│       │   ├─ get_bd_parallel_group()        Line: XXX
│       │   └─ BD group initialization        Line: 262-307
│       │
│       └── utils.py
│           └─ get_model_config()
│
└── asymmetric_bitpipe/
    └── configs/
        ├── 4_devices/
        ├── 8_devices/
        └── 12_devices/
            └── *.json  [ASYMMETRIC CONFIGURATIONS]
```

---

## Execution Flow - File Sequence

### 1. ENTRY POINT
```
┌─ pretrain_gpt.py
│  └─ Calls pretrain(
│         model_provider=gpt_model_provider,
│         forward_step_func=forward_step,
│         ...
│     )
```

### 2. INITIALIZATION
```
megatron/training.py:91
└─ initialize_megatron()
   ├─ megatron/initialize.py: Parse arguments
   └─ megatron/arguments.py:255: Check --enable-bitpipe-schedule
      └─ Set virtual_pipeline_model_parallel_size = 4
```

### 3. MODEL BUILDING
```
megatron/training.py:150-250
└─ model = model_provider()
   └─ megatron/model/transformer.py:1250-1550
      ├─ TransformerLayer.__init__() called for each VR
      ├─ Double layers (for BitPipe)         Line: 1267
      ├─ Calculate offset (V-shaped pattern) Line: 1466
      └─ Create only assigned layers         Line: 1520
```

### 4. SCHEDULER SELECTION
```
megatron/training.py:474
└─ get_forward_backward_func()
   └─ megatron/core/pipeline_parallel/schedules.py:21-119
      ├─ Check enable_bitpipe_schedule       Line: 104
      └─ Return forward_backward_pipelining_with_bitpipe_4vr
```

### 5. SCHEDULER EXECUTION
```
megatron/training.py:480
└─ forward_backward_func(
       forward_step_func=...,
       data_iterator=...,
       model=...,
       num_microbatches=4,
       ...
   )
   └─ megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py:130-800
      ├─ Setup (compute schedules)                      Line: 260-380
      ├─ Warmup phase                                   Line: 514-595
      │  └─ P2P via: megatron/core/pipeline_parallel/p2p_communication.py
      │     ├─ recv_forward()
      │     ├─ send_forward()
      │     └─ Boundary checks via: megatron/core/parallel_state.py
      │        ├─ is_pipeline_first_stage()
      │        └─ is_pipeline_last_stage()
      ├─ Steady-state (1F1B) phase                       Line: 596-650
      ├─ Cooldown phase                                  Line: 651-730
      └─ Gradient sync (critical!)                       Line: 432-457
         ├─ allreduce_gradients()
         └─ Use BD groups: megatron/core/parallel_state.py:262-307
            └─ get_bd_parallel_group()
```

### 6. AFTER SCHEDULER RETURNS
```
megatron/training.py:500-544
├─ optimizer.reduce_model_grads()  [DDP sync]
├─ optimizer.step()                [Update weights]
├─ Aggregate losses                [Logging]
└─ Loop back to step 4 for next iteration
```

---

## Key Function Signatures

### Layer Distribution Function
```python
# File: megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py:66-96
def get_bitpipe_offset(pipeline_rank, vp_rank, num_layers_per_vr, num_devices):
    """
    Calculate which layers to assign to current device/VR.

    Args:
        pipeline_rank: Device index (0 to num_devices-1)
        vp_rank: Virtual rank (0, 1, 2, 3 for BitPipe)
        num_layers_per_vr: Layers per virtual rank
        num_devices: Total pipeline devices

    Returns:
        (offset, num_layers) tuple
        - offset: Starting layer index for this VR
        - num_layers: Number of layers this VR handles
    """
```

### Scheduler Function
```python
# File: megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py:130-800
def forward_backward_pipelining_with_bitpipe_4vr(
    *,
    forward_step_func,           # User's forward function
    data_iterator,               # List of data iterators (one per VR)
    model,                        # List of model chunks (one per VR)
    num_microbatches,            # User-specified count
    seq_length,
    micro_batch_size,
    decoder_seq_length=None,
    forward_only=False,
    collect_non_loss_data=False,
):
    """
    Main BitPipe scheduler orchestrating all forward/backward passes.

    Returns:
        losses_reduced: List of loss dictionaries per microbatch
    """
```

### Boundary Check Functions
```python
# File: megatron/core/parallel_state.py
def is_pipeline_first_stage(ignore_virtual=False):
    """Check if current rank is first pipeline stage."""
    # Used to skip recv_forward() on first stage

def is_pipeline_last_stage(ignore_virtual=False):
    """Check if current rank is last pipeline stage."""
    # Used to set output_tensor=None on last stage (don't send)
```

### Gradient Synchronization Function
```python
# File: megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py:432-457
def allreduce_gradients(model):
    """
    Synchronize gradients across bidirectional pipeline pairs.

    Why needed:
    - Paired devices (e.g., Device 0 & 3) share layers
    - Both accumulate gradients for same weights
    - Must average gradients for correctness
    """
```

---

## Critical Code Locations

### DEADLOCK PREVENTION
```python
# File: megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py:548-555
recv_prev = True
if parallel_state.is_pipeline_first_stage():
    recv_prev = False  # ← DO NOT receive if first stage!

if parallel_state.is_pipeline_last_stage():
    output_tensor = None  # ← DO NOT send if last stage!
```

**Why Chimera deadlocks:** Missing these checks!

### LAYER ASSIGNMENT (V-SHAPED PATTERN)
```python
# File: megatron/model/transformer.py:1466-1510
if args.enable_bitpipe_schedule:
    from megatron.core.pipeline_parallel.schedule_impl.bitpipe import get_bitpipe_offset

    pipeline_rank = mpu.get_pipeline_model_parallel_rank()
    vp_rank = mpu.get_virtual_pipeline_model_parallel_rank()

    offset, num_layers = get_bitpipe_offset(
        pipeline_rank, vp_rank,
        num_layers_per_vr,
        mpu.get_pipeline_model_parallel_world_size()
    )
```

### SCHEDULER DISPATCHER
```python
# File: megatron/core/pipeline_parallel/schedules.py:21-119
def get_forward_backward_func():
    if get_args().enable_bitpipe_schedule:
        from megatron.core.pipeline_parallel.schedule_impl.bitpipe import (
            forward_backward_pipelining_with_bitpipe_4vr,
        )
        return forward_backward_pipelining_with_bitpipe_4vr
    # ... other schedulers ...
```

### BD GROUP INITIALIZATION
```python
# File: megatron/core/parallel_state.py:262-307
if enable_bitpipe_schedule:
    num_bd_parallel_groups = pipeline_model_parallel_size // 2
    for j in range(num_bd_parallel_groups):
        bd_ranks = [ranks[0+j], ranks[-1-j]]  # Pair first+j with last-j
        # Create NCCL group for gradient synchronization
```

---

## How to Debug BitPipe Execution

### 1. Add prints at each phase:

```python
# In bitpipe_4vr.py

# Setup phase
print(f"[SETUP] Pipeline size: {pipeline_parallel_size}, Rank: {pipeline_parallel_rank}")
print(f"[SETUP] Total microbatches: {total_num_microbatches}")
print(f"[SETUP] Warmup: {num_warmup_microbatches}, Remaining: {num_microbatches_remaining}")

# Warmup phase
print(f"[WARMUP {i}] MB {microbatch_id}, VR {model_chunk_id}")
print(f"[WARMUP {i}] First stage: {parallel_state.is_pipeline_first_stage()}")
print(f"[WARMUP {i}] Last stage: {parallel_state.is_pipeline_last_stage()}")

# Steady state
print(f"[STEADY {i}] Forward: MB {fwd_microbatch_id}, Backward: MB {bwd_microbatch_id}")

# Cooldown
print(f"[COOLDOWN {i}] Backward: MB {bwd_microbatch_id}")
```

### 2. Verify layer distribution:

```bash
grep -n "Creating layers\|Offset:" /path/to/logs
# Should show V-shaped pattern:
# Device 0, VR0: Offset 0, Layers 2
# Device 0, VR1: Offset 6, Layers 2
# Device 0, VR2: Offset 14, Layers 2
# Device 0, VR3: Offset 8, Layers 2
```

### 3. Check scheduler selection:

```bash
grep -n "\[DEBUG\].*schedule ENABLED" /path/to/logs
# Should show: "[DEBUG] BitPipe 4-VR schedule ENABLED"
```

### 4. Monitor P2P communication:

```python
# Add prints in p2p_communication.py
print(f"[P2P] send_forward from rank {rank}")
print(f"[P2P] recv_forward to rank {rank}")
```

---

## Summary

- **Entry:** `pretrain_gpt.py` → `megatron/training.py:pretrain()`
- **Initialization:** `megatron/initialize.py` + `megatron/arguments.py`
- **Layer Assignment:** `megatron/model/transformer.py:TransformerLayer`
- **Scheduler Selection:** `megatron/core/pipeline_parallel/schedules.py:get_forward_backward_func()`
- **Scheduler Execution:** `megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py`
- **P2P Communication:** `megatron/core/pipeline_parallel/p2p_communication.py`
- **Parallel State:** `megatron/core/parallel_state.py`
- **Gradient Sync:** `bitpipe_4vr.py:allreduce_gradients()` + BD groups

