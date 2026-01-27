# BitPipe Concrete Example - Step by Step Trace

This document walks through a **concrete example** with 4 devices, 24 layers, and 4 user microbatches, showing exactly what happens at each step.

---

## Setup

**Command:**
```bash
torchrun --nproc_per_node 4 \
    pretrain_gpt.py \
    --enable-bitpipe-schedule \
    --pipeline-model-parallel-size 4 \
    --micro-batch-size 8 \
    --global-batch-size 32 \
    --num-layers 24 \
    --train-iters 1
```

**Hardware:** 4 GPUs (Device 0, 1, 2, 3)

**Key Parameters:**
- Base layers: 24
- User microbatches: 4 (from global_batch_size / micro_batch_size = 32 / 8)
- Virtual pipeline size: 4 (automatically set for BitPipe)

---

## PHASE 1: LAYER DISTRIBUTION

### Step 1.1: Calculate total layers needed

**File:** `megatron/model/transformer.py:1267`
```python
num_layers = 24  # Input

if args.enable_bitpipe_schedule:
    num_layers = num_layers * 2  # 24 → 48
    # Reason: 4 VRs means we need 2 layers per VR per device
    # Total = 24 layers * 2 = 48 layers
```

### Step 1.2: For each device/VR combination, calculate which layers it gets

**File:** `megatron/model/transformer.py:1466-1510`

Let's trace through all device/VR combinations:

#### Device 0, VR 0 (pipeline_rank=0, vp_rank=0)
```python
# File: megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py:66-96
# get_bitpipe_offset(pipeline_rank=0, vp_rank=0, num_layers_per_vr=6, num_devices=4)

vp_idx = vp_rank if vp_rank < 2 else (vp_rank - 1) % 2
       = 0 if 0 < 2 else ...
       = 0

offset = (
    pipeline_rank * num_layers_per_vr +
    vp_idx * (pipeline_world_size - 1 - 2 * pipeline_rank) * num_layers_per_vr +
    (vp_rank // 2) * (total_layers // 2)
)
       = (0 * 6) + (0 * (4 - 1 - 0) * 6) + (0 // 2) * 24
       = 0 + 0 + 0
       = 0

Result: Device 0, VR0 gets layers [0, 1]  (offset=0, count=2... wait let's recalculate)
```

# Complete V-Shaped Layer Assignment (4 devices, 48 layers total)

Device 0:
  ├─ VR0: layers [1, 2, 3]       
  ├─ VR1: layers [10, 11, 12]    
  ├─ VR2: layers [22, 23, 24]    
  └─ VR3: layers [13, 14, 15]   

Device 1:
  ├─ VR0: layers [4, 5, 6]    
  ├─ VR1: layers [7, 8, 9]   
  ├─ VR2: layers [19, 20, 21]     
  └─ VR3: layers [16, 17, 18]     

Device 2:
  ├─ VR0: layers [7, 8, 9]       
  ├─ VR1: layers [4, 5, 6]    
  ├─ VR2: layers [16, 17, 18]     
  └─ VR3: layers [19, 20, 21]   

Device 3:
  ├─ VR0: layers [10, 11, 12]      
  ├─ VR1: layers [1, 2, 3]       
  ├─ VR2: layers [13, 14, 15]     
  └─ VR3: layers [22, 23, 24]   


### Step 1.3: Each device creates its model chunks

For **Device 0**, the following happens in parallel across all devices:

```python
# Device 0, VR0 execution in megatron/model/transformer.py
# (Same happens for VR1, VR2, VR3 but on different ranks)

class GPT(nn.Module):
    def __init__(self):
        # ... setup ...

        # For Device 0, VR 0:
        pipeline_rank = 0
        vp_rank = 0

        # Calculate offset
        offset = 0  # (from get_bitpipe_offset)
        num_layers = 2

        # Create ONLY layers [0, 1]
        self.layers = nn.ModuleList([
            TransformerLayer(layer_id=0, ...),
            TransformerLayer(layer_id=1, ...),
        ])
```

**Result after all device/VR combinations are initialized:**

```
Device 0 model chunk 0 (VR0): Transformer layers [0, 1]
Device 0 model chunk 1 (VR1): Transformer layers [14, 15]
Device 0 model chunk 2 (VR2): Transformer layers [32, 33]
Device 0 model chunk 3 (VR3): Transformer layers [16, 17]

Device 1 model chunk 0 (VR0): Transformer layers [2, 3]
Device 1 model chunk 1 (VR1): Transformer layers [12, 13]
... and so on for all devices and VRs
```

---

## PHASE 2: SCHEDULER SELECTION

### Step 2.1: Training loop calls get_forward_backward_func()

**File:** `megatron/training.py:474`
```python
forward_backward_func = get_forward_backward_func()
```

### Step 2.2: Dispatcher checks which scheduler to use

**File:** `megatron/core/pipeline_parallel/schedules.py:104-109`
```python
if get_args().enable_bitpipe_schedule:
    print("[DEBUG] BitPipe 4-VR schedule ENABLED")
    from megatron.core.pipeline_parallel.schedule_impl.bitpipe import (
        forward_backward_pipelining_with_bitpipe_4vr,
    )
    return forward_backward_pipelining_with_bitpipe_4vr
```

**Result:** `forward_backward_func` now points to `forward_backward_pipelining_with_bitpipe_4vr`

---

## PHASE 3: SCHEDULER EXECUTION

### Step 3.1: Training calls the scheduler with data

**File:** `megatron/training.py:480-488`
```python
losses_reduced = forward_backward_func(
    forward_step_func=forward_step,
    data_iterator=[iter0, iter1, iter2, iter3],  # One for each VR
    model=[model0, model1, model2, model3],      # One for each VR
    num_microbatches=4,                          # User-specified
    seq_length=256,
    micro_batch_size=8,
    ...
)
```

**Inputs to scheduler:**
- `num_microbatches=4` (user-specified)
- `model=[chunk0, chunk1, chunk2, chunk3]` (4 chunks)
- `data_iterator=[iter0, iter1, iter2, iter3]` (4 iterators)

### Step 3.2: Scheduler initializes

**File:** `megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py:260-380`

```python
def forward_backward_pipelining_with_bitpipe_4vr(...):
    # Setup
    pipeline_parallel_size = 4  # Number of devices
    pipeline_parallel_rank = 0  # Current device (varies per rank)
    num_model_chunks = 4        # VRs per device

    # CRITICAL: Double microbatches for bidirectional execution
    total_num_microbatches = 4 * (4 // 2)  # = 4 * 2 = 8
    # User said 4, BitPipe creates 8!

    # Calculate warmup microbatches
    num_warmup_microbatches = pipeline_parallel_size  # 4
    num_microbatches_remaining = 8 - 4 = 4

    print("Total microbatches: 8")
    print("Warmup: 4")
    print("Remaining: 4")
```

**Result:**
- Total microbatches to execute: 8 (doubled!)
- Warmup iterations: 4
- Steady-state iterations: 4 (not used here since 4 remaining ≥ 4)

### Step 3.3: Build forward and backward schedules

**File:** `megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py:180-250`

The scheduler computes optimal execution order:

```python
# After complex scheduling algorithm (considering V-shaped layout)
# Forward order (example for rank 0):
forward_schedule = [0, 1, 2, 10, 3, 11, 8, 9]

# Backward order (example for rank 0):
backward_schedule = [8, 9, 10, 2, 11, 3, 0, 1]

# These schedules are DIFFERENT for each device rank!
# Device 0 executes different microbatches than Device 3
```

The schedules ensure:
- Data flows properly through all devices
- All 4 devices always have work to do (minimize idle time)
- Gradients are synchronized at the right times

### Step 3.4: Setup data structures

**File:** `megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py:380-410`

```python
# For each VR, track forward pass outputs
input_tensors = [[], [], [], []]       # One list per VR
output_tensors = [[], [], [], []]      # One list per VR
output_tensor_grads = [[], [], [], []] # One list per VR for backward

# Track synchronized VRs (for gradient sync)
synchronized_model_chunks = set()

# Create profiler if enabled
profiler = initialize_bitpipe_profiler(args) if args.enable_bitpipe_profiling else None

# Tensor shape
tensor_shape = (256, 8, 2048)  # (seq_len, batch_size, hidden_size)
```

---

## PHASE 4: WARMUP PHASE (Fill Pipeline)

### Step 4.1: Warmup iteration 0 (All devices)

The following happens **in parallel** across all 4 devices:

#### Device 0:
```python
# Iteration k=0 of warmup
microbatch_id = forward_schedule[0] = 0
forward_model_chunk_id = get_model_chunk_id(0) = 0  # VR0

# Rank 0 is first stage, so don't recv
if parallel_state.is_pipeline_first_stage():
    recv_prev = False
input_tensor = None  # Create new input from data

# Forward pass
output_tensor = forward_step(
    forward_step_func=forward_step,
    data_iterator=data_iterator[0],    # Iterator for VR0
    model=model[0],                    # VR0's model (layers [0,1])
    input_tensor=None,
    ...
)

# Rank 0 is first stage, so send output to Device 1
if not parallel_state.is_pipeline_last_stage():
    send_forward_recv_forward(output_tensor, ...)

output_tensors[0].append(output_tensor)
```

#### Device 1:
```python
# Warmup iteration k=0
microbatch_id = forward_schedule[0] = 0
forward_model_chunk_id = get_model_chunk_id(0) = 0  # VR0

# Rank 1 is NOT first stage, so receive from Device 0
if not parallel_state.is_pipeline_first_stage():
    input_tensor = recv_forward(...)  # Waits for Device 0's output

# Forward pass
output_tensor = forward_step(
    forward_step_func=forward_step,
    data_iterator=data_iterator[0],    # Iterator for VR0
    model=model[0],                    # VR0's model (layers [2,3])
    input_tensor=input_tensor,
    ...
)

# Device 1 is not last stage, so send to Device 2
if not parallel_state.is_pipeline_last_stage():
    send_forward_recv_forward(output_tensor, ...)

output_tensors[0].append(output_tensor)
```

#### Device 2:
```python
# Similar to Device 1
# Receives from Device 1, processes, sends to Device 3
input_tensor = recv_forward(...)
output_tensor = forward_step(...)
send_forward_recv_forward(output_tensor, ...)
```

#### Device 3:
```python
# Device 3 is last stage
input_tensor = recv_forward(...)  # Receives from Device 2

# Forward pass
output_tensor = forward_step(
    ...,
    input_tensor=input_tensor,
    ...
)

# Rank 3 is last stage - DON'T send anywhere!
# This prevents the deadlock that Chimera has!
output_tensor = None  # Important!

# Store loss for later reduction
if output_tensor is not None:
    losses_reduced.append(output_tensor)
```

**Timeline after warmup iteration 0:**
```
Time t0→t1:   Device 0 computes, sends output
Time t1→t2:   Device 1 receives from 0, computes, sends
Time t2→t3:   Device 2 receives from 1, computes, sends
Time t3→t4:   Device 3 receives from 2, computes

Result: Microbatch 0 has moved through all 4 devices
```

### Step 4.2: Warmup iterations 1-3

Same pattern repeats with different microbatch IDs:

```python
# Warmup iteration 1
microbatch_id = forward_schedule[1]
# ... (repeat for all devices)

# Warmup iteration 2
microbatch_id = forward_schedule[2]
# ... (repeat for all devices)

# Warmup iteration 3
microbatch_id = forward_schedule[3]
# ... (repeat for all devices)
```

**Result after warmup (4 iterations):**
- All 4 devices have data in flight
- Pipeline is "full" - at any moment, each device is computing
- Ready to start the efficient 1F1B phase

---

## PHASE 5: STEADY STATE (1F1B - One Forward, One Backward)

### Step 5.1: Iteration 0 of steady state

**Simultaneously on all devices:**

#### Device 0 (Forward):
```python
# Forward for microbatch at forward_schedule[4]
forward_k = num_warmup_microbatches + 0 = 4
forward_id = forward_schedule[4]

forward_model_chunk_id = get_model_chunk_id(forward_id)
output_tensor = forward_step(...)

# Since Device 0 is NOT last stage:
# Send forward output AND receive backward gradient (overlapped!)
output_tensor_grad = send_forward_recv_backward(output_tensor, ...)
output_tensor_grads[...].append(output_tensor_grad)
```

#### Device 1 (Backward):
```python
# Backward for microbatch at backward_schedule[0]
bwd_id = backward_schedule[0]

backward_model_chunk_id = get_model_chunk_id(bwd_id)

# Receive backward gradient from Device 2
input_tensor_grad = recv_backward(...)

# Pop stored tensors from warmup
output_tensor = output_tensors[...].pop(0)
input_tensor = input_tensors[...].pop(0)

# Backward pass
output_tensor_grad = backward_step(
    input_tensor, output_tensor, input_tensor_grad, ...
)

# Send backward gradient to Device 0
send_backward(output_tensor_grad, ...)
```

#### Device 2 (Mixed):
```python
# Similar: processes forward from earlier
# and backward from earlier
```

#### Device 3 (Backward):
```python
# Device 3 is last stage
# Backward doesn't send upstream
input_tensor_grad = recv_backward(...)  # From Device 2

output_tensor = output_tensors[...].pop(0)
input_tensor = input_tensors[...].pop(0)

output_tensor_grad = backward_step(...)

# Device 3 is last stage - don't send backward!
# Backward gradient ends here
```

**Timeline - beautifully overlapped!**
```
Device 0: [F: MB4] [B: MB0] [F: MB5] [B: MB1] ...
Device 1:       [B: MB0] [F: MB4] [B: MB1] [F: MB5] ...
Device 2:              [B: MB0] [F: MB4] [B: MB1] ...
Device 3:                     [B: MB0] [F: MB4] ...

Key: While Device 0 computes backward, Device 1 can compute forward!
     This overlap maximizes GPU utilization!
```

### Step 5.2: Gradient Synchronization

At critical points during backward pass:

```python
# File: bitpipe_4vr.py:475-479
if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(microbatch_id):
    enable_grad_sync()
    synchronized_model_chunks.add(model_chunk_id)
```

When a VR finishes its last microbatch, trigger:

```python
# File: bitpipe_4vr.py:432-457
def allreduce_gradients(model):
    """Sync gradients across BD group pairs"""
    if parallel_state.is_rank_in_bd_group():
        # All devices in BD group synchronize
        # Example: Device 0 and Device 3 sync together
        torch.distributed.all_reduce(
            gradients,
            group=parallel_state.get_bd_parallel_group()
        )
```

**BD Group Synchronization:**
```
Before allreduce:
Device 0 VR0 gradients: [g1_from_forward, g1_from_backward]  (from forward + backward)
Device 3 VR1 gradients: [g1_from_forward, g1_from_backward]  (from forward + backward)

Both processed the SAME layers, so gradients must be averaged:

AllReduce:
Device 0 & 3: all_reduce([g1, g1]) → [(g1+g1)/2, (g1+g1)/2]

After allreduce:
Device 0 VR0 gradients: [avg_g1, avg_g1]  (synchronized!)
Device 3 VR1 gradients: [avg_g1, avg_g1]  (synchronized!)
```

---

## PHASE 6: COOLDOWN (Drain Pipeline)

### Step 6.1: Cooldown iteration 0

After all forward passes complete, finish remaining backward passes:

```python
remaining_backward = backward_schedule[num_microbatches_remaining:]

for i, bwd_microbatch_id in enumerate(remaining_backward):
    # Similar to steady state but ONLY backward
    # No new forward passes

    input_tensor_grad = recv_backward(...)
    output_tensor = output_tensors[...].pop(0)
    input_tensor = input_tensors[...].pop(0)

    output_tensor_grad = backward_step(...)

    if not parallel_state.is_pipeline_first_stage():
        send_backward(output_tensor_grad, ...)
```

**Result:** All microbatches have completed forward and backward passes

---

## PHASE 7: AFTER SCHEDULER

### Step 7.1: Training loop resumes

**File:** `megatron/training.py:500-544`

```python
# Scheduler returns:
losses_reduced = [loss_dict_0, loss_dict_1, ..., loss_dict_7]

# Now reduce gradients across data-parallel replicas (if used)
optimizer.reduce_model_grads(args, timers)

# Update weights
update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)

# Aggregate losses for logging
if is_pipeline_stage_containing_loss():
    loss_reduced = {}
    for key in losses_reduced[0]:
        losses_reduced_for_key = [x[key] for x in losses_reduced]
        loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
    print(f"Loss: {loss_reduced['lm loss']}")
```

---

## Summary

### What happened in one training iteration:

1. **Layer Distribution:** 48 layers split across 4 devices in V-shaped pattern
2. **Scheduler Selection:** BitPipe 4-VR scheduler selected
3. **Warmup:** 4 forward passes to fill pipeline
4. **1F1B:** 4 pairs of (forward, backward) with overlap
5. **Cooldown:** Remaining backward passes
6. **Gradient Sync:** AllReduce within BD groups at critical points
7. **Optimizer:** Update weights using synchronized gradients

### Key advantages:

- **Doubled throughput:** 8 effective microbatches from 4 user inputs
- **Overlapped computation:** Forward on one device while backward on another
- **Correct gradients:** BD groups ensure paired devices average their results
- **No deadlocks:** Boundary checks prevent P2P on first/last stages
- **Pipeline efficiency:** Minimal idle time (bubbles)

### Key files involved:

1. `megatron/model/transformer.py` - Layer distribution
2. `megatron/core/pipeline_parallel/schedules.py` - Scheduler dispatcher
3. `megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py` - Main scheduler
4. `megatron/core/pipeline_parallel/p2p_communication.py` - P2P operations
5. `megatron/core/parallel_state.py` - Parallel groups and boundary checks
6. `megatron/training.py` - Main training loop

