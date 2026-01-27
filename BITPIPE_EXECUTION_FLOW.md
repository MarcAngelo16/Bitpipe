# BitPipe Complete Execution Flow - Step by Step Guide

This guide traces the complete flow from program start to training completion, showing where each piece of code is located.

---

## PHASE 0: PROGRAM INITIALIZATION

### Step 0.1: Start the training script
**File:** `pretrain_gpt.py` or `gpt_dummy.py`

```bash
python pretrain_gpt.py \
    --enable-bitpipe-schedule \
    --pipeline-model-parallel-size 4 \
    --num-layers 24 \
    --micro-batch-size 8 \
    --global-batch-size 32 \
    ...
```

**What happens:** Command line arguments are parsed and stored in `args` object.

---

### Step 0.2: Initialize Megatron Framework
**File:** `megatron/training.py:91-92`
**Code:**
```python
initialize_megatron(extra_args_provider=extra_args_provider,
                    args_defaults=args_defaults)
```

**What happens:**
- Parses command line arguments
- Initializes distributed PyTorch
- Sets up parallel groups (data parallel, tensor parallel, pipeline parallel)
- Creates global `args` object accessible via `get_args()`

**Important check:**
```python
# File: megatron/arguments.py:255-293
if args.enable_bitpipe_schedule:
    args.virtual_pipeline_model_parallel_size = 4  # BitPipe uses 4 VRs
    # ... other BitPipe initialization
```

---

## PHASE 1: MODEL BUILDING

### Step 1.1: Create Model Provider Function
**File:** `pretrain_gpt.py` defines the model provider

**What it does:** Returns a function that creates the GPT model

---

### Step 1.2: Build Model Chunks (Multiple copies for interleaving)
**File:** `megatron/training.py:150-250` (main training function)
**Code:**
```python
model = model_provider()
# For BitPipe: model is a list of 4 model chunks
# model = [GPT_chunk_0, GPT_chunk_1, GPT_chunk_2, GPT_chunk_3]
```

**What happens:**
- For BitPipe: Creates 4 copies of the model (one for each virtual rank)
- Each copy is wrapped in DDP for gradient synchronization
- Each copy is then wrapped in Float16Module for mixed precision

---

### Step 1.3: Distribute Layers Across Devices
**File:** `megatron/model/transformer.py:1250-1550` (in TransformerLayer __init__)

This is the **CRITICAL** step where layers get assigned to devices and VRs.

#### 1.3a: Calculate total layers (accounts for virtual pipeline)
**File:** `megatron/model/transformer.py:1267-1271`
```python
if args.enable_bitpipe_schedule:
    # Double the layers for bidirectional execution
    num_layers = num_layers * 2  # 24 → 48 layers total
```

**Why:** BitPipe has 4 VRs per device, so we need 2x the layers (for 2x pipelines)

#### 1.3b: Calculate offset for current device/VR combination
**File:** `megatron/model/transformer.py:1466-1510`
```python
if args.enable_bitpipe_schedule:
    # Import the offset calculation function
    from megatron.core.pipeline_parallel.schedule_impl.bitpipe import get_bitpipe_offset

    # Get current device and VR rank
    pipeline_rank = mpu.get_pipeline_model_parallel_rank()
    vp_rank = mpu.get_virtual_pipeline_model_parallel_rank()

    # Calculate offset using V-shaped pattern
    offset, num_layers = get_bitpipe_offset(
        pipeline_rank, vp_rank,
        num_layers_per_vr,  # total_layers // (num_devices * 2)
        mpu.get_pipeline_model_parallel_world_size()
    )
```

**What offset calculation does (for 4 devices, 24 base layers → 48 total):**
```
Device Layout:
Device 0: VR0[0,1]   VR1[6,7]   VR2[14,15] VR3[8,9]
Device 1: VR0[2,3]   VR1[4,5]   VR2[12,13] VR3[10,11]
Device 2: VR0[4,5]   VR1[2,3]   VR2[10,11] VR3[12,13]
Device 3: VR0[6,7]   VR1[0,1]   VR2[8,9]   VR3[14,15]

# V-shaped pattern ensures optimal P2P communication
# See get_bitpipe_offset() in bitpipe_4vr.py:66-96
```

#### 1.3c: Create only assigned layers for this VR
**File:** `megatron/model/transformer.py:1520-1550`
```python
# Example for Device 0, VR0:
# - Offset: 0, Num_layers: 2
# - Creates layers 0-1 out of 0-47

layer_indices = list(range(offset, offset + num_layers))
self.layers = torch.nn.ModuleList(
    [TransformerLayer(...) for _ in layer_indices]
)
```

**Result:** Each device/VR combination has DIFFERENT layers:
```
Device 0, VR0: Has layers [0, 1]
Device 0, VR1: Has layers [6, 7]
Device 0, VR2: Has layers [14, 15]
Device 0, VR3: Has layers [8, 9]
```

---

## PHASE 2: SCHEDULER SELECTION

### Step 2.1: Retrieve Forward-Backward Function
**File:** `megatron/training.py:474`
**Code:**
```python
forward_backward_func = get_forward_backward_func()
```

### Step 2.2: Determine Which Scheduler to Use
**File:** `megatron/core/pipeline_parallel/schedules.py:21-119`
**Code:**
```python
def get_forward_backward_func():
    # Check if BitPipe is enabled
    if get_args().enable_bitpipe_schedule:
        print("[DEBUG] BitPipe 4-VR schedule ENABLED")
        from megatron.core.pipeline_parallel.schedule_impl.bitpipe import (
            forward_backward_pipelining_with_bitpipe_4vr,
        )
        return forward_backward_pipelining_with_bitpipe_4vr

    # Fallback to other schedulers...
```

**What it returns:** Function that will orchestrate all forward/backward passes

---

## PHASE 3: TRAINING LOOP

### Step 3.1: Call Forward-Backward Function
**File:** `megatron/training.py:480-488`
**Code:**
```python
losses_reduced = forward_backward_func(
    forward_step_func=forward_step_func,
    data_iterator=data_iterator,         # List of 4 iterators (one per VR)
    model=model,                          # List of 4 model chunks
    num_microbatches=get_num_microbatches(),  # User-specified count (e.g., 4)
    seq_length=args.seq_length,
    micro_batch_size=args.micro_batch_size,
    decoder_seq_length=args.decoder_seq_length,
    forward_only=False
)
```

---

## PHASE 4: BITPIPE SCHEDULER EXECUTION

Now we enter the BitPipe scheduler. This is where the magic happens!

**File:** `megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py`

### Step 4.1: Setup Phase
**Location:** `bitpipe_4vr.py:260-380`

```python
def forward_backward_pipelining_with_bitpipe_4vr(
    *,
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    ...
):
    # Get configuration
    args = get_args()
    config = get_model_config(model[0])

    # Get pipeline info
    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()  # 4
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()  # 0-3

    # Double microbatches for bidirectional execution
    # User says: 4 microbatches
    # BitPipe creates: 4 * (4 chunks / 2) = 8 microbatches total
    num_model_chunks = len(model)  # 4
    total_num_microbatches = num_microbatches * (num_model_chunks // 2)  # 4 * 2 = 8
```

**Key insight:** BitPipe creates 8 total microbatches from the user's 4!

### Step 4.2: Build Execution Schedules
**Location:** `bitpipe_4vr.py:380-420`

The scheduler creates two lists that determine the ORDER of microbatch execution:

```python
# Forward schedule - which microbatches to execute in forward pass, and in what order
forward_schedule = get_forward_schedule(...)
# Example result: [0, 1, 2, 10, 3, 11, 8, 9, 4, 5, 6, 14, 7, 15, 12, 13]

# Backward schedule - which microbatches to backward, and in what order
backward_schedule = get_backward_schedule(...)
# Example result: [8, 9, 10, 2, 11, 3, 0, 1, 12, 13, 14, 6, 15, 7, 4, 5]
```

**Purpose:** These schedules optimize pipeline utilization by:
- Interleaving forward and backward passes
- Minimizing "pipeline bubbles" (idle GPU time)
- Respecting the V-shaped layer distribution

### Step 4.3: WARMUP PHASE
**Location:** `bitpipe_4vr.py:514-595`

**Purpose:** Fill the pipeline with initial forward passes

```python
num_warmup_microbatches = pipeline_parallel_size  # 4

for k in range(num_warmup_microbatches):  # k = 0, 1, 2, 3
    # Get which microbatch and VR to execute
    forward_model_chunk_id = get_model_chunk_id(microbatch_idx[k])
    parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)

    # Execute forward pass for this microbatch on this VR
    output_tensor = forward_step_helper(microbatch_idx[k], None, 0)

    # CRITICAL: Check if first/last stage before communicating
    if parallel_state.is_pipeline_last_stage():
        output_tensor = None  # Don't send if you're last stage

    # Send output to next device (unless you're last stage)
    if not parallel_state.is_pipeline_last_stage():
        input_tensor = send_forward_recv_forward(output_tensor, ...)

    # Receive input from previous device (unless you're first stage)
    if not parallel_state.is_pipeline_first_stage():
        input_tensor = recv_forward(...)
```

**Warmup Timeline (4 devices, 4 warmup iterations):**
```
Time →

Device 0: [MB0,VR0] [MB1,VR1] [MB2,VR0] [MB10,VR1]
Device 1:           [MB0,VR0] [MB1,VR1] [MB2,VR0]
Device 2:                     [MB0,VR0] [MB1,VR1]
Device 3:                               [MB0,VR0]

After warmup: All devices have data flowing, pipeline is "full"
```

**Key protection (Line 548-549):**
```python
recv_prev = True
if parallel_state.is_pipeline_first_stage():
    recv_prev = False  # ← This prevents deadlock!
```

### Step 4.4: STEADY STATE (1F1B) PHASE
**Location:** `bitpipe_4vr.py:596-650`

**Purpose:** For remaining microbatches, do 1 Forward + 1 Backward on each iteration

```python
num_steady_state_iterations = num_microbatches_remaining

for j in range(n_loop):  # Multiple loops
    for k in range(unit_remaining * 2):  # Pairs of (forward, backward)
        if k % 2 == 0:
            # FORWARD PASS
            forward_k = k // 2 + num_warmup_microbatches + offset
            forward_model_chunk_id = get_model_chunk_id(microbatch_idx[forward_k])

            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
            output_tensor = forward_step_helper(microbatch_idx[forward_k], None, 0)

            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            # Send forward, receive backward gradient simultaneously (overlap!)
            if not parallel_state.is_pipeline_last_stage():
                output_tensor_grad = send_forward_recv_backward(
                    output_tensor,
                    tensor_shape=tensor_shape,
                    config=config,
                )
        else:
            # BACKWARD PASS
            backward_k = k // 2 + offset
            backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[backward_k])

            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            input_tensor_grad = backward_step_helper(microbatch_idx_b[backward_k])

            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Send backward, receive forward activation simultaneously
            if not parallel_state.is_pipeline_first_stage():
                input_tensor = send_backward_recv_forward(
                    input_tensor_grad,
                    ...
                )
```

**1F1B Timeline (Overlapping operations):**
```
Device 0: [F:0] [B:0] [F:1] [B:1] [F:2] [B:2] ...
Device 1:     [F:0] [B:0] [F:1] [B:1] [F:2] ...
Device 2:           [F:0] [B:0] [F:1] [B:1] ...
Device 3:                 [F:0] [B:0] [F:1] ...

F = Forward, B = Backward
Numbers = Microbatch ID
```

**Benefit:** While Device 0 computes backward, Device 1-3 can compute forward → Better GPU utilization!

### Step 4.5: COOLDOWN PHASE
**Location:** `bitpipe_4vr.py:651-730`

**Purpose:** Finish remaining backward passes after all forwards are done

```python
remaining_backward = backward_schedule[num_microbatches_remaining:]

for i, bwd_microbatch_id in enumerate(remaining_backward):
    # ... similar to steady state but only backward
```

### Step 4.6: GRADIENT SYNCHRONIZATION
**Location:** `bitpipe_4vr.py:432-457`

**Critical for correctness!**

```python
def allreduce_gradients(model):
    """
    Sync gradients across bidirectional pipeline pairs.

    Why needed: In BitPipe, paired devices process SAME layers from
    opposite directions. Example:

    Device 0, VR0: layers [0,1]  ← forward direction
    Device 3, VR1: layers [0,1]  ← backward direction (same layers!)

    Both accumulate gradients for the same weights, so we must average them.
    """
    if parallel_state.is_rank_in_bd_group():
        # Get all gradients for this VR
        buckets = {}
        for param in model.module.parameters():
            if param.requires_grad and param.main_grad is not None:
                tp = param.data.type()
                if tp not in buckets:
                    buckets[tp] = []
                buckets[tp].append(param)

        # AllReduce gradients within BD group
        for tp in buckets:
            bucket = buckets[tp]
            grads = [param.main_grad.data for param in bucket]
            coalesced = _flatten_dense_tensors(grads)
            coalesced /= torch.distributed.get_world_size(group=parallel_state.get_bd_parallel_group())
            torch.distributed.all_reduce(coalesced, group=parallel_state.get_bd_parallel_group())
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)
```

**Bidirectional (BD) Groups created in `parallel_state.py:262-307`:**
```python
# For 4 devices with BitPipe V-shaped ranks:
# Rank order: [0, 3, 2, 1]
# BD Group 0: [0, 3] - These two share layers and must sync gradients
# BD Group 1: [2, 1] - These two share layers and must sync gradients
```

---

## PHASE 5: AFTER FORWARD-BACKWARD

### Step 5.1: Reduce Gradients (Data Parallel)
**File:** `megatron/training.py:500`
```python
optimizer.reduce_model_grads(args, timers)
```

If using data parallelism, average gradients across all data-parallel replicas.

### Step 5.2: Optimizer Step
**File:** `megatron/training.py:510`
```python
update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)
```

Update model weights using the averaged gradients.

### Step 5.3: Loss Aggregation and Logging
**File:** `megatron/training.py:537-544`
```python
if is_pipeline_stage_containing_loss():
    # Average loss across microbatches
    loss_reduced = {}
    for key in losses_reduced[0]:
        losses_reduced_for_key = [x[key] for x in losses_reduced]
        loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
    return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
```

---

## SUMMARY DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│ User starts: python pretrain_gpt.py --enable-bitpipe-schedule  │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 0: initialize_megatron()                                  │
│ - Parse arguments                                               │
│ - Set virtual_pipeline_model_parallel_size = 4 (BitPipe)        │
│ - Initialize distributed PyTorch                                │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: Build Model                                            │
│ - Create 4 model chunks (megatron/model/transformer.py)         │
│ - Each chunk gets DIFFERENT layers based on V-shaped pattern    │
│   Device 0, VR0: layers [0,1]                                   │
│   Device 0, VR1: layers [6,7]                                   │
│   Device 0, VR2: layers [14,15]                                 │
│   Device 0, VR3: layers [8,9]                                   │
│   ... (repeat pattern for devices 1, 2, 3)                      │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: Select Scheduler (schedules.py:21-119)                 │
│ - get_forward_backward_func()                                   │
│ - Detects --enable-bitpipe-schedule                             │
│ - Returns: forward_backward_pipelining_with_bitpipe_4vr         │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: Call Scheduler (training.py:480-488)                   │
│ forward_backward_func(                                           │
│     forward_step_func=...,                                      │
│     data_iterator=[iter0, iter1, iter2, iter3],                 │
│     model=[model0, model1, model2, model3],                     │
│     num_microbatches=4,  (user-specified)                       │
│     ...                                                          │
│ )                                                               │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: BitPipe Scheduler (bitpipe_4vr.py)                     │
│                                                                  │
│ 4.1 Setup:                                                      │
│     - Double microbatches: 4 → 8 total                          │
│     - Build forward/backward schedules                          │
│     - Initialize BD groups for gradient sync                    │
│                                                                  │
│ 4.2 WARMUP: Forward passes only (fill pipeline)                 │
│     for k in range(4):  # 4 warmup iterations                   │
│         forward_step_helper(...)                                │
│         send_forward(...) / recv_forward(...)                   │
│                                                                  │
│ 4.3 STEADY STATE: 1F1B (interleaved forward/backward)           │
│     for each iteration:                                         │
│         forward_step(...)                                       │
│         backward_step(...)                                      │
│         (with gradient sync when needed)                        │
│                                                                  │
│ 4.4 COOLDOWN: Remaining backward passes                         │
│     for each remaining backward:                                │
│         backward_step(...)                                      │
│         allreduce_gradients(...) [CRITICAL!]                    │
│                                                                  │
│ 4.5 Return: losses_reduced (list of losses per microbatch)      │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 5: Finish Iteration (training.py:500-544)                 │
│ - optimizer.reduce_model_grads()   [DDP sync if needed]         │
│ - optimizer.step()                 [Update weights]             │
│ - Aggregate losses for logging                                  │
│ - Continue to next iteration                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## KEY CONCEPTS TO REMEMBER

### 1. Layer Assignment (Why V-shape?)
Each device gets DIFFERENT layers for each VR to enable:
- Two pipelines running in opposite directions simultaneously
- Gradient synchronization between paired devices
- Optimal utilization of all GPUs

**File:** `megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py:66-96`

### 2. Scheduler Selection (One dispatcher, many backends)
The function `get_forward_backward_func()` is a **dispatcher** that:
- Checks which schedule is enabled (BitPipe, Chimera, standard 1F1B, etc.)
- Returns the appropriate scheduler function
- The training loop calls whatever function is returned

**File:** `megatron/core/pipeline_parallel/schedules.py:21-119`

### 3. Microbatch Scheduling (Order matters!)
The forward and backward SCHEDULES determine:
- Which microbatch executes when
- Which VR each microbatch uses
- When to synchronize gradients

These schedules are carefully computed to:
- Maximize pipeline utilization
- Minimize idle GPU time (bubbles)
- Ensure correct gradient synchronization

**File:** `megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py:200-350`

### 4. Boundary Checks (Prevent deadlock)
CRITICAL before any P2P communication:
```python
if not parallel_state.is_pipeline_first_stage():
    input_tensor = recv_forward(...)

if not parallel_state.is_pipeline_last_stage():
    send_forward(output_tensor, ...)
```

This prevents:
- First stage trying to receive from non-existent previous stage
- Last stage trying to send to non-existent next stage

**Why Chimera deadlocks:** Missing these checks!

### 5. Bidirectional Gradient Sync (Critical for correctness)
After backward passes, BitPipe calls:
```python
allreduce_gradients(model[chunk_id])
```

This averages gradients across paired devices that share layers:
```
Device 0, VR0: layers [0,1]  ← accumulates gradients forward
Device 3, VR1: layers [0,1]  ← accumulates gradients backward
→ Must average gradients for these shared layers!
```

**File:** `megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py:432-457`

---

## HOW TO TRACE YOUR OWN EXECUTION

1. **Start here:** Your training script (`pretrain_gpt.py`)
2. **Follow to:** `megatron/training.py:474` - `get_forward_backward_func()`
3. **Which calls:** `megatron/core/pipeline_parallel/schedules.py:21-119`
4. **Which returns:** `megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py`
5. **Which is called by:** `megatron/training.py:480` with your data and model
6. **Each iteration:** Repeats 1-5 above until training completes

Use this to debug issues:
- Add print statements at each phase
- Check which scheduler is selected
- Verify layer distribution matches expected V-shape
- Confirm boundary checks prevent P2P on first/last stage

