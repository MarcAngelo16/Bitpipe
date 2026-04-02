# Session Summary: Understanding BitPipe Tensor Arrays and Data Flow

**Date:** January 8, 2026
**Topic:** Deep dive into BitPipe's tensor queue management, VR transitions, and data flow

---

## Overview

This session focused on understanding **how BitPipe manages tensors** through three critical arrays (`input_tensors`, `output_tensors`, `output_tensor_grads`) and how data flows through Virtual Ranks (VRs) during training.

---

## Key Questions Explored

### 1. **How are the three tensor arrays organized?**
   - Are they per-rank or shared?
   - How are they indexed?
   - What is `len(model)`?

### 2. **When and how are these arrays filled?**
   - When does `input_tensors` get populated?
   - When does `output_tensors` get populated?
   - When does `output_tensor_grads` get populated?

### 3. **Where does the actual training data come from?**
   - How does MB0 get its data on the first stage?
   - What is the role of `data_iterator`?

### 4. **What is the microbatch schedule?**
   - Is `microbatch_idx[k]` just a label or an actual schedule?
   - How do MB2 and MB6 relate to each other?
   - What happens during VR transitions?

---

## Core Data Structures

### Three Arrays Per Rank

**Every rank maintains its own local arrays:**

```python
# Line 152-156 in bitpipe_4vr.py
input_tensors = [[] for _ in range(len(model))]        # len(model) = 4 for BitPipe 4-VR
output_tensors = [[] for _ in range(len(model))]       # One queue per VR
forward_data_store = []
if not forward_only:
    output_tensor_grads = [[] for _ in range(len(model))]  # Gradient queues
```

**Structure:**
```python
# ========== EACH RANK INDEPENDENTLY HAS ==========

input_tensors = [
    [],  # VR0 input queue
    [],  # VR1 input queue
    [],  # VR2 input queue
    [],  # VR3 input queue
]

output_tensors = [
    [],  # VR0 output queue
    [],  # VR1 output queue
    [],  # VR2 output queue
    [],  # VR3 output queue
]

output_tensor_grads = [
    [],  # VR0 gradient queue
    [],  # VR1 gradient queue
    [],  # VR2 gradient queue
    [],  # VR3 gradient queue
]
```

**Key Properties:**
- ✅ **Per-rank**: Each GPU has its own separate arrays (no shared memory)
- ✅ **Per-VR indexing**: Arrays indexed by Virtual Rank (0-3)
- ✅ **FIFO queues**: Use `append()` to add, `pop(0)` to remove (oldest first)

---

## When Arrays Are Filled

### 1. `input_tensors` - Filled During **FORWARD** Phase

**Three scenarios for filling:**

#### **Scenario A: First Stage (Generates Data)**
```python
# Line 402-406 in forward_step_helper
if parallel_state.is_pipeline_first_stage():
    if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
        input_tensors[model_chunk_id].append(None)
        # ↑ None signals "generate from data_iterator"
```

**Example: Rank 0, VR0 (First Stage)**
```python
# Before MB0:
input_tensors[VR0] = []

# During forward_step_helper(MB0):
input_tensors[VR0].append(None)  # For MB0
# input_tensors[VR0] = [None]

# After P2P (prepare for MB1):
input_tensors[VR0].append(None)  # For MB1
# input_tensors[VR0] = [None, None]
```

---

#### **Scenario B: Middle/Last Stage (Receives via P2P)**
```python
# Line 587 in warmup loop
input_tensor = p2p_communication.send_forward_recv_forward(
    output_tensor,
    recv_prev=recv_prev,
    tensor_shape=tensor_shape,
    config=config,
)
input_tensors[next_forward_model_chunk_id].append(input_tensor)
# ↑ Receives activation from previous rank
```

**Example: Rank 1, VR0 (Middle Stage)**
```python
# After Rank 0 sends out_R0_MB0:

# P2P receive
input_tensor = recv_forward(...)  # Receives out_R0_MB0
input_tensors[VR0].append(input_tensor)

# State:
# input_tensors[VR0] = [out_R0_MB0]
```

---

#### **Scenario C: Same-Device VR Transition**
```python
# Line 569-572 in warmup loop
if (is_pipeline_last_stage(ignore_virtual=True) and next_forward_model_chunk_id == v_size-2) or \
   (is_pipeline_first_stage(ignore_virtual=True) and next_forward_model_chunk_id == v_size-1):
    detached_output_tensor = output_tensor.detach()
    detached_output_tensor.requires_grad_()
    input_tensor = detached_output_tensor
    # ↑ No P2P communication needed!

# Line 587
input_tensors[next_forward_model_chunk_id].append(input_tensor)
```

**Example: Rank 0, VR1→VR3 Transition**
```python
# After processing MB2 in VR1:
output_tensor = out_R0_MB2  # VR1 output

# Transition to VR3 (MB6)
detached = output_tensor.detach()
input_tensors[VR3].append(detached)

# State:
# input_tensors[VR3] = [out_R0_MB2_detached]  # Ready for MB6!
```

---

### 2. `output_tensors` - Filled During **FORWARD** Phase

**Filled after forward computation:**

```python
# Line 419 in forward_step_helper
output_tensor = forward_step(
    forward_step_func,
    data_iterator[model_chunk_id],
    model[model_chunk_id],
    num_microbatches // 2,
    input_tensor,
    forward_data_store,
    config,
    collect_non_loss_data,
    checkpoint_activations_microbatch,
)
output_tensors[model_chunk_id].append(output_tensor)
# ↑ Always appended after forward computation
```

**Key Insight: Tensors Stay Until Backward!**

```python
# Forward MB0:
output_tensors[VR0].append(out_R0_MB0)
# output_tensors[VR0] = [out_R0_MB0]

# P2P: Send out_R0_MB0 to next rank
send_forward(out_R0_MB0)

# Line 589: Deallocate memory (optional)
deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
# ↑ May free GPU memory, but queue entry REMAINS!

# output_tensors[VR0] = [out_R0_MB0]  ← Still there!

# Forward MB1:
output_tensors[VR0].append(out_R0_MB1)
# output_tensors[VR0] = [out_R0_MB0, out_R0_MB1]

# ... Queue grows during warmup ...

# Backward MB0: ONLY NOW is it removed!
# Line 480 in backward_step_helper
output_tensor = output_tensors[VR0].pop(0)  # Removes out_R0_MB0
# output_tensors[VR0] = [out_R0_MB1]
```

**Why keep outputs?**
- Needed for backward pass gradient computation
- FIFO ensures correct pairing: first forward → first backward

---

### 3. `output_tensor_grads` - Filled During **BACKWARD** Phase

**Four scenarios for filling:**

#### **Scenario A: Last Stage (Generate from Loss)**
```python
# Line 476-478 in backward_step_helper
if parallel_state.is_pipeline_last_stage():
    if len(output_tensor_grads[model_chunk_id]) == 0:
        output_tensor_grads[model_chunk_id].append(None)
        # ↑ None means "generate gradient from loss"
```

**Example: Rank 0, VR1 (Last Stage)**
```python
# Backward MB2:
# Rank 0 VR1 is last stage, no gradient to receive

# Generate gradient from loss
output_tensor_grads[VR1].append(None)
# output_tensor_grads[VR1] = [None]

# During backward_step:
# None is converted to loss gradient (dloss/doutput = 1.0)
```

---

#### **Scenario B: 1F1B Phase (Receive After Forward)**
```python
# Line 613-622 in steady-state loop
if not parallel_state.is_pipeline_last_stage():
    output_tensor_grad = p2p_communication.send_forward_recv_backward(
        output_tensor,  # Send forward output
        tensor_shape=tensor_shape,
        config=config,
    )
    # ↑ Receives gradient from next rank

    output_tensor_grads[next_backward_model_chunk_id].append(
        output_tensor_grad
    )
```

**Example: Rank 0, VR0 (First Stage)**
```python
# 1F1B: After forward MB4, prepare for backward MB0

# P2P: Send out_R0_MB4 forward, Receive grad_R1_MB0 backward
output_tensor_grad = send_forward_recv_backward(out_R0_MB4)
# Receives: grad_R1_MB0 (gradient from Rank 1)

# Store for upcoming backward MB0
output_tensor_grads[VR0].append(grad_R1_MB0)
# output_tensor_grads[VR0] = [grad_R1_MB0]
```

---

#### **Scenario C: Backward Phase (Receive After Backward)**
```python
# Line 697-705 in cooldown loop
output_tensor_grad = p2p_communication.send_backward_recv_backward_bd(
    input_tensor_grad,  # Send gradient to previous rank
    recv_prev=recv_prev,
    tensor_shape=tensor_shape,
    config=config,
)
# ↑ Receives gradient from next rank

if recv_prev:
    output_tensor_grads[next_backward_model_chunk_id].append(
        output_tensor_grad
    )
```

---

#### **Scenario D: Same-Device Gradient Transition**
```python
# Line 693-694
if (is_pipeline_last_stage(ignore_virtual=True) and next_backward_model_chunk_id == 0) or \
   (is_pipeline_first_stage(ignore_virtual=True) and next_backward_model_chunk_id == 1):
    output_tensor_grads[next_backward_model_chunk_id].append(input_tensor_grad)
    # ↑ Direct gradient passing, no P2P!
```

**Example: Rank 3, VR2→VR0 Gradient Transition**
```python
# During backward MB8 (VR2):
input_tensor_grad = backward_step(...)  # grad_R3_MB8

# Next backward is MB0 in VR0 (same device)
# Direct gradient passing
output_tensor_grads[VR0].append(grad_R3_MB8)

# State:
# output_tensor_grads[VR0] = [grad_R3_MB8]  # Ready for backward MB0!
```

---

## Where Training Data Comes From

### The Data Iterator - Source of Truth

**Critical insight:** MB0 doesn't just appear - it comes from the **data_iterator**!

```python
# Line 408-418 in forward_step_helper
output_tensor = forward_step(
    forward_step_func,
    data_iterator[model_chunk_id],  # ← THE ACTUAL TRAINING DATA SOURCE!
    model[model_chunk_id],
    num_microbatches // 2,
    input_tensor,  # None for first stage
    forward_data_store,
    config,
    collect_non_loss_data,
    checkpoint_activations_microbatch,
)
```

### Inside `forward_step` Function

**Line 204 in schedules.py:**
```python
def forward_step(forward_step_func, data_iterator, model, ...):
    # Call user-defined forward_step_func
    output_tensor, loss_func = forward_step_func(data_iterator, model)
    # ↑ This is where training data is fetched!
```

### User's `forward_step_func`

**Example from Line 61-64 in schedules.py:**
```python
def forward_step(data_iterator, model):
    data, loss_mask = next(data_iterator)  # ← FETCH REAL TRAINING BATCH!
    output = model(data)  # Process the batch
    return output, partial(loss_func, loss_mask)
```

### Complete Flow: MB0 Data Acquisition

```
1. Warmup loop k=0
   ↓
2. forward_step_helper(microbatch_idx[0]=0, input_tensor=None, ...)
   ↓
3. forward_step(..., data_iterator[VR0], model[VR0], input_tensor=None, ...)
   ↓
4. forward_step_func(data_iterator[VR0], model[VR0])
   ↓
5. data, loss_mask = next(data_iterator[VR0])  ← GET BATCH 0 FROM DATASET!
   ↓
6. output = model[VR0](data)  ← Process through layers [0,1]
   ↓
7. Return output_tensor = out_R0_MB0
```

**Key Points:**
- **`data_iterator[VR0]`** is a Python iterator over the training dataset
- **`next(data_iterator)`** fetches the next batch of real training data
- **Each VR has its own data iterator** for proper data distribution
- **Microbatch IDs are just labels** - actual data comes from iterator!

---

## Microbatch Schedule: Labels vs Execution Order

### The Schedule is BOTH a Label and Timeline

```python
# Example schedule for Rank 0 (warmup):
microbatch_idx = [0, 1, 2, 6, 3, 7]
#                 ↑  ↑  ↑  ↑  ↑  ↑
#                k=0 k=1 k=2 k=3 k=4 k=5

# What this means:
# k=0 (Timestep 0): Process microbatch with ID 0 (MB0) in VR0
# k=1 (Timestep 1): Process microbatch with ID 1 (MB1) in VR0
# k=2 (Timestep 2): Process microbatch with ID 2 (MB2) in VR1
# k=3 (Timestep 3): Process microbatch with ID 6 (MB6) in VR3 ← Transition!
# k=4 (Timestep 4): Process microbatch with ID 3 (MB3) in VR1
# k=5 (Timestep 5): Process microbatch with ID 7 (MB7) in VR3
```

### Loop Variable `k` = Timestep (When to Execute)

```python
for k in range(num_warmup_microbatches):
    # k is the TIMESTEP in execution
    microbatch_id = microbatch_idx[k]  # Which MB to process
    forward_model_chunk_id = get_model_chunk_id(microbatch_id)  # Which VR
```

### Microbatch ID Determines VR

```python
def get_model_chunk_id(microbatch_id):
    """Maps microbatch ID to Virtual Rank"""
    microbatch_id_in_group = microbatch_id % pipeline_parallel_size
    chunk_offset = 0 if microbatch_id < (total_num_microbatches // 2) else 2
    model_chunk_id = microbatch_id_in_group // (pipeline_parallel_size // 2)
    model_chunk_id += chunk_offset
    return model_chunk_id

# Examples:
# get_model_chunk_id(0) = VR0  # First half, pipeline 0
# get_model_chunk_id(1) = VR0  # First half, pipeline 0
# get_model_chunk_id(2) = VR1  # First half, pipeline 1
# get_model_chunk_id(6) = VR3  # Second half, pipeline 1
```

---

## VR Transitions: The MB2→MB6 Mystery Explained

### Critical Insight: MB2 and MB6 are THE SAME DATA!

**MB2 and MB6 are NOT separate microbatches - they're the same data flowing through different parts of the model!**

```python
# At k=2 (Timestep 2):
microbatch_id = microbatch_idx[2] = 2  # MB2
forward_model_chunk_id = get_model_chunk_id(2) = VR1

# Process MB2 through VR1 (first half of model, layers 0-7)
input_tensor = input_tensors[VR1][-1]  # out_R1_MB2 (from Rank 1)
output_tensor = forward_step(input_tensor, ...)  # out_R0_MB2

# Determine next microbatch:
next_microbatch_id = microbatch_idx[3] = 6
next_forward_model_chunk_id = get_model_chunk_id(6) = VR3

# Different chunks → TRANSITION!
if forward_model_chunk_id != next_forward_model_chunk_id:
    detached_output_tensor = output_tensor.detach()  # out_R0_MB2
    input_tensors[VR3].append(detached_output_tensor)

# ==========================================

# At k=3 (Timestep 3):
microbatch_id = microbatch_idx[3] = 6  # MB6!
forward_model_chunk_id = get_model_chunk_id(6) = VR3

# Process MB6 through VR3 (second half of model, layers 8-15)
input_tensor = input_tensors[VR3][-1]  # out_R0_MB2 (detached) ← SAME DATA!
output_tensor = forward_step(input_tensor, ...)  # out_R0_MB6
```

### Data Flow Visualization

```
MB2 Journey (Complete Model):
┌─────────────────────────────────────┐
│  VR1 Pipeline (First Half)          │
│  Rank 3 → Rank 2 → Rank 1 → Rank 0  │
│                           out_R1_MB2 │
└──────────────┬──────────────────────┘
               │ P2P send
               ↓
┌──────────────────────────────────────┐
│  Rank 0 VR1 (Last Stage of VR1)     │
│  Label: MB2                          │
│  Input: out_R1_MB2                   │
│  Process: Layers [6,7]               │
│  Output: out_R0_MB2                  │
└──────────────┬───────────────────────┘
               │ Same-device transition (.detach())
               │ Timestep: k=2 → k=3
               ↓
┌──────────────────────────────────────┐
│  Rank 0 VR3 (First Stage of VR3)    │
│  Label: MB6                          │
│  Input: out_R0_MB2 (detached)        │
│  Process: Layers [8,9]               │
│  Output: out_R0_MB6                  │
└──────────────┬───────────────────────┘
               │ P2P send
               ↓
         VR3 Pipeline (Second Half)
         Rank 0 → Rank 1 → Rank 2 → Rank 3
```

### Why Different IDs?

**Two reasons:**

1. **Scheduler tracking**: Different IDs help scheduler know which VR to use
2. **Queue management**: Separate IDs for first/second half processing

**But they represent the SAME logical training example!**

```python
# Conceptually:
# MB2 = Process training_batch_2 through first half (layers 0-7)
# MB6 = Process training_batch_2 through second half (layers 8-15)

# The data_iterator only gets called ONCE for this data
# The transition just passes activations between VRs
```

---

## Complete Example: Rank 0 Tensor Array Evolution

### Schedule for Rank 0 (Warmup)
```
Rank 0: MB0(VR0) → MB1(VR0) → MB2(VR1) → MB6(VR3) → MB3(VR1) → MB7(VR3)
        k=0        k=1        k=2        k=3        k=4        k=5
```

### Step-by-Step Array States

#### **Initial State**
```python
input_tensors = [[], [], [], []]
output_tensors = [[], [], [], []]
output_tensor_grads = [[], [], [], []]
```

---

#### **After k=0: Process MB0 in VR0**

**Actions:**
- Rank 0 is FIRST stage for VR0
- Generate data from `next(data_iterator[VR0])`
- Compute forward pass
- Send to Rank 1, prepare for MB1

**Arrays:**
```python
input_tensors = [
    [None, None],  # VR0: [MB0_used, MB1_prepared]
    [],
    [],
    [],
]

output_tensors = [
    [out_R0_MB0],  # VR0: Stored for backward
    [],
    [],
    [],
]

output_tensor_grads = [[], [], [], []]  # No backward yet
```

---

#### **After k=1: Process MB1 in VR0**

**Actions:**
- Use prepared None for MB1
- Generate data from `next(data_iterator[VR0])`
- Compute forward pass
- Send to Rank 1
- **Prepare for MB2 in VR1**: Receive from Rank 1 VR1

**Arrays:**
```python
input_tensors = [
    [None, None, None],  # VR0: [MB0, MB1, next_MB]
    [out_R1_MB2],        # VR1: ← Received from Rank 1 for MB2!
    [],
    [],
]

output_tensors = [
    [out_R0_MB0, out_R0_MB1],  # VR0: 2 outputs
    [],
    [],
    [],
]

output_tensor_grads = [[], [], [], []]
```

---

#### **After k=2: Process MB2 in VR1**

**Actions:**
- Rank 0 is LAST stage for VR1
- Input from Rank 1: `out_R1_MB2`
- Compute forward pass → `out_R0_MB2`
- **Transition to VR3**: Next is MB6 in VR3
- Detach and store for MB6

**Arrays:**
```python
input_tensors = [
    [None, None, None],          # VR0
    [out_R1_MB2],                # VR1: Consumed by MB2
    [],
    [out_R0_MB2_detached],       # VR3: ← Transition! Ready for MB6
]

output_tensors = [
    [out_R0_MB0, out_R0_MB1],    # VR0
    [out_R0_MB2],                # VR1: ← Stored for backward!
    [],
    [],
]

output_tensor_grads = [[], [], [], []]
```

---

#### **After k=3: Process MB6 in VR3**

**Actions:**
- Rank 0 is FIRST stage for VR3
- Input from transition: `out_R0_MB2_detached`
- Compute forward pass → `out_R0_MB6`
- Send to Rank 1 VR3
- **Prepare for MB3 in VR1**: Receive from Rank 1 VR1

**Arrays:**
```python
input_tensors = [
    [None, None, None],                # VR0
    [out_R1_MB2, out_R1_MB3],          # VR1: ← Received from Rank 1 for MB3!
    [],
    [out_R0_MB2_detached, ???],        # VR3
]

output_tensors = [
    [out_R0_MB0, out_R0_MB1],          # VR0
    [out_R0_MB2],                      # VR1
    [],
    [out_R0_MB6],                      # VR3: ← Stored for backward!
]

output_tensor_grads = [[], [], [], []]
```

---

#### **After k=4: Process MB3 in VR1**

**Actions:**
- Similar to MB2
- Input from Rank 1: `out_R1_MB3`
- Compute forward pass → `out_R0_MB3`
- Transition to VR3 for MB7

**Arrays:**
```python
input_tensors = [
    [None, None, None],                              # VR0
    [out_R1_MB2, out_R1_MB3],                        # VR1: Both consumed
    [],
    [out_R0_MB2_detached, ???, out_R0_MB3_detached], # VR3: MB7 ready!
]

output_tensors = [
    [out_R0_MB0, out_R0_MB1],                        # VR0
    [out_R0_MB2, out_R0_MB3],                        # VR1: ← 2 outputs!
    [],
    [out_R0_MB6],                                    # VR3
]

output_tensor_grads = [[], [], [], []]
```

---

#### **After k=5: Process MB7 in VR3**

**Actions:**
- Input from transition: `out_R0_MB3_detached`
- Compute forward pass → `out_R0_MB7`
- Send to Rank 1 VR3

**Final Arrays After Warmup:**
```python
input_tensors = [
    [None, None, None],                                   # VR0: First stage
    [out_R1_MB2, out_R1_MB3],                             # VR1: From Rank 1
    [],                                                   # VR2: Not used
    [out_R0_MB2_detached, ???, out_R0_MB3_detached, ???], # VR3: Transitions
]

output_tensors = [
    [out_R0_MB0, out_R0_MB1],        # VR0: 2 microbatches
    [out_R0_MB2, out_R0_MB3],        # VR1: 2 microbatches
    [],                              # VR2: Not used
    [out_R0_MB6, out_R0_MB7],        # VR3: 2 microbatches
]

output_tensor_grads = [[], [], [], []]  # Filled during backward phase
```

---

## Summary of Key Insights

### 1. **Three Arrays, Three Phases**

| Array | Filled During | Consumed During | Purpose |
|-------|---------------|-----------------|---------|
| `input_tensors` | Forward (recv/append) | Forward (pop for compute) | Store inputs for forward |
| `output_tensors` | Forward (append after compute) | Backward (pop for gradient) | Store outputs for backward |
| `output_tensor_grads` | Backward (recv/append) | Backward (pop for gradient) | Store gradients for backward |

### 2. **Per-Rank, Per-VR Organization**

- Each rank maintains its own arrays (no shared memory)
- Each array has 4 queues (one per VR) for BitPipe 4-VR
- FIFO queue management ensures correct forward-backward pairing

### 3. **Tensors Persist Until Consumed**

```python
# Forward: Append
output_tensors[VR0].append(out_R0_MB0)

# P2P: Send (but DON'T remove from queue!)
send_forward(out_R0_MB0)

# Deallocate: May free memory (but queue entry remains!)
deallocate_output_tensor(out_R0_MB0)

# Backward: Pop (ONLY time tensors are removed)
output_tensors[VR0].pop(0)  # Finally removed!
```

### 4. **Training Data Comes from Iterator**

- `data_iterator[VR]` is a Python iterator over the dataset
- `next(data_iterator)` fetches the next real training batch
- First stage generates data, others receive via P2P

### 5. **Microbatch Schedule is Timeline**

- `k` = timestep (when to execute)
- `microbatch_idx[k]` = which microbatch ID (what to execute)
- Microbatch IDs determine which VR to use

### 6. **VR Transitions: Same Data, Different Labels**

- MB2 (VR1) and MB6 (VR3) are the **same training data**
- Transition happens via `detach()` on same device (no P2P)
- Different IDs help scheduler track first/second half processing

### 7. **Same-Device Transitions are Zero-Copy**

```python
# No network communication needed!
detached_output = output_tensor.detach()
input_tensors[next_VR].append(detached_output)

# Breaks computation graph for independent backward passes
# Much faster than P2P communication
```

---

## Visual Summary: Complete Tensor Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                    FORWARD PHASE                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Prepare Input                                          │
│     ├─ First stage: input_tensors.append(None)             │
│     ├─ Other stages: input_tensors.append(recv_forward())  │
│     └─ Transitions: input_tensors.append(prev_output.detach()) │
│                                                             │
│  2. Compute Forward                                        │
│     ├─ input = input_tensors[-1]                           │
│     ├─ data = next(data_iterator) [if input is None]       │
│     └─ output = forward_step(input, data, model)           │
│                                                             │
│  3. Store Output                                           │
│     └─ output_tensors.append(output)  ← STAYS UNTIL BACKWARD! │
│                                                             │
│  4. Send Forward                                           │
│     └─ send_forward(output)  ← Doesn't remove from queue!  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    BACKWARD PHASE                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Prepare Gradient                                       │
│     ├─ Last stage: output_tensor_grads.append(None)        │
│     ├─ Other stages: output_tensor_grads.append(recv_backward()) │
│     └─ Transitions: output_tensor_grads.append(next_grad)  │
│                                                             │
│  2. Pop Tensors (FIFO)                                     │
│     ├─ input_tensor = input_tensors.pop(0)  ← REMOVED!     │
│     ├─ output_tensor = output_tensors.pop(0)  ← REMOVED!   │
│     └─ output_grad = output_tensor_grads.pop(0)  ← REMOVED! │
│                                                             │
│  3. Compute Backward                                       │
│     └─ input_grad = backward_step(input, output, output_grad) │
│                                                             │
│  4. Send Gradient                                          │
│     └─ send_backward(input_grad)                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Open Questions for Future Exploration

1. **Backward phase timing**: When exactly do gradients start flowing?
2. **1F1B phase**: How do queue sizes stabilize during steady-state?
3. **Gradient synchronization**: How does BD (bidirectional) gradient sync work?
4. **Memory optimization**: How does `deallocate_pipeline_outputs` affect memory?
5. **Chimera 2-VR**: How do the queues differ with only 2 VRs per device?

---

## References

- **Code locations**: `megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py`
- **Key functions**: `forward_step_helper`, `backward_step_helper`, `forward_step`
- **P2P communication**: `megatron/core/pipeline_parallel/p2p_communication.py`
- **Project documentation**: `CLAUDE.md`, `PHASE2_SUMMARY.md`

---

## Acknowledgments

This session clarified fundamental aspects of BitPipe's tensor management that are critical for understanding:
- How data flows through the pipeline
- When and where tensors are stored and removed
- The relationship between microbatch labels and actual execution
- The elegant design of same-device VR transitions

These insights form the foundation for deeper exploration of BitPipe's performance characteristics and optimization strategies.
