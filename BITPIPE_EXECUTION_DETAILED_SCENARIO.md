# BitPipe Execution: Detailed Scenario with Boundary Checks

This document shows a **complete, step-by-step execution** of BitPipe with actual data flowing through the pipeline, including **exactly where and why** each boundary check is used.

---

## Scenario Setup

**Configuration:**
- **4 devices** (Device 0, 1, 2, 3)
- **16 layers** total (4 layers per device × 4 VRs)
- **4 microbatches** (user-specified) → **8 total** (BitPipe doubles)
- **4 VRs per device** (VR0, VR1, VR2, VR3)

**Layer Distribution (V-shaped):**
```
Device 0: VR0[0,1]   VR1[6,7]   VR2[14,15] VR3[8,9]
Device 1: VR0[2,3]   VR1[4,5]   VR2[12,13] VR3[10,11]
Device 2: VR0[4,5]   VR1[2,3]   VR2[10,11] VR3[12,13]
Device 3: VR0[6,7]   VR1[0,1]   VR2[8,9]   VR3[14,15]
```

**Forward Schedule (computed by `get_microbatch_idx()`):**
```
Device 0: [0, 1, 2, 10, 3, 11, 8, 9]      (8 microbatches)
Device 1: [0, 2, 1, 3, 10, 8, 11, 9]
Device 2: [2, 0, 3, 1, 8, 10, 9, 11]
Device 3: [2, 3, 0, 8, 1, 9, 10, 11]
```

**Backward Schedule (computed by `get_bkmicrobatch_idx()`):**
```
Device 0: [8, 9, 10, 2, 11, 3, -1, 0, 1, -1]   (-1 = gradient sync)
Device 1: [8, 10, 9, 11, 2, 0, 3, -1, 1, -1]
Device 2: [10, 8, 11, 9, 0, 2, 1, -1, 3, -1]
Device 3: [10, 11, 8, 0, 9, -1, 1, 2, 3, -1]
```

**Microbatch to VR Mapping:**
```
MB 0,1,4,5   → VR0
MB 2,3,6,7   → VR1
MB 8,9,12,13 → VR2
MB 10,11,14,15 → VR3
```

**Phase Breakdown:**
```
Warmup: 6 iterations (fill pipeline)
Steady: 0 iterations (n_loop=0 for this small example)
Cooldown: 10 iterations (drain pipeline)
```

---

## WARMUP PHASE: Step-by-Step

**Purpose:** Fill the pipeline with forward passes only.

### Iteration 0: First Forward Pass

**Device 0 (First Stage for VR0):**

```python
# Step 1: Get microbatch from schedule
forward_model_chunk_id = get_model_chunk_id(0)  # MB 0 → VR0
parallel_state.set_virtual_pipeline_model_parallel_rank(0)

# Step 2: Execute forward_step_helper
def forward_step_helper(microbatch_id=0, ...):
    model_chunk_id = 0  # VR0

    # ✅ BOUNDARY CHECK #1 (Lines 402-406)
    if parallel_state.is_pipeline_first_stage():  # ← True for Device 0, VR0
        if len(input_tensors[0]) == len(output_tensors[0]):  # Both empty
            input_tensors[0].append(None)  # ← Add None for first stage

    # Get input (None for first stage)
    input_tensor = input_tensors[0][-1]  # = None

    # Execute forward (embedding layer + layers [0,1])
    output_tensor = forward_step(
        forward_step_func,
        data_iterator[0],
        model[0],
        input_tensor=None,  # ← First stage has no input
        ...
    )
    # output_tensor = Tensor[seq_len, batch, hidden] with activations after layer 1

    # Store output
    output_tensors[0].append(output_tensor)

    return output_tensor

# Step 3: Prepare for P2P communication
output_tensor = result_from_helper  # Activations after layer [0,1]

# ✅ BOUNDARY CHECK #2 (Lines 547-548)
if parallel_state.is_pipeline_last_stage():  # ← False for Device 0, VR0
    output_tensor = None  # Skip (not last stage)

# ✅ BOUNDARY CHECK #3 (Lines 556-567)
recv_prev = True
if parallel_state.is_pipeline_first_stage():  # ← True for Device 0
    recv_prev = False  # ← Don't receive from non-existent Device -1

# Step 4: Send forward, receive next input
if not parallel_state.is_pipeline_last_stage():  # ← True (Device 0 is not last)
    input_tensor = send_forward_recv_forward(
        output_tensor,        # Send: activations after [0,1] → Device 1
        recv_prev=False,      # Don't receive (first stage)
        tensor_shape=...,
        config=config,
    )
    # Result: Device 1 receives activations, Device 0 gets nothing back

# Store next input (nothing to store for first stage in first iteration)
```

**What happened:**
- ✅ Device 0 added `None` input for first stage (CHECK #1)
- ✅ Device 0 processed MB0 through layers [0,1]
- ✅ Device 0 sent activations to Device 1 (CHECK #3 prevented receive)
- ✅ Device 0 didn't try to receive from non-existent Device -1 (CHECK #3)

---

**Device 1 (Middle Stage for VR0):**

```python
# Step 1: Get microbatch from schedule
forward_model_chunk_id = get_model_chunk_id(0)  # MB 0 → VR0
parallel_state.set_virtual_pipeline_model_parallel_rank(0)

# Device 1 receives first! (from Device 0's send)
# Receive happens automatically in send_forward_recv_forward

# Step 2: Execute forward_step_helper
def forward_step_helper(microbatch_id=0, ...):
    model_chunk_id = 0  # VR0

    # ✅ BOUNDARY CHECK #1 (Lines 402-406)
    if parallel_state.is_pipeline_first_stage():  # ← False for Device 1
        # Skip (not first stage)

    # Get input from queue (received from Device 0)
    input_tensor = input_tensors[0][-1]  # Activations from Device 0

    # Execute forward (layers [2,3])
    output_tensor = forward_step(
        forward_step_func,
        data_iterator[0],
        model[0],
        input_tensor=input_tensor,  # ← Use activations from Device 0
        ...
    )
    # output_tensor = Tensor with activations after layer 3

    # Store output
    output_tensors[0].append(output_tensor)

    return output_tensor

# Step 3: Send to next device
if parallel_state.is_pipeline_last_stage():  # ← False
    output_tensor = None  # Skip

recv_prev = True
if parallel_state.is_pipeline_first_stage():  # ← False
    recv_prev = False  # Skip

if not parallel_state.is_pipeline_last_stage():  # ← True
    input_tensor = send_forward_recv_forward(
        output_tensor,        # Send: activations after [2,3] → Device 2
        recv_prev=True,       # Will receive from Device 0 (for next MB)
        tensor_shape=...,
        config=config,
    )
```

**What happened:**
- ✅ Device 1 received activations from Device 0
- ✅ Device 1 processed MB0 through layers [2,3]
- ✅ Device 1 sent activations to Device 2

---

**Device 3 (Last Stage for VR0):**

```python
# Eventually, after Device 2 processes...

# Step 1: Receive from Device 2
# (happens automatically in previous device's send)

# Step 2: Execute forward_step_helper
def forward_step_helper(microbatch_id=0, ...):
    model_chunk_id = 0  # VR0

    # ✅ BOUNDARY CHECK #1 (Lines 402-406)
    if parallel_state.is_pipeline_first_stage():  # ← False
        # Skip

    # Get input
    input_tensor = input_tensors[0][-1]  # From Device 2

    # Execute forward (layers [6,7] + compute loss)
    output_tensor = forward_step(
        forward_step_func,
        data_iterator[0],
        model[0],
        input_tensor=input_tensor,
        ...
    )
    # output_tensor = Loss tensor (scalar)

    # Store output
    output_tensors[0].append(output_tensor)

    return output_tensor

# Step 3: Handle last stage
# ✅ BOUNDARY CHECK #2 (Lines 547-548)
if parallel_state.is_pipeline_last_stage():  # ← TRUE for Device 3, VR0
    output_tensor = None  # ← Don't send (no next device!)

recv_prev = True
if parallel_state.is_pipeline_first_stage():  # ← False
    recv_prev = False

# ✅ BOUNDARY CHECK #3 (Lines 556-567)
if not parallel_state.is_pipeline_last_stage():  # ← FALSE for Device 3
    # SKIP send_forward_recv_forward
    # ← This prevents deadlock! Device 3 won't try to send to Device 4
```

**What happened:**
- ✅ Device 3 received activations from Device 2
- ✅ Device 3 processed MB0 through layers [6,7] and computed loss
- ✅ Device 3 set `output_tensor = None` (CHECK #2)
- ✅ Device 3 **DIDN'T** try to send to non-existent Device 4 (CHECK #3)
- ✅ **NO DEADLOCK!** Last stage correctly stops propagation

---

### Warmup Summary (All 6 Iterations)

**Timeline:**

```
Time Step →
┌─────────────────────────────────────────────────────────────────┐
│ WARMUP PHASE (6 iterations)                                     │
└─────────────────────────────────────────────────────────────────┘

Device 0: [F:MB0] [F:MB1] [F:MB2] [F:MB10] [F:MB3] [F:MB11]
Device 1:         [F:MB0] [F:MB2] [F:MB1]  [F:MB3] [F:MB10]
Device 2:                 [F:MB0] [F:MB2]  [F:MB1] [F:MB3]
Device 3:                         [F:MB0]  [F:MB2] [F:MB1]

Legend:
F:MB0 = Forward pass for Microbatch 0
```

**Key Observations:**

1. **Pipeline fills gradually** (1 → 2 → 3 → 4 devices active)
2. **Each device processes different microbatches** (from schedule)
3. **Boundary checks prevent deadlocks:**
   - First stage doesn't receive from Device -1
   - Last stage doesn't send to Device 4
   - All middle stages send and receive normally

---

## STEADY STATE (1F1B): Step-by-Step

**Purpose:** One Forward + One Backward per iteration (pipeline full, maximum utilization)

**Note:** In our small example (8 microbatches, 4 devices), there's actually no steady state phase (n_loop=0). But let's show what it would look like with more microbatches.

### Iteration 7: First 1F1B Pair

**Device 0:**

```python
# k = 0 (even, so forward)
if k % 2 == 0:
    # FORWARD PASS
    forward_k = 0 + 6  # = 6 (index into forward schedule)
    forward_model_chunk_id = get_model_chunk_id(microbatch_idx[6])  # = get_model_chunk_id(8) = VR2

    parallel_state.set_virtual_pipeline_model_parallel_rank(2)  # VR2

    # Execute forward
    output_tensor = forward_step_helper(microbatch_idx[6], None, 0)  # MB 8

    # ✅ BOUNDARY CHECK #4 (Lines 607-608)
    if parallel_state.is_pipeline_last_stage():  # ← False for Device 0, VR2
        output_tensor = None

    # ✅ BOUNDARY CHECK #5 (Lines 612-622)
    if not parallel_state.is_pipeline_last_stage():  # ← True
        # Send forward, receive backward gradient (overlap!)
        output_tensor_grad = send_forward_recv_backward(
            output_tensor,        # Send: activations for MB8 → Device 1
            tensor_shape=...,
            config=config,
        )
        # Receive: gradient for previous microbatch (from MB 8's backward later)
        output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)
```

**What happened:**
- ✅ Device 0 did forward pass for MB8 (VR2, layers [14,15])
- ✅ Device 0 sent activations to Device 1
- ✅ Device 0 received gradient for a previous microbatch (overlapping communication!)
- ✅ Boundary checks ensured correct communication

---

```python
# k = 1 (odd, so backward)
else:
    # BACKWARD PASS
    backward_k = 0  # (first backward)
    backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[0])  # = get_model_chunk_id(8) = VR2

    parallel_state.set_virtual_pipeline_model_parallel_rank(2)  # VR2

    # Execute backward
    def backward_step_helper(microbatch_id=8):
        model_chunk_id = 2  # VR2

        # Check if last microbatch for this VR (enable gradient sync)
        if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(8):
            enable_grad_sync()  # Maybe enable (if last)

        # ✅ BOUNDARY CHECK #6 (Lines 476-478)
        if parallel_state.is_pipeline_last_stage():  # ← True for Device 0, VR2
            if len(output_tensor_grads[2]) == 0:
                output_tensor_grads[2].append(None)  # ← No gradient from next stage

        # Pop tensors from queues (FIFO)
        input_tensor = input_tensors[2].pop(0)        # Oldest input
        output_tensor = output_tensors[2].pop(0)      # Oldest output
        output_tensor_grad = output_tensor_grads[2].pop(0)  # Oldest gradient (None for last stage)

        # Execute backward
        input_tensor_grad = backward_step(
            input_tensor,
            output_tensor,
            output_tensor_grad,  # None for last stage
            model_type,
            config
        )
        # input_tensor_grad = gradient w.r.t. input (to send to previous device)

        disable_grad_sync()

        return input_tensor_grad

    input_tensor_grad = backward_step_helper(8)

    # ✅ BOUNDARY CHECK #7 (Lines 632-633)
    if parallel_state.is_pipeline_first_stage():  # ← False for Device 0, VR2
        input_tensor_grad = None

    # ✅ BOUNDARY CHECK #8 (Lines 637, 657-664)
    if not parallel_state.is_pipeline_first_stage():  # ← True for Device 0, VR2
        # Send backward gradient, receive forward activation (overlap!)
        input_tensor = send_backward_recv_forward(
            input_tensor_grad,    # Send: gradient for MB8 ← Device 1
            tensor_shape=...,
            config=config,
        )
        # Receive: activation for next forward pass
        input_tensors[next_forward_model_chunk_id].append(input_tensor)
```

**What happened:**
- ✅ Device 0 did backward pass for MB8 (VR2)
- ✅ Device 0 added `None` gradient for last stage (CHECK #6)
- ✅ Device 0 sent gradient to Device 1 (previous stage)
- ✅ Device 0 received activation for next forward pass
- ✅ **Overlap!** Forward and backward communication happen together

---

### 1F1B Summary

**Key Pattern:**
```
for k in range(unit_remaining * 2):
    if k % 2 == 0:
        # FORWARD: Send activations →, receive gradients ←
        send_forward_recv_backward(...)
    else:
        # BACKWARD: Send gradients ←, receive activations →
        send_backward_recv_forward(...)
```

**Boundary Checks:**
- ✅ Forward: Check if last stage before sending
- ✅ Backward: Check if first stage before sending
- ✅ Last stage adds `None` gradient (no next stage)
- ✅ First stage sets `input_tensor_grad = None` (no previous stage)

---

## COOLDOWN PHASE: Step-by-Step

**Purpose:** Drain remaining backward passes from pipeline + synchronize gradients

### Iteration 16: Backward with Gradient Sync

**Device 0:**

```python
backward_k = 6  # Index into backward schedule
backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[6])

# Special case: gradient sync marker
if backward_model_chunk_id == -1:  # ← TRUE! Gradient sync marker
    # Determine which VR chunks to sync
    offset = range(num_model_chunks // 2)  # [0, 1] for first half devices
    # For Device 0: sync VR2 and VR3 (backward pipeline chunks)

    # ✅ GRADIENT SYNC (Lines 901-914)
    if backward_k < total_num_microbatches + 1:  # First sync point
        for i_chunk in offset:  # i_chunk = 0, 1
            # Sync VR2 (chunk 2) and VR3 (chunk 3)
            allreduce_gradients(model[num_model_chunks // 2 + i_chunk])
            # model[2] = VR2, model[3] = VR3

        # After sync, receive next gradient if needed
        if not next_backward_model_chunk_id == -1:
            output_tensor_grads[next_backward_model_chunk_id].append(
                recv_backward(tensor_shape=tensor_shape, config=config)
            )

    elif backward_k == total_num_microbatches + 1:  # Second sync point
        for i_chunk in offset:  # i_chunk = 0, 1
            # Sync VR0 (chunk 0) and VR1 (chunk 1)
            allreduce_gradients(model[i_chunk])
```

**What is `allreduce_gradients()` doing?**

```python
def allreduce_gradients(model):
    """
    Sync gradients across BD (bidirectional) groups

    For Device 0, VR2 [layers 14,15]:
    - Device 0 has accumulated gradients for layers [14,15]
    - Device 3, VR3 also has layers [14,15] (same layers!)
    - Must average gradients between Device 0 and Device 3

    BD Groups:
    - Group 0: [Device 0, Device 1]
    - Group 1: [Device 3, Device 2]
    """

    if parallel_state.is_rank_in_bd_group():  # Am I in a BD group?
        # Barrier to sync
        torch.distributed.barrier(group=parallel_state.get_bd_parallel_group())

        # Group parameters by type
        buckets = {}
        for param in model.module.parameters():
            if param.requires_grad and param.main_grad is not None:
                tp = param.data.type()
                if tp not in buckets:
                    buckets[tp] = []
                buckets[tp].append(param)

        # AllReduce each bucket
        for tp in buckets:
            bucket = buckets[tp]
            grads = [param.main_grad.data for param in bucket]

            # Flatten all gradients
            coalesced = _flatten_dense_tensors(grads)

            # Average (divide by BD group size, which is 2)
            coalesced /= torch.distributed.get_world_size(
                group=parallel_state.get_bd_parallel_group()
            )
            # Example: Device 0 has grad=0.5, Device 1 has grad=0.3
            # After average: both have grad=(0.5+0.3)/2 = 0.4

            # AllReduce (sum operation, but already divided above)
            torch.distributed.all_reduce(
                coalesced, group=parallel_state.get_bd_parallel_group()
            )

            # Unflatten and copy back
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)
```

**Concrete Example:**

```
Before allreduce_gradients(model[2]):  # VR2

Device 0, VR2, layer 14 weight:
  - grad = [0.5, 0.3, 0.7]  (accumulated from MB 8, 9, 12, 13)

Device 1, VR3, layer 14 weight:
  - grad = [0.4, 0.6, 0.2]  (accumulated from MB 10, 11, 14, 15)

After allreduce_gradients(model[2]):

Device 0, VR2, layer 14 weight:
  - grad = [(0.5+0.4)/2, (0.3+0.6)/2, (0.7+0.2)/2]
  - grad = [0.45, 0.45, 0.45]

Device 1, VR3, layer 14 weight:
  - grad = [0.45, 0.45, 0.45]  (same!)

Now both devices have the SAME averaged gradients!
Optimizer step will update both devices identically.
```

---

### Normal Backward (No Sync Marker)

**Device 1:**

```python
backward_k = 6
backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[6])  # Not -1

if backward_model_chunk_id != -1:  # Normal backward
    # Execute backward_step_helper
    input_tensor_grad = backward_step_helper(microbatch_idx_b[6])

    # ✅ BOUNDARY CHECK #9 (Lines 921-923)
    if parallel_state.is_pipeline_first_stage():  # ← Depends on VR
        recv_prev = False
        input_tensor_grad = None  # ← Don't send if first stage

    # Determine next microbatch
    next_backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[7])

    # Check if same-device VR transition
    if (parallel_state.is_pipeline_last_stage(ignore_virtual=True) and
        next_backward_model_chunk_id == 0) or \
       (parallel_state.is_pipeline_first_stage(ignore_virtual=True) and
        next_backward_model_chunk_id == 1):
        # Same-device transition (VR2→VR0 or VR3→VR1)
        output_tensor_grads[next_backward_model_chunk_id].append(input_tensor_grad)
    else:
        # Normal P2P communication
        if not parallel_state.is_pipeline_first_stage():  # ← Boundary check
            output_tensor_grads[next_backward_model_chunk_id].append(
                send_backward_recv_backward_bd(
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    config=config,
                )
            )
```

**What happened:**
- ✅ Device 1 did backward pass
- ✅ Device 1 checked if first stage (CHECK #9)
- ✅ Device 1 sent gradient to previous device
- ✅ Device 1 received gradient for next backward

---

### Cooldown Summary

**Timeline:**

```
Time Step →
┌─────────────────────────────────────────────────────────────────┐
│ COOLDOWN PHASE (10 iterations)                                  │
└─────────────────────────────────────────────────────────────────┘

Device 0: [B:MB8] [B:MB9] [B:MB10] [B:MB2] [B:MB11] [B:MB3] [SYNC] [B:MB0] [B:MB1] [SYNC]
Device 1: [B:MB8] [B:MB10] [B:MB9] [B:MB11] [B:MB2] [B:MB0] [B:MB3] [SYNC] [B:MB1] [SYNC]
Device 2: [B:MB10] [B:MB8] [B:MB11] [B:MB9] [B:MB0] [B:MB2] [B:MB1] [SYNC] [B:MB3] [SYNC]
Device 3: [B:MB10] [B:MB11] [B:MB8] [B:MB0] [B:MB9] [SYNC] [B:MB1] [B:MB2] [B:MB3] [SYNC]

Legend:
B:MB0 = Backward pass for Microbatch 0
SYNC = Gradient synchronization (allreduce_gradients)
```

**Key Operations:**

1. **Backward passes drain from pipeline**
2. **Gradient sync markers (`-1`) trigger `allreduce_gradients()`**
3. **Two sync points:**
   - First sync: VR2/VR3 (backward pipeline chunks)
   - Second sync: VR0/VR1 (forward pipeline chunks)
4. **Eager sync overlaps with computation** (while other devices compute)

---

## Complete Boundary Check Table (with Context)

| Check | Location | Condition | Purpose | Example Scenario |
|-------|----------|-----------|---------|------------------|
| #1 | Lines 402-406 | `if is_pipeline_first_stage()` | Add `None` input for first stage | Device 0, VR0: No previous device, needs `input_tensor=None` |
| #2 | Lines 547-548 | `if is_pipeline_last_stage()` | Set `output_tensor = None` | Device 3, VR0: No next device, don't send |
| #3 | Lines 556-567 | `if not is_pipeline_first_stage()` | Only receive if not first | Device 0: Skip receive from Device -1 |
| #4 | Lines 607-608 | `if is_pipeline_last_stage()` | Set `output_tensor = None` (1F1B) | Device 3, VR2: Don't send in steady state |
| #5 | Lines 612-622 | `if not is_pipeline_last_stage()` | Only send if not last (1F1B) | Device 3: Skip send to Device 4 |
| #6 | Lines 476-478 | `if is_pipeline_last_stage()` | Add `None` gradient for last stage | Device 0, VR2: No gradient from next stage |
| #7 | Lines 632-633 | `if is_pipeline_first_stage()` | Set `input_tensor_grad = None` | Device 0, VR0: Don't send gradient to Device -1 |
| #8 | Lines 657-664 | `if not is_pipeline_first_stage()` | Only send if not first | Device 0, VR0: Skip send_backward to Device -1 |
| #9 | Lines 921-923 | `if is_pipeline_first_stage()` | Set `recv_prev = False`, `grad = None` | Device 0, VR0: Cooldown boundary handling |
| #10 | Lines 677-679 | `if is_pipeline_first_stage()` | Set `recv_prev = False` | Prevent receive deadlock in cooldown |
| #11 | Lines 717-718 | `if is_pipeline_last_stage()` | Set `output_tensor = None` | Prevent send deadlock in mid-loop |
| #12 | Lines 722-732 | `if not is_pipeline_last_stage()` | Only send if not last | Prevent send deadlock in mid-loop |
| #13 | Lines 743-745 | `if is_pipeline_first_stage()` | Set `grad = None`, `recv_prev = False` | Prevent backward send deadlock |
| #14 | Lines 749-757 | `if not is_pipeline_first_stage()` | Only send if not first | Prevent backward send deadlock |
| #15 | Lines 777-778 | `if is_pipeline_first_stage()` | Set `recv_prev = False` | Mid-loop warmup boundary |
| #16 | Lines 822-823 | `if is_pipeline_last_stage()` | Set `output_tensor = None` | Final 1F1B forward boundary |
| #17 | Lines 827-837 | `if not is_pipeline_last_stage()` | Only send if not last | Final 1F1B forward send |
| #18 | Lines 847-848 | `if is_pipeline_first_stage()` | Set `grad = None` | Final 1F1B backward boundary |

---

## Deadlock Scenarios Prevented

### ❌ Without Boundary Checks (DEADLOCK)

**Scenario 1: First Stage Receives**

```python
# Device 0, VR0 (first stage)
input_tensor = recv_forward(tensor_shape, config)
# ↑ BLOCKS FOREVER waiting for Device -1 (doesn't exist!)
```

**Scenario 2: Last Stage Sends**

```python
# Device 3, VR0 (last stage)
send_forward(output_tensor, ...)
# ↑ BLOCKS FOREVER waiting for Device 4 (doesn't exist!)
```

### ✅ With Boundary Checks (NO DEADLOCK)

**Scenario 1 Fixed:**

```python
# Device 0, VR0 (first stage)
if not parallel_state.is_pipeline_first_stage():  # ← False, skip
    input_tensor = recv_forward(tensor_shape, config)
# ↑ Doesn't execute! No deadlock!
```

**Scenario 2 Fixed:**

```python
# Device 3, VR0 (last stage)
if not parallel_state.is_pipeline_last_stage():  # ← False, skip
    send_forward(output_tensor, ...)
# ↑ Doesn't execute! No deadlock!
```

---

## Summary: Why Boundary Checks Matter

1. **Prevent deadlock:** First/last stages don't communicate with non-existent devices
2. **Correct gradients:** Last stage has `None` gradient, first stage doesn't send backward
3. **Proper queue management:** `None` values added at boundaries
4. **Clean execution:** Each device knows its role (first/middle/last)

**Total:** 18+ boundary checks prevent deadlocks throughout warmup, steady state, and cooldown phases!

---

## Key Takeaways

1. **Warmup:** Fill pipeline gradually, forward-only
   - First stage: Add `None` input (CHECK #1)
   - Last stage: Don't send forward (CHECK #2, #3)

2. **Steady State:** 1F1B pattern, maximum utilization
   - Forward: Check before send (CHECK #4, #5)
   - Backward: Check before send, add `None` grad for last stage (CHECK #6, #7, #8)

3. **Cooldown:** Drain backward passes, sync gradients
   - Backward: Check before send (CHECK #9, #10, etc.)
   - Gradient sync: Call `allreduce_gradients()` at markers
   - BD groups average gradients for shared layers

4. **Every P2P operation needs boundary checks!**
   - 17+ checks throughout the code
   - Missing even ONE check → deadlock!
