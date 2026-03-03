# BitPipe Phase Looping Logic - Deep Dive Analysis

This document provides a comprehensive analysis of BitPipe 4-VR's phase looping logic to understand how it can be adapted for Chimera 2-VR.

---

## 1. Warmup Count Determination (Per-Rank)

**Location:** `bitpipe_4vr.py:203-230`

### Base Calculation

```python
total_num_microbatches = num_microbatches * (num_model_chunks // 2)  # Line 205
# Example: 8 microbatches * 2 = 16 total (doubled for bidirectional)

n_loop = total_num_microbatches // pipeline_parallel_size // 2 - 1  # Line 207
# Example: 16 // 4 // 2 - 1 = 1 loop iteration

if forward_only:
    num_warmup_microbatches = total_num_microbatches
else:
    if total_num_microbatches == pipeline_parallel_size:
        num_warmup_microbatches = total_num_microbatches
    else:
        # Base warmup count
        num_warmup_microbatches = pipeline_parallel_size + pipeline_parallel_size // 2
        # Example: 4 + 2 = 6 base warmup
```

### Per-Rank Adjustment (V-Shaped Pattern)

**This is the KEY to understanding per-rank warmup:**

```python
num_warmup_microbatches += (
    pipeline_parallel_rank
    if pipeline_parallel_rank < pipeline_parallel_size // 2
    else pipeline_parallel_size - 1 - pipeline_parallel_rank
)  # Lines 223-227
```

**Example with 4 devices:**

| Rank | Base | Adjustment | Total Warmup | Reasoning |
|------|------|------------|--------------|-----------|
| 0    | 6    | +0         | 6            | First rank needs minimum warmup |
| 1    | 6    | +1         | 7            | One more than rank 0 |
| 2    | 6    | +(4-1-2)=1 | 7            | Symmetric with rank 1 |
| 3    | 6    | +(4-1-3)=0 | 6            | Symmetric with rank 0 |

**Pattern:** Ranks form a **V-shape** - edges need less warmup, middle ranks need more.

### Phase Breakdown

```python
unit_remaining = 2 * pipeline_parallel_size - num_warmup_microbatches  # Line 228
num_microbatches_mid = n_loop * pipeline_parallel_size * 2  # Line 229
num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches - num_microbatches_mid  # Line 230
```

**Example (4 devices, 16 total microbatches, rank 0):**
- `num_warmup_microbatches = 6`
- `unit_remaining = 2*4 - 6 = 2`
- `num_microbatches_mid = 1 * 4 * 2 = 8`
- `num_microbatches_remaining = 16 - 6 - 8 = 2`

---

## 2. Forward/Backward Schedule Generation

### Forward Schedule: `get_microbatch_idx()` (Lines 256-300)

**Purpose:** Generate the order in which microbatches are executed in forward passes.

**Key Logic:**
```python
i_half = pipeline_parallel_rank // num_unit  # Which half (0 or 1)
num_initial = (
    num_unit - pipeline_parallel_rank
    if pipeline_parallel_rank < num_unit
    else pipeline_parallel_rank - num_unit + 1
)  # Lines 268-272
```

**Example Output (4 devices, 16 microbatches):**
```
Rank 0: [0, 1, 2, 10, 3, 11, 8,  9,  4, 5, 6, 14, 7, 15, 12, 13]
Rank 1: [0, 2, 1,  3, 10,  8, 11, 9,  4, 6, 5,  7, 14, 12, 15, 13]
Rank 2: [2, 0, 3,  1,  8, 10,  9, 11, 6, 4, 7,  5, 12, 14, 13, 15]
Rank 3: [2, 3, 0,  8,  1,  9, 10, 11, 6, 7, 4, 12,  5, 13, 14, 15]
```

**Pattern Analysis:**
- First half uses VR0/VR1 microbatches (0-7)
- Second half uses VR2/VR3 microbatches (8-15)
- Interleaving ensures bidirectional pipeline stays balanced

### Backward Schedule: `get_bkmicrobatch_idx()` (Lines 303-351)

**Purpose:** Generate backward pass order with gradient sync markers.

**Example Output (4 devices, 16 microbatches):**
```
Rank 0: [8, 9, 10, 2, 11, 3, 0, 1, 12, 13, 14, 6, 15, 7, 4, 5, -1, -1]
Rank 1: [8, 10, 9, 11, 2, 0, 3, 1, 12, 14, 13, 15, 6, 4, 7, 5, -1, -1]
Rank 2: [10, 8, 11, 9, 0, 2, 1, 3, 14, 12, 15, 13, 4, 6, 5, 7, -1, -1]
Rank 3: [10, 11, 8, 0, 9, 1, 2, 3, 14, 15, 12, 4, 13, 5, 6, 7, -1, -1]
```

**Sync Marker Insertion (Lines 335-339):**
```python
if pipeline_parallel_rank == pipeline_parallel_size // 2 or \
   pipeline_parallel_rank == pipeline_parallel_size // 2 - 1:
    microbatch_idx.append(-1)  # Middle ranks: eager sync at end
else:
    microbatch_idx.insert(-1, -1)  # Other ranks: sync second-to-last
microbatch_idx.append(-1)  # All ranks: final sync
```

**Result:** Each rank has TWO `-1` markers for gradient synchronization.

---

## 3. Helper Functions and Tensor Queue Management

### forward_step_helper() (Lines 386-425)

**Tensor Queue Access Pattern:**

```python
def forward_step_helper(microbatch_id, checkpoint_activations_microbatch, offset):
    model_chunk_id = get_model_chunk_id(microbatch_id)

    # RETRIEVE input tensor from END of queue (with offset)
    if parallel_state.is_pipeline_first_stage():
        if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
            input_tensors[model_chunk_id].append(None)  # First stage generates input

    input_tensor = input_tensors[model_chunk_id][-1 - offset]  # Line 407
    # -1 = most recent, -2 = second most recent, etc.

    # RUN forward computation
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

    # STORE output at END of queue
    output_tensors[model_chunk_id].append(output_tensor)  # Line 419

    return output_tensor
```

**Key Insights:**
1. **Input retrieval:** `[-1-offset]` - takes from END with offset (default offset=0)
2. **Output storage:** `.append()` - adds to END
3. **Offset usage:** Almost always 0 in current code (used for potential future optimizations)

### backward_step_helper() (Lines 455-507)

**Tensor Queue Access Pattern:**

```python
def backward_step_helper(microbatch_id):
    model_chunk_id = get_model_chunk_id(microbatch_id)

    # RETRIEVE from FRONT of queues (FIFO order)
    if parallel_state.is_pipeline_last_stage():
        if len(output_tensor_grads[model_chunk_id]) == 0:
            output_tensor_grads[model_chunk_id].append(None)  # Last stage generates loss gradient

    input_tensor = input_tensors[model_chunk_id].pop(0)  # Line 479 - FIFO!
    output_tensor = output_tensors[model_chunk_id].pop(0)  # Line 480 - FIFO!
    output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)  # Line 481 - FIFO!

    # RUN backward computation
    input_tensor_grad = backward_step(
        input_tensor, output_tensor, output_tensor_grad, model_type, config
    )

    return input_tensor_grad
```

**Key Insights:**
1. **All retrievals:** `.pop(0)` - removes from FRONT (FIFO)
2. **Ensures correctness:** Backward processes oldest forward pass first
3. **Queue draining:** Each backward consumes one forward from each queue

### Queue Invariant

**CRITICAL PROPERTY:**
```
For each model_chunk_id:
- Forward appends to END: input_tensors[chunk].append(x)
- Forward appends to END: output_tensors[chunk].append(y)
- Backward pops from FRONT: input_tensors[chunk].pop(0)
- Backward pops from FRONT: output_tensors[chunk].pop(0)

Result: FIFO queue ensures correct forward-backward pairing
```

---

## 4. Pre-Receiving vs Inline Receiving

### Pre-Receive Phase (Lines 510-519)

**Before warmup loop starts:**

```python
# Pre-receive the FIRST tensor to kick off the pipeline
if pipeline_parallel_rank < pipeline_parallel_size // 2:
    # First half ranks use VR0
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(recv_forward(tensor_shape, config))
else:
    # Second half ranks use VR1
    parallel_state.set_virtual_pipeline_model_parallel_rank(1)
    input_tensors[1].append(recv_forward(tensor_shape, config))
```

**Why pre-receive?**
- Ensures every rank has ONE tensor ready to start computation
- Prevents deadlock - first rank can't send until someone is ready to receive
- Primes the pipeline pump

### Inline Receive Pattern (Warmup Phase Lines 554-587)

**After each forward step:**

```python
# Compute forward
output_tensor = forward_step_helper(microbatch_idx[k], None, 0)

# Determine next VR
next_forward_model_chunk_id = get_model_chunk_id(microbatch_idx[k + 1])

# CASE 1: Same VR (sequential pipeline stages)
if forward_model_chunk_id == next_forward_model_chunk_id:
    if parallel_state.is_pipeline_first_stage():
        recv_prev = False  # Don't receive, generate input

    input_tensor = send_forward_recv_forward(
        output_tensor,
        recv_prev=recv_prev,
        tensor_shape=tensor_shape,
        config=config,
    )

# CASE 2: Different VR (VR transition)
else:
    # SPECIAL CASE: Same-device VR transition (VR0→VR2 or VR1→VR3)
    if (is_pipeline_last_stage(ignore_virtual=True) and next_chunk == VR2) or \
       (is_pipeline_first_stage(ignore_virtual=True) and next_chunk == VR3):
        # NO COMMUNICATION! Just detach and reattach gradient tracking
        detached_output_tensor = output_tensor.detach()
        detached_output_tensor.requires_grad_()
        input_tensor = detached_output_tensor  # Lines 569-572

    # REGULAR CASE: Bidirectional P2P communication
    else:
        if parallel_state.is_pipeline_last_stage():
            recv_next = False  # Last stage doesn't receive from next

        input_tensor = send_forward_recv_forward_bd0(
            output_tensor,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            config=config,
        )

# Store for next iteration
input_tensors[next_forward_model_chunk_id].append(input_tensor)
```

**Key Decision Tree:**

```
After forward step with output_tensor:
├─ Is next microbatch same VR?
│  └─ YES: Use send_forward_recv_forward (normal pipeline P2P)
└─ NO: Different VR
   ├─ Is it same-device transition (VR0→VR2 or VR1→VR3)?
   │  └─ YES: Detach/reattach (no communication!)
   └─ NO: Use send_forward_recv_forward_bd0 (bidirectional P2P)
```

### 1F1B Phase Inline Receive Pattern (Lines 599-665)

**More complex - interleaves forward and backward:**

```python
for k in range(unit_remaining * 2):
    forward_k = k // 2 + num_warmup_microbatches + offset
    backward_k = k // 2 + offset

    if k % 2 == 0:
        # FORWARD STEP
        output_tensor = forward_step_helper(microbatch_idx[forward_k], None, 0)

        # Send forward output, receive backward gradient
        if not parallel_state.is_pipeline_last_stage():
            output_tensor_grad = send_forward_recv_backward(
                output_tensor,
                tensor_shape=tensor_shape,
                config=config,
            )  # Lines 613-619
            output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    else:
        # BACKWARD STEP
        input_tensor_grad = backward_step_helper(microbatch_idx_b[backward_k])

        # Send backward gradient, receive forward input
        if k < unit_remaining * 2 - 1:
            input_tensor = send_backward_recv_forward(
                input_tensor_grad,
                tensor_shape=tensor_shape,
                config=config,
            )  # Lines 657-663
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
```

**Pattern:** Every communication is **bidirectional** - send one tensor, receive another.

---

## 5. Gradient Sync Markers (-1) Integration

### Sync Marker Insertion (Lines 335-339)

**In `get_bkmicrobatch_idx()`:**

```python
# Eager sync marker placement
if pipeline_parallel_rank == pipeline_parallel_size // 2 or \
   pipeline_parallel_rank == pipeline_parallel_size // 2 - 1:
    microbatch_idx.append(-1)  # Middle ranks sync at the end
else:
    microbatch_idx.insert(-1, -1)  # Edge ranks sync second-to-last

microbatch_idx.append(-1)  # All ranks have final sync
```

**Result for 4 devices:**
- Rank 0 (edge): `[..., MB, -1, -1]` - two syncs at end
- Rank 1 (middle): `[..., MB, -1, -1]` - two syncs at end
- Rank 2 (middle): `[..., MB, -1, -1]` - two syncs at end
- Rank 3 (edge): `[..., MB, -1, -1]` - two syncs at end

### Sync Marker Processing (Cooldown Phase Lines 901-914)

**When `backward_model_chunk_id == -1`:**

```python
if backward_model_chunk_id == -1:  # Sync marker
    # Determine chunk order based on rank
    offset = (
        range(num_model_chunks // 2)  # Forward order [0, 1]
        if pipeline_parallel_rank < pipeline_parallel_size // 2
        else reversed(range(num_model_chunks // 2))  # Reverse order [1, 0]
    )

    if backward_k < total_num_microbatches + 1:
        # FIRST SYNC: Synchronize VR2 and VR3 (backward pipeline chunks)
        for i_chunk in offset:
            allreduce_gradients(model[num_model_chunks // 2 + i_chunk])
            # num_model_chunks = 4, so this syncs model[2] and model[3]

        # Continue receiving if more backward passes coming
        if not next_backward_model_chunk_id == -1:
            output_tensor_grads[next_backward_model_chunk_id].append(
                recv_backward(tensor_shape, config)
            )

    elif backward_k == total_num_microbatches + 1:
        # FINAL SYNC: Synchronize VR0 and VR1 (forward pipeline chunks)
        for i_chunk in offset:
            allreduce_gradients(model[i_chunk])
            # This syncs model[0] and model[1]
```

### allreduce_gradients() Implementation (Lines 427-452)

**Performs gradient synchronization across BD (bidirectional) groups:**

```python
def allreduce_gradients(model):
    if (
        parallel_state.is_rank_in_bd_group()
        and parallel_state.get_pipeline_model_parallel_world_size() > 1
    ):
        # Wait for all ranks in BD group
        torch.distributed.barrier(group=parallel_state.get_bd_parallel_group())

        # Pack gradients by data type
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
            coalesced = _flatten_dense_tensors(grads)

            # Average across BD group
            coalesced /= torch.distributed.get_world_size(
                group=parallel_state.get_bd_parallel_group()
            )

            # Synchronize
            torch.distributed.all_reduce(
                coalesced, group=parallel_state.get_bd_parallel_group()
            )

            # Unpack back to parameters
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)
```

**BD Groups for 4 devices:**
- BD Group 0: [Device 0, Device 3] - share VR0/VR1 layers
- BD Group 1: [Device 1, Device 2] - share VR0/VR1 layers

**Why two sync points?**
1. **First sync (-1):** Sync VR2/VR3 gradients early (eager sync)
   - These chunks finish backward first
   - Overlap sync with remaining computation
2. **Final sync (-1):** Sync VR0/VR1 gradients at the end
   - Ensures all gradients are synchronized before optimizer step

---

## 6. Phase Loop Structure Summary

### Warmup Phase (Lines 529-589)

```
FOR k = 0 to num_warmup_microbatches - 1:
    1. Set VR for current microbatch
    2. Run forward_step_helper()
    3. Determine next VR
    4. Send current output, receive next input:
       - Same VR: send_forward_recv_forward
       - Different VR (same device): detach/reattach
       - Different VR (different device): send_forward_recv_forward_bd0
    5. Store input for next iteration
```

**Purpose:** Fill the pipeline with forward passes, no backward yet.

### 1F1B Mid-Phase Loop (Lines 596-809)

```
FOR j = 0 to n_loop - 1:
    OFFSET = j * pipeline_parallel_size * 2

    # Subphase 1: Interleaved 1F1B (Lines 599-665)
    FOR k = 0 to unit_remaining * 2 - 1:
        IF k is even:
            - Run forward_step_helper()
            - Send forward, receive backward gradient
        ELSE:
            - Run backward_step_helper()
            - Send backward gradient, receive forward input

    # Subphase 2: Cooldown backward (Lines 669-705)
    FOR k = 0 to num_warmup_microbatches - unit_remaining:
        - Run backward_step_helper()
        - Send backward gradient, receive backward gradient (or forward input)

    # Subphase 3: 1F1B again (Lines 708-758)
    FOR k = 0 to 2 * (unit_remaining - 1) - 1:
        IF k is even:
            - Run forward_step_helper()
            - Send forward, receive backward gradient
        ELSE:
            - Run backward_step_helper()
            - Send backward gradient, receive forward input

    # Subphase 4: Warmup again (Lines 761-809)
    FOR k = 0 to num_warmup_microbatches - unit_remaining:
        - Run forward_step_helper()
        - Send forward, receive forward input
```

**Purpose:** Steady-state execution with balanced forward/backward.

### Final 1F1B Phase (Lines 813-885)

```
FOR k = 0 to 2 * unit_remaining - 1:
    IF k is even:
        - Run forward_step_helper()
        - Send forward, receive backward gradient
    ELSE:
        - Run backward_step_helper()
        - Send backward gradient, receive forward/backward input
```

**Purpose:** Complete remaining forward/backward passes before cooldown.

### Cooldown Phase (Lines 894-992)

```
FOR k = 0 to 2 * pipeline_parallel_size - unit_remaining + 1:
    backward_k = k + num_microbatches_mid + unit_remaining
    backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[backward_k])

    IF backward_model_chunk_id == -1:
        # GRADIENT SYNC MARKER
        IF first sync:
            - Sync VR2 and VR3 gradients (eager sync)
            - Receive next backward gradient if needed
        ELIF final sync:
            - Sync VR0 and VR1 gradients (final sync)
    ELSE:
        # REGULAR BACKWARD PASS
        - Run backward_step_helper()
        - Send backward gradient, receive backward gradient
        - Handle special cases for VR transitions
```

**Purpose:** Drain pipeline with remaining backward passes + gradient sync.

---

## 7. Key Takeaways for Chimera Adaptation

### What Stays the Same

1. **Queue management pattern:**
   - Forward: append to end, retrieve from end
   - Backward: pop from front (FIFO)

2. **Pre-receive + inline receive pattern:**
   - Pre-receive first tensor before loop
   - Inline receive after each computation

3. **Gradient sync markers (-1):**
   - Insert into backward schedule
   - Process during cooldown phase

4. **Phase structure:**
   - Warmup → 1F1B → Cooldown

### What Changes for Chimera 2-VR

1. **Warmup count calculation:**
   - Chimera has 2 VRs instead of 4
   - May need different V-shaped adjustment

2. **Schedule generation:**
   - Simpler with 2 VRs
   - No VR0→VR2, VR1→VR3 transitions
   - Just VR0 ↔ VR1 bidirectional flow

3. **Same-device transitions:**
   - **NO same-device transitions in Chimera**
   - Each VR completes full model on its own path
   - Remove lines 569-572, 789-792, 980-981, 693-694

4. **BD group sync:**
   - Still need BD groups for VR0 ↔ VR1 pairing
   - Simpler: only ONE sync point (not VR2/VR3 then VR0/VR1)

5. **P2P communication patterns:**
   - Simplified: only `send_forward_recv_forward_bd0` for bidirectional
   - No need for complex VR transition logic

### Critical Simplifications

**Chimera Advantage:**
- **50% fewer VR transitions** (2 VRs vs 4 VRs)
- **No same-device logic** (no VR0→VR2 transitions)
- **Simpler schedule generation** (sequential, not V-shaped)
- **Same queue management** (proven pattern from BitPipe)
- **Same sync strategy** (eager BD group sync)

---

## 8. Recommended Next Steps

1. **Simplify schedule generation:**
   - Adapt `get_microbatch_idx()` for 2-VR pattern
   - Remove VR2/VR3 references

2. **Simplify warmup calculation:**
   - Test if same V-shaped adjustment works for 2-VR
   - May need different formula

3. **Remove same-device transition logic:**
   - Delete detach/reattach code paths
   - All communication is P2P (no same-device shortcuts)

4. **Simplify gradient sync:**
   - One sync point for VR0+VR1 (not separate VR2/VR3 then VR0/VR1)

5. **Test with minimal example:**
   - 4 devices, 48 layers, 8 microbatches
   - Verify queue invariants hold
   - Check BD group synchronization

