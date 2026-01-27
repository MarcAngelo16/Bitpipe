# BitPipe 4-VR: Detailed Code Study

## Key Insight: The Three Separate Schedules

BitPipe manages **THREE independent schedules** for each rank:

1. **Forward Schedule** (`microbatch_idx`): Which microbatches to process in warmup/1F1B phases, in what order
2. **Backward Schedule** (`microbatch_idx_b`): Which microbatches to process in 1F1B/cooldown phases, in what order
3. **Model Chunk Mapping** (`get_model_chunk_id()`): Maps any microbatch ID to its VR chunk (0,1,2,3)

**Critical**: Forward and backward schedules are INDEPENDENT and have DIFFERENT orders on each rank!

---

## Pre-Receive Phase (Lines 510-519)

```python
# Run warmup forward passes.
if pipeline_parallel_rank < pipeline_parallel_size // 2:  # Ranks 0,1
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(recv_forward(...))  # Pre-receive for VR0
else:  # Ranks 2,3
    parallel_state.set_virtual_pipeline_model_parallel_rank(1)
    input_tensors[1].append(recv_forward(...))  # Pre-receive for VR1
```

**What this does:**
- Rank 0: Sets VR0, receives initial forward tensor → stores in `input_tensors[0]`
- Rank 1: Sets VR0, receives initial forward tensor → stores in `input_tensors[0]`
- Rank 2: Sets VR1, receives initial forward tensor → stores in `input_tensors[1]`
- Rank 3: Sets VR1, receives initial forward tensor → stores in `input_tensors[1]`

**Why rank-based and not first-stage only?**
- Multiple stages need initial tensors
- Each rank needs its OWN initial tensor for the VR it's handling
- Rank 1 can't just receive from Rank 0 - Rank 1 is the first stage for some pipelines

---

## Schedule Generation (Lines 256-351)

### `get_microbatch_idx()` - Forward Schedule (Lines 256-300)

Generates the **forward microbatch order** for THIS rank. Example for 4 devices, 8 total MBs:

```
Forward schedules (DIFFERENT for each rank):
- Rank 0: [0, 1, 2, 10, 3, 11, 8, 9, 4, 5, 6, 14, 7, 15, 12, 13]
- Rank 1: [0, 2, 1, 3, 10, 8, 11, 9, 4, 6, 5, 7, 14, 12, 15, 13]
- Rank 2: [2, 0, 3, 1, 8, 10, 9, 11, 6, 4, 7, 5, 12, 14, 13, 15]
- Rank 3: [2, 3, 0, 8, 1, 9, 10, 11, 6, 7, 4, 12, 5, 13, 14, 15]
```

**Key insight:** Each rank has a different order! This is intentional for load balancing and pipelineization.

### `get_bkmicrobatch_idx()` - Backward Schedule (Lines 303-351)

Generates the **backward microbatch order** for THIS rank. Example:

```
Backward schedules (DIFFERENT for each rank):
- Rank 0: [8, 9, 10, 2, 11, 3, 0, 1, 12, 13, 14, 6, 15, 7, 4, 5, -1, -1]
- Rank 1: [8, 10, 9, 11, 2, 0, 3, 1, 12, 14, 13, 15, 6, 4, 7, 5, -1, -1]
- Rank 2: [10, 8, 11, 9, 0, 2, 1, 3, 14, 12, 15, 13, 4, 6, 5, 7, -1, -1]
- Rank 3: [10, 11, 8, 0, 9, 1, 2, 3, 14, 15, 12, 4, 13, 5, 6, 7, -1, -1]
```

**Key differences from forward schedule:**
1. Starts with HIGHER microbatch IDs (8, 9, 10... instead of 0, 1, 2...)
2. Different interleaving pattern
3. Ends with gradient sync markers (-1, -1)

---

## Model Chunk Mapping (Lines 353-361)

```python
def get_model_chunk_id(microbatch_id):
    """Map microbatch ID to VR chunk (0,1,2,3)"""
    microbatch_id_in_group = microbatch_id % pipeline_parallel_size
    chunk_offset = 0 if microbatch_id < (total_num_microbatches // 2) else 2
    model_chunk_id = microbatch_id_in_group // (pipeline_parallel_size // 2)
    model_chunk_id += chunk_offset
    if microbatch_id == -1:
        model_chunk_id = -1
    return model_chunk_id
```

**Example with 4 devices, 16 total MBs:**

```
MB 0: (0 % 4) // 2 + 0 = 0 → VR0
MB 1: (1 % 4) // 2 + 0 = 0 → VR0
MB 2: (2 % 4) // 2 + 0 = 1 → VR1
MB 3: (3 % 4) // 2 + 0 = 1 → VR1
MB 4: (4 % 4) // 2 + 0 = 0 → VR0
...
MB 8: (0 % 4) // 2 + 2 = 2 → VR2  (offset changes!)
MB 9: (1 % 4) // 2 + 2 = 2 → VR2
MB 10: (2 % 4) // 2 + 2 = 3 → VR3
MB 11: (3 % 4) // 2 + 2 = 3 → VR3
```

**Key insight:** First 8 MBs → VR0/VR1, Second 8 MBs → VR2/VR3

---

## Warmup Phase (Lines 528-589)

```python
for k in range(num_warmup_microbatches):
    # Step 1: Get which microbatch to process (from forward schedule for THIS rank)
    forward_model_chunk_id = get_model_chunk_id(microbatch_idx[k])

    # Step 2: Set VR for this microbatch
    parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)

    # Step 3: Execute forward
    output_tensor = forward_step_helper(microbatch_idx[k], None, 0)

    # Step 4: Determine next microbatch's VR
    next_forward_model_chunk_id = get_model_chunk_id(microbatch_idx[k + 1])

    # Step 5: Route output tensor based on VR transition
    if forward_model_chunk_id == next_forward_model_chunk_id:
        # Same VR: Send to next pipeline stage, receive from prev
        input_tensor = send_forward_recv_forward(output_tensor, ...)
    else:
        # Different VR: Check if same device (lines 569-572)
        if (is_last_stage and next_vr == VR2) or (is_first_stage and next_vr == VR3):
            # Same device: Just detach and re-enable gradients
            input_tensor = output_tensor.detach()
            input_tensor.requires_grad_()
        else:
            # Different device: Use bidirectional P2P
            input_tensor = send_forward_recv_forward_bd0(output_tensor, ...)

    # Step 6: Store input tensor for next microbatch
    input_tensors[next_forward_model_chunk_id].append(input_tensor)

    # Step 7: Deallocate output tensor
    deallocate_output_tensor(output_tensor, ...)
```

**Critical insight:** `microbatch_idx[k]` determines ORDER, but `get_model_chunk_id()` determines VR

---

## 1F1B Phase (Lines 596-688)

This is trickier - it interleaves forward and backward in a 1F1B pattern:

```python
for k in range(unit_remaining * 2):  # Each iteration handles 2 MBs (F and B)
    if k % 2 == 0:  # FORWARD pass
        forward_k = k // 2 + num_warmup_microbatches + offset
        forward_model_chunk_id = get_model_chunk_id(microbatch_idx[forward_k])
        parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
        output_tensor = forward_step_helper(microbatch_idx[forward_k], None, 0)

        # Forward also prepares for backward:
        next_backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[backward_k])
        # Send forward output, receive backward gradient
        output_tensor_grad = send_forward_recv_backward(output_tensor, ...)
        output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    else:  # BACKWARD pass
        backward_k = k // 2 + offset
        backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[backward_k])
        parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
        input_tensor_grad = backward_step_helper(microbatch_idx_b[backward_k])

        # Backward determines next forward's VR
        next_forward_model_chunk_id = get_model_chunk_id(microbatch_idx[forward_k + 1])
        # ... route input_tensor_grad similarly to warmup ...
```

**Key insight:** Forward and backward use DIFFERENT schedules and can be processing different VRs!

Example timeline:
```
Time 0: Forward MB 4 (VR0)
Time 1: Backward MB 8 (VR2)  ← Different VR, different schedule!
Time 2: Forward MB 5 (VR0)
Time 3: Backward MB 9 (VR2)
```

---

## Cooldown Phase (Lines 889-992)

Processes remaining backward passes:

```python
for k in range(2*pipeline_parallel_size - unit_remaining + 2):
    backward_k = k + num_microbatches_mid + unit_remaining
    backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[backward_k])

    if backward_model_chunk_id == -1:  # GRADIENT SYNC MARKER
        # Synchronize gradients across bidirectional pairs
        for i_chunk in range(num_model_chunks // 2):
            allreduce_gradients(model[num_model_chunks // 2 + i_chunk])

    else:  # Regular BACKWARD
        parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
        input_tensor_grad = backward_step_helper(microbatch_idx_b[backward_k])
        # ... route similar to 1F1B ...
```

**Key insight:** Cooldown is PURELY backward passes + gradient syncs

---

## Forward Step Helper (Lines 386-425)

```python
def forward_step_helper(microbatch_id, checkpoint_activations_microbatch, offset):
    """Helper method to run forward step with model split into chunks"""

    # Step 1: Map microbatch to its VR chunk
    model_chunk_id = get_model_chunk_id(microbatch_id)

    # Step 2: Determine which pipeline this belongs to (for profiling)
    # Pipeline 0: VR0 and VR2 (forward direction)
    # Pipeline 1: VR1 and VR3 (backward direction)
    pipeline_id = 0 if model_chunk_id in [0, 2] else 1

    # Step 3: Start profiling (if enabled)
    if profiler:
        profiler.start_microbatch(microbatch_id, pipeline_id, model_chunk_id, 'forward')

    # Step 4: Get input tensor for this microbatch
    # Special case for first stage: append None if no input available
    if parallel_state.is_pipeline_first_stage():
        if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
            input_tensors[model_chunk_id].append(None)  # Input is generated internally

    # Get input tensor from queue (usually index -1, offset for special cases)
    input_tensor = input_tensors[model_chunk_id][-1-offset]

    # Step 5: Execute actual forward pass
    output_tensor = forward_step(
        forward_step_func,
        data_iterator[model_chunk_id],      # Use data iterator for this chunk
        model[model_chunk_id],               # Use model for this chunk
        num_microbatches // 2,               # Batch size for this chunk
        input_tensor,                        # Input computed in Step 4
        forward_data_store,                  # Store for backward
        config,
        collect_non_loss_data,
        checkpoint_activations_microbatch,
    )

    # Step 6: Store output tensor for later use (backward will need it)
    output_tensors[model_chunk_id].append(output_tensor)

    # Step 7: End profiling
    if profiler:
        profiler.end_microbatch(microbatch_id, pipeline_id, model_chunk_id, 'forward')

    return output_tensor
```

**Key Points:**

1. **Chunk-specific processing**: Uses `model[model_chunk_id]`, `data_iterator[model_chunk_id]`, etc.
2. **Queue management**: Gets input from `input_tensors[model_chunk_id]` queue, appends output to `output_tensors[model_chunk_id]`
3. **First stage special case**: First stage generates input internally (None), other stages receive via P2P
4. **Pipeline ID for profiling**: Tracks whether this is forward pipeline (0,2) or backward pipeline (1,3)
5. **Offset parameter**: Used in mid-phase loops for accessing specific items in queue

---

## Backward Step Helper (Lines 455-505)

```python
def backward_step_helper(microbatch_id):
    """Helper method to run backward step with model split into chunks"""

    # Step 1: Map microbatch to its VR chunk
    model_chunk_id = get_model_chunk_id(microbatch_id)

    # Step 2: Determine which pipeline this belongs to (for profiling)
    pipeline_id = 0 if model_chunk_id in [0, 2] else 1

    # Step 3: Start profiling
    if profiler:
        profiler.start_microbatch(microbatch_id, pipeline_id, model_chunk_id, 'backward')

    # Step 4: Handle gradient synchronization (Phase 1: Default sync)
    if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(microbatch_id):
        enable_grad_sync()  # Enable AllReduce for this chunk
        synchronized_model_chunks.add(model_chunk_id)

    # Step 5: Prepare gradient for backward
    # Last stage special case: append None if no gradient available
    if parallel_state.is_pipeline_last_stage():
        if len(output_tensor_grads[model_chunk_id]) == 0:
            output_tensor_grads[model_chunk_id].append(None)  # Loss gradient generated here

    # Step 6: Pop tensors from queues (FIFO order)
    # These were saved during forward passes
    input_tensor = input_tensors[model_chunk_id].pop(0)              # First input from forward
    output_tensor = output_tensors[model_chunk_id].pop(0)            # First output from forward
    output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)  # Gradient from next stage

    # Step 7: Execute actual backward pass
    input_tensor_grad = backward_step(
        input_tensor,        # Input to this chunk (from forward)
        output_tensor,       # Output from this chunk (from forward)
        output_tensor_grad,  # Gradient of output (from next stage)
        model_type,
        config,
    )

    # Step 8: Handle gradient synchronization (Phase 2: Custom sync)
    if config.grad_sync_func is not None:
        grad_sync_microbatch_id = microbatch_id
        if grad_sync_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(grad_sync_microbatch_id):
            grad_sync_chunk_id = get_model_chunk_id(grad_sync_microbatch_id)
            enable_grad_sync()
            config.grad_sync_func(model[grad_sync_chunk_id].parameters())
            synchronized_model_chunks.add(grad_sync_chunk_id)

    # Step 9: Disable gradient sync (will be re-enabled only for specific microbatches)
    disable_grad_sync()

    # Step 10: End profiling
    if profiler:
        profiler.end_microbatch(microbatch_id, pipeline_id, model_chunk_id, 'backward')

    return input_tensor_grad
```

**Key Points:**

1. **FIFO queue management**: Pops tensors in order they were appended (first forward → first backward)
2. **Gradient sync control**:
   - Only sync on LAST microbatch for each chunk (reduces communication overhead)
   - Two phases: default sync and custom sync
3. **Last stage special case**: Last stage generates loss gradient internally (None)
4. **Pipeline tracking**: Uses same pipeline_id as forward for consistency
5. **Always disable sync at end**: Ensures gradients don't sync except on specific microbatches

---

## How Forward/Backward Helpers are Used

**In Warmup Phase:**
```python
for k in range(num_warmup_microbatches):
    forward_model_chunk_id = get_model_chunk_id(microbatch_idx[k])
    parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
    output_tensor = forward_step_helper(microbatch_idx[k], None, 0)  # Helper is called
    # ... P2P communication ...
    deallocate_output_tensor(output_tensor, ...)
```

**In 1F1B Phase (Forward):**
```python
if k % 2 == 0:  # FORWARD
    forward_model_chunk_id = get_model_chunk_id(microbatch_idx[forward_k])
    parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
    output_tensor = forward_step_helper(microbatch_idx[forward_k], None, 0)  # Helper is called
    # ... P2P communication ...
```

**In 1F1B Phase (Backward):**
```python
else:  # BACKWARD
    backward_model_chunk_id = get_model_chunk_id(microbatch_idx_b[backward_k])
    parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
    input_tensor_grad = backward_step_helper(microbatch_idx_b[backward_k])  # Helper is called
    # ... P2P communication ...
```

**Pattern:**
1. Set virtual rank for correct model chunk
2. Call helper to execute forward/backward
3. Handle P2P communication with output/gradient
4. Store in queues for next phase

---

## Execution Synchronization: The CRITICAL Pattern

BitPipe's genius is that **different ranks are at different points in the forward/backward schedules**, but they stay synchronized via P2P communication:

```
Forward Pass:
- Rank 0 executes: MB 0 → (send) → Rank 1 receives
- Rank 1 executes: MB 0 → (send) → Rank 2 receives
- Rank 2 executes: MB 0 → (send) → Rank 3 receives

Backward Pass:
- Rank 3 starts with loss gradient
- Rank 3 executes: MB 8 → (send_backward) → Rank 2 receives
- Rank 2 executes: MB 8 → (send_backward) → Rank 1 receives
- Rank 1 executes: MB 8 → (send_backward) → Rank 0 receives
```

Each rank's **forward schedule** matches the prev rank's **forward schedule** (shifted in time).
Each rank's **backward schedule** matches the next rank's **backward schedule** (shifted in time).

**This keeps the pipeline flowing without deadlock!**

---

## Why Chimera's Deadlock Happened

Chimera is trying to run both VR0 and VR1 forward passes before any backward passes. But BitPipe interleaves them!

**BitPipe pattern (correct):**
```
Rank 0: [Warmup VR0] [1F1B VR0+VR1] [Cooldown VR1]
Rank 1: [Warmup VR0] [1F1B VR0+VR1] [Cooldown VR1]
Rank 2: [Warmup VR1] [1F1B VR1+VR0] [Cooldown VR0]  ← Reversed!
Rank 3: [Warmup VR1] [1F1B VR1+VR0] [Cooldown VR0]  ← Reversed!
```

**Chimera current (deadlock):**
```
Rank 0: [Warmup VR0] [Warmup VR1] [Cooldown] ← Gets stuck when Rank 1 still in warmup VR1
Rank 1: [Warmup VR0] [recv_forward for VR1...] ← Waiting for Rank 0 to send
```

---

## Solution for Chimera

Chimera should probably:
1. **NOT separate VR0 and VR1 warmup phases** - they should be interleaved or follow BitPipe's forward schedule
2. **Use a single forward schedule** that respects P2P communication ordering
3. **Match BitPipe's pattern** but simplified for 2 VRs instead of 4

The forward schedule MUST ensure that when Rank 1 tries to recv_forward, Rank 0 has already sent from the same phase.

