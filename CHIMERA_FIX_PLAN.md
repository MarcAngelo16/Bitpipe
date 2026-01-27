# Chimera 2-VR: Fix Plan Based on BitPipe 4-VR Structure

## Problem Statement

Chimera is experiencing deadlocks because it's missing critical components that BitPipe has. Even though Chimera has only 2 VRs (vs BitPipe's 4 VRs), it **still needs the same fundamental structure** because:

1. ✅ Chimera is bidirectional (like BitPipe)
2. ✅ Chimera has BD groups (paired devices share layers)
3. ✅ Chimera needs boundary checks (first/last stage handling)
4. ✅ Chimera needs proper queue management (FIFO for backward)
5. ✅ Chimera needs gradient synchronization (allreduce_gradients)

---

## ROOT CAUSE: Warmup Calculation Issue ⚠️ CRITICAL!

### The Core Problem

**BitPipe doubles microbatches, Chimera doesn't!**

```python
# BitPipe:
num_microbatches = 4  # User specified
total_num_microbatches = num_microbatches * 2 = 8  # DOUBLED!

# Chimera:
num_microbatches = 4  # User specified
total_num_microbatches = num_microbatches = 4  # NO DOUBLING!
```

**But the warmup calculation assumes BitPipe's doubling:**

```python
# From bitpipe_4vr.py (lines 221-227):
num_warmup_microbatches = pipeline_parallel_size + pipeline_parallel_size // 2
num_warmup_microbatches += (
    pipeline_parallel_rank
    if pipeline_parallel_rank < pipeline_parallel_size // 2
    else pipeline_parallel_size - 1 - pipeline_parallel_rank
)

# For 4 devices:
# Device 0: 4 + 2 + 0 = 6 warmup microbatches
# Device 1: 4 + 2 + 1 = 7 warmup microbatches
# Device 2: 4 + 2 + 1 = 7 warmup microbatches
# Device 3: 4 + 2 + 0 = 6 warmup microbatches
```

**The Deadlock:**

```
Chimera with 4 microbatches, 4 devices:
  total_num_microbatches = 4

But warmup loop tries to process:
  Device 0: 6 microbatches  ❌ (only 4 exist!)
  Device 1: 7 microbatches  ❌ (only 4 exist!)
  Device 2: 7 microbatches  ❌ (only 4 exist!)
  Device 3: 6 microbatches  ❌ (only 4 exist!)

Result: Warmup loop tries to access microbatches that don't exist!
→ Index out of bounds or waiting for data that never arrives
→ DEADLOCK!
```

---

### The Solution for Chimera

**Divide warmup/mid/remaining by 2** (since Chimera doesn't double):

```python
# CHIMERA WARMUP CALCULATION (NEW)

# Step 1: Calculate base warmup (same as BitPipe)
if total_num_microbatches == pipeline_parallel_size:
    # Special case: all warmup, no 1F1B
    num_warmup_microbatches = total_num_microbatches
    # ✅ DO NOT DIVIDE! Already correct.
else:
    # Normal case: warmup + 1F1B + cooldown
    num_warmup_microbatches = pipeline_parallel_size + pipeline_parallel_size // 2

    num_warmup_microbatches += (
        pipeline_parallel_rank
        if pipeline_parallel_rank < pipeline_parallel_size // 2
        else pipeline_parallel_size - 1 - pipeline_parallel_rank
    )

    # ✅ CHIMERA FIX: Divide by 2 (no microbatch doubling)
    num_warmup_microbatches = num_warmup_microbatches // 2

# Step 2: Calculate unit_remaining
unit_remaining = 2 * pipeline_parallel_size - num_warmup_microbatches
# ✅ CHIMERA FIX: Divide by 2
unit_remaining = unit_remaining // 2

# Step 3: Calculate mid-phase
n_loop = total_num_microbatches // pipeline_parallel_size // 2 - 1
num_microbatches_mid = n_loop * pipeline_parallel_size * 2
# ✅ CHIMERA FIX: Divide by 2
num_microbatches_mid = num_microbatches_mid // 2

# Step 4: Calculate remaining
num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches - num_microbatches_mid
# ✅ CHIMERA FIX: Divide by 2
num_microbatches_remaining = num_microbatches_remaining // 2
```

---

### Concrete Example: 4 Devices, 4 Microbatches

**BitPipe (8 total MBs after doubling):**
```
Device 0: warmup=6, mid=0, remaining=2, total=8 ✓
Device 1: warmup=7, mid=0, remaining=1, total=8 ✓
Device 2: warmup=7, mid=0, remaining=1, total=8 ✓
Device 3: warmup=6, mid=0, remaining=2, total=8 ✓
```

**Chimera WITHOUT fix (4 total MBs, no doubling):**
```
Device 0: warmup=6, remaining=? → TRIES TO ACCESS 6 MBs (only 4 exist!) ❌
Device 1: warmup=7, remaining=? → TRIES TO ACCESS 7 MBs (only 4 exist!) ❌
→ DEADLOCK!
```

**Chimera WITH fix (4 total MBs, no doubling):**
```
# Special case: total_num_microbatches (4) == pipeline_parallel_size (4)
if total_num_microbatches == pipeline_parallel_size:
    num_warmup_microbatches = total_num_microbatches  # = 4
    # All forward passes in warmup, no 1F1B, only cooldown

Device 0: warmup=4, mid=0, remaining=0, total=4 ✓
Device 1: warmup=4, mid=0, remaining=0, total=4 ✓
Device 2: warmup=4, mid=0, remaining=0, total=4 ✓
Device 3: warmup=4, mid=0, remaining=0, total=4 ✓
```

**Chimera WITH fix (8 MBs example):**
```
# Normal case: 8 microbatches, 4 devices
total_num_microbatches = 8

# Before division by 2:
Device 0: warmup_base = 6
Device 1: warmup_base = 7

# After division by 2:
Device 0: warmup = 6 // 2 = 3
Device 1: warmup = 7 // 2 = 3 (or 4, depending on rounding)

# This matches the 8 total microbatches correctly!
```

---

### Implementation Code for Chimera

```python
# In chimera_2vr.py (around line 200-240)

# Compute number of warmup and remaining microbatches
num_model_chunks = len(model)  # 2 for Chimera
total_num_microbatches = num_microbatches  # NO DOUBLING!

n_loop = total_num_microbatches // pipeline_parallel_size // 2 - 1

if forward_only:
    num_warmup_microbatches = total_num_microbatches
else:
    # ✅ SPECIAL CASE: All warmup, no 1F1B
    if total_num_microbatches == pipeline_parallel_size:
        num_warmup_microbatches = total_num_microbatches
        # DO NOT DIVIDE! This is already correct for Chimera
    else:
        # ✅ NORMAL CASE: Calculate warmup with division
        num_warmup_microbatches = pipeline_parallel_size + pipeline_parallel_size // 2

        num_warmup_microbatches += (
            pipeline_parallel_rank
            if pipeline_parallel_rank < pipeline_parallel_size // 2
            else pipeline_parallel_size - 1 - pipeline_parallel_rank
        )

        # ✅ CHIMERA FIX: Divide by 2 (no microbatch doubling)
        num_warmup_microbatches = num_warmup_microbatches // 2

# Calculate remaining phases
unit_remaining = 2 * pipeline_parallel_size - num_warmup_microbatches
if total_num_microbatches != pipeline_parallel_size:
    # ✅ CHIMERA FIX: Divide by 2
    unit_remaining = unit_remaining // 2

num_microbatches_mid = n_loop * pipeline_parallel_size * 2
if total_num_microbatches != pipeline_parallel_size:
    # ✅ CHIMERA FIX: Divide by 2
    num_microbatches_mid = num_microbatches_mid // 2

num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches - num_microbatches_mid
# No need to divide remaining (it's already calculated correctly from the above)
```

---

### Why This Matters

**Without this fix:**
- Warmup loop tries to process MORE microbatches than exist
- Schedule indices go out of bounds
- Devices wait for data that never arrives
- **DEADLOCK!**

**With this fix:**
- Warmup, mid, and cooldown phases correctly sized
- All microbatches accounted for
- No index out of bounds
- No deadlock!

**This is likely the PRIMARY cause of Chimera's deadlock issue!**

---

## What Chimera Currently Has vs Needs

| Component | BitPipe 4-VR | Chimera 2-VR Current | Chimera 2-VR Needs |
|-----------|--------------|----------------------|--------------------|
| `forward_step_helper()` | ✅ Has (lines 386-425) | ❌ **MISSING** | ✅ **MUST ADD** |
| `backward_step_helper()` | ✅ Has (lines 455-507) | ❌ **MISSING** | ✅ **MUST ADD** |
| `allreduce_gradients()` | ✅ Has (lines 427-452) | ✅ Has (lines 414-449) | ✅ Already exists |
| Boundary checks (17+) | ✅ Has throughout | ❌ **INCOMPLETE** | ✅ **MUST ADD** |
| Queue management | ✅ Has (lists per VR) | ❌ **INCOMPLETE** | ✅ **MUST ADD** |
| BD group sync calls | ✅ Has (in cooldown) | ⚠️ **PARTIAL** | ✅ **MUST FIX** |

---

## Critical Missing Components

### 1. `forward_step_helper()` - **HIGH PRIORITY**

**Why needed:**
- Manages input/output tensor queues for each VR
- Handles first stage boundary (adds `None` input)
- Sets virtual pipeline rank before execution
- Provides consistent interface for all forward passes

**BitPipe implementation (lines 386-425):**

```python
def forward_step_helper(microbatch_id, checkpoint_activations_microbatch, offset):
    """Helper method to run forward step with model split into chunks"""

    # STEP 1: Determine which VR to use
    model_chunk_id = get_model_chunk_id(microbatch_id)

    # STEP 2: Set current VR rank
    parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

    # STEP 3: Handle first stage (no input from previous stage)
    if parallel_state.is_pipeline_first_stage():
        if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
            input_tensors[model_chunk_id].append(None)

    # STEP 4: Get input tensor from queue
    input_tensor = input_tensors[model_chunk_id][-1 - offset]

    # STEP 5: Execute forward pass
    output_tensor = forward_step(
        forward_step_func,
        data_iterator[model_chunk_id],
        model[model_chunk_id],
        num_microbatches // 2,  # For BitPipe; Chimera: num_microbatches (no doubling)
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data,
        checkpoint_activations_microbatch,
    )

    # STEP 6: Store output tensor in queue
    output_tensors[model_chunk_id].append(output_tensor)

    return output_tensor
```

**Chimera adaptation:**

```python
def forward_step_helper(microbatch_id, checkpoint_activations_microbatch, offset):
    """Helper method to run forward step with model split into chunks (Chimera 2-VR)"""

    # Chimera has 2 VRs: 0 and 1
    model_chunk_id = get_model_chunk_id(microbatch_id)  # Returns 0 or 1

    parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

    # Handle first stage
    if parallel_state.is_pipeline_first_stage():
        if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
            input_tensors[model_chunk_id].append(None)

    input_tensor = input_tensors[model_chunk_id][-1 - offset]

    output_tensor = forward_step(
        forward_step_func,
        data_iterator[model_chunk_id],
        model[model_chunk_id],
        num_microbatches,  # No doubling for Chimera!
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data,
        checkpoint_activations_microbatch,
    )

    output_tensors[model_chunk_id].append(output_tensor)

    return output_tensor
```

---

### 2. `backward_step_helper()` - **HIGH PRIORITY**

**Why needed:**
- Manages FIFO queue popping (`.pop(0)` for oldest first)
- Handles last stage boundary (adds `None` gradient)
- Enables gradient sync for last microbatch per VR
- Provides consistent interface for all backward passes

**BitPipe implementation (lines 455-507):**

```python
def backward_step_helper(microbatch_id):
    """Helper method to run backward step with model split into chunks"""

    # STEP 1: Determine which VR to use
    model_chunk_id = get_model_chunk_id(microbatch_id)

    # STEP 2: Enable gradient sync if last microbatch for this VR
    if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(microbatch_id):
        enable_grad_sync()
        synchronized_model_chunks.add(model_chunk_id)

    # STEP 3: Handle last stage (no gradient from next stage)
    if parallel_state.is_pipeline_last_stage():
        if len(output_tensor_grads[model_chunk_id]) == 0:
            output_tensor_grads[model_chunk_id].append(None)

    # STEP 4: Pop tensors from queues (FIFO order)
    input_tensor = input_tensors[model_chunk_id].pop(0)
    output_tensor = output_tensors[model_chunk_id].pop(0)
    output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)

    # STEP 5: Execute backward pass
    input_tensor_grad = backward_step(
        input_tensor, output_tensor, output_tensor_grad, model_type, config
    )

    # STEP 6: Disable gradient sync
    disable_grad_sync()

    return input_tensor_grad
```

**Chimera adaptation:**

```python
def backward_step_helper(microbatch_id):
    """Helper method to run backward step with model split into chunks (Chimera 2-VR)"""

    model_chunk_id = get_model_chunk_id(microbatch_id)  # Returns 0 or 1

    # Enable gradient sync if last microbatch for this VR
    if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(microbatch_id):
        enable_grad_sync()
        synchronized_model_chunks.add(model_chunk_id)

    # Handle last stage
    if parallel_state.is_pipeline_last_stage():
        if len(output_tensor_grads[model_chunk_id]) == 0:
            output_tensor_grads[model_chunk_id].append(None)

    # CRITICAL: Pop from front (FIFO)
    input_tensor = input_tensors[model_chunk_id].pop(0)
    output_tensor = output_tensors[model_chunk_id].pop(0)
    output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)

    # Execute backward
    input_tensor_grad = backward_step(
        input_tensor, output_tensor, output_tensor_grad, model_type, config
    )

    disable_grad_sync()

    return input_tensor_grad
```

---

### 3. Boundary Checks - **HIGH PRIORITY**

**Where to add checks:**

```python
# ❌ WRONG (causes deadlock):
input_tensor = recv_forward(tensor_shape, config)

# ✅ CORRECT:
if not parallel_state.is_pipeline_first_stage():
    input_tensor = recv_forward(tensor_shape, config)

# ❌ WRONG (causes deadlock):
send_forward(output_tensor, ...)

# ✅ CORRECT:
if not parallel_state.is_pipeline_last_stage():
    send_forward(output_tensor, ...)
```

**Critical checks needed in Chimera:**

1. **Before receiving forward:**
   ```python
   if not parallel_state.is_pipeline_first_stage():
       input_tensor = recv_forward(...)
   ```

2. **Before sending forward:**
   ```python
   if not parallel_state.is_pipeline_last_stage():
       send_forward(output_tensor, ...)
   ```

3. **Before receiving backward:**
   ```python
   if not parallel_state.is_pipeline_last_stage():
       output_tensor_grad = recv_backward(...)
   ```

4. **Before sending backward:**
   ```python
   if not parallel_state.is_pipeline_first_stage():
       send_backward(input_tensor_grad, ...)
   ```

5. **Set tensors to None at boundaries:**
   ```python
   # Last stage
   if parallel_state.is_pipeline_last_stage():
       output_tensor = None

   # First stage
   if parallel_state.is_pipeline_first_stage():
       input_tensor_grad = None
   ```

---

### 4. Queue Management

**Requirements:**

```python
# At scheduler start (Chimera: 2 lists, BitPipe: 4 lists)
input_tensors = [[] for _ in range(len(model))]      # [[], []]
output_tensors = [[] for _ in range(len(model))]     # [[], []]
output_tensor_grads = [[] for _ in range(len(model))] # [[], []]

# Forward: Append to end (newest)
output_tensors[model_chunk_id].append(output_tensor)

# Backward: Pop from front (oldest, FIFO)
input_tensor = input_tensors[model_chunk_id].pop(0)
output_tensor = output_tensors[model_chunk_id].pop(0)
output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
```

---

### 5. BD Gradient Sync

**Already implemented in Chimera** (lines 414-449), but must be **called correctly**:

```python
# In cooldown phase, when microbatch_id == -1
if backward_model_chunk_id == -1:
    # Sync both VR0 and VR1
    for i_chunk in range(num_model_chunks):  # 0, 1 for Chimera
        allreduce_gradients(model[i_chunk])

    # Receive next gradient if needed
    if not next_backward_model_chunk_id == -1:
        output_tensor_grads[next_backward_model_chunk_id].append(
            recv_backward(tensor_shape=tensor_shape, config=config)
        )
```

---

## Implementation Checklist

### Phase 1: Add Helper Functions ✅ CRITICAL

- [ ] Add `forward_step_helper()` to `chimera_2vr.py`
  - [ ] Queue management
  - [ ] First stage handling (append `None`)
  - [ ] VR switching (`set_virtual_pipeline_model_parallel_rank`)
  - [ ] Return output tensor

- [ ] Add `backward_step_helper()` to `chimera_2vr.py`
  - [ ] FIFO queue popping (`.pop(0)`)
  - [ ] Last stage handling (append `None` gradient)
  - [ ] Gradient sync triggering (`enable_grad_sync()`)
  - [ ] Return input gradient

### Phase 2: Add Boundary Checks ✅ CRITICAL

- [ ] Warmup phase:
  - [ ] Check `is_pipeline_first_stage()` before `recv_forward()`
  - [ ] Check `is_pipeline_last_stage()` before `send_forward()`
  - [ ] Set `output_tensor = None` if last stage

- [ ] Steady state (1F1B):
  - [ ] Forward: Check before `send_forward_recv_backward()`
  - [ ] Backward: Check before `send_backward_recv_forward()`
  - [ ] Set boundary tensors to `None`

- [ ] Cooldown phase:
  - [ ] Check `is_pipeline_first_stage()` before `recv_backward()`
  - [ ] Check before `send_backward_recv_backward()`
  - [ ] Set `input_tensor_grad = None` if first stage

### Phase 3: Fix Gradient Sync ✅ IMPORTANT

- [ ] Verify `allreduce_gradients()` is correct for 2 VRs
- [ ] Add gradient sync markers (`-1`) to backward schedule
- [ ] Call `allreduce_gradients()` in cooldown phase
- [ ] Test that BD groups sync correctly

### Phase 4: Replace Direct Calls

- [ ] Replace all direct `forward_step()` calls with `forward_step_helper()`
- [ ] Replace all direct `backward_step()` calls with `backward_step_helper()`
- [ ] Verify all P2P calls have boundary checks

### Phase 5: Testing

- [ ] Test with 4 GPUs, 24 layers
- [ ] Verify no deadlock
- [ ] Check loss convergence
- [ ] Compare with BitPipe 4-VR performance

---

## Example: Warmup Phase (Before vs After)

### ❌ BEFORE (Current Chimera - Causes Deadlock)

```python
for k in range(num_warmup_microbatches):
    # Direct forward_step call (no helper)
    output_tensor = forward_step(...)

    # NO BOUNDARY CHECK! ← Deadlock if last stage!
    send_forward(output_tensor, ...)

    # NO BOUNDARY CHECK! ← Deadlock if first stage!
    input_tensor = recv_forward(...)
```

### ✅ AFTER (Fixed Chimera)

```python
for k in range(num_warmup_microbatches):
    forward_model_chunk_id = get_model_chunk_id(microbatch_idx[k])
    parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)

    # Use helper function
    output_tensor = forward_step_helper(microbatch_idx[k], None, 0)

    # Boundary check before sending
    if parallel_state.is_pipeline_last_stage():
        output_tensor = None

    # Boundary check before communication
    if not parallel_state.is_pipeline_last_stage():
        input_tensor = send_forward_recv_forward(
            output_tensor,
            recv_prev=not parallel_state.is_pipeline_first_stage(),
            tensor_shape=tensor_shape,
            config=config,
        )
```

---

## Key Differences: Chimera vs BitPipe

| Aspect | BitPipe 4-VR | Chimera 2-VR |
|--------|--------------|--------------|
| Number of VRs | 4 | 2 |
| Microbatch doubling | Yes (4 → 8) | No (4 → 4) |
| Same-device transitions | Yes (VR0→VR2, VR1→VR3) | No |
| Layer pattern | V-shaped | Sequential + swap |
| BD groups | [0,1], [3,2] (V-shaped ranks) | [0,3], [1,2] (sequential ranks) |
| Helper functions | ✅ Has both | ❌ Missing both |
| Boundary checks | ✅ 17+ checks | ❌ Incomplete |
| Queue management | ✅ Correct | ❌ Needs fixing |

**Key Insight:** Despite differences, Chimera needs the **same fundamental structure** as BitPipe because both are bidirectional pipelines!

---

## Summary

Chimera is **not just a simpler BitPipe** - it needs the **same robust infrastructure**:

1. ✅ **Helper functions** - Abstract away complexity, ensure consistency
2. ✅ **Boundary checks** - Prevent deadlocks at pipeline edges
3. ✅ **Queue management** - Maintain correct temporal ordering
4. ✅ **BD gradient sync** - Ensure correctness of shared layers

Without these, Chimera will deadlock or produce incorrect results!

---

## Next Steps

1. **Implement helper functions** (highest priority)
2. **Add all boundary checks** (prevents deadlock)
3. **Test on 4 GPUs** (minimum for bidirectional pipeline)
4. **Verify BD sync works** (check gradients are averaged)
5. **Compare performance with BitPipe** (validate simplified design)
