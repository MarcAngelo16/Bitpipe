# Session Summary - December 10, 2025

## Context

Continuing work on Chimera 2-VR deadlock fixes. Previous session identified that BitPipe's complex warmup/1F1B/cooldown phase structure was causing phase mismatch deadlocks in Chimera.

## What We Tried: Simple 2-Phase Approach

### Attempt: Remove Complex Phases

**Goal**: Simplify execution by replacing BitPipe's warmup/1F1B/cooldown with a simple 2-phase loop:
- Phase 1: Execute ALL forward passes sequentially
- Phase 2: Execute ALL backward passes sequentially

**Changes Made**:
1. Removed `forward_step_helper()` and `backward_step_helper()` dependencies
2. Replaced helper calls with direct `forward_step()` / `backward_step()` calls
3. Inline tensor recv/send operations (no pre-queuing)
4. File reduced from 839 lines → ~720 lines

**Code Structure**:
```python
# Phase 1: All Forwards
for microbatch_id in microbatch_idx:
    model_chunk_id = get_model_chunk_id(microbatch_id)
    set_virtual_rank(model_chunk_id)

    # Recv (if not first stage)
    input_tensor = recv_forward(...) if not is_first_stage() else None

    # Execute
    output_tensor = forward_step(...)

    # Store for backward
    input_tensors[model_chunk_id].append(input_tensor)
    output_tensors[model_chunk_id].append(output_tensor)

    # Send (if not last stage)
    if not is_last_stage():
        send_forward(output_tensor)

# Phase 2: All Backwards
for microbatch_id in microbatch_idx_b:
    # Handle sync markers (-1)
    if microbatch_id == -1:
        allreduce_gradients(...)
        continue

    model_chunk_id = get_model_chunk_id(microbatch_id)
    set_virtual_rank(model_chunk_id)

    # Pop stored tensors
    input_tensor = input_tensors[model_chunk_id].pop(0)
    output_tensor = output_tensors[model_chunk_id].pop(0)

    # Recv gradient (if not last in backward)
    output_tensor_grad = recv_backward(...) if not is_last_for_backward() else None

    # Execute
    input_tensor_grad = backward_step(...)

    # Send gradient (if not first in backward)
    if not is_first_for_backward():
        send_backward(input_tensor_grad)
```

## Why It Doesn't Work

### Issue 1: Schedule Order Mismatch (Phase Desynchronization)

**Symptom**: Ranks finish forwards at different times, causing deadlock when one rank enters Phase 2 while others are still in Phase 1.

**Example from logs**:
```
Rank 0: [FWD DONE] Processed 8 forwards
Rank 0: === PHASE 2: All Backwards ===
Rank 0: [BWD 0] MB 2, VR 1
        ↓ Tries to recv_backward from next stage

Rank 3: [FWD 6] MB 4, VR 0  ← Still in Phase 1!
        ↓ Not ready to send_backward yet

Result: Rank 0 blocks waiting, Rank 3 blocks on forward recv → DEADLOCK
```

**Root Cause**: The forward schedules are designed for **interleaved execution**, NOT sequential execution:

```python
# Rank 0 forward schedule: [0, 4, 1, 5, 2, 6, 3, 7]
# Rank 1 forward schedule: [0, 2, 1, 3, 4, 6, 5, 7]

# Sequential execution breaks coordination:
Rank 0 sends MB 0 → Rank 1 expects MB 0 ✓
Rank 0 sends MB 4 → Rank 1 expects MB 2 ✗ MISMATCH
```

The schedules assume **pipeline coordination** where ranks alternate sending/receiving in a specific pattern. Sequential execution breaks this coordination because:
- Sender executes its schedule in order
- Receiver executes its (different) schedule in order
- They fall out of sync immediately

### Issue 2: Bidirectional Ambiguity in Boundary Checks

**Problem**: `is_pipeline_first_stage()` and `is_pipeline_last_stage()` return **global device positions**, but Chimera has **bidirectional pipelines** where each device is BOTH first AND last depending on VR.

**Example**:
```python
Device 0 (first device globally):
  - VR0 forward:  FIRST stage (sends to Dev 1)     ← is_first = True ✓
  - VR1 forward:  LAST stage  (receives from Dev 1) ← is_last = False ✗
  - VR0 backward: LAST stage  (no recv)            ← is_last = False ✗
  - VR1 backward: FIRST stage (no send)            ← is_first = True ✓

Device 3 (last device globally):
  - VR0 forward:  LAST stage  (receives from Dev 2) ← is_last = True ✓
  - VR1 forward:  FIRST stage (sends to Dev 2)     ← is_first = False ✗
  - VR0 backward: FIRST stage (no send)            ← is_first = False ✗
  - VR1 backward: LAST stage  (no recv)            ← is_last = True ✓
```

**What We Tried**: VR-aware boundary checks
```python
is_last_stage_for_backward = (
    (model_chunk_id == 0 and is_pipeline_last_stage()) or
    (model_chunk_id == 1 and is_pipeline_first_stage())
)
```

**Still didn't work** because the fundamental issue is Issue 1 (schedule mismatch), not just boundary checks.

### Issue 3: Missing Coordination Protocol

Sequential execution lacks the **implicit coordination** that warmup/1F1B/cooldown provides:

**BitPipe's coordination (implicit in phases)**:
- Warmup: Ranks know exactly how many forwards before first backward
- 1F1B: Alternate forward/backward maintains sync
- Cooldown: Drain backwards with gradient sync markers

**Our sequential approach**:
- No coordination on when to transition between phases
- No mechanism to ensure sender/receiver are processing the same microbatch
- Ranks execute independently → inevitable desync

## Key Realization

### Chimera Needs Its Own Phase Logic

**User Insight**:
> "chimera have its own interleaving phase and its cooldown phase but just different algorithm, so we might need to create our own logic"

**Why We Can't Just Copy BitPipe's Phases**:
1. **Microbatch doubling**: BitPipe doubles microbatches (8 → 16), Chimera doesn't
2. **V-shaped pattern**: BitPipe uses 4 VRs with complex V-shape, Chimera uses 2 VRs sequential
3. **Warmup calculation**: BitPipe's ceiling division assumes doubling: `(num_warmup + 1) // 2`

**Why We Can't Use Simple Sequential Execution**:
- Chimera is still a **bidirectional pipeline** (2 VRs going opposite directions)
- Needs interleaved forward/backward to avoid bubbles
- Schedules are designed for interleaved execution, not sequential

**What Chimera Needs**:
- Custom warmup phase (different count per rank, no doubling assumption)
- Custom 1F1B-like interleaving phase (but with 2 VRs, not 4)
- Custom cooldown phase (with BD gradient sync for 2 VRs)

## Current Blocker

**Problem**: Don't understand BitPipe's phase looping logic well enough to adapt it for Chimera.

**What we need to understand**:
1. How does BitPipe determine warmup count per rank?
2. How does the 1F1B phase coordinate forward/backward interleaving?
3. How do helper functions (`forward_step_helper`, `backward_step_helper`) maintain tensor queues?
4. When does BitPipe pre-receive tensors vs receive inline?
5. How do sync markers (-1) integrate with the phase loop?

**Current state of knowledge**:
- ✓ Understand the schedules (forward/backward microbatch order)
- ✓ Understand layer distribution (offset calculation)
- ✓ Understand gradient sync (BD groups, allreduce_gradients)
- ✗ Don't understand the phase loop execution pattern
- ✗ Don't understand how BitPipe maintains synchronization across ranks

## Next Steps

### Immediate Next Session

1. **Deep dive into BitPipe phase logic**:
   - Read `bitpipe_4vr.py` warmup/1F1B/cooldown loops carefully
   - Trace execution: when does rank i send? when does rank i+1 receive?
   - Understand helper function tensor queue management
   - Document the coordination protocol

2. **Design Chimera phase logic**:
   - Adapt warmup calculation for 2 VRs (no doubling)
   - Design 1F1B interleaving for bidirectional 2-VR
   - Determine when to pre-receive vs inline receive
   - Handle boundary conditions for bidirectional flow

3. **Implement Chimera phases**:
   - Write custom warmup loop
   - Write custom 1F1B loop
   - Write custom cooldown loop
   - Integrate with existing schedules

### Open Questions

1. **Warmup count**: Should all ranks have same warmup? Or different like BitPipe?
2. **Helper functions**: Should Chimera use helpers or inline recv/send?
3. **Tensor queues**: When to queue vs when to use directly?
4. **Sync markers**: Should they be in backward schedule or handled separately?

## Files Modified

### megatron/core/pipeline_parallel/schedule_impl/chimera/chimera_2vr.py

**Lines 620-661**: Forward loop (replaced helper with direct calls)
```python
# Before
output_tensor = forward_step_helper(microbatch_id, None)

# After
if not is_pipeline_first_stage():
    input_tensor = recv_forward(tensor_shape, config)
else:
    input_tensor = None

output_tensor = forward_step(...)
input_tensors[model_chunk_id].append(input_tensor)
output_tensors[model_chunk_id].append(output_tensor)

if not is_pipeline_last_stage():
    send_forward(output_tensor)
```

**Lines 663-730**: Backward loop (replaced helper with direct calls + VR-aware boundaries)
```python
# Pop stored tensors
input_tensor = input_tensors[model_chunk_id].pop(0)
output_tensor = output_tensors[model_chunk_id].pop(0)

# VR-aware boundary check
is_last_stage_for_backward = (
    (model_chunk_id == 0 and is_pipeline_last_stage()) or
    (model_chunk_id == 1 and is_pipeline_first_stage())
)

if is_last_stage_for_backward:
    output_tensor_grad = None
else:
    output_tensor_grad = recv_backward(tensor_shape, config)

input_tensor_grad = backward_step(...)

is_first_stage_for_backward = (
    (model_chunk_id == 0 and is_pipeline_first_stage()) or
    (model_chunk_id == 1 and is_pipeline_last_stage())
)

if not is_first_stage_for_backward:
    send_backward(input_tensor_grad)
```

## Conclusion

The simple 2-phase approach fails because:
1. **Schedules assume interleaved execution** → sequential breaks coordination
2. **Global boundary checks don't work** → bidirectional pipelines need VR-aware logic
3. **Missing coordination protocol** → ranks desynchronize without phase structure

**The path forward**: Chimera needs custom warmup/1F1B/cooldown phases adapted for 2-VR bidirectional pipeline, but we need to understand BitPipe's phase logic first before we can design Chimera's equivalent.

**User's key insight**: "we need to create our own phase logic, like chimera still is a bidirectional pipelines so we cant just make it like 1f1b, even though we have the schedluing logic, we need a looping patern phase to support it"
