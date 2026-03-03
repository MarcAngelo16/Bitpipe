# Chimera 2-VR Implementation Breakdown

## Overview

This document breaks down the exact changes needed to implement the redesigned communication pattern from `CHIMERA_DESIGN.md`.

---

## Current Code Structure (chimera_2vr.py)

### Phase 1: Warmup (lines ~970-1040)
- **Status**: ✅ Already correct (no changes needed)
- Uses combined send+recv except for first step on edge ranks (safe)

### Phase 2: 1F1B (lines ~1040-1301)
- **Status**: ⚠️ Needs minor fix
- Location: Line 1222 (bridge index bug)

### Phase 3: Bridge (lines 1217-1281)
- **Status**: ⚠️ Needs 1 critical fix
- Location: Line 1222
- **BUG**: `first_cooldown_mb = microbatch_idx_b[bwd_idx]` (wrong index)
- **FIX**: `first_cooldown_mb = microbatch_idx_b[bwd_idx + 1]`

### Phase 4: Cooldown (lines 1303-1483)
- **Status**: ❌ Needs major redesign
- **Current pattern**: Deferred send (causes deadlock)
- **New pattern**: Chain (pre-fetched grad → compute → send+recv)

---

## Function Breakdown: What Needs to Change

### 1. Bridge Fix (CRITICAL - Line 1222)

**Current code:**
```python
first_cooldown_mb = microbatch_idx_b[bwd_idx]  # BUG: reads CURRENT backward
```

**Fixed code:**
```python
first_cooldown_mb = microbatch_idx_b[bwd_idx + 1]  # First cooldown item
```

**Why**: At this point, `bwd_idx` hasn't been incremented yet (increment at line 1299). Reading `bwd_idx` gives us the CURRENT backward MB, not the first cooldown MB.

**Impact**:
- Edge ranks (R0, R3): Bridge fails → falls back to standalone send → deadlock
- Middle ranks (R1, R2): Bridge succeeds but pre-fetches WRONG VR → cooldown k=0 still does bare recv → deadlock

---

### 2. Cooldown Redesign (MAJOR - Lines 1303-1483)

The current cooldown has TWO patterns we need to consolidate into ONE chain pattern.

#### 2a. New Helper Function: `get_next_item_info()`

**Purpose**: Peek at next backward item to determine if we need to recv for it

**Location**: Add before cooldown loop (around line 1316)

**Signature**:
```python
def get_next_item_info(microbatch_idx_b, current_bwd_idx):
    """
    Returns info about the next non-sync backward item.

    Returns:
        (next_mb_id, next_vr, needs_recv, is_sync)
        - next_mb_id: microbatch ID (-1 if sync, None if end)
        - next_vr: VR for next item (None if sync or end)
        - needs_recv: True if next item needs gradient recv
        - is_sync: True if next is a sync marker
    """
```

**Implementation**:
```python
def get_next_item_info(microbatch_idx_b, current_bwd_idx, pipeline_parallel_rank, pipeline_parallel_size):
    """Look ahead to next backward item"""
    if current_bwd_idx + 1 >= len(microbatch_idx_b):
        return None, None, False, False  # End of schedule

    next_mb = microbatch_idx_b[current_bwd_idx + 1]

    if next_mb == -1:
        return -1, None, False, True  # Sync marker

    next_vr = get_model_chunk_id(next_mb, pipeline_parallel_size)
    next_is_grad_first = is_vr_first_stage_for_gradient(next_vr, pipeline_parallel_rank, pipeline_parallel_size)
    needs_recv = not next_is_grad_first

    return next_mb, next_vr, needs_recv, False
```

---

#### 2b. New Helper Function: `post_sync_recv_if_needed()`

**Purpose**: After allreduce sync, recv gradient if next item needs it

**Location**: Add before cooldown loop (around line 1317)

**Signature**:
```python
def post_sync_recv_if_needed(next_mb, next_vr, needs_recv, tensor_shape, config):
    """
    Safely recv after allreduce sync.
    Returns the received gradient or None.
    """
```

**Implementation**:
```python
def post_sync_recv_if_needed(next_mb, next_vr, needs_recv, tensor_shape, config, pipeline_parallel_rank):
    """Recv gradient after sync marker (safe because allreduce synchronized all ranks)"""
    if not needs_recv or next_mb is None:
        return None

    print_all_ranks(f"[Rank{pipeline_parallel_rank}] POST-SYNC RECV: VR{next_vr} grad for MB{next_mb}")

    if next_vr == 0:
        # VR0 grads flow 3→2→1→0 (recv from next)
        grad = p2p_communication.chimera_grad_recv_next_only(tensor_shape, config)
    else:
        # VR1 grads flow 0→1→2→3 (recv from prev)
        grad = p2p_communication.chimera_grad_recv_prev_only(tensor_shape, config)

    print_all_ranks(f"[Rank{pipeline_parallel_rank}] POST-SYNC RECV done for VR{next_vr}")
    return grad
```

---

#### 2c. Rewrite Cooldown Loop Logic

**Current pattern (BROKEN)**:
```
1. Check if have bridged grad → use it
2. Check if have pending send + need recv → combine them
3. Check if have pending send + no recv → send only
4. Check if no pending + need recv → recv only (DEADLOCK!)
5. Compute backward
6. Save result as pending grad
```

**New pattern (CHAIN)**:
```
1. If sync marker:
   a. Flush pending grad (standalone send - safe)
   b. Allreduce sync
   c. Post-sync recv if next needs grad (standalone recv - safe)
   d. Store received grad for next iteration
2. If backward item:
   a. Get gradient input:
      - If have bridged grad and VR matches → use it
      - Else if have pending send + need recv → combined send+recv
      - Else if have pending send + no recv → send only
      - Else if need recv → ERROR (bridge should have pre-fetched!)
      - Else → grad first (no input needed)
   b. Compute backward
   c. If not grad_last → save result as pending for next iteration
```

---

### 3. Detailed Cooldown Code Structure

**New cooldown loop (pseudocode with line numbers)**:

```python
# Line ~1314: Initialize cooldown state
cooldown_k = 0
pending_grad_to_send = None
pending_grad_vr = None
pre_received_grad = None  # NEW: for post-sync recvs
pre_received_vr = None    # NEW

while bwd_idx < len(microbatch_idx_b):
    microbatch_id = microbatch_idx_b[bwd_idx]

    # ===== SYNC MARKER HANDLING =====
    if microbatch_id == -1:
        # Step 1: Flush pending grad BEFORE allreduce
        if pending_grad_to_send is not None:
            print_all_ranks(f"PRE-SYNC FLUSH: VR{pending_grad_vr}")
            if pending_grad_vr == 0:
                p2p_communication.chimera_grad_send_prev_only(pending_grad_to_send, config)
            else:
                p2p_communication.chimera_grad_send_next_only(pending_grad_to_send, config)
            pending_grad_to_send = None
            pending_grad_vr = None

        # Step 2: Allreduce sync
        print_all_ranks(f"SYNC MARKER at bwd_idx={bwd_idx}")
        enable_grad_sync()
        for chunk_id in range(len(model)):
            if chunk_id not in synchronized_model_chunks:
                allreduce_gradients(model[chunk_id])
                synchronized_model_chunks.add(chunk_id)
        disable_grad_sync()

        # Step 3: Post-sync recv if next item needs grad
        next_mb, next_vr, needs_recv, _ = get_next_item_info(
            microbatch_idx_b, bwd_idx, pipeline_parallel_rank, pipeline_parallel_size
        )
        if needs_recv:
            pre_received_grad = post_sync_recv_if_needed(
                next_mb, next_vr, needs_recv, tensor_shape, config, pipeline_parallel_rank
            )
            pre_received_vr = next_vr

        bwd_idx += 1
        cooldown_k += 1
        continue

    # ===== BACKWARD ITEM HANDLING =====
    model_chunk_id = get_model_chunk_id(microbatch_id, pipeline_parallel_size)
    parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

    print_all_ranks(f"[COOLDOWN k={cooldown_k}] BWD MB{microbatch_id}, VR{model_chunk_id}")

    # Pop tensors
    input_tensor = input_tensors[model_chunk_id].pop(0)
    output_tensor = output_tensors[model_chunk_id].pop(0)

    # Determine gradient properties
    cool_is_grad_first = is_vr_first_stage_for_gradient(model_chunk_id, pipeline_parallel_rank, pipeline_parallel_size)
    cool_is_grad_last = is_vr_last_stage_for_gradient(model_chunk_id, pipeline_parallel_rank, pipeline_parallel_size)

    # ===== GET GRADIENT INPUT =====
    output_tensor_grad = None

    # Check if we have pre-received grad (from post-sync or bridge)
    have_pre_received = (pre_received_grad is not None and pre_received_vr == model_chunk_id)
    have_bridged_grad = (bridged_cooldown_grad is not None and bridged_cooldown_vr == model_chunk_id)
    have_pending_send = pending_grad_to_send is not None
    need_recv = not cool_is_grad_first

    if have_bridged_grad:
        # Use bridged gradient from 1F1B→cooldown transition
        print_all_ranks(f"Using BRIDGED grad for VR{model_chunk_id} MB{microbatch_id}")
        output_tensor_grad = bridged_cooldown_grad
        bridged_cooldown_grad = None
        bridged_cooldown_vr = None

        # If also have pending send, must flush it
        # This happens when middle ranks use bridged grad at cooldown k=0
        # but still have a pending send from... wait, no.
        # At cooldown k=0, there's no pending send yet (first cooldown item).
        # The bridge happens at end of 1F1B, so pending_grad is None at start of cooldown.
        # So this case (have_bridged_grad + have_pending_send) should never happen.
        # But keep the check for safety.
        if have_pending_send:
            print_all_ranks(f"WARNING: Have both bridged grad and pending send - flushing pending")
            if pending_grad_vr == 0:
                p2p_communication.chimera_grad_send_prev_only(pending_grad_to_send, config)
            else:
                p2p_communication.chimera_grad_send_next_only(pending_grad_to_send, config)
            pending_grad_to_send = None
            pending_grad_vr = None

    elif have_pre_received:
        # Use pre-received gradient from post-sync recv
        print_all_ranks(f"Using POST-SYNC pre-received grad for VR{model_chunk_id} MB{microbatch_id}")
        output_tensor_grad = pre_received_grad
        pre_received_grad = None
        pre_received_vr = None

        # Similar to above - should not have pending send right after sync
        # because we flushed before allreduce. But check for safety.
        if have_pending_send:
            print_all_ranks(f"WARNING: Have both post-sync grad and pending send")
            if pending_grad_vr == 0:
                p2p_communication.chimera_grad_send_prev_only(pending_grad_to_send, config)
            else:
                p2p_communication.chimera_grad_send_next_only(pending_grad_to_send, config)
            pending_grad_to_send = None
            pending_grad_vr = None

    elif have_pending_send and need_recv:
        # CHAIN: Combined send previous + recv current
        print_all_ranks(f"CHAIN: send VR{pending_grad_vr} + recv VR{model_chunk_id} for MB{microbatch_id}")

        if pending_grad_vr == 0 and model_chunk_id == 0:
            # VR0→VR0: send_prev + recv_next
            output_tensor_grad = p2p_communication.chimera_grad_send_prev_recv_next(
                pending_grad_to_send, tensor_shape, config
            )
        elif pending_grad_vr == 0 and model_chunk_id == 1:
            # VR0→VR1: send_prev + recv_prev
            output_tensor_grad = p2p_communication.chimera_grad_send_prev_recv_prev(
                pending_grad_to_send, tensor_shape, config
            )
        elif pending_grad_vr == 1 and model_chunk_id == 0:
            # VR1→VR0: send_next + recv_next
            output_tensor_grad = p2p_communication.chimera_grad_send_next_recv_next(
                pending_grad_to_send, tensor_shape, config
            )
        else:  # pending_grad_vr == 1 and model_chunk_id == 1
            # VR1→VR1: send_next + recv_prev
            output_tensor_grad = p2p_communication.chimera_grad_send_next_recv_prev(
                pending_grad_to_send, tensor_shape, config
            )

        pending_grad_to_send = None
        pending_grad_vr = None
        print_all_ranks(f"CHAIN done for MB{microbatch_id}")

    elif have_pending_send and not need_recv:
        # Send only (current is grad_first, no incoming grad needed)
        print_all_ranks(f"Send only VR{pending_grad_vr} (grad_first for VR{model_chunk_id})")
        if pending_grad_vr == 0:
            p2p_communication.chimera_grad_send_prev_only(pending_grad_to_send, config)
        else:
            p2p_communication.chimera_grad_send_next_only(pending_grad_to_send, config)
        pending_grad_to_send = None
        pending_grad_vr = None
        output_tensor_grad = None  # grad_first

    elif not have_pending_send and need_recv:
        # ERROR: This should NOT happen with correct bridge!
        # If we reach here, it means:
        # - No bridged grad (bridge failed or wrong VR)
        # - No post-sync pre-received grad
        # - No pending send from previous iteration
        # - But we need a recv
        # This is the DEADLOCK case from the old code!
        raise RuntimeError(
            f"[Rank{pipeline_parallel_rank}] DEADLOCK DETECTED: Bare recv at cooldown k={cooldown_k}, "
            f"MB{microbatch_id}, VR{model_chunk_id}. "
            f"Bridge should have pre-fetched this gradient! "
            f"have_bridged_grad={have_bridged_grad}, have_pre_received={have_pre_received}, "
            f"have_pending_send={have_pending_send}, need_recv={need_recv}"
        )

    else:
        # No pending send, no recv needed (grad_first)
        print_all_ranks(f"Grad first for VR{model_chunk_id} MB{microbatch_id} - no recv")
        output_tensor_grad = None

    # ===== COMPUTE BACKWARD =====
    input_tensor_grad = backward_step(
        input_tensor, output_tensor, output_tensor_grad, model_type, config
    )
    print_all_ranks(f"Backward done for MB{microbatch_id}, VR{model_chunk_id}")

    # ===== SAVE FOR NEXT ITERATION =====
    if not cool_is_grad_last:
        pending_grad_to_send = input_tensor_grad
        pending_grad_vr = model_chunk_id
        print_all_ranks(f"Saved grad VR{model_chunk_id} for chain")
    else:
        print_all_ranks(f"Grad last for VR{model_chunk_id} - no send needed")

    bwd_idx += 1
    cooldown_k += 1

# ===== FINAL FLUSH =====
if pending_grad_to_send is not None:
    print_all_ranks(f"FINAL FLUSH: VR{pending_grad_vr}")
    if pending_grad_vr == 0:
        p2p_communication.chimera_grad_send_prev_only(pending_grad_to_send, config)
    else:
        p2p_communication.chimera_grad_send_next_only(pending_grad_to_send, config)
```

---

## Summary of Changes

### ✅ Simple Changes (1 line)
1. **Line 1222**: `bwd_idx` → `bwd_idx + 1`

### 🔧 Moderate Changes (new helper functions)
2. **Add `get_next_item_info()`**: ~20 lines
3. **Add `post_sync_recv_if_needed()`**: ~15 lines

### 🔨 Major Changes (rewrite)
4. **Cooldown loop**: ~150 lines (replace lines 1314-1469)
   - Add pre-receive state tracking
   - Add error detection for bare recvs
   - Reorganize logic flow to use chain pattern

---

## Testing Strategy

### Phase 1: Bridge fix only
1. Apply line 1222 fix
2. Test with `CHIMERA_DEBUG=1`
3. Verify all ranks successfully bridge
4. Verify correct VR is pre-fetched

### Phase 2: Cooldown redesign
1. Add helper functions
2. Rewrite cooldown loop
3. Test with 4 GPUs, 4 microbatches
4. Verify no deadlocks
5. Verify correct gradient flow

### Phase 3: Validation
1. Test with different GPU counts (4, 6, 8)
2. Test with different microbatch counts (4, 8, 16)
3. Verify convergence matches BitPipe
4. Profile performance

---

## Key Insights for Debugging

### What to watch for:
1. **Bridge success logs**: All 4 ranks should log "BRIDGE done: pre-fetched VR{X}"
2. **No bare recv logs**: Should never see "Recv only VR{X}" in cooldown (except POST-SYNC RECV)
3. **Chain logs**: Should see "CHAIN: send VR{X} + recv VR{Y}" in cooldown
4. **Edge ranks grad_last**: R0 and R3 should never log "Saved grad for chain" in cooldown

### Common errors:
1. **"Recv only" in cooldown k=0**: Bridge failed or wrong VR
2. **Hanging**: Unmatched send/recv (check NCCL batch matching)
3. **Wrong gradients**: VR mismatch in bridge or post-sync recv

---

## Files to Modify

1. **`megatron/core/pipeline_parallel/schedule_impl/chimera/chimera_2vr.py`**
   - Line 1222: Fix bridge index
   - Lines ~1316-1317: Add helper functions
   - Lines 1314-1469: Rewrite cooldown loop

2. **No P2P changes needed**: All required functions already exist in `p2p_communication.py`

---

## Next Steps

1. Review this breakdown
2. Implement changes incrementally (bridge fix first, then cooldown)
3. Test each phase before moving to next
4. Use `git diff chimera_2vr_old.py chimera_2vr.py` to review changes
