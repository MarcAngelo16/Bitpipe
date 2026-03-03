# Simplified Chimera Implementation with Flag-Based P2P

## Goal

Replace all the `if/elif` communication branches with **unified flag-based functions**, just like BitPipe does with `send_forward=True/False`, `recv_forward=True/False`.

---

## Current Problem: Too Many Branches

**Current warmup/1F1B/cooldown code:**
```python
# Current approach (complicated!)
if bwd_model_chunk_id == 0 and next_fwd_vr == 0:
    # VR0 bwd send_prev + VR0 fwd recv_prev
    next_input = p2p_communication.chimera_send_prev_recv_prev(...)
elif bwd_model_chunk_id == 0 and next_fwd_vr == 1:
    # VR0 bwd send_prev + VR1 fwd recv_next
    next_input = p2p_communication.chimera_send_prev_recv_next(...)
elif bwd_model_chunk_id == 1 and next_fwd_vr == 0:
    # VR1 bwd send_next + VR0 fwd recv_prev
    next_input = p2p_communication.chimera_send_next_recv_prev(...)
else:  # bwd_model_chunk_id == 1 and next_fwd_vr == 1
    # VR1 bwd send_next + VR1 fwd recv_next
    next_input = p2p_communication.chimera_send_next_recv_next(...)
```

**16+ lines** just to handle 4 combinations!

---

## Solution: Unified Flag-Based Functions

### New Approach (clean!)

```python
# Determine what to send/recv based on VR
send_to_prev = (bwd_model_chunk_id == 0)  # VR0 sends to prev
send_to_next = (bwd_model_chunk_id == 1)  # VR1 sends to next
recv_from_prev = (next_fwd_vr == 1)       # VR1 recvs from prev
recv_from_next = (next_fwd_vr == 0)       # VR0 recvs from next

# Single unified call!
next_input = chimera_communicate(
    tensor_send_prev=input_tensor_grad if send_to_prev else None,
    tensor_send_next=input_tensor_grad if send_to_next else None,
    recv_prev=recv_from_prev,
    recv_next=recv_from_next,
    tensor_shape=tensor_shape,
    config=config
)
```

**6 lines** instead of 16! Much cleaner!

---

## Implementation Plan

### Step 1: Add Unified Chimera P2P Function

**Location:** `megatron/core/pipeline_parallel/p2p_communication.py` (after existing Chimera functions, around line 1520)

```python
def chimera_communicate(
    tensor_send_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Unified Chimera communication with flags (like BitPipe's approach).

    Handles all combinations of send/recv in a single function.
    Uses _communicate() directly to bypass stage checks.

    Args:
        tensor_send_prev: Tensor to send to previous rank (None = no send)
        tensor_send_next: Tensor to send to next rank (None = no send)
        recv_prev: Whether to receive from previous rank
        recv_next: Whether to receive from next rank
        tensor_shape: Shape for receiving tensors
        config: Model config

    Returns:
        (recv_prev_tensor, recv_next_tensor)
        - recv_prev_tensor: Tensor received from prev (None if recv_prev=False)
        - recv_next_tensor: Tensor received from next (None if recv_next=False)

    Examples:
        # VR0 grad: send_prev + recv_next
        grad_in, _ = chimera_communicate(
            tensor_send_prev=grad_out, tensor_send_next=None,
            recv_prev=False, recv_next=True, ...
        )

        # VR1 grad: send_next + recv_prev
        grad_in, _ = chimera_communicate(
            tensor_send_prev=None, tensor_send_next=grad_out,
            recv_prev=True, recv_next=False, ...
        )

        # VR0 grad send only (grad_last): send_prev, no recv
        _, _ = chimera_communicate(
            tensor_send_prev=grad_out, tensor_send_next=None,
            recv_prev=False, recv_next=False, ...
        )

        # VR0 grad recv only (post-sync): no send, recv_next
        grad_in, _ = chimera_communicate(
            tensor_send_prev=None, tensor_send_next=None,
            recv_prev=False, recv_next=True, ...
        )
    """
    if config.timers is not None:
        config.timers('chimera-communicate', log_level=2).start()

    # Use _communicate directly (bypasses stage checks)
    recv_prev_tensor, recv_next_tensor, _ = _communicate(
        tensor_send_next=tensor_send_next,
        tensor_send_prev=tensor_send_prev,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        wait_on_reqs=True,
        config=config
    )

    if config.timers is not None:
        config.timers('chimera-communicate').stop()

    return recv_prev_tensor, recv_next_tensor
```

---

### Step 2: Simplified Warmup Phase

**Current code (lines ~975-1040):** Many `if/elif` branches

**New code:**
```python
# Warmup phase
for k in range(num_warmup_microbatches):
    microbatch_id = microbatch_idx[fwd_idx]
    model_chunk_id = get_model_chunk_id(microbatch_id, pipeline_parallel_size)
    parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

    print_all_ranks(f"[WARMUP k={k}] FWD MB{microbatch_id}, VR{model_chunk_id}")

    # Pop input if not first stage
    is_first = is_vr_first_stage_for_activation(model_chunk_id, ...)
    is_last = is_vr_last_stage_for_activation(model_chunk_id, ...)

    if not is_first:
        input_tensor = input_tensors[model_chunk_id].pop(0)
    else:
        input_tensor = None

    # Forward step
    output_tensor = forward_step(input_tensor, ...)

    # Determine send/recv based on current and next VR
    fwd_idx += 1
    if fwd_idx < len(microbatch_idx):
        next_microbatch_id = microbatch_idx[fwd_idx]
        next_model_chunk_id = get_model_chunk_id(next_microbatch_id, ...)
        next_is_first = is_vr_first_stage_for_activation(next_model_chunk_id, ...)
    else:
        next_is_first = False  # No more forwards

    # Determine what to send/recv
    send_to_prev = not is_last and model_chunk_id == 1  # VR1 sends to prev
    send_to_next = not is_last and model_chunk_id == 0  # VR0 sends to next
    recv_from_prev = not next_is_first and next_model_chunk_id == 1  # VR1 recvs from prev
    recv_from_next = not next_is_first and next_model_chunk_id == 0  # VR0 recvs from next

    # Unified communication!
    recv_prev_tensor, recv_next_tensor = chimera_communicate(
        tensor_send_prev=output_tensor if send_to_prev else None,
        tensor_send_next=output_tensor if send_to_next else None,
        recv_prev=recv_from_prev,
        recv_next=recv_from_next,
        tensor_shape=tensor_shape,
        config=config
    )

    # Store received input for next forward
    if recv_prev_tensor is not None:
        input_tensors[next_model_chunk_id].append(recv_prev_tensor)
    if recv_next_tensor is not None:
        input_tensors[next_model_chunk_id].append(recv_next_tensor)

    print_all_ranks(f"[WARMUP k={k}] FWD done, sent={send_to_prev or send_to_next}, "
                    f"recv_p={recv_from_prev}, recv_n={recv_from_next}")
```

---

### Step 3: Simplified 1F1B Phase

**Current code (lines ~1040-1301):** Separate branches for FWD comm and BWD comm

**New code pattern:**
```python
# 1F1B phase
while ...:
    # === FORWARD ===
    # ... forward compute ...

    # Determine FWD send/recv
    is_fwd_last = is_vr_last_stage_for_activation(fwd_model_chunk_id, ...)
    need_recv_grad = not is_fwd_last  # Recv grad if not last stage

    # VR0 fwd sends to next, VR1 fwd sends to prev
    send_fwd_to_prev = not is_fwd_last and fwd_model_chunk_id == 1
    send_fwd_to_next = not is_fwd_last and fwd_model_chunk_id == 0

    # Receive gradient for upcoming backward
    is_bwd_grad_first = is_vr_first_stage_for_gradient(bwd_model_chunk_id, ...)
    recv_grad_from_prev = need_recv_grad and bwd_model_chunk_id == 1  # VR1 grads from prev
    recv_grad_from_next = need_recv_grad and bwd_model_chunk_id == 0  # VR0 grads from next

    # Combined FWD send + BWD grad recv
    recv_prev_grad, recv_next_grad = chimera_communicate(
        tensor_send_prev=output_tensor if send_fwd_to_prev else None,
        tensor_send_next=output_tensor if send_fwd_to_next else None,
        recv_prev=recv_grad_from_prev,
        recv_next=recv_grad_from_next,
        tensor_shape=tensor_shape,
        config=config
    )

    # Extract the grad we need
    output_tensor_grad = recv_prev_grad if recv_prev_grad is not None else recv_next_grad

    # === BACKWARD ===
    # ... backward compute ...

    # Determine BWD send/recv
    need_send_bwd = not is_bwd_grad_last
    need_recv_fwd = (fwd_idx < len(microbatch_idx))  # More forwards?

    if need_recv_fwd:
        next_fwd_vr = get_model_chunk_id(microbatch_idx[fwd_idx], ...)
        next_is_first = is_vr_first_stage_for_activation(next_fwd_vr, ...)
        need_recv_fwd = not next_is_first

    # Determine directions
    send_bwd_to_prev = need_send_bwd and bwd_model_chunk_id == 0  # VR0 grads to prev
    send_bwd_to_next = need_send_bwd and bwd_model_chunk_id == 1  # VR1 grads to next
    recv_fwd_from_prev = need_recv_fwd and next_fwd_vr == 1  # VR1 activations from prev
    recv_fwd_from_next = need_recv_fwd and next_fwd_vr == 0  # VR0 activations from next

    # Combined BWD send + FWD recv (or bridge!)
    if need_send_bwd and not need_recv_fwd:
        # BRIDGE CASE: send BWD grad + recv first cooldown grad
        first_cooldown_mb = microbatch_idx_b[bwd_idx + 1]  # FIX: use bwd_idx+1
        can_bridge = (first_cooldown_mb != -1)

        if can_bridge:
            first_cool_vr = get_model_chunk_id(first_cooldown_mb, ...)
            cool_is_grad_first = is_vr_first_stage_for_gradient(first_cool_vr, ...)
            can_bridge = not cool_is_grad_first

        if can_bridge:
            # Bridge: send BWD + recv cooldown grad
            recv_cool_from_prev = (first_cool_vr == 1)  # VR1 grads from prev
            recv_cool_from_next = (first_cool_vr == 0)  # VR0 grads from next

            recv_prev_grad, recv_next_grad = chimera_communicate(
                tensor_send_prev=input_tensor_grad if send_bwd_to_prev else None,
                tensor_send_next=input_tensor_grad if send_bwd_to_next else None,
                recv_prev=recv_cool_from_prev,
                recv_next=recv_cool_from_next,
                tensor_shape=tensor_shape,
                config=config
            )

            bridged_cooldown_grad = recv_prev_grad if recv_prev_grad else recv_next_grad
            bridged_cooldown_vr = first_cool_vr
            print_all_ranks(f"BRIDGE success: pre-fetched VR{first_cool_vr}")
        else:
            # Fallback: send only (should be rare with bwd_idx+1 fix)
            chimera_communicate(
                tensor_send_prev=input_tensor_grad if send_bwd_to_prev else None,
                tensor_send_next=input_tensor_grad if send_bwd_to_next else None,
                recv_prev=False,
                recv_next=False,
                tensor_shape=tensor_shape,
                config=config
            )
            print_all_ranks(f"Bridge fallback: send only")
    else:
        # Normal 1F1B: combined BWD send + FWD recv
        recv_prev_fwd, recv_next_fwd = chimera_communicate(
            tensor_send_prev=input_tensor_grad if send_bwd_to_prev else None,
            tensor_send_next=input_tensor_grad if send_bwd_to_next else None,
            recv_prev=recv_fwd_from_prev,
            recv_next=recv_fwd_from_next,
            tensor_shape=tensor_shape,
            config=config
        )

        # Store received fwd activation
        next_input = recv_prev_fwd if recv_prev_fwd else recv_next_fwd
        if next_input is not None:
            input_tensors[next_fwd_vr].append(next_input)
```

---

### Step 4: Simplified Cooldown Phase

**Current code (lines ~1314-1469):** Many nested if/elif branches

**New code:**
```python
# Cooldown phase
pending_grad = None
pending_vr = None
pre_received_grad = None
pre_received_vr = None

while bwd_idx < len(microbatch_idx_b):
    microbatch_id = microbatch_idx_b[bwd_idx]

    # === SYNC MARKER ===
    if microbatch_id == -1:
        # Flush pending grad before sync
        if pending_grad is not None:
            send_to_prev = (pending_vr == 0)
            send_to_next = (pending_vr == 1)

            chimera_communicate(
                tensor_send_prev=pending_grad if send_to_prev else None,
                tensor_send_next=pending_grad if send_to_next else None,
                recv_prev=False,
                recv_next=False,
                tensor_shape=tensor_shape,
                config=config
            )
            pending_grad = None
            pending_vr = None

        # Allreduce
        enable_grad_sync()
        for chunk_id in range(len(model)):
            if chunk_id not in synchronized_model_chunks:
                allreduce_gradients(model[chunk_id])
                synchronized_model_chunks.add(chunk_id)
        disable_grad_sync()

        # Post-sync recv if needed
        if bwd_idx + 1 < len(microbatch_idx_b):
            next_mb = microbatch_idx_b[bwd_idx + 1]
            if next_mb != -1:
                next_vr = get_model_chunk_id(next_mb, ...)
                next_is_grad_first = is_vr_first_stage_for_gradient(next_vr, ...)

                if not next_is_grad_first:
                    recv_from_prev = (next_vr == 1)
                    recv_from_next = (next_vr == 0)

                    recv_prev_grad, recv_next_grad = chimera_communicate(
                        tensor_send_prev=None,
                        tensor_send_next=None,
                        recv_prev=recv_from_prev,
                        recv_next=recv_from_next,
                        tensor_shape=tensor_shape,
                        config=config
                    )

                    pre_received_grad = recv_prev_grad if recv_prev_grad else recv_next_grad
                    pre_received_vr = next_vr

        bwd_idx += 1
        continue

    # === BACKWARD ITEM ===
    model_chunk_id = get_model_chunk_id(microbatch_id, ...)
    parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

    # Pop tensors
    input_tensor = input_tensors[model_chunk_id].pop(0)
    output_tensor = output_tensors[model_chunk_id].pop(0)

    # Determine gradient properties
    is_grad_first = is_vr_first_stage_for_gradient(model_chunk_id, ...)
    is_grad_last = is_vr_last_stage_for_gradient(model_chunk_id, ...)

    # === GET GRADIENT INPUT ===
    have_bridged = (bridged_cooldown_grad is not None and bridged_cooldown_vr == model_chunk_id)
    have_pre_received = (pre_received_grad is not None and pre_received_vr == model_chunk_id)
    have_pending = (pending_grad is not None)
    need_recv = not is_grad_first

    output_tensor_grad = None

    if have_bridged:
        # Use bridged gradient
        output_tensor_grad = bridged_cooldown_grad
        bridged_cooldown_grad = None
        bridged_cooldown_vr = None

    elif have_pre_received:
        # Use post-sync gradient
        output_tensor_grad = pre_received_grad
        pre_received_grad = None
        pre_received_vr = None

    elif have_pending and (need_recv or not is_grad_first):
        # CHAIN: send pending + recv current
        send_to_prev = (pending_vr == 0)
        send_to_next = (pending_vr == 1)
        recv_from_prev = need_recv and (model_chunk_id == 1)
        recv_from_next = need_recv and (model_chunk_id == 0)

        recv_prev_grad, recv_next_grad = chimera_communicate(
            tensor_send_prev=pending_grad if send_to_prev else None,
            tensor_send_next=pending_grad if send_to_next else None,
            recv_prev=recv_from_prev,
            recv_next=recv_from_next,
            tensor_shape=tensor_shape,
            config=config
        )

        output_tensor_grad = recv_prev_grad if recv_prev_grad else recv_next_grad
        pending_grad = None
        pending_vr = None

    elif not have_pending and need_recv:
        # ERROR: bare recv (should never happen!)
        raise RuntimeError(f"DEADLOCK: Bare recv at cooldown, bridge should have pre-fetched!")

    # === COMPUTE BACKWARD ===
    input_tensor_grad = backward_step(
        input_tensor, output_tensor, output_tensor_grad, model_type, config
    )

    # === SAVE FOR NEXT ===
    if not is_grad_last:
        pending_grad = input_tensor_grad
        pending_vr = model_chunk_id

    bwd_idx += 1

# Final flush
if pending_grad is not None:
    send_to_prev = (pending_vr == 0)
    send_to_next = (pending_vr == 1)

    chimera_communicate(
        tensor_send_prev=pending_grad if send_to_prev else None,
        tensor_send_next=pending_grad if send_to_next else None,
        recv_prev=False,
        recv_next=False,
        tensor_shape=tensor_shape,
        config=config
    )
```

---

## Summary of Benefits

### Before (current code):
- **16+ if/elif branches** for communication combinations
- **4 separate functions** per phase (send_prev_recv_next, send_prev_recv_prev, send_next_recv_next, send_next_recv_prev)
- **3 types** of standalone functions (send_only, recv_only for each direction)
- **Difficult to read** and understand flow
- **Error-prone** to modify

### After (flag-based):
- **1 unified function** (`chimera_communicate`)
- **Boolean flags** determine what to send/recv
- **Much cleaner** code (~50% fewer lines)
- **Easier to debug** (all communication goes through same function)
- **Consistent pattern** matching BitPipe's style

---

## Implementation Order

1. ✅ **Add `chimera_communicate()` to `p2p_communication.py`** (~30 lines)
2. ✅ **Rewrite warmup phase** with flags (~50 lines, replace ~100 lines)
3. ✅ **Rewrite 1F1B phase** with flags (~80 lines, replace ~150 lines)
4. ✅ **Rewrite cooldown phase** with flags (~100 lines, replace ~160 lines)
5. ✅ **Test with 4 GPUs** (verify no deadlocks, correct gradients)
6. ✅ **Test with 6, 8 GPUs** (verify scalability)

---

## Next Steps

Ready to implement? We'll start with:
1. Add the unified `chimera_communicate()` function
2. Test it with a simple warmup case
3. Then progressively replace each phase

Let me know when you're ready to start coding!
