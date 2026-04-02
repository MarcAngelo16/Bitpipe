# Chimera 2-VR Deadlock Analysis & Communication Redesign Plan

## Last Updated: February 7, 2026 (Session 2)

---

## Overview

Two sessions analyzed deadlocks in the Chimera 2-VR scheduler's cooldown phase. The first session identified a deadlock at the **1F1B→cooldown transition** and implemented a bridge fix. The second session found the bridge only partially worked — a **new deadlock in cooldown k=0** was discovered. Root cause analysis revealed the bridge has a code bug AND a fundamental timing problem. The conclusion: Chimera needs a **full communication redesign** matching BitPipe's pattern of always-combined send+recv operations.

**Current status:** Bridge fix partially implemented but **NOT sufficient**. Full redesign needed.

---

## Setup

- 4 GPUs, 4 microbatches, 24 layers
- `CUDA_DEVICE_MAX_CONNECTIONS=1`
- `CHIMERA_DEBUG=1`

### Schedules (4 devices, 4 microbatches)

| Rank | Forward | Backward | warmup | 1f1b | cooldown items |
|------|---------|----------|--------|------|----------------|
| 0 | [0, 1, 2, 3] | [2, 3, 0, -1, 1, -1] | 2 | 2 | 4 |
| 1 | [0, 2, 1, 3] | [2, 0, 3, 1, -1, -1] | 3 | 1 | 5 |
| 2 | [2, 0, 3, 1] | [0, 2, 1, 3, -1, -1] | 3 | 1 | 5 |
| 3 | [2, 3, 0, 1] | [0, 1, 2, -1, 3, -1] | 2 | 2 | 4 |

**VR assignments:** MB0,MB1 → VR0 | MB2,MB3 → VR1

**Gradient flow directions:**
- VR0: forward 0→1→2→3, gradients 3→2→1→0 (send_prev, recv_next)
- VR1: forward 3→2→1→0, gradients 0→1→2→3 (send_next, recv_prev)

**Where gradients originate (loss computation):**
- VR0: loss at Rank 3 (last fwd stage). Gradient first stage = R3, last stage = R0
- VR1: loss at Rank 0 (last fwd stage). Gradient first stage = R0, last stage = R3

**Key asymmetry:** Ranks 0,3 have **2** 1F1B iterations. Ranks 1,2 have **1**.

### Cooldown Schedule Per Rank (backward items after 1F1B)

Derived from backward schedule, starting at `bwd_idx` after 1F1B completes:

| Rank | bwd_idx after 1F1B | Cooldown items (in order) |
|------|-------------------|---------------------------|
| 0 | 2 | MB0(VR0), -1(sync), MB1(VR0), -1(sync) |
| 1 | 1 | MB0(VR0), MB3(VR1), MB1(VR0), -1(sync), -1(sync) |
| 2 | 1 | MB2(VR1), MB1(VR0), MB3(VR1), -1(sync), -1(sync) |
| 3 | 2 | MB2(VR1), -1(sync), MB3(VR1), -1(sync) |

**Key observation:** At cooldown k=0:
- R0 processes VR0, R1 processes VR0, R2 processes VR1, R3 processes VR1
- R0 and R1 both need VR0 grads (flowing 3→2→1→0, recv_next)
- R2 and R3 both need VR1 grads (flowing 0→1→2→3, recv_prev)

### Schedule Generation Code

- Forward schedule: `get_chimera_microbatch_idx()` at `chimera_2vr.py:397-450`
- Backward schedule: `get_chimera_bkmicrobatch_idx()` at `chimera_2vr.py:453-499`
- MB→VR mapping: `get_model_chunk_id()` at `chimera_2vr.py:502-522`
- Warmup count: `get_num_warmup_microbatches()` at `chimera_2vr.py:525-561`

---

## Deadlock #1: 1F1B→Cooldown Transition (Session 1)

### What Happened

At the end of 1F1B, edge ranks (R0, R3) did standalone sends (`chimera_send_next_only` / `chimera_send_prev_only`) to middle ranks (R1, R2) who were already in cooldown doing recv from the opposite direction.

| Rank | Phase | Operation | Waiting for |
|------|-------|-----------|-------------|
| R0 | End 1F1B k=1 | `chimera_send_next_only` → R1 | R1 to post recv from R0 |
| R1 | Cooldown k=0 | `chimera_grad_recv_next_only` ← R2 | R2 to send VR0 grad |
| R2 | Cooldown k=0 | `chimera_grad_recv_prev_only` ← R1 | R1 to send VR1 grad |
| R3 | End 1F1B k=1 | `chimera_send_prev_only` → R2 | R2 to post recv from R3 |

Root cause: standalone blocking sends at `chimera_2vr.py:1211-1218` (the `need_send_bwd and not need_recv_fwd` case).

### Bridge Fix Applied

Replaced standalone send with combined send+recv to pre-fetch first cooldown grad. See "Bridge Implementation" section below for code details.

**Result:** Bridge partially fixed the deadlock — middle ranks no longer freeze at the transition. But a NEW deadlock appeared in cooldown k=0.

---

## Deadlock #2: Cooldown k=0 (Session 2) ← CURRENT PROBLEM

### Logs Showing the Deadlock

After the bridge fix, training progresses further but freezes at cooldown k=0:

```
[Rank 0] 1F1B k=1 BWD MB3 VR1 → "Send bwd only VR1 (no bridge: first_cooldown_mb=3)" → fallback standalone send
[Rank 0] COOLDOWN k=0: BWD MB0 VR0 → "Recv only VR0 grad for MB0"  ← STUCK

[Rank 1] 1F1B k=0 BWD MB2 VR1 → "BRIDGE: send VR1 bwd grad + recv VR1 cooldown grad (MB2)" → bridge succeeded
[Rank 1] COOLDOWN k=0: BWD MB0 VR0 → "Recv only VR0 grad for MB0"  ← STUCK

[Rank 2] 1F1B k=0 BWD MB0 VR0 → "BRIDGE: send VR0 bwd grad + recv VR0 cooldown grad (MB0)" → bridge succeeded
[Rank 2] COOLDOWN k=0: BWD MB2 VR1 → "Recv only VR1 grad for MB2"  ← STUCK

[Rank 3] 1F1B k=1 BWD MB1 VR0 → "Send bwd only VR0 (no bridge: first_cooldown_mb=1)" → fallback standalone send
[Rank 3] COOLDOWN k=0: BWD MB2 VR1 → "Recv only VR1 grad for MB2"  ← STUCK
```

### The Deadlock: All 4 Ranks Post Standalone Recv

At cooldown k=0, every rank hits the `not have_pending_send and need_recv` path at `chimera_2vr.py:1426-1435`:

| Rank | Cooldown k=0 | recv direction | Waiting for |
|------|-------------|----------------|-------------|
| R0 | BWD MB0 VR0 | recv_next (from R1) | R1 to send VR0 grad |
| R1 | BWD MB0 VR0 | recv_next (from R2) | R2 to send VR0 grad |
| R2 | BWD MB2 VR1 | recv_prev (from R1) | R1 to send VR1 grad |
| R3 | BWD MB2 VR1 | recv_prev (from R2) | R2 to send VR1 grad |

```
DEADLOCK:
R0 ←(recv)← R1    R1 has nothing to send (it's trying to recv from R2)
R1 ←(recv)← R2    R2 has nothing to send (it's trying to recv from R1)
R2 ←(recv)← R1    CIRCULAR: R1 and R2 both recv from each other
R3 ←(recv)← R2    R2 has nothing to send
```

Central deadlock: **R1 ↔ R2 mutual recv**. Both wait for the other. R0 and R3 are blocked downstream.

### Why It Happens: No Pending Send at Cooldown Start

The cooldown communication design (`chimera_2vr.py:1394-1435`) uses a "deferred send" pattern:
1. Compute backward → produces gradient
2. Save gradient as `pending_grad_to_send`
3. On NEXT cooldown step, combine pending send + current recv

**The problem:** At cooldown k=0, step 3 can't happen because step 1 and 2 haven't executed yet. There's no `pending_grad_to_send` from a previous cooldown iteration. The code falls into the bare `need_recv` path (line 1426) which does a standalone recv. All 4 ranks do this simultaneously → deadlock.

---

## Root Cause Analysis: Two Bugs in the Bridge

### Bug 1: Wrong Index (`bwd_idx` not incremented)

**Location:** `chimera_2vr.py:1222`

```python
first_cooldown_mb = microbatch_idx_b[bwd_idx]  # BUG: reads CURRENT backward, not first cooldown
```

At this point in the code, `bwd_idx` hasn't been incremented yet (increment happens at line 1299). So `microbatch_idx_b[bwd_idx]` returns the **current** backward MB (the one being processed), NOT the first cooldown MB.

**Should be:**
```python
first_cooldown_mb = microbatch_idx_b[bwd_idx + 1]  # First cooldown item
```

**Impact on each rank:**

| Rank | bwd_idx at bridge | Reads `bwd[bwd_idx]` | Actual first cooldown `bwd[bwd_idx+1]` | Effect |
|------|------------------|----------------------|----------------------------------------|--------|
| R0 | 1 | MB3 (VR1) | MB0 (VR0) | VR1 grad_first=R0 → `can_bridge=False` → standalone send |
| R1 | 0 | MB2 (VR1) | MB0 (VR0) | Bridges with VR1, but cooldown k=0 needs VR0 → wrong VR pre-fetched |
| R2 | 0 | MB0 (VR0) | MB2 (VR1) | Bridges with VR0, but cooldown k=0 needs VR1 → wrong VR pre-fetched |
| R3 | 1 | MB1 (VR0) | MB2 (VR1) | VR0 grad_first=R3 → `can_bridge=False` → standalone send |

**Edge ranks** (R0, R3): Bridge fails completely because it checks `is_vr_first_stage_for_gradient` on the CURRENT MB (which happens to be their own grad-first VR), so `can_bridge=False`.

**Middle ranks** (R1, R2): Bridge succeeds but pre-fetches the WRONG VR's gradient:
- R1 pre-fetches VR1 grad (`bridged_cooldown_vr=1`), but cooldown k=0 is VR0 (`model_chunk_id=0`) → `have_bridged_grad = (1 == 0) = False`
- R2 pre-fetches VR0 grad (`bridged_cooldown_vr=0`), but cooldown k=0 is VR1 (`model_chunk_id=1`) → `have_bridged_grad = (0 == 1) = False`

The `have_bridged_grad` check at `chimera_2vr.py:1365`:
```python
have_bridged_grad = (bridged_cooldown_grad is not None and bridged_cooldown_vr == model_chunk_id)
```
Returns False because the VR doesn't match. So cooldown k=0 falls through to standalone recv anyway.

### Bug 2: Timing Asymmetry (Fundamental Design Issue)

Even with Bug 1 fixed (`bwd_idx+1`), the bridge STILL can't prevent the cooldown k=0 deadlock.

**Why:** Middle and edge ranks bridge at **different timesteps**. At the timestep when middle ranks bridge, edge ranks are doing their 1F1B k=0 communication. At the timestep when edge ranks bridge, middle ranks are already in cooldown. The communication partners don't align.

**Detailed timing analysis with FIXED bridge (`bwd_idx+1`):**

**T_middle: Middle ranks bridge (end of their single 1F1B iteration)**

What each rank does at this timestep:

| Rank | Phase | What it does | NCCL batch |
|------|-------|-------------|------------|
| R0 | 1F1B k=0 BWD comm | send VR1 grad + recv VR1 fwd input | [isend→R1, irecv←R1] |
| R1 | Bridge (fixed) | send VR1 grad + recv VR0 cooldown grad | [isend→R2, irecv←R2] |
| R2 | Bridge (fixed) | send VR0 grad + recv VR1 cooldown grad | [isend→R1, irecv←R1] |
| R3 | 1F1B k=0 BWD comm | send VR0 grad + recv VR0 fwd input | [isend→R2, irecv←R2] |

How bridges are computed (with fix):
- R1: `bwd_model_chunk_id=1(VR1)`, `first_cool_vr=get_model_chunk_id(microbatch_idx_b[1])=get_model_chunk_id(MB0)=VR0` → VR1 send_next + VR0 recv_next → `chimera_grad_send_next_recv_next` → [isend→R2, irecv←R2]
- R2: `bwd_model_chunk_id=0(VR0)`, `first_cool_vr=get_model_chunk_id(microbatch_idx_b[1])=get_model_chunk_id(MB2)=VR1` → VR0 send_prev + VR1 recv_prev → `chimera_grad_send_prev_recv_prev` → [isend→R1, irecv←R1]

**Matching analysis at T_middle:**

```
R0 isend→R1  ↔  R1 irecv←R2  ✗ MISMATCH (R1 recvs from R2, not R0)
R0 irecv←R1  ↔  R1 isend→R2  ✗ MISMATCH (R1 sends to R2, not R0)
R1 isend→R2  ↔  R2 irecv←R1  ✓ MATCH
R2 isend→R1  ↔  R1 irecv←R2  ✓ MATCH
R3 isend→R2  ↔  R2 irecv←R1  ✗ MISMATCH (R2 recvs from R1, not R3)
R3 irecv←R2  ↔  R2 isend→R1  ✗ MISMATCH (R2 sends to R1, not R3)
```

**R1↔R2 exchange works. R0↔R1 and R3↔R2 are completely unmatched.** R0 and R3's groups block (both their isend and irecv have no match), which blocks their entire 1F1B k=0 communication.

This means R0 and R3 can't even REACH their bridge (at 1F1B k=1) because they're stuck at k=0.

**Conclusion:** The bridge approach is fundamentally incompatible with Chimera's timing asymmetry. Middle ranks' bridges leave edge ranks stranded because the middle ranks switch their communication partner (from edge rank to each other) at the bridge timestep.

---

## How BitPipe Handles Cooldown (Reference for Redesign)

### The Chain Pattern

BitPipe's cooldown at `bitpipe_4vr.py:889-992` uses a **chain** where every step has its input pre-received:

```
Bridge (line 862, last 1F1B backward):
  send_backward_recv_backward_bd(input_tensor_grad, ...)
  → Sends 1F1B result, receives first cooldown grad
  → Stores in output_tensor_grads[next_backward_model_chunk_id]

Cooldown loop (line 894):
  for each backward k:
    if sync_marker:
      allreduce_gradients(...)            # Synchronizes all ranks
      recv_backward(...)                  # Safe standalone recv (post-sync)
    else:
      input_tensor_grad = backward_step_helper(...)  # Uses pre-stored grad
      send_backward_recv_backward(input_tensor_grad, ...)  # Send result + recv next
      → Stores received grad for next iteration
```

### Key Communication Functions Used in Cooldown

| Function | Code location | Send dir | Recv dir | When used |
|----------|--------------|----------|----------|-----------|
| `send_backward_recv_backward` | `p2p_communication.py:733-759` | send_prev | recv_next | Same-VR cooldown steps |
| `send_backward_recv_backward_bd` | `p2p_communication.py:762-785` | send_prev | recv_prev | Cross-VR cooldown steps (direction switch) |
| `recv_backward` (standalone) | `p2p_communication.py:598-618` | none | recv_next | After sync marker (allreduce syncs all ranks first) |

### Why It Works: No Bare Recv at Cooldown Start

1. **Last 1F1B backward** (`bitpipe_4vr.py:862`): uses `send_backward_recv_backward_bd` to simultaneously send the 1F1B grad AND recv the first cooldown grad. When cooldown starts, the grad is already in `output_tensor_grads`.

2. **Each cooldown step**: pops pre-received grad → computes backward → immediately sends result + recvs next input via combined send_backward_recv_backward. Never a bare recv (except after sync markers which synchronize all ranks via allreduce first).

3. **Special cases in cooldown** (`bitpipe_4vr.py:925-992`):
   - `backward_k == total-2`: `send_backward_recv_backward_bd` (cross-direction)
   - `backward_k == total-1`: `send_backward_recv_backward(recv_next=False)` (send only, last step)
   - `backward_k == total`: `send_backward_recv_backward(recv_next=False)` (send only)
   - Same VR: `send_backward_recv_backward` (standard chain)
   - Different VR, same device: direct tensor copy (line 981: `output_tensor_grads[next].append(input_tensor_grad)`)
   - Different VR, cross device: `send_backward_recv_backward_bd` (line 983-991)

### The Critical Design Principle

**BitPipe NEVER starts any phase with a bare recv.** The chain is:

```
pre-fetch (from previous phase) → compute → send+recv → compute → send+recv → ... → send-only (end)
```

Each step's recv feeds the next step's compute. The chain never breaks because every recv is combined with a send from the current step's backward result.

**Standalone recvs** only appear after sync markers (`allreduce_gradients`), which is safe because allreduce is a collective that synchronizes all ranks — after allreduce completes, all ranks know the matching send is coming.

---

## Chimera's Fundamental Problem

### Why the Deferred-Send Pattern Fails

Chimera's current cooldown (`chimera_2vr.py:1303-1468`) uses a "deferred send" pattern:

```python
# chimera_2vr.py:1426-1435 (the problematic path)
elif not have_pending_send and need_recv:
    # Recv only (no pending send from previous iteration)
    if model_chunk_id == 0:
        output_tensor_grad = chimera_grad_recv_next_only(tensor_shape, config)
    else:
        output_tensor_grad = chimera_grad_recv_prev_only(tensor_shape, config)
```

At cooldown k=0:
- `have_pending_send = False` (no previous cooldown iteration)
- `have_bridged_grad = False` (bridge bug: wrong VR pre-fetched)
- `need_recv = True` (not grad-first stage)
- Falls into bare recv path → deadlock

### The 14 Standalone Send/Recv Sites

Current standalone communication sites in `chimera_2vr.py`:

| # | Location (line ~) | Phase | Type | Risk |
|---|-------------------|-------|------|------|
| 1 | 1014-1016 | Warmup | send_only | Low — pipeline filling |
| 2 | 1134-1136 | 1F1B fwd | send_only | Low — loss stage |
| 3 | 1277-1279 | 1F1B bridge fallback | send_only | **HIGH** — causes deadlock #1 |
| 4 | 1284-1290 | 1F1B recv fwd | recv_only | Medium |
| 5 | 1327-1329 | Cooldown sync flush | send_only | Medium |
| 6 | 1388-1390 | Cooldown bridged flush | send_only | Medium |
| 7 | 1419-1421 | Cooldown pending send | send_only | Low — grad-first |
| 8 | 1426-1435 | **Cooldown bare recv** | **recv_only** | **HIGH** — causes deadlock #2 |
| 9 | 1466-1468 | Post-cooldown flush | send_only | Low — end of schedule |

---

## Plan for Next Session: Full Communication Redesign

### Goal

Redesign Chimera's communication to match BitPipe's "always-combined" pattern. **Every communication should be a combined send+recv.** Standalone sends/recvs should only exist where provably safe (e.g., after a collective sync, or at pipeline edges where one direction doesn't exist).

### Design Principles (from BitPipe)

1. **Chain pattern**: pre-fetch first grad → compute → send+recv → compute → send+recv → ...
2. **Never a bare recv** at the start of any phase
3. **Every send is paired with a recv** in the same NCCL batch
4. **When one direction isn't needed**: pass `recv_*=False` to the combined function (not a separate standalone)

### Key Challenge: Chimera's Bidirectional Cooldown

In BitPipe, cooldown backward items alternate between VRs that flow in the SAME direction (because of 4 VR pairs). Consecutive items communicate with the same neighbors.

In Chimera, cooldown alternates between VR0 (sends prev, recvs next) and VR1 (sends next, recvs prev). Consecutive items may communicate with DIFFERENT neighbors. This makes the chain pattern harder to implement.

**Example for R1 cooldown:**
```
k=0: BWD MB0 (VR0) — recv VR0 grad from R2 (recv_next), send VR0 result to R0 (send_prev)
k=1: BWD MB3 (VR1) — recv VR1 grad from R0 (recv_prev), send VR1 result to R2 (send_next)
k=2: BWD MB1 (VR0) — recv VR0 grad from R2 (recv_next), send VR0 result to R0 (send_prev)
```

For the chain to work: k=0's send (send_prev→R0) needs to be combined with k=1's recv (recv_prev←R0). Both go to/from R0! This is a `send_prev + recv_prev` combination. Similarly, k=1's send (send_next→R2) + k=2's recv (recv_next←R2) → both go to/from R2. This is `send_next + recv_next`.

**So the chain DOES work for middle ranks:** consecutive cooldown items alternate directions, and send+recv for adjacent items go to/from the SAME neighbor.

For edge ranks (R0, R3), cooldown items are all the same VR, so consecutive send+recv is straightforward.

### Proposed Approach

1. **Fix the bridge** (use `bwd_idx + 1` for first cooldown item)
2. **Restructure 1F1B→cooldown transition** so that ALL ranks (not just middle) enter cooldown with their first grad pre-fetched. This may require restructuring the timing so middle ranks' bridges don't break edge ranks' 1F1B communication.
3. **Rewrite cooldown loop** to always use combined send+recv:
   ```python
   # Proposed cooldown structure:
   # Step 0: use pre-fetched grad (from bridge), compute backward, combine send+recv for next
   # Step k: use recv from step k-1, compute backward, combine send+recv for next
   # Last step: use recv from previous, compute backward, send-only (or skip if grad-last)
   ```
4. **Handle sync markers** carefully — allreduce synchronizes all ranks, so a bare recv after sync is safe (BitPipe does this too)
5. **Eliminate all unnecessary standalone operations**

### Specific Code Locations to Modify

| File | Lines | What to change |
|------|-------|---------------|
| `chimera_2vr.py:1222` | Bridge index bug | Change `microbatch_idx_b[bwd_idx]` → `microbatch_idx_b[bwd_idx + 1]` |
| `chimera_2vr.py:1217-1282` | Bridge logic | Redesign to handle timing asymmetry (see challenge above) |
| `chimera_2vr.py:1303-1468` | Entire cooldown loop | Rewrite with chain pattern (no bare recvs) |
| `chimera_2vr.py:1394-1435` | Cooldown recv paths | Replace bare recv with combined send+recv |
| `chimera_2vr.py:1014-1016` | Warmup standalone send | Consider combining with recv |
| `chimera_2vr.py:1134-1136` | 1F1B fwd standalone send | Consider combining with recv |

### Open Questions for Next Session

1. **Timing asymmetry resolution**: How to make edge ranks' 1F1B k=0 communication compatible with middle ranks' bridge? Options:
   - Restructure the schedule so all ranks bridge at the same timestep
   - Add intermediate combined operations that bridge the gap
   - Accept the asymmetry and design cooldown k=0 to handle it (e.g., middle ranks don't do bare recv because they have bridged grad; edge ranks have their grad delivered by middle ranks' first cooldown send)

2. **Warmup standalone sends**: Are the warmup standalone sends (line 1014) safe? They work now, but should be reviewed for consistency.

3. **Testing strategy**: After redesign, need to verify with multiple configurations (4, 6, 8 GPUs; varying microbatch counts).

---

## Appendix A: Bridge Implementation (Current Code, Partially Working)

### Change 1: Bridge state variables (`chimera_2vr.py` ~line 875)

```python
bridged_cooldown_grad = None
bridged_cooldown_vr = None
```

### Change 2: Bridge logic at end of 1F1B (`chimera_2vr.py:1217-1282`)

Replaces the original standalone send (`need_send_bwd and not need_recv_fwd` case):

```python
elif need_send_bwd and not need_recv_fwd:
    # BRIDGE: combine last 1F1B send + first cooldown recv
    first_cooldown_mb = microbatch_idx_b[bwd_idx]  # BUG: should be bwd_idx + 1
    can_bridge = (first_cooldown_mb != -1)

    if can_bridge:
        first_cool_vr = get_model_chunk_id(first_cooldown_mb, pipeline_parallel_size)
        cool_is_grad_first = is_vr_first_stage_for_gradient(first_cool_vr, ...)
        can_bridge = not cool_is_grad_first

    if can_bridge:
        # Combined send+recv based on VR directions
        # bwd_model_chunk_id determines send direction
        # first_cool_vr determines recv direction
        # 4 combinations: send_prev/next + recv_prev/next
        bridged_grad = chimera_grad_send_*_recv_*(input_tensor_grad, tensor_shape, config)
        bridged_cooldown_grad = bridged_grad
        bridged_cooldown_vr = first_cool_vr
    else:
        # Fallback: standalone send (still dangerous)
        chimera_send_*_only(input_tensor_grad, config)
```

### Change 3: Cooldown uses pre-fetched gradient (`chimera_2vr.py:1364-1392`)

```python
have_bridged_grad = (bridged_cooldown_grad is not None and bridged_cooldown_vr == model_chunk_id)

if have_bridged_grad:
    output_tensor_grad = bridged_cooldown_grad
    bridged_cooldown_grad = None
    bridged_cooldown_vr = None
    # Flush any pending send alongside (standalone — risky)
    if have_pending_send:
        chimera_grad_send_*_only(pending_grad_to_send, config)
```

---

## Appendix B: First Deadlock Analysis (Session 1, for reference)

### Original Deadlock Chain

```
R0 ──send_next──→ R1    (R1 is recv from R2, NOT from R0)
R3 ──send_prev──→ R2    (R2 is recv from R1, NOT from R3)
R1 ──recv_next──← R2    (mutual)
R2 ──recv_prev──← R1    (mutual)
```

### Root Cause

Standalone blocking sends at `chimera_2vr.py:1211-1218` (original code before bridge):

```python
elif need_send_bwd and not need_recv_fwd:
    if bwd_model_chunk_id == 0:
        p2p_communication.chimera_send_prev_only(input_tensor_grad, config)
    else:
        p2p_communication.chimera_send_next_only(input_tensor_grad, config)
```

Under `CUDA_DEVICE_MAX_CONNECTIONS=1`, the `isend` with no matching `irecv` blocks the entire CUDA stream.

---

## Appendix C: Key P2P Functions Reference

### Chimera Combined Functions (`p2p_communication.py`)

| Function | Line | Send dir | Recv dir | NCCL batch |
|----------|------|----------|----------|------------|
| `chimera_send_prev_recv_prev` | ~1200 | send_prev | recv_prev | [isend→prev, irecv←prev] |
| `chimera_send_prev_recv_next` | ~1230 | send_prev | recv_next | [isend→prev, irecv←next] |
| `chimera_send_next_recv_prev` | ~1240 | send_next | recv_prev | [isend→next, irecv←prev] |
| `chimera_send_next_recv_next` | ~1250 | send_next | recv_next | [isend→next, irecv←next] |
| `chimera_grad_send_prev_recv_next` | 1270 | send_prev | recv_next | [isend→prev, irecv←next] |
| `chimera_grad_send_next_recv_prev` | 1304 | send_next | recv_prev | [isend→next, irecv←prev] |
| `chimera_grad_send_prev_recv_prev` | 1338 | send_prev | recv_prev | [isend→prev, irecv←prev] |
| `chimera_grad_send_next_recv_next` | 1372 | send_next | recv_next | [isend→next, irecv←next] |

### Chimera Standalone Functions (`p2p_communication.py`)

| Function | Line | Direction |
|----------|------|-----------|
| `chimera_send_next_only` | 1145 | isend→next |
| `chimera_send_prev_only` | 1173 | isend→prev |
| `chimera_recv_next_only` | 1201 | irecv←next |
| `chimera_recv_prev_only` | 1229 | irecv←prev |
| `chimera_grad_send_prev_only` | 1406 | isend→prev |
| `chimera_grad_send_next_only` | 1434 | isend→next |
| `chimera_grad_recv_next_only` | 1462 | irecv←next |
| `chimera_grad_recv_prev_only` | 1490 | irecv←prev |

### BitPipe Cooldown Functions (`p2p_communication.py`)

| Function | Line | Send dir | Recv dir |
|----------|------|----------|----------|
| `send_backward_recv_backward` | 733 | send_prev | recv_next |
| `send_backward_recv_backward_bd` | 762 | send_prev | recv_prev |
| `recv_backward` (standalone) | 598 | — | recv_next |

### Helper Functions in Chimera Scheduler (`chimera_2vr.py`)

| Function | Line | Purpose |
|----------|------|---------|
| `is_vr_first_stage_for_activation` | 168 | VR0: R0, VR1: R3 |
| `is_vr_last_stage_for_activation` | 182 | VR0: R3, VR1: R0 |
| `is_vr_first_stage_for_gradient` | 196 | VR0: R3, VR1: R0 (where loss is) |
| `is_vr_last_stage_for_gradient` | 223 | VR0: R0, VR1: R3 (where grads end) |

---

## Appendix D: Lessons Learned

1. **Every `isend` needs a matching `irecv`.** Under `CUDA_DEVICE_MAX_CONNECTIONS=1`, an unmatched send/recv blocks the entire GPU's NCCL stream.

2. **Combined operations create mutual exchanges.** A batch of `[isend, irecv]` works because the partner also has `[isend, irecv]`. Neither side blocks.

3. **Phase asymmetry creates timing gaps.** Edge ranks (0, N-1) have more 1F1B iterations than middle ranks (N/2-1, N/2). This means they arrive at phase transitions at different times.

4. **The bridge approach has a timing problem.** When middle ranks bridge with each other, they change their communication partner, leaving edge ranks stranded mid-1F1B.

5. **BitPipe's chain pattern is the right model.** Pre-fetch first grad → compute → send+recv → compute → send+recv. Never start with a bare recv.

6. **Standalone operations should be eliminated** wherever possible. Use combined functions with `recv_*=False` instead.

7. **The `bwd_idx` vs `bwd_idx+1` bug** is easy to fix but doesn't solve the fundamental timing issue.
