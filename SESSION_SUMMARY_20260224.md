# Session Summary – 2026-02-24

## What We Worked On

This session continued from the previous deadlock-fix session. Two main things were accomplished:
1. Replaced the DEADLOCK GUARD with a proper standalone recv for grad-last ranks
2. Diagnosed and attempted to fix an NCCL "Duplicate GPU" error in the BD groups

---

## Fix 1: Standalone Recv for Grad-Last Ranks (WORKS ✅)

### Problem

In the cooldown phase, the scheduler detected a "bare recv" situation for ranks R0 (for VR0) and R3 (for VR1). The previous session had added a `RuntimeError` guard here to detect deadlocks. But it was firing legitimately for the grad-last edge ranks.

**Why these ranks are special:** R0 is the start of the VR0 gradient chain (the gradient originates there — no incoming grad from any higher stage). R3 is the start of the VR1 gradient chain. These ranks never produce a `pending_grad_to_send`, so the chain pattern `have_pending_send=True, need_recv=True` never applies to them.

### Solution

Replaced the `RuntimeError` with a genuine standalone recv. The partner rank (R1 for VR0, R2 for VR1) flushes its pending gradient at the `-1` sync marker (PRE-SYNC FLUSH). NCCL buffers this send until the standalone recv is posted by R0/R3.

**Code location:** `chimera_2vr.py` around line 1623–1644:

```python
elif not have_pending_send and need_recv:
    # STANDALONE RECV: Valid for grad_last ranks (R0 for VR0, R3 for VR1).
    recv_from_prev = (model_chunk_id == 1)  # VR1 recvs from prev
    recv_from_next = (model_chunk_id == 0)  # VR0 recvs from next
    recv_prev_grad, recv_next_grad = p2p_communication.chimera_communicate(
        tensor_send_prev=None, tensor_send_next=None,
        recv_prev=recv_from_prev, recv_next=recv_from_next,
        tensor_shape=tensor_shape, config=config,
    )
    output_tensor_grad = recv_prev_grad if recv_prev_grad is not None else recv_next_grad
```

**Verified working:** Logs confirmed R0 and R3 successfully received their gradients.

---

## Fix 2: Single End-Sync Allreduce (Context from previous session, confirmed still in place)

The scheduler uses **one allreduce at the end of cooldown** (not two). The `-1` sync markers in both the forward and backward schedules both map to the same single `allreduce_gradients()` call. This is the correct design for Chimera 2-VR.

---

## Problem: NCCL "Duplicate GPU" Error in BD Groups

After the standalone recv fix, a new error appeared:

```
torch.distributed.DistBackendError: NCCL error in:
    AllReduce: ncclInvalidUsage: Duplicate GPU detected:
    rank 1 and rank 2 both on CUDA device 22000
```

This error fires at `torch.distributed.barrier(group=parallel_state.get_bd_parallel_group())`.

### Root Cause

The machine has 4x RTX 3060 with a two-NUMA-node PCIe topology:
- GPU 0: PCIe bus `21:00.0`
- GPU 1: PCIe bus `22:00.0`  ← same root complex as GPU 0
- GPU 2: PCIe bus `41:00.0`  ← different root complex
- GPU 3: PCIe bus `42:00.0`  ← same root complex as GPU 2

Without V-shaped rank reordering, Chimera uses sequential pipeline ranks `[0,1,2,3]`. This produces BD groups:
- BD group 0: `[0, 3]` → GPU 0 + GPU 3 (cross-NUMA!)
- BD group 1: `[1, 2]` → GPU 1 + GPU 2 (cross-NUMA!)

NCCL 2.18.5 on this hardware has a bug/quirk with cross-NUMA GPU pairs in 2-rank collective communicator initialization. It reports "Duplicate GPU" even though `check_ranks.py` confirmed all 4 ranks are on distinct GPUs.

BitPipe works because it uses V-shaped rank reordering `[0,3,2,1]`, which produces BD groups `[0,1]` and `[3,2]` — both same-NUMA pairs.

---

## Attempted Fix: V-Shaped Rank Reordering for Chimera

### What Was Done

Two changes in `megatron/core/parallel_state.py`:

**Change 1 — Apply V-shape reordering to Chimera (lines 258–266):**
```python
_enable_chimera = hasattr(get_args(), 'enable_chimera_schedule') and get_args().enable_chimera_schedule
if get_args().enable_bitpipe_schedule or _enable_chimera:
    p_ranks = ranks
    ranks = []
    for k in range(pipeline_model_parallel_size):
        if k % 2 == 0:
            ranks.append(p_ranks[k])
        else:
            ranks.append(p_ranks[pipeline_model_parallel_size - k])
```

This changes `[0,1,2,3]` → `[0,3,2,1]` for Chimera, so BD groups become `[0,1]` and `[3,2]` (same-NUMA).

**Change 2 — Apply position_embedding_ranks to Chimera (lines 320–323):**
```python
if get_args().enable_bitpipe_schedule or _enable_chimera:
    position_embedding_ranks = [ranks[0], ranks[-1]]
```

### What This Changes

With V-shaped ranks `[0,3,2,1]`, the LOGICAL pipeline rank ordering changes:
- Physical rank 0 → Pipeline stage 0 (unchanged)
- Physical rank 3 → Pipeline stage 1  (was: phys 1)
- Physical rank 2 → Pipeline stage 2  (unchanged)
- Physical rank 1 → Pipeline stage 3  (was: phys 3)

The layer distribution and pre/post_process flags adapt automatically — they are computed from LOGICAL pipeline rank, not physical rank. Verified from VR_INIT logs:

```
[VR_INIT] Chimera 2-VR | Device 2 | VR 1 | pre_process=False | post_process=False  ✓ (middle)
[VR_INIT] Chimera 2-VR | Device 1 | VR 1 | pre_process=False | post_process=False  ✓ (middle)
[VR_INIT] Chimera 2-VR | Device 3 | VR 1 | pre_process=True  | post_process=False  ✓ (phys 1 = PR3 = last stage)
```

Layer assignment with V-shape (correct bidirectional pairing):

| Pipeline rank | Physical rank | VR0 layers | VR1 layers |
|---|---|---|---|
| 0 | 0 | 1–6 | 19–24 |
| 1 | 3 | 7–12 | 13–18 |
| 2 | 2 | 13–18 | 7–12 |
| 3 | 1 | 19–24 | 1–6 |

BD groups are now `[0,1]` and `[3,2]` — same-NUMA pairs. The NCCL "Duplicate GPU" error should be gone.

### Current Status: STUCK ⚠️

The script gets through model initialization (all 4 ranks print parameter counts) but then hangs. The hang is somewhere **after model creation**, not during initialization. Based on the logs, pre/post process is correct and layer distribution is correct.

The hang is most likely one of:
- The first `torch.distributed.barrier()` call after model setup (in `print_datetime()`)
- The first training step's P2P communication (a new deadlock pattern caused by the changed physical routing)

**We did not have time to diagnose this further.** The next session should run the script with full output capture (`tee /tmp/chimera_out.txt`) and identify whether the hang is before or after training starts.

---

## How to Revert the V-Shaped Reordering for Chimera

If the reordering turns out to cause problems and you want to go back to sequential ranks for Chimera while keeping it for BitPipe, make two changes in `megatron/core/parallel_state.py`:

**Revert Change 1** — Restore the original BitPipe-only condition (line ~259):
```python
# BEFORE (current - both BitPipe and Chimera):
if get_args().enable_bitpipe_schedule or _enable_chimera:

# AFTER (revert - BitPipe only):
if get_args().enable_bitpipe_schedule:
```

**Revert Change 2** — Restore the original position_embedding_ranks condition (line ~320):
```python
# BEFORE (current):
if get_args().enable_bitpipe_schedule or _enable_chimera:
    position_embedding_ranks = [ranks[0], ranks[-1]]

# AFTER (revert):
if get_args().enable_bitpipe_schedule:
    position_embedding_ranks = [ranks[0], ranks[-1]]
```

Note: `_enable_chimera` variable definition on line ~258 can stay — it's harmless.

**Why you might want to revert:** The V-shaped reordering changes the physical P2P routing for Chimera (VR0 forward: 0→3→2→1 instead of 0→1→2→3). If this causes issues with the Chimera scheduler's assumptions about rank ordering, reverting is the first debugging step. The NCCL "Duplicate GPU" error is hardware-specific — it only occurs on machines where cross-NUMA BD group pairs trigger the NCCL bug. On machines without this PCIe topology quirk, sequential ranks work fine.

**Alternative fix (no reordering):** Instead of reordering ranks, you could create the BD groups using P2P-style sends instead of `new_group()` collectives. This avoids NCCL collectives entirely and would work regardless of NUMA topology. However, this requires more invasive changes to `allreduce_gradients()` in `chimera_2vr.py`.

---

## Files Changed This Session

| File | Change |
|------|--------|
| `megatron/core/pipeline_parallel/schedule_impl/chimera/chimera_2vr.py` | Replaced DEADLOCK GUARD RuntimeError with standalone recv for grad-last ranks (~line 1623) |
| `megatron/core/parallel_state.py` | Extended V-shaped rank reordering and `position_embedding_ranks` to Chimera; fixed BD group comment |

---

## Next Steps

1. Run `bash singlenode_gpt_chimera.sh 2>&1 | tee /tmp/chimera_out.txt` and share full output
2. Identify exact hang point: before or after `[after model, optimizer... built]` datetime log
3. If hang is in training: inspect whether V-shaped routing causes a new P2P deadlock
4. If hang is in a barrier: check whether BD group creation itself is the issue (try adding prints around `new_group()` calls)
