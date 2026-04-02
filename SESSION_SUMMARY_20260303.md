# Session Summary – 2026-03-03

## What We Discussed

This session was a conceptual/analysis session — no code was changed. We reviewed two
outstanding bugs in Chimera and clarified the NUMA topology issue from last session.

---

## Topic 1: NUMA Topology Clarification

Reviewed why the NCCL "Duplicate GPU" error occurs on this machine.

### Hardware Topology

```
NUMA Node 0 (PCIe root complex 1):   GPU 0 (bus 21:00.0)  GPU 1 (bus 22:00.0)
NUMA Node 1 (PCIe root complex 2):   GPU 2 (bus 41:00.0)  GPU 3 (bus 42:00.0)
```

Same-NUMA GPU pairs can communicate directly via shared PCIe root — fast.
Cross-NUMA GPU pairs must route through the CPU — slow, and triggers a bug in NCCL 2.18.5.

### NCCL "Duplicate GPU" Error Is a False Positive

The error message `rank 1 and rank 2 both on CUDA device 22000` is **misleading**.
The GPUs are genuinely different physical devices. It is a bug in NCCL 2.18.5 where the
internal GPU fingerprinting incorrectly identifies cross-NUMA GPU pairs as the same device.

- `check_ranks.py` confirmed all 4 ranks map to distinct GPUs (correct rank definitions)
- Error fires at `torch.distributed.barrier(group=bd_group)` — the first operation that
  forces NCCL to initialize the communicator for that group
- The actual allreduce code never runs because the barrier already throws

### Sequential vs V-Shape BD Group Physical Pairing

| | Chimera sequential `[0,1,2,3]` | BitPipe V-shape `[0,3,2,1]` |
|---|---|---|
| Pipeline stage 0 | Physical GPU 0 (NUMA 0) | Physical GPU 0 (NUMA 0) |
| Pipeline stage 1 | Physical GPU 1 (NUMA 0) | Physical GPU 3 (NUMA 1) |
| Pipeline stage 2 | Physical GPU 2 (NUMA 1) | Physical GPU 2 (NUMA 1) |
| Pipeline stage 3 | Physical GPU 3 (NUMA 1) | Physical GPU 1 (NUMA 0) |
| BD Group 0 | GPU 0 + GPU 3 = cross-NUMA ❌ | GPU 0 + GPU 1 = same-NUMA ✅ |
| BD Group 1 | GPU 1 + GPU 2 = cross-NUMA ❌ | GPU 3 + GPU 2 = same-NUMA ✅ |

The V-shape reordering is a workaround that avoids creating cross-NUMA BD groups.
Current code in `parallel_state.py` applies V-shape to BitPipe only (Chimera reverted
after a hang was observed — see SESSION_SUMMARY_20260224.md).

---

## Topic 2: BD Group Membership (Layer-Sharing Analysis)

Clarified which devices belong to the same BD group and why.

### Rule: BD Group = Devices That Share the Same Layers

Both BitPipe and Chimera follow the same pairing: **pipeline stage `i` pairs with stage `N-1-i`**.

**BitPipe (4 devices, 24 layers):**
```
Device0: VR0[1-3]   VR1[10-12]  VR2[22-24]  VR3[13-15]
Device3: VR0[10-12] VR1[1-3]    VR2[13-15]  VR3[22-24]
  → share layers 1-3, 10-12, 13-15, 22-24  → BD Group 0: [Device0, Device3]

Device1: VR0[4-6]   VR1[7-9]    VR2[19-21]  VR3[16-18]
Device2: VR0[7-9]   VR1[4-6]    VR2[16-18]  VR3[19-21]
  → share layers 4-6, 7-9, 16-18, 19-21    → BD Group 1: [Device1, Device2]
```

**Chimera (4 devices, 24 layers):**
```
Device0: VR0[1-6]    VR1[19-24]
Device3: VR0[19-24]  VR1[1-6]
  → share layers 1-6, 19-24  → BD Group 0: [Device0, Device3]

Device1: VR0[7-12]   VR1[13-18]
Device2: VR0[13-18]  VR1[7-12]
  → share layers 7-12, 13-18 → BD Group 1: [Device1, Device2]
```

The logical BD grouping is IDENTICAL for both architectures. The V-shape only affects
which physical GPUs the logical pipeline stages map to.

---

## Topic 3: Bug Found — Wrong Allreduce Chunk Order in Chimera ⚠️

**This is a correctness bug that would produce wrong gradients even if the NCCL issue
were resolved.**

### Root Cause

For a BD allreduce to be correct, both devices in the group must call allreduce on
tensors containing **the same model layers**. Because the layer assignment is mirrored
between paired devices (VR0 on stage 0 = VR1 on stage N-1, and vice versa), the chunk
iteration order must be reversed for second-half ranks.

**BitPipe handles this correctly:**
```python
offset = range(num_model_chunks//2) if pipeline_parallel_rank < pipeline_parallel_size//2 \
         else reversed(range(num_model_chunks//2))
for i_chunk in offset:
    allreduce_gradients(model[num_model_chunks//2 + i_chunk])
```

- First-half ranks (stages 0,1): iterate [0,1] → allreduce VR2 then VR3
- Second-half ranks (stages 2,3): iterate [1,0] → allreduce VR3 then VR2 (reversed)

This ensures that when stage 0 calls allreduce(VR2) and stage 3 calls allreduce(VR3),
both are contributing gradients for the **same layers** (e.g. layers 22-24).

**Chimera is missing the reversal:**
```python
for chunk_id in range(len(model)):   # Always [0, 1] for ALL ranks — BUG
    allreduce_gradients(model[chunk_id])
```

With the same order on all ranks, the allreduce pairs wrong layers:

```
Stage 0 calls allreduce(model[0]) → VR0 = layers 1-6
Stage 3 calls allreduce(model[0]) → VR0 = layers 19-24   ← DIFFERENT LAYERS ❌

Stage 0 calls allreduce(model[1]) → VR1 = layers 19-24
Stage 3 calls allreduce(model[1]) → VR1 = layers 1-6     ← DIFFERENT LAYERS ❌
```

NCCL will execute without error (tensors have the same shape) but the averaged gradient
is a meaningless mix of two different model layers. Training would be incorrect.

### Fix (Not Yet Applied)

```python
# Reverse chunk iteration for second-half ranks, like BitPipe
if pipeline_parallel_rank < pipeline_parallel_size // 2:
    chunk_order = range(len(model))            # [0, 1]
else:
    chunk_order = reversed(range(len(model)))  # [1, 0]

for chunk_id in chunk_order:
    if chunk_id not in synchronized_model_chunks:
        allreduce_gradients(model[chunk_id])
        synchronized_model_chunks.add(chunk_id)
```

With the fix, BD Group 0 = [stage 0, stage 3]:
```
Stage 0 (logical rank 0 < 2): order [0,1] → allreduce VR0[1-6]  then VR1[19-24]
Stage 3 (logical rank 3 ≥ 2): order [1,0] → allreduce VR1[1-6]  then VR0[19-24]

Allreduce 1: VR0[1-6]   ↔ VR1[1-6]   = same layers ✓
Allreduce 2: VR1[19-24] ↔ VR0[19-24] = same layers ✓
```

**File to change:** `megatron/core/pipeline_parallel/schedule_impl/chimera/chimera_2vr.py`
Look for the sync marker block (`if microbatch_id == -1:`) in the COOLDOWN PHASE loop.

---

## Summary of Outstanding Bugs

| Bug | Status | Fix |
|-----|--------|-----|
| NCCL "Duplicate GPU" on cross-NUMA BD groups | Known — V-shape reorder applied but caused hang (see 20260224) | Apply V-shape to Chimera + debug hang |
| Wrong allreduce chunk order (different layers averaged) | **Newly found this session** | Reverse `chunk_order` for second-half ranks |

---

## Next Steps

1. Apply the allreduce chunk order fix in `chimera_2vr.py`
2. Re-apply V-shape reordering to Chimera in `parallel_state.py` (see revert instructions
   in SESSION_SUMMARY_20260224.md)
3. Run `bash singlenode_gpt_chimera.sh 2>&1 | tee /tmp/chimera_out.txt` to diagnose
   the hang that appeared after V-shape was applied last session
4. Verify that after both fixes, BD allreduce is producing correct layer-matched gradients
   (check via debug prints before/after allreduce)
