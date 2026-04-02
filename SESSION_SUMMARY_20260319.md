# Session Summary — 2026-03-19

## Topics Covered

1. CPU-mode feasibility for Chimera (short answer: not possible)
2. Log rank confusion due to V-shape reordering → fix applied to `print_all_ranks`
3. **Chimera deadlock with 4 devices + 8 microbatches** (main investigation)

---

## 1. CPU Mode Not Feasible

**Question**: Can Chimera run with CPU processes (gloo backend) instead of GPUs?

**Answer**: No, without significant code surgery. `p2p_communication.py` hard-codes `torch.cuda.current_device()` for every buffer allocation (lines ~47-61, ~315-446). Megatron-LM assumes CUDA everywhere. The `gloo` backend can handle collectives on CPU tensors, but the model tensors are always CUDA.

**Workaround**: Share one GPU across all 4 processes:
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0   # all 4 processes share GPU 0
torchrun --nproc-per-node 4 gpt_dummy.py --enable-chimera-schedule ...
# Scale down: --num-layers 4 --hidden-size 64 --seq-length 64
```

---

## 2. Log Rank Confusion Fix

**Problem**: `print_all_ranks` used `torch.distributed.get_rank()` (global/physical rank) in the header, but log message bodies used `pipeline_parallel_rank` (logical pipeline rank). With V-shape reordering these differ for physical ranks 1 and 3:

| Physical (global) | Pipeline rank |
|---|---|
| 0 | 0 |
| **1** | **3** ← swapped |
| 2 | 2 |
| **3** | **1** ← swapped |

**Root cause** (`parallel_state.py:257-264`): V-shape IS applied to Chimera (despite comment saying it isn't). The group is created with sequential ranks `[0,1,2,3]` at line 247, BEFORE the reorder. So `rank_in_group == global_rank`. Then `get_pipeline_model_parallel_rank()` at lines 503-508 maps odd ranks: `rank_in_group=1 → pipeline=(4-1)=3`, `rank_in_group=3 → pipeline=(4-3)=1`.

Output would show e.g.:
```
[Chimera Rank 1/World 1]  Chimera Config: pipeline_size=4, rank=3   ← confusing!
[Chimera Rank 3/World 3]  Chimera Config: pipeline_size=4, rank=1
```

**Fix applied** (`chimera_2vr.py:98-113`): Changed header to show `G{global}/P{pipeline}`:
```python
global_rank = torch.distributed.get_rank()
pipeline_rank = mpu.get_pipeline_model_parallel_rank()
print(f"[Chimera G{global_rank}/P{pipeline_rank}] {message}")
```
Now `[Chimera G1/P3]` makes it clear: physical device 1 has pipeline rank 3.

---

## 3. Chimera Deadlock: 4 Devices + 8 Microbatches

### The Problem

Chimera works with 4 devices + 4 microbatches but **deadlocks** with 4 devices + 8 microbatches.
BitPipe handles 8 MBs fine because it has a fundamentally different loop structure.

---

### Key Structural Difference: BitPipe has `n_loop`, Chimera does not

**BitPipe** (`bitpipe_4vr.py:207`):
```python
n_loop = total_num_microbatches // pipeline_parallel_size // 2 - 1
# For 8 user MBs: total=16 (doubled), 4 stages → n_loop = 16//4//2 - 1 = 1
```

BitPipe wraps the 1F1B in an **outer loop over `n_loop` chunks**. Each chunk processes exactly `pipeline_size` microbatches in one clean 1F1B block before moving to the next chunk. The structure:
```
Warmup (fill pipeline)
For each chunk (n_loop times):
    1F1B for unit_remaining*2 steps  (one chunk = pipeline_size MBs)
    Mini-cooldown (drain this chunk's backwards)
Final cooldown
```

**Chimera** (`chimera_2vr.py:697-708`):
```python
num_warmup_microbatches = get_num_warmup_microbatches(...)
num_1f1b_microbatches = total_num_microbatches - num_warmup_microbatches
num_cooldown_microbatches = num_warmup_microbatches + 1
```

Chimera runs **one flat warmup + one big 1F1B loop + one cooldown**. For 4 MBs, the 1F1B loop runs 2 iterations. For 8 MBs, it runs 6 iterations. There is no outer chunk loop.

---

### Schedule Simulation: 4 Devices, 8 Microbatches

VR assignment: `(mb_id % 4) // 2` → MB 0,1,4,5 = VR0; MB 2,3,6,7 = VR1

**Forward schedules** (from `get_chimera_microbatch_idx(8, 4, rank)`):
```
Pipeline 0: [0,1, 2,3, 4,5, 6,7]        warmup=2, 1f1b=6
Pipeline 1: [0,2, 1,3, 4,6, 5,7]        warmup=3, 1f1b=5
Pipeline 2: [2,0, 3,1, 6,4, 7,5]        warmup=3, 1f1b=5
Pipeline 3: [2,3, 0,1, 6,7, 4,5]        warmup=2, 1f1b=6
```

**Backward schedules** (= paired rank's fwd + [-1]):
```
Pipeline 0 bwd: [2,3, 0,1, 6,7, 4,5, -1]   (paired with pipeline 3)
Pipeline 1 bwd: [2,0, 3,1, 6,4, 7,5, -1]   (paired with pipeline 2)
Pipeline 2 bwd: [0,2, 1,3, 4,6, 5,7, -1]   (paired with pipeline 1)
Pipeline 3 bwd: [0,1, 2,3, 4,5, 6,7, -1]   (paired with pipeline 0)
```

**VR pattern in 1F1B for pipeline 0** (6 iterations):
```
k=0: fwd MB2(VR1), bwd MB2(VR1)
k=1: fwd MB3(VR1), bwd MB3(VR1)   ← VR switch coming: next fwd is VR0 (first stage)
k=2: fwd MB4(VR0), bwd MB0(VR0)
k=3: fwd MB5(VR0), bwd MB1(VR0)   ← VR switch: next fwd is VR1 (not first stage)
k=4: fwd MB6(VR1), bwd MB6(VR1)
k=5: fwd MB7(VR1), bwd MB7(VR1)   ← True end of 1F1B
```

---

### The Bug: Bridge Fires Mid-1F1B

**Code location**: `chimera_2vr.py:1357-1440` (the `elif need_send_bwd and not need_recv_fwd:` branch).

**What triggers the bridge at k=1 (mid-1F1B)**:

After backward of MB3(VR1) at pipeline 0:
```python
has_next_forward = (fwd_idx < len(microbatch_idx))  # True: fwd_idx=4, len=8
next_fwd_mb = microbatch_idx[4] = MB4(VR0)
next_fwd_is_first = is_vr_first_stage_for_activation(VR0, pipeline_rank=0, ...)  # True! rank 0 IS first stage for VR0
need_recv_fwd = has_next_forward AND NOT next_fwd_is_first  # = True AND NOT True = False
```

So `need_send_bwd=True` (VR1 grad must go to pipeline 1) but `need_recv_fwd=False` (no recv needed, next fwd starts at this device). This falls into the bridge branch:

```python
# chimera_2vr.py:1363
first_cooldown_mb = microbatch_idx_b[bwd_idx + 1]  # = microbatch_idx_b[2] = MB0(VR0)
can_bridge = not cool_is_grad_first  # True for VR0 at pipeline 0
# Bridge fires: send VR1 grad to pipeline 1 + recv VR0 grad from pipeline 1
```

**Why this deadlocks**:

At the same moment (k=1 of 1F1B), pipeline 2 is at its own k=0 backward doing `chimera_send_prev_recv_next`:
- Sends VR0 grad to pipeline 1 (prev)
- Receives VR1 fwd from pipeline 3 (next)

Pipeline 3's bridge also fires at k=1 of its 1F1B (symmetric case):
- Sends VR0 grad to pipeline 2 (prev)
- Expects to recv VR1 grad from pipeline 2

But pipeline 2 sends to pipeline 1, NOT pipeline 3. Pipeline 3 waits forever for a recv from pipeline 2 that never comes.

```
Pipeline 3 bridge:  SEND VR0→pipeline2,  RECV VR1←pipeline2
Pipeline 2 k=0 bwd: SEND VR0→pipeline1,  RECV VR1←pipeline3
                         ↑ pipeline3 expects this but it goes to pipeline1 instead
                                           ↑ pipeline3 is trying to send to pipeline2, not here
→ DEADLOCK
```

The bridge was **designed for the true end of 1F1B** (when `not has_next_forward`). It fires incorrectly at mid-1F1B VR-switch points where `has_next_forward=True` but `next_fwd_is_first=True`.

---

### Why 4 MBs Does NOT Deadlock

For 4 MBs, pipeline 0 1F1B (2 iterations):
```
k=0: fwd MB2(VR1), bwd MB2(VR1)  → normal combined comm
k=1: fwd MB3(VR1), bwd MB3(VR1)  → TRUE end (no more forwards), bridge fires correctly
```

At k=1, `has_next_forward=False` (fwd schedule fully consumed). The bridge fires at the TRUE end, where pipeline 1 is simultaneously doing its backward and sending the VR0 grad that pipeline 0 expects to pre-fetch. The communication aligns because all ranks converge at the same phase boundary together.

With 8 MBs, ranks are at different positions in their longer 1F1B loops, so the bridge fires at k=1 while the other ranks are still mid-loop at their own k=0 or k=1 with incompatible communication.

---

### The Required Fix

**Option A (Recommended): Add `n_loop` outer loop, like BitPipe**

```python
# chimera_2vr.py — replace flat 1F1B with chunked loop
n_loop = num_microbatches // pipeline_parallel_size - 1
# For 8 MBs, 4 stages: n_loop = 8//4 - 1 = 1

# Warmup (unchanged, fills pipeline with first chunk)
...

# Outer loop: one iteration per extra chunk
for loop_iter in range(n_loop):
    # Inner 1F1B: process unit_remaining*2 steps for this chunk
    unit_remaining = pipeline_parallel_size // 2 - position_offset
    for k in range(unit_remaining * 2):
        fwd + bwd (interleaved)
    # Mini-cooldown: drain this chunk's remaining backwards
    for k in range(warmup_offset):
        bwd only

# Final cooldown (same as current)
...
```

**Option B (Quick patch): Tighten bridge condition**

Change the bridge branch from triggering on ANY `not need_recv_fwd` to ONLY triggering at the true end:

```python
# chimera_2vr.py:1357
elif need_send_bwd and not need_recv_fwd:
    if not has_next_forward:
        # TRUE end of 1F1B — bridge is safe
        first_cooldown_mb = microbatch_idx_b[bwd_idx + 1]
        ... (existing bridge logic)
    else:
        # Mid-1F1B VR switch (next_fwd_is_first=True but more forwards remain)
        # Standalone send only — next fwd iteration handles itself (no recv needed)
        if bwd_model_chunk_id == 0:
            p2p_communication.chimera_send_prev_only(input_tensor_grad, config)
        else:
            p2p_communication.chimera_send_next_only(input_tensor_grad, config)
```

**Warning on Option B**: Standalone sends can still deadlock if the receiver is not posting a matching recv at the same time. Option B needs careful verification that mid-1F1B standalone sends always have a matching recv on the other rank. Option A (n_loop) is safer because the chunk structure guarantees all ranks are at the same phase simultaneously.

---

### Current Status

| Scenario | Status |
|---|---|
| Chimera 4 devices, 4 MBs | ✅ Works |
| Chimera 4 devices, 8 MBs | ❌ Deadlocks (bridge fires mid-1F1B) |
| Chimera 8 devices, any MBs | ❌ Not tested, same structural issue |
| BitPipe 4 devices, 8 MBs | ✅ Works (n_loop handles extra chunk) |

**Root cause**: Chimera has no `n_loop` outer chunk structure. The flat 1F1B loop with 6 iterations creates mid-loop VR switches that trigger the bridge at the wrong time, causing communication mismatches between ranks.

**Next step**: Implement `n_loop` in Chimera, modeled after `bitpipe_4vr.py:597-669`.

---

## 4. n_loop Implementation Plan (From Scratch)

### Formula Derivation

**Setup (4 devices, 8 MBs):**
- `pipeline_parallel_size = 4`
- `num_microbatches = 8`

**n_loop**: Number of extra chunks beyond the first (which warmup handles):
```
n_loop = num_microbatches // pipeline_parallel_size - 1
       = 8 // 4 - 1 = 1
```
For 4 MBs: `4 // 4 - 1 = 0` → no outer loop, falls back to existing flat schedule. ✅

**Warmup counts per rank** (from `get_num_warmup_microbatches()`):
```
Pipeline 0: warmup = 2   (position_in_half=0, num_unit=2)
Pipeline 1: warmup = 3   (position_in_half=1, num_unit=2)
Pipeline 2: warmup = 3   (position_in_half=1, num_unit=2)
Pipeline 3: warmup = 2   (position_in_half=0, num_unit=2)
```

**unit_remaining per rank** = forwards to do in 1F1B per chunk = `pipeline_parallel_size - warmup_count`:
```
Pipeline 0: unit_remaining = 4 - 2 = 2
Pipeline 1: unit_remaining = 4 - 3 = 1
Pipeline 2: unit_remaining = 4 - 3 = 1
Pipeline 3: unit_remaining = 4 - 2 = 2
```

**mini-cooldown per rank** = backwards to drain between chunks = `warmup_count`:
```
Pipeline 0: mini-cooldown = 2 bwd-only steps
Pipeline 1: mini-cooldown = 3 bwd-only steps
Pipeline 2: mini-cooldown = 3 bwd-only steps
Pipeline 3: mini-cooldown = 2 bwd-only steps
```

### Per-Rank Simulation (4 devices, 8 MBs)

Forward schedule (abbreviated fwd[i] = microbatch_idx[i]):
```
Pipeline 0 fwd: [0,1, 2,3, 4,5, 6,7]
Pipeline 1 fwd: [0,2, 1,3, 4,6, 5,7]
Pipeline 2 fwd: [2,0, 3,1, 6,4, 7,5]
Pipeline 3 fwd: [2,3, 0,1, 6,7, 4,5]
```

**Pipeline 0** (warmup=2, unit_remaining=2, mini-cooldown=2):
```
Phase            | Fwd idx | Fwd MB | Bwd idx | Bwd MB
--- WARMUP ---
step 0 (fwd)     | 0       | MB0    | —       | —
step 1 (fwd)     | 1       | MB1    | —       | —
--- CHUNK 0: 1F1B (unit_remaining=2 pairs) ---
step 0 (fwd+bwd) | 2       | MB2    | 0       | MB2(pair0-fwd)
step 1 (fwd+bwd) | 3       | MB3    | 1       | MB3(pair0-fwd)
--- CHUNK 0: mini-cooldown (warmup=2 bwd-only) ---
step 0 (bwd)     | —       | —      | 2       | bwd MB0
step 1 (bwd)     | —       | —      | 3       | bwd MB1
--- CHUNK 1: 1F1B (unit_remaining=2 pairs) ---
step 0 (fwd+bwd) | 4       | MB4    | 4       | bwd MB4(pair1-fwd)
step 1 (fwd+bwd) | 5       | MB5    | 5       | bwd MB5(pair1-fwd)
--- CHUNK 1: mini-cooldown → becomes FINAL COOLDOWN ---
step 0 (bwd)     | —       | —      | 6       | bwd MB6
step 1 (bwd)     | —       | —      | 7       | bwd MB7
+ ALLREDUCE SYNC
```
Total fwd=8 ✅, bwd=8 ✅

**Pipeline 1** (warmup=3, unit_remaining=1, mini-cooldown=3):
```
Phase            | Fwd idx | Fwd MB | Bwd idx | Bwd MB
--- WARMUP ---
step 0 (fwd)     | 0       | MB0    | —       | —
step 1 (fwd)     | 1       | MB2    | —       | —
step 2 (fwd)     | 2       | MB1    | —       | —
--- CHUNK 0: 1F1B (unit_remaining=1 pair) ---
step 0 (fwd+bwd) | 3       | MB3    | 0       | bwd MB0(pair)
--- CHUNK 0: mini-cooldown (warmup=3 bwd-only) ---
step 0 (bwd)     | —       | —      | 1       | bwd MB2
step 1 (bwd)     | —       | —      | 2       | bwd MB1
step 2 (bwd)     | —       | —      | 3       | bwd MB3
--- CHUNK 1: 1F1B (unit_remaining=1 pair) ---
step 0 (fwd+bwd) | 4       | MB4    | 4       | bwd MB4(pair)
--- CHUNK 1: mini-cooldown → FINAL COOLDOWN ---
step 0 (bwd)     | —       | —      | 5       | bwd MB6
step 1 (bwd)     | —       | —      | 6       | bwd MB5
step 2 (bwd)     | —       | —      | 7       | bwd MB7
+ ALLREDUCE SYNC
```
Total fwd=8 ✅, bwd=8 ✅

Pipelines 2 and 3 are symmetric mirrors of 1 and 0 respectively.

### Key Insight: Why All Ranks Align at Chunk Boundaries

At the end of each chunk (after mini-cooldown), all ranks have processed exactly `pipeline_parallel_size` microbatches in fwd and the matching bwds. Since every rank does exactly the same number of steps in warmup + 1F1B + mini-cooldown per chunk, the chunk boundary is a **natural synchronization point** where no rank can get ahead. This prevents the bridge from firing while another rank is still mid-1F1B.

### Pseudocode Skeleton

```python
# chimera_2vr.py — replace flat 1F1B section

n_loop = num_microbatches // pipeline_parallel_size - 1
# n_loop = 0 for original 4MB case → no outer loop, same behavior
# n_loop = 1 for 8MB case → one extra chunk

num_warmup = get_num_warmup_microbatches(...)
unit_remaining = pipeline_parallel_size - num_warmup   # 1F1B pairs per chunk
# mini-cooldown backwards = num_warmup (drain warmup debt before next chunk)

# --- WARMUP (unchanged) ---
for k in range(num_warmup):
    run_forward(microbatch_idx[fwd_idx])
    fwd_idx += 1

# --- CHUNKED LOOP ---
for chunk in range(n_loop + 1):
    is_last_chunk = (chunk == n_loop)

    # Inner 1F1B
    for k in range(unit_remaining):
        run_forward(microbatch_idx[fwd_idx]);  fwd_idx += 1
        run_backward(microbatch_idx_b[bwd_idx]); bwd_idx += 1

    if not is_last_chunk:
        # Mini-cooldown: drain warmup debt before next chunk
        for k in range(num_warmup):
            run_backward(microbatch_idx_b[bwd_idx]); bwd_idx += 1
    else:
        # Final cooldown: drain remaining backwards + allreduce sync
        for k in range(num_warmup):
            if microbatch_idx_b[bwd_idx] == -1:
                allreduce_gradients(model)
            else:
                run_backward(microbatch_idx_b[bwd_idx])
            bwd_idx += 1
```

**Notes:**
- When `n_loop=0` (4 MBs), the outer loop runs once with `chunk=0=n_loop`, so it goes directly to final cooldown — identical to current behavior.
- The bridge logic (`elif need_send_bwd and not need_recv_fwd`) should be guarded to only fire in final cooldown (i.e., when `not has_next_forward` is truly the end). With n_loop in place this is naturally satisfied.
- The backward schedule `microbatch_idx_b` includes `-1` sync markers; the mini-cooldown steps must skip these or handle them correctly (same as current cooldown logic).

---

## Files Modified This Session

| File | Change |
|---|---|
| `megatron/core/pipeline_parallel/schedule_impl/chimera/chimera_2vr.py` | `print_all_ranks` now shows `G{global}/P{pipeline}` in header |

## Files to Modify Next Session

| File | Change Needed |
|---|---|
| `megatron/core/pipeline_parallel/schedule_impl/chimera/chimera_2vr.py` | Add `n_loop` outer chunk loop to 1F1B phase |
| `asymmetric_bitpipe/scripts/scheduling_sim.py` | Verify/update simulator to match n_loop structure |
