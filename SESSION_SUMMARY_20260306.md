# Session Summary — 2026-03-06

## Status at Start of Session

Chimera 2-VR was running (V-shape rank reordering working, BD groups correct), but crashed on the **second training iteration** with:

```
File "megatron/training.py", line 568, in train_step
    for key in losses_reduced[0]:
TypeError: iteration over a 0-d tensor
```

First iteration completed successfully. Error appeared at the start of the second iteration.

---

## Root Cause: Duplicate (Broken) Loss Collection in Chimera Scheduler

The Chimera scheduler had **two separate loss collection mechanisms** that conflicted:

### Mechanism 1: `forward_data_store` (correct — shared with BitPipe)
Inside `forward_step()` (schedules.py line 210-215):
```python
if parallel_state.is_pipeline_last_stage():
    output_tensor = loss_func(output_tensor)
    loss, loss_reduced = output_tensor
    output_tensor = loss / num_microbatches
    forward_data_store.append(loss_reduced)  # ← appends a DICT: {'lm loss': tensor}
```

### Mechanism 2: `losses_reduced` (broken — added manually in Chimera)
Inside `chimera_2vr.py` at warmup and 1F1B phases:
```python
losses_reduced = []          # line 686

# At warmup "last stage":
losses_reduced.append(output_tensor)  # line 992 — output_tensor is a 0-d SCALAR!

# At 1F1B "last stage":
losses_reduced.append(output_tensor)  # line 1120 — same problem

return losses_reduced        # line 1704 — returns scalars, not dicts
```

At the last stage, `forward_step` returns `loss / num_microbatches` — a **0-d scalar tensor**.
The Chimera scheduler was collecting these raw scalars, then returning them as `losses_reduced`.
The training loop expects dicts (`{'lm loss': tensor}`), so iterating `losses_reduced[0]` threw the TypeError.

### Why BitPipe Never Had This Bug

BitPipe uses only 3 lines for the same purpose:
```python
forward_data_store = []      # line 154
forward_step(..., forward_data_store, ...)   # line 414 — forward_step fills it correctly
return forward_data_store    # line 1010
```

No manual appending, no separate `losses_reduced`. The Chimera scheduler was a mistaken deviation from this pattern.

---

## Secondary Bug: `is_pipeline_stage_containing_loss()` Missing Chimera

In `megatron/utils.py`:
```python
def is_pipeline_stage_containing_loss():
    if get_args().enable_bitpipe_schedule:
        return mpu.is_pipeline_first_stage(ignore_virtual=True) or mpu.is_pipeline_last_stage(ignore_virtual=True)
    else:
        return mpu.is_pipeline_last_stage(ignore_virtual=True)  # ← Chimera fell here
```

For Chimera, this returned True **only for rank N-1** (last device). But Chimera has loss at **both ends**:
- **VR0 pipeline** ends at rank N-1 (last device)
- **VR1 pipeline** ends at rank 0 (first device)

So rank 0 (which correctly had VR1 loss in `forward_data_store`) was being silently skipped, and its losses were never aggregated.

---

## Fixes Applied

### Fix 1: `megatron/core/pipeline_parallel/schedule_impl/chimera/chimera_2vr.py`

1. Removed `losses_reduced = []` initialization
2. Removed `losses_reduced.append(output_tensor)` in warmup phase
3. Removed `losses_reduced.append(output_tensor)` in 1F1B phase
4. Changed `return losses_reduced` → `return forward_data_store`

Chimera now matches BitPipe's pattern exactly.

### Fix 2: `megatron/utils.py`

```python
def is_pipeline_stage_containing_loss():
    if get_args().enable_bitpipe_schedule or (hasattr(get_args(), 'enable_chimera_schedule') and get_args().enable_chimera_schedule):
        # Both BitPipe and Chimera: VR0 ends at rank N-1, VR1 ends at rank 0
        return mpu.is_pipeline_first_stage(ignore_virtual=True) or mpu.is_pipeline_last_stage(ignore_virtual=True)
    else:
        return mpu.is_pipeline_last_stage(ignore_virtual=True)
```

---

## Result

Training runs successfully past the first iteration. The TypeError is resolved.

---

## Known Remaining Issues

1. **Log readability**: V-shape rank reordering (`_PIPELINE_GLOBAL_RANKS = [0,3,2,1]`) makes logs hard to follow because P2P shows `R0: send→R3` which looks wrong but is correct for same-NUMA BD groups. Consider cleaning up verbose debug prints after validation is complete.

2. **Loss averaging**: With both rank 0 and rank N-1 now entering the loss averaging block in `train_step`, both ranks return a `loss_reduced` dict. This is the same behavior as BitPipe and appears to work correctly, but has not been verified for correctness of the averaged loss values across both pipeline directions.

3. **Verbose debug prints**: Many `print_all_ranks` and `[forward_step]`, `[Transformer]`, `[P2P ACTUAL]` debug prints are still active. These should be removed/reduced once the implementation is confirmed stable.

---

## Files Modified This Session

- `megatron/core/pipeline_parallel/schedule_impl/chimera/chimera_2vr.py` — removed broken `losses_reduced`, return `forward_data_store`
- `megatron/utils.py` — extended `is_pipeline_stage_containing_loss()` to Chimera
