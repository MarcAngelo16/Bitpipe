# Session Summary - December 2, 2025

## Critical Bug Fix: Chimera Cooldown Loop Skipping Backward Passes

### Problem Identified

The Chimera 2-VR profiler was not capturing backward passes during the cooldown phase, resulting in profiles showing `num_backward_passes: 0` even though backward execution was happening.

**Root Cause:**
- When `num_microbatches_remaining = 0` (warmup phase fills entire pipeline, no 1F1B phase)
- The cooldown loop was incorrectly slicing the backward schedule
- All backward microbatches were being skipped, only sync markers were processed

### Example Scenario

Configuration: 4 microbatches, 4 devices

```
Forward Schedule (all MBs):
  Rank 0: [0, 1, 2, 3]  (all 4 MBs in warmup)

Backward Schedule (generated):
  Rank 0: [2, 3, 0, 1, -1, -1]  (4 MBs for backward + 2 sync markers)

Phase Breakdown:
  num_warmup_microbatches = 4
  num_microbatches_remaining = 0  (NO 1F1B phase!)

Buggy Loop:
  microbatch_idx_b[num_warmup_microbatches:]
  = microbatch_idx_b[4:]
  = [-1, -1]  ← ONLY SYNCS! Missing [2,3,0,1]
```

### Changes Made

#### File: `megatron/core/pipeline_parallel/schedule_impl/chimera/chimera_2vr.py`

**Location:** Lines 662-671 (Cooldown Phase Loop)

##### BEFORE (Buggy):
```python
    # === COOLDOWN PHASE ===
    if not forward_only:
        print_all_ranks("=== Starting COOLDOWN phase ===")

        if profiler:
            profiler.record_phase_transition("cooldown_start")

        # Process remaining backward passes
        cooldown_start = num_warmup_microbatches + num_microbatches_remaining
        for i, bwd_microbatch_id in enumerate(microbatch_idx_b[num_warmup_microbatches:]):
            # BUG: When num_microbatches_remaining=0, this skips ALL backward MBs!
            # Only processes sync markers (-1 items)
```

##### AFTER (Fixed):
```python
    # === COOLDOWN PHASE ===
    if not forward_only:
        print_all_ranks("=== Starting COOLDOWN phase ===")

        if profiler:
            profiler.record_phase_transition("cooldown_start")

        # Process remaining backward passes
        # CRITICAL: If num_microbatches_remaining = 0, there's NO 1F1B phase, so we process entire backward schedule
        # Otherwise, we skip the backward MBs already handled in 1F1B phase
        if num_microbatches_remaining > 0:
            cooldown_start_idx = num_warmup_microbatches
        else:
            # No 1F1B phase, process all backward MBs starting from index 0
            cooldown_start_idx = 0

        for i, bwd_microbatch_id in enumerate(microbatch_idx_b[cooldown_start_idx:]):
            # NOW: Correctly processes all backward MBs when no 1F1B phase exists
```

### Why This Change

**The Issue:**
The original code assumed that `num_warmup_microbatches` items in the backward schedule corresponded to the warmup phase and should be skipped. However, the backward schedule is **separate** from the forward schedule:

- **Forward Schedule:** Used in warmup + 1F1B phases
- **Backward Schedule:** Used in 1F1B (partially) + cooldown phases

When there's **no 1F1B phase** (`num_microbatches_remaining = 0`), the backward schedule should be processed entirely during cooldown, not skipped.

**Impact on Profiling:**
- With buggy code: Backward profiler calls never executed
- With fixed code: All backward microbatches are profiled correctly
- Result: Profiles now show accurate `num_backward_passes` count

### Expected Profile Behavior After Fix

**Before Fix:**
```json
{
  "metadata": {"schedule_type": "chimera"},
  "summary": {
    "num_forward_passes": 4,
    "num_backward_passes": 0    ← WRONG!
  }
}
```

**After Fix (Expected):**
```json
{
  "metadata": {"schedule_type": "chimera"},
  "summary": {
    "num_forward_passes": 4,
    "num_backward_passes": 4    ← CORRECT!
  }
}
```

### How to Revert (If Needed)

If you want to test a different scheduling pattern and need to revert this change:

```python
# Simply remove lines 663-669 and restore the original line:
# Replace this block:
        # Process remaining backward passes
        # CRITICAL: If num_microbatches_remaining = 0, there's NO 1F1B phase, so we process entire backward schedule
        # Otherwise, we skip the backward MBs already handled in 1F1B phase
        if num_microbatches_remaining > 0:
            cooldown_start_idx = num_warmup_microbatches
        else:
            # No 1F1B phase, process all backward MBs starting from index 0
            cooldown_start_idx = 0

        for i, bwd_microbatch_id in enumerate(microbatch_idx_b[cooldown_start_idx:]):

# Back to:
        # Process remaining backward passes
        cooldown_start = num_warmup_microbatches + num_microbatches_remaining
        for i, bwd_microbatch_id in enumerate(microbatch_idx_b[num_warmup_microbatches:]):
```

### Related Code Sections

The cooldown loop continues with the rest of the backward processing (lines 672+):
```python
            if bwd_microbatch_id == -1:
                # Gradient sync marker
                enable_grad_sync()
                for chunk_id in range(len(model)):
                    if chunk_id not in synchronized_model_chunks:
                        allreduce_gradients(model[chunk_id])
                        synchronized_model_chunks.add(chunk_id)
                disable_grad_sync()
            else:
                # Regular backward microbatch
                if profiler:
                    profiler.start_microbatch(bwd_microbatch_id, 0, model_chunk_id, 'backward')
                # ... backward step execution ...
                if profiler:
                    profiler.end_microbatch(bwd_microbatch_id, 0, model_chunk_id, 'backward')
```

This code was never being reached for backward MBs due to the incorrect slice.

### Testing

To verify the fix works:
1. Run Chimera training with profiling enabled: `--enable-bitpipe-profiling`
2. Check the generated profile JSON files in `asymmetric_bitpipe/profiles/raw/`
3. Verify `num_backward_passes` matches expected count
4. Re-run `analyze_bitpipe_profile.py` to verify visualizations show backward passes

### Notes

- This fix **only affects the cooldown phase** when `num_microbatches_remaining = 0`
- Schedules with larger `num_microbatches_remaining` are unaffected
- The profiler infrastructure itself was working correctly; only the scheduler loop was skipping items
- All 4 files in `chimera_profile_rank*.json` should now show backward passes in their microbatch_events
