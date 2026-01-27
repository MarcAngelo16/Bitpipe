## Session Summary - November 16, 2025

---

## Overview

This document captures the detailed discussion on understanding BitPipe's 4-VR bidirectional pipeline scheduling and designing Chimera's 2-VR simplified scheduling algorithm.

**Key Participants:** Discussing the core differences between BitPipe (V-shaped, 4-VR) and Chimera (linear, 2-VR) scheduling.

**Goal:** Create correct `chimera_get_microbatch_idx()` and `chimera_get_bkmicrobatch_idx()` functions for Chimera 2-VR scheduler.

---

## Core Understanding Achieved

### 1. BitPipe Architecture (4-VR, Bidirectional, V-Shaped)

**Two Concurrent Pipelines:**

```
PIPELINE 0 (starts from Rank 0):
  Forward:  MB0 (VR0: rank0→1→2→3) → [Transform: MB0 becomes MB8] → MB8 (VR2: rank3→2→1→0)
  Backward: MB8 (VR2: rank0→1→2→3) → [Transform: MB8 becomes MB0] → MB0 (VR0: rank3→2→1→0)
  Key: Starts at rank0, ends at rank0 (V-shaped pattern)

PIPELINE 1 (starts from Rank 3):
  Forward:  MB2 (VR1: rank3→2→1→0) → [Transform: MB2 becomes MB9] → MB9 (VR3: rank0→1→2→3)
  Backward: MB9 (VR3: rank3→2→1→0) → [Transform: MB9 becomes MB2] → MB2 (VR1: rank0→1→2→3)
  Key: Starts at rank3, ends at rank3 (V-shaped pattern)

These two pipelines run CONCURRENTLY!
```

**Microbatch Grouping (8 base MBs → 16 total):**
- **VR0**: [0, 1, 4, 5] (first pipeline forward, first half)
- **VR1**: [2, 3, 6, 7] (second pipeline forward, first half)
- **VR2**: [8, 9, 12, 13] (first pipeline backward, second half = same data as VR0 but transformed)
- **VR3**: [10, 11, 14, 15] (second pipeline backward, second half = same data as VR1 but transformed)

**Why times 2?** The same tensor data flows through DIFFERENT VR paths with different IDs tracked separately:
- MB0 (tensor) → VR0 as MB0 → VR2 as MB8 (same tensor, different ID)
- MB2 (tensor) → VR1 as MB2 → VR3 as MB9 (same tensor, different ID)

**Critical Code:** `get_model_chunk_id()` in bitpipe_4vr.py (line 353)
```python
def get_model_chunk_id(microbatch_id):  # 0,1,4,5 #2,3,6,7 # 8,9,12,13 #10,11,14,15
    microbatch_id_in_group = microbatch_id % (pipeline_parallel_size)
    chunk_offset = 0 if microbatch_id < (total_num_microbatches//2) else 2
    model_chunk_id = microbatch_id_in_group // (pipeline_parallel_size // 2)
    model_chunk_id += chunk_offset
    return model_chunk_id
```

---

### 2. Chimera Architecture (2-VR, Bidirectional, LINEAR)

**Two Concurrent Pipelines:**

```
PIPELINE 0 (starts from Rank 0):
  Forward:  MB0 (VR0: rank0→1→2→3) [Done, stays MB0]
  Backward: MB0 (VR0: rank3→2→1→0) [Done]
  Key: Starts at rank0, ends at rank3 (linear pattern, NOT V-shaped)

PIPELINE 1 (starts from Rank 3):
  Forward:  MB2 (VR1: rank3→2→1→0) [Done, stays MB2]
  Backward: MB2 (VR1: rank0→1→2→3) [Done]
  Key: Starts at rank3, ends at rank0 (linear pattern, NOT V-shaped)

These two pipelines run CONCURRENTLY!
```

**Microbatch Grouping (8 base MBs → stays 8, NO doubling):**
- **VR0**: [0, 1, 4, 5] (first pipeline, forward and backward)
- **VR1**: [2, 3, 6, 7] (second pipeline, forward and backward)

**Why NO times 2?** The tensor keeps the same identity throughout:
- MB0 (tensor) → VR0 forward → VR0 backward as MB0 (same tensor, same ID throughout)
- MB2 (tensor) → VR1 forward → VR1 backward as MB2 (same tensor, same ID throughout)

**Key Difference from BitPipe:** No transformation at rank boundaries. The data doesn't change identity.

---

## BitPipe Forward & Backward Schedules (Example)

**Configuration:** 8 base microbatches, 4 devices → 16 total MBs (doubled)

### Forward Schedule (get_microbatch_idx):
```
Rank 0: [0, 1, 2, 10, 3, 11, 8, 9, 4, 5, 6, 14, 7, 15, 12, 13]
Rank 1: [0, 2, 1, 3, 10, 8, 11, 9, 4, 6, 5, 7, 14, 12, 15, 13]
Rank 2: [2, 0, 3, 1, 8, 10, 9, 11, 6, 4, 7, 5, 12, 14, 13, 15]
Rank 3: [2, 3, 0, 8, 1, 9, 10, 11, 6, 7, 4, 12, 5, 13, 14, 15]
```

### Backward Schedule (get_bkmicrobatch_idx):
```
Rank 0: [8, 9, 10, 2, 11, 3, 0, 1, 12, 13, 14, 6, 15, 7, 4, -1, 5, -1]
Rank 1: [8, 10, 9, 11, 2, 0, 3, 1, 12, 14, 13, 15, 6, 4, 7, 5, -1, -1]
Rank 2: [10, 8, 11, 9, 0, 2, 1, 3, 14, 12, 15, 13, 4, 6, 5, 7, -1, -1]
Rank 3: [10, 11, 8, 0, 9, 1, 2, 3, 14, 15, 12, 4, 13, 5, 6, -1, 7, -1]
```

**Notes on Backward Schedule:**
- Contains `-1` sync markers (gradient synchronization points)
- Different placement per rank (some have -1 at end, some in middle)

---

## Chimera Forward & Backward Schedules (Target Design)

**Configuration:** 8 base microbatches, 4 devices → stays 8 MBs (NOT doubled)

### Target Forward Schedule (with f/b suffix for clarity):
```
Rank 0: [f0, f1, f2, f3, f4, f5, f6, f7]
Rank 1: [f0, f2, f1, f3, f4, f6, f5, f7]
Rank 2: [f2, f0, f3, f1, f6, f4, f7, f5]
Rank 3: [f2, f3, f0, f6, f7, f4, ...]
```

### Target Backward Schedule (with f/b suffix for clarity):
```
Rank 0: [b2, b3, b0, b1, b6, b7, b4, b5, -1, -1]
Rank 1: [b2, b0, b3, b1, b6, b4, b7, b5, -1, -1]
Rank 2: [b0, b2, b1, b3, b4, b6, b5, b7, -1, -1]
Rank 3: [b0, b1, b2, b3, b4, b5, b6, b7, -1, -1]
```

**Key Observations:**
- Forward and backward are separate (unlike BitPipe's mixed schedule)
- Backward reverses the VR processing order (B before A instead of A before B)
- Each rank has exactly 2 sync markers (-1)
- Total 10 items per rank (8 MBs + 2 syncs)

---

## BitPipe Forward Pass Algorithm (5-Phase Loop)

**Code Location:** `megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py:256-300`

**Key Variables:**
```python
i_loop = total_num_microbatches // pipeline_parallel_size // 2  # = 2 loops
num_unit = pipeline_parallel_size // 2  # = 2
i_half = pipeline_parallel_rank // num_unit  # 0 for ranks 0-1, 1 for ranks 2-3
num_initial = (
    num_unit - pipeline_parallel_rank if pipeline_parallel_rank < num_unit
    else pipeline_parallel_rank - num_unit + 1
)
# Rank 0: num_initial = 2, Rank 1: num_initial = 1
# Rank 2: num_initial = 1, Rank 3: num_initial = 2
```

**5-Phase Pattern (in each loop iteration j):**
```python
Phase 1: for i in range(num_initial):
    microbatch_idx.append(microbatch_id01[i_half][j*num_unit+i])

Phase 2: for i in range(num_unit - num_initial):
    microbatch_idx.append(microbatch_id01[1-i_half][j*num_unit+i])
    microbatch_idx.append(microbatch_id01[i_half][j*num_unit+i+num_initial])

Phase 3: for i in range(num_initial):
    microbatch_idx.append(microbatch_id01[1-i_half][j*num_unit+i+(num_unit-num_initial)])
    microbatch_idx.append(microbatch_id01[3-i_half][j*num_unit+i])

Phase 4: for i in range(num_unit - num_initial):
    microbatch_idx.append(microbatch_id01[2+i_half][j*num_unit+i])
    microbatch_idx.append(microbatch_id01[3-i_half][j*num_unit+i+num_initial])

Phase 5: for i in range(num_initial):
    microbatch_idx.append(microbatch_id01[2+i_half][j*num_unit+i+(num_unit-num_initial)])
```

**What This Does:**
- Carefully interleaves VR0/VR1 (first half) with VR2/VR3 (second half)
- Uses `i_half` and `num_initial` to create rank-dependent ordering
- Early ranks (0,1) prioritize their own VR first
- Later ranks (2,3) also adapt their pattern
- Maintains pipeline fill with strategic interleaving

---

## BitPipe Backward Pass Algorithm (5-Phase Loop with Sync)

**Code Location:** `megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py:303-351`

**5-Phase Pattern (similar to forward but processes VR2/VR3 first):**
```python
Phase 1: for k in range(num_initial):
    microbatch_idx.append(microbatch_id01[2+i_half][j*num_unit+k])

Phase 2: for k in range(num_unit-num_initial):
    microbatch_idx.append(microbatch_id01[3-i_half][j*num_unit+k])
    microbatch_idx.append(microbatch_id01[2+i_half][j*num_unit+k+num_initial])

Phase 3: for k in range(num_initial):
    microbatch_idx.append(microbatch_id01[3-i_half][(j*num_unit+k+num_unit-num_initial)])
    microbatch_idx.append(microbatch_id01[1-i_half][j*num_unit+k])

Phase 4: for k in range(num_unit-num_initial):
    microbatch_idx.append(microbatch_id01[i_half][j*num_unit+k])
    microbatch_idx.append(microbatch_id01[1-i_half][j*num_unit+k+num_initial])

Phase 5: for k in range(num_initial):
    microbatch_idx.append(microbatch_id01[i_half][j*num_unit+k+num_unit-num_initial])

Sync Points (after all phases):
if pipeline_parallel_rank == pipeline_parallel_size // 2 or pipeline_parallel_rank == pipeline_parallel_size // 2 - 1:
    microbatch_idx.append(-1)
else:
    microbatch_idx.insert(-1, -1)  # Insert before last element
microbatch_idx.append(-1)  # Always add final sync
```

**What This Does:**
- Reverses the VR processing order: starts with VR2/VR3 (second half) instead of VR0/VR1
- Creates symmetric inverse pattern to forward pass
- Adds gradient synchronization markers (-1)
- Middle ranks (0,1) have sync marker at the end, outer ranks insert before last

---

## Chimera Design Challenge

### Question 1: Forward Pass Ordering Rule

**Current observation from example:**
```
Rank0 Forward: f0, f1, f2, f3, f4, f5, f6, f7
Rank1 Forward: f0, f2, f1, f3, f4, f6, f5, f7
Rank2 Forward: f2, f0, f3, f1, f6, f4, f7, f5
Rank3 Forward: f2, f3, f0, f6, f7, f4, ...
```

**Need to determine:**
- What is the algorithm that produces this ordering?
- Does it follow a pattern similar to BitPipe's 5-phase approach (adapted for 2-VR)?
- How do `i_half` and `num_initial` determine the order for Chimera?

### Question 2: Backward Pass Pattern

**Current observation from example:**
```
Rank0 Backward: b2, b3, b0, b1, b6, b7, b4, b5
Rank1 Backward: b2, b0, b3, b1, b6, b4, b7, b5
```

**Need to determine:**
- Is backward a reverse of forward (same rank pattern but inverted)?
- Or does it follow its own algorithm?
- How are the -1 sync markers placed?

### Question 3: Microbatch Grouping for Chimera

**Proposed approach:**
- Use adapted version of BitPipe's `get_model_chunk_id()` for 2-VR
- VR0 gets MBs where: `(mb_id % num_devices) // (num_devices // 2) == 0`
- VR1 gets MBs where: `(mb_id % num_devices) // (num_devices // 2) == 1`

**This gives:**
- Chimera VR0: [0, 1, 4, 5]
- Chimera VR1: [2, 3, 6, 7]

---

## Key Differences: BitPipe vs Chimera

| Aspect | BitPipe 4-VR | Chimera 2-VR |
|--------|--------------|--------------|
| **Total MBs** | 2N (doubled) | N (not doubled) |
| **VRs** | 4 (VR0, VR1, VR2, VR3) | 2 (VR0, VR1) |
| **Pattern** | V-shaped (Rank 0→N→0) | Linear (Rank 0→N or N→0) |
| **Transformation** | MB0→MB8 at rank boundary | No transformation |
| **Microbatch ID** | Changes mid-journey | Stays same throughout |
| **Pipelines** | 2 concurrent with V-paths | 2 concurrent with linear paths |
| **Scheduling** | Complex 5-phase interleaving | Simpler interleaving (TBD) |
| **Sync Markers** | Variable placement | Standard (2 per rank) |

---

## Next Steps

1. **Verify BitPipe's 5-phase algorithm** by hand-tracing Rank 0, Rank 1, etc.
2. **Identify the exact forward pass ordering rule** for Chimera
3. **Determine the backward pass ordering rule** for Chimera
4. **Design equivalent functions:**
   - `chimera_get_microbatch_idx()`
   - `chimera_get_bkmicrobatch_idx()`
5. **Implement in:** `megatron/core/pipeline_parallel/schedule_impl/chimera/chimera_2vr.py`
6. **Update scheduling_sim.py** to generate Chimera schedules

---

## Files Involved

- **BitPipe Implementation:** `/workspace/Bitpipe/megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py`
- **Chimera Implementation:** `/workspace/Bitpipe/megatron/core/pipeline_parallel/schedule_impl/chimera/chimera_2vr.py`
- **Simulator:** `/workspace/Bitpipe/asymmetric_bitpipe/scripts/scheduling_sim.py`
- **Session Summary:** `/workspace/Bitpipe/asymmetric_bitpipe/scripts/session_summary_20251116.md` (this file)

---

## References

- CLAUDE.md: Complete BitPipe and Chimera documentation
- BitPipe Paper: Bidirectional Interleaved Pipeline Parallelism
- Megatron-LM: Base framework for pipeline parallelism

---

**Last Updated:** November 16, 2025
**Status:** Analysis phase - awaiting verification of BitPipe algorithm and Chimera design decisions