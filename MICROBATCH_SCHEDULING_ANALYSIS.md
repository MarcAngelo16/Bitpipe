# Microbatch Scheduling: BitPipe vs Chimera

## Executive Summary

| Aspect | BitPipe 4-VR | Chimera 2-VR |
|--------|--------------|--------------|
| **Scheduling Function** | `get_microbatch_idx()` + `get_bkmicrobatch_idx()` | `build_chimera_forward_schedule()` + `build_chimera_backward_schedule()` |
| **VRs per Device** | 4 (VR0, VR1, VR2, VR3) | 2 (VR0, VR1) |
| **Layer Pattern** | V-shaped | Sequential |
| **Algorithm Complexity** | 5-phase nested loops | Simple 2-branch interleaving |
| **Lines of Code** | ~90 | ~170 (split across 2 functions) |
| **Readability** | 🔴 Complex | 🟢 Clear |

---

## BitPipe: `get_microbatch_idx()` Algorithm

### Location
File: `megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py`, lines 256-300

### What It Does
Computes a **single globally optimized execution order** for all microbatches, specifically designed for the 4-VR V-shaped pattern.

### Key Insight
**Different ranks execute microbatches in different orders** - this is intentional and required for optimal V-shaped pipeline utilization.

### Algorithm Structure
```python
def get_microbatch_idx(total_num_microbatches, pipeline_parallel_rank):
    # Step 1: Group microbatches by VR
    microbatch_id01 = get_microbatch(total_num_microbatches)
    # Returns: microbatch_id01[0,1,2,3] = lists of microbatch IDs per VR

    # Step 2: Calculate rank-dependent parameters
    num_unit = pipeline_parallel_size // 2           # 2 (for 4 devices)
    i_half = pipeline_parallel_rank // num_unit      # 0 or 1
    num_initial = ...  # Rank-dependent initial count

    # Step 3: Build order using 5-phase loop
    for j in range(i_loop):
        # Phase 1: Initial VR0/VR1 microbatches
        # Phase 2: Alternation between halves
        # Phase 3: More complex alternation
        # Phase 4: Final alternation pattern
        # Phase 5: Final grouped microbatches
```

### Example Output (4 Devices, 8 Base MBs → 16 Total)
```
Device 0: [0,    1, 2, 6, 3, 7, 4,    5, ...]
Device 1: [0, 2, 1, 3, 6, 4, 7, 5, ...]
Device 2: [2, 0, 3, 1, 4, 6, 5, 7, ...]
Device 3: [2,    3, 0, 4, 1, 5, 6,    7, ...]
```

**Observation**: Notice each device has a different order! This is the V-shaped optimization at work.

### Why So Complex?
BitPipe's V-shaped pattern creates these constraints:
- 4 VRs with specific device-pairing relationships
- Rank reordering affects the logical flow
- Must balance pipeline utilization across all 4 VRs simultaneously
- Paper-optimized algorithm for maximum throughput

---

## Chimera: `build_chimera_forward_schedule()` Algorithm

### Location
File: `megatron/core/pipeline_parallel/schedule_impl/chimera/chimera_2vr.py`, lines 118-200 & 203-285

### What It Does
Builds **two separate schedules** (forward and backward) optimized for 2-VR sequential pattern.

### Algorithm Structure
```python
def build_chimera_forward_schedule(total_num_microbatches, pipeline_parallel_size, pipeline_parallel_rank):
    # Step 1: Simple grouping by VR
    microbatch_groups = get_chimera_microbatch_groups(total_num_microbatches)
    # Returns: [VR0_list, VR1_list]

    # Step 2: Calculate rank parameters
    num_unit = pipeline_parallel_size // 2
    is_first_half = pipeline_parallel_rank < num_unit

    # Step 3: Build schedule (simple 2-branch logic)
    if is_first_half:
        # Rank 0, 1 (first half): prioritize VR0 first
        offset = pipeline_parallel_rank
        Add initial VR0 microbatches
        Interleave remaining VR0 and VR1
    else:
        # Rank 2, 3 (second half): prioritize VR1 first
        offset = pipeline_parallel_size - 1 - pipeline_parallel_rank
        Add initial VR1 microbatches
        Interleave remaining VR0 and VR1
```

### Example Output (4 Devices, 8 Base MBs → 16 Total)
```
VR0: [0, 1, 2, 3, 4, 5, 6, 7]       (first half)
VR1: [8, 9, 10, 11, 12, 13, 14, 15] (second half)

Rank 0: [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7]
Rank 1: [0, 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7]
Rank 2: [8, 9, 0, 10, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 7]
Rank 3: [0, 14, 1, 15, 2, 12, 3, 13, 4, 10, 5, 11, 6, 7]
```

**Observation**: Much simpler pattern - mostly sequential with rank-dependent starting offset.

### Why Simpler?
With only 2 VRs and sequential layer assignment:
- No complex device-pairing relationships
- Simple alternation between VR0 and VR1 is sufficient
- Rank differences are less critical
- More intuitive to understand and extend

---

## Backward Schedule

### BitPipe: `get_bkmicrobatch_idx()`
- **Lines**: 303-351
- **Pattern**: Similar 5-phase algorithm but processes VR2/VR3 first (backward pipeline flows opposite)
- **Includes**: Sync markers (-1) placed at specific positions
- **Complexity**: High (similar to forward schedule)

### Chimera: `build_chimera_backward_schedule()`
- **Lines**: 203-285
- **Pattern**: Reverses VR0/VR1 processing order (VR1 first, then VR0)
- **Includes**: Simple sync marker placement logic
- **Complexity**: Low (basic branching)

---

## Usage in Training Loop

### BitPipe
```python
# Line 249-254: Compute schedules ONCE at start
microbatch_idx = get_microbatch_idx(total_num_microbatches, pipeline_parallel_rank)
bkmicrobatch_idx = get_bkmicrobatch_idx(total_num_microbatches, pipeline_parallel_rank)

# Line 560-596: Warmup - use pre-computed microbatch_idx
for k in range(num_warmup_microbatches):
    microbatch_id = microbatch_idx[k]
    model_chunk_id = get_model_chunk_id(microbatch_id)
    # Execute forward pass
```

### Chimera
```python
# Line 391-396: Build schedules ONCE at start
forward_schedule = build_chimera_forward_schedule(...)
backward_schedule = build_chimera_backward_schedule(...)

# Line 511-549: Warmup - use pre-built forward_schedule
for i in range(num_warmup_microbatches):
    microbatch_id = forward_schedule[i]
    model_chunk_id = get_model_chunk_id(microbatch_id)
    # Execute forward pass
```

---

## Comparison Table

| Feature | BitPipe | Chimera |
|---------|---------|---------|
| **Function Count** | 4 (get_microbatch, get_bkmicrobatch_idx, get_model_chunk_id, is_first/last_mb) | 4 (2 schedule builders + helper functions) |
| **VR Count** | 4 | 2 |
| **Max Nesting Depth** | 4 levels | 2 levels |
| **Sync Marker Placement** | Complex (lines 335-339) | Simple (lines 278-283) |
| **Rank-Dependent Variations** | High (all ranks different) | Moderate (2 halves differ) |
| **Theoretical Optimality** | ✅ Paper-optimized | ⚠️ Pragmatic heuristic |
| **Maintainability** | 🔴 Hard | 🟢 Easy |

---

## Key Differences Summary

1. **Architecture**:
   - BitPipe: Single unified scheduling algorithm for all 4 VRs
   - Chimera: Separate forward/backward schedules for 2 VRs

2. **Complexity**:
   - BitPipe: 5-phase interleaving with complex indexing
   - Chimera: Simple alternation with rank-based starting offset

3. **Rank Variation**:
   - BitPipe: All 4 ranks get completely different orders
   - Chimera: First half (0,1) vs second half (2,3) differ, but within each half similar patterns

4. **Optimization Target**:
   - BitPipe: Optimized for V-shaped 4-VR pattern (from paper)
   - Chimera: Optimized for simplicity while maintaining good efficiency

---

## Performance Implications

**BitPipe Advantages**:
- Theoretically optimal for 4-VR V-shaped execution
- Proven in research paper
- Likely achieves highest throughput

**Chimera Trade-offs**:
- Simpler to understand and debug
- Easier to extend for asymmetric mode
- Likely 1-3% performance penalty vs BitPipe
- Worth it for maintainability if performance difference is acceptable
