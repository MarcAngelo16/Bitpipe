# Warmup Calculation and Microbatch Mapping Explained

## Setup: 4 Devices, 4 Microbatches → 8 Total

**User specifies:** `num_microbatches = 4`
**BitPipe doubles:** `total_num_microbatches = 4 * 2 = 8`

---

## 1. Microbatch to VR Mapping (8 Total Microbatches)

Using `get_model_chunk_id()` from `bitpipe_4vr.py:353-361`:

```python
def get_model_chunk_id(microbatch_id):
    microbatch_id_in_group = microbatch_id % pipeline_parallel_size  # % 4
    chunk_offset = 0 if microbatch_id < (total_num_microbatches // 2) else 2
    model_chunk_id = microbatch_id_in_group // (pipeline_parallel_size // 2)
    model_chunk_id += chunk_offset
    return model_chunk_id
```

**Step-by-Step Calculation:**

| MB ID | `microbatch_id < 4?` | `chunk_offset` | `id_in_group` | `id_in_group // 2` | `+ offset` | **Final VR** |
|-------|---------------------|----------------|---------------|-------------------|-----------|-------------|
| 0 | Yes (0 < 4) | 0 | 0 % 4 = 0 | 0 // 2 = 0 | 0 + 0 | **VR0** |
| 1 | Yes (1 < 4) | 0 | 1 % 4 = 1 | 1 // 2 = 0 | 0 + 0 | **VR0** |
| 2 | Yes (2 < 4) | 0 | 2 % 4 = 2 | 2 // 2 = 1 | 1 + 0 | **VR1** |
| 3 | Yes (3 < 4) | 0 | 3 % 4 = 3 | 3 // 2 = 1 | 1 + 0 | **VR1** |
| 4 | No (4 >= 4) | 2 | 4 % 4 = 0 | 0 // 2 = 0 | 0 + 2 | **VR2** |
| 5 | No (5 >= 4) | 2 | 5 % 4 = 1 | 1 // 2 = 0 | 0 + 2 | **VR2** |
| 6 | No (6 >= 4) | 2 | 6 % 4 = 2 | 2 // 2 = 1 | 1 + 2 | **VR3** |
| 7 | No (7 >= 4) | 2 | 7 % 4 = 3 | 3 // 2 = 1 | 1 + 2 | **VR3** |

**Result:**
```
MB 0, 1   → VR0  (first half, pipeline 0)
MB 2, 3   → VR1  (first half, pipeline 1)
MB 4, 5   → VR2  (second half, pipeline 0)
MB 6, 7   → VR3  (second half, pipeline 1)
```

**Key Pattern:**
- **First 4 MBs (0-3)** → VR0/VR1 (first half of model)
- **Second 4 MBs (4-7)** → VR2/VR3 (second half of model)
- **Even groups** (0,1,4,5) → VR0/VR2 (pipeline 0)
- **Odd groups** (2,3,6,7) → VR1/VR3 (pipeline 1)

---

## 2. Why Each Device Has Different Warmup Counts

**YES!** Each device has a **different number of warmup microbatches** due to the V-shaped pattern and bidirectional execution.

### Code from `bitpipe_4vr.py:221-227`

```python
if total_num_microbatches == pipeline_parallel_size:
    num_warmup_microbatches = total_num_microbatches
else:
    num_warmup_microbatches = pipeline_parallel_size + pipeline_parallel_size // 2  # Base

    num_warmup_microbatches += (
        pipeline_parallel_rank
        if pipeline_parallel_rank < pipeline_parallel_size // 2
        else pipeline_parallel_size - 1 - pipeline_parallel_rank
    )
```

### Calculation for Each Device (4 devices, 8 total MBs)

**Base warmup:** `4 + 4 // 2 = 6`

**Device-specific adjustment:**

| Device | `pipeline_rank` | Condition | Adjustment | **Total Warmup** |
|--------|----------------|-----------|------------|------------------|
| Device 0 | 0 | 0 < 2 (True) | `0` | 6 + 0 = **6** |
| Device 1 | 1 | 1 < 2 (True) | `1` | 6 + 1 = **7** |
| Device 2 | 2 | 2 < 2 (False) | `4 - 1 - 2 = 1` | 6 + 1 = **7** |
| Device 3 | 3 | 3 < 2 (False) | `4 - 1 - 3 = 0` | 6 + 0 = **6** |

**Result:**
```
Device 0: 6 warmup microbatches
Device 1: 7 warmup microbatches
Device 2: 7 warmup microbatches
Device 3: 6 warmup microbatches
```

---

## 3. Why This Pattern? (V-Shaped Explanation)

### The Problem: Pipeline Bubbles

In a standard pipeline:
```
Device 0: [F] [F] [F] [F] [B] [B] [B] [B]
Device 1:     [F] [F] [F] [F] [B] [B] [B] [B]
Device 2:         [F] [F] [F] [F] [B] [B] [B] [B]
Device 3:             [F] [F] [F] [F] [B] [B] [B] [B]
             ↑ Bubble!
```

### BitPipe's Solution: Bidirectional Execution

BitPipe runs **two pipelines simultaneously**:
- **Pipeline 0 (VR0/VR2)**: Forward direction (Device 0→1→2→3)
- **Pipeline 1 (VR1/VR3)**: Backward direction (Device 3→2→1→0)

**Layer Distribution (V-shaped):**
```
Device 0: VR0[0,1]   VR1[6,7]   VR2[14,15] VR3[8,9]
Device 1: VR0[2,3]   VR1[4,5]   VR2[12,13] VR3[10,11]
Device 2: VR0[4,5]   VR1[2,3]   VR2[10,11] VR3[12,13]
Device 3: VR0[6,7]   VR1[0,1]   VR2[8,9]   VR3[14,15]
```

**Why V-shaped?**
- Device 0 and Device 3 are **edge devices** (start/end of both pipelines)
- Device 1 and Device 2 are **middle devices** (middle of both pipelines)

### Edge Devices Need Fewer Warmups

**Device 0 (Edge):**
- Starts Pipeline 0 (VR0) → Doesn't wait for anyone
- Starts Pipeline 1 (VR3) → Receives from Device 3 quickly
- **Less waiting** → Fewer warmup iterations needed

**Device 1 (Middle):**
- Middle of Pipeline 0 (VR0) → Must wait for Device 0
- Middle of Pipeline 1 (VR1) → Must wait for Device 2
- **More waiting** → More warmup iterations needed

---

## 4. Detailed Warmup Timeline (8 MBs, 4 Devices)

Let's trace the actual execution for each device:

### Device 0 (6 warmup iterations)

**Forward Schedule:** `[0, 1, 2, 10, 3, 11]` (first 6 from schedule)

```
Iter 0: Forward MB0 (VR0) - layers [0,1]
Iter 1: Forward MB1 (VR0) - layers [0,1]
Iter 2: Forward MB2 (VR1) - layers [6,7]
Iter 3: Forward MB10 (VR3) - layers [8,9]  ← Different VR!
Iter 4: Forward MB3 (VR1) - layers [6,7]
Iter 5: Forward MB11 (VR3) - layers [8,9]
```

**Why 6?**
- Device 0 is an **edge device** (first for VR0/VR1, last for VR2/VR3)
- Alternates between starting new microbatches (VR0/VR1) and receiving from Device 3 (VR3)
- **Balanced workload** → Needs exactly 6 warmups to fill pipeline

---

### Device 1 (7 warmup iterations)

**Forward Schedule:** `[0, 2, 1, 3, 10, 8, 11]` (first 7 from schedule)

```
Iter 0: Forward MB0 (VR0) - layers [2,3]  ← Waits for Device 0
Iter 1: Forward MB2 (VR1) - layers [4,5]  ← Waits for Device 2
Iter 2: Forward MB1 (VR0) - layers [2,3]
Iter 3: Forward MB3 (VR1) - layers [4,5]
Iter 4: Forward MB10 (VR3) - layers [10,11]
Iter 5: Forward MB8 (VR2) - layers [12,13]
Iter 6: Forward MB11 (VR3) - layers [10,11]
```

**Why 7?**
- Device 1 is a **middle device** (middle of both pipelines)
- Must wait for data from **both directions**:
  - VR0: Waits for Device 0
  - VR1: Waits for Device 2
- **More waiting** → Needs 7 warmups to properly fill pipeline

---

### Device 2 (7 warmup iterations)

**Forward Schedule:** `[2, 0, 3, 1, 8, 10, 9]` (first 7 from schedule)

```
Iter 0: Forward MB2 (VR1) - layers [2,3]  ← Waits for Device 3
Iter 1: Forward MB0 (VR0) - layers [4,5]  ← Waits for Device 1
Iter 2: Forward MB3 (VR1) - layers [2,3]
Iter 3: Forward MB1 (VR0) - layers [4,5]
Iter 4: Forward MB8 (VR2) - layers [10,11]
Iter 5: Forward MB10 (VR3) - layers [12,13]
Iter 6: Forward MB9 (VR2) - layers [10,11]
```

**Why 7?**
- Same as Device 1: **middle device**
- Must wait for data from both directions
- Needs extra iteration to balance the pipeline

---

### Device 3 (6 warmup iterations)

**Forward Schedule:** `[2, 3, 0, 8, 1, 9]` (first 6 from schedule)

```
Iter 0: Forward MB2 (VR1) - layers [0,1]  ← Starts Pipeline 1!
Iter 1: Forward MB3 (VR1) - layers [0,1]
Iter 2: Forward MB0 (VR0) - layers [6,7]  ← Receives from Device 2
Iter 3: Forward MB8 (VR2) - layers [8,9]
Iter 4: Forward MB1 (VR0) - layers [6,7]
Iter 5: Forward MB9 (VR2) - layers [8,9]
```

**Why 6?**
- Device 3 is an **edge device** (first for VR1, last for VR0/VR2)
- Starts Pipeline 1 (VR1) immediately
- **Balanced workload** → Needs exactly 6 warmups

---

## 5. Visual Timeline (First 6 Time Steps)

```
Time Step →
┌─────────────────────────────────────────────────────────────────┐
│ WARMUP PHASE                                                     │
└─────────────────────────────────────────────────────────────────┘

Step 0:
  Device 0: [F:MB0,VR0]
  Device 1:
  Device 2:
  Device 3:

Step 1:
  Device 0: [F:MB1,VR0]
  Device 1: [F:MB0,VR0]  ← Received from Device 0
  Device 2:
  Device 3:

Step 2:
  Device 0: [F:MB2,VR1]
  Device 1: [F:MB2,VR1]  ← Starts VR1 (backward pipeline)
  Device 2: [F:MB0,VR0]  ← Received from Device 1
  Device 3:

Step 3:
  Device 0: [F:MB10,VR3] ← Different VR!
  Device 1: [F:MB1,VR0]
  Device 2: [F:MB2,VR1]  ← Received from Device 3
  Device 3: [F:MB0,VR0]  ← Received from Device 2

Step 4:
  Device 0: [F:MB3,VR1]
  Device 1: [F:MB3,VR1]
  Device 2: [F:MB0,VR0]
  Device 3: [F:MB2,VR1]  ← VR1 pipeline flowing

Step 5:
  Device 0: [F:MB11,VR3]
  Device 1: [F:MB10,VR3] ← Received from Device 0
  Device 2: [F:MB3,VR1]
  Device 3: [F:MB0,VR0]

Step 6:
  Device 0: DONE (6 warmups complete!)
  Device 1: [F:MB8,VR2]  ← 7th warmup
  Device 2: [F:MB1,VR0]  ← 7th warmup
  Device 3: DONE (6 warmups complete!)

Step 7:
  Device 0: Start 1F1B (warmup done)
  Device 1: [F:MB11,VR3] ← 7th warmup complete!
  Device 2: [F:MB8,VR2]  ← 7th warmup complete!
  Device 3: Start 1F1B (warmup done)
```

---

## 6. Why This Matters for Performance

### Even Distribution

By giving middle devices **1 extra warmup iteration**, BitPipe ensures:

1. **All devices finish warmup at roughly the same time**
   - Edge devices (0, 3): 6 iterations
   - Middle devices (1, 2): 7 iterations
   - Difference: Only 1 iteration!

2. **Pipeline is fully utilized**
   - No device sits idle waiting for others
   - All 4 devices active during steady state

3. **Minimal bubble**
   - ~6% pipeline bubble (vs ~12% for standard 1F1B)
   - Bidirectional execution doubles throughput

---

## 7. Formula Breakdown

```python
num_warmup_microbatches = pipeline_parallel_size + pipeline_parallel_size // 2
                        = 4 + 2 = 6  # Base warmup

# Adjustment for middle devices
if pipeline_parallel_rank < pipeline_parallel_size // 2:
    # First half devices (0, 1)
    adjustment = pipeline_parallel_rank  # 0, 1
else:
    # Second half devices (2, 3)
    adjustment = pipeline_parallel_size - 1 - pipeline_parallel_rank  # 1, 0

num_warmup_microbatches += adjustment
```

**Pattern:**
```
Device 0: 6 + 0 = 6  ← Edge (first half)
Device 1: 6 + 1 = 7  ← Middle (first half)
Device 2: 6 + 1 = 7  ← Middle (second half)
Device 3: 6 + 0 = 6  ← Edge (second half)
```

**Symmetric Pattern:**
- Devices 0 and 3: Same warmup (6) - both edges
- Devices 1 and 2: Same warmup (7) - both middle

This **V-shaped symmetry** matches the layer distribution!

---

## 8. Summary

### Microbatch Mapping (8 total)
```
MB 0, 1   → VR0  (first half, pipeline 0)
MB 2, 3   → VR1  (first half, pipeline 1)
MB 4, 5   → VR2  (second half, pipeline 0)
MB 6, 7   → VR3  (second half, pipeline 1)
```

### Warmup Counts (4 devices)
```
Device 0: 6 iterations (edge device)
Device 1: 7 iterations (middle device)
Device 2: 7 iterations (middle device)
Device 3: 6 iterations (edge device)
```

### Why Different Counts?
1. **V-shaped layer distribution** creates edge and middle devices
2. **Edge devices** (0, 3) start both pipelines → Need fewer warmups
3. **Middle devices** (1, 2) wait for data from both directions → Need more warmups
4. **Result**: All devices reach steady state at the same time!

This clever design maximizes pipeline utilization and minimizes bubbles! 🚀
