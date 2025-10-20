# BitPipe Scheduling Deep Dive - Discussion Notes

**Date:** 2025-10-08
**Topic:** Understanding VR-to-Microbatch Assignment and Scheduling Function Hierarchy

---

## Table of Contents
1. [Key Discoveries](#key-discoveries)
2. [Layer Assignment Pattern](#layer-assignment-pattern)
3. [Microbatch ID and Real Data Mapping](#microbatch-id-and-real-data-mapping)
4. [VR Pairing and Pipeline Flow](#vr-pairing-and-pipeline-flow)
5. [Scheduling Function Hierarchy](#scheduling-function-hierarchy)
6. [Execution Timeline Example](#execution-timeline-example)
7. [Open Questions](#open-questions)

---

## Key Discoveries

### Discovery 1: Microbatch IDs are Labels, Not Data
**Critical Insight:** There are only **8 real data batches**, but they get **16 microbatch IDs** as they flow through the two halves of the model.

- **MB0 and MB8 are the SAME DATA**
  - MB0 is the ID when processing first half (layers 1-12)
  - MB8 is the ID when processing second half (layers 13-24)

- **MB2 and MB10 are the SAME DATA**
  - MB2 is the ID when processing first half (layers 1-12)
  - MB10 is the ID when processing second half (layers 13-24)

### Discovery 2: VR Pairing Determines Path Through Model
**Key Pattern:**
- **VR0 + VR1** = Together process layers 1-12 (first half)
- **VR2 + VR3** = Together process layers 13-24 (second half)

### Discovery 3: Bidirectional Means Opposite Device Order
- **VR0** flows: Device 0 → Device 1 → Device 2 → Device 3
- **VR1** flows: Device 3 → Device 2 → Device 1 → Device 0 (REVERSE!)
- **VR2** flows: Device 3 → Device 2 → Device 1 → Device 0 (REVERSE!)
- **VR3** flows: Device 0 → Device 1 → Device 2 → Device 3

This allows **2x concurrent execution** through the same layers!

---

## Layer Assignment Pattern

From actual BitPipe logs (4 devices, 24 layers):

```
Device 0:
  VR0: [1, 2, 3]      ← First chunk of first half
  VR1: [10, 11, 12]   ← Last chunk of first half
  VR2: [22, 23, 24]   ← Last chunk of second half
  VR3: [13, 14, 15]   ← First chunk of second half
  Total: 12 layers

Device 1:
  VR0: [4, 5, 6]
  VR1: [7, 8, 9]
  VR2: [19, 20, 21]
  VR3: [16, 17, 18]
  Total: 12 layers

Device 2:
  VR0: [7, 8, 9]      ← SWAPPED with Device 1's VR1!
  VR1: [4, 5, 6]      ← SWAPPED with Device 1's VR0!
  VR2: [16, 17, 18]   ← SWAPPED with Device 1's VR3!
  VR3: [19, 20, 21]   ← SWAPPED with Device 1's VR2!
  Total: 12 layers

Device 3:
  VR0: [10, 11, 12]   ← SWAPPED with Device 0's VR1!
  VR1: [1, 2, 3]      ← SWAPPED with Device 0's VR0!
  VR2: [13, 14, 15]   ← SWAPPED with Device 0's VR3!
  VR3: [22, 23, 24]   ← SWAPPED with Device 0's VR2!
  Total: 12 layers
```

### Device Pairing Pattern
- **Device 0 ↔ Device 3** (paired)
- **Device 1 ↔ Device 2** (paired)

**Rule:** Paired devices swap VR0↔VR1 and VR2↔VR3 layer assignments.

---

## Microbatch ID and Real Data Mapping

### Configuration
- **Base microbatches:** 8
- **Total microbatches (BitPipe doubled):** 16
- **Real data batches:** 8

### Microbatch Assignment by VR

From `get_model_chunk_id()` function:
```python
# Formula:
microbatch_id_in_group = microbatch_id % pipeline_parallel_size
chunk_offset = 0 if microbatch_id < (total_num_microbatches // 2) else 2
model_chunk_id = (microbatch_id_in_group // (pipeline_parallel_size // 2)) + chunk_offset
```

**Result for 4 devices, 16 total microbatches:**
```
VR0 handles: [0, 1, 4, 5]     (first half IDs)
VR1 handles: [2, 3, 6, 7]     (first half IDs)
VR2 handles: [8, 9, 12, 13]   (second half IDs)
VR3 handles: [10, 11, 14, 15] (second half IDs)
```

### Real Data to Microbatch ID Mapping

**8 Real Data Batches → 16 Microbatch IDs:**

| Real Data Batch | First Half ID | First Half VR | Second Half ID | Second Half VR | Pipeline |
|-----------------|---------------|---------------|----------------|----------------|----------|
| 0               | MB0           | VR0           | MB8            | VR2            | Pipeline 0 |
| 1               | MB2           | VR1           | MB10           | VR3            | Pipeline 1 |
| 2               | MB1           | VR0           | MB9            | VR2            | Pipeline 0 |
| 3               | MB3           | VR1           | MB11           | VR3            | Pipeline 1 |
| 4               | MB4           | VR0           | MB12           | VR2            | Pipeline 0 |
| 5               | MB6           | VR1           | MB14           | VR3            | Pipeline 1 |
| 6               | MB5           | VR0           | MB13           | VR2            | Pipeline 0 |
| 7               | MB7           | VR1           | MB15           | VR3            | Pipeline 1 |

**Key Insight:** The transition from first-half ID to second-half ID happens when the data finishes the first 12 layers and begins processing layers 13-24.

---

## VR Pairing and Pipeline Flow

### Pipeline 0 (VR0 → VR2 path)

**Real Data Batch 0 Complete Journey:**

#### Forward Pass - First Half (MB0, VR0)
```
Layers 1-12 processing:
D0.VR0[1,2,3] → D1.VR0[4,5,6] → D2.VR0[7,8,9] → D3.VR0[10,11,12]
```

#### Forward Pass - Second Half (MB8, VR2)
```
Layers 13-24 processing:
D3.VR2[13,14,15] → D2.VR2[16,17,18] → D1.VR2[19,20,21] → D0.VR2[22,23,24]
                   ↑
           (Starts from opposite end!)
```

#### Backward Pass - Second Half (MB8, VR2)
```
Gradients flow backward through layers 24→13:
D0.VR2[24→22] → D1.VR2[21→19] → D2.VR2[18→16] → D3.VR2[15→13]
```

#### Backward Pass - First Half (MB0, VR0)
```
Gradients flow backward through layers 12→1:
D3.VR0[12→10] → D2.VR0[9→7] → D1.VR0[6→4] → D0.VR0[3→1]
```

### Pipeline 1 (VR1 → VR3 path)

**Real Data Batch 1 Complete Journey:**

#### Forward Pass - First Half (MB2, VR1)
```
Layers 1-12 processing (REVERSE device order):
D3.VR1[1,2,3] → D2.VR1[4,5,6] → D1.VR1[7,8,9] → D0.VR1[10,11,12]
```

#### Forward Pass - Second Half (MB10, VR3)
```
Layers 13-24 processing:
D0.VR3[13,14,15] → D1.VR3[16,17,18] → D2.VR3[19,20,21] → D3.VR3[22,23,24]
```

#### Backward Pass - Second Half (MB10, VR3)
```
Gradients flow backward through layers 24→13:
D3.VR3[24→22] → D2.VR3[21→19] → D1.VR3[18→16] → D0.VR3[15→13]
```

#### Backward Pass - First Half (MB2, VR1)
```
Gradients flow backward through layers 12→1:
D0.VR1[12→10] → D1.VR1[9→7] → D2.VR1[6→4] → D3.VR1[3→1]
```

### The Bidirectional Magic

**For First Half (Layers 1-12):**
```
Pipeline 0 (MB0): D0 → D1 → D2 → D3  (using VR0)
Pipeline 1 (MB2): D3 → D2 → D1 → D0  (using VR1)  ← SIMULTANEOUS!
                  ↑
          Same layers, opposite direction!
```

**For Second Half (Layers 13-24):**
```
Pipeline 0 (MB8):  D3 → D2 → D1 → D0  (using VR2)
Pipeline 1 (MB10): D0 → D1 → D2 → D3  (using VR3)  ← SIMULTANEOUS!
                   ↑
          Same layers, opposite direction!
```

**Result:** 2x concurrent microbatch execution through the same model layers!

---

## Scheduling Function Hierarchy

### Function Call Flow

```
forward_backward_pipelining_with_BitPipe()
│
├─ SETUP PHASE (Called ONCE at start)
│  │
│  ├─ get_microbatch(total_num_microbatches)  [Line 238]
│  │  │
│  │  │  Purpose: Groups microbatch IDs by VR assignment
│  │  │
│  │  │  Returns: microbatch_id01 = [
│  │  │    [0, 1, 4, 5],      # VR0's microbatches
│  │  │    [2, 3, 6, 7],      # VR1's microbatches
│  │  │    [8, 9, 12, 13],    # VR2's microbatches
│  │  │    [10, 11, 14, 15]   # VR3's microbatches
│  │  │  ]
│  │  │
│  │  └─ Calls: get_model_chunk_id(microbatch_id) for each ID [Line 241]
│  │
│  ├─ get_microbatch_idx(total_num_microbatches, pipeline_parallel_rank)  [Line 510]
│  │  │
│  │  │  Purpose: Generates FORWARD execution schedule for this device
│  │  │
│  │  │  Example output for Device 0:
│  │  │  [0, 1, 2, 10, 3, 11, 8, 9, 4, 5, 6, 14, 7, 15, 12, 13]
│  │  │
│  │  │  This is the ORDER in which forward passes execute on Device 0
│  │  │
│  │  └─ Calls: get_microbatch(total_num_microbatches) [Line 252]
│  │
│  └─ get_bkmicrobatch_idx(total_num_microbatches, pipeline_parallel_rank)  [Line 511]
│     │
│     │  Purpose: Generates BACKWARD execution schedule for this device
│     │
│     │  Example output for Device 0:
│     │  [8, 9, 10, 2, 11, 3, 0, 1, 12, 13, 14, 6, 15, 7, 4, -1, 5, -1]
│     │                                                        ↑       ↑
│     │                                              Gradient sync points
│     │
│     │  This is the ORDER in which backward passes execute on Device 0
│     │
│     └─ Calls: get_microbatch(total_num_microbatches) [Line 294]
│
│
├─ WARMUP PHASE (Fill the pipeline with forward passes)  [Lines 518-578]
│  │
│  │  Loop: for k in range(num_warmup_microbatches):
│  │
│  ├─ get_model_chunk_id(microbatch_idx[k])  [Line 519]
│  │  │
│  │  │  Purpose: Determine which VR to use for microbatch_idx[k]
│  │  │
│  │  │  Example: If microbatch_idx[k] = 0, returns VR0
│  │  │           If microbatch_idx[k] = 8, returns VR2
│  │  │
│  │  └─ Result: Tells device which model chunk (VR) to execute
│  │
│  ├─ set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)  [Line 520]
│  │  │
│  │  │  Purpose: Switch context to the correct VR
│  │  │
│  │  └─ This makes model[VR] active for the next forward pass
│  │
│  ├─ forward_step_helper(microbatch_idx[k], ...)  [Line 521]
│  │  │
│  │  │  Purpose: Execute forward pass for this microbatch
│  │  │
│  │  │  Steps inside:
│  │  │  1. get_model_chunk_id(microbatch_id)  [Line 380]
│  │  │     → Determines VR (redundant check for safety)
│  │  │
│  │  │  2. pipeline_id = 0 if model_chunk_id in [0, 2] else 1  [Line 384]
│  │  │     → VR0/VR2 = Pipeline 0, VR1/VR3 = Pipeline 1
│  │  │
│  │  │  3. forward_step(model[model_chunk_id], input_tensor, ...)  [Line 397]
│  │  │     → Actual neural network forward computation
│  │  │
│  │  │  4. Returns output_tensor
│  │  │
│  │  └─ Output stored in: output_tensors[model_chunk_id]
│  │
│  └─ P2P Communication (send output to next stage, recv input from prev stage)
│     │
│     ├─ If same VR on next device: send_forward_recv_forward()
│     └─ If different VR on next device: send_forward_recv_forward_bd0()
│
│
├─ STEADY STATE PHASE (Interleaved 1F1B)  [Lines 585-676]
│  │
│  │  Loop: for j in range(n_loop):
│  │    Loop: for k in range(unit_remaining*2):
│  │
│  ├─ if k%2 == 0: FORWARD PASS
│  │  │
│  │  │  Calculate: forward_k = k//2 + num_warmup_microbatches + offset
│  │  │
│  │  ├─ get_model_chunk_id(microbatch_idx[forward_k])  [Line 593]
│  │  │
│  │  ├─ set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)  [Line 594]
│  │  │
│  │  ├─ forward_step_helper(microbatch_idx[forward_k], ...)  [Line 595]
│  │  │
│  │  └─ P2P: send_forward_recv_backward()  [Line 602]
│  │     → Send forward output, receive backward gradient
│  │
│  └─ else: BACKWARD PASS
│     │
│     │  Calculate: backward_k = k//2 + offset
│     │
│     ├─ get_model_chunk_id(microbatch_idx_b[backward_k])  [Line 615]
│     │
│     ├─ set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)  [Line 616]
│     │
│     ├─ backward_step_helper(microbatch_idx_b[backward_k])  [Line 619]
│     │  │
│     │  │  Steps inside:
│     │  │  1. get_model_chunk_id(microbatch_id)  [Line 449]
│     │  │
│     │  │  2. pipeline_id = 0 if model_chunk_id in [0, 2] else 1
│     │  │
│     │  │  3. backward_step(input_tensor_grad, ...)
│     │  │     → Actual neural network backward computation
│     │  │
│     │  │  4. Returns input_tensor_grad
│     │  │
│     │  └─ Gradient stored for this VR
│     │
│     └─ P2P: send_backward_recv_forward() or send_backward_recv_backward_bd()
│
│
└─ COOLDOWN PHASE (Drain remaining backward passes)  [Lines 677+]
   │
   │  Similar to steady state, but only backward passes
   │
   ├─ Process remaining backward passes from microbatch_idx_b
   │
   └─ When microbatch_id == -1: GRADIENT SYNCHRONIZATION
      │
      └─ allreduce_gradients(model[chunk])  [Line 416]
         │
         │  Purpose: All-reduce gradients across data parallel ranks
         │
         │  Synchronizes gradients for:
         │  - VR2 and VR3 first (eager sync)
         │  - Then VR0 and VR1 (final sync)
         │
         └─ Uses bidirectional parallel group for communication
```

### Key Scheduling Functions

#### 1. `get_model_chunk_id(microbatch_id)` [Line 342]
```python
def get_model_chunk_id(microbatch_id):
    microbatch_id_in_group = microbatch_id % pipeline_parallel_size
    chunk_offset = 0 if microbatch_id < (total_num_microbatches // 2) else 2
    model_chunk_id = microbatch_id_in_group // (pipeline_parallel_size // 2)
    model_chunk_id += chunk_offset
    return model_chunk_id
```

**Purpose:** Maps microbatch ID to VR (0-3)

**Example:**
- MB0: 0 % 4 = 0, offset=0, 0//2 = 0, result = **VR0**
- MB2: 2 % 4 = 2, offset=0, 2//2 = 1, result = **VR1**
- MB8: 8 % 4 = 0, offset=2, 0//2 = 0, result = **VR2**
- MB10: 10 % 4 = 2, offset=2, 2//2 = 1, result = **VR3**

#### 2. `get_microbatch(total_num_microbatches)` [Line 238]
```python
def get_microbatch(total_num_microbatches):
    microbatch_id01 = [[] for _ in range(num_model_chunks)]
    for i in range(total_num_microbatches):
        model_chunk_id = get_model_chunk_id(i)
        microbatch_id01[model_chunk_id].append(i)
    return microbatch_id01
```

**Purpose:** Groups all microbatch IDs by their VR assignment

**Output:**
```python
microbatch_id01 = [
    [0, 1, 4, 5],      # VR0's microbatches
    [2, 3, 6, 7],      # VR1's microbatches
    [8, 9, 12, 13],    # VR2's microbatches
    [10, 11, 14, 15]   # VR3's microbatches
]
```

#### 3. `get_microbatch_idx(total_num_microbatches, pipeline_parallel_rank)` [Line 245]

**Purpose:** Generates device-specific forward execution schedule

**Algorithm:** Complex interleaving pattern based on:
- `i_half`: Which half of devices (0-1 or 2-3)
- `num_initial`: How many microbatches to process before interleaving
- Loops through VR assignments in a V-shaped pattern

**Output Examples:**
```
Device 0: [0, 1, 2, 10, 3, 11, 8, 9, 4, 5, 6, 14, 7, 15, 12, 13]
Device 1: [0, 2, 1, 3, 10, 8, 11, 9, 4, 6, 5, 7, 14, 12, 15, 13]
Device 2: [2, 0, 3, 1, 8, 10, 9, 11, 6, 4, 7, 5, 12, 14, 13, 15]
Device 3: [2, 3, 0, 8, 1, 9, 10, 11, 6, 7, 4, 12, 5, 13, 14, 15]
```

#### 4. `get_bkmicrobatch_idx(total_num_microbatches, pipeline_parallel_rank)` [Line 292]

**Purpose:** Generates device-specific backward execution schedule

**Special Features:**
- Inserts `-1` for gradient synchronization points
- Two sync points per device:
  - "Eager sync" after processing most microbatches
  - "Final sync" at the very end

**Output Examples:**
```
Device 0: [8, 9, 10, 2, 11, 3, 0, 1, 12, 13, 14, 6, 15, 7, 4, -1, 5, -1]
Device 1: [8, 10, 9, 11, 2, 0, 3, 1, 12, 14, 13, 15, 6, 4, 7, 5, -1, -1]
Device 2: [10, 8, 11, 9, 0, 2, 1, 3, 14, 12, 15, 13, 4, 6, 5, 7, -1, -1]
Device 3: [10, 11, 8, 0, 9, 1, 2, 3, 14, 15, 12, 4, 13, 5, 6, -1, 7, -1]
```

---

## Execution Timeline Example

Let's trace **Real Data Batch 0** through the entire pipeline on **Device 0**:

### Device 0 Forward Schedule
```
[0, 1, 2, 10, 3, 11, 8, 9, 4, 5, 6, 14, 7, 15, 12, 13]
 ↑              ↑
MB0            MB8 (same data!)
```

### Device 0 Backward Schedule
```
[8, 9, 10, 2, 11, 3, 0, 1, 12, 13, 14, 6, 15, 7, 4, -1, 5, -1]
 ↑                    ↑
MB8                  MB0 (same data!)
```

### Timeline

| Time Step | Phase | Microbatch ID | VR  | Real Data | Layers Processing | Device Flow |
|-----------|-------|---------------|-----|-----------|-------------------|-------------|
| 0         | FWD   | MB0           | VR0 | Batch 0   | [1, 2, 3]         | D0.VR0 |
| ...       | FWD   | ...           | ... | ...       | ...               | ... |
| (MB0 moves to D1, D2, D3 through their VR0 chunks)                              |||||
| 6         | FWD   | MB8           | VR2 | Batch 0   | [22, 23, 24]      | D0.VR2 |
| ...       | ...   | ...           | ... | ...       | ...               | ... |
| (MB8 already started on D3, D2, D1, now arrives at D0)                          |||||
| 15        | BWD   | MB8           | VR2 | Batch 0   | [24→22] grads     | D0.VR2 |
| ...       | BWD   | ...           | ... | ...       | ...               | ... |
| (MB8 backward continues through D1, D2, D3)                                     |||||
| 22        | BWD   | MB0           | VR0 | Batch 0   | [3→1] grads       | D0.VR0 |
| ...       | ...   | ...           | ... | ...       | ...               | ... |

**Complete Path for Real Data Batch 0:**
1. Forward through layers 1-12 (as MB0, using VR0)
2. Forward through layers 13-24 (as MB8, using VR2)
3. Backward through layers 24-13 (as MB8, using VR2)
4. Backward through layers 12-1 (as MB0, using VR0)

### The Transition Point

**When does MB0 become MB8?**

The transition is **implicit** in the schedule:
- When Device 3 finishes processing MB0 through its VR0 chunk [10,11,12]
- The output tensor is sent to the "next stage" which is Device 3's VR2 chunk [13,14,15]
- At this point, the data is now identified as MB8
- The same physical data continues processing, but with a new ID and different VR

**Code Location:** Lines 544-576 in `bitpipe_schedule.py`
```python
if forward_model_chunk_id == next_forward_model_chunk_id:
    # Same VR, stay in same pipeline
    input_tensor = send_forward_recv_forward(...)
else:
    # Different VR, transition to next part of model
    if (parallel_state.is_pipeline_last_stage(ignore_virtual=True) and
        next_forward_model_chunk_id == v_size-2):
        # Same device transition (e.g., VR0→VR2 on Device 3)
        detached_output_tensor = output_tensor.detach()
        detached_output_tensor.requires_grad_()
        input_tensor = detached_output_tensor
    else:
        # Cross-device transition
        input_tensor = send_forward_recv_forward_bd0(...)
```

---

## Open Questions

### Question 1: Detailed Interleaving Logic
**What we understand:** The schedules are generated by `get_microbatch_idx()` and `get_bkmicrobatch_idx()`

**What's unclear:** The exact mathematical pattern of how the interleaving works
- Why does Device 0's schedule start with `[0, 1, 2, 10, ...]`?
- What determines the specific order?
- How does `num_initial` and `i_half` create the V-shaped pattern?

**Next steps:**
- Trace through the nested loops in `get_microbatch_idx()` [Lines 267-280]
- Understand the index calculations: `j*num_unit+i`, `i_half`, `1-i_half`, `3-i_half`

### Question 2: Gradient Sync Timing
**What we understand:** `-1` markers indicate gradient sync points

**What's unclear:**
- Why different devices have sync markers at different positions?
- Device 0: `[..., 4, -1, 5, -1]` (sync between last two microbatches)
- Device 1: `[..., 5, -1, -1]` (sync after last microbatch)
- What determines this pattern?

**Code location:** Lines 324-328
```python
if pipeline_parallel_rank == pipeline_parallel_size // 2 or \
   pipeline_parallel_rank == pipeline_parallel_size // 2 - 1:
    microbatch_idx.append(-1)
else:
    microbatch_idx.insert(-1, -1)
microbatch_idx.append(-1)
```

### Question 3: Communication Pattern Details
**What we understand:** Different P2P functions for same-VR vs. different-VR transitions

**What's unclear:**
- When exactly is `send_forward_recv_forward_bd0()` used vs. `send_forward_recv_forward()`?
- What's the difference between them?
- How does the "bidirectional" communication work?

**Next steps:** Examine `p2p_communication.py`

### Question 4: Warmup/Steady/Cooldown Phase Boundaries
**What we understand:** Three execution phases with different microbatch processing patterns

**What's unclear:**
- Exact calculation of `num_warmup_microbatches` [Lines 207-216]
- Why does it depend on `pipeline_parallel_rank`?
- How does `n_loop` and `unit_remaining` work?

**Formulas to investigate:**
```python
n_loop = total_num_microbatches // pipeline_parallel_size // 2 - 1
num_warmup_microbatches = pipeline_parallel_size + pipeline_parallel_size//2
num_warmup_microbatches += (
    pipeline_parallel_rank if pipeline_parallel_rank < pipeline_parallel_size // 2
    else pipeline_parallel_size - 1 - pipeline_parallel_rank
)
unit_remaining = 2*pipeline_parallel_size - num_warmup_microbatches
num_microbatches_mid = n_loop * pipeline_parallel_size * 2
num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches - num_microbatches_mid
```

---

## Summary of Understanding

### What We NOW Understand Clearly ✅

1. **Microbatch IDs are labels**: 8 real data batches get 16 IDs (first-half ID + second-half ID)

2. **VR pairing**:
   - VR0 + VR1 = First half of model (layers 1-12)
   - VR2 + VR3 = Second half of model (layers 13-24)

3. **Layer assignment follows V-shaped pattern**: Paired devices swap VR assignments

4. **Bidirectional execution**: Two pipelines run simultaneously through same layers in opposite directions

5. **Data transition**: MB0→MB8 is the same data transitioning from first half to second half

6. **Function hierarchy**: Setup phase generates schedules once, execution phase loops through them

7. **Schedule generation**: Device-specific forward and backward schedules created by `get_microbatch_idx()` and `get_bkmicrobatch_idx()`

### What We Want to Explore Next 🔍

1. Detailed interleaving algorithm math
2. Gradient synchronization timing logic
3. P2P communication patterns
4. Phase boundary calculations
5. Performance implications of the V-shaped pattern

---

## Related Files

- **Main scheduler:** `megatron/core/pipeline_parallel/bitpipe_schedule.py`
- **P2P communication:** `megatron/core/pipeline_parallel/p2p_communication.py`
- **Layer distribution:** `megatron/model/transformer.py`
- **Simulation tool:** `asymmetric_bitpipe/scripts/simulation_microbatch.py`
- **Profiling:** `megatron/core/pipeline_parallel/bitpipe_profiler.py`

---

## Simulation Results

Running `python asymmetric_bitpipe/scripts/simulation_microbatch.py`:

### VR Assignment
```
VR0 (Forward Pipeline 1st half): [0, 1, 4, 5]
VR1 (Forward Pipeline 1st half): [2, 3, 6, 7]
VR2 (Backward Pipeline 2nd half): [8, 9, 12, 13]
VR3 (Backward Pipeline 2nd half): [10, 11, 14, 15]
```

### Device 0 Schedule
```
Forward:  [0, 1, 2, 10, 3, 11, 8, 9, 4, 5, 6, 14, 7, 15, 12, 13]
Backward: [8, 9, 10, 2, 11, 3, 0, 1, 12, 13, 14, 6, 15, 7, 4, -1, 5, -1]
```

### Device 1 Schedule
```
Forward:  [0, 2, 1, 3, 10, 8, 11, 9, 4, 6, 5, 7, 14, 12, 15, 13]
Backward: [8, 10, 9, 11, 2, 0, 3, 1, 12, 14, 13, 15, 6, 4, 7, 5, -1, -1]
```

### Device 2 Schedule
```
Forward:  [2, 0, 3, 1, 8, 10, 9, 11, 6, 4, 7, 5, 12, 14, 13, 15]
Backward: [10, 8, 11, 9, 0, 2, 1, 3, 14, 12, 15, 13, 4, 6, 5, 7, -1, -1]
```

### Device 3 Schedule
```
Forward:  [2, 3, 0, 8, 1, 9, 10, 11, 6, 7, 4, 12, 5, 13, 14, 15]
Backward: [10, 11, 8, 0, 9, 1, 2, 3, 14, 15, 12, 4, 13, 5, 6, -1, 7, -1]
```

---

**End of Notes**
