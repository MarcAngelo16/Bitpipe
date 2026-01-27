# BitPipe Execution Trace - Concrete Example

This document traces a concrete example of BitPipe execution to show exactly how ranks coordinate.

**Configuration:**
- 4 devices (ranks 0, 1, 2, 3)
- 16 layers total
- 8 base microbatches → 16 total microbatches (doubled for bidirectional)
- 4 VRs per device

---

## Layer Distribution

```
Device 0: VR0[0,1]  VR1[6,7]  VR2[14,15] VR3[8,9]
Device 1: VR0[2,3]  VR1[4,5]  VR2[12,13] VR3[10,11]
Device 2: VR0[4,5]  VR1[2,3]  VR2[10,11] VR3[12,13]
Device 3: VR0[6,7]  VR1[0,1]  VR2[8,9]   VR3[14,15]
```

**VR Assignments:**
- VR0: Forward pipeline first quarter (Device 0→1→2→3)
- VR1: Forward pipeline second quarter (Device 3→2→1→0 - reversed)
- VR2: Backward pipeline first quarter (Device 3→2→1→0 - reversed, second half layers)
- VR3: Backward pipeline second quarter (Device 0→1→2→3, second half layers)

**Microbatch to VR Mapping:**
- MB 0,1,4,5 → VR0 (Pipeline 0, first quarter)
- MB 2,3,6,7 → VR1 (Pipeline 1, first quarter)
- MB 8,9,12,13 → VR2 (Pipeline 0, second quarter)
- MB 10,11,14,15 → VR3 (Pipeline 1, second quarter)

---

## Phase Calculations

**Per-Rank Warmup:**

```
Base: pipeline_parallel_size + pipeline_parallel_size // 2 = 4 + 2 = 6

Rank 0: 6 + 0 = 6 warmup microbatches
Rank 1: 6 + 1 = 7 warmup microbatches
Rank 2: 6 + (4-1-2) = 6 + 1 = 7 warmup microbatches
Rank 3: 6 + (4-1-3) = 6 + 0 = 6 warmup microbatches
```

**Phase Breakdown (Rank 0 example):**

```
total_num_microbatches = 16
n_loop = 16 // 4 // 2 - 1 = 1
num_warmup_microbatches = 6
unit_remaining = 2 * 4 - 6 = 2
num_microbatches_mid = 1 * 4 * 2 = 8
num_microbatches_remaining = 16 - 6 - 8 = 2
```

---

## Forward Schedules

```
Rank 0: [0,  1,  2, 10,  3, 11,  8,  9,  4,  5,  6, 14,  7, 15, 12, 13]
Rank 1: [0,  2,  1,  3, 10,  8, 11,  9,  4,  6,  5,  7, 14, 12, 15, 13]
Rank 2: [2,  0,  3,  1,  8, 10,  9, 11,  6,  4,  7,  5, 12, 14, 13, 15]
Rank 3: [2,  3,  0,  8,  1,  9, 10, 11,  6,  7,  4, 12,  5, 13, 14, 15]
```

**VR for each position (Rank 0 example):**

```
Position: 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
MB:       0   1   2  10   3  11   8   9   4   5   6  14   7  15  12  13
VR:      VR0 VR0 VR1 VR3 VR1 VR3 VR2 VR2 VR0 VR0 VR1 VR3 VR1 VR3 VR2 VR2
```

---

## Backward Schedules (with Sync Markers)

```
Rank 0: [8,  9, 10,  2, 11,  3,  0,  1, 12, 13, 14,  6, 15,  7,  4,  5, -1, -1]
Rank 1: [8, 10,  9, 11,  2,  0,  3,  1, 12, 14, 13, 15,  6,  4,  7,  5, -1, -1]
Rank 2: [10, 8, 11,  9,  0,  2,  1,  3, 14, 12, 15, 13,  4,  6,  5,  7, -1, -1]
Rank 3: [10,11,  8,  0,  9,  1,  2,  3, 14, 15, 12,  4, 13,  5,  6,  7, -1, -1]
```

**Note:** Each schedule has two `-1` sync markers at positions 16 and 17.

---

## Warmup Phase Trace

### Time Step 0: Pre-Receive (Before Loop)

**Operation:** Each rank receives its first input tensor.

| Rank | VR Set | Action | Receives From | Result |
|------|--------|--------|---------------|--------|
| 0 | VR0 | `recv_forward()` | None (first stage) | `input_tensors[VR0] = [None]` |
| 1 | VR0 | `recv_forward()` | Rank 0 | `input_tensors[VR0] = [tensor]` |
| 2 | VR1 | `recv_forward()` | Rank 3 | `input_tensors[VR1] = [tensor]` |
| 3 | VR1 | `recv_forward()` | None (first stage) | `input_tensors[VR1] = [None]` |

**Why different VRs?**
- Ranks 0-1 (first half): Start with VR0
- Ranks 2-3 (second half): Start with VR1

### Time Step 1: Warmup Iteration 0

**Forward Computation:**

| Rank | MB | VR | Input From | Compute | Output To | Next MB | Next VR |
|------|----|----|------------|---------|-----------|---------|---------|
| 0 | 0 | VR0 | None (generated) | Layers [0,1] | Rank 1 VR0 | 1 | VR0 |
| 1 | 0 | VR0 | Rank 0 VR0 | Layers [2,3] | Rank 2 VR0 | 2 | VR1 |
| 2 | 2 | VR1 | Rank 3 VR1 | Layers [2,3] | Rank 1 VR1 | 0 | VR0 |
| 3 | 2 | VR1 | None (generated) | Layers [0,1] | Rank 2 VR1 | 3 | VR1 |

**P2P Communication:**

| Rank | Send To | Recv From | Comm Type | Reason |
|------|---------|-----------|-----------|--------|
| 0 | Rank 1 | Rank 0 (self-loop) | `send_forward_recv_forward` | Same VR (VR0→VR0) |
| 1 | Rank 2 | Rank 3 | `send_forward_recv_forward_bd0` | Different VR (VR0→VR1) |
| 2 | Rank 1 | Rank 0 | `send_forward_recv_forward_bd0` | Different VR (VR1→VR0) |
| 3 | Rank 2 | Rank 3 (self-loop) | `send_forward_recv_forward` | Same VR (VR1→VR1) |

**Queue State After:**

```
Rank 0:
  input_tensors[VR0] = [None, from_rank_0_self]
  output_tensors[VR0] = [out_0]

Rank 1:
  input_tensors[VR0] = [from_rank_0, future]
  input_tensors[VR1] = [from_rank_3]
  output_tensors[VR0] = [out_0]

Rank 2:
  input_tensors[VR0] = [from_rank_0]
  input_tensors[VR1] = [from_rank_3, from_rank_1]
  output_tensors[VR1] = [out_2]

Rank 3:
  input_tensors[VR1] = [None, from_rank_3_self]
  output_tensors[VR1] = [out_2]
```

### Time Step 2: Warmup Iteration 1

**Forward Computation:**

| Rank | MB | VR | Input From | Compute | Output To | Next MB | Next VR |
|------|----|----|------------|---------|-----------|---------|---------|
| 0 | 1 | VR0 | Self (VR0) | Layers [0,1] | Rank 1 VR0 | 2 | VR1 |
| 1 | 2 | VR1 | Rank 2 VR1 | Layers [4,5] | Rank 0 VR1 | 1 | VR0 |
| 2 | 0 | VR0 | Rank 1 VR0 | Layers [4,5] | Rank 3 VR0 | 3 | VR1 |
| 3 | 3 | VR1 | Self (VR1) | Layers [0,1] | Rank 2 VR1 | 0 | VR0 |

**P2P Communication:**

| Rank | Send To | Recv From | Comm Type | Reason |
|------|---------|-----------|-----------|--------|
| 0 | Rank 1 | Rank 0 | `send_forward_recv_forward_bd0` | VR0→VR1 (bidirectional) |
| 1 | Rank 0 | Rank 2 | `send_forward_recv_forward_bd0` | VR1→VR0 (bidirectional) |
| 2 | Rank 3 | Rank 1 | `send_forward_recv_forward_bd0` | VR0→VR1 (bidirectional) |
| 3 | Rank 2 | Rank 3 | `send_forward_recv_forward_bd0` | VR1→VR0 (bidirectional) |

### VR Transition Example: Rank 1, Iteration 5

**Scenario:** MB 3 (VR1) → MB 11 (VR3)

```python
# Current forward output
output_tensor = forward_step_helper(MB_3, VR1, ...)  # Layers [4,5] output

# Next VR
next_forward_model_chunk_id = get_model_chunk_id(11)  # Returns VR3

# Check if same-device transition
if is_pipeline_first_stage(ignore_virtual=True) and next_chunk == VR3:
    # YES! VR1 last stage (rank 0 layers [0,1]) → VR3 first stage (rank 0 layers [8,9])
    # Same device, no communication
    detached_output_tensor = output_tensor.detach()
    detached_output_tensor.requires_grad_()
    input_tensor = detached_output_tensor  # SAME-DEVICE TRANSITION
else:
    # Normal P2P communication
    input_tensor = send_forward_recv_forward_bd0(...)

input_tensors[VR3].append(input_tensor)
```

**Key Insight:** VR1→VR3 transition on Device 0 requires NO network communication!

---

## 1F1B Phase Trace (Mid-Phase Loop, j=0)

### Time Step N: 1F1B Iteration 0

**Even Step (k=0): Forward**

| Rank | MB | VR | Action | Send | Recv |
|------|----|----|--------|------|------|
| 0 | 4 | VR0 | Forward layers [0,1] | Output to Rank 1 | Gradient from Rank 1 |
| 1 | 4 | VR0 | Forward layers [2,3] | Output to Rank 2 | Gradient from Rank 2 |
| 2 | 6 | VR1 | Forward layers [2,3] | Output to Rank 1 | Gradient from Rank 1 |
| 3 | 6 | VR1 | Forward layers [0,1] | Output to Rank 2 | Gradient from Rank 2 |

**Communication Pattern:** `send_forward_recv_backward`
- Send forward activation
- Receive backward gradient (from NEXT stage's backward pass)

**Odd Step (k=1): Backward**

| Rank | MB | VR | Action | Send | Recv |
|------|----|----|--------|------|------|
| 0 | 8 | VR2 | Backward layers [14,15] | Gradient to Rank 3 | Input from Rank 1 |
| 1 | 8 | VR2 | Backward layers [12,13] | Gradient to Rank 2 | Input from Rank 2 |
| 2 | 10 | VR3 | Backward layers [12,13] | Gradient to Rank 1 | Input from Rank 3 |
| 3 | 10 | VR3 | Backward layers [14,15] | Gradient to Rank 2 | Input from Rank 2 |

**Communication Pattern:** `send_backward_recv_forward`
- Send backward gradient
- Receive forward input (for NEXT forward pass)

**1F1B Invariant:**
```
At each time step:
- Half the ranks do forward, half do backward
- Each forward is paired with a backward from a PREVIOUS microbatch
- Queues remain balanced (forward appends, backward pops)
```

---

## Cooldown Phase Trace

### Gradient Sync Marker Processing

**First Sync Marker (backward_k = 16, microbatch_idx_b[16] = -1):**

| Rank | Action | Model Chunks Synced | BD Group |
|------|--------|---------------------|----------|
| 0 | `allreduce_gradients(model[2])` then `model[3]` | VR2, VR3 | Group 0 (with Rank 3) |
| 1 | `allreduce_gradients(model[2])` then `model[3]` | VR2, VR3 | Group 1 (with Rank 2) |
| 2 | `allreduce_gradients(model[3])` then `model[2]` | VR3, VR2 (reversed) | Group 1 (with Rank 1) |
| 3 | `allreduce_gradients(model[3])` then `model[2]` | VR3, VR2 (reversed) | Group 0 (with Rank 0) |

**Why reversed order for ranks 2-3?**
```python
offset = (
    range(num_model_chunks // 2)  # [0, 1] for ranks 0-1
    if pipeline_parallel_rank < pipeline_parallel_size // 2
    else reversed(range(num_model_chunks // 2))  # [1, 0] for ranks 2-3
)

# Ensures better load balancing during sync
```

**BD Group AllReduce:**
```
BD Group 0 (Rank 0 ↔ Rank 3):
- Both have gradients for layers [0,1] from VR0 and VR1
- Both have gradients for layers [8,9] from VR2 and VR3
- Need to average these gradients

BD Group 1 (Rank 1 ↔ Rank 2):
- Both have gradients for layers [2,3] from VR0 and VR1
- Both have gradients for layers [10,11] from VR2 and VR3
- Need to average these gradients
```

**Final Sync Marker (backward_k = 17, microbatch_idx_b[17] = -1):**

| Rank | Action | Model Chunks Synced |
|------|--------|---------------------|
| 0 | `allreduce_gradients(model[0])` then `model[1]` | VR0, VR1 |
| 1 | `allreduce_gradients(model[0])` then `model[1]` | VR0, VR1 |
| 2 | `allreduce_gradients(model[1])` then `model[0]` | VR1, VR0 |
| 3 | `allreduce_gradients(model[1])` then `model[0]` | VR1, VR0 |

**Result:** All gradients synchronized across BD groups, ready for optimizer step.

---

## Queue State Evolution Example (Rank 1, VR0)

### Initial State

```
input_tensors[VR0] = []
output_tensors[VR0] = []
output_tensor_grads[VR0] = []
```

### After Pre-Receive

```
input_tensors[VR0] = [from_rank_0]
output_tensors[VR0] = []
output_tensor_grads[VR0] = []
```

### After Warmup Iteration 0 (MB 0)

```
# Forward consumed input, produced output, received next input
input_tensors[VR0] = [from_rank_0_next]  # Old input consumed, new received
output_tensors[VR0] = [out_MB0]  # Stored for backward
output_tensor_grads[VR0] = []
```

### After Warmup Iteration 1 (MB 2, different VR)

```
# Did not touch VR0, still has queued state
input_tensors[VR0] = [from_rank_0_next]
output_tensors[VR0] = [out_MB0]
output_tensor_grads[VR0] = []
```

### After 1F1B Forward (MB 4)

```
# Forward consumed input, produced output, received gradient
input_tensors[VR0] = []  # Consumed
output_tensors[VR0] = [out_MB0, out_MB4]  # Added new output
output_tensor_grads[VR0] = [grad_from_next]  # Received for MB0
```

### After 1F1B Backward (MB 0)

```
# Backward consumed oldest forward, sent gradient, received input
input_tensors[VR0] = [from_rank_0_new]  # Received for next forward
output_tensors[VR0] = [out_MB4]  # MB0 consumed
output_tensor_grads[VR0] = []  # MB0 gradient consumed
```

**FIFO Invariant Maintained:**
- Forward appends to end: `output_tensors[VR0].append(out_MB4)`
- Backward pops from front: `output_tensors[VR0].pop(0)` → gets `out_MB0`
- Ensures correct pairing: backward for MB0 uses forward output from MB0

---

## Synchronization Guarantees

### Send-Recv Pairing

**Every send MUST have a matching receive:**

```
Time Step T:
  Rank A: send_forward(tensor) to Rank B
  Rank B: recv_forward() from Rank A

Both operations BLOCK until completed → implicit synchronization
```

**BitPipe uses combined operations:**

```
send_forward_recv_forward:
  - Send forward tensor to next rank
  - Receive forward tensor from previous rank
  - BOTH happen atomically (or with proper ordering)

send_backward_recv_forward:
  - Send backward gradient to previous rank
  - Receive forward tensor from previous rank
  - BOTH happen atomically
```

### Deadlock Prevention

**Pre-receive before warmup loop:**
- Ensures first rank can send (someone is ready to receive)
- Primes the pipeline

**Careful send/recv ordering in code:**
- Send operations happen before receive in many places
- Combined operations handle ordering internally

**recv_prev and recv_next flags:**
```python
recv_prev = True
if parallel_state.is_pipeline_first_stage():
    recv_prev = False  # First stage doesn't receive from previous

recv_next = True
if parallel_state.is_pipeline_last_stage():
    recv_next = False  # Last stage doesn't receive from next
```

**Prevents deadlock:** Don't wait for receives that will never come.

---

## Key Patterns for Chimera Adaptation

### 1. Queue Management (KEEP AS-IS)

```python
# Forward: append to end, retrieve from end
input_tensor = input_tensors[chunk_id][-1-offset]
output_tensors[chunk_id].append(output_tensor)

# Backward: pop from front (FIFO)
input_tensor = input_tensors[chunk_id].pop(0)
output_tensor = output_tensors[chunk_id].pop(0)
output_tensor_grad = output_tensor_grads[chunk_id].pop(0)
```

### 2. Pre-Receive Pattern (SIMPLIFY FOR CHIMERA)

```python
# BitPipe: Different VRs for different rank halves
if pipeline_parallel_rank < pipeline_parallel_size // 2:
    input_tensors[0].append(recv_forward())  # VR0
else:
    input_tensors[1].append(recv_forward())  # VR1

# Chimera: ALL ranks start with VR0 (forward direction)
input_tensors[0].append(recv_forward())
```

### 3. Same-Device Transition (REMOVE FOR CHIMERA)

```python
# BitPipe: Has same-device VR transitions
if (is_last_stage and next_chunk == VR2) or (is_first_stage and next_chunk == VR3):
    input_tensor = output_tensor.detach()  # NO COMMUNICATION

# Chimera: NO same-device transitions
# Always use P2P communication for VR0 ↔ VR1
input_tensor = send_forward_recv_forward_bd0(...)
```

### 4. Gradient Sync (SIMPLIFY FOR CHIMERA)

```python
# BitPipe: Two sync points (VR2/VR3 then VR0/VR1)
if backward_k == first_sync:
    allreduce_gradients(model[2])  # VR2
    allreduce_gradients(model[3])  # VR3
elif backward_k == final_sync:
    allreduce_gradients(model[0])  # VR0
    allreduce_gradients(model[1])  # VR1

# Chimera: One sync point (VR0 and VR1 together)
if backward_k == sync_marker:
    allreduce_gradients(model[0])  # VR0
    allreduce_gradients(model[1])  # VR1
```

### 5. Schedule Simplification (NEW FOR CHIMERA)

**BitPipe schedule complexity:**
- 4 VRs with complex interleaving
- V-shaped layer assignment complicates routing
- Same-device transitions require special handling

**Chimera opportunities:**
- 2 VRs with simpler interleaving
- Sequential layer assignment (no V-shape)
- All transitions are P2P (no special cases)

**Potential Chimera schedule:**
```
# Example: 4 devices, 8 base microbatches → 16 total

Forward schedule (all ranks identical pattern):
Rank 0: [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]
Rank 1: [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]
Rank 2: [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]
Rank 3: [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]

Pattern: Alternate VR0 (0-7) and VR1 (8-15)
```

This is MUCH simpler than BitPipe's per-rank custom schedules!

---

## Summary

**BitPipe's phase loop maintains these invariants:**

1. **FIFO queue ordering:** Forward appends, backward pops from front
2. **Paired send/recv:** Every send has a matching receive
3. **Pre-receive priming:** Pipeline starts with one tensor already received
4. **Inline receive:** After each step, receive next tensor
5. **Gradient sync markers:** Special `-1` entries trigger synchronization
6. **BD group sync:** Paired devices synchronize gradients together

**For Chimera, these can be simplified:**

1. **Keep FIFO queue pattern** (proven to work)
2. **Simplify pre-receive** (all ranks same VR)
3. **Remove same-device logic** (all P2P)
4. **Simplify sync** (one point, not two)
5. **Simplify schedules** (sequential, not V-shaped)

The core pattern is sound - just needs VR count reduction and simplification!
