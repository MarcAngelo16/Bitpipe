# BitPipe Scheduling: Complete Architecture and Flow

## Table of Contents
1. [Overview](#overview)
2. [How get_forward_backward_func() Works](#how-get_forward_backward_func-works)
3. [BitPipe 4-VR Complete Scheduling Flow](#bitpipe-4vr-complete-scheduling-flow)
4. [Microbatch-to-VR Mapping](#microbatch-to-vr-mapping)
5. [Three Communication Scenarios](#three-communication-scenarios)
6. [Scenario 1: Same VR (Normal Pipeline Flow)](#scenario-1-same-vr-normal-pipeline-flow)
7. [Scenario 2: Same-Device VR Transition](#scenario-2-same-device-vr-transition)
8. [Scenario 3: Cross-Device Bidirectional Transition](#scenario-3-cross-device-bidirectional-transition)
9. [Complete Warmup Timeline](#complete-warmup-timeline)
10. [recv_forward() vs schedules.py](#recv_forward-vs-schedulespy)

---

## Overview

BitPipe is a bidirectional interleaved pipeline parallelism scheduler that:
- Divides the model into **4 Virtual Ranks (VR)** per device
- Creates **two bidirectional pipelines** running simultaneously
- Doubles effective microbatches without doubling memory
- Achieves ~6% pipeline bubble (vs ~12% for standard 1F1B)

**Key constraint:** Requires minimum 4 GPUs with even number of devices (4, 6, 8, 10, 12...)

---

## How get_forward_backward_func() Works

### The Selection Function

**Location:** `megatron/core/pipeline_parallel/schedules.py:21-119`

```python
def get_forward_backward_func():
    """
    Retrieves the appropriate forward_backward function based on configuration.

    Returns a function object (not a result, but the function itself!)
    """

    # Priority 1: Chimera 2-VR (higher priority)
    if hasattr(get_args(), 'enable_chimera_schedule') and get_args().enable_chimera_schedule:
        from megatron.core.pipeline_parallel.schedule_impl.chimera import (
            forward_backward_pipelining_with_chimera_2vr,
        )
        return forward_backward_pipelining_with_chimera_2vr  # ← Returns function object

    # Priority 2: BitPipe 4-VR
    if get_args().enable_bitpipe_schedule:
        from megatron.core.pipeline_parallel.schedule_impl.bitpipe import (
            forward_backward_pipelining_with_bitpipe_4vr,
        )
        return forward_backward_pipelining_with_bitpipe_4vr  # ← Returns function object

    # Priority 3-5: Standard pipelines
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    if pipeline_model_parallel_size > 1:
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining

    return forward_backward_func
```

### Call Flow in Training

```
Training Loop:
    │
    ├─ ONE TIME at start:
    │  forward_backward_func = get_forward_backward_func()
    │
    └─ EVERY ITERATION:
       for iteration in range(num_iterations):
           loss = forward_backward_func(
               forward_step_func=forward_step,
               data_iterator=[iter0, iter1, iter2, iter3],  ← 4 iterators
               model=[chunk0, chunk1, chunk2, chunk3],      ← 4 model chunks
               num_microbatches=8,
               seq_length=2048,
               micro_batch_size=32,
               ...
           )
```

**Key Point:** `get_forward_backward_func()` is called **ONCE** to select the scheduler function, then that function is called **repeatedly** during training.

---

## BitPipe 4-VR Complete Scheduling Flow

### Phase 1: Initialization

**Lines 94-243 in bitpipe_4vr.py**

```python
def forward_backward_pipelining_with_bitpipe_4vr(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches=8,
    seq_length=2048,
    micro_batch_size=32,
    ...
):
    # Step 1: Get configuration
    config = get_model_config(model[0])

    # Step 2: Pipeline info
    pipeline_parallel_size = 4          # Example: 4 devices
    pipeline_parallel_rank = 0,1,2,3    # Current device

    # Step 3: Calculate effective microbatches
    num_model_chunks = 4                          # BitPipe always 4 chunks
    total_num_microbatches = 8 * (4//2) = 16     # Double the microbatches!

    # Step 4: Create tensor shape (ONCE for entire iteration)
    tensor_shape = (2048, 32, 4096)   # seq_len, batch_size, hidden_size

    # This tensor_shape is a SINGLE TUPLE, not a list
    # Used for ALL recv_forward() calls in this iteration
```

### Phase 2: Setup Phase (Lines 514-532)

Initial data loading before warmup loop:

```python
# Lines 515-524: Load initial tensor based on device rank
if pipeline_parallel_rank < pipeline_parallel_size // 2:  # Ranks 0,1
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(_profile_p2p_comm(
        p2p_communication.recv_forward,    # ← LOW-LEVEL API (NOT wrapper!)
        "recv_forward",
        tensor_shape,                      # ← SINGLE TUPLE
        config
    ))
else:  # Ranks 2,3
    parallel_state.set_virtual_pipeline_model_parallel_rank(1)
    input_tensors[1].append(_profile_p2p_comm(
        p2p_communication.recv_forward,    # ← LOW-LEVEL API
        "recv_forward",
        tensor_shape,                      # ← SINGLE TUPLE
        config
    ))
```

**What this does:**
- Devices 0,1 prepare to receive data for VR0 (first quarter of model)
- Devices 2,3 prepare to receive data for VR1 (second quarter of model)
- Uses `p2p_communication.recv_forward()` directly (NOT the wrapper from schedules.py!)
- Gets back a **single tensor**, not a list

### Phase 3: Warmup Loop (Lines 533-596)

Fill the pipeline with forward passes:

```python
for k in range(num_warmup_microbatches):
    # Get which VR this microbatch uses
    forward_model_chunk_id = get_model_chunk_id(microbatch_idx[k])
    parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)

    # Execute forward pass
    output_tensor = forward_step_helper(microbatch_idx[k], None, 0)

    # Determine next microbatch's VR
    next_forward_model_chunk_id = get_model_chunk_id(microbatch_idx[k+1])

    # Route based on VR transition (see Scenarios below)
    recv_prev = True
    if parallel_state.is_pipeline_first_stage():
        recv_prev = False

    if forward_model_chunk_id == next_forward_model_chunk_id:
        # Scenario 1: Same VR (normal pipeline)
        input_tensor = _profile_p2p_comm(
            p2p_communication.send_forward_recv_forward,
            "send_forward_recv_forward",
            output_tensor,
            recv_prev=recv_prev,
            tensor_shape=tensor_shape,
            config=config,
        )
    else:
        # Scenario 2 or 3: Different VR
        if (is_last_stage and next == VR2) or (is_first_stage and next == VR3):
            # Scenario 2: Same-device transition
            detached_output_tensor = output_tensor.detach()
            detached_output_tensor.requires_grad_()
            input_tensor = detached_output_tensor  # No network!
        else:
            # Scenario 3: Cross-device bidirectional
            input_tensor = _profile_p2p_comm(
                p2p_communication.send_forward_recv_forward_bd0,
                "send_forward_recv_forward_bd0",
                output_tensor,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                config=config,
            )

    input_tensors[next_forward_model_chunk_id].append(input_tensor)
```

### Phase 4: Steady State (Lines 597-980)

Interleaved 1F1B (one forward, one backward):

```python
for j in range(n_loop):
    for k in range(unit_remaining*2):
        if k%2 == 0:
            # Forward pass
            forward_model_chunk_id = get_model_chunk_id(...)
            output_tensor = forward_step_helper(...)
            # Communication similar to warmup
        else:
            # Backward pass
            backward_model_chunk_id = get_model_chunk_id(...)
            input_tensor_grad = backward_step_helper(...)
            # Gradient communication
```

### Phase 5: Cooldown (Lines 981-1000)

Complete remaining backward passes with eager gradient sync:

```python
for k in range(num_cooldown_microbatches):
    backward_model_chunk_id = get_model_chunk_id(...)
    output_tensor_grad = recv_backward(tensor_shape, config)
    input_tensor_grad = backward_step_helper(...)
    send_backward(input_tensor_grad, config)
```

---

## Microbatch-to-VR Mapping

### The get_model_chunk_id() Function

**Location:** Lines 358-366 in bitpipe_4vr.py

```python
def get_model_chunk_id(microbatch_id):
    """Maps microbatch ID to VR (Virtual Rank)"""
    microbatch_id_in_group = microbatch_id % pipeline_parallel_size  # 0-3
    chunk_offset = 0 if microbatch_id < (total_num_microbatches//2) else 2
    model_chunk_id = microbatch_id_in_group // (pipeline_parallel_size // 2)  # 0 or 1
    model_chunk_id += chunk_offset  # Add 0 or 2
    return model_chunk_id
```

### Example: 4 Devices, 8 Base Microbatches (16 total after doubling)

```
MB ID  | in_group | offset | chunk/2 | Final VR | Pipeline
────── | ──────── | ────── | ─────── | ──────── | ────────
MB 0   |    0     |   0    |    0    |    0     | Forward (VR0)
MB 1   |    1     |   0    |    0    |    0     | Forward (VR0)
MB 2   |    2     |   0    |    1    |    1     | Backward (VR1)
MB 3   |    3     |   0    |    1    |    1     | Backward (VR1)
MB 4   |    0     |   0    |    0    |    0     | Forward (VR0)
MB 5   |    1     |   0    |    0    |    0     | Forward (VR0)
MB 6   |    2     |   0    |    1    |    1     | Backward (VR1)
MB 7   |    3     |   0    |    1    |    1     | Backward (VR1)
MB 8   |    0     |   2    |    0    |    2     | Forward (VR2)
MB 9   |    1     |   2    |    0    |    2     | Forward (VR2)
MB 10  |    2     |   2    |    1    |    3     | Backward (VR3)
MB 11  |    3     |   2    |    1    |    3     | Backward (VR3)
MB 12  |    0     |   2    |    0    |    2     | Forward (VR2)
MB 13  |    1     |   2    |    0    |    2     | Forward (VR2)
MB 14  |    2     |   2    |    1    |    3     | Backward (VR3)
MB 15  |    3     |   2    |    1    |    3     | Backward (VR3)
```

**Summary:**
- MB 0,1,4,5,... → **VR0** (first quarter of model, forward direction)
- MB 2,3,6,7,... → **VR1** (second quarter of model, backward direction)
- MB 8,9,12,13,... → **VR2** (third quarter of model, forward direction)
- MB 10,11,14,15,... → **VR3** (fourth quarter of model, backward direction)

---

## Three Communication Scenarios

### Overview

The routing logic determines **how to communicate** based on whether consecutive microbatches use the same VR:

```
┌─────────────────────────────────────────────────────────────────┐
│  Get next microbatch's VR                                       │
│  next_vr = get_model_chunk_id(microbatch_idx[k+1])            │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
   SAME VR?      DIFFERENT VR?      DIFFERENT VR?
   (Scenario 1)   Same Device?       Cross Device?
                  (Scenario 2)        (Scenario 3)
         │               │               │
    ┌────▼────┐     ┌────▼────┐    ┌────▼─────┐
    │ NORMAL  │     │ SAME-   │    │BIDIRECTIONAL
    │PIPELINE │     │DEVICE   │    │ ROUTING
    │         │     │TRANSITION    │
    │send_fwd │     │         │    │send_fwd
    │recv_fwd │     │detach() │    │recv_bwd
    │         │     │no-copy  │    │
    └─────────┘     └─────────┘    └──────────┘

    Uses NCCL    Uses in-GPU      Uses NCCL
    P2P          memory (fast!)   P2P
```

---

## Scenario 1: Same VR (Normal Pipeline Flow)

### When It Happens

**When:** Current and next microbatches use the **same VR**

Example: MB 0 (VR0) → MB 1 (VR0)

### Device Layout Example

```
Device 0: VR0[layers 0-3]  VR1[layers 6-7]  VR2[layers 14-15] VR3[layers 8-9]
Device 1: VR0[layers 2-3]  VR1[layers 4-5]  VR2[layers 12-13] VR3[layers 10-11]
Device 2: VR0[layers 4-5]  VR1[layers 2-3]  VR2[layers 10-11] VR3[layers 12-13]
Device 3: VR0[layers 6-7]  VR1[layers 0-1]  VR2[layers 8-9]   VR3[layers 14-15]
```

### Code (Lines 560-572)

```python
if forward_model_chunk_id == next_forward_model_chunk_id:
    if parallel_state.is_pipeline_first_stage():
        recv_prev = False  # First stage doesn't receive from previous

    if k==total_num_microbatches-1 and forward_only:
        recv_prev = False  # Last iteration, evaluation mode

    input_tensor = _profile_p2p_comm(
        p2p_communication.send_forward_recv_forward,
        "send_forward_recv_forward",
        output_tensor,           # ← Send current output
        recv_prev=recv_prev,     # ← Receive next input
        tensor_shape=tensor_shape,
        config=config,
    )
```

### Execution Flow

```
Device 0 (VR0)    Device 1 (VR0)    Device 2 (VR0)    Device 3 (VR0)
    │                  │                 │                 │
    │                  │                 │                 │
MB0 forward        MB0 forward       MB0 forward       MB0 forward
output             output            output            output
    │                  │                 │                 │
    ├─ send ──────────>│ recv            │                 │
    │                  ├─ send ──────────>│ recv            │
    │                  │                  ├─ send ──────────>│ recv
    │                  │                  │                  │
    │<──── ring continues ────────────────────────────────>│
    │                  │                 │                 │
MB1 forward        MB1 forward       MB1 forward       MB1 forward
```

**Key Point:** This is the **normal pipeline communication** - standard ring pattern through all devices.

---

## Scenario 2: Same-Device VR Transition

### When It Happens

**When:**
- MB transitions from VR0→VR2 **on the same device** (last stage)
- OR MB transitions from VR1→VR3 **on the same device** (first stage)
- Both happen on the **same physical device**

### The Magic: Zero-Copy Tensor Transition

```python
# Lines 574-577
if (parallel_state.is_pipeline_last_stage(ignore_virtual=True) and next_forward_model_chunk_id == v_size-2) or \
   (parallel_state.is_pipeline_first_stage(ignore_virtual=True) and next_forward_model_chunk_id == v_size-1):

    # SAME-DEVICE TRANSITION
    detached_output_tensor = output_tensor.detach()
    detached_output_tensor.requires_grad_()
    input_tensor = detached_output_tensor  # ← NO NETWORK COMMUNICATION!
```

### Understanding the Condition

```python
# Condition 1: Last stage transitioning VR0→VR2
parallel_state.is_pipeline_last_stage(ignore_virtual=True)  # Device is last stage
and next_forward_model_chunk_id == v_size-2                # Next is VR2 (value 2)

# Condition 2: First stage transitioning VR1→VR3
parallel_state.is_pipeline_first_stage(ignore_virtual=True) # Device is first stage
and next_forward_model_chunk_id == v_size-1                # Next is VR3 (value 3)
```

### Example: MB5 (VR0) → MB8 (VR2) on Device 3

```
Timeline:
─────────

Device 3, MB5 (VR0):
  ├─ Forward pass through VR0 layers [6-7]
  └─ output_tensor shape: (2048, 32, 4096)

Check transition:
  ├─ is_pipeline_last_stage() = True ✓
  ├─ next_vr == 2 (VR2) = True ✓
  └─ SCENARIO 2! (Same-device transition)

Transition (NO NETWORK!):
  ├─ detached_output = output_tensor.detach()
  │  └─ Detaches from VR0's computation graph
  │  └─ Saves memory - activations no longer tied to VR0
  │  └─ Output: tensor shape (2048, 32, 4096)
  │
  ├─ detached_output.requires_grad_()
  │  └─ Enable gradient computation for backward pass
  │  └─ Now ready to be input to VR2
  │
  └─ input_tensor = detached_output  ← SAME TENSOR OBJECT!

Device 3, MB8 (VR2):
  ├─ Input: input_tensor (same tensor from above)
  ├─ Forward pass through VR2 layers [8-9]
  └─ output_tensor shape: (2048, 32, 4096)
```

### Why This Is Brilliant

```
Normal Pipeline (between devices):
output_tensor ──────────────────────> Network (NCCL)
                 Serialization
                 Send over network
                 Deserialization
                 └─ Microseconds of latency!
                 └─ Network bandwidth used!
                 └─ Add communication overhead!

Same-Device Transition:
output_tensor (Device 3, VR0)
     │
     └─ detach() (in-GPU memory)
     │
     └─ input_tensor (Device 3, VR2)
        └─ Nanoseconds! (no network!)
        └─ In-GPU memory transfer only!
        └─ Zero network overhead!
```

### The Comment Explained

```python
# last stage of chunk0 and first stage of chunk2 are in the same device
```

Translation:
- "last stage of chunk0" = Last device (Device 3) executing VR0
- "first stage of chunk2" = Same device (Device 3) executing VR2
- "in the same device" = Both happen on Device 3
- Therefore, no network communication needed!

---

## Scenario 3: Cross-Device Bidirectional Transition

### When It Happens

**When:**
- Current and next microbatches use **DIFFERENT VRs**
- **AND it's NOT a same-device transition** (Scenario 2)
- Data must go to a **DIFFERENT device**

This is the **most frequent scenario** in the warmup phase!

### Code (Lines 578-591)

```python
else:
    if (parallel_state.is_pipeline_last_stage(ignore_virtual=True) and next_forward_model_chunk_id == v_size-2) or \
       (parallel_state.is_pipeline_first_stage(ignore_virtual=True) and next_forward_model_chunk_id == v_size-1):
        # Scenario 2: Same-device (handled above)
        detached_output_tensor = output_tensor.detach()
        ...
    else:
        # Scenario 3: Cross-device bidirectional
        if parallel_state.is_pipeline_last_stage():
            recv_next = False  # Last stage doesn't receive

        if k==total_num_microbatches-1 and forward_only:
            recv_next = False  # Last iteration, evaluation mode

        input_tensor = _profile_p2p_comm(
            p2p_communication.send_forward_recv_forward_bd0,
            "send_forward_recv_forward_bd0",
            output_tensor,           # ← Send to paired device
            recv_next=recv_next,     # ← Receive from paired device
            tensor_shape=tensor_shape,
            config=config,
        )
```

### What is `send_forward_recv_forward_bd0`?

- `bd0` = "Bidirectional, group 0"
- Used for **bidirectional pairing** in BitPipe
- Device 0 ↔ Device 3, Device 1 ↔ Device 2
- Different from Scenario 1 because devices are working on **different VRs**

### Concrete Example Timeline

**Device 0, Warmup Phase:**

```
Time  Current MB   Current VR   Next MB   Next VR   Scenario   Communication
────  ──────────   ──────────   ───────   ────────  ────────   ──────────────

k=0   MB0          VR0          MB1       VR0       Scenario 1 send_forward_recv_forward
                                          (same VR)            (normal pipeline)

k=1   MB1          VR0          MB2       VR1       Scenario 3 send_forward_recv_forward_bd0
                                          (diff VR)            ↑ CROSS-DEVICE!
                                                    └─ Send to Device 3's VR1
                                                    └─ Receive from Device 3's VR1

k=2   MB2          VR1          MB3       VR1       Scenario 1 send_forward_recv_forward
                                          (same VR)            (normal pipeline)

k=3   MB3          VR1          MB4       VR0       Scenario 3 send_forward_recv_forward_bd0
                                          (diff VR)            ↑ CROSS-DEVICE!
                                                    └─ Bidirectional routing

k=4   MB4          VR0          MB5       VR0       Scenario 1 send_forward_recv_forward
                                          (same VR)            (normal pipeline)

k=5   MB5          VR0          MB8       VR2       Scenario 2 detach()
                                          (diff VR)            (same-device!)
                                          BUT SAME
                                          DEVICE!

k=6   MB8          VR2          MB9       VR2       Scenario 1 send_forward_recv_forward
                                          (same VR)            (normal pipeline)

k=7   MB9          VR2          MB10      VR3       Scenario 3 send_forward_recv_forward_bd0
                                          (diff VR)            ↑ CROSS-DEVICE!
```

### Device 3 Perspective

```
Time  Current MB   Current VR   Next MB   Next VR   Scenario   Why
────  ──────────   ──────────   ───────   ────────  ────────   ────

k=0   MB0          VR0          MB1       VR0       Scenario 1 Same VR

k=1   MB1          VR0          MB2       VR1       Scenario 3 Different VR,
                                                                device 3 is last stage
                                                                but next is VR1 (not VR2)
                                                                → Cross-device!
                                                                → Send to Device 0's VR1

k=2   MB2          VR1          MB3       VR1       Scenario 1 Same VR

k=3   MB3          VR1          MB4       VR0       Scenario 3 Different VR,
                                                                device 3 is last stage
                                                                but next is VR0 (not VR2)
                                                                → Cross-device!

k=4   MB4          VR0          MB5       VR0       Scenario 1 Same VR

k=5   MB5          VR0          MB8       VR2       Scenario 2 DIFFERENT VR
                                                     BUT SAME DEVICE!
                                                     is_pipeline_last_stage=True
                                                     next == v_size-2 (VR2)
                                                     → Use detach()!
                                                     → NO NETWORK!

k=6   MB8          VR2          MB9       VR2       Scenario 1 Same VR

k=7   MB9          VR2          MB10      VR3       Scenario 3 Different VR,
                                                                but NOT same-device
                                                                (VR3 is different
                                                                 from VR2 transition)
                                                                → Cross-device!
```

### Visual Timeline Showing All Three Scenarios

```
Device 0 Timeline:
─────────────────

k=0:  MB0(VR0) ──┐ Scenario 1
              ├─ send_fwd ──> D1
              └─ recv_fwd <── D1

k=1:  MB1(VR0) ──┐ Scenario 3  ← CROSS-DEVICE!
              ├─ send_fwd ──> D3(VR1)  ← Goes to Device 3!
              └─ recv_fwd <── D3(VR1)  ← Receives from Device 3!

k=2:  MB2(VR1) ──┐ Scenario 1
              ├─ send_fwd ──> D3(VR1)
              └─ recv_fwd <── D1

k=3:  MB3(VR1) ──┐ Scenario 3  ← CROSS-DEVICE!
              ├─ send_fwd ──> D1(VR0)?  ← Bidirectional routing
              └─ recv_fwd <── D1(VR0)?

k=4:  MB4(VR0) ──┐ Scenario 1
              ├─ send_fwd ──> D1
              └─ recv_fwd <── D1

k=5:  MB5(VR0) ──┐ Scenario 1
              ├─ send_fwd ──> D1
              └─ recv_fwd <── D1

k=6:  MB6(VR1) ──┐ Scenario 1
              ├─ send_fwd ──> D3
              └─ recv_fwd <── D1

k=7:  MB7(VR1) ──┐ Scenario 3  ← CROSS-DEVICE!
              ├─ send_fwd ──> D1(VR0)?
              └─ recv_fwd <── D1(VR0)?


Device 3 Timeline:
─────────────────

k=0:  MB0(VR0) ──┐ Scenario 1
              ├─ send_fwd ──> (none, last stage)
              └─ recv_fwd <── D2

k=1:  MB1(VR0) ──┐ Scenario 3  ← CROSS-DEVICE!
              ├─ send_fwd ──> D0(VR1)  ← Sends to Device 0!
              └─ recv_fwd <── D2

k=2:  MB2(VR1) ──┐ Scenario 3  ← CROSS-DEVICE!
              ├─ send_fwd ──> D0(VR1)  ← Sends to Device 0!
              └─ recv_fwd <── (bidirectional from D0)

k=3:  MB3(VR1) ──┐ Scenario 1
              ├─ send_fwd ──> (none, last stage)
              └─ recv_fwd <── D2

k=4:  MB4(VR0) ──┐ Scenario 1
              ├─ send_fwd ──> (none, last stage)
              └─ recv_fwd <── D2

k=5:  MB5(VR0) ──┐ Scenario 2  ← SAME-DEVICE!
              ├─ NO NETWORK!
              └─ detach() ──> MB8(VR2) on Device 3

k=6:  MB8(VR2) ──┐ Scenario 1
              ├─ send_fwd ──> (none, last stage)
              └─ recv_fwd <── D2

k=7:  MB9(VR2) ──┐ Scenario 3  ← CROSS-DEVICE!
              ├─ send_fwd ──> D0(VR3)  ← Sends to Device 0!
              └─ recv_fwd <── (bidirectional)
```

### When Scenario 3 Happens - Summary

**Scenario 3 happens at EVERY VR transition that is NOT same-device:**

```
VR Transitions in Warmup:
────────────────────────

MB0(VR0) → MB1(VR0)    Same VR → Scenario 1
MB1(VR0) → MB2(VR1)    Different, cross-device → Scenario 3 ✓
MB2(VR1) → MB3(VR1)    Same VR → Scenario 1
MB3(VR1) → MB4(VR0)    Different, cross-device → Scenario 3 ✓
MB4(VR0) → MB5(VR0)    Same VR → Scenario 1
MB5(VR0) → MB8(VR2)    Different, SAME-DEVICE → Scenario 2 (NOT Scenario 3!)
MB8(VR2) → MB9(VR2)    Same VR → Scenario 1
MB9(VR2) → MB10(VR3)   Different, cross-device → Scenario 3 ✓
MB10(VR3) → MB11(VR3)  Same VR → Scenario 1
MB11(VR3) → MB12(VR2)  Different, cross-device → Scenario 3 ✓
```

**Frequency:** Scenario 3 happens **almost every 2-3 microbatches** in the warmup phase!

---

## Complete Warmup Timeline

### Full Execution: 4 Devices, 16 Total Microbatches

```
Time  Device 0         Device 1         Device 2         Device 3
      (Rank 0)        (Rank 1)        (Rank 2)        (Rank 3)
──────────────────────────────────────────────────────────────────

k=0:  MB0(VR0)S1 ─────> MB0(VR0)S1 ──> MB0(VR0)S1 ──> MB0(VR0)S1
      recv from ext   recv from D0    recv from D1    loss compute

k=1:  MB1(VR0)S3 ────────────────────────────────────> MB1(VR0)
      send to D3 recv from D3

k=2:  MB2(VR1)S1 ────> MB2(VR1)S1 ──> MB2(VR1)S3 <─── recv from D3

k=3:  MB3(VR1)S3 <────────────────────────────────── MB3(VR1)
      recv from D3 send to D0

k=4:  MB4(VR0)S1 ────> MB4(VR0)S1 ──> MB4(VR0)S1 ──> MB4(VR0)

k=5:  MB5(VR0)S1 ────> MB5(VR0)S1 ──> MB5(VR0)S1 ──> MB5(VR0)
      (normally)      (normally)      (normally)      S2: detach()
                                                       ↓ to MB8(VR2)

k=6:  MB6(VR1)S1 ────> MB6(VR1)S1 ──> MB6(VR1)S1 ──> MB8(VR2)
      (overlapping)                                   (from detach)

k=7:  MB7(VR1)S3 <────────────────────────────────── MB9(VR2)
      recv from D3 send to D0                        S3: send to D0

...

Legend:
───────
S1 = Scenario 1 (send_forward_recv_forward) - Normal pipeline
S2 = Scenario 2 (detach) - Same-device, no network
S3 = Scenario 3 (send_forward_recv_forward_bd0) - Cross-device bidirectional
```

---

## recv_forward() vs schedules.py

### The Critical Issue: Two Different APIs

**BitPipe uses the LOW-LEVEL API:**

```python
# Location: bitpipe_4vr.py:518-523
from megatron.core.pipeline_parallel import p2p_communication

input_tensor = _profile_p2p_comm(
    p2p_communication.recv_forward,    # ← LOW-LEVEL API
    "recv_forward",
    tensor_shape,                      # ← SINGLE TUPLE (2048, 32, 4096)
    config
)

# Returns: Single Tensor ✅
# Type: torch.Tensor with shape (2048, 32, 4096)
```

**Chimera INCORRECTLY uses the HIGH-LEVEL WRAPPER:**

```python
# Location: chimera_2vr.py:524
from megatron.core.pipeline_parallel.schedules import recv_forward

input_tensor = recv_forward(
    tensor_shape,      # ← SINGLE TUPLE (2048, 32, 4096)
    config
)

# Expected: Single Tensor
# Actually Returns: LIST of Tensors ❌
# Type: [torch.Tensor, torch.Tensor, torch.Tensor] (3 items!)
# Error: input_tensor.shape → AttributeError! ❌
```

### Why They're Different

**Low-Level API (p2p_communication.recv_forward):**

```python
# Location: megatron/core/pipeline_parallel/p2p_communication.py
def recv_forward(tensor_shape, config):
    """Receives ONE tensor from previous pipeline stage"""
    # Receives via NCCL directly
    return tensor  # ← Single tensor object
```

**High-Level Wrapper (schedules.recv_forward):**

```python
# Location: megatron/core/pipeline_parallel/schedules.py:973
def recv_forward(tensor_shapes, config):
    """
    Designed for encoder-decoder models with multiple tensors

    Input: List of shapes (for encoder, decoder, etc.)
    Output: List of tensors (one per shape)
    """
    input_tensors = []

    for tensor_shape in tensor_shapes:  # ← ITERATES over shapes!
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensors.append(p2p_communication.recv_forward(tensor_shape, config))

    return input_tensors  # ← RETURNS LIST!
```

### What Happens in Chimera (The Bug)

```python
# Chimera passes single tuple to wrapper expecting list:
tensor_shape = (2048, 32, 4096)

# Inside schedules.recv_forward():
input_tensors = []
for tensor_shape in (2048, 32, 4096):  # ← ITERATES 3 TIMES!
    # Iteration 1: tensor_shape = 2048 (not valid!)
    # Iteration 2: tensor_shape = 32
    # Iteration 3: tensor_shape = 4096
    input_tensors.append(p2p_communication.recv_forward(tensor_shape, config))

# Tries to call p2p_communication.recv_forward(2048, config) ← INVALID!

# Later:
input_tensor.shape  # ← Crashes! input_tensor = [Tensor, Tensor, Tensor]
# AttributeError: 'list' object has no attribute 'shape'
```

### BitPipe Imports But Never Uses

Interestingly, BitPipe **imports** the wrapper functions but **never calls them:**

```python
# bitpipe_4vr.py:34-43
from megatron.core.pipeline_parallel.schedules import (
    recv_forward,      # ❌ Imported but NEVER CALLED
    send_forward,      # ❌ Imported but NEVER CALLED
    recv_backward,     # ❌ Imported but NEVER CALLED
    send_backward,     # ❌ Imported but NEVER CALLED
    deallocate_output_tensor,  # ✅ Called at line 594
    forward_step,      # ✅ Called at line 413
    backward_step,     # ✅ Called at line 487
    get_tensor_shapes, # ❌ Imported but NEVER CALLED
)
```

**These are leftover from copy-paste!** BitPipe only uses the **low-level P2P API** for communication.

### The Fix for Chimera

Change Chimera to use the same low-level API as BitPipe:

```python
# File: megatron/core/pipeline_parallel/schedule_impl/chimera/chimera_2vr.py

# OLD (Line 39):
from megatron.core.pipeline_parallel.schedules import (
    recv_forward,      # ← Remove this
    send_forward,      # ← Remove this
    ...
)

# NEW (Add this):
from megatron.core.pipeline_parallel import p2p_communication

# OLD (Line 524):
input_tensor = recv_forward(tensor_shape, config)

# NEW:
if not parallel_state.is_pipeline_first_stage():
    input_tensor = p2p_communication.recv_forward(tensor_shape, config)
else:
    input_tensor = None

# OLD (Line 548):
send_forward(output_tensor, config)

# NEW:
if not parallel_state.is_pipeline_last_stage():
    p2p_communication.send_forward(output_tensor, config)

# Similar fixes for recv_backward and send_backward
```

---

## Helper Functions

### forward_step_helper (Lines 391-430)

Executes the forward pass for one microbatch through one model chunk:

```python
def forward_step_helper(microbatch_id, checkpoint_activations_microbatch, offset):
    """Run forward step with model split into chunks"""

    # Get which VR this microbatch uses
    model_chunk_id = get_model_chunk_id(microbatch_id)

    # Get the input tensor (previously received or created)
    if parallel_state.is_pipeline_first_stage():
        if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
            input_tensors[model_chunk_id].append(None)

    input_tensor = input_tensors[model_chunk_id][-1-offset]

    # Call schedules.py's forward_step (computation, not communication!)
    output_tensor = forward_step(
        forward_step_func,
        data_iterator[model_chunk_id],
        model[model_chunk_id],
        num_microbatches//2,
        input_tensor,              # ← Single tensor from P2P
        forward_data_store,
        config,
        collect_non_loss_data,
        checkpoint_activations_microbatch,
    )

    output_tensors[model_chunk_id].append(output_tensor)
    return output_tensor
```

**Key Point:** `forward_step()` from schedules.py is a **computation helper**, not communication!

### backward_step_helper (Lines 460-512)

Executes the backward pass for one microbatch through one model chunk:

```python
def backward_step_helper(microbatch_id):
    """Run backward step with model split into chunks"""

    model_chunk_id = get_model_chunk_id(microbatch_id)

    # Enable grad sync when appropriate
    if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(microbatch_id):
        enable_grad_sync()
        synchronized_model_chunks.add(model_chunk_id)

    # Get stored tensors
    input_tensor = input_tensors[model_chunk_id].pop(0)
    output_tensor = output_tensors[model_chunk_id].pop(0)
    output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)

    # Call schedules.py's backward_step (computation, not communication!)
    input_tensor_grad = backward_step(
        input_tensor,
        output_tensor,
        output_tensor_grad,
        model_type,
        config
    )

    # Handle custom gradient sync
    if config.grad_sync_func is not None:
        grad_sync_microbatch_id = microbatch_id
        if grad_sync_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(...):
            enable_grad_sync()
            config.grad_sync_func(model[grad_sync_chunk_id].parameters())
            synchronized_model_chunks.add(grad_sync_chunk_id)

    disable_grad_sync()
    return input_tensor_grad
```

---

## Summary Table

### Scenario Comparison

| Aspect | Scenario 1 | Scenario 2 | Scenario 3 |
|--------|-----------|-----------|-----------|
| **Condition** | Same VR | Different VR, same device | Different VR, cross-device |
| **Example** | MB0(VR0)→MB1(VR0) | MB5(VR0)→MB8(VR2) on D3 | MB1(VR0)→MB2(VR1) across devices |
| **Function** | `send_forward_recv_forward` | `detach()` | `send_forward_recv_forward_bd0` |
| **Network** | NCCL P2P | None (in-GPU) | NCCL P2P |
| **Latency** | Microseconds | Nanoseconds | Microseconds |
| **Frequency** | ~50% of transitions | ~25% of transitions | ~25% of transitions |

### API Comparison

| Aspect | BitPipe | Chimera (Broken) | Standard Pipeline |
|--------|---------|-----------------|-------------------|
| **P2P API** | `p2p_communication.recv_forward()` | `schedules.recv_forward()` | `schedules.recv_forward()` |
| **Input** | Single tuple | Single tuple (wrong!) | List of tuples |
| **Output** | Single tensor | List of tensors | List of tensors |
| **Works?** | ✅ Yes | ❌ No | ✅ Yes |
| **Model type** | Single-stack (GPT) | Single-stack (GPT) | Both (encoder-decoder) |

---

## Key Insights

1. **BitPipe doubles microbatches** without doubling memory by creating 4 VRs that share the pipeline
2. **Three routing scenarios** handle different VR transitions efficiently
3. **Scenario 2 (same-device transition) is the key optimization** - zero-copy tensor passing
4. **Scenario 3 (cross-device) is the most frequent** - happens at almost every VR transition
5. **Chimera breaks because it uses the wrong API** - high-level wrapper instead of low-level P2P
6. **The fix is straightforward** - replace `schedules.recv_forward()` with `p2p_communication.recv_forward()`

---

## References

- **BitPipe 4-VR Implementation**: `megatron/core/pipeline_parallel/schedule_impl/bitpipe/bitpipe_4vr.py`
- **Chimera 2-VR Implementation**: `megatron/core/pipeline_parallel/schedule_impl/chimera/chimera_2vr.py`
- **P2P Communication**: `megatron/core/pipeline_parallel/p2p_communication.py`
- **Schedules Helpers**: `megatron/core/pipeline_parallel/schedules.py`
- **Parallel State**: `megatron/core/parallel_state.py`
- **Architecture Documentation**: `CLAUDE.md` in repository root
