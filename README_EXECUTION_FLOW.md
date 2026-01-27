# BitPipe Execution Flow Documentation

This directory contains three comprehensive guides explaining how BitPipe works, from code files to concrete execution traces.

## Documents Included

### 1. BITPIPE_EXECUTION_FLOW.md
**Start here if you want to understand the complete step-by-step flow.**

Covers:
- Phase 0: Program initialization and argument parsing
- Phase 1: Model building and layer distribution
- Phase 2: Scheduler selection
- Phase 3: Training loop and scheduler invocation
- Phase 4: BitPipe scheduler execution (detailed breakdown of all phases)
- Phase 5: Post-scheduler processing (gradient reduction, optimizer step)

**Key concepts explained:**
- Why layers are distributed differently for each VR
- How the scheduler dispatcher works
- Complete breakdown of warmup, steady-state, and cooldown phases
- Bidirectional gradient synchronization
- How boundary checks prevent deadlocks

**Visual diagrams included** showing the complete pipeline

---

### 2. BITPIPE_CODE_MAP.md
**Use this as a quick reference to find specific code.**

Contains:
- Complete file structure with line numbers
- Function signatures for key functions
- File execution sequence (which file calls which)
- Critical code locations for:
  - Deadlock prevention
  - Layer assignment (V-shaped pattern)
  - Scheduler dispatcher
  - BD group initialization
- Debugging tips and how to add print statements
- Summary of key locations

**Perfect for:**
- Finding where a specific piece of code is located
- Understanding call chains
- Quick reference while reading code

---

### 3. BITPIPE_CONCRETE_EXAMPLE.md
**Read this for a concrete walkthrough with actual numbers.**

Uses this configuration:
- 4 devices (GPUs)
- 24 base layers → 48 total layers (after doubling for BitPipe)
- 4 user microbatches → 8 total microbatches (after doubling for bidirectional)
- Micro batch size: 8
- Global batch size: 32

Walks through:
- **PHASE 1:** Exact layer distribution for each device/VR combination
  - Shows the complete V-shaped pattern
  - Explains what layers each device gets

- **PHASE 2:** Scheduler selection with exact code flow

- **PHASE 3:** Scheduler invocation with specific parameters

- **PHASE 4:** Warmup phase execution (step by step)
  - Shows what happens on each device during each warmup iteration
  - Includes timeline diagram of data flow

- **PHASE 5:** Steady-state (1F1B) phase
  - Shows overlapped forward and backward passes
  - Explains why this improves GPU utilization

- **PHASE 6:** Cooldown phase

- **PHASE 7:** Post-scheduler processing

**Great for:**
- Understanding concrete data flow with real numbers
- Seeing how devices coordinate with each other
- Understanding why the V-shaped pattern is beneficial

---

## Quick Start Guide

### If you're new to BitPipe:
1. Start with `BITPIPE_EXECUTION_FLOW.md` (summary section first)
2. Read the "Key Concepts to Remember" section
3. Move to `BITPIPE_CONCRETE_EXAMPLE.md` for a walkthrough
4. Use `BITPIPE_CODE_MAP.md` as reference while reading code

### If you're debugging an issue:
1. Check `BITPIPE_CODE_MAP.md` for file locations
2. Look at "Critical Code Locations" section
3. Add debug prints to the suggested locations
4. Refer to `BITPIPE_EXECUTION_FLOW.md` to understand the flow

### If you're comparing with Chimera:
1. Read about deadlock prevention in `BITPIPE_EXECUTION_FLOW.md`
2. Check the boundary checks in `BITPIPE_CODE_MAP.md`
3. Understand why Chimera needs the same checks

---

## Key Takeaways

### Layer Distribution
- BitPipe creates 4 VRs per device
- Layers are distributed in a **V-shaped pattern** for optimal scheduling
- Each device gets DIFFERENT layers for each VR
- Paired devices (e.g., Device 0 & 3) share layers and must sync gradients

### Scheduler Flow
1. `get_forward_backward_func()` is a **dispatcher** (schedules.py)
2. It detects `--enable-bitpipe-schedule` flag
3. Returns the BitPipe scheduler function
4. Training loop calls this function with data and model

### Three Execution Phases
1. **Warmup:** Fill the pipeline with forward passes
2. **Steady-State:** 1F1B (one forward, one backward) with overlapping
3. **Cooldown:** Finish remaining backward passes

### Critical for Correctness
1. **Boundary checks:** First/last stages don't send/recv from non-existent neighbors
2. **BD group sync:** Paired devices average gradients for shared layers
3. **Microbatch scheduling:** Order matters for optimal utilization

### Why Chimera Deadlocks
- Missing boundary checks on first/last stages
- Code tries to recv_forward() on first stage → blocks forever
- No mechanism to skip P2P on boundary conditions

---

## Files in Repository

Main files explained in these guides:

```
megatron/
├── training.py                                    ← Main training loop
├── arguments.py                                   ← BitPipe arg parsing
├── initialize.py                                  ← Megatron init
├── model/
│   └── transformer.py                             ← Layer distribution
├── core/
│   ├── pipeline_parallel/
│   │   ├── schedules.py                           ← Scheduler dispatcher
│   │   ├── schedule_impl/bitpipe/
│   │   │   └── bitpipe_4vr.py                     ← Main BitPipe scheduler
│   │   ├── p2p_communication.py                   ← P2P operations
│   │   └── parallel_state.py                      ← Parallel groups & checks
```

---

## Questions?

Refer to the specific document:
- "How does the scheduler know which VR to execute?" → BITPIPE_CODE_MAP.md
- "What happens in warmup phase?" → BITPIPE_EXECUTION_FLOW.md
- "Show me step-by-step with numbers" → BITPIPE_CONCRETE_EXAMPLE.md
- "Why does Chimera deadlock?" → BITPIPE_EXECUTION_FLOW.md (deadlock prevention section)

---

## Documentation Structure

Each document is self-contained but cross-referenced:
- EXECUTION_FLOW: High-level overview, concepts
- CODE_MAP: File locations, function signatures
- CONCRETE_EXAMPLE: Step-by-step trace with actual numbers

Read them in any order based on your learning style:
- **Visual learners:** Start with BITPIPE_EXECUTION_FLOW.md (has diagrams)
- **Code readers:** Start with BITPIPE_CODE_MAP.md (has file locations)
- **Hands-on learners:** Start with BITPIPE_CONCRETE_EXAMPLE.md (has numbers)

