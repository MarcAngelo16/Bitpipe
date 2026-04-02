# Session Summary — 2026-03-31

## Scope
Profiling fixes and explanation only. No scheduler logic changes.

---

## Problem
Chimera P2P Comm Time showed `0.001s` (effectively zero) vs BitPipe's `0.068s` and 1F1B's `0.227s` — clearly not capturing real communication.

**Root cause:** All `p2p_communication.chimera_*` calls in warmup, 1F1B, and cooldown phases were made directly without any profiler wrapping. The existing `_profile_p2p_comm` helper only handled standard unidirectional send/recv, not Chimera's combined `send_next_recv_prev`-style functions.

---

## Fix

**File changed:** `megatron/core/pipeline_parallel/schedule_impl/chimera/chimera_2vr.py`

Added `_profile_chimera_p2p_call(comm_func, comm_type_name, *args, **kwargs)` helper and wrapped all ~20 call sites:

| Phase | Operations wrapped |
|---|---|
| Warmup pre-recv | `recv_prev_only`, `recv_next_only` |
| Warmup 1F1B fwd | 4 combined `send_*_recv_*` variants, `send_*_only`, `recv_*_only` |
| 1F1B fwd comms | Combined send+recv, send-only, recv-only |
| 1F1B bwd comms | 4 combined variants, bridge `chimera_communicate` |
| Cooldown chain | `chain_send_recv`, `chain_send_only`, `chain_recv_only` |
| Final flush | `final_flush_send` |

The pre-sync flush (`send_backward_flush`) at the -1 marker already had inline profiling — left unchanged.

---

## Context: Previous Session (same profiling work)
The following were done in the prior session (not today):
- `bitpipe_profiler.py` — full GPU-aligned timestamp rewrite (CUDA events replace CPU wall clock for start_time)
- `analyze_bitpipe_profile.py` — updated for new format (sync blocks, -1 markers, `cuda_event_gpu_aligned` timing label)
- `chimera_2vr.py` + `bitpipe_4vr.py` — added sync block profiling (`start_sync_block`/`end_sync_block`) around BD allreduce (-1 marker)

---

## Metric Explanation (how the comparison table is built)
- **Pipeline Duration** = `max(total_time)` across all ranks (wall-clock, slowest rank)
- **Avg Total Time** = `mean(total_time)` across ranks (wall-clock)
- **Forward/Backward/P2P** = mean of GPU-measured sums across ranks
- **BD Sync** = mean of `total_sync_time_ms` (GPU-measured allreduce duration)
- **Efficiency** = `(fwd + bwd) / avg_total_time × 100` — pure compute fraction
- **Throughput** = `(num_fwd + num_bwd passes) / total_time` per rank — note BitPipe has 2x microbatches by design, making direct throughput comparison misleading

Numbers don't sum to total_time because P2P timing overlaps with microbatch timing (end_microbatch is called after P2P in the loop).
