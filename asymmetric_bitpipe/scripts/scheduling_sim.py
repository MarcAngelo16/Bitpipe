#!/usr/bin/env python3
"""
BitPipe Scheduling Simulator

This program simulates and compares microbatch scheduling strategies:
- BitPipe 4-VR with get_microbatch_idx() (complex 5-phase algorithm)
- Chimera 2-VR with build_chimera_forward_schedule() (simple interleaving)

Usage:
    python scheduling_sim.py

Or directly:
    python scheduling_sim.py --num-microbatches 8 --num-devices 4
"""

import argparse
import sys
from typing import List, Tuple
import textwrap


class BitPipeScheduler:
    """BitPipe 4-VR scheduling simulator"""

    def __init__(self, total_num_microbatches: int, pipeline_parallel_size: int):
        self.total_num_microbatches = total_num_microbatches
        self.pipeline_parallel_size = pipeline_parallel_size
        self.num_model_chunks = 4  # BitPipe always has 4 VRs

    def get_microbatch(self) -> List[List[int]]:
        """
        Group microbatches by VR.

        Returns 4 lists, one for each VR:
        - microbatch_id01[0]: VR0 microbatches
        - microbatch_id01[1]: VR1 microbatches
        - microbatch_id01[2]: VR2 microbatches
        - microbatch_id01[3]: VR3 microbatches
        """
        microbatch_id01 = [[] for _ in range(self.num_model_chunks)]

        for i in range(self.total_num_microbatches):
            model_chunk_id = self._get_model_chunk_id(i)
            microbatch_id01[model_chunk_id].append(i)

        return microbatch_id01

    def _get_model_chunk_id(self, microbatch_id: int) -> int:
        """Map microbatch ID to VR (0-3)"""
        microbatch_id_in_group = microbatch_id % self.pipeline_parallel_size
        chunk_offset = 0 if microbatch_id < (self.total_num_microbatches // 2) else 2
        model_chunk_id = microbatch_id_in_group // (self.pipeline_parallel_size // 2)
        model_chunk_id += chunk_offset
        return model_chunk_id

    def get_microbatch_idx(self, pipeline_parallel_rank: int) -> List[int]:
        """
        Compute execution order for BitPipe 4-VR (forward pass)

        This is the main BitPipe scheduling algorithm with 5 phases.
        Different ranks get different orders.

        Args:
            pipeline_parallel_rank: Current device rank (0 to num_devices-1)

        Returns:
            List of microbatch IDs in execution order
        """
        microbatch_id01 = self.get_microbatch()
        i_loop = self.total_num_microbatches // self.pipeline_parallel_size // 2

        num_unit = self.pipeline_parallel_size // 2
        i_half = pipeline_parallel_rank // num_unit
        num_initial = (
            num_unit - pipeline_parallel_rank
            if pipeline_parallel_rank < num_unit
            else pipeline_parallel_rank - num_unit + 1
        )

        microbatch_idx = []

        for j in range(i_loop):
            # Phase 1: Initial microbatches from first half
            for i in range(num_initial):
                if j * num_unit + i < len(microbatch_id01[i_half]):
                    microbatch_idx.append(microbatch_id01[i_half][j * num_unit + i])

            # Phase 2: Alternating microbatches from both halves
            for i in range(num_unit - num_initial):
                if j * num_unit + i < len(microbatch_id01[1 - i_half]):
                    microbatch_idx.append(microbatch_id01[1 - i_half][j * num_unit + i])
                if j * num_unit + i + num_initial < len(microbatch_id01[i_half]):
                    microbatch_idx.append(microbatch_id01[i_half][j * num_unit + i + num_initial])

            # Phase 3: More alternation
            for i in range(num_initial):
                if j * num_unit + i + (num_unit - num_initial) < len(microbatch_id01[1 - i_half]):
                    microbatch_idx.append(microbatch_id01[1 - i_half][j * num_unit + i + (num_unit - num_initial)])
                if j * num_unit + i < len(microbatch_id01[3 - i_half]):
                    microbatch_idx.append(microbatch_id01[3 - i_half][j * num_unit + i])

            # Phase 4: Final alternation
            for i in range(num_unit - num_initial):
                if j * num_unit + i < len(microbatch_id01[2 + i_half]):
                    microbatch_idx.append(microbatch_id01[2 + i_half][j * num_unit + i])
                if j * num_unit + i + num_initial < len(microbatch_id01[3 - i_half]):
                    microbatch_idx.append(microbatch_id01[3 - i_half][j * num_unit + i + num_initial])

            # Phase 5: Final grouped microbatches
            for i in range(num_initial):
                if j * num_unit + i + (num_unit - num_initial) < len(microbatch_id01[2 + i_half]):
                    microbatch_idx.append(microbatch_id01[2 + i_half][j * num_unit + i + (num_unit - num_initial)])

        return microbatch_idx

    def get_bkmicrobatch_idx(self, pipeline_parallel_rank: int) -> List[int]:
        """
        Compute execution order for BitPipe 4-VR (backward pass)

        Similar to forward but processes VR2/VR3 first.
        Includes sync markers (-1).

        Args:
            pipeline_parallel_rank: Current device rank (0 to num_devices-1)

        Returns:
            List of microbatch IDs + sync markers in execution order
        """
        microbatch_id01 = self.get_microbatch()
        i_loop = self.total_num_microbatches // self.pipeline_parallel_size // 2

        num_unit = self.pipeline_parallel_size // 2
        i_half = pipeline_parallel_rank // num_unit
        num_initial = (
            num_unit - pipeline_parallel_rank
            if pipeline_parallel_rank < num_unit
            else pipeline_parallel_rank - num_unit + 1
        )

        microbatch_idx = []

        for j in range(i_loop):
            # Phase 1: Initial microbatches from VR2/VR3
            for k in range(num_initial):
                if j * num_unit + k < len(microbatch_id01[2 + i_half]):
                    microbatch_idx.append(microbatch_id01[2 + i_half][j * num_unit + k])

            # Phase 2: Alternation
            for k in range(num_unit - num_initial):
                if j * num_unit + k < len(microbatch_id01[3 - i_half]):
                    microbatch_idx.append(microbatch_id01[3 - i_half][j * num_unit + k])
                if j * num_unit + k + num_initial < len(microbatch_id01[2 + i_half]):
                    microbatch_idx.append(microbatch_id01[2 + i_half][j * num_unit + k + num_initial])

            # Phase 3: More alternation
            for k in range(num_initial):
                if j * num_unit + k + (num_unit - num_initial) < len(microbatch_id01[3 - i_half]):
                    microbatch_idx.append(microbatch_id01[3 - i_half][j * num_unit + k + (num_unit - num_initial)])
                if j * num_unit + k < len(microbatch_id01[1 - i_half]):
                    microbatch_idx.append(microbatch_id01[1 - i_half][j * num_unit + k])

            # Phase 4: Final alternation
            for k in range(num_unit - num_initial):
                if j * num_unit + k < len(microbatch_id01[i_half]):
                    microbatch_idx.append(microbatch_id01[i_half][j * num_unit + k])
                if j * num_unit + k + num_initial < len(microbatch_id01[1 - i_half]):
                    microbatch_idx.append(microbatch_id01[1 - i_half][j * num_unit + k + num_initial])

            # Phase 5: Final grouped microbatches
            for k in range(num_initial):
                if j * num_unit + k + (num_unit - num_initial) < len(microbatch_id01[i_half]):
                    microbatch_idx.append(microbatch_id01[i_half][j * num_unit + k + (num_unit - num_initial)])

        # Add sync markers
        if pipeline_parallel_rank == self.pipeline_parallel_size // 2 or \
           pipeline_parallel_rank == self.pipeline_parallel_size // 2 - 1:
            microbatch_idx.append(-1)
        else:
            microbatch_idx.insert(-1, -1)
        microbatch_idx.append(-1)

        return microbatch_idx


class ChimeraScheduler:
    """Chimera 2-VR scheduling simulator"""

    def __init__(self, total_num_microbatches: int, pipeline_parallel_size: int):
        self.total_num_microbatches = total_num_microbatches
        self.pipeline_parallel_size = pipeline_parallel_size
        self.num_model_chunks = 2  # Chimera always has 2 VRs

    def get_chimera_microbatch_groups(self) -> List[List[int]]:
        """
        Simple grouping for Chimera 2-VR

        Returns 2 lists:
        - [0]: First half (VR0)
        - [1]: Second half (VR1)
        """
        # For Chimera, both VRs process all microbatches
        # Just return all MBs in both groups (no doubling)
        all_mbs = list(range(self.total_num_microbatches))
        return [all_mbs, all_mbs]

    def chimera_get_microbatch_idx(self, pipeline_parallel_rank: int) -> List[int]:
        """
        Generate forward microbatch schedule for Chimera 2-VR (NEW ALGORITHM)

        Chimera uses simple 2-VR architecture without V-shaped transformation.
        No doubling of microbatches - just 8 MBs total.

        Pattern:
        - Process microbatches in chunks of 4 (num_unit * 2)
        - Apply rank-dependent reordering to each chunk
        - Different pattern for first half vs second half ranks

        Args:
            pipeline_parallel_rank: Current device rank (0 to num_devices-1)

        Returns:
            List of microbatch IDs in execution order
        """
        microbatch_idx = []
        num_unit = self.pipeline_parallel_size // 2  # 2 for 4 devices
        i_half = pipeline_parallel_rank // num_unit  # 0 for ranks 0-1, 1 for ranks 2-3
        position_in_half = pipeline_parallel_rank % num_unit  # 0 or 1

        # Process in chunks of 4 microbatches (num_unit * 2)
        chunk_size = num_unit * 2
        num_chunks = self.total_num_microbatches // chunk_size

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            # Get the 4 MBs in this chunk: [0,1,2,3], [4,5,6,7]
            chunk_mbs = [chunk_start, chunk_start + 1, chunk_start + 2, chunk_start + 3]

            if i_half == 0:  # First half ranks (0, 1)
                if position_in_half == 0:
                    # Rank 0: sequential order
                    # [0,1,2,3] → [0,1,2,3]
                    microbatch_idx.extend(chunk_mbs)
                else:
                    # Rank 1: swap pairs within chunk
                    # [0,1,2,3] → [0,2,1,3]
                    microbatch_idx.extend([chunk_mbs[0], chunk_mbs[2], chunk_mbs[1], chunk_mbs[3]])
            else:  # Second half ranks (2, 3)
                if position_in_half == 0:
                    # Rank 2: reverse each pair
                    # [0,1,2,3] → [2,0,3,1]
                    microbatch_idx.extend([chunk_mbs[2], chunk_mbs[0], chunk_mbs[3], chunk_mbs[1]])
                else:
                    # Rank 3: reverse pair order (group pairs)
                    # [0,1,2,3] → [2,3,0,1]
                    microbatch_idx.extend([chunk_mbs[2], chunk_mbs[3], chunk_mbs[0], chunk_mbs[1]])

        return microbatch_idx

    def chimera_get_bkmicrobatch_idx(self, pipeline_parallel_rank: int) -> List[int]:
        """
        Generate backward microbatch schedule for Chimera 2-VR (NEW ALGORITHM)

        Pattern:
        - Backward is derived from the paired rank's forward schedule
        - Paired rank = num_devices - 1 - current_rank
        - Add sync markers (-1) at specific positions

        Args:
            pipeline_parallel_rank: Current device rank (0 to num_devices-1)

        Returns:
            List of microbatch IDs + sync markers in execution order
        """
        microbatch_idx = []
        num_unit = self.pipeline_parallel_size // 2

        # Get paired rank's forward schedule
        paired_rank = self.pipeline_parallel_size - 1 - pipeline_parallel_rank
        paired_forward = self.chimera_get_microbatch_idx(paired_rank)

        # Use paired rank's forward schedule as backward for current rank
        microbatch_idx.extend(paired_forward)

        # Add sync markers (exactly 2 per rank)
        # Position based on rank
        if pipeline_parallel_rank == num_unit or pipeline_parallel_rank == num_unit - 1:
            # Middle ranks: append both at end
            microbatch_idx.append(-1)
            microbatch_idx.append(-1)
        else:
            # Outer ranks: insert before last element
            microbatch_idx.insert(-1, -1)
            microbatch_idx.append(-1)

        return microbatch_idx

    def build_chimera_forward_schedule(self, pipeline_parallel_rank: int) -> List[int]:
        """
        Build forward execution schedule for Chimera 2-VR

        Simple rank-dependent interleaving strategy.
        First half ranks prioritize VR0, second half prioritize VR1.

        Args:
            pipeline_parallel_rank: Current device rank (0 to num_devices-1)

        Returns:
            List of microbatch IDs in execution order
        """
        microbatch_groups = self.get_chimera_microbatch_groups()
        num_unit = self.pipeline_parallel_size // 2
        is_first_half = pipeline_parallel_rank < num_unit

        forward_schedule = []

        if is_first_half:
            # First half ranks: prioritize VR0 → VR1
            offset = pipeline_parallel_rank

            # Initial VR0 microbatches
            for i in range(offset + 1):
                if i < len(microbatch_groups[0]):
                    forward_schedule.append(microbatch_groups[0][i])

            # Interleave remaining
            vr0_idx = offset + 1
            vr1_idx = 0

            while vr0_idx < len(microbatch_groups[0]) or vr1_idx < len(microbatch_groups[1]):
                # Add VR1
                if vr1_idx < len(microbatch_groups[1]):
                    forward_schedule.append(microbatch_groups[1][vr1_idx])
                    vr1_idx += 1

                # Add VR0
                if vr0_idx < len(microbatch_groups[0]):
                    forward_schedule.append(microbatch_groups[0][vr0_idx])
                    vr0_idx += 1

        else:
            # Second half ranks: prioritize VR1 → VR0
            offset = self.pipeline_parallel_size - 1 - pipeline_parallel_rank

            # Initial VR1 microbatches
            for i in range(offset + 1):
                if i < len(microbatch_groups[1]):
                    forward_schedule.append(microbatch_groups[1][i])

            # Interleave remaining
            vr1_idx = offset + 1
            vr0_idx = 0

            while vr0_idx < len(microbatch_groups[0]) or vr1_idx < len(microbatch_groups[1]):
                # Add VR0
                if vr0_idx < len(microbatch_groups[0]):
                    forward_schedule.append(microbatch_groups[0][vr0_idx])
                    vr0_idx += 1

                # Add VR1
                if vr1_idx < len(microbatch_groups[1]):
                    forward_schedule.append(microbatch_groups[1][vr1_idx])
                    vr1_idx += 1

        return forward_schedule

    def build_chimera_backward_schedule(self, pipeline_parallel_rank: int) -> List[int]:
        """
        Build backward execution schedule for Chimera 2-VR

        Reverses VR0/VR1 processing order.
        Includes gradient sync markers (-1).

        Args:
            pipeline_parallel_rank: Current device rank (0 to num_devices-1)

        Returns:
            List of microbatch IDs + sync markers in execution order
        """
        microbatch_groups = self.get_chimera_microbatch_groups()
        num_unit = self.pipeline_parallel_size // 2
        is_first_half = pipeline_parallel_rank < num_unit

        backward_schedule = []

        # Backward processes opposite VR order from forward
        if is_first_half:
            # Forward: VR0 → VR1, Backward: VR1 → VR0
            offset = pipeline_parallel_rank

            # Initial VR1 microbatches
            for i in range(offset + 1):
                if i < len(microbatch_groups[1]):
                    backward_schedule.append(microbatch_groups[1][i])

            # Interleave remaining
            vr1_idx = offset + 1
            vr0_idx = 0

            while vr0_idx < len(microbatch_groups[0]) or vr1_idx < len(microbatch_groups[1]):
                # Add VR0
                if vr0_idx < len(microbatch_groups[0]):
                    backward_schedule.append(microbatch_groups[0][vr0_idx])
                    vr0_idx += 1

                # Add VR1
                if vr1_idx < len(microbatch_groups[1]):
                    backward_schedule.append(microbatch_groups[1][vr1_idx])
                    vr1_idx += 1

        else:
            # Forward: VR1 → VR0, Backward: VR0 → VR1
            offset = self.pipeline_parallel_size - 1 - pipeline_parallel_rank

            # Initial VR0 microbatches
            for i in range(offset + 1):
                if i < len(microbatch_groups[0]):
                    backward_schedule.append(microbatch_groups[0][i])

            # Interleave remaining
            vr0_idx = offset + 1
            vr1_idx = 0

            while vr0_idx < len(microbatch_groups[0]) or vr1_idx < len(microbatch_groups[1]):
                # Add VR1
                if vr1_idx < len(microbatch_groups[1]):
                    backward_schedule.append(microbatch_groups[1][vr1_idx])
                    vr1_idx += 1

                # Add VR0
                if vr0_idx < len(microbatch_groups[0]):
                    backward_schedule.append(microbatch_groups[0][vr0_idx])
                    vr0_idx += 1

        # Add sync markers
        num_unit = self.pipeline_parallel_size // 2
        if pipeline_parallel_rank == num_unit or pipeline_parallel_rank == num_unit - 1:
            backward_schedule.append(-1)
        else:
            backward_schedule.insert(-1, -1)
        backward_schedule.append(-1)

        return backward_schedule


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def explain_bitpipe_transformation(num_devices: int, total_num_mbs: int):
    """Explain how BitPipe transforms microbatch IDs through V-shaped pipeline"""
    print("\nBitPipe V-Shaped Microbatch Transformation:")
    print("-" * 80)

    num_unit = num_devices // 2
    base_mbs = total_num_mbs // 2

    print(f"\nConfiguration: {base_mbs} base microbatches, {num_devices} devices")
    print(f"Total microbatches (doubled): {total_num_mbs}")
    print(f"\nMicrobatch Grouping by VR:")
    print(f"  VR0: MB0-MB{num_unit-1}         (first {num_unit} base MBs, ranks 0→{num_devices-1})")
    print(f"  VR1: MB{num_unit}-MB{base_mbs-1}      (second {num_unit} base MBs, ranks {num_devices-1}→0)")
    print(f"  VR2: MB{base_mbs}-MB{base_mbs+num_unit-1}    (first {num_unit} base MBs again, ranks {num_devices-1}→0)")
    print(f"  VR3: MB{base_mbs+num_unit}-MB{total_num_mbs-1} (second {num_unit} base MBs again, ranks 0→{num_devices-1})")

    print(f"\nExample: Journey of MB0:")
    print(f"  1. Rank 0 (VR0) → Process first quarter of model")
    print(f"  2. Rank 1 (VR0) → Continue processing")
    print(f"  3. Rank 2 (VR0) → Continue processing")
    print(f"  4. Rank {num_devices-1} (VR0) → Last rank of VR0")
    print(f"     ↓ [Same-device transition: MB0 → MB{base_mbs}]")
    print(f"  5. Rank {num_devices-1} (VR2) → MB{base_mbs} starts second half")
    print(f"  6. Rank {num_devices-2} (VR2) → Continue processing")
    print(f"  7. Rank 1 (VR2) → Continue processing")
    print(f"  8. Rank 0 (VR2) → Complete forward pass")

    print(f"\nWhy doubling (2N)?")
    print(f"  • VR0 and VR1 process {base_mbs} different microbatches concurrently")
    print(f"  • VR2 and VR3 process the SAME data but at different times/paths")
    print(f"  • To satisfy V-shaped scheduling, we need {base_mbs}×4 = {total_num_mbs} total MB slots")
    print(f"  • Result: 2× effective microbatches for better pipeline utilization")


def explain_chimera_transformation(num_devices: int, total_num_mbs: int):
    """Explain how Chimera keeps microbatch IDs constant (no V-shape)"""
    print("\nChimera Linear Microbatch Flow (No Transformation):")
    print("-" * 80)

    num_unit = num_devices // 2
    base_mbs = total_num_mbs  # Chimera doesn't double

    print(f"\nConfiguration: {base_mbs} microbatches, {num_devices} devices")
    print(f"Total microbatches: {base_mbs} (NO doubling)")
    print(f"\nMicrobatch Grouping by VR:")
    print(f"  VR0: MB0-MB{base_mbs-1}  (all MBs in sequential order, ranks 0→{num_devices-1})")
    print(f"  VR1: MB0-MB{base_mbs-1}  (SAME MBs again, ranks {num_devices-1}→0)")

    print(f"\nExample: Journey of MB0:")
    print(f"  1. Rank 0 (VR0) → Process first half of model")
    print(f"  2. Rank 1 (VR0) → Continue processing")
    print(f"  3. Rank 2 (VR0) → Continue processing")
    print(f"  4. Rank {num_devices-1} (VR0) → Last rank, MB0 completes first half")
    print(f"     ↓ [NO transition: MB0 stays MB0]")
    print(f"  5. Rank {num_devices-1} (VR1) → MB0 starts second half")
    print(f"  6. Rank {num_devices-2} (VR1) → Continue processing")
    print(f"  7. Rank 1 (VR1) → Continue processing")
    print(f"  8. Rank 0 (VR1) → Complete forward pass, MB0 done")

    print(f"\nWhy NO doubling?")
    print(f"  • VR0 and VR1 are sequential, not concurrent")
    print(f"  • Each MB follows single path: Rank 0→{num_devices-1} (VR0), then {num_devices-1}→0 (VR1)")
    print(f"  • No need to separate forward and backward paths")
    print(f"  • Total MBs = {base_mbs} (not doubled)")
    print(f"  • Simpler scheduling, easier to debug")


def print_schedule_comparison(bitpipe_sched: List[int], chimera_sched: List[int], total_num_mbs: int, num_devices: int):
    """Compare BitPipe and Chimera schedules side by side with VR annotations"""
    max_len = max(len(bitpipe_sched), len(chimera_sched))
    num_unit = num_devices // 2

    print(f"{'Step':<6} {'BitPipe':<20} {'VR':<6} {'Chimera':<20} {'VR':<6}")
    print(f"{'-'*6} {'-'*20} {'-'*6} {'-'*20} {'-'*6}")

    for i in range(max_len):
        bp = str(bitpipe_sched[i]) if i < len(bitpipe_sched) else "—"
        ch = str(chimera_sched[i]) if i < len(chimera_sched) else "—"

        # Determine VR for BitPipe
        bp_vr = "—"
        if i < len(bitpipe_sched) and bitpipe_sched[i] != -1:
            mb_id = bitpipe_sched[i]
            mb_id_in_group = mb_id % num_devices
            chunk_offset = 0 if mb_id < (total_num_mbs // 2) else 2
            bp_vr = str((mb_id_in_group // num_unit) + chunk_offset)

        # Determine VR for Chimera (simpler)
        ch_vr = "—"
        if i < len(chimera_sched) and chimera_sched[i] != -1:
            ch_vr = "0" if i < (len(chimera_sched) // 2) else "1"

        print(f"{i:<6} {bp:<20} {bp_vr:<6} {ch:<20} {ch_vr:<6}")


def print_schedule_stats(name: str, schedule: List[int]):
    """Print statistics about a schedule"""
    non_sync = [x for x in schedule if x != -1]
    sync_count = sum(1 for x in schedule if x == -1)

    print(f"\n{name}")
    print(f"  Total items:      {len(schedule)}")
    print(f"  Microbatches:     {len(non_sync)}")
    print(f"  Sync markers:     {sync_count}")
    print(f"  Schedule:         {schedule}")


def interactive_mode():
    """Interactive mode for testing different configurations"""
    print_section("BitPipe Scheduling Simulator - Interactive Mode")

    print("This tool simulates microbatch scheduling for:")
    print("  - BitPipe 4-VR with get_microbatch_idx()")
    print("  - Chimera 2-VR with build_chimera_forward_schedule()")
    print()

    while True:
        try:
            # Get user input
            num_microbatches = int(input("Enter number of microbatches (user-specified, will be doubled): "))
            num_devices = int(input("Enter number of devices (must be even): "))

            # Validation
            if num_microbatches <= 0:
                print("❌ Microbatches must be positive!")
                continue

            if num_devices <= 0 or num_devices % 2 != 0:
                print("❌ Devices must be positive and even!")
                continue

            if num_microbatches % num_devices != 0:
                print(f"⚠️  Warning: {num_microbatches} is not divisible by {num_devices}")
                print("   Some ranks may have unbalanced schedules")

            # Calculate total microbatches
            total_num_microbatches = num_microbatches * 2  # Double for bidirectional

            print_section(f"Configuration: {num_microbatches} base MBs → {total_num_microbatches} total MBs, {num_devices} devices")

            # BitPipe simulation
            print_section("BitPipe 4-VR Scheduling")
            bp_scheduler = BitPipeScheduler(total_num_microbatches, num_devices)

            print("Forward Schedule (get_microbatch_idx):\n")
            bp_schedules = []
            for rank in range(num_devices):
                bp_sched = bp_scheduler.get_microbatch_idx(rank)
                bp_schedules.append(bp_sched)
                print_schedule_stats(f"  Rank {rank}", bp_sched)

            print("\n\nBackward Schedule (get_bkmicrobatch_idx):\n")
            for rank in range(num_devices):
                bk_sched = bp_scheduler.get_bkmicrobatch_idx(rank)
                print_schedule_stats(f"  Rank {rank}", bk_sched)

            # Chimera simulation
            print_section("Chimera 2-VR Scheduling")
            ch_scheduler = ChimeraScheduler(total_num_microbatches, num_devices)

            print("Forward Schedule (build_chimera_forward_schedule):\n")
            ch_schedules = []
            for rank in range(num_devices):
                ch_sched = ch_scheduler.build_chimera_forward_schedule(rank)
                ch_schedules.append(ch_sched)
                print_schedule_stats(f"  Rank {rank}", ch_sched)

            print("\n\nBackward Schedule (build_chimera_backward_schedule):\n")
            for rank in range(num_devices):
                bk_sched = ch_scheduler.build_chimera_backward_schedule(rank)
                print_schedule_stats(f"  Rank {rank}", bk_sched)

            # Comparison for Rank 0
            print_section("Comparison: Rank 0 Forward Schedule")
            print_schedule_comparison(bp_schedules[0], ch_schedules[0], total_num_microbatches, num_devices)

            # Analysis
            print_section("Analysis & Explanations")

            # Show BitPipe explanation
            explain_bitpipe_transformation(num_devices, total_num_microbatches)

            # Show Chimera explanation
            explain_chimera_transformation(num_devices, total_num_microbatches)

            print("\n" + "="*80)
            print("Key Insight:")
            print("="*80)
            print(f"BitPipe DOUBLES microbatches because:")
            print(f"  • MB0 goes: Rank 0→{num_devices-1} (VR0) then becomes MB{total_num_microbatches//2} and goes {num_devices-1}→0 (VR2)")
            print(f"  • This V-shaped transformation requires tracking them separately")
            print(f"  • Result: 2N microbatches total for N base microbatches")
            print()
            print(f"Chimera does NOT double microbatches because:")
            print(f"  • MB0 goes: Rank 0→{num_devices-1} (VR0) then continues to {num_devices-1}→0 (VR1)")
            print(f"  • No transformation: MB0 stays MB0 throughout")
            print(f"  • Result: N microbatches total (same as input)")

        except ValueError:
            print("❌ Invalid input! Please enter integers.")
            continue
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

        # Ask to continue
        again = input("\n\nRun another simulation? (y/n): ").strip().lower()
        if again != 'y':
            break

    print("\nThank you for using the scheduler simulator!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="BitPipe Scheduling Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python scheduling_sim.py                          # Interactive mode
          python scheduling_sim.py --num-microbatches 8 --num-devices 4  # Single run
        """)
    )

    parser.add_argument('--num-microbatches', type=int, default=None,
                        help='Number of base microbatches (will be doubled)')
    parser.add_argument('--num-devices', type=int, default=None,
                        help='Number of devices (must be even)')
    parser.add_argument('--interactive', action='store_true', default=None,
                        help='Force interactive mode')

    args = parser.parse_args()

    # Determine mode
    if args.num_microbatches is not None and args.num_devices is not None:
        # Non-interactive mode
        num_microbatches = args.num_microbatches
        num_devices = args.num_devices

        # Validation
        if num_microbatches <= 0 or num_devices <= 0 or num_devices % 2 != 0:
            print("❌ Invalid configuration!")
            print("   - Microbatches must be positive")
            print("   - Devices must be positive and even")
            sys.exit(1)

        # BitPipe doubles, Chimera does NOT
        total_num_microbatches_bp = num_microbatches * 2  # Double for BitPipe
        total_num_microbatches_ch = num_microbatches      # NO doubling for Chimera

        print_section(f"Configuration: {num_microbatches} base MBs, {num_devices} devices")
        print(f"  BitPipe: {total_num_microbatches_bp} total MBs (doubled)")
        print(f"  Chimera: {total_num_microbatches_ch} total MBs (not doubled)\n")

        # Run simulations
        bp_scheduler = BitPipeScheduler(total_num_microbatches_bp, num_devices)
        ch_scheduler = ChimeraScheduler(total_num_microbatches_ch, num_devices)

        print_section("BitPipe 4-VR Forward Schedules")
        for rank in range(num_devices):
            sched = bp_scheduler.get_microbatch_idx(rank)
            print_schedule_stats(f"Rank {rank}", sched)

        print_section("BitPipe 4-VR Backward Schedules")
        for rank in range(num_devices):
            sched = bp_scheduler.get_bkmicrobatch_idx(rank)
            print_schedule_stats(f"Rank {rank}", sched)

        print_section("Chimera 2-VR Forward Schedules (NEW ALGORITHM)")
        for rank in range(num_devices):
            sched = ch_scheduler.chimera_get_microbatch_idx(rank)
            print_schedule_stats(f"Rank {rank}", sched)

        print_section("Chimera 2-VR Backward Schedules (NEW ALGORITHM)")
        for rank in range(num_devices):
            sched = ch_scheduler.chimera_get_bkmicrobatch_idx(rank)
            print_schedule_stats(f"Rank {rank}", sched)

    else:
        # Interactive mode
        interactive_mode()


if __name__ == '__main__':
    main()
