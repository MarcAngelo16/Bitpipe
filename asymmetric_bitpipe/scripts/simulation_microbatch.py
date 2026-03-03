#!/usr/bin/env python3
"""
Simulation of BitPipe Microbatch Assignment Pattern

This script simulates how microbatches are assigned to different Virtual Ranks (VRs)
across pipeline stages in BitPipe's bidirectional interleaved pipeline parallelism.

The functions here are EXACT COPIES from megatron/core/pipeline_parallel/bitpipe_schedule.py
to help understand the scheduling logic.

Usage:
    python simulation_microbatch.py --pipeline-size 4 --num-microbatches 8
"""

import argparse
from typing import List, Dict
from collections import defaultdict


class BitPipeScheduleSimulator:
    """
    Simulates BitPipe scheduling logic with exact same functions as the main program.

    This class encapsulates the scheduling context (pipeline_parallel_size,
    total_num_microbatches, num_model_chunks) similar to how they exist as
    closure variables in the main BitPipe scheduler.
    """

    def __init__(self, pipeline_parallel_size: int, num_microbatches: int):
        """
        Initialize the simulator.

        Args:
            pipeline_parallel_size: Number of pipeline stages/devices
            num_microbatches: Base number of microbatches (before BitPipe doubling)
        """
        self.pipeline_parallel_size = pipeline_parallel_size
        self.num_model_chunks = 4  # BitPipe always uses 4 VRs per device
        self.num_microbatches = num_microbatches
        self.total_num_microbatches = num_microbatches * (self.num_model_chunks // 2)

    def get_model_chunk_id(self, microbatch_id: int) -> int:
        """
        EXACT COPY from bitpipe_schedule.py:342-349

        Helper method to get the model chunk ID (VR) given the microbatch ID.

        Args:
            microbatch_id: The microbatch index (0 to total_num_microbatches-1)

        Returns:
            model_chunk_id: Which VR (0-3) should process this microbatch, or -1 for sync
        """
        microbatch_id_in_group = microbatch_id % (self.pipeline_parallel_size)
        chunk_offset = 0 if microbatch_id < (self.total_num_microbatches // 2) else 2
        model_chunk_id = microbatch_id_in_group // (self.pipeline_parallel_size // 2)
        model_chunk_id += chunk_offset

        if microbatch_id == -1:
            model_chunk_id = -1

        return model_chunk_id

    def get_microbatch(self, total_num_microbatches: int) -> List[List[int]]:
        """
        EXACT COPY from bitpipe_schedule.py:238-243

        Groups microbatch IDs by which VR they belong to.

        Args:
            total_num_microbatches: Total number of microbatches

        Returns:
            microbatch_id01: List of 4 lists, each containing microbatch IDs for each VR
                            microbatch_id01[0] = VR0's microbatches
                            microbatch_id01[1] = VR1's microbatches
                            microbatch_id01[2] = VR2's microbatches
                            microbatch_id01[3] = VR3's microbatches
        """
        microbatch_id01 = [[] for _ in range(self.num_model_chunks)]
        for i in range(total_num_microbatches):
            model_chunk_id = self.get_model_chunk_id(i)
            microbatch_id01[model_chunk_id].append(i)
        return microbatch_id01

    def get_microbatch_idx(self, total_num_microbatches: int, pipeline_parallel_rank: int) -> List[int]:
        """
        EXACT COPY from bitpipe_schedule.py:245-289

        Generate forward pass execution schedule for a given device rank.

        Args:
            total_num_microbatches: Total number of microbatches
            pipeline_parallel_rank: Device rank in pipeline (0 to pipeline_parallel_size-1)

        Returns:
            microbatch_idx: List of microbatch IDs in execution order for forward passes

        Example output for device 0 with 8 devices, 16 total microbatches:
            [0, 1, 2, 10, 3, 11, 8, 9, 4, 5, 6, 14, 7, 15, 12, 13]
        """
        microbatch_idx = []
        microbatch_id01 = self.get_microbatch(total_num_microbatches)
        i_loop = total_num_microbatches // self.pipeline_parallel_size // 2

        num_unit = self.pipeline_parallel_size // 2
        i_half = pipeline_parallel_rank // num_unit
        num_initial = (
            num_unit - pipeline_parallel_rank
            if pipeline_parallel_rank < num_unit
            else pipeline_parallel_rank - num_unit + 1
        )

        # Comments from original code showing example patterns:
        # [0, 1, 2, 10, 3, 11, 8, 9, 4, 5, 6, 14, 7, 15, 12, 13]
        # [0, 2, 1, 3, 10, 8, 11, 9, 4, 6, 5, 7, 14, 12, 15, 13]
        # [2, 0, 3, 1, 8, 10, 9, 11, 6, 4, 7, 5, 12, 14, 13, 15]
        # [2, 3, 0, 8, 1, 9, 10, 11, 6, 7, 4, 12, 5, 13, 14, 15]

        for j in range(i_loop):
            for i in range(num_initial):
                microbatch_idx.append(microbatch_id01[i_half][j * num_unit + i])
            for i in range(num_unit - num_initial):
                microbatch_idx.append(microbatch_id01[1 - i_half][j * num_unit + i])
                microbatch_idx.append(microbatch_id01[i_half][j * num_unit + i + num_initial])
            for i in range(num_initial):
                microbatch_idx.append(microbatch_id01[1 - i_half][j * num_unit + i + (num_unit - num_initial)])
                microbatch_idx.append(microbatch_id01[3 - i_half][j * num_unit + i])
            for i in range(num_unit - num_initial):
                microbatch_idx.append(microbatch_id01[2 + i_half][j * num_unit + i])
                microbatch_idx.append(microbatch_id01[3 - i_half][j * num_unit + i + num_initial])
            for i in range(num_initial):
                microbatch_idx.append(microbatch_id01[2 + i_half][j * num_unit + i + (num_unit - num_initial)])

        return microbatch_idx

    def get_bkmicrobatch_idx(self, total_num_microbatches: int, pipeline_parallel_rank: int) -> List[int]:
        """
        EXACT COPY from bitpipe_schedule.py:292-340

        Generate backward pass execution schedule for a given device rank.
        Includes -1 markers for gradient synchronization points.

        Args:
            total_num_microbatches: Total number of microbatches
            pipeline_parallel_rank: Device rank in pipeline (0 to pipeline_parallel_size-1)

        Returns:
            microbatch_idx: List of microbatch IDs in execution order for backward passes
                           -1 indicates gradient synchronization point

        Example output for device 0 with 8 devices, 16 total microbatches:
            [8, 9, 10, 2, 11, 3, 0, 1, 12, 13, 14, 6, 15, 7, 4, 5, -1, -1]
        """
        microbatch_idx = []
        microbatch_id01 = self.get_microbatch(total_num_microbatches)
        i_loop = total_num_microbatches // self.pipeline_parallel_size // 2

        num_unit = self.pipeline_parallel_size // 2
        i_half = pipeline_parallel_rank // num_unit
        num_initial = (
            num_unit - pipeline_parallel_rank
            if pipeline_parallel_rank < num_unit
            else pipeline_parallel_rank - num_unit + 1
        )

        # Comments from original code showing example patterns:
        # [8, 9, 10, 2, 11, 3, 0, 1, 12, 13, 14, 6, 15, 7, 4, 5]
        # [8, 10, 9, 11, 2, 0, 3, 1, 12, 14, 13, 15, 6, 4, 7, 5]
        # [10, 8, 11, 9, 0, 2, 1, 3, 14, 12, 15, 13, 4, 6, 5, 7]
        # [10, 11, 8, 0, 9, 1, 2, 3, 14, 15, 12, 4, 13, 5, 6, 7]

        for j in range(i_loop):
            for k in range(num_initial):
                microbatch_idx.append(microbatch_id01[2 + i_half][j * num_unit + k])
            for k in range(num_unit - num_initial):
                microbatch_idx.append(microbatch_id01[3 - i_half][j * num_unit + k])
                microbatch_idx.append(microbatch_id01[2 + i_half][j * num_unit + k + num_initial])
            for k in range(num_initial):
                microbatch_idx.append(microbatch_id01[3 - i_half][(j * num_unit + k + num_unit - num_initial)])
                microbatch_idx.append(microbatch_id01[1 - i_half][j * num_unit + k])
            for k in range(num_unit - num_initial):
                microbatch_idx.append(microbatch_id01[i_half][j * num_unit + k])
                microbatch_idx.append(microbatch_id01[1 - i_half][j * num_unit + k + num_initial])
            for k in range(num_initial):
                microbatch_idx.append(microbatch_id01[i_half][j * num_unit + k + num_unit - num_initial])

        # Eager sync markers
        if pipeline_parallel_rank == self.pipeline_parallel_size // 2 or \
           pipeline_parallel_rank == self.pipeline_parallel_size // 2 - 1:
            microbatch_idx.append(-1)
        else:  # second last sync
            microbatch_idx.insert(-1, -1)
        microbatch_idx.append(-1)  # last sync

        return microbatch_idx


def simulate_microbatch_assignment(pipeline_parallel_size: int, num_microbatches: int):
    """
    Simulates and visualizes microbatch assignment across all devices and VRs.

    Args:
        pipeline_parallel_size: Number of pipeline stages/devices
        num_microbatches: Base number of microbatches (before BitPipe doubling)
    """
    # Create simulator instance
    simulator = BitPipeScheduleSimulator(pipeline_parallel_size, num_microbatches)

    print("=" * 100)
    print(f"BitPipe Microbatch Scheduling Simulation")
    print("=" * 100)
    print(f"Pipeline Parallel Size: {pipeline_parallel_size} devices")
    print(f"Base Microbatches: {num_microbatches}")
    print(f"Total Microbatches (BitPipe doubled): {simulator.total_num_microbatches}")
    print(f"Virtual Ranks per Device: {simulator.num_model_chunks}")
    print("=" * 100)
    print()

    # Step 1: Show VR assignment (which microbatches belong to which VR)
    print("STEP 1: MICROBATCH ASSIGNMENT BY VIRTUAL RANK (VR)")
    print("-" * 100)
    print("This shows which VR each microbatch is assigned to based on get_model_chunk_id()")
    print()

    microbatch_id01 = simulator.get_microbatch(simulator.total_num_microbatches)

    for vr_id in range(simulator.num_model_chunks):
        microbatches = microbatch_id01[vr_id]
        pipeline_half = "Forward Pipeline (1st half)" if vr_id < 2 else "Backward Pipeline (2nd half)"
        print(f"VR{vr_id} ({pipeline_half}):")
        print(f"  Handles {len(microbatches)} microbatches: {microbatches}")
        print()

    print("=" * 100)
    print()

    # Step 2: Show per-device forward and backward schedules
    print("STEP 2: PER-DEVICE FORWARD AND BACKWARD SCHEDULES")
    print("-" * 100)
    print("This shows the execution order for each device rank")
    print("Generated by get_microbatch_idx() for forward and get_bkmicrobatch_idx() for backward")
    print()

    for device_rank in range(pipeline_parallel_size):
        print(f"{'=' * 100}")
        print(f"DEVICE {device_rank}")
        print(f"{'=' * 100}")

        # Get forward schedule
        forward_schedule = simulator.get_microbatch_idx(
            simulator.total_num_microbatches,
            device_rank
        )

        # Get backward schedule
        backward_schedule = simulator.get_bkmicrobatch_idx(
            simulator.total_num_microbatches,
            device_rank
        )

        print(f"\nForward Schedule (length={len(forward_schedule)}):")
        print(f"  {forward_schedule}")

        print(f"\nBackward Schedule (length={len(backward_schedule)}):")
        print(f"  {backward_schedule}")
        print(f"  Note: -1 indicates gradient synchronization point")

        # Show detailed execution order with VR information
        print(f"\nDetailed Forward Execution Order:")
        for time_step, mb_id in enumerate(forward_schedule):
            chunk_id = simulator.get_model_chunk_id(mb_id)
            print(f"  Step {time_step:2d}: Microbatch {mb_id:2d} → VR{chunk_id}")

        print(f"\nDetailed Backward Execution Order:")
        for time_step, mb_id in enumerate(backward_schedule):
            if mb_id == -1:
                print(f"  Step {time_step:2d}: GRADIENT SYNC")
            else:
                chunk_id = simulator.get_model_chunk_id(mb_id)
                print(f"  Step {time_step:2d}: Microbatch {mb_id:2d} → VR{chunk_id}")

        print()

    print("=" * 100)


def run_custom_simulation():
    """
    Interactive mode for custom simulations.
    """
    print("\nCustom Simulation Mode")
    print("-" * 50)

    while True:
        try:
            pp_size = int(input("\nEnter pipeline parallel size (or 0 to exit): "))
            if pp_size == 0:
                break

            num_mb = int(input("Enter base number of microbatches: "))

            simulator = BitPipeScheduleSimulator(pp_size, num_mb)

            print(f"\nSimulator initialized:")
            print(f"  Pipeline size: {pp_size}")
            print(f"  Base microbatches: {num_mb}")
            print(f"  Total microbatches: {simulator.total_num_microbatches}")

            mb_id = int(input(f"\nEnter microbatch ID (0-{simulator.total_num_microbatches-1}): "))

            if mb_id < 0 or mb_id >= simulator.total_num_microbatches:
                print(f"Error: Microbatch ID must be between 0 and {simulator.total_num_microbatches-1}")
                continue

            chunk_id = simulator.get_model_chunk_id(mb_id)

            print(f"\n  Microbatch {mb_id} → VR{chunk_id}")

            # Show calculation breakdown
            half = "First Half" if mb_id < simulator.total_num_microbatches // 2 else "Second Half"
            chunk_offset = 0 if mb_id < simulator.total_num_microbatches // 2 else 2
            group = mb_id % pp_size
            base_chunk = group // (pp_size // 2)

            print(f"\n  Calculation:")
            print(f"    - Half: {half} (offset = {chunk_offset})")
            print(f"    - Group ID: {mb_id} % {pp_size} = {group}")
            print(f"    - Base chunk: {group} // {pp_size // 2} = {base_chunk}")
            print(f"    - Final VR: {base_chunk} + {chunk_offset} = {chunk_id}")

        except (ValueError, KeyboardInterrupt):
            break


def main():
    parser = argparse.ArgumentParser(
        description="Simulate BitPipe microbatch scheduling with exact functions from main program",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simulate 4 devices with 8 base microbatches (16 total) - DEFAULT
  python simulation_microbatch.py --pipeline-size 4 --num-microbatches 8

  # Simulate 8 devices with 8 base microbatches (16 total)
  python simulation_microbatch.py --pipeline-size 8 --num-microbatches 8

  # Simulate 4 devices with 4 base microbatches (8 total)
  python simulation_microbatch.py --pipeline-size 4 --num-microbatches 4

  # Interactive custom mode
  python simulation_microbatch.py --custom
        """
    )

    parser.add_argument(
        '--pipeline-size',
        type=int,
        default=4,
        help='Number of pipeline parallel stages/devices (must be even, default: 4)'
    )

    parser.add_argument(
        '--num-microbatches',
        type=int,
        default=8,
        help='Base number of microbatches before BitPipe doubling (default: 8, total will be 16)'
    )

    parser.add_argument(
        '--custom',
        action='store_true',
        help='Run in custom interactive mode'
    )

    args = parser.parse_args()

    if args.custom:
        run_custom_simulation()
    else:
        # Validate inputs
        if args.pipeline_size % 2 != 0:
            print(f"Error: pipeline-size must be even, got {args.pipeline_size}")
            return

        if args.pipeline_size < 2:
            print(f"Error: pipeline-size must be >= 2, got {args.pipeline_size}")
            return

        if args.num_microbatches < 1:
            print(f"Error: num-microbatches must be >= 1, got {args.num_microbatches}")
            return

        simulate_microbatch_assignment(args.pipeline_size, args.num_microbatches)


if __name__ == "__main__":
    main()
