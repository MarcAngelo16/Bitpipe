#!/usr/bin/env python
"""
Profile Analysis and Visualization Tool

This script analyzes and visualizes BitPipe/Chimera profiling data to understand:
- Pipeline execution timeline (including BD allreduce sync blocks)
- Communication patterns
- Performance bottlenecks
- Pipeline efficiency

Supports both old profiles (wall-clock timing) and new profiles (cuda_event timing).
Sync events (-1 markers) are visualized as BD Sync blocks on the timeline.
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from datetime import datetime
import pandas as pd

# Paths relative to this script's location, so the script works regardless
# of which directory it is invoked from.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PROFILE_DIR = os.path.join(_SCRIPT_DIR, "..", "profiles", "raw")
_DEFAULT_VIZ_DIR     = os.path.join(_SCRIPT_DIR, "..", "visualizations")

# ─────────────────────────────────────────────────────────────────────────────
# Iteration selection for cross-schedule comparison
# Set a specific iteration number per schedule type to pin which profile is
# used in the comparison table, or None to average all available iterations.
#
# Example — compare chimera iter 3 vs 1f1b iter 5:
#   COMPARE_ITERS = {'chimera': 3, '1f1b': 5, ...}
# ─────────────────────────────────────────────────────────────────────────────
COMPARE_ITERS = {
    'bitpipe':      None,
    'bitpipe_asym': None,
    'chimera':      None,
    'chimera_asym': None,
    '1f1b':         None,
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_profile_data(profile_dir=None):
    if profile_dir is None:
        profile_dir = _DEFAULT_PROFILE_DIR
    """Load all profile files from directory"""
    profile_files = glob.glob(os.path.join(profile_dir, "*.json"))
    profiles = []

    for file in sorted(profile_files):
        with open(file, 'r') as f:
            data = json.load(f)
            filename = os.path.basename(file)
            data['filename'] = filename
            data['profile_type'] = identify_profile_type(data, filename)
            profiles.append(data)

    return profiles

def identify_profile_type(profile, filename=""):
    """Identify if profile is training or validation"""
    if "_train_" in filename:
        return "training"
    elif "_validation_" in filename or "_val_" in filename:
        return "validation"
    num_backward = profile['summary']['num_backward_passes']
    return "training" if num_backward > 0 else "validation"

def get_schedule_type(profile):
    """Get the schedule type (bitpipe, bitpipe_asym, chimera, chimera_asym, or 1f1b)"""
    return profile['metadata'].get('schedule_type', 'unknown')

def get_timing_method(profile):
    """Return 'cuda_event' for new profiles, 'wall_clock' for old ones"""
    return profile['metadata'].get('timing_method', 'wall_clock')

def get_sync_events(profile):
    """Return sync_events list (empty list for old profiles without it)"""
    return profile.get('sync_events', [])

def get_summary_value(profile, key, default=0.0):
    """Safely get a summary value with fallback for old profiles"""
    return profile['summary'].get(key, default)

def get_iteration(profile):
    """Extract iteration number from profile filename, or None if not present."""
    import re
    filename = profile.get('filename', '')
    m = re.search(r'_iter(\d+)_', filename)
    return int(m.group(1)) if m else None

def filter_profiles_for_comparison(profiles_by_schedule, iter_map):
    """
    Filter each schedule's profile list to a specific iteration when requested.

    Args:
        profiles_by_schedule: dict of {schedule_name: [profile, ...]}
        iter_map: COMPARE_ITERS dict — {schedule_name: int | None}

    Returns:
        dict of {schedule_name: [profile, ...]} with iteration filtering applied.
        Prints a note for each schedule showing which iteration is being used.
    """
    result = {}
    for name, profs in profiles_by_schedule.items():
        target_iter = iter_map.get(name, None)
        if target_iter is None:
            result[name] = profs
            iters = sorted(set(i for p in profs if (i := get_iteration(p)) is not None))
            iter_label = f"all iters {iters}" if iters else "unknown iter"
        else:
            filtered = [p for p in profs if get_iteration(p) == target_iter]
            if not filtered:
                available = sorted(set(i for p in profs if (i := get_iteration(p)) is not None))
                print(f"  [WARNING] {name}: iter {target_iter} not found "
                      f"(available: {available}). Using all.")
                filtered = profs
            result[name] = filtered
            iter_label = f"iter {target_iter}"
        print(f"  {name}: using {iter_label} ({len(result[name])} profiles)")
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Timeline visualization
# ─────────────────────────────────────────────────────────────────────────────

def create_timeline_visualization(profiles, output_file="timeline.png"):
    """Create a timeline visualization of microbatch execution and sync blocks"""

    training_profiles = [p for p in profiles if p.get('profile_type', 'validation') == "training"]

    if not training_profiles:
        print("No training profiles found!")
        return

    bitpipe_profiles      = {p['metadata']['rank']: p for p in training_profiles if get_schedule_type(p) == "bitpipe"}
    bitpipe_asym_profiles = {p['metadata']['rank']: p for p in training_profiles if get_schedule_type(p) == "bitpipe_asym"}
    chimera_profiles      = {p['metadata']['rank']: p for p in training_profiles if get_schedule_type(p) == "chimera"}
    chimera_asym_profiles = {p['metadata']['rank']: p for p in training_profiles if get_schedule_type(p) == "chimera_asym"}
    f1b_profiles          = {p['metadata']['rank']: p for p in training_profiles if get_schedule_type(p) == "1f1b"}

    if bitpipe_asym_profiles:
        rank_profiles = bitpipe_asym_profiles
        schedule_type = "BitPipe Asymmetric"
    elif chimera_asym_profiles:
        rank_profiles = chimera_asym_profiles
        schedule_type = "Chimera 2-VR Asymmetric"
    elif chimera_profiles:
        rank_profiles = chimera_profiles
        schedule_type = "Chimera 2-VR"
    elif bitpipe_profiles:
        rank_profiles = bitpipe_profiles
        schedule_type = "BitPipe"
    else:
        rank_profiles = f1b_profiles
        schedule_type = "Standard 1F1B"

    fig, ax = plt.subplots(figsize=(18, 10))

    # Color scheme
    if schedule_type in ["BitPipe", "BitPipe Asymmetric", "Chimera 2-VR", "Chimera 2-VR Asymmetric"]:
        forward_colors  = {0: '#1f77b4', 1: '#ff7f0e'}
        backward_colors = {0: '#2ca02c', 1: '#d62728'}
    else:
        forward_colors  = {0: '#1f77b4'}
        backward_colors = {0: '#2ca02c'}

    sync_color = '#9467bd'  # purple for BD allreduce sync blocks

    # Compute x-axis bounds from microbatch events
    all_events = []
    for profile in rank_profiles.values():
        all_events.extend(profile['microbatch_events'])

    if all_events:
        earliest_start = min(e['start_time'] for e in all_events)
        latest_end     = max(e['end_time']   for e in all_events)
        # Also extend to cover sync events if present
        for profile in rank_profiles.values():
            for se in get_sync_events(profile):
                if se['wall_clock_end'] > latest_end:
                    latest_end = se['wall_clock_end']
        computation_duration = latest_end - earliest_start
        padding = computation_duration * 0.05
        x_min = max(0, earliest_start - padding)
        x_max = latest_end + padding
        print(f"Timeline bounds: {earliest_start:.4f}s to {latest_end:.4f}s (duration: {computation_duration:.4f}s)")
        print(f"Plot x-axis: {x_min:.4f}s to {x_max:.4f}s")
    else:
        x_min, x_max = 0, 1

    y_positions = {}
    row_height  = 1.8   # total height per rank row
    bar_height  = 0.7   # individual compute bar height

    for rank in sorted(rank_profiles.keys()):
        y_base = rank * row_height
        y_positions[rank] = y_base
        profile = rank_profiles[rank]

        # ── Microbatch compute bars ──────────────────────────────────────────
        for event in profile['microbatch_events']:
            start    = event['start_time']
            duration = event['end_time'] - event['start_time']
            pipeline_id = event.get('pipeline_id', 0)

            if event['phase'] == 'forward':
                color   = forward_colors.get(pipeline_id, '#1f77b4')
                y_off   = 0.0
            else:
                color   = backward_colors.get(pipeline_id, '#2ca02c')
                y_off   = bar_height + 0.05

            rect = patches.Rectangle(
                (start, y_base + y_off), duration, bar_height,
                linewidth=1, edgecolor='black', facecolor=color, alpha=0.7
            )
            ax.add_patch(rect)

            mb_id   = event['microbatch_id']
            chunk_id = event.get('model_chunk_id', 'N/A')
            ax.text(
                start + duration / 2, y_base + y_off + bar_height / 2,
                f"MB{mb_id}\nC{chunk_id}",
                ha='center', va='center', fontsize=6, weight='bold'
            )

        # ── BD Allreduce sync blocks (new: from sync_events) ─────────────────
        for se in get_sync_events(profile):
            start    = se['wall_clock_start']
            end      = se['wall_clock_end']
            duration = end - start
            if duration <= 0:
                continue

            # Span the full row height so it stands out over compute bars
            rect = patches.Rectangle(
                (start, y_base - 0.05), duration, row_height - 0.1,
                linewidth=2, edgecolor=sync_color, facecolor=sync_color,
                alpha=0.25, hatch='//', zorder=3
            )
            ax.add_patch(rect)

            total_ms = se.get('total_duration_ms', 0.0)
            label = f"BD\nSync\n{total_ms:.1f}ms"
            ax.text(
                start + duration / 2, y_base + row_height / 2,
                label, ha='center', va='center',
                fontsize=6, color=sync_color, weight='bold', zorder=4
            )

    # ── Phase transition lines ───────────────────────────────────────────────
    if rank_profiles:
        first_profile = list(rank_profiles.values())[0]
        for transition in first_profile['phase_transitions']:
            ax.axvline(x=transition['timestamp'], color='gray', linestyle='--', alpha=0.5)
            ax.text(
                transition['timestamp'],
                max(y_positions.values()) + row_height,
                transition['phase_name'],
                rotation=45, ha='right', fontsize=8
            )

    # ── Formatting ───────────────────────────────────────────────────────────
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Rank', fontsize=12)
    first_profile = list(rank_profiles.values())[0]
    timing_label  = get_timing_method(first_profile).replace('_', ' ').title()
    synced_label  = 'synced t=0' if first_profile['metadata'].get('synchronized', False) else 'unsynchronized clocks'
    ax.set_title(
        f'{schedule_type} Pipeline Execution Timeline  [{timing_label} | {synced_label}]',
        fontsize=14, weight='bold'
    )

    ax.set_yticks([y_positions[r] + row_height / 2 for r in sorted(y_positions.keys())])
    ax.set_yticklabels([f'Rank {r}' for r in sorted(y_positions.keys())])
    ax.set_xlim(x_min, x_max)

    if y_positions:
        ax.set_ylim(-0.3, max(y_positions.values()) + row_height + 0.5)

    # Legend
    if schedule_type in ["BitPipe", "BitPipe Asymmetric", "Chimera 2-VR", "Chimera 2-VR Asymmetric"]:
        legend_elements = [
            patches.Patch(facecolor=forward_colors[0],  alpha=0.7, label='Pipeline 0 Forward'),
            patches.Patch(facecolor=forward_colors[1],  alpha=0.7, label='Pipeline 1 Forward'),
            patches.Patch(facecolor=backward_colors[0], alpha=0.7, label='Pipeline 0 Backward'),
            patches.Patch(facecolor=backward_colors[1], alpha=0.7, label='Pipeline 1 Backward'),
            patches.Patch(facecolor=sync_color, alpha=0.4, hatch='//', label='BD Allreduce Sync (-1)'),
        ]
    else:
        legend_elements = [
            patches.Patch(facecolor=forward_colors[0],  alpha=0.7, label='Forward Pass'),
            patches.Patch(facecolor=backward_colors[0], alpha=0.7, label='Backward Pass'),
            patches.Patch(facecolor=sync_color, alpha=0.4, hatch='//', label='BD Allreduce Sync (-1)'),
        ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    base_name = output_file.replace('.png', '')
    schedule_suffix = {
        "BitPipe": "bitpipe",
        "BitPipe Asymmetric": "bitpipe_asym",
        "Chimera 2-VR": "chimera",
        "Chimera 2-VR Asymmetric": "chimera_asym",
        "Standard 1F1B": "1f1b",
    }.get(schedule_type, 'unknown')
    final_output_file = f"{base_name}_{schedule_suffix}.png"
    plt.savefig(final_output_file, dpi=300, bbox_inches='tight')
    print(f"{schedule_type} timeline visualization saved to {final_output_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Sync event analysis
# ─────────────────────────────────────────────────────────────────────────────

def create_sync_analysis(profiles, output_file="sync_analysis.png"):
    """Visualize BD allreduce sync event breakdown across ranks.

    Shows:
      - Total sync time per rank
      - Sync overhead as % of total pipeline time
      - Per-chunk allreduce duration heatmap across ranks
      - Time budget breakdown (compute / p2p / sync)
    """
    training_profiles = [p for p in profiles if p.get('profile_type', 'validation') == "training"]
    profiles_with_sync = [p for p in training_profiles if get_sync_events(p)]

    if not profiles_with_sync:
        print("No sync events found in profiles (either old profile format or no -1 markers executed).")
        return

    # Collect per-rank data
    ranks           = sorted(set(p['metadata']['rank'] for p in profiles_with_sync))
    rank_profile    = {p['metadata']['rank']: p for p in profiles_with_sync}

    total_sync_ms   = []
    sync_pct        = []
    compute_pct     = []
    p2p_pct         = []
    idle_pct        = []

    for r in ranks:
        p        = rank_profile[r]
        total_s  = p['metadata']['total_time']
        fwd_s    = p['summary']['total_forward_time']
        bwd_s    = p['summary']['total_backward_time']
        p2p_s    = p['summary']['total_p2p_time']
        sync_ms  = get_summary_value(p, 'total_sync_time_ms', 0.0)
        sync_s   = sync_ms / 1000.0

        total_sync_ms.append(sync_ms)
        sync_pct.append(sync_s / total_s * 100)
        compute_pct.append((fwd_s + bwd_s) / total_s * 100)
        p2p_pct.append(p2p_s / total_s * 100)
        idle_pct.append(max(0.0, 100 - (fwd_s + bwd_s + p2p_s + sync_s) / total_s * 100))

    # Collect per-chunk allreduce durations for heatmap
    # Structure: chunk_id -> list of (rank, duration_ms)
    all_chunk_ids = sorted(set(
        chunk['chunk_id']
        for p in profiles_with_sync
        for se in get_sync_events(p)
        for chunk in se.get('chunk_durations_ms', [])
    ))

    # Build matrix [rank x chunk]
    chunk_matrix = np.zeros((len(ranks), len(all_chunk_ids)))
    chunk_counts  = np.zeros((len(ranks), len(all_chunk_ids)))
    for ri, r in enumerate(ranks):
        p = rank_profile[r]
        for se in get_sync_events(p):
            for chunk_entry in se.get('chunk_durations_ms', []):
                cid = chunk_entry['chunk_id']
                if cid in all_chunk_ids:
                    ci = all_chunk_ids.index(cid)
                    chunk_matrix[ri, ci] += chunk_entry['duration_ms']
                    chunk_counts[ri, ci] += 1
    # Average over multiple sync events
    with np.errstate(invalid='ignore'):
        chunk_matrix = np.where(chunk_counts > 0, chunk_matrix / chunk_counts, 0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    schedule_type = get_schedule_type(profiles_with_sync[0])
    fig.suptitle(f'BD Allreduce Sync Analysis  [{schedule_type.upper()}]', fontsize=14, weight='bold')

    x = np.arange(len(ranks))
    rank_labels = [f'Rank {r}' for r in ranks]

    # ── 1. Total sync time per rank (ms) ────────────────────────────────────
    ax = axes[0, 0]
    bars = ax.bar(x, total_sync_ms, color='#9467bd', alpha=0.8, edgecolor='black')
    ax.set_title('Total BD Allreduce Time per Rank')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Total Sync Time (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(rank_labels)
    ax.bar_label(bars, fmt='%.1fms', padding=3, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # ── 2. Sync overhead % ───────────────────────────────────────────────────
    ax = axes[0, 1]
    bars = ax.bar(x, sync_pct, color='#e377c2', alpha=0.8, edgecolor='black')
    ax.set_title('BD Sync Overhead (% of total time)')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Overhead (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(rank_labels)
    ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # ── 3. Per-chunk allreduce heatmap ───────────────────────────────────────
    ax = axes[1, 0]
    if all_chunk_ids:
        im = ax.imshow(chunk_matrix, cmap='YlOrRd', aspect='auto')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Avg Duration (ms)')
        ax.set_xticks(range(len(all_chunk_ids)))
        ax.set_xticklabels([f'VR{c}' for c in all_chunk_ids])
        ax.set_yticks(range(len(ranks)))
        ax.set_yticklabels(rank_labels)
        ax.set_title('Avg Per-Chunk Allreduce Duration (ms)')
        ax.set_xlabel('Model Chunk (VR)')
        ax.set_ylabel('Rank')
        for ri in range(len(ranks)):
            for ci in range(len(all_chunk_ids)):
                if chunk_matrix[ri, ci] > 0:
                    ax.text(ci, ri, f'{chunk_matrix[ri, ci]:.1f}',
                            ha='center', va='center', fontsize=9, color='black')
    else:
        ax.text(0.5, 0.5, 'No chunk data\n(chunk_durations_ms empty)',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Per-Chunk Allreduce Duration')

    # ── 4. Time budget breakdown (stacked bar) ───────────────────────────────
    ax = axes[1, 1]
    bottom = np.zeros(len(ranks))
    bar_data = [
        (compute_pct, '#1f77b4', 'Compute (fwd+bwd)'),
        (p2p_pct,     '#ff7f0e', 'P2P Comm'),
        (sync_pct,    '#9467bd', 'BD Sync (-1)'),
        (idle_pct,    '#cccccc', 'Idle/Other'),
    ]
    for values, color, label in bar_data:
        ax.bar(x, values, bottom=bottom, color=color, label=label, alpha=0.85, edgecolor='black', linewidth=0.5)
        bottom += np.array(values)
    ax.set_title('Time Budget Breakdown per Rank')
    ax.set_xlabel('Rank')
    ax.set_ylabel('% of total time')
    ax.set_xticks(x)
    ax.set_xticklabels(rank_labels)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Sync analysis saved to {output_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline efficiency analysis (text)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_pipeline_efficiency(profiles):
    """Analyze pipeline efficiency metrics"""

    print("\n" + "="*60)
    print("PIPELINE PERFORMANCE ANALYSIS")
    print("="*60)

    training_profiles   = [p for p in profiles if p.get('profile_type', 'validation') == "training"]
    validation_profiles = [p for p in profiles if p.get('profile_type', 'validation') == "validation"]

    bitpipe_profiles      = [p for p in training_profiles if get_schedule_type(p) == "bitpipe"]
    bitpipe_asym_profiles = [p for p in training_profiles if get_schedule_type(p) == "bitpipe_asym"]
    chimera_profiles      = [p for p in training_profiles if get_schedule_type(p) == "chimera"]
    chimera_asym_profiles = [p for p in training_profiles if get_schedule_type(p) == "chimera_asym"]
    f1b_profiles          = [p for p in training_profiles if get_schedule_type(p) == "1f1b"]

    print(f"\nFound {len(training_profiles)} training profiles and {len(validation_profiles)} validation profiles")
    print(f"BitPipe: {len(bitpipe_profiles)}, BitPipe Asym: {len(bitpipe_asym_profiles)}, "
          f"Chimera: {len(chimera_profiles)}, Chimera Asym: {len(chimera_asym_profiles)}, "
          f"1F1B: {len(f1b_profiles)}")

    if training_profiles:
        print("\n--- TRAINING PERFORMANCE ---")

        total_time   = []
        forward_time = []
        backward_time = []
        p2p_time     = []
        sync_time_ms = []

        for p in training_profiles:
            rank    = p['metadata']['rank']
            summary = p['summary']
            t_total = p['metadata']['total_time']

            t_fwd   = summary['total_forward_time']
            t_bwd   = summary['total_backward_time']
            t_p2p   = summary['total_p2p_time']
            t_sync  = get_summary_value(p, 'total_sync_time_ms', 0.0)  # ms
            n_sync  = int(get_summary_value(p, 'num_sync_events', 0))
            timing  = get_timing_method(p)
            it      = get_iteration(p)
            iter_tag = f"  iter={it}" if it is not None else ""

            total_time.append(t_total)
            forward_time.append(t_fwd)
            backward_time.append(t_bwd)
            p2p_time.append(t_p2p)
            sync_time_ms.append(t_sync)

            schedule      = get_schedule_type(p)
            t_sync_s      = t_sync / 1000.0
            needs_sync    = schedule in ('chimera', 'chimera_asym', 'bitpipe', 'bitpipe_asym')
            useful_time   = t_fwd + t_bwd + (t_sync_s if needs_sync else 0.0)
            efficiency    = useful_time / t_total * 100
            comm_overhead = t_p2p / t_total * 100
            sync_overhead = t_sync_s / t_total * 100

            print(f"\nRank {rank}  [{timing} timing{iter_tag}]:")
            print(f"  Total time:           {t_total:.3f}s")
            print(f"  Forward time:         {t_fwd:.3f}s")
            print(f"  Backward time:        {t_bwd:.3f}s")
            print(f"  P2P comm time:        {t_p2p:.3f}s")
            if n_sync > 0:
                print(f"  BD Sync time:         {t_sync:.1f}ms  ({n_sync} sync event(s), {sync_overhead:.1f}% of total)")
            print(f"  Pipeline efficiency:  {efficiency:.1f}%  "
                  f"({'fwd+bwd+sync' if needs_sync else 'fwd+bwd'} / total)")
            print(f"  Comm overhead:        {comm_overhead:.1f}%")

            transitions = p['phase_transitions']
            if transitions:
                start_mem = transitions[0]['memory_allocated'] / 1024**2
                peak_mem  = max(t['memory_allocated'] for t in transitions) / 1024**2
                print(f"  Memory:               {start_mem:.1f}MB → {peak_mem:.1f}MB (peak)")

        print(f"\nOverall Statistics (across {len(training_profiles)} ranks):")
        print(f"  Pipeline duration (max):  {max(total_time):.3f}s")
        print(f"  Avg total time:           {np.mean(total_time):.3f}s (±{np.std(total_time):.3f}s)")
        print(f"  Avg forward time:         {np.mean(forward_time):.3f}s")
        print(f"  Avg backward time:        {np.mean(backward_time):.3f}s")
        print(f"  Avg P2P time:             {np.mean(p2p_time):.3f}s")
        if any(ms > 0 for ms in sync_time_ms):
            print(f"  Avg BD Sync time:         {np.mean(sync_time_ms):.1f}ms (±{np.std(sync_time_ms):.1f}ms)")

    # Schedule comparison table
    available_schedules = []
    if f1b_profiles:
        available_schedules.append(('1F1B', f1b_profiles))
    if bitpipe_profiles:
        available_schedules.append(('BitPipe', bitpipe_profiles))
    if bitpipe_asym_profiles:
        available_schedules.append(('BitPipe Asym', bitpipe_asym_profiles))
    if chimera_profiles:
        available_schedules.append(('Chimera', chimera_profiles))
    if chimera_asym_profiles:
        available_schedules.append(('Chimera Asym', chimera_asym_profiles))

    if len(available_schedules) >= 2:
        print("\n--- SCHEDULE COMPARISON (one column per iteration) ---")

        # Build columns: one per (schedule_display_name, iteration) pair,
        # preserving schedule order and sorting iterations within each schedule.
        columns = []  # list of (col_name, schedule_key, profs_for_this_iter)
        for display_name, profs in available_schedules:
            iter_groups = {}
            for p in profs:
                it = get_iteration(p)
                key = it if it is not None else 'unknown'
                iter_groups.setdefault(key, []).append(p)
            for it in sorted(iter_groups.keys(), key=lambda x: (x == 'unknown', x)):
                col_name = f"{display_name} iter{it}" if it != 'unknown' else display_name
                columns.append((col_name, display_name, iter_groups[it]))

        # Compute metrics per column
        metrics = {}
        for col_name, schedule_name, profs in columns:
            avg_time     = np.mean([p['metadata']['total_time'] for p in profs])
            duration     = max([p['metadata']['total_time'] for p in profs])
            avg_fwd      = np.mean([p['summary']['total_forward_time'] for p in profs])
            avg_bwd      = np.mean([p['summary']['total_backward_time'] for p in profs])
            avg_p2p      = np.mean([p['summary']['total_p2p_time'] for p in profs])
            avg_sync     = np.mean([get_summary_value(p, 'total_sync_time_ms', 0.0) for p in profs])
            schedule_key = schedule_name.lower().replace(' ', '_')
            needs_sync   = schedule_key in ('chimera', 'chimera_asym', 'bitpipe', 'bitpipe_asym')
            useful_time  = avg_fwd + avg_bwd + (avg_sync / 1000.0 if needs_sync else 0.0)
            efficiency   = useful_time / avg_time * 100

            metrics[col_name] = {
                'duration':      duration,
                'avg_time':      avg_time,
                'forward':       avg_fwd,
                'backward':      avg_bwd,
                'p2p':           avg_p2p,
                'sync_ms':       avg_sync,
                'efficiency':    efficiency,
                'schedule_name': schedule_name,
            }

        col_width = 18
        header = f"{'Metric':<30}"
        for col_name, _, _ in columns:
            header += f" {col_name:<{col_width}}"
        print(f"\nPerformance Comparison:\n{header}")
        print("-" * (30 + (col_width + 1) * len(columns)))

        metric_rows = [
            ('Pipeline Duration (s)',    'duration',   f'{{:<{col_width}.3f}}'),
            ('Avg Total Time (s)',        'avg_time',   f'{{:<{col_width}.3f}}'),
            ('Forward Time (s)',          'forward',    f'{{:<{col_width}.3f}}'),
            ('Backward Time (s)',         'backward',   f'{{:<{col_width}.3f}}'),
            ('P2P Comm Time (s)',         'p2p',        f'{{:<{col_width}.3f}}'),
            ('BD Sync Time (ms)',         'sync_ms',    f'{{:<{col_width}.1f}}'),
            ('Pipeline Efficiency (%)*', 'efficiency', f'{{:<{col_width}.1f}}'),
        ]

        for display_name, key, fmt in metric_rows:
            row = f"{display_name:<30}"
            for col_name, _, _ in columns:
                row += ' ' + fmt.format(metrics[col_name][key])
            print(row)

        print("\n* Pipeline Efficiency: (fwd+bwd+sync)/total for Chimera/BitPipe, "
              "(fwd+bwd)/total for 1F1B")
        print("  BD sync is necessary work for bidirectional schedules, not overhead.")
        print("  Remaining loss = pipeline bubble + P2P comm.")

        # Speedup vs 1F1B: compare each non-1F1B column against the 1f1b column(s)
        f1b_cols = [(cn, m) for cn, m in metrics.items() if m['schedule_name'] == '1F1B']
        if f1b_cols:
            # Use average duration across all 1F1B iterations as baseline
            baseline = np.mean([m['duration'] for _, m in f1b_cols])
            print(f"\nSpeedup vs 1F1B (baseline avg duration = {baseline:.3f}s):")
            for col_name, _, _ in columns:
                if metrics[col_name]['schedule_name'] != '1F1B':
                    d = metrics[col_name]['duration']
                    tag = f"{baseline/d:.2f}x faster" if d < baseline else f"{d/baseline:.2f}x slower"
                    print(f"  {col_name}: {tag}")


# ─────────────────────────────────────────────────────────────────────────────
# Microbatch distribution charts
# ─────────────────────────────────────────────────────────────────────────────

def analyze_microbatch_distribution(profiles, output_file="microbatch_distribution.png"):
    """Analyze how microbatches are distributed across ranks and pipelines"""

    training_profiles = [p for p in profiles if p.get('profile_type', 'validation') == "training"]
    if not training_profiles:
        return

    mb_data = []
    for p in training_profiles:
        rank = p['metadata']['rank']
        for event in p['microbatch_events']:
            mb_data.append({
                'rank':           rank,
                'microbatch_id':  event['microbatch_id'],
                'pipeline_id':    event['pipeline_id'],
                'model_chunk_id': event['model_chunk_id'],
                'phase':          event['phase'],
                'duration':       event['end_time'] - event['start_time']
            })

    df = pd.DataFrame(mb_data)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    pivot = df.groupby(['rank', 'pipeline_id']).size().unstack(fill_value=0)
    pivot.plot(kind='bar', ax=ax)
    ax.set_title('Microbatch Count by Rank and Pipeline')
    ax.set_xlabel('Rank'); ax.set_ylabel('Count')
    ax.legend(title='Pipeline ID')

    ax = axes[0, 1]
    phase_duration = df.groupby(['rank', 'phase'])['duration'].mean().unstack()
    phase_duration.plot(kind='bar', ax=ax)
    ax.set_title('Average Microbatch Duration by Phase')
    ax.set_xlabel('Rank'); ax.set_ylabel('Duration (s)')
    ax.legend(title='Phase')

    ax = axes[1, 0]
    chunk_util = df.groupby(['rank', 'model_chunk_id']).size().unstack(fill_value=0)
    chunk_util.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Model Chunk Utilization by Rank')
    ax.set_xlabel('Rank'); ax.set_ylabel('Microbatch Count')
    ax.legend(title='Model Chunk ID')

    ax = axes[1, 1]
    pipeline_time = df.groupby(['rank', 'pipeline_id'])['duration'].sum().unstack()
    pipeline_time.plot(kind='bar', ax=ax)
    ax.set_title('Total Execution Time by Pipeline')
    ax.set_xlabel('Rank'); ax.set_ylabel('Total Time (s)')
    ax.legend(title='Pipeline ID')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nMicrobatch distribution analysis saved to {output_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Communication matrix
# ─────────────────────────────────────────────────────────────────────────────

def create_communication_matrix(profiles, output_file="communication_matrix.png"):
    """Create a communication matrix showing P2P patterns"""

    training_profiles = [p for p in profiles if p.get('profile_type', 'validation') == "training"]
    if not training_profiles:
        return

    world_size  = training_profiles[0]['metadata']['world_size']
    comm_matrix = np.zeros((world_size, world_size))

    for p in training_profiles:
        for event in p['p2p_events']:
            if 'send' in event['comm_type']:
                src = event['source_rank']
                dst = event['dest_rank']
                comm_matrix[src, dst] += event['end_time'] - event['start_time']

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(comm_matrix, cmap='YlOrRd', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Total Communication Time (s)')

    ax.set_xticks(range(world_size)); ax.set_yticks(range(world_size))
    ax.set_xlabel('Destination Rank'); ax.set_ylabel('Source Rank')
    ax.set_title('P2P Communication Matrix')

    for i in range(world_size):
        for j in range(world_size):
            if comm_matrix[i, j] > 0:
                ax.text(j, i, f'{comm_matrix[i, j]:.3f}',
                        ha="center", va="center", color="black", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nCommunication matrix saved to {output_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Execution order print (includes sync events)
# ─────────────────────────────────────────────────────────────────────────────

def print_microbatch_execution_order(profiles):
    """Print the execution order of microbatches (and sync events) for each rank"""

    print("\n" + "="*60)
    print("MICROBATCH EXECUTION ORDER BY RANK")
    print("="*60)

    training_profiles = [p for p in profiles if p.get('profile_type', 'validation') == "training"]
    if not training_profiles:
        print("No training profiles found!")
        return

    rank_profiles = {}
    for p in training_profiles:
        rank = p['metadata']['rank']
        if rank not in rank_profiles:
            rank_profiles[rank] = p

    schedule_type = get_schedule_type(list(rank_profiles.values())[0])
    print(f"\nSchedule Type: {schedule_type.upper()}")

    for rank in sorted(rank_profiles.keys()):
        profile = rank_profiles[rank]

        # Build a unified event list: microbatch events + sync events
        unified = []
        for e in profile['microbatch_events']:
            unified.append({
                'kind':        'mb',
                'start_time':  e['start_time'],
                'end_time':    e['end_time'],
                'event':       e,
            })
        for se in get_sync_events(profile):
            unified.append({
                'kind':        'sync',
                'start_time':  se['wall_clock_start'],
                'end_time':    se['wall_clock_end'],
                'event':       se,
            })

        unified.sort(key=lambda x: x['start_time'])

        print(f"\n{'='*50}")
        print(f"RANK {rank} - Execution Order  [{get_timing_method(profile)} timing]:")
        print(f"{'='*50}")

        for i, item in enumerate(unified):
            start    = item['start_time']
            end      = item['end_time']
            duration = end - start

            if item['kind'] == 'mb':
                e         = item['event']
                phase     = e['phase'].upper()
                mb_id     = e['microbatch_id']
                pip_id    = e['pipeline_id']
                chunk_id  = e.get('model_chunk_id', 'N/A')

                if schedule_type in ["bitpipe", "bitpipe_asym", "chimera", "chimera_asym"]:
                    print(f"{i+1:3d}. MB{mb_id:2d}  {phase:8s}  VR{chunk_id}  P{pip_id}  "
                          f"[{start:7.4f}s – {end:7.4f}s]  {duration*1000:6.1f}ms")
                else:
                    print(f"{i+1:3d}. MB{mb_id:2d}  {phase:8s}  "
                          f"[{start:7.4f}s – {end:7.4f}s]  {duration*1000:6.1f}ms")

            else:  # sync
                se      = item['event']
                total   = se.get('total_duration_ms', duration * 1000)
                chunks  = se.get('chunk_durations_ms', [])
                phase   = se.get('phase', 'cooldown')
                chunk_str = '  '.join(f"VR{c['chunk_id']}:{c['duration_ms']:.1f}ms"
                                       for c in chunks) if chunks else 'n/a'
                print(f"{i+1:3d}. --- BD SYNC ({phase})  "
                      f"[{start:7.4f}s – {end:7.4f}s]  total:{total:.1f}ms  [{chunk_str}]")

        # Summary for this rank
        mb_events   = [x for x in unified if x['kind'] == 'mb']
        sync_events = [x for x in unified if x['kind'] == 'sync']
        fwd_count   = sum(1 for x in mb_events if x['event']['phase'] == 'forward')
        bwd_count   = sum(1 for x in mb_events if x['event']['phase'] == 'backward')
        total_comp  = sum(x['end_time'] - x['start_time'] for x in mb_events)

        print(f"\n  Microbatches: {len(mb_events)} (fwd={fwd_count}, bwd={bwd_count}), "
              f"compute={total_comp*1000:.1f}ms,  sync events: {len(sync_events)}")

    print("\n" + "="*60)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level orchestration
# ─────────────────────────────────────────────────────────────────────────────

def create_visualizations_by_schedule_type(profiles):
    """Create visualizations for each available schedule type"""
    os.makedirs(_DEFAULT_VIZ_DIR, exist_ok=True)

    training_profiles = [p for p in profiles if p.get('profile_type', 'validation') == "training"]
    if not training_profiles:
        print("No training profiles found for visualization!")
        return []

    schedule_groups = {
        '1f1b':         [p for p in training_profiles if get_schedule_type(p) == "1f1b"],
        'bitpipe':      [p for p in training_profiles if get_schedule_type(p) == "bitpipe"],
        'bitpipe_asym': [p for p in training_profiles if get_schedule_type(p) == "bitpipe_asym"],
        'chimera':      [p for p in training_profiles if get_schedule_type(p) == "chimera"],
        'chimera_asym': [p for p in training_profiles if get_schedule_type(p) == "chimera_asym"],
    }
    available = {name: profs for name, profs in schedule_groups.items() if profs}

    print(f"\n--- GENERATING VISUALIZATIONS ---")
    print(f"Available schedule types: {list(available.keys())}")

    generated_files = []

    for schedule_name, schedule_profs in available.items():
        print(f"\nGenerating visualizations for {schedule_name.upper()}...")

        timeline_file = os.path.join(_DEFAULT_VIZ_DIR, f"timeline_{schedule_name}.png")
        create_timeline_visualization(schedule_profs, timeline_file)
        generated_files.append(timeline_file)

        distribution_file = os.path.join(_DEFAULT_VIZ_DIR, f"microbatch_distribution_{schedule_name}.png")
        analyze_microbatch_distribution(schedule_profs, distribution_file)
        generated_files.append(distribution_file)

        comm_file = os.path.join(_DEFAULT_VIZ_DIR, f"communication_matrix_{schedule_name}.png")
        create_communication_matrix(schedule_profs, comm_file)
        generated_files.append(comm_file)

        # Sync analysis only if any profile has sync events
        if any(get_sync_events(p) for p in schedule_profs):
            sync_file = os.path.join(_DEFAULT_VIZ_DIR, f"sync_analysis_{schedule_name}.png")
            create_sync_analysis(schedule_profs, sync_file)
            generated_files.append(sync_file)
            print(f"  Generated: {timeline_file}, {distribution_file}, {comm_file}, {sync_file}")
        else:
            print(f"  Generated: {timeline_file}, {distribution_file}, {comm_file}")
            print(f"  (No sync events in profiles — skipping sync_analysis chart)")

    print(f"\nGenerated {len(generated_files)} visualization files ({len(available)} schedule types)")
    return generated_files


def main():
    """Main analysis function"""
    profiles = load_profile_data()

    if not profiles:
        print(f"No profile files found in {_DEFAULT_PROFILE_DIR}")
        return

    print(f"Loaded {len(profiles)} profile files")
    for p in profiles:
        timing   = get_timing_method(p)
        synced   = p['metadata'].get('synchronized', False)
        n_sync   = int(get_summary_value(p, 'num_sync_events', 0))
        print(f"  {p['filename']}  rank={p['metadata']['rank']}  "
              f"schedule={get_schedule_type(p)}  timing={timing}  "
              f"synchronized={synced}  sync_events={n_sync}")

    generated_files = create_visualizations_by_schedule_type(profiles)
    analyze_pipeline_efficiency(profiles)
    print_microbatch_execution_order(profiles)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    if generated_files:
        print(f"\nGenerated {len(generated_files)} visualization files:")
        for f in generated_files:
            print(f"  {f}")
    else:
        print("\nNo visualization files generated (no training profiles found)")
    print("="*60)


if __name__ == "__main__":
    main()
