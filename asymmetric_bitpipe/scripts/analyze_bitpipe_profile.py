#!/usr/bin/env python
"""
BitPipe Profile Analysis and Visualization Tool

This script analyzes and visualizes BitPipe profiling data to understand:
- Pipeline execution timeline
- Communication patterns
- Performance bottlenecks
- Pipeline efficiency
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from datetime import datetime
import pandas as pd

def load_profile_data(profile_dir="../profiles/raw"):
    """Load all profile files from directory"""
    profile_files = glob.glob(os.path.join(profile_dir, "*.json"))
    profiles = []
    
    for file in sorted(profile_files):
        with open(file, 'r') as f:
            data = json.load(f)
            data['filename'] = os.path.basename(file)
            profiles.append(data)
    
    return profiles

def identify_profile_type(profile):
    """Identify if profile is training or validation based on backward passes"""
    num_backward = profile['summary']['num_backward_passes']
    return "training" if num_backward > 0 else "validation"

def get_schedule_type(profile):
    """Get the schedule type (bitpipe, bitpipe_asym, or 1f1b)"""
    return profile['metadata'].get('schedule_type', 'unknown')

def create_timeline_visualization(profiles, output_file="bitpipe_timeline.png"):
    """Create a timeline visualization of microbatch execution"""
    
    # Separate training and validation profiles
    training_profiles = [p for p in profiles if identify_profile_type(p) == "training"]
    
    if not training_profiles:
        print("No training profiles found!")
        return
    
    # Separate by schedule type and use the first training profile from each rank
    bitpipe_profiles = {p['metadata']['rank']: p for p in training_profiles if get_schedule_type(p) == "bitpipe"}
    bitpipe_asym_profiles = {p['metadata']['rank']: p for p in training_profiles if get_schedule_type(p) == "bitpipe_asym"}
    f1b_profiles = {p['metadata']['rank']: p for p in training_profiles if get_schedule_type(p) == "1f1b"}
    
    # Use profiles in priority order: BitPipe Asym > BitPipe > 1F1B
    if bitpipe_asym_profiles:
        rank_profiles = bitpipe_asym_profiles
        schedule_type = "BitPipe Asymmetric"
    elif bitpipe_profiles:
        rank_profiles = bitpipe_profiles
        schedule_type = "BitPipe"
    else:
        rank_profiles = f1b_profiles
        schedule_type = "Standard 1F1B"
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Color schemes - handle both BitPipe (2 pipelines) and 1F1B (1 pipeline)
    if schedule_type in ["BitPipe", "BitPipe Asymmetric"]:
        forward_colors = {
            0: '#1f77b4',  # Pipeline 0 - blue shades
            1: '#ff7f0e'   # Pipeline 1 - orange shades
        }
        backward_colors = {
            0: '#2ca02c',  # Pipeline 0 - green shades
            1: '#d62728'   # Pipeline 1 - red shades
        }
    else:  # Standard 1F1B
        forward_colors = {0: '#1f77b4'}   # Single pipeline - blue
        backward_colors = {0: '#2ca02c'}  # Single pipeline - green
    
    # Calculate computation time bounds across all ranks
    all_events = []
    for profile in rank_profiles.values():
        all_events.extend(profile['microbatch_events'])
    
    if all_events:
        earliest_start = min(event['start_time'] for event in all_events)
        latest_end = max(event['end_time'] for event in all_events)
        computation_duration = latest_end - earliest_start
        
        # Add small padding (5% on each side)
        padding = computation_duration * 0.05
        x_min = max(0, earliest_start - padding)  # Don't go below 0
        x_max = latest_end + padding
        
        print(f"Timeline bounds: {earliest_start:.4f}s to {latest_end:.4f}s (duration: {computation_duration:.4f}s)")
        print(f"Plot x-axis: {x_min:.4f}s to {x_max:.4f}s")
    else:
        x_min, x_max = 0, 1  # Default if no events
    
    # Plot each rank
    y_positions = {}
    for rank in sorted(rank_profiles.keys()):
        y_positions[rank] = rank * 2
        
        profile = rank_profiles[rank]
        
        # Plot microbatch events
        for event in profile['microbatch_events']:
            y_pos = y_positions[rank]
            start = event['start_time']
            duration = event['end_time'] - event['start_time']
            
            # Choose color based on phase and pipeline
            if event['phase'] == 'forward':
                color = forward_colors[event['pipeline_id']]
                y_offset = 0
            else:
                color = backward_colors[event['pipeline_id']]
                y_offset = 0.8
            
            # Create rectangle
            rect = patches.Rectangle(
                (start, y_pos + y_offset), 
                duration, 
                0.7,
                linewidth=1, 
                edgecolor='black',
                facecolor=color,
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add microbatch ID and chunk ID
            mb_id = event['microbatch_id']
            chunk_id = event.get('model_chunk_id', 'N/A')
            label = f"MB{mb_id}\nC{chunk_id}"
            
            ax.text(
                start + duration/2, 
                y_pos + y_offset + 0.35, 
                label, 
                ha='center', 
                va='center', 
                fontsize=7,  # Slightly smaller font to fit both lines
                weight='bold'
            )
    
    # Add phase transitions
    if rank_profiles:
        first_profile = list(rank_profiles.values())[0]
        for transition in first_profile['phase_transitions']:
            ax.axvline(
                x=transition['timestamp'], 
                color='gray', 
                linestyle='--', 
                alpha=0.5
            )
            ax.text(
                transition['timestamp'], 
                max(y_positions.values()) + 2, 
                transition['phase_name'], 
                rotation=45, 
                ha='right'
            )
    
    # Formatting
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Rank', fontsize=12)
    ax.set_title(f'{schedule_type} Pipeline Execution Timeline (Computation Focus)', fontsize=16, weight='bold')
    
    # Set y-axis
    ax.set_yticks([y_positions[r] + 0.4 for r in sorted(y_positions.keys())])
    ax.set_yticklabels([f'Rank {r}' for r in sorted(y_positions.keys())])
    
    # Set x-axis to focus on computation time
    ax.set_xlim(x_min, x_max)
    
    # Set y-axis limits to ensure all ranks are visible
    # Calculate the maximum y position needed
    if y_positions:
        max_rank = max(y_positions.keys())
        # Need space for: base position + backward offset (0.8) + rectangle height (0.7) + padding
        y_max = y_positions[max_rank] + 0.8 + 0.7 + 0.5  # 0.5 for padding
        y_min = -0.5  # Small padding at bottom
        ax.set_ylim(y_min, y_max)
    
    # Add legend
    if schedule_type in ["BitPipe", "BitPipe Asymmetric"]:
        legend_elements = [
            patches.Patch(facecolor=forward_colors[0], alpha=0.7, label='Pipeline 0 Forward'),
            patches.Patch(facecolor=forward_colors[1], alpha=0.7, label='Pipeline 1 Forward'),
            patches.Patch(facecolor=backward_colors[0], alpha=0.7, label='Pipeline 0 Backward'),
            patches.Patch(facecolor=backward_colors[1], alpha=0.7, label='Pipeline 1 Backward')
        ]
    else:  # Standard 1F1B
        legend_elements = [
            patches.Patch(facecolor=forward_colors[0], alpha=0.7, label='Forward Pass'),
            patches.Patch(facecolor=backward_colors[0], alpha=0.7, label='Backward Pass')
        ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Update filename based on schedule type
    base_name = output_file.replace('.png', '')
    if schedule_type == "BitPipe":
        schedule_suffix = 'bitpipe'
    elif schedule_type == "BitPipe Asymmetric":
        schedule_suffix = 'bitpipe_asym'
    else:
        schedule_suffix = '1f1b'
    final_output_file = f"{base_name}_{schedule_suffix}.png"
    plt.savefig(final_output_file, dpi=300)
    print(f"{schedule_type} timeline visualization saved to {final_output_file}")
    
def analyze_pipeline_efficiency(profiles):
    """Analyze pipeline efficiency metrics"""
    
    print("\n" + "="*60)
    print("PIPELINE PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Separate by type and schedule
    training_profiles = [p for p in profiles if identify_profile_type(p) == "training"]
    validation_profiles = [p for p in profiles if identify_profile_type(p) == "validation"]
    
    bitpipe_profiles = [p for p in training_profiles if get_schedule_type(p) == "bitpipe"]
    bitpipe_asym_profiles = [p for p in training_profiles if get_schedule_type(p) == "bitpipe_asym"]
    f1b_profiles = [p for p in training_profiles if get_schedule_type(p) == "1f1b"]
    
    print(f"\nFound {len(training_profiles)} training profiles and {len(validation_profiles)} validation profiles")
    print(f"BitPipe profiles: {len(bitpipe_profiles)}, BitPipe Asymmetric: {len(bitpipe_asym_profiles)}, Standard 1F1B: {len(f1b_profiles)}")
    
    # Analyze training profiles
    if training_profiles:
        print("\n--- TRAINING PERFORMANCE ---")
        
        total_time = []
        forward_time = []
        backward_time = []
        p2p_time = []
        throughput_values = []
        
        for p in training_profiles:
            rank = p['metadata']['rank']
            summary = p['summary']
            
            total_time.append(p['metadata']['total_time'])
            forward_time.append(summary['total_forward_time'])
            backward_time.append(summary['total_backward_time'])
            p2p_time.append(summary['total_p2p_time'])
            
            # Store throughput for overall statistics
            total_microbatches = summary['num_forward_passes'] + summary['num_backward_passes']
            throughput = total_microbatches / p['metadata']['total_time']
            throughput_values.append(throughput)
            
            # Calculate efficiency and throughput
            compute_time = summary['total_forward_time'] + summary['total_backward_time']
            efficiency = compute_time / p['metadata']['total_time'] * 100
            comm_overhead = summary['total_p2p_time'] / p['metadata']['total_time'] * 100
            
            # Calculate throughput (microbatches per second)
            total_microbatches = summary['num_forward_passes'] + summary['num_backward_passes']
            throughput = total_microbatches / p['metadata']['total_time']
            
            print(f"\nRank {rank}:")
            print(f"  Total time: {p['metadata']['total_time']:.3f}s")
            print(f"  Forward time: {summary['total_forward_time']:.3f}s")
            print(f"  Backward time: {summary['total_backward_time']:.3f}s")
            print(f"  P2P comm time: {summary['total_p2p_time']:.3f}s")
            print(f"  Pipeline efficiency: {efficiency:.1f}%")
            print(f"  Communication overhead: {comm_overhead:.1f}%")
            print(f"  Throughput: {throughput:.2f} microbatches/second")
            
            # Memory usage
            transitions = p['phase_transitions']
            if transitions:
                start_mem = transitions[0]['memory_allocated'] / 1024**2
                peak_mem = max(t['memory_allocated'] for t in transitions) / 1024**2
                print(f"  Memory: {start_mem:.1f}MB → {peak_mem:.1f}MB (peak)")
        
        # Overall statistics
        print(f"\nOverall Statistics (across {len(training_profiles)} ranks):")
        print(f"  Total pipeline duration: {max(total_time):.3f}s (max across ranks)")
        print(f"  Average total time: {np.mean(total_time):.3f}s (±{np.std(total_time):.3f}s)")
        print(f"  Average forward time: {np.mean(forward_time):.3f}s")
        print(f"  Average backward time: {np.mean(backward_time):.3f}s")
        print(f"  Average P2P time: {np.mean(p2p_time):.3f}s")
        print(f"  Average throughput: {np.mean(throughput_values):.2f} microbatches/second (±{np.std(throughput_values):.2f})")
        
    # Compare all available schedule types
    available_schedules = []
    schedule_profiles = {}
    
    if f1b_profiles:
        available_schedules.append(('1F1B', f1b_profiles))
        schedule_profiles['1F1B'] = f1b_profiles
    if bitpipe_profiles:
        available_schedules.append(('BitPipe', bitpipe_profiles))
        schedule_profiles['BitPipe'] = bitpipe_profiles
    if bitpipe_asym_profiles:
        available_schedules.append(('BitPipe Asym', bitpipe_asym_profiles))
        schedule_profiles['BitPipe Asym'] = bitpipe_asym_profiles
    
    if len(available_schedules) >= 2:
        print("\n--- SCHEDULE COMPARISON ---")
        
        # Calculate metrics for each schedule type
        metrics = {}
        for name, profiles in available_schedules:
            avg_time = np.mean([p['metadata']['total_time'] for p in profiles])
            duration = max([p['metadata']['total_time'] for p in profiles])
            avg_forward = np.mean([p['summary']['total_forward_time'] for p in profiles])
            avg_backward = np.mean([p['summary']['total_backward_time'] for p in profiles])
            avg_p2p = np.mean([p['summary']['total_p2p_time'] for p in profiles])
            efficiency = (avg_forward + avg_backward) / avg_time * 100
            
            # Calculate average throughput for this schedule
            throughputs = []
            for p in profiles:
                total_microbatches = p['summary']['num_forward_passes'] + p['summary']['num_backward_passes']
                throughput = total_microbatches / p['metadata']['total_time']
                throughputs.append(throughput)
            avg_throughput = np.mean(throughputs)
            
            metrics[name] = {
                'duration': duration,
                'avg_time': avg_time,
                'forward': avg_forward,
                'backward': avg_backward,
                'p2p': avg_p2p,
                'efficiency': efficiency,
                'throughput': avg_throughput
            }
        
        # Print comparison table
        print(f"\nPerformance Comparison:")
        
        # Header
        header = f"{'Metric':<25}"
        for name, _ in available_schedules:
            header += f" {name:<15}"
        print(header)
        print("-" * (25 + 16 * len(available_schedules)))
        
        # Metrics rows
        metric_names = [
            ('Pipeline Duration (s)', 'duration'),
            ('Avg Total Time (s)', 'avg_time'),
            ('Forward Time (s)', 'forward'),
            ('Backward Time (s)', 'backward'),
            ('P2P Comm Time (s)', 'p2p'),
            ('Pipeline Efficiency (%)', 'efficiency'),
            ('Throughput (samples/s)', 'throughput')
        ]
        
        for display_name, key in metric_names:
            row = f"{display_name:<25}"
            for name, _ in available_schedules:
                value = metrics[name][key]
                if key == 'efficiency':
                    row += f" {value:<15.1f}"
                elif key == 'throughput':
                    row += f" {value:<15.2f}"
                else:
                    row += f" {value:<15.3f}"
            print(row)
        
        # Calculate speedups relative to 1F1B if available
        if '1F1B' in metrics:
            print("\nSpeedup vs 1F1B:")
            baseline_duration = metrics['1F1B']['duration']
            for name, _ in available_schedules:
                if name != '1F1B':
                    duration = metrics[name]['duration']
                    if duration < baseline_duration:
                        speedup = baseline_duration / duration
                        print(f"  {name}: {speedup:.2f}x faster 🚀")
                    else:
                        slowdown = duration / baseline_duration
                        print(f"  {name}: {slowdown:.2f}x slower ⚠️")
        
def analyze_microbatch_distribution(profiles, output_file="microbatch_distribution.png"):
    """Analyze how microbatches are distributed across ranks and pipelines"""
    
    training_profiles = [p for p in profiles if identify_profile_type(p) == "training"]
    
    if not training_profiles:
        return
    
    # Collect microbatch data
    mb_data = []
    for p in training_profiles:
        rank = p['metadata']['rank']
        for event in p['microbatch_events']:
            mb_data.append({
                'rank': rank,
                'microbatch_id': event['microbatch_id'],
                'pipeline_id': event['pipeline_id'],
                'model_chunk_id': event['model_chunk_id'],
                'phase': event['phase'],
                'duration': event['end_time'] - event['start_time']
            })
    
    df = pd.DataFrame(mb_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Microbatch count by rank and pipeline
    ax = axes[0, 0]
    pivot = df.groupby(['rank', 'pipeline_id']).size().unstack(fill_value=0)
    pivot.plot(kind='bar', ax=ax)
    ax.set_title('Microbatch Count by Rank and Pipeline')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Count')
    ax.legend(title='Pipeline ID')
    
    # 2. Average duration by phase
    ax = axes[0, 1]
    phase_duration = df.groupby(['rank', 'phase'])['duration'].mean().unstack()
    phase_duration.plot(kind='bar', ax=ax)
    ax.set_title('Average Microbatch Duration by Phase')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Duration (s)')
    ax.legend(title='Phase')
    
    # 3. Model chunk utilization
    ax = axes[1, 0]
    chunk_util = df.groupby(['rank', 'model_chunk_id']).size().unstack(fill_value=0)
    chunk_util.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Model Chunk Utilization by Rank')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Microbatch Count')
    ax.legend(title='Model Chunk ID')
    
    # 4. Pipeline balance
    ax = axes[1, 1]
    pipeline_time = df.groupby(['rank', 'pipeline_id'])['duration'].sum().unstack()
    pipeline_time.plot(kind='bar', ax=ax)
    ax.set_title('Total Execution Time by Pipeline')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Total Time (s)')
    ax.legend(title='Pipeline ID')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"\nMicrobatch distribution analysis saved to {output_file}")

def print_microbatch_execution_order(profiles):
    """Print the execution order of microbatches for each rank"""
    
    print("\n" + "="*60)
    print("MICROBATCH EXECUTION ORDER BY RANK")
    print("="*60)
    
    # Get all training profiles
    training_profiles = [p for p in profiles if identify_profile_type(p) == "training"]
    
    if not training_profiles:
        print("No training profiles found!")
        return
    
    # Group profiles by rank
    rank_profiles = {}
    for p in training_profiles:
        rank = p['metadata']['rank']
        if rank not in rank_profiles:
            rank_profiles[rank] = p
    
    # Get schedule type
    schedule_type = get_schedule_type(list(rank_profiles.values())[0])
    print(f"\nSchedule Type: {schedule_type.upper()}")
    
    # Print execution order for each rank
    for rank in sorted(rank_profiles.keys()):
        profile = rank_profiles[rank]
        events = profile['microbatch_events']
        
        # Sort events by start time
        sorted_events = sorted(events, key=lambda x: x['start_time'])
        
        print(f"\n{'='*40}")
        print(f"RANK {rank} - Microbatch Execution Order:")
        print(f"{'='*40}")
        
        for i, event in enumerate(sorted_events):
            phase = event['phase'].upper()
            mb_id = event['microbatch_id']
            pipeline_id = event['pipeline_id']
            start_time = event['start_time']
            end_time = event['end_time']
            duration = end_time - start_time
            
            # Format the phase name with consistent width
            phase_str = f"{phase:8s}"  # 8 characters wide, left-aligned
            
            # Create the output string
            if schedule_type in ["bitpipe", "bitpipe_asym"]:
                print(f"{i+1:3d}. MB{mb_id:2d} - {phase_str} (Pipeline {pipeline_id}) "
                      f"[{start_time:6.3f}s - {end_time:6.3f}s] Duration: {duration:5.3f}s")
            else:  # 1f1b
                print(f"{i+1:3d}. MB{mb_id:2d} - {phase_str} "
                      f"[{start_time:6.3f}s - {end_time:6.3f}s] Duration: {duration:5.3f}s")
        
        # Add summary statistics for this rank
        forward_count = sum(1 for e in sorted_events if e['phase'] == 'forward')
        backward_count = sum(1 for e in sorted_events if e['phase'] == 'backward')
        total_duration = sum(e['end_time'] - e['start_time'] for e in sorted_events)
        
        print(f"\nRank {rank} Summary:")
        print(f"  - Total microbatches: {len(sorted_events)}")
        print(f"  - Forward passes: {forward_count}")
        print(f"  - Backward passes: {backward_count}")
        print(f"  - Total compute time: {total_duration:.3f}s")
    
    print("\n" + "="*60)

def create_communication_matrix(profiles, output_file="communication_matrix.png"):
    """Create a communication matrix showing P2P patterns"""
    
    training_profiles = [p for p in profiles if identify_profile_type(p) == "training"]
    
    if not training_profiles:
        return
    
    # Get world size
    world_size = training_profiles[0]['metadata']['world_size']
    
    # Initialize communication matrix
    comm_matrix = np.zeros((world_size, world_size))
    
    # Aggregate P2P communications
    for p in training_profiles:
        for event in p['p2p_events']:
            if 'send' in event['comm_type']:
                src = event['source_rank']
                dst = event['dest_rank']
                comm_matrix[src, dst] += event['end_time'] - event['start_time']
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(comm_matrix, cmap='YlOrRd', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Total Communication Time (s)')
    
    # Labels
    ax.set_xticks(range(world_size))
    ax.set_yticks(range(world_size))
    ax.set_xlabel('Destination Rank')
    ax.set_ylabel('Source Rank')
    ax.set_title('P2P Communication Matrix')
    
    # Add values to cells
    for i in range(world_size):
        for j in range(world_size):
            if comm_matrix[i, j] > 0:
                text = ax.text(j, i, f'{comm_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"\nCommunication matrix saved to {output_file}")

def create_visualizations_by_schedule_type(profiles):
    """Create visualizations for each available schedule type"""
    # Ensure visualizations directory exists
    os.makedirs("../visualizations", exist_ok=True)
    
    training_profiles = [p for p in profiles if identify_profile_type(p) == "training"]
    
    if not training_profiles:
        print("No training profiles found for visualization!")
        return []
    
    # Separate profiles by schedule type
    schedule_types = {
        '1f1b': [p for p in training_profiles if get_schedule_type(p) == "1f1b"],
        'bitpipe': [p for p in training_profiles if get_schedule_type(p) == "bitpipe"],
        'bitpipe_asym': [p for p in training_profiles if get_schedule_type(p) == "bitpipe_asym"]
    }
    
    # Remove empty schedule types
    available_schedules = {name: profiles for name, profiles in schedule_types.items() if profiles}
    
    print(f"\n--- GENERATING VISUALIZATIONS ---")
    print(f"Available schedule types: {list(available_schedules.keys())}")
    
    generated_files = []
    
    # Generate visualizations for each schedule type
    for schedule_name, schedule_profiles in available_schedules.items():
        print(f"\nGenerating visualizations for {schedule_name.upper()}...")
        
        # Timeline visualization
        timeline_file = f"../visualizations/timeline_{schedule_name}.png"
        create_timeline_visualization(schedule_profiles, timeline_file)
        generated_files.append(timeline_file)
        
        # Microbatch distribution
        distribution_file = f"../visualizations/microbatch_distribution_{schedule_name}.png"
        analyze_microbatch_distribution(schedule_profiles, distribution_file)
        generated_files.append(distribution_file)
        
        # Communication matrix
        communication_file = f"../visualizations/communication_matrix_{schedule_name}.png"
        create_communication_matrix(schedule_profiles, communication_file)
        generated_files.append(communication_file)
        
        print(f"  ✓ Generated: {timeline_file}, {distribution_file}, {communication_file}")
    
    total_files = len(generated_files)
    total_schedules = len(available_schedules)
    print(f"\n🎉 Generated {total_files} visualization files ({total_schedules} schedule types × 3 visualizations)")
    
    return generated_files

def main():
    """Main analysis function"""
    
    # Load profiles
    profiles = load_profile_data()
    
    if not profiles:
        print("No profile files found in ./bitpipe_profiles/")
        return
    
    print(f"Loaded {len(profiles)} profile files")
    
    # Create visualizations for each schedule type
    generated_files = create_visualizations_by_schedule_type(profiles)
    analyze_pipeline_efficiency(profiles)
    #print_microbatch_execution_order(profiles)
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    if generated_files:
        print(f"\n📊 Generated {len(generated_files)} visualization files:")
        for file in generated_files:
            print(f"  ✓ {file}")
    else:
        print("\n⚠️  No visualization files generated (no training profiles found)")
    print("="*60)

if __name__ == "__main__":
    main()