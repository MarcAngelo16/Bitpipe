import heapq
import numpy as np
import itertools
from tqdm.notebook import tqdm

# stripped of all visualization logic for raw performance

def fast_makespan(task_forward, multiplier=2.0):
    # Pre-calculate backward pass
    # Using integer arithmetic for speed
    task_backward = [int(x * multiplier) for x in task_forward]
    task_backward.reverse()

    task_full = task_forward + task_backward

    # Machine Availability Times [M0, M1, M2, M3]
    machine_avail = [0, 0, 0, 0]

    # Event Heap: (time, neg_stage_priority, task_id, duration)
    events = []

    # Initial Population

    # Initialize events
    for task_id in range(4):
        duration = task_full[0] # Stage 0 duration
        heapq.heappush(events, (0, 0, task_id, duration))

    max_finish_time = 0

    while events:
        time, neg_stage, task_id, duration = heapq.heappop(events)
        stage = -neg_stage

        # Hardcoded get_machine logic for speed
        # Symmetry: M1=M4 (idx 0=3), M2=M3 (idx 1=2)
        if task_id < 2: # Tasks 0, 1 (Left->Right)
            if stage == 0 or stage == 7: machine = 0
            elif stage == 1 or stage == 6: machine = 1
            elif stage == 2 or stage == 5: machine = 2
            else: machine = 3
        else: # Tasks 2, 3 (Right->Left)
            if stage == 0 or stage == 7: machine = 3
            elif stage == 1 or stage == 6: machine = 2
            elif stage == 2 or stage == 5: machine = 1
            else: machine = 0

        ready_time = max(time, 0)

        if machine_avail[machine] <= ready_time:
            # Process Task
            end = ready_time + duration
            machine_avail[machine] = end
            max_finish_time = max(max_finish_time, end)

            # Queue Next Stage
            if stage < 7:
                next_stage = stage + 1
                next_dur = task_full[next_stage]
                heapq.heappush(events, (end, -next_stage, task_id, next_dur))
        else:
            # Wait for machine
            heapq.heappush(events, (machine_avail[machine], neg_stage, task_id, duration))

    return max_finish_time

TARGET = 64
MULTIPLIER = 2.0

best_time = float('inf')
best_variance = float('inf')
best_vector = None

# We iterate s1, s2, s3. s4 is the remainder.

# Using ranges to ensure at least 1 unit per stage to be realistic (1 to 117)
# or allowing 0 if your system permits 0-duration stages. Let's assume >= 1.
r = range(1, TARGET)

count = 0

for s1 in r:
    # Optimization: If s1 alone is already too big to allow others, break
    if s1 > TARGET - 3: break

    for s2 in range(1, TARGET - s1):
        for s3 in range(1, TARGET - s1 - s2):
            s4 = TARGET - s1 - s2 - s3

            # We now have a valid vector [s1, s2, s3, s4]
            current_vec = [s1, s2, s3, s4]

            # Run Sim
            time = fast_makespan(current_vec, MULTIPLIER)

            # Check if this is the new best
            if time < best_time:
                best_time = time
                best_variance = np.var(current_vec)
                best_vector = current_vec
            elif time == best_time:
                # Tie-breaker: Minimize Variance
                current_var = np.var(current_vec)
                if current_var < best_variance:
                    best_variance = current_var
                    best_vector = current_vec

            count += 1

print(f"Processed {count} states.")
print("="*40)
print(f"GLOBALLY OPTIMAL DIVISION: {best_vector}")
print(f"Minimum Total Time: {best_time}")
print(f"Variance: {best_variance:.2f}")
print("="*40)

# Visualize the result using the original visualizer
# (Make sure to run the first cell with 'run_schedule' and 'visualize' first)
if 'visualize' in globals():
    full_details = run_schedule(best_vector, MULTIPLIER)
    visualize(full_details, best_vector, MULTIPLIER)
