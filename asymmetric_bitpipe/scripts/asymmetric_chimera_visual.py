import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ipywidgets as widgets
from IPython.display import display, clear_output

# --- 1. Core Logic Functions ---

def visualize(scheduling_details, task_durations, multiplier):
    # --- Custom Color Palettes ---
    c_blue = {0: '#6BAED6', 1: '#9ECAE1', 2: '#C6DBEF', 3: '#DEEBF7'}
    c_orange = {0: '#FD8D3C', 1: '#FDBE85', 2: '#FEEDDE', 3: '#FFF5EB'}

    fig, ax = plt.subplots(figsize=(12, 6))

    max_time = 0

    # Plot each machine's tasks
    for task_id, stages in scheduling_details.items():
        for stage, machine, start, end in stages:
            if stage < 4:
                color = c_blue.get(stage, 'gray')
            else:
                color = c_orange.get(stage - 4, 'gray')

            lbl = f'Task {task_id}' if stage == 0 and machine == 0 else ""

            # Plot Bar
            ax.broken_barh([(start, end - start)], (machine * 10, 9),facecolors=color, edgecolor='black', label=lbl)

            # Text Label (Task + Stage)
            task_id = task_id-2 if task_id >= 2 else task_id
            ax.text(start + (end - start) / 2, machine * 10 + 5, f'T{task_id}S{stage}',ha='center', va='center', color='black', fontsize=8, fontweight='bold')

            max_time = max(max_time, end)

    # Add Flush Line
    ax.axvline(x=max_time, color='#C0392B', linestyle='-', linewidth=2.5)
    ax.text(max_time, 42, f'Flush\n{max_time}', color='#C0392B', ha='center', fontweight='bold', fontsize=12)

    # Formatting
    ax.set_ylim(0, 48)
    ax.set_xlim(0, max(max_time * 1.05, 100))
    ax.set_xlabel('Time')
    ax.set_yticks([5, 15, 25, 35])
    ax.set_yticklabels(['Machine 0', 'Machine 1', 'Machine 2', 'Machine 3'])
    ax.grid(axis='x', linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.title(f'Chimera Pipeline | Multiplier: x{multiplier} | Forward Tasks: {task_durations}')
    plt.show()

def get_machine(task_id, stage):
    # Logic: Tasks 0/1 go 0->3 then 3->0. Tasks 2/3 go 3->0 then 0->3.
    if task_id < 2:
        if stage in [0, 7]: return 0
        elif stage in [1, 6]: return 1
        elif stage in [2, 5]: return 2
        elif stage in [3, 4]: return 3
    else:
        if stage in [0, 7]: return 3
        elif stage in [1, 6]: return 2
        elif stage in [2, 5]: return 1
        elif stage in [3, 4]: return 0


def run_schedule(task_forward, multiplier):
    # Calculate backward pass and enforce Integer constraint
    task_backward = [int(x * multiplier) for x in task_forward]
    task_backward.reverse()

    task_full = task_forward + task_backward
    # task_matrix structure: [Task 0 Stages, Task 1 Stages, ...]
    task_matrix = [task_full for _ in range(4)]

    machine_availability = [0] * 4
    events = []
    scheduling_details = {i: [] for i in range(4)}

    # Initialize first stage for all tasks
    for task_id in range(4):
        duration = task_matrix[task_id][0]
        # Priority Queue Tuple: (time, negative_stage_priority, task_id, duration)
        heapq.heappush(events, (0, 0, task_id, duration))

    while events:
        time, neg_stage, task_id, duration = heapq.heappop(events)
        stage = -neg_stage
        ready_time = max(time, 0)
        machine = get_machine(task_id, stage)

        if machine_availability[machine] <= ready_time:
            # Machine is free
            start = ready_time
            end = start + duration
            machine_availability[machine] = end
            scheduling_details[task_id].append((stage, machine, start, end))

            # Queue next stage if exists
            if stage + 1 < 8:
                next_stage = stage + 1
                next_dur = task_matrix[task_id][next_stage]
                heapq.heappush(events, (end, -next_stage, task_id, next_dur))
        else:
            # Machine is busy, wait until it is free
            heapq.heappush(events, (machine_availability[machine], neg_stage, task_id, duration))

    return scheduling_details

# --- 2. Interactive Widget Logic ---

TARGET_SUM = 100
style = {'description_width': 'initial'}
layout_slider = widgets.Layout(width='300px')
layout_text = widgets.Layout(width='80px')

# Multiplier Slider (Float, but results will be cast to Int)
s_mult = widgets.FloatSlider(value=2.0, min=0.5, max=4.0, step=0.5, description='Backward Multiplier', style=style, layout=layout_slider)

# Stage Sliders (Now IntSlider)
s1 = widgets.IntSlider(value=35, min=0, max=TARGET_SUM, step=1, description='Stage 1', style=style, layout=layout_slider)
t1 = widgets.IntText(value=35, step=1, layout=layout_text)
widgets.jslink((s1, 'value'), (t1, 'value'))

s2 = widgets.IntSlider(value=25, min=0, max=TARGET_SUM, step=1, description='Stage 2', style=style, layout=layout_slider)
t2 = widgets.IntText(value=25, step=1, layout=layout_text)
widgets.jslink((s2, 'value'), (t2, 'value'))

s3 = widgets.IntSlider(value=15, min=0, max=TARGET_SUM, step=1, description='Stage 3', style=style, layout=layout_slider)
t3 = widgets.IntText(value=15, step=1, layout=layout_text)
widgets.jslink((s3, 'value'), (t3, 'value'))

t4_label = widgets.Label(value="Stage 4 (Remainder):")
t4 = widgets.IntText(value=45, disabled=True, layout=widgets.Layout(width='100px'))

out = widgets.Output()

def update_simulation(change=None):
    val1, val2, val3 = s1.value, s2.value, s3.value
    mult = s_mult.value

    current_total = val1 + val2 + val3
    remainder = TARGET_SUM - current_total

    if remainder < 0:
        remainder = 0

    t4.value = remainder

    task_forward = [val1, val2, val3, remainder]

    with out:
        clear_output(wait=True)
        if current_total > TARGET_SUM:
            print(f"Warning: Input sum ({current_total}) exceeds {TARGET_SUM}. Stage 4 set to 0.")

        details = run_schedule(task_forward, mult)
        visualize(details, task_forward, mult)

s1.observe(update_simulation, names='value')
s2.observe(update_simulation, names='value')
s3.observe(update_simulation, names='value')
s_mult.observe(update_simulation, names='value')

row_mult = widgets.HBox([s_mult])
row1 = widgets.HBox([s1, t1])
row2 = widgets.HBox([s2, t2])
row3 = widgets.HBox([s3, t3])
row4 = widgets.HBox([t4_label, t4])

ui = widgets.VBox([row_mult, row1, row2, row3, row4, out])

update_simulation()
display(ui)