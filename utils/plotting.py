import os
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
import numpy as np

# Path to the event file
event_file_path = "/home/smondal/Desktop/DexterousHands/bidexhands/logs/CubeLift/ppo/ppo_without_affordance/events.out.tfevents.1735821776.ar-MS-7E26.648553.0"

# Initialize the EventAccumulator
event_acc = event_accumulator.EventAccumulator(event_file_path)
event_acc.Reload()  # Load the event file data

# List all available tags
scalar_tags = event_acc.Tags()['scalars']
print(f"Available scalar tags: {scalar_tags}")

# Choose a scalar tag to plot
tag_to_plot = scalar_tags[1]  # 0,1,7,6

# Extract the scalar data (step, wall_time, value)
steps, wall_times, values = [], [], []
for event in event_acc.Scalars(tag_to_plot):
    steps.append(event.step)
    wall_times.append(event.wall_time)
    values.append(event.value)

steps_np = np.array(steps)
values_np = np.array(values)
# Apply Gaussian smoothing
sigma = 100  # Standard deviation for Gaussian kernel; increase for more smoothing
smooth_values = gaussian_filter1d(values, sigma)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(steps, smooth_values, label=tag_to_plot, linestyle='-', linewidth=2)
plt.xlabel('Iteration Step')
plt.ylabel('Percentage Success Rate') # plt.ylabel('Episode Length'), plt.ylabel('Reward Value'), plt.ylabel('Percentage Success Rate')
# plt.title(f"{tag_to_plot}")
plt.legend()
plt.grid(True)
plt.show()
