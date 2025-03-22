import numpy as np
import os
import matplotlib.pyplot as plt
#from scipy.signal import morlet2
#from scipy.signal._cwt import cwt
import pywt

#load the raw binary file for patient 001
file_path = "data/preprocessed/movement/001_ml.bin"
data = np.fromfile(file_path, dtype=np.float32)
# Try guessing number of time steps
for n_channels in range(100, 150):
    if len(data) % n_channels == 0:
        print(f"✓ Can reshape to (time_steps, {n_channels}):", len(data) // n_channels)

data = data.reshape((-1, 132))
print("Shape of data:", data.shape)  # should say (976, 132)

channel_names = []
TASKS = [
    "Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2", "StretchHold", "HoldWeight",
    "DrinkGlas", "CrossArms", "TouchNose", "Entrainment1", "Entrainment2"
]
SENSORS = ["Gyroscope", "Acceleration"]
WRISTS = ["RightWrist", "LeftWrist"]
AXES = ["X", "Y", "Z"]

for task in TASKS:
    for sensor in SENSORS:
        for wrist in WRISTS:
            for axis in AXES:
                name = f"{task}_{sensor}_{wrist}_{axis}"
                channel_names.append(name)

print("Total channel names:", len(channel_names))
print("Example channels:", channel_names[:5])
import json

# Load the filtered metadata
with open("C:\\Users\\Venora\\Downloads\\LMP1210-Final-Project\\processed_data\\filtered_metadata_gyroscope_right_PD_HC.json", "r") as f:
    metadata = json.load(f)["tasks"]

# Prepare a list of segments to process
segments_to_process = []

for entry in metadata:
    task = entry["name"]
    sensor = entry["sensor"]
    wrist = entry["wrist"]
    axis = entry["axis"]
    start = entry["start"]
    end = entry["end"]

    # Build the channel name (same format as in our list)
    channel_name = f"{task}_{sensor}_{wrist}_{axis}"

    # Find the column index
    if channel_name in channel_names:
        col_idx = channel_names.index(channel_name)
        segments_to_process.append({
            "task": task,
            "axis": axis,
            "start": start,
            "end": end,
            "channel_name": channel_name,
            "column_index": col_idx
        })
    else:
        print(f"⚠️ Channel not found: {channel_name}")

print(f"\nFound {len(segments_to_process)} segments to process.")
print("Example segment:", segments_to_process[0])

#output_dir = "images"
#os.makedirs(output_dir, exist_ok=True)

# Frequencies: 1 to 19 Hz
#frequencies = np.arange(1, 20)
#sampling_rate = 128  # Hz (assumed)

#for seg in segments_to_process:
#    signal = data[seg["start"]:seg["end"], seg["column_index"]]
    
#    widths = sampling_rate / (2 * np.pi * frequencies)  # Convert Hz to wavelet scales

#    coef_matrix = cwt(signal, morlet2, widths, w=6)

 #   power = np.abs(coef_matrix) ** 2  # Power = magnitude squared

    # Plot & save the heatmap
  #  plt.figure(figsize=(10, 4))
   # plt.imshow(power, aspect="auto", cmap="viridis", origin="lower",
    #           extent=[0, len(signal), frequencies[0], frequencies[-1]])
    #plt.colorbar(label="Power")
    #plt.xlabel("Time")
    #plt.ylabel("Frequency (Hz)")
    #plt.title(f'{seg["task"]} - {seg["axis"]} (CWT)')

#    filename = f'{seg["task"]}_{seg["axis"]}_CWT.png'
#    plt.savefig(os.path.join(output_dir, filename), dpi=150)
 #   plt.close()

  #  print(f"✅ Saved: {filename}")


# Output folder for saved images
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

# CWT settings
sampling_rate = 128  # Hz
scales = np.arange(1, 20)  # Corresponds to frequencies 1–19 Hz

# Loop through all valid segments
for seg in segments_to_process:
    task = seg["task"]
    axis = seg["axis"]
    start = seg["start"]
    end = seg["end"]
    col_idx = seg["column_index"]

    # Skip segments that exceed the available data range
    if start >= data.shape[0] or end > data.shape[0]:
        print(f"Skipping out-of-bounds segment: {task} {axis} ({start}–{end})")
        continue

    # Extract the signal from the specified channel and time range
    signal = data[start:end, col_idx]
    print(f"Processing {task} - {axis} | Signal length: {len(signal)}")

    # Skip if the signal is empty or constant
    if signal.size == 0 or np.all(signal == signal[0]):
        print(f"Skipping empty or constant signal: {task} {axis}")
        continue

    try:
        # Apply Continuous Wavelet Transform using a complex Morlet wavelet
        coef_matrix, freqs = pywt.cwt(
            signal,
            scales=scales,
            wavelet='cmor1.5-1.0',
            sampling_period=1/sampling_rate
        )

        power = np.abs(coef_matrix) ** 2  # Compute power

        # Plot the scaleogram and save the image
        plt.figure(figsize=(10, 4))
        plt.imshow(
            power,
            aspect='auto',
            cmap='viridis',
            origin='lower',
            extent=[0, len(signal), freqs[0], freqs[-1]]
        )
        plt.colorbar(label='Power')
        plt.xlabel('Time (samples)')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'{task} - {axis} (CWT)')

        filename = f"{task}_{axis}_CWT.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150)
        plt.close()

        print(f"Saved: {filename}")

    except Exception as e:
        print(f"Error generating CWT for {task} {axis}: {e}")
        continue