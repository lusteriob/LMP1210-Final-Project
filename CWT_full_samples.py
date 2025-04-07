import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from tqdm import tqdm  # type: ignore

# === Paths ===
BIN_META_PATH = r"C:\Users\Venora\Downloads\LMP1210-Final-Project\processed_data\bin_metadata.csv"
SEGMENT_META_PATH = r"C:\Users\Venora\Downloads\LMP1210-Final-Project\processed_data\filtered_metadata_gyroscope_right_PD_HC.json"
OUTPUT_DIR = "cwt_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load metadata ===
bin_metadata = pd.read_csv(BIN_META_PATH)

with open(SEGMENT_META_PATH, "r") as f:
    segment_template = json.load(f)["tasks"]

# === CWT Parameters ===
sampling_rate = 100
freq_min = 1
freq_max = 19
num_freqs = 32
frequencies = np.linspace(freq_min, freq_max, num_freqs)
wavelet = 'cmor1.5-1.0'
central_freq = pywt.central_frequency(wavelet)
scales = central_freq * sampling_rate / frequencies

# === Main Loop ===
for _, row in tqdm(bin_metadata.iterrows(), total=len(bin_metadata), desc="Generating CWT Images"):
    patient_id = str(row["PatientID"]).zfill(3)
    label = row["Label"]
    file_path = row["FilePath"]

    if not os.path.exists(file_path):
        print(f"Missing file: {file_path}")
        continue

    try:
        data = np.fromfile(file_path, dtype=np.float32)
        n_samples = int(row["TotalSamples"])
        data = data.reshape((n_samples, -1))  # (8784, 9) for 9 segments
    except Exception as e:
        print(f"Error loading data for patient {patient_id}: {e}")
        continue

    out_dir = os.path.join(OUTPUT_DIR, patient_id)
    os.makedirs(out_dir, exist_ok=True)

    for seg in segment_template:
        task = seg["name"]
        axis = seg["axis"]
        start = seg["start"]
        end = seg["end"]

        if start >= data.shape[0] or end > data.shape[0]:
            print(f"Skipping out-of-bounds segment: {task}_{axis} ({start}:{end})")
            continue

        try:
            signal = data[start:end]
        except Exception as e:
            print(f"Error indexing segment {task}_{axis} for {patient_id}: {e}")
            continue

        if signal.size == 0 or np.all(signal == signal[0]):
            continue

        try:
            coef_matrix, freqs = pywt.cwt(
                signal.flatten(),
                scales=scales,
                wavelet=wavelet,
                sampling_period=1/sampling_rate
            )
            power = np.abs(coef_matrix) ** 2

            plt.figure(figsize=(10, 4))
            plt.imshow(
                power,
                aspect='auto',
                cmap='viridis',
                origin='lower',
                extent=[0, len(signal), frequencies[0], frequencies[-1]]
            )
            plt.colorbar(label='Power')
            plt.xlabel('Time (samples)')
            plt.ylabel('Frequency (Hz)')
            plt.title(f"{task} - {axis} (CWT)")

            out_path = os.path.join(out_dir, f"{task}_{axis}_CWT.png")
            plt.savefig(out_path, dpi=150)
            plt.close()

        except Exception as e:
            print(f"Error generating CWT image for {patient_id} {task}_{axis}: {e}")
            continue
