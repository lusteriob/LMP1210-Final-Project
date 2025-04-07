import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# === Paths ===
DATA_DIR = "data/preprocessed"
BIN_DIR = os.path.join(DATA_DIR, "movement")
META_PATH = r"C:\Users\Venora\Downloads\LMP1210-Final-Project\processed_data\filtered_metadata_gyroscope_right_PD_HC.json"
FILE_LIST_PATH = os.path.join(DATA_DIR, "file_list.csv")
OUTPUT_DIR = "regenerated_bins"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# === Load segment metadata ===
with open(META_PATH, "r") as f:
    segment_metadata = json.load(f)["tasks"]

# === Load patient list and filter for PD or HC
df = pd.read_csv(FILE_LIST_PATH)
df = df[df["label"].isin([0, 1])]
patient_ids = df["id"].astype(str).str.zfill(3).tolist()
print(f"Found {len(patient_ids)} valid patients.")

# === Rebuild channel names in bin file order ===
channel_names = []
TASKS = [
    "Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2",
    "StretchHold", "HoldWeight", "DrinkGlas", "CrossArms",
    "TouchNose", "Entrainment1", "Entrainment2"
]
SENSORS = ["Gyroscope", "Acceleration"]
WRISTS = ["RightWrist", "LeftWrist"]
AXES = ["X", "Y", "Z"]

for task in TASKS:
    for sensor in SENSORS:
        for wrist in WRISTS:
            for axis in AXES:
                channel_names.append(f"{task}_{sensor}_{wrist}_{axis}")

# === Process each patient ===
for patient_id in patient_ids:
    bin_path = os.path.join(BIN_DIR, f"{patient_id}_ml.bin")

    if not os.path.exists(bin_path):
        print(f"Skipping {patient_id} — bin file not found.")
        continue

    try:
        data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 132)
    except ValueError:
        print(f"Skipping {patient_id} — reshape failed.")
        continue

    # Filter metadata entries by filename
    patient_segments = [
        seg for seg in segment_metadata if seg.get("filename") == f"{patient_id}_ml.bin"
    ]

    if len(patient_segments) < 9:
        print(f"Skipping {patient_id} — only found {len(patient_segments)} segments.")
        continue

    all_segments = []

    for seg in patient_segments:
        task = seg["name"]
        sensor = seg["sensor"]
        wrist = seg["wrist"]
        axis = seg["axis"]
        start_idx = seg["start"]
        end_idx = seg["end"]

        channel_name = f"{task}_{sensor}_{wrist}_{axis}"
        if channel_name not in channel_names:
            print(f"Missing channel: {channel_name}")
            continue

        col_idx = channel_names.index(channel_name)
        segment_data = data[start_idx:end_idx, col_idx]

        if segment_data.size == 0:
            print(f"Empty segment for {patient_id}: {channel_name}")
            continue

        all_segments.append(segment_data.astype(np.float32))

    # Save final full-segment .bin file
    if len(all_segments) == 9:
        full_array = np.concatenate(all_segments)
        out_path = os.path.join(OUTPUT_DIR, f"{patient_id}_ml_full.bin")
        full_array.tofile(out_path)
        print(f"Saved: {out_path} | Shape: {full_array.shape}")
    else:
        print(f"Incomplete segments for {patient_id}, skipping save.")
