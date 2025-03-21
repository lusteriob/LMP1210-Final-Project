import os
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Define paths
data_dir = "data/preprocessed"
movement_dir = os.path.join(data_dir, "movement")
metadata_json_path = os.path.join(data_dir, "preprocessed_metadata.json")
file_list_path = os.path.join(data_dir, "file_list.csv")

# Output directory for all tasks and sensors
output_dir = "all_processed_data"
Path(output_dir).mkdir(parents=True, exist_ok=True)
output_metadata_file = os.path.join(output_dir, "bin_metadata.csv")
filtered_metadata_output = os.path.join(output_dir, "filtered_metadata_all_right_PD_HC.json")

# Load patient metadata
def load_patient_data(file_list_path):
    df = pd.read_csv(file_list_path)
    df = df[df["label"].isin([0, 1])]
    patient_data = dict(zip(df["id"].astype(str).str.zfill(3), df["label"]))
    return df, patient_data

# Load preprocessed metadata from JSON
def load_preprocessed_metadata(metadata_json_path):
    with open(metadata_json_path, "r") as f:
        metadata = json.load(f)["tasks"]
    return metadata

# Filter metadata: RightWrist + Accelerometer or Gyroscope + all tasks
def filter_metadata(metadata):
    return [entry for entry in metadata if entry["wrist"] == "RightWrist"]

# Process and save filtered .bin files
def process_movement_data(movement_dir, patient_data, filtered_metadata):
    filtered_metadata_records = []
    corrected_metadata = []

    for patient_id in patient_data.keys():
        file_path = os.path.join(movement_dir, f"{patient_id}_ml.bin")
        if not os.path.exists(file_path):
            continue

        data = np.fromfile(file_path, dtype=np.float32)
        filtered_segments = []
        current_index = 0

        for entry in filtered_metadata:
            segment = data[entry["start"]:entry["end"]]
            filtered_segments.append(segment)
            segment_length = len(segment)
            corrected_metadata.append({
                "name": entry["name"],
                "wrist": entry["wrist"],
                "sensor": entry["sensor"],
                "start": current_index,
                "end": current_index + segment_length
            })
            current_index += segment_length

        if not filtered_segments:
            continue

        filtered_data = np.concatenate(filtered_segments)
        output_bin_path = os.path.join(output_dir, f"{patient_id}.bin")
        filtered_data.astype(np.float32).tofile(output_bin_path)

        label = patient_data[patient_id]
        filtered_metadata_records.append([patient_id, label, output_bin_path, len(filtered_data)])

    metadata_df = pd.DataFrame(filtered_metadata_records, columns=["PatientID", "Label", "FilePath", "TotalSamples"])
    metadata_df.to_csv(output_metadata_file, index=False)

    # Save corrected metadata
    with open(filtered_metadata_output, "w") as f:
        json.dump({"tasks": corrected_metadata}, f, indent=4)

# Run
if __name__ == "__main__":
    patient_df, patient_data = load_patient_data(file_list_path)
    metadata = load_preprocessed_metadata(metadata_json_path)
    filtered_metadata = filter_metadata(metadata)
    process_movement_data(movement_dir, patient_data, filtered_metadata)
    print(f"✅ Processed data saved to {output_dir}. Metadata saved as {filtered_metadata_output}.")
