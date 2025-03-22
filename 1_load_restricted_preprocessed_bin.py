"""
THIS SCRIPT GRABS RIGHT HANDED, GYROSCOPE DATA FROM 3 SPECIFIC TASKS FROM THE PREPROCESSED .BIN FILES AND SAVES THEM IN A NEW DIRECTORY.
THE METADATA IS ALSO UPDATED TO REFLECT THE NEW FILE LOCATIONS.
"""
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

# Output directory
output_dir = "processed_data"
Path(output_dir).mkdir(parents=True, exist_ok=True)
output_metadata_file = os.path.join(output_dir, "bin_metadata.csv")

# Load patient metadata (from file_list.csv)
def load_patient_data(file_list_path):
    df = pd.read_csv(file_list_path)
    df = df[df["label"].isin([0, 1])]  # Keep only PD (1) and HC (0) subjects
    patient_data = dict(zip(df["id"].astype(str).str.zfill(3), df["label"]))
    return df, patient_data

# Load preprocessed metadata from JSON
def load_preprocessed_metadata(metadata_json_path):
    with open(metadata_json_path, "r") as f:
        metadata = json.load(f)["tasks"]
    return metadata

# Filter metadata based on criteria
def filter_metadata(metadata):
    filtered = [
        entry for entry in metadata
        if (
            entry["wrist"] == "RightWrist" and
            entry["sensor"] == "Gyroscope" and
            entry["name"] in ["TouchNose", "RelaxedTask1", "Relaxed1"]
        )
    ]
    return filtered

# Process and filter each .bin file
def process_movement_data(movement_dir, patient_data, filtered_metadata):
    filtered_metadata_records = []
    
    for patient_id in patient_data.keys():
        file_path = os.path.join(movement_dir, f"{patient_id}_ml.bin")
        if not os.path.exists(file_path):
            print(f"Skipping: {patient_id} (No file found)")
            continue
        
        # Load binary data
        data = np.fromfile(file_path, dtype=np.float32)
        
        # Initialize list to collect filtered data segments
        filtered_segments = []
        
        for entry in filtered_metadata:
            start_idx = entry["start"]
            end_idx = entry["end"]
            segment = data[start_idx:end_idx]
            filtered_segments.append(segment)
        
        if not filtered_segments:
            print(f"No matching data for patient {patient_id}. Skipping...")
            continue
        
        # Concatenate all filtered segments
        filtered_data = np.concatenate(filtered_segments)
        
        # Save filtered .bin file
        output_bin_path = os.path.join(output_dir, f"{patient_id}.bin")
        filtered_data.astype(np.float32).tofile(output_bin_path)
        
        # Store metadata
        label = patient_data[patient_id]
        filtered_metadata_records.append([patient_id, label, output_bin_path, len(filtered_data)])
    
    # Save updated metadata
    metadata_df = pd.DataFrame(filtered_metadata_records, columns=["PatientID", "Label", "FilePath", "TotalSamples"])
    metadata_df.to_csv(output_metadata_file, index=False)

# Main execution
if __name__ == "__main__":
    print("Loading patient data...")
    patient_df, patient_data = load_patient_data(file_list_path)
    
    print("Loading preprocessed metadata...")
    metadata = load_preprocessed_metadata(metadata_json_path)
    
    print("Filtering metadata...")
    filtered_metadata = filter_metadata(metadata)
    
    print("Processing and filtering movement data...")
    process_movement_data(movement_dir, patient_data, filtered_metadata)
    
    print(f"✅ Preprocessed data saved to {output_dir}. Metadata saved as {output_metadata_file}.")
