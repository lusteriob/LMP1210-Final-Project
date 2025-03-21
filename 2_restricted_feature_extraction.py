import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyts.transformation import BOSS

# Paths
DATA_DIR = "processed_data"
BIN_META_PATH = os.path.join(DATA_DIR, "bin_metadata.csv")
SEGMENT_META_PATH = os.path.join(DATA_DIR, "filtered_metadata_gyroscope_right_PD_HC.json")

# Load metadata
bin_metadata = pd.read_csv(BIN_META_PATH)
with open(SEGMENT_META_PATH, 'r') as f:
    segment_metadata = json.load(f)  # List of 9 segments (task × axis)

def load_segments(file_path, global_segment_meta):
    data = np.fromfile(file_path, dtype=np.float32)
    segments = [data[s['start']:s['end']] for s in global_segment_meta["tasks"]]
    return segments

def apply_boss_multiscale(segment, window_sizes=[20, 40, 80], word_size=4, n_bins=4):
    """Apply BOSS with different window sizes and concatenate histograms."""
    features = []
    segment_2d = segment.reshape(1, -1)  # pyts expects 2D: (n_samples, n_timestamps)
    for window_size in window_sizes:
        boss = BOSS(window_size=window_size, word_size=word_size, n_bins=n_bins, norm_mean=False)
        hist = boss.fit_transform(segment_2d)
        features.append(hist.toarray())  # 🔥 THIS LINE FIXES THE ERROR
    return np.hstack(features)

def extract_all_boss_features(bin_metadata, segment_metadata, max_length=5000):
    feature_list = []
    label_list = []
    patient_ids = []

    for _, row in tqdm(bin_metadata.iterrows(), total=len(bin_metadata), desc="Extracting BOSS features"):
        file_path = os.path.join(row["FilePath"])
        label = row["Label"]
        patient_id = row["PatientID"]

        segments = load_segments(file_path, segment_metadata)
        patient_features = []
        for segment in segments:
            segment_features = apply_boss_multiscale(segment)
            patient_features.append(segment_features.ravel())

        # Combine all segment features
        full_feature_vector = np.hstack(patient_features)

        # Pad or truncate to fixed length
        padded = np.zeros(max_length, dtype=np.float32)
        length = min(len(full_feature_vector), max_length)
        padded[:length] = full_feature_vector[:length]

        feature_list.append(padded)
        label_list.append(label)
        patient_ids.append(patient_id)

    # Convert to DataFrame
    X = pd.DataFrame(feature_list)
    X["PatientID"] = patient_ids
    y = pd.Series(label_list, name="Label")
    return X, y


def main():
    os.makedirs("features", exist_ok=True)  # Ensure the folder exists

    X, y = extract_all_boss_features(bin_metadata, segment_metadata)

    # Save to /features/
    X.to_pickle("features/boss_features_pads_style.pkl")
    y.to_csv("features/boss_labels_pads_style.csv", index=False)
    
    print("✅ Feature extraction complete and saved to /features/")

if __name__ == "__main__":
    main()
