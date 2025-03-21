import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyts.transformation import BOSS

# Configs
WINDOW_SIZES = [20, 40, 80]
WORD_SIZES = [4, 6]
N_BINS = [2, 4]

# Paths
DATA_DIR = "processed_data"
BIN_META_PATH = os.path.join(DATA_DIR, "bin_metadata.csv")
SEGMENT_META_PATH = os.path.join(DATA_DIR, "filtered_metadata_gyroscope_right_PD_HC.json")

# Load metadata
bin_metadata = pd.read_csv(BIN_META_PATH)
with open(SEGMENT_META_PATH, "r") as f:
    segment_metadata = json.load(f)

def load_segments(file_path, global_segment_meta):
    data = np.fromfile(file_path, dtype=np.float32)
    return [data[s["start"]:s["end"]] for s in global_segment_meta["tasks"]]

def extract_boss_multi_config(segment):
    """Extracts multiple BOSS histograms from a single segment."""
    segment_2d = segment.reshape(1, -1)
    histograms = []

    for w in WINDOW_SIZES:
        for ws in WORD_SIZES:
            for nb in N_BINS:
                boss = BOSS(window_size=w, word_size=ws, n_bins=nb, norm_mean=False)
                hist = boss.fit_transform(segment_2d).toarray().ravel()
                histograms.append(hist)

    return np.hstack(histograms)

def extract_all_features(bin_metadata, segment_metadata):
    feature_list = []
    label_list = []
    patient_ids = []

    for _, row in tqdm(bin_metadata.iterrows(), total=len(bin_metadata), desc="Extracting Smart BOSS"):
        file_path = os.path.join(row["FilePath"])
        label = row["Label"]
        patient_id = row["PatientID"]

        segments = load_segments(file_path, segment_metadata)
        patient_features = []

        for segment in segments:
            try:
                segment_features = extract_boss_multi_config(segment)
                patient_features.append(segment_features)
            except Exception as e:
                print(f"⚠️ Error extracting segment in {file_path}: {e}")
                continue

        full_feature_vector = np.hstack(patient_features)
        feature_list.append(full_feature_vector)
        label_list.append(label)
        patient_ids.append(patient_id)

    # Padding to max feature length (optional)
    max_len = max(len(f) for f in feature_list)
    X_padded = np.zeros((len(feature_list), max_len), dtype=np.float32)
    for i, f in enumerate(feature_list):
        X_padded[i, :len(f)] = f

    X = pd.DataFrame(X_padded)
    X["PatientID"] = patient_ids
    y = pd.Series(label_list, name="Label")

    return X, y

def main():
    os.makedirs("features", exist_ok=True)
    X, y = extract_all_features(bin_metadata, segment_metadata)

    X.to_pickle("features/smart_boss_features.pkl")
    y.to_csv("features/smart_boss_labels.csv", index=False)
    print("✅ Smart BOSS features saved to features/")

if __name__ == "__main__":
    main()
