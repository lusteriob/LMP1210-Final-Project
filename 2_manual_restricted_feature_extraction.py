import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import welch

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
    segments = [data[s["start"]:s["end"]] for s in global_segment_meta["tasks"]]
    return segments

def extract_psd_features(signal, fs=128, max_freq=19):
    freqs, psd = welch(signal, fs=fs, nperseg=256)
    psd_log = np.log10(psd + 1e-10)  # avoid log(0)
    selected = psd_log[(freqs > 0) & (freqs <= max_freq)]
    return selected[:19] if len(selected) >= 19 else np.pad(selected, (0, 19 - len(selected)))

def extract_time_domain_features(signal):
    features = []
    segment_length = len(signal) // 4
    for i in range(4):
        seg = signal[i * segment_length : (i + 1) * segment_length]
        features.append(np.std(seg))
        features.append(np.max(np.abs(seg)))
        features.append(np.sum(np.abs(seg)**2))
    return features  # total 12 features

def extract_manual_features(bin_metadata, segment_metadata):
    feature_list = []
    label_list = []
    patient_ids = []

    for _, row in tqdm(bin_metadata.iterrows(), total=len(bin_metadata), desc="Extracting Manual Features"):
        file_path = os.path.join(row["FilePath"])
        label = row["Label"]
        patient_id = row["PatientID"]

        segments = load_segments(file_path, segment_metadata)
        patient_features = []

        for segment in segments:
            # segment = 1D time series for one axis of one task (976 samples)
            psd_feats = extract_psd_features(segment)            # 19
            time_feats = extract_time_domain_features(segment)   # 12
            patient_features.extend(psd_feats)
            patient_features.extend(time_feats)

        feature_list.append(patient_features)
        label_list.append(label)
        patient_ids.append(patient_id)

    X = pd.DataFrame(feature_list)
    X["PatientID"] = patient_ids
    y = pd.Series(label_list, name="Label")
    return X, y

def main():
    os.makedirs("features", exist_ok=True)

    X, y = extract_manual_features(bin_metadata, segment_metadata)

    # Save to /features/
    X.to_pickle("features/manual_features_pads_style.pkl")
    y.to_csv("features/manual_labels_pads_style.csv", index=False)

    print("✅ Manual feature extraction complete and saved to /features/")

if __name__ == "__main__":
    main()
