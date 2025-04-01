"""
THIS IS THE CORRECT CODE TO EXTRACT MULTIBOSS FEATURES.
"""
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from ml.multi_boss import MultiBOSS
import warnings

warnings.filterwarnings("ignore")

"""
# Uncomment to work with the restricted data
# === Paths
DATA_DIR = "processed_data"
BIN_META_PATH = os.path.join(DATA_DIR, "bin_metadata.csv")
SEGMENT_META_PATH = os.path.join(DATA_DIR, "filtered_metadata_gyroscope_right_PD_HC.json")
OUTPUT_FEATURES = "features/multiboss_features.pkl"
OUTPUT_LABELS = "features/multiboss_labels.csv"
"""

# Uncomment to work with the original right data
# === Paths
DATA_DIR = "all_processed_data"
BIN_META_PATH = os.path.join(DATA_DIR, "bin_metadata.csv")
SEGMENT_META_PATH = os.path.join(DATA_DIR, "filtered_metadata_all_right_PD_HC.json")
OUTPUT_FEATURES = "features_all/multiboss_features.pkl"
OUTPUT_LABELS = "features_all/multiboss_labels.csv"

# === Load metadata
bin_metadata = pd.read_csv(BIN_META_PATH)
with open(SEGMENT_META_PATH, "r") as f:
    segment_meta = json.load(f)

# === Load 9 segments per patient (3 tasks × 3 axes)
def load_segments(file_path, segment_meta):
    data = np.fromfile(file_path, dtype=np.float32)
    segments = [data[s["start"]:s["end"]] for s in segment_meta["tasks"]]
    return np.stack(segments, axis=0)  # shape: (9, 976) or (66, 976) depending on the task

# === Load dataset
def extract_features(bin_metadata, segment_meta):
    X_list, y_list, pid_list = [], [], []

    for _, row in tqdm(bin_metadata.iterrows(), total=len(bin_metadata), desc="Extracting MultiBOSS"):
        file_path = os.path.join(row["FilePath"])
        pid = row["PatientID"]
        label = row["Label"]

        try:
            segment_array = load_segments(file_path, segment_meta)  # (9, 976) or (66, 976)
            X_list.append(segment_array)
            y_list.append(label)
            pid_list.append(pid)
        except Exception as e:
            print(f"⚠️ Failed for {pid}: {e}")

    X_raw = np.stack(X_list, axis=0)  # shape: (n_samples, 9, 976)
    y = pd.Series(y_list, name="Label")
    pids = pd.Series(pid_list, name="PatientID")
    return X_raw, y, pids

# === Main
def main():
    # Uncomment to work with the right data
    os.makedirs("features_all", exist_ok=True)
    """
    # Uncomment to work with the restricted data
    os.makedirs("features_all", exist_ok=True)
    """
    X_raw, y, pids = extract_features(bin_metadata, segment_meta)

    print(f"🧠 Fitting MultiBOSS on shape: {X_raw.shape}")
    model = MultiBOSS(
        data_shape=(66, 976),
        window_sizes=(40,),
        word_size=2,
        n_bins=3,
        window_step=2,
        buf_path="./boss_cache/"
    )
    """ 
    # Uncomment to work with the restricted data
    print(f"🧠 Fitting MultiBOSS on shape: {X_raw.shape}")
    model = MultiBOSS(
        data_shape=(9, 976),
        window_sizes=(40,),
        word_size=2,
        n_bins=3,
        window_step=2,
        buf_path="./boss_cache/"
    )
    """

    X_boss = model.fit_transform(X_raw, y)

    print("💾 Saving features and labels...")
    X_df = pd.DataFrame(X_boss)
    X_df["PatientID"] = pids.values
    X_df.to_pickle(OUTPUT_FEATURES)
    y.to_csv(OUTPUT_LABELS, index=False)

    print("✅ MultiBOSS feature extraction complete!")

if __name__ == "__main__":
    main()
