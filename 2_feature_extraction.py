import os
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from pyts.transformation import BOSS
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

# Define paths
bin_output_dir = "processed_data"
metadata_file = os.path.join(bin_output_dir, "bin_metadata.csv")
segment_labels_file = os.path.join(bin_output_dir, "segment_labels.json")
boss_feature_file = os.path.join(bin_output_dir, "boss_features.pkl")
tsfresh_feature_file = os.path.join(bin_output_dir, "tsfresh_features.pkl")

# Load segment labels
with open(segment_labels_file, "r") as f:
    segment_info = json.load(f)

segment_labels = segment_info["segment_labels"]
segment_length = segment_info["segment_length"]

# Load metadata
metadata_df = pd.read_csv(metadata_file)

# MultiBOSS class
class MultiBOSS(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, window_size=40, window_step=2, word_size=2, n_bins=3):
        self.window_size = window_size
        self.window_step = window_step
        self.word_size = word_size
        self.n_bins = n_bins
        self.boss_models = {}

    def fit_transform(self, X):
        boss_model = BOSS(
            sparse=False, window_size=self.window_size, window_step=self.window_step,
            word_size=self.word_size, n_bins=self.n_bins, norm_std=False, norm_mean=False,
            anova=False, drop_sum=False
        )
        boss_model.fit(X)
        return boss_model.transform(X)

# Process each patient individually
def process_patient(patient_id, label, file_path):
    print(f"\n🟢 Processing patient {patient_id}...")

    # Load binary data
    data = np.fromfile(file_path, dtype=np.float32).reshape(-1, len(segment_labels))

    # ==========================
    # 🔹 BOSS Feature Extraction
    # ==========================
    print(f"⚡ Extracting BOSS features for patient {patient_id}...")
    boss_extractor = MultiBOSS(window_size=40)
    patient_boss_features = []

    for segment_idx in range(len(segment_labels)):
        segment_data = data[:, segment_idx].reshape(1, -1)
        transformed = boss_extractor.fit_transform(segment_data)
        patient_boss_features.append(transformed.flatten())

    boss_features = np.hstack(patient_boss_features)
    boss_df = pd.DataFrame([boss_features], columns=[f"BOSS_{i}" for i in range(len(boss_features))])
    boss_df.insert(0, "PatientID", patient_id)
    boss_df.insert(1, "Label", label)

    # Save BOSS features incrementally
    if os.path.exists(boss_feature_file):
        with open(boss_feature_file, "rb") as f:
            existing_data = pickle.load(f)
        boss_df = pd.concat([existing_data, boss_df], ignore_index=True)

    with open(boss_feature_file, "wb") as f:
        pickle.dump(boss_df, f)

    print(f"✅ BOSS features saved for patient {patient_id}.")

    # ============================
    # 🔹 TSFresh Feature Extraction
    # ============================
    print(f"⚡ Extracting TSFresh features for patient {patient_id}...")

    df_list = []
    for segment_idx, segment in enumerate(segment_labels):
        segment_data = data[:, segment_idx]
        df, _ = make_forecasting_frame(segment_data, kind=segment, rolling_direction=1, max_timeshift=1)

        # ✅ FIX: Ensure `id` column is correctly formatted
        df["id"] = df["id"].apply(lambda x: x[1] if isinstance(x, tuple) else x)
        df["PatientID"] = patient_id
        df_list.append(df)

    ts_df_combined = pd.concat(df_list, axis=0)

    print(f"🔍 TSFresh input data for patient {patient_id}:\n{ts_df_combined.head()}")

    try:
        ts_fresh_features = extract_features(
            ts_df_combined, column_id="PatientID", column_sort="time",
            disable_progressbar=False, n_jobs=2, chunksize=50
        )
        ts_fresh_features.reset_index(inplace=True)
        ts_fresh_features.rename(columns={"index": "PatientID"}, inplace=True)
        ts_fresh_df = ts_fresh_features.merge(metadata_df[["PatientID", "Label"]], on="PatientID", how="left")

        # Save TSFresh features incrementally
        if os.path.exists(tsfresh_feature_file):
            with open(tsfresh_feature_file, "rb") as f:
                existing_data = pickle.load(f)
            ts_fresh_df = pd.concat([existing_data, ts_fresh_df], ignore_index=True)

        with open(tsfresh_feature_file, "wb") as f:
            pickle.dump(ts_fresh_df, f)

        print(f"✅ TSFresh features saved for patient {patient_id}.")

    except Exception as e:
        print(f"❌ ERROR extracting TSFresh features for patient {patient_id}: {e}")
        print("🛠 Investigate segment data before processing again.")

# ==========================
# 🔹 Main Execution
# ==========================
if __name__ == '__main__':
    print("🚀 Starting feature extraction...")

    for _, row in metadata_df.iterrows():
        process_patient(row["PatientID"], row["Label"], row["FilePath"])

    print("\n✅ Feature extraction complete!")
