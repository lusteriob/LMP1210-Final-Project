import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# === Paths ===
CWT_IMAGE_DIR = "features_all/cwt_all_images"
BIN_META_PATH = "all_processed_data/bin_metadata.csv"
SEGMENT_META_PATH = "all_processed_data/filtered_metadata_all_right_PD_HC.json"
OUTPUT_DIR = "features_all/cwt_numpy_volumes_6_tasks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load metadata ===
bin_metadata = pd.read_csv(BIN_META_PATH)
with open(SEGMENT_META_PATH, "r") as f:
    segment_template = json.load(f)["tasks"]

# === Define tasks to keep ===
tasks_to_keep = {
    "TouchNose",
    "Relaxed1",
    "Entrainment1",
    "CrossArms",
    "Entrainment2",
    "RelaxedTask1"
}

# === Build filtered (task, sensor, axis) list ===
channel_list = []
for task in segment_template:
    if task["name"] in tasks_to_keep:
        channel = (task["name"], task["sensor"], task["axis"])
        if channel not in channel_list:
            channel_list.append(channel)

# Optional sanity check
print(f"✅ Using {len(channel_list)} channels from top 7 tasks.")
print("Included channels:")
for ch in channel_list:
    print("  -", ch)

# === Stack CWT images for each patient ===
image_shape = None  # auto-detect shape

for _, row in tqdm(bin_metadata.iterrows(), total=len(bin_metadata), desc="Stacking CWT Images"):
    patient_id = str(row["PatientID"]).zfill(3)
    label = row["Label"]
    patient_folder = os.path.join(CWT_IMAGE_DIR, patient_id)

    if not os.path.exists(patient_folder):
        continue

    patient_stack = []
    all_found = True

    for task, sensor, axis in channel_list:
        img_path = os.path.join(patient_folder, f"{task}_{axis}_CWT.png")
        if not os.path.exists(img_path):
            print(f"❌ Missing image: {img_path}")
            all_found = False
            break

        img = Image.open(img_path).convert("L")  # Convert to grayscale
        img_arr = np.array(img)

        if image_shape is None:
            image_shape = img_arr.shape
        elif img_arr.shape != image_shape:
            print(f"⚠️ Skipping {patient_id} due to inconsistent image shape: {img_arr.shape}")
            all_found = False
            break

        patient_stack.append(img_arr)

    if not all_found:
        continue

    volume = np.stack(patient_stack, axis=0)  # Shape: (channels, H, W)
    np.save(os.path.join(OUTPUT_DIR, f"{patient_id}_volume.npy"), volume)

    # Save label
    with open(os.path.join(OUTPUT_DIR, f"{patient_id}_label.txt"), "w") as f:
        f.write(str(label))

print("✅ All eligible CWT volumes saved as .npy files!")
