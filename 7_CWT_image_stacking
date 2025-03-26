import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# === Paths ===
CWT_IMAGE_DIR = "features/cwt_images"
BIN_META_PATH = "processed_data/bin_metadata.csv"
OUTPUT_DIR = "features/cwt_numpy"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load metadata ===
bin_metadata = pd.read_csv(BIN_META_PATH)

# === Expected image order ===
TASKS = ["Relaxed1", "RelaxedTask1", "TouchNose"]
AXES = ["X", "Y", "Z"]

# === Loop through patients ===
image_shape = None  # Will auto-detect based on first image

for _, row in tqdm(bin_metadata.iterrows(), total=len(bin_metadata), desc="Stacking CWT Images"):
    patient_id = str(row["PatientID"]).zfill(3)
    label = row["Label"]
    patient_folder = os.path.join(CWT_IMAGE_DIR, patient_id)

    if not os.path.exists(patient_folder):
        continue

    patient_stack = []
    all_found = True

    for task in TASKS:
        for axis in AXES:
            img_path = os.path.join(patient_folder, f"{task}_{axis}_CWT.png")
            if not os.path.exists(img_path):
                all_found = False
                break
            img = Image.open(img_path).convert("L")  # Grayscale
            img_arr = np.array(img)

            if image_shape is None:
                image_shape = img_arr.shape
            elif img_arr.shape != image_shape:
                print(f"Skipping {patient_id} due to inconsistent image shape: {img_arr.shape}")
                all_found = False
                break

            patient_stack.append(img_arr)
        if not all_found:
            break

    if not all_found:
        continue

    # Stack into (9, height, width)
    volume = np.stack(patient_stack, axis=0)
    np.save(os.path.join(OUTPUT_DIR, f"{patient_id}_volume.npy"), volume)

    # Save label alongside
    with open(os.path.join(OUTPUT_DIR, f"{patient_id}_label.txt"), "w") as f:
        f.write(str(label))

print("All eligible CWT volumes saved as .npy files!")
