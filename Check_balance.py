import os
from collections import Counter
import matplotlib.pyplot as plt

# === CONFIG ===
DATA_DIR = "features/cwt_numpy"  # folder where patient volume/label files are stored
SPLIT_FILE = os.path.join(DATA_DIR, "splits", "train.txt")  # change this to "val.txt" or "test.txt" as needed

# === Load Labels ===
with open(SPLIT_FILE, "r") as f:
    patient_ids = [line.strip() for line in f if line.strip()]

labels = []
for pid in patient_ids:
    label_path = os.path.join(DATA_DIR, f"{pid}_label.txt")
    with open(label_path, "r") as f:
        labels.append(int(f.read().strip()))

# === Count & Plot ===
label_counts = Counter(labels)
print("Label Counts:", label_counts)

plt.bar(label_counts.keys(), label_counts.values(), color='skyblue')
plt.xticks([0, 1], ['Healthy', 'PD'])
plt.ylabel("Number of Samples")
plt.title("Class Distribution in Split")
plt.grid(axis='y')
plt.tight_layout()
plt.show()
