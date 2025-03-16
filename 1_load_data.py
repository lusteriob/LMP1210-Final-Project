import numpy as np
import pandas as pd
from utils import get_right_wrist_data

# Load and preprocess
X, y, channel_names = get_right_wrist_data()

# Save to disk for later steps
np.save("results/X_right.npy", X)
np.save("results/y.npy", y)

print(f"Saved right-wrist dataset: {X.shape}, Labels: {y.shape}")
