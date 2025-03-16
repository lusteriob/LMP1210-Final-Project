import numpy as np
from tsfresh import extract_features
from ml.multi_boss import MultiBOSS

# Load right-wrist data
X = np.load("results/X_right.npy")
y = np.load("results/y.npy")

# Apply BOSS feature extraction
boss_extractor = MultiBOSS(window_sizes=[20, 40, 80], window_step=2)
X_boss = boss_extractor.fit_transform(X)
np.save("results/X_boss.npy", X_boss)

# Apply TSFresh feature extraction
X_tsfresh = extract_features(X, column_id="id", column_sort="time")
np.save("results/X_tsfresh.npy", X_tsfresh)

print(f"Saved extracted features: BOSS {X_boss.shape}, TSFresh {X_tsfresh.shape}")
