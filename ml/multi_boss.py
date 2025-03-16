"""How This Works
BOSS extracts frequency-domain features from movement signals.
Multiple window sizes (20, 40, 80) capture different levels of movement patterns.
The features are concatenated for better classification."""

import numpy as np
from pyts.transformation import BOSS

class MultiBOSS:
    def __init__(self, window_sizes=[20, 40, 80], window_step=2):
        self.window_sizes = window_sizes
        self.window_step = window_step
        self.models = []

    def fit_transform(self, X):
        """Apply BOSS transformation across multiple window sizes."""
        transformed_features = []

        for win_size in self.window_sizes:
            boss = BOSS(word_size=4, n_bins=4, window_size=win_size)
            X_boss = boss.fit_transform(X)
            transformed_features.append(X_boss)

        return np.hstack(transformed_features)  # Combine features from different window sizes
