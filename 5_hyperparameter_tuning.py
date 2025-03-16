import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from utils import nested_cv

# Load data
X_boss = np.load("results/X_boss.npy")
X_tsfresh = np.load("results/X_tsfresh.npy")
y = np.load("results/y.npy")

# Define parameter grids
svm_params = {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]}
best_svm = nested_cv(SVC(kernel="rbf"), X_boss, y, svm_params)
print("Best SVM hyperparameters:", best_svm.best_params_)
