import numpy as np
import joblib
import torch
from utils import train_svm, train_cnn_lstm

# Load features
X_boss = np.load("results/X_boss.npy")
X_tsfresh = np.load("results/X_tsfresh.npy")
y = np.load("results/y.npy")

# Train SVM on BOSS
svm = train_svm(X_boss, y)
joblib.dump(svm, "results/svm_model.pkl")

# Train CNN+LSTM on TSFresh
cnn_lstm = train_cnn_lstm(X_tsfresh, y)
torch.save(cnn_lstm.state_dict(), "results/cnn_lstm.pth")

print("Saved trained models: SVM & CNN-LSTM")
