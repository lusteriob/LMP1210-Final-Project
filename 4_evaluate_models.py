import numpy as np
import joblib
import torch
from utils import evaluate_model
from ml.cnn_lstm import CNN_LSTM

# Load test data
X_boss = np.load("results/X_boss.npy")
X_tsfresh = np.load("results/X_tsfresh.npy")
y = np.load("results/y.npy")

# Load trained models
svm = joblib.load("results/svm_model.pkl")
cnn_lstm = CNN_LSTM()
cnn_lstm.load_state_dict(torch.load("results/cnn_lstm.pth"))

# Evaluate models
evaluate_model(svm, X_boss, y, "SVM")
evaluate_model(cnn_lstm, X_tsfresh, y, "CNN+LSTM")
