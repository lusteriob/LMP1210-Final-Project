import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from ml.cnn_lstm import CNN_LSTM

def get_right_wrist_data():
    # Load .bin files and extract right-wrist data
    # (Implement based on your file format)
    pass

def train_svm(X, y):
    svm = SVC(kernel="rbf", C=1.0, gamma="scale")
    svm.fit(X, y)
    return svm

def train_cnn_lstm(X, y):
    model = CNN_LSTM()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Convert X & y to PyTorch tensors
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.long)

    # Train CNN+LSTM model
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_train.unsqueeze(1))
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    return model

def evaluate_model(model, X, y, name):
    y_pred = model.predict(X) if hasattr(model, "predict") else model(torch.tensor(X, dtype=torch.float32)).argmax(dim=1).numpy()
    acc = balanced_accuracy_score(y, y_pred)
    print(f"{name} Balanced Accuracy: {acc:.4f}")

from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np

def nested_cv(model, X, y, param_grid, outer_splits=5, inner_splits=5):
    """
    Perform nested cross-validation to tune hyperparameters while avoiding overfitting.
    
    Parameters:
        model (sklearn-compatible model): The classifier to optimize.
        X (numpy array): Feature matrix.
        y (numpy array): Target labels.
        param_grid (dict): Hyperparameter grid for GridSearchCV.
        outer_splits (int): Number of outer CV folds.
        inner_splits (int): Number of inner CV folds.
    
    Returns:
        best_model (trained model): The model with the best hyperparameters.
    """

    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=42)

    best_models = []
    best_scores = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Grid search on inner folds
        grid_search = GridSearchCV(model, param_grid, cv=inner_cv, scoring="balanced_accuracy")
        grid_search.fit(X_train, y_train)

        # Store best model
        best_models.append(grid_search.best_estimator_)
        best_scores.append(grid_search.best_score_)

        print(f"Best parameters: {grid_search.best_params_} | Balanced Accuracy: {grid_search.best_score_:.4f}")

    # Return the best-performing model
    best_idx = np.argmax(best_scores)
    best_model = best_models[best_idx]

    return best_model
