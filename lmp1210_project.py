import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tsfresh.feature_extraction import extract_features
from pyts.transformation import BOSS
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from skorch.helper import SliceDataset

# Dataset directory
dataset_root = "../data/"

# Step 1: Load and Preprocess Right-Wrist Data
def get_channels(type="movement"):
    """ Generate a list of movement channels based on tasks and sensor types. """
    channels = []
    if type == "movement":
        for task in ["Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2", "StretchHold", "HoldWeight",
                     "DrinkGlas", "CrossArms", "TouchNose", "Entrainment1", "Entrainment2"]:
            for device_location in ["RightWrist"]:  # Exclude LeftWrist
                for sensor in ["Acceleration", "Rotation"]:
                    for axis in ["X", "Y", "Z"]:
                        channel = f"{task}_{sensor}_{device_location}_{axis}"
                        channels.append(channel)
    return channels

def get_dataset(type="movement", mode=None, feature_extraction=None, return_idxs=False):
    """
    Load dataset and filter only right-wrist movement data.
    """
    channels = pd.Series(get_channels(type))  # Get only right-wrist channels
    file_list = pd.read_csv(f"{dataset_root}/file_list.csv")
    y = file_list["label"].values  # Extract labels
    
    print("Loading dataset (Right-Wrist Data Only)")
    
    subfolder = "movement/" if type == "movement" else "questionnaire/"
    
    x = []
    for idx, file_idx in enumerate(file_list["id"]):
        if idx % 100 == 0:
            print(f"Processing {idx + 1} / {len(file_list)}")
        
        # Load binary data
        data = np.fromfile(f"{dataset_root}/{subfolder}{file_idx:03d}_ml.bin", dtype=np.float32).reshape((-1, 976))
        
        # Filter only right-wrist data
        channel_mask = channels.str.contains("RightWrist")
        data = data[:, channel_mask.values]  # Apply mask to select columns
        
        new_channels = channels[channel_mask].values  # Update channel list
        
        # Apply feature extraction if provided
        if feature_extraction is not None:
            data = feature_extraction(data)
            n_feature_per_channel = data.shape[0] // len(new_channels)
            new_channels = np.repeat(new_channels, n_feature_per_channel)
        
        x.append(data)
    
    x = np.stack(x)
    
    if return_idxs:
        idxs = file_list.index.values
        return x, y, new_channels, idxs
    return x, y, new_channels

# Step 2: Compute Sample Weights
def compute_sample_weights(labels):
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return {cls: weight for cls, weight in zip(np.unique(labels), class_weights)}

# Step 3: Feature Extraction (BOSS vs. TSFresh)
def extract_features_boss(data):
    boss = BOSS(word_size=4, n_bins=4, window_size=10)
    return boss.fit_transform(data)

def extract_features_tsfresh(data):
    return extract_features(data, column_id='id', column_sort='time')

# Step 4: Feature Selection (RFE, Lasso, Permutation Importance)
def feature_selection_rfe(model, data, labels):
    rfe = RFE(model, n_features_to_select=10)
    return rfe.fit_transform(data, labels)

def feature_selection_lasso(data, labels):
    lasso = Lasso(alpha=0.01)
    lasso.fit(data, labels)
    selected_features = data.columns[lasso.coef_ != 0]
    return data[selected_features]

# Step 5: Classifier Definitions
class CNN1D(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * input_size, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Step 6: Model Training with Nested CV
def train_model(data, labels, use_tsfresh=True):
    feature_extractor = extract_features_tsfresh if use_tsfresh else extract_features_boss
    extracted_features = feature_extractor(data)
    
    # Define classifiers
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', class_weight='balanced'))
    ])
    
    cnn_clf = NeuralNetClassifier(
        CNN1D(input_size=extracted_features.shape[1]),
        max_epochs=50,
        lr=0.001,
        optimizer=optim.Adam,
        criterion=nn.CrossEntropyLoss,
        batch_size=32,
        train_split=None,
        verbose=0
    )
    
    # Perform nested CV
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    param_grid_svm = {'svm__C': [0.1, 1, 10, 100], 'svm__gamma': ['scale', 'auto']}
    param_grid_cnn = {'lr': [0.001, 0.01], 'max_epochs': [50, 100]}
    
    grid_svm = GridSearchCV(svm, param_grid_svm, cv=inner_cv, scoring='balanced_accuracy')
    grid_cnn = GridSearchCV(cnn_clf, param_grid_cnn, cv=inner_cv, scoring='balanced_accuracy')
    
    for train_idx, test_idx in outer_cv.split(extracted_features, labels):
        X_train, X_test = extracted_features.iloc[train_idx], extracted_features.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
        
        grid_svm.fit(X_train, y_train)
        grid_cnn.fit(X_train.values.astype(np.float32), y_train.values.astype(np.longlong))
        
        print(f'SVM Best Score: {grid_svm.best_score_}')
        print(f'CNN Best Score: {grid_cnn.best_score_}')
