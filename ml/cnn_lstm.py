"""How This Works
A 1D CNN extracts local features from the right-wrist movement data.
A LSTM (Recurrent Layer) learns long-term dependencies in movement patterns.
A fully connected layer makes the final classification"""

import torch
import torch.nn as nn
import torch.optim as optim

class CNN_LSTM(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, num_classes)  # Binary classification (PD vs HC)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Take last LSTM output
        return x
