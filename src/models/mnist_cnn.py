"""
SimpleCNN model for MNIST classification in federated learning.

Architecture:
  Input (1, 28, 28)
  → Conv(1→32, kernel=5, pad=0) → ReLU → MaxPool(2)   # (32, 12, 12)
  → Conv(32→64, kernel=5, pad=0) → ReLU → MaxPool(2)  # (64, 4, 4)
  → Flatten                                             # (1024,)
  → FC(1024→128) → ReLU
  → FC(128→10)
  → LogSoftmax (via CrossEntropyLoss)

Author: FL Experiment System
Date: 2026
"""

from __future__ import annotations
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple convolutional network for MNIST (grayscale, 28×28).

    Designed for federated learning: exposes get_parameters() and
    set_parameters() methods compatible with the Flower framework.

    Attributes:
        conv1: First convolutional layer (1→32 channels, 5×5 kernel).
        conv2: Second convolutional layer (32→64 channels, 5×5 kernel).
        fc1: First fully-connected layer (1024→128).
        fc2: Output layer (128→10 classes).
    """

    def __init__(self):
        """Initialize layers with default weight initialization."""
        super().__init__()
        # Block 1: (1,28,28) → (32,24,24) → (32,12,12)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        # Block 2: (32,12,12) → (64,8,8) → (64,4,4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        # Classifier
        self.fc1 = nn.Linear(64 * 4 * 4, 128)   # 1024 → 128
        self.fc2 = nn.Linear(128, 10)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (N, 1, 28, 28).

        Returns:
            Logits tensor of shape (N, 10).
        """
        # Block 1
        x = self.pool(F.relu(self.conv1(x)))   # → (N, 32, 12, 12)
        # Block 2
        x = self.pool(F.relu(self.conv2(x)))   # → (N, 64, 4, 4)
        # Flatten
        x = x.view(x.size(0), -1)              # → (N, 1024)
        # FC
        x = F.relu(self.fc1(x))                # → (N, 128)
        x = self.fc2(x)                         # → (N, 10)
        return x

    def get_parameters(self) -> List[np.ndarray]:
        """
        Extract model parameters as a list of NumPy arrays.

        Used by Flower to serialize parameters for communication.

        Returns:
            List of numpy arrays, one per parameter tensor
            (weights and biases in layer order).

        Example:
            >>> model = SimpleCNN()
            >>> params = model.get_parameters()
            >>> len(params)
            8
        """
        return [val.cpu().numpy() for val in self.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """
        Load parameters from a list of NumPy arrays into the model.

        Used by Flower to deserialize aggregated parameters from the server.

        Args:
            parameters: List of numpy arrays matching the model's state_dict
                        order and shapes.

        Raises:
            ValueError: If the number of parameter arrays doesn't match.

        Example:
            >>> model = SimpleCNN()
            >>> params = model.get_parameters()
            >>> model.set_parameters(params)  # Round-trip, no-op
        """
        state_dict = self.state_dict()
        keys = list(state_dict.keys())

        if len(parameters) != len(keys):
            raise ValueError(
                f"Expected {len(keys)} parameter arrays, got {len(parameters)}"
            )

        new_state = {
            key: torch.tensor(param)
            for key, param in zip(keys, parameters)
        }
        self.load_state_dict(new_state, strict=True)
