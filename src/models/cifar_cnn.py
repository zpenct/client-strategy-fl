"""
CNN with BatchNorm for CIFAR-10 classification in federated learning.

Architecture:
  Input (3, 32, 32)
  → Conv(3→32, kernel=3, pad=1) → BN → ReLU → MaxPool(2)   # (32, 16, 16)
  → Conv(32→64, kernel=3, pad=1) → BN → ReLU → MaxPool(2)  # (64, 8, 8)
  → Flatten                                                   # (4096,)
  → FC(4096→256) → ReLU → Dropout(0.5)
  → FC(256→10)

Author: FL Experiment System
Date: 2026
"""

from __future__ import annotations
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFARCNN(nn.Module):
    """
    CNN with batch normalization for CIFAR-10 (RGB, 32×32).

    Includes Dropout for regularization — important when each client
    has limited, skewed data. Exposes Flower-compatible get/set_parameters.

    Attributes:
        conv1: First conv block (3→32 channels, 3×3, padding=1).
        bn1: Batch normalization after conv1.
        conv2: Second conv block (32→64 channels, 3×3, padding=1).
        bn2: Batch normalization after conv2.
        fc1: Dense layer (64*8*8=4096 → 256).
        dropout: Dropout with p=0.5.
        fc2: Output layer (256 → 10).
    """

    def __init__(self, dropout_p: float = 0.5):
        """
        Initialize CIFAR-10 CNN.

        Args:
            dropout_p: Dropout probability (default 0.5).
        """
        super().__init__()

        # Block 1: (3,32,32) → (32,32,32) → (32,16,16)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2: (32,16,16) → (64,16,16) → (64,8,8)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier: 64*8*8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (N, 3, 32, 32).

        Returns:
            Logits tensor of shape (N, 10).
        """
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # → (N, 32, 16, 16)
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # → (N, 64, 8, 8)
        # Flatten
        x = x.view(x.size(0), -1)                        # → (N, 4096)
        # Classifier
        x = F.relu(self.fc1(x))                          # → (N, 256)
        x = self.dropout(x)
        x = self.fc2(x)                                   # → (N, 10)
        return x

    def get_parameters(self) -> List[np.ndarray]:
        """
        Extract model parameters as a list of NumPy arrays.

        Note: BatchNorm running_mean and running_var (non-trainable buffers)
        are included in state_dict but are NOT parameters in the autograd
        sense. They are included here so that BN statistics are also
        federated across clients.

        Returns:
            List of numpy arrays in state_dict order.

        Example:
            >>> model = CIFARCNN()
            >>> params = model.get_parameters()
            >>> len(params)  # conv weights+bias, bn params+buffers, fc weights+bias
            14
        """
        return [val.cpu().numpy() for val in self.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """
        Load parameters from a list of NumPy arrays.

        Args:
            parameters: List of numpy arrays matching state_dict order.

        Raises:
            ValueError: If length of parameters doesn't match state_dict.

        Example:
            >>> model = CIFARCNN()
            >>> params = model.get_parameters()
            >>> model.set_parameters(params)
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
