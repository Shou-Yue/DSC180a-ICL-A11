import torch
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from torch.utils.data import Dataset


class GaussianMixtureDataset(Dataset):
    """Dataset class for generating Gaussian mixture data"""
    def __init__(self, d: int, N: int, B: int, R: float, is_validation: bool=False, label_flip_p: float =0.0):
        self.d = d
        self.N = N 
        self.B = B
        self.R = R
        self.is_validation = is_validation
        self.label_flip_p = label_flip_p
        assert 0 <= label_flip_p < 0.5, f'label flip probab must be in [0,1/2), got {label_flip_p}'

        if is_validation:
            # use differents eed for validation data
            torch.manual_seed(42)

        # Generate all data at once and store in tensors
        self.context_x, self.context_y, self.target_x, self.target_y = self._generate_data()

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate all data at once and return fixed tensors.

        Returns:
            context_x: Shape (B, N, d)
            context_y: Shape (B, N)
            target_x: Shape (B, d)
            target_y: Shape (B)
        """
        # Generate mean vectors for all tasks - Shape: (B, d)
        mus = torch.randn(self.B, self.d)
        # Normalize and scale - Shape: (B, d)
        mus = mus / torch.norm(mus, dim=1, keepdim=True) * self.R

        # Generate clean labels at once - Shape: (B, N+1)
        y_all = (torch.rand(self.B, self.N + 1) > 0.5).float()


        # Convert to {-1, 1} for signal generation - Shape: (B, N+1)
        y_signal = 2 * y_all - 1

        # Generate noise - Shape: (B, N+1, d)
        z = torch.randn(self.B, self.N + 1, self.d)

        # Broadcast for multiplication:
        # mus[:, None, :] shape: (B, 1, d)
        # y_signal[..., None] shape: (B, N+1, 1)
        # Result shape: (B, N+1, d)
        x = y_signal[..., None] * mus[:, None, :] + z

        # Flip labels with probability label_flip_p
        if self.label_flip_p:
            flip_mask = (torch.rand(self.B, self.N + 1) < self.label_flip_p)
            y_all = torch.where(flip_mask, 1 - y_all, y_all)

        # Split into context and target
        context_x = x[:, :self.N, :]  # (B, N, d)
        target_x = x[:, -1, :]        # (B, d)
        context_y = y_all[:, :self.N]  # (B, N)
        target_y = y_all[:, -1]       # (B)

        return context_x, context_y, target_x, target_y

    def __len__(self) -> int:
        return 1  # Only one batch for full-batch GD

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the entire dataset as one batch"""
        assert idx == 0, "Only one batch supported for full-batch GD"
        return self.context_x, self.context_y, self.target_x, self.target_y
