"""
XGBoost Model Wrapper for Time Series Classification.

Uses a neural network that mimics gradient boosting behavior for compatibility
with the PyTorch training pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class GradientBoostingBlock(nn.Module):
    """A single gradient boosting-like block."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.tree_like = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )

        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection like boosting
        return x + self.scale * self.tree_like(x)


class Model(nn.Module):
    """
    XGBoost-inspired model for time series classification.

    Uses multiple gradient boosting-like blocks with residual connections.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.num_class = getattr(configs, 'num_class', 2)
        self.dropout = getattr(configs, 'dropout', 0.1)
        self.n_estimators = getattr(configs, 'e_layers', 3)  # Number of boosting rounds

        # Feature extraction
        self.n_stats = 6  # mean, std, min, max, median approximation, range
        subsample_factor = max(1, self.seq_len // 16)
        self.subsampled_len = (self.seq_len + subsample_factor - 1) // subsample_factor
        self.subsample_factor = subsample_factor

        feature_dim = self.enc_in * self.n_stats + self.subsampled_len * self.enc_in

        # Initial projection
        hidden_dim = 256
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        # Gradient boosting blocks
        self.boosting_blocks = nn.ModuleList([
            GradientBoostingBlock(hidden_dim, hidden_dim * 2, self.dropout)
            for _ in range(self.n_estimators)
        ])

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.num_class)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract statistical and temporal features from time series.

        Args:
            x: Input tensor (batch_size, seq_len, enc_in)

        Returns:
            Feature tensor (batch_size, feature_dim)
        """
        batch_size = x.shape[0]

        # Statistical features per channel
        mean_feat = x.mean(dim=1)
        std_feat = x.std(dim=1)
        min_feat = x.min(dim=1)[0]
        max_feat = x.max(dim=1)[0]
        range_feat = max_feat - min_feat

        # Median approximation (using middle value)
        mid_idx = x.shape[1] // 2
        median_approx = x[:, mid_idx, :]

        # Subsample temporal features
        subsampled = x[:, ::self.subsample_factor, :]
        subsampled_flat = subsampled.reshape(batch_size, -1)

        # Concatenate all features
        features = torch.cat([
            mean_feat, std_feat, min_feat, max_feat, range_feat, median_approx,
            subsampled_flat
        ], dim=1)

        return features

    def classification(self, x_enc: torch.Tensor, x_mark_enc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Classification forward pass.

        Args:
            x_enc: Input tensor (batch_size, seq_len, enc_in)
            x_mark_enc: Time marks (unused)

        Returns:
            Class logits (batch_size, num_class)
        """
        # Normalize input
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # Extract features
        features = self.extract_features(x_enc)

        # Initial projection
        hidden = self.input_proj(features)

        # Apply boosting blocks sequentially
        for block in self.boosting_blocks:
            hidden = block(hidden)

        # Classification
        logits = self.classifier(hidden)

        return logits

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Forward pass with task routing."""
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        else:
            return self.classification(x_enc, x_mark_enc)
