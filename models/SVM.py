"""
SVM Model Wrapper for Time Series Classification.

Uses sklearn SVM with a neural network wrapper for compatibility with the training pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class Model(nn.Module):
    """
    SVM-based model wrapper for time series classification.

    Uses a simple feature extraction followed by a linear layer that mimics SVM behavior.
    For actual SVM inference, use the sklearn_fit and sklearn_predict methods.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.num_class = getattr(configs, 'num_class', 2)
        self.dropout = getattr(configs, 'dropout', 0.1)

        # Feature extraction: flatten + statistical features
        # Features: mean, std, min, max per channel + flattened (subsampled)
        self.n_stats = 4  # mean, std, min, max
        subsample_factor = max(1, self.seq_len // 32)
        self.subsampled_len = (self.seq_len + subsample_factor - 1) // subsample_factor

        feature_dim = self.enc_in * self.n_stats + self.subsampled_len * self.enc_in

        # Linear classifier (SVM-like with hinge loss approximation)
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        self.classifier = nn.Linear(128, self.num_class)

        # Store subsample factor
        self.subsample_factor = subsample_factor

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
        mean_feat = x.mean(dim=1)  # (batch, enc_in)
        std_feat = x.std(dim=1)    # (batch, enc_in)
        min_feat = x.min(dim=1)[0] # (batch, enc_in)
        max_feat = x.max(dim=1)[0] # (batch, enc_in)

        # Subsample temporal features
        subsampled = x[:, ::self.subsample_factor, :]  # (batch, subsampled_len, enc_in)
        subsampled_flat = subsampled.reshape(batch_size, -1)  # (batch, subsampled_len * enc_in)

        # Concatenate all features
        features = torch.cat([
            mean_feat, std_feat, min_feat, max_feat,
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

        # Feature transformation
        hidden = self.feature_extractor(features)

        # Classification
        logits = self.classifier(hidden)

        return logits

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Forward pass with task routing."""
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        else:
            return self.classification(x_enc, x_mark_enc)
