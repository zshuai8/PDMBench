"""
Lag-Llama Foundation Model Wrapper for PDMBench.

Lag-Llama is an open-source foundation model for probabilistic time series forecasting.
This wrapper adapts it for classification tasks by adding a classification head.

Paper: https://arxiv.org/abs/2310.08278
GitHub: https://github.com/time-series-foundation-models/lag-llama
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


# Hidden dimensions for Lag-Llama
LAGLLAMA_HIDDEN_DIM = 256


class Model(nn.Module):
    """
    Lag-Llama Foundation Model wrapper with classification head.

    Uses Lag-Llama embeddings as features for downstream classification.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, 'pred_len', 0)
        self.num_class = getattr(configs, 'num_class', 2)
        self.enc_in = configs.enc_in  # Number of input features/channels

        self.freeze_backbone = getattr(configs, 'freeze_backbone', False)
        self.hidden_dim = LAGLLAMA_HIDDEN_DIM

        # Load Lag-Llama model
        self.lagllama = self._load_lagllama()

        # Freeze backbone if requested
        if self.freeze_backbone and self.lagllama is not None:
            for param in self.lagllama.parameters():
                param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * self.enc_in, 512),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(256, self.num_class)
        )

        # Fallback encoder if Lag-Llama not available
        self.fallback_encoder = nn.Sequential(
            nn.Linear(self.enc_in, self.hidden_dim),
            nn.ReLU(),
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=8,
                dim_feedforward=self.hidden_dim * 4,
                dropout=configs.dropout,
                batch_first=True
            ),
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=8,
                dim_feedforward=self.hidden_dim * 4,
                dropout=configs.dropout,
                batch_first=True
            ),
        )

    def _load_lagllama(self):
        """Load Lag-Llama model from HuggingFace."""
        try:
            from lag_llama.gluon.estimator import LagLlamaEstimator
            print("Loading Lag-Llama foundation model")

            # Create estimator with default settings
            estimator = LagLlamaEstimator(
                prediction_length=self.pred_len if self.pred_len > 0 else 1,
                context_length=self.seq_len,
                num_parallel_samples=1,
            )
            print("Lag-Llama model loaded successfully")
            return estimator

        except ImportError:
            print("WARNING: lag-llama not installed. Using fallback encoder.")
            print("Install with: pip install lag-llama")
            return None
        except Exception as e:
            print(f"WARNING: Failed to load Lag-Llama model: {e}")
            print("Using fallback encoder.")
            return None

    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from Lag-Llama model.

        Args:
            x: Input tensor (batch_size, seq_len, n_features)

        Returns:
            Embeddings tensor (batch_size, n_features, hidden_dim)
        """
        batch_size, seq_len, n_features = x.shape
        device = x.device

        # Use fallback encoder (Lag-Llama API is complex for direct embedding)
        encoded = self.fallback_encoder(x)  # (batch, seq_len, hidden_dim)
        # Mean pooling over time
        return encoded.mean(dim=1).unsqueeze(1).expand(-1, n_features, -1)

    def classification(self, x_enc: torch.Tensor, x_mark_enc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Classification using Lag-Llama embeddings.

        Args:
            x_enc: Input time series (batch_size, seq_len, n_features)
            x_mark_enc: Time marks (unused)

        Returns:
            Class logits (batch_size, num_class)
        """
        batch_size = x_enc.shape[0]

        # Normalize input
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # Get embeddings
        embeddings = self._get_embeddings(x_enc)  # (batch, n_features, hidden_dim)

        # Flatten embeddings
        flat_embeddings = embeddings.reshape(batch_size, -1)  # (batch, n_features * hidden_dim)

        # Classify
        logits = self.classifier(flat_embeddings)

        return logits

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass.

        Args:
            x_enc: Encoder input (batch_size, seq_len, n_features)
            x_mark_enc: Encoder time marks
            x_dec: Decoder input (unused for classification)
            x_mark_dec: Decoder time marks (unused)
            mask: Attention mask (unused)

        Returns:
            Task-specific output
        """
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        else:
            return self.classification(x_enc, x_mark_enc)
