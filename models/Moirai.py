"""
MOIRAI Foundation Model Wrapper for PDMBench.

MOIRAI is a Masked Encoder-based Universal Time Series Forecasting Transformer
developed by Salesforce Research.

Paper: https://arxiv.org/abs/2402.02592
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

# Size to model configuration mapping
MOIRAI_MODELS = {
    'small': 'salesforce/moirai-1.0-R-small',
    'base': 'salesforce/moirai-1.0-R-base',
    'large': 'salesforce/moirai-1.0-R-large',
}

# Hidden dimensions for each model size
MOIRAI_HIDDEN_DIMS = {
    'small': 384,
    'base': 768,
    'large': 1024,
}


class Model(nn.Module):
    """
    MOIRAI Foundation Model wrapper with classification head.

    Uses MOIRAI embeddings as features for downstream classification.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, 'pred_len', 0)
        self.num_class = getattr(configs, 'num_class', 2)
        self.enc_in = configs.enc_in  # Number of input features/channels

        # MOIRAI configuration
        self.moirai_size = getattr(configs, 'moirai_size', 'small')
        self.freeze_backbone = getattr(configs, 'freeze_backbone', False)
        self.hidden_dim = MOIRAI_HIDDEN_DIMS.get(self.moirai_size, 384)

        # Load MOIRAI model
        self.moirai = self._load_moirai()

        # Freeze backbone if requested
        if self.freeze_backbone and self.moirai is not None:
            for param in self.moirai.parameters():
                param.requires_grad = False

        # Classification head
        # Process embeddings from all channels
        self.channel_aggregator = nn.Sequential(
            nn.Linear(self.enc_in, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(256, self.num_class)
        )

        # Fallback encoder if MOIRAI not available
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

    def _load_moirai(self):
        """Load MOIRAI model from HuggingFace."""
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

            model_path = MOIRAI_MODELS.get(self.moirai_size, MOIRAI_MODELS['small'])
            print(f"Loading MOIRAI model: {model_path}")

            # Load MOIRAI module
            moirai = MoiraiModule.from_pretrained(model_path)
            print("MOIRAI model loaded successfully")
            return moirai

        except ImportError:
            print("WARNING: uni2ts not installed. Using fallback encoder.")
            print("Install with: pip install uni2ts")
            return None
        except Exception as e:
            print(f"WARNING: Failed to load MOIRAI model: {e}")
            print("Using fallback encoder.")
            return None

    def _get_moirai_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from MOIRAI model.

        Args:
            x: Input tensor (batch_size, seq_len, n_features)

        Returns:
            Embeddings tensor (batch_size, hidden_dim)
        """
        batch_size, seq_len, n_features = x.shape
        device = x.device

        if self.moirai is None:
            # Use fallback encoder
            encoded = self.fallback_encoder(x)  # (batch, seq_len, hidden_dim)
            # Mean pooling over time
            return encoded.mean(dim=1)  # (batch, hidden_dim)

        try:
            # Move MOIRAI to same device if needed
            self.moirai = self.moirai.to(device)

            # MOIRAI processes each channel, we need to aggregate
            all_embeddings = []

            with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
                # Process each channel
                for i in range(n_features):
                    channel_data = x[:, :, i:i+1]  # (batch, seq_len, 1)

                    # MOIRAI expects specific input format
                    # Prepare input for MOIRAI
                    if hasattr(self.moirai, 'encode'):
                        # Use encode method if available
                        channel_embed = self.moirai.encode(channel_data)
                    elif hasattr(self.moirai, 'encoder'):
                        # Access encoder directly
                        channel_embed = self.moirai.encoder(channel_data)
                        if hasattr(channel_embed, 'last_hidden_state'):
                            channel_embed = channel_embed.last_hidden_state
                    else:
                        # Use full forward pass
                        channel_embed = self.moirai(channel_data)

                    # Pool over time dimension if needed
                    if channel_embed.dim() == 3:
                        channel_embed = channel_embed.mean(dim=1)

                    all_embeddings.append(channel_embed)

                # Stack and aggregate across channels
                stacked = torch.stack(all_embeddings, dim=1)  # (batch, n_features, hidden_dim)

                # Weighted aggregation
                weights = self.channel_aggregator(
                    torch.ones(batch_size, seq_len, n_features, device=device)
                ).mean(dim=1)  # (batch, 1)
                weights = torch.softmax(weights.expand(-1, n_features), dim=1)  # (batch, n_features)

                # Weighted sum
                embeddings = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden_dim)

                return embeddings

        except Exception as e:
            print(f"Warning: MOIRAI embedding failed: {e}, using fallback")
            encoded = self.fallback_encoder(x)
            return encoded.mean(dim=1)

    def classification(self, x_enc: torch.Tensor, x_mark_enc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Classification using MOIRAI embeddings.

        Args:
            x_enc: Input time series (batch_size, seq_len, n_features)
            x_mark_enc: Time marks (unused)

        Returns:
            Class logits (batch_size, num_class)
        """
        # Normalize input
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # Get embeddings
        embeddings = self._get_moirai_embeddings(x_enc)  # (batch, hidden_dim)

        # Classify
        logits = self.classifier(embeddings)

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
            # For other tasks, return classification by default
            return self.classification(x_enc, x_mark_enc)
