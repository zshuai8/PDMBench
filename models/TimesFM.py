"""
TimesFM Foundation Model Wrapper for PDMBench.

TimesFM is a foundation model for time series forecasting developed by Google.
This wrapper adapts it for classification tasks by adding a classification head.

Paper: https://arxiv.org/abs/2310.10688
GitHub: https://github.com/google-research/timesfm
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

# Workaround for cublasSgemmStridedBatched bug in PyTorch 2.10+cu128
if torch.cuda.is_available():
    torch.backends.cuda.preferred_blas_library('cublaslt')


# TimesFM model dimensions
TIMESFM_MODEL_DIMS = 1280  # Internal model dimension
TIMESFM_HORIZON = 128  # Default forecast horizon


class Model(nn.Module):
    """
    TimesFM Foundation Model wrapper with classification head.

    Uses TimesFM forecast outputs as features for downstream classification.
    Since TimesFM doesn't expose embeddings directly, we use forecast statistics
    as representation.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, 'pred_len', TIMESFM_HORIZON)
        self.num_class = getattr(configs, 'num_class', 2)
        self.enc_in = configs.enc_in

        self.freeze_backbone = getattr(configs, 'freeze_backbone', False)
        # Use forecast length + quantile features as hidden dim per channel
        self.forecast_len = min(self.pred_len if self.pred_len > 0 else TIMESFM_HORIZON, TIMESFM_HORIZON)
        # Features: mean forecast + 10 quantiles statistics (mean, std) = forecast_len + 20
        self.hidden_dim = self.forecast_len + 20

        # Load TimesFM model
        self.timesfm = self._load_timesfm()

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

        # Fallback encoder (used if TimesFM fails to load)
        self.fallback_hidden_dim = 512
        self.fallback_encoder = nn.Sequential(
            nn.Linear(self.enc_in, self.fallback_hidden_dim),
            nn.ReLU(),
            nn.TransformerEncoderLayer(
                d_model=self.fallback_hidden_dim,
                nhead=8,
                dim_feedforward=self.fallback_hidden_dim * 4,
                dropout=configs.dropout,
                batch_first=True
            ),
            nn.TransformerEncoderLayer(
                d_model=self.fallback_hidden_dim,
                nhead=8,
                dim_feedforward=self.fallback_hidden_dim * 4,
                dropout=configs.dropout,
                batch_first=True
            ),
        )

        # Fallback classifier
        self.fallback_classifier = nn.Sequential(
            nn.Linear(self.fallback_hidden_dim * self.enc_in, 512),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(256, self.num_class)
        )

    def _load_timesfm(self):
        """Load TimesFM model using the new API (version 1.3.0+)."""
        try:
            import timesfm
            print("Loading TimesFM foundation model")

            # New API for TimesFM 1.3.0+
            backend = 'gpu' if torch.cuda.is_available() else 'cpu'
            tfm = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend=backend,
                    per_core_batch_size=32,
                    horizon_len=self.forecast_len,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
                )
            )
            print(f"TimesFM model loaded successfully (context_len={tfm.context_len}, horizon_len={tfm.horizon_len})")
            return tfm

        except ImportError:
            print("WARNING: timesfm not installed. Using fallback encoder.")
            print("Install with: pip install timesfm")
            return None
        except Exception as e:
            print(f"WARNING: Failed to load TimesFM model: {e}")
            print("Using fallback encoder.")
            import traceback
            traceback.print_exc()
            return None

    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from TimesFM model.

        Since TimesFM is a forecasting model, we use forecast outputs as features:
        - Mean forecast values
        - Statistics from quantile predictions

        Args:
            x: Input tensor (batch_size, seq_len, n_features)

        Returns:
            Features tensor (batch_size, n_features, hidden_dim)
        """
        batch_size, seq_len, n_features = x.shape
        device = x.device

        if self.timesfm is not None:
            try:
                embeddings_list = []

                for i in range(n_features):
                    channel_data = x[:, :, i].cpu().numpy()

                    with torch.no_grad():
                        # TimesFM forecast returns (mean_forecast, quantile_forecast)
                        # mean_forecast: (batch, horizon_len)
                        # quantile_forecast: (batch, horizon_len, num_quantiles)
                        mean_forecast, quantile_forecast = self.timesfm.forecast(channel_data)

                        # Create feature vector from forecasts
                        # 1. Mean forecast values (horizon_len features)
                        mean_features = torch.tensor(mean_forecast, dtype=torch.float32)

                        # 2. Quantile statistics (mean and std for each timestep across quantiles)
                        quantile_mean = np.mean(quantile_forecast, axis=-1)  # (batch, horizon_len)
                        quantile_std = np.std(quantile_forecast, axis=-1)  # (batch, horizon_len)

                        # 3. Global statistics across horizon
                        global_mean = np.mean(quantile_forecast, axis=(1, 2), keepdims=True)  # (batch, 1, 1)
                        global_std = np.std(quantile_forecast, axis=(1, 2), keepdims=True)
                        per_quantile_mean = np.mean(quantile_forecast, axis=1)  # (batch, num_quantiles)
                        per_quantile_std = np.std(quantile_forecast, axis=1)

                        # Combine features
                        channel_embed = np.concatenate([
                            mean_forecast,  # (batch, horizon_len)
                            global_mean.reshape(-1, 1),  # (batch, 1)
                            global_std.reshape(-1, 1),  # (batch, 1)
                            np.mean(per_quantile_mean, axis=-1, keepdims=True),  # (batch, 1)
                            np.std(per_quantile_mean, axis=-1, keepdims=True),  # (batch, 1)
                            np.mean(quantile_mean, axis=-1, keepdims=True),  # (batch, 1)
                            np.std(quantile_mean, axis=-1, keepdims=True),  # (batch, 1)
                            np.mean(quantile_std, axis=-1, keepdims=True),  # (batch, 1)
                            np.std(quantile_std, axis=-1, keepdims=True),  # (batch, 1)
                            np.min(mean_forecast, axis=-1, keepdims=True),  # (batch, 1)
                            np.max(mean_forecast, axis=-1, keepdims=True),  # (batch, 1)
                            np.median(mean_forecast, axis=-1, keepdims=True),  # (batch, 1)
                            np.percentile(mean_forecast, 25, axis=-1, keepdims=True),  # (batch, 1)
                            np.percentile(mean_forecast, 75, axis=-1, keepdims=True),  # (batch, 1)
                            np.mean(mean_forecast, axis=-1, keepdims=True),  # (batch, 1)
                            np.std(mean_forecast, axis=-1, keepdims=True),  # (batch, 1)
                            np.var(mean_forecast, axis=-1, keepdims=True),  # (batch, 1)
                            np.ptp(mean_forecast, axis=-1, keepdims=True),  # (batch, 1) - range
                        ], axis=-1)

                        # Truncate/pad to hidden_dim
                        if channel_embed.shape[-1] > self.hidden_dim:
                            channel_embed = channel_embed[:, :self.hidden_dim]
                        elif channel_embed.shape[-1] < self.hidden_dim:
                            pad_width = self.hidden_dim - channel_embed.shape[-1]
                            channel_embed = np.pad(channel_embed, ((0, 0), (0, pad_width)), mode='constant')

                    embeddings_list.append(torch.tensor(channel_embed, dtype=torch.float32))

                # Stack: (batch_size, n_features, hidden_dim)
                stacked = torch.stack(embeddings_list, dim=1)
                return stacked.to(device)

            except Exception as e:
                print(f"Warning: TimesFM embedding failed: {e}, using fallback")
                import traceback
                traceback.print_exc()

        # Fallback: use transformer encoder
        encoded = self.fallback_encoder(x)  # (batch, seq_len, fallback_hidden_dim)
        return encoded.mean(dim=1).unsqueeze(1).expand(-1, n_features, -1)

    def classification(self, x_enc: torch.Tensor, x_mark_enc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Classification using TimesFM features."""
        batch_size = x_enc.shape[0]
        n_features = x_enc.shape[2]

        # Normalize input
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # Get embeddings
        embeddings = self._get_embeddings(x_enc)

        # Flatten and classify
        flat_embeddings = embeddings.reshape(batch_size, -1)

        # Use appropriate classifier based on embedding source
        if self.timesfm is not None and embeddings.shape[-1] == self.hidden_dim:
            logits = self.classifier(flat_embeddings)
        else:
            # Fallback classifier for fallback encoder
            logits = self.fallback_classifier(flat_embeddings)

        return logits

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Forward pass."""
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        else:
            return self.classification(x_enc, x_mark_enc)
