"""
MOMENT Foundation Model Wrapper for PDMBench.

MOMENT is a family of open-source foundation models for general-purpose time series analysis.
This wrapper adapts it for classification tasks.

Paper: https://arxiv.org/abs/2402.03885
GitHub: https://github.com/moment-timeseries-foundation-model/moment
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


# Hidden dimensions for MOMENT models (based on T5 encoder)
MOMENT_HIDDEN_DIMS = {
    'small': 768,   # MOMENT-1-small uses 768
    'base': 768,    # MOMENT-1-base uses 768
    'large': 1024,  # MOMENT-1-large uses 1024
}


class Model(nn.Module):
    """
    MOMENT Foundation Model wrapper with classification head.

    Uses MOMENT embeddings as features for downstream classification.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, 'pred_len', 0)
        self.num_class = getattr(configs, 'num_class', 2)
        self.enc_in = configs.enc_in

        self.moment_size = getattr(configs, 'moment_size', 'base')
        self.freeze_backbone = getattr(configs, 'freeze_backbone', False)
        self.hidden_dim = MOMENT_HIDDEN_DIMS.get(self.moment_size, 768)

        # Load MOMENT model - store in a list to prevent PyTorch from registering it as a submodule
        # This prevents it from being moved to GPU when model.to(device) is called
        self._moment_cpu = [self._load_moment()]

        # Freeze backbone if requested
        if self.freeze_backbone and self._moment_cpu[0] is not None:
            for param in self._moment_cpu[0].parameters():
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

        # Fallback encoder
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

    def _load_moment(self):
        """Load MOMENT model."""
        try:
            from momentfm import MOMENTPipeline
            print(f"Loading MOMENT foundation model (size: {self.moment_size})")

            model = MOMENTPipeline.from_pretrained(
                f"AutonLab/MOMENT-1-{self.moment_size}",
                model_kwargs={'task_name': 'embedding'}
            )
            model.init()
            # Keep MOMENT on CPU to avoid CUDA CUBLAS errors
            model = model.cpu()
            print("MOMENT model loaded successfully (on CPU)")
            return model

        except ImportError:
            print("WARNING: momentfm not installed. Using fallback encoder.")
            print("Install with: pip install momentfm")
            return None
        except Exception as e:
            print(f"WARNING: Failed to load MOMENT model: {e}")
            print("Using fallback encoder.")
            return None

    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from MOMENT model."""
        batch_size, seq_len, n_features = x.shape
        device = x.device
        moment = self._moment_cpu[0]

        if moment is not None:
            try:
                # Clear CUDA state before CPU operations to avoid stale CUDA context issues
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                # MOMENT expects (batch, n_channels, seq_len)
                # Keep everything on CPU to avoid CUDA CUBLAS errors
                x_cpu = x.detach().cpu()
                x_transposed = x_cpu.transpose(1, 2)

                # Ensure MOMENT stays on CPU
                moment = moment.cpu()
                self._moment_cpu[0] = moment

                with torch.no_grad():
                    # MOMENT expects keyword arguments: x_enc, input_mask
                    output = moment(x_enc=x_transposed)
                    if hasattr(output, 'embeddings'):
                        embeddings = output.embeddings
                    else:
                        embeddings = output

                # Ensure correct shape: (batch, n_features, hidden_dim)
                if embeddings.dim() == 2:
                    # embeddings is (batch, hidden_dim)
                    embeddings = embeddings.unsqueeze(1).expand(-1, n_features, -1)
                elif embeddings.dim() == 3 and embeddings.shape[1] != n_features:
                    # embeddings might be (batch, seq, hidden) - pool over sequence
                    embeddings = embeddings.mean(dim=1).unsqueeze(1).expand(-1, n_features, -1)

                return embeddings.to(device)

            except Exception as e:
                print(f"Warning: MOMENT embedding failed: {e}, using fallback")
                import traceback
                traceback.print_exc()

        # Fallback
        encoded = self.fallback_encoder(x)
        return encoded.mean(dim=1).unsqueeze(1).expand(-1, n_features, -1)

    def classification(self, x_enc: torch.Tensor, x_mark_enc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Classification using MOMENT embeddings."""
        batch_size = x_enc.shape[0]

        # Normalize input
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # Get embeddings
        embeddings = self._get_embeddings(x_enc)

        # Flatten and classify
        flat_embeddings = embeddings.reshape(batch_size, -1)
        logits = self.classifier(flat_embeddings)

        return logits

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Forward pass."""
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        else:
            return self.classification(x_enc, x_mark_enc)
