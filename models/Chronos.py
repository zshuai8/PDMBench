"""
CHRONOS Foundation Model Wrapper for PDMBench.

CHRONOS is a foundation model for time series forecasting developed by Amazon.
This wrapper adapts it for classification tasks by adding a classification head.

Paper: https://arxiv.org/abs/2403.07815
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

# Workaround for cublasSgemmStridedBatched bug in PyTorch 2.10+cu128
if torch.cuda.is_available():
    torch.backends.cuda.preferred_blas_library('cublaslt')

# Size to model path mapping
CHRONOS_MODELS = {
    'tiny': 'amazon/chronos-t5-tiny',
    'mini': 'amazon/chronos-t5-mini',
    'small': 'amazon/chronos-t5-small',
    'base': 'amazon/chronos-t5-base',
    'large': 'amazon/chronos-t5-large',
}

# Hidden dimensions for each model size
CHRONOS_HIDDEN_DIMS = {
    'tiny': 256,
    'mini': 384,
    'small': 512,
    'base': 768,
    'large': 1024,
}


class Model(nn.Module):
    """
    CHRONOS Foundation Model wrapper with classification head.

    Uses CHRONOS embeddings as features for downstream classification.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, 'pred_len', 0)
        self.num_class = getattr(configs, 'num_class', 2)
        self.enc_in = configs.enc_in  # Number of input features/channels

        # CHRONOS configuration
        self.chronos_size = getattr(configs, 'chronos_size', 'small')
        self.freeze_backbone = getattr(configs, 'freeze_backbone', False)
        self.hidden_dim = CHRONOS_HIDDEN_DIMS.get(self.chronos_size, 512)

        # Determine GPU device
        self.gpu_device = torch.device(f'cuda:{getattr(configs, "gpu", 0)}')

        # Load CHRONOS model - store in list to prevent PyTorch from registering
        # as submodule (which would cause device mismatch issues with tokenizer)
        self._chronos_ref = [self._load_chronos()]

        # Freeze backbone if requested
        if self.freeze_backbone and self._chronos_ref[0] is not None:
            if hasattr(self._chronos_ref[0], 'model'):
                for param in self._chronos_ref[0].model.parameters():
                    param.requires_grad = False

        # Classification head
        # Pooled embedding from all channels
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * self.enc_in, 512),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(256, self.num_class)
        )

        # Projection layer if CHRONOS not available (fallback)
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

    def _load_chronos(self):
        """Load CHRONOS model from HuggingFace."""
        try:
            from chronos import ChronosPipeline

            model_path = CHRONOS_MODELS.get(self.chronos_size, CHRONOS_MODELS['small'])
            print(f"Loading CHRONOS model: {model_path}")

            # Load pipeline - tokenizer stays on CPU by design (uses torch.bucketize
            # with CPU boundaries), but the T5 model goes on GPU via device_map.
            # The library handles CPU->GPU transfers internally.
            if torch.cuda.is_available():
                chronos = ChronosPipeline.from_pretrained(
                    model_path,
                    device_map=str(self.gpu_device),
                    torch_dtype=torch.float32,
                )
                print(f"CHRONOS loaded: tokenizer=CPU, T5 model={self.gpu_device}")
            else:
                chronos = ChronosPipeline.from_pretrained(
                    model_path,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                )
                print("CHRONOS model loaded successfully (CPU)")

            return chronos

        except ImportError:
            print("WARNING: chronos-forecasting not installed. Using fallback encoder.")
            print("Install with: pip install chronos-forecasting")
            return None
        except Exception as e:
            print(f"WARNING: Failed to load CHRONOS model: {e}")
            print("Using fallback encoder.")
            return None

    def _get_chronos_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from CHRONOS model.
        Tokenization on CPU, T5 encoder on GPU, all channels batched together.

        Args:
            x: Input tensor (batch_size, seq_len, n_features)

        Returns:
            Embeddings tensor (batch_size, n_features, hidden_dim)
        """
        batch_size, seq_len, n_features = x.shape
        device = x.device

        chronos = self._chronos_ref[0]

        if chronos is None:
            encoded = self.fallback_encoder(x)
            return encoded.mean(dim=1).unsqueeze(1).expand(-1, n_features, -1)

        try:
            # Reshape all channels into one big batch for a single forward pass
            # (batch_size, seq_len, n_features) -> (batch_size * n_features, seq_len)
            x_cpu = x.detach().cpu().float()
            x_flat = x_cpu.permute(0, 2, 1).reshape(batch_size * n_features, seq_len)

            with torch.no_grad():
                # Tokenize on CPU (bucketize needs CPU boundaries)
                token_ids, attention_mask, scale = chronos.tokenizer.context_input_transform(x_flat)

                # Encode on GPU (model loaded there via device_map)
                embeddings = chronos.model.encode(
                    input_ids=token_ids.to(chronos.model.device),
                    attention_mask=attention_mask.to(chronos.model.device),
                )  # (batch_size * n_features, seq_len+1, hidden_dim)

                # Mean pool over time
                pooled = embeddings.mean(dim=1)  # (batch_size * n_features, hidden_dim)

            # Reshape back: (batch_size, n_features, hidden_dim)
            result = pooled.reshape(batch_size, n_features, self.hidden_dim)
            return result.to(device)

        except Exception as e:
            print(f"Warning: CHRONOS embedding failed: {e}, using fallback")
            import traceback
            traceback.print_exc()
            encoded = self.fallback_encoder(x)
            return encoded.mean(dim=1).unsqueeze(1).expand(-1, n_features, -1)

    def classification(self, x_enc: torch.Tensor, x_mark_enc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Classification using CHRONOS embeddings.

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
        embeddings = self._get_chronos_embeddings(x_enc)  # (batch, n_features, hidden_dim)

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
            # For other tasks, return classification by default
            return self.classification(x_enc, x_mark_enc)
