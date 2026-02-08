"""
PDM Models Module

Base classes and utilities for PDM models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod


class PDMModel(nn.Module, ABC):
    """
    Abstract base class for PDM models.

    All PDM models should inherit from this class and implement
    the forward method.

    The expected input shape is [batch_size, seq_len, num_features].
    The expected output shape is [batch_size, num_classes].
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        **kwargs
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, num_features]

        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        pass

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


class SimpleMLP(PDMModel):
    """
    Simple MLP baseline for time series classification.

    Flattens the input and passes through fully connected layers.
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1
    ):
        super().__init__(seq_len, num_features, num_classes)

        input_dim = seq_len * num_features
        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten
        return self.network(x)


class SimpleLSTM(PDMModel):
    """
    Simple LSTM baseline for time series classification.
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True
    ):
        super().__init__(seq_len, num_features, num_classes)

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        fc_input = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)
        # Use last hidden state
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


class SimpleCNN(PDMModel):
    """
    Simple 1D CNN baseline for time series classification.
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        num_filters: List[int] = [64, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__(seq_len, num_features, num_classes)

        conv_layers = []
        in_channels = num_features

        for out_channels in num_filters:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        # Calculate output size after convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, num_features, seq_len)
            conv_out = self.conv(dummy)
            self.flat_size = conv_out.view(1, -1).size(1)

        self.fc = nn.Linear(self.flat_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SimpleTransformer(PDMModel):
    """
    Simple Transformer baseline for time series classification.
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__(seq_len, num_features, num_classes)

        self.input_projection = nn.Linear(num_features, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, features]
        x = self.input_projection(x)
        x = x + self.pos_encoding[:, :x.size(1), :]

        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)
        x = self.dropout(x)

        return self.fc(x)


class ModelWrapper(PDMModel):
    """
    Wrapper to adapt external models to PDM interface.

    Use this to wrap user-provided models that have different interfaces.

    Example:
        # User's model expects [B, C, T] instead of [B, T, C]
        user_model = MyCustomModel(...)
        wrapped = ModelWrapper(
            user_model,
            seq_len=512,
            num_features=7,
            num_classes=3,
            input_transform=lambda x: x.permute(0, 2, 1)
        )
    """

    def __init__(
        self,
        model: nn.Module,
        seq_len: int,
        num_features: int,
        num_classes: int,
        input_transform=None,
        output_transform=None
    ):
        super().__init__(seq_len, num_features, num_classes)
        self.model = model
        self.input_transform = input_transform
        self.output_transform = output_transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_transform:
            x = self.input_transform(x)

        out = self.model(x)

        if self.output_transform:
            out = self.output_transform(out)

        return out


def get_model(
    model_name: str,
    seq_len: int,
    num_features: int,
    num_classes: int,
    **kwargs
) -> PDMModel:
    """
    Factory function to get a model by name.

    Args:
        model_name: Name of the model ('mlp', 'lstm', 'cnn', 'transformer')
        seq_len: Sequence length
        num_features: Number of input features/channels
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments

    Returns:
        Initialized model
    """
    models = {
        'mlp': SimpleMLP,
        'lstm': SimpleLSTM,
        'cnn': SimpleCNN,
        'transformer': SimpleTransformer
    }

    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    return models[model_name.lower()](
        seq_len=seq_len,
        num_features=num_features,
        num_classes=num_classes,
        **kwargs
    )


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module) -> str:
    """Get a summary of model architecture."""
    lines = [
        "=" * 60,
        "Model Summary",
        "=" * 60,
        f"Total parameters: {count_parameters(model):,}",
        "",
        "Layers:",
        "-" * 60
    ]

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in module.parameters())
            lines.append(f"  {name}: {module.__class__.__name__} ({params:,} params)")

    lines.append("=" * 60)
    return "\n".join(lines)
