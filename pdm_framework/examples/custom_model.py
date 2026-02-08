"""
Custom Model Example

This example shows how to use the ModelWrapper to integrate
your own custom models with the PDM Framework.
"""

import sys
sys.path.insert(0, '../..')

import torch
import torch.nn as nn
from pdm_framework import PDMDataset, Trainer, Evaluator, TrainerConfig
from pdm_framework.models import ModelWrapper, PDMModel


# Example 1: Simple custom model inheriting from PDMModel
class MyCustomRNN(PDMModel):
    """
    Custom RNN model that inherits from PDMModel.

    This is the recommended approach for new models.
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        hidden_size: int = 64
    ):
        super().__init__(seq_len, num_features, num_classes)

        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, features]
        gru_out, _ = self.gru(x)  # [batch, seq_len, hidden]

        # Attention
        attn_weights = self.attention(gru_out)  # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum
        context = (gru_out * attn_weights).sum(dim=1)  # [batch, hidden]

        return self.classifier(context)


# Example 2: Wrapping an existing model with different interface
class ExternalModel(nn.Module):
    """
    Example of an external model that expects [B, C, T] input format
    instead of [B, T, C].
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # Expects x: [batch, channels, time]
        x = self.conv(x)
        x = x.squeeze(-1)
        return self.fc(x)


def main():
    # Load dataset by name
    print("Loading dataset...")
    dataset = PDMDataset('Paderborn')
    print(dataset)
    print()

    train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=32)

    # Example 1: Use custom model that inherits from PDMModel
    print("=" * 50)
    print("Example 1: Custom PDMModel")
    print("=" * 50)

    custom_model = MyCustomRNN(
        seq_len=dataset.seq_len,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        hidden_size=64
    )

    config = TrainerConfig(epochs=10, patience=3)
    trainer = Trainer(custom_model, train_loader, val_loader, config=config)
    trainer.fit()

    evaluator = Evaluator(custom_model, test_loader)
    metrics = evaluator.evaluate()
    print(f"Test Accuracy: {metrics.accuracy:.4f}")
    print()

    # Example 2: Wrap external model with ModelWrapper
    print("=" * 50)
    print("Example 2: Wrapped External Model")
    print("=" * 50)

    external = ExternalModel(
        in_channels=dataset.num_features,
        num_classes=dataset.num_classes
    )

    # Wrap with input transform to convert [B, T, C] -> [B, C, T]
    wrapped_model = ModelWrapper(
        model=external,
        seq_len=dataset.seq_len,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        input_transform=lambda x: x.permute(0, 2, 1)
    )

    trainer = Trainer(wrapped_model, train_loader, val_loader, config=config)
    trainer.fit()

    evaluator = Evaluator(wrapped_model, test_loader)
    metrics = evaluator.evaluate()
    print(f"Test Accuracy: {metrics.accuracy:.4f}")


if __name__ == '__main__':
    main()
