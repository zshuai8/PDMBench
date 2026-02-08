"""
Data Augmentation Example

This example shows how to use data augmentations with the PDM Framework.
"""

import sys
sys.path.insert(0, '../..')

from pdm_framework import PDMDataset, Trainer, Evaluator, TrainerConfig
from pdm_framework.models import SimpleLSTM
from pdm_framework.transforms import (
    Compose, Jitter, Scale, TimeWarp, MagnitudeWarp,
    Cutout, GaussianNoise, Normalize
)


def main():
    # Define augmentation pipeline
    augmentation = Compose([
        Normalize(),           # Normalize to [0, 1]
        Jitter(sigma=0.03),    # Add small noise
        Scale(sigma=0.1),      # Random scaling
        TimeWarp(sigma=0.2),   # Time warping
    ])

    # Load dataset with augmentation
    print("Loading dataset with augmentation...")
    dataset = PDMDataset('Paderborn', transform=augmentation)
    print(dataset)
    print()

    # Get loaders - augmentation only applied to training
    train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=32)

    # Create model
    model = SimpleLSTM(
        seq_len=dataset.seq_len,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes
    )

    # Train with augmented data
    config = TrainerConfig(epochs=20, patience=5)
    print("Training with augmentation...")
    trainer = Trainer(model, train_loader, val_loader, config=config)
    trainer.fit()

    # Evaluate
    evaluator = Evaluator(model, test_loader)
    metrics = evaluator.evaluate()
    print(f"\nTest Accuracy (with augmentation): {metrics.accuracy:.4f}")
    print()

    # Compare with no augmentation
    print("=" * 50)
    print("Training without augmentation for comparison...")
    print("=" * 50)

    dataset_no_aug = PDMDataset('Paderborn')  # No transform
    train_loader_no_aug, val_loader_no_aug, _ = dataset_no_aug.get_loaders(batch_size=32)

    model_no_aug = SimpleLSTM(
        seq_len=dataset.seq_len,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes
    )

    trainer_no_aug = Trainer(model_no_aug, train_loader_no_aug, val_loader_no_aug, config=config)
    trainer_no_aug.fit()

    evaluator_no_aug = Evaluator(model_no_aug, test_loader)
    metrics_no_aug = evaluator_no_aug.evaluate()
    print(f"\nTest Accuracy (without augmentation): {metrics_no_aug.accuracy:.4f}")

    print("\n" + "=" * 50)
    print("Comparison:")
    print(f"  With augmentation:    {metrics.accuracy:.4f}")
    print(f"  Without augmentation: {metrics_no_aug.accuracy:.4f}")
    print(f"  Difference:           {metrics.accuracy - metrics_no_aug.accuracy:+.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
