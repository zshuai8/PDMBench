"""
Quick Start Example

This example demonstrates the basic usage of the PDM Framework
for training and evaluating a model on PDMBench datasets.
"""

import sys
sys.path.insert(0, '../..')

from pdm_framework import PDMDataset, Trainer, Evaluator, TrainerConfig
from pdm_framework.models import SimpleLSTM, get_model


def main():
    # 1. Load a dataset by name
    print("Loading Paderborn Bearing dataset...")
    dataset = PDMDataset('Paderborn')
    print(dataset)
    print()

    # 2. Get data loaders
    train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=32)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print()

    # 3. Create a model
    print("Creating LSTM model...")
    model = SimpleLSTM(
        seq_len=dataset.seq_len,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        hidden_size=128,
        num_layers=2
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # 4. Configure training
    config = TrainerConfig(
        optimizer='adamw',
        learning_rate=1e-3,
        scheduler='cosine',
        epochs=10,  # Use more epochs for real training
        patience=5
    )

    # 5. Train the model
    print("Training model...")
    trainer = Trainer(model, train_loader, val_loader, config=config)
    history = trainer.fit()
    print()

    # 6. Evaluate on test set
    print("Evaluating on test set...")
    evaluator = Evaluator(model, test_loader)
    metrics = evaluator.evaluate()
    print(metrics)

    # 7. Error analysis
    print("\nError Analysis:")
    analysis = evaluator.error_analysis()
    print(f"  Misclassified samples: {analysis['num_misclassified']}")
    print(f"  Error rate: {analysis['error_rate']:.4f}")
    print(f"  Avg correct confidence: {analysis['avg_correct_confidence']:.4f}")
    print(f"  Avg incorrect confidence: {analysis['avg_incorrect_confidence']:.4f}")


if __name__ == '__main__':
    main()
