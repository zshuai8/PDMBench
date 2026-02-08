"""
Model Comparison Example

This example shows how to compare multiple models on the same dataset.
"""

import sys
sys.path.insert(0, '../..')

from pdm_framework import PDMDataset, Trainer, TrainerConfig
from pdm_framework.models import SimpleMLP, SimpleLSTM, SimpleCNN, SimpleTransformer
from pdm_framework.evaluators import compare_models, print_comparison_table


def main():
    # Load dataset by name
    print("Loading HUST Bearing dataset...")
    dataset = PDMDataset('HUST')
    print(dataset)
    print()

    train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=32)

    # Define models to compare
    model_configs = {
        'MLP': SimpleMLP(
            seq_len=dataset.seq_len,
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
            hidden_dims=[256, 128]
        ),
        'LSTM': SimpleLSTM(
            seq_len=dataset.seq_len,
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
            hidden_size=128
        ),
        'CNN': SimpleCNN(
            seq_len=dataset.seq_len,
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
            num_filters=[64, 128, 256]
        ),
        'Transformer': SimpleTransformer(
            seq_len=dataset.seq_len,
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
            d_model=64,
            n_heads=4
        )
    }

    # Training configuration
    config = TrainerConfig(
        optimizer='adamw',
        learning_rate=1e-3,
        epochs=20,
        patience=5,
        verbose=False  # Quiet training
    )

    # Train each model
    trained_models = {}
    for name, model in model_configs.items():
        print(f"Training {name}...", end=" ", flush=True)
        trainer = Trainer(model, train_loader, val_loader, config=config)
        history = trainer.fit()
        trained_models[name] = model
        print(f"Done! Best val acc: {max(history['val_acc']):.4f}")

    print()

    # Compare all models
    results = compare_models(trained_models, test_loader)
    print_comparison_table(results)


if __name__ == '__main__':
    main()
