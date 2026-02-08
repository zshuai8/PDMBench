"""
PDM Framework - A Modular Framework for Predictive Maintenance

This framework provides clean abstractions for working with PDM problems:
- Datasets: Easy access to PDMBench datasets
- Models: Base classes and pre-built models
- Trainers: Flexible training pipelines
- Evaluators: Comprehensive evaluation metrics

Example:
    from pdm_framework import PDMDataset, Trainer, Evaluator
    from pdm_framework.models import TimesNet

    # Load dataset by name
    dataset = PDMDataset('Paderborn')
    train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=32)

    # Create model
    model = TimesNet(
        seq_len=dataset.seq_len,
        num_classes=dataset.num_classes,
        enc_in=dataset.num_features
    )

    # Train
    trainer = Trainer(model, train_loader, val_loader)
    trainer.fit(epochs=50)

    # Evaluate
    evaluator = Evaluator(model, test_loader)
    results = evaluator.evaluate()
    print(results)
"""

from .datasets import PDMDataset, get_dataset_info, list_datasets
from .trainers import Trainer, TrainerConfig
from .evaluators import Evaluator, EvaluationMetrics, compare_models, print_comparison_table
from .transforms import (
    Compose, Normalize, Standardize, Jitter, Scale,
    TimeWarp, MagnitudeWarp, Mixup, Cutout, FrequencyMask,
    GaussianNoise, RandomCrop, ChannelDropout, RandomFlip
)
from .models import (
    PDMModel, SimpleMLP, SimpleLSTM, SimpleCNN, SimpleTransformer,
    ModelWrapper, get_model, count_parameters, get_model_summary
)

__version__ = "1.0.0"
__all__ = [
    # Datasets
    'PDMDataset',
    'get_dataset_info',
    'list_datasets',
    # Models
    'PDMModel',
    'SimpleMLP',
    'SimpleLSTM',
    'SimpleCNN',
    'SimpleTransformer',
    'ModelWrapper',
    'get_model',
    'count_parameters',
    'get_model_summary',
    # Trainers
    'Trainer',
    'TrainerConfig',
    # Evaluators
    'Evaluator',
    'EvaluationMetrics',
    'compare_models',
    'print_comparison_table',
    # Transforms
    'Compose',
    'Normalize',
    'Standardize',
    'Jitter',
    'Scale',
    'TimeWarp',
    'MagnitudeWarp',
    'Mixup',
    'Cutout',
    'FrequencyMask',
    'GaussianNoise',
    'RandomCrop',
    'ChannelDropout',
    'RandomFlip',
]
