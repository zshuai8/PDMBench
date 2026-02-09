# PDMBench: A Comprehensive Benchmark for Predictive Maintenance

PDMBench is a unified benchmark platform for evaluating deep learning models on predictive maintenance (PdM) tasks. It provides standardized access to 14 industrial datasets, 33 models (including 4 foundation models), an interactive web UI, a REST API, and a Python framework for programmatic experimentation.

**Key Features:**

- **14 PdM datasets** spanning bearings, gearboxes, motors, and telemetry
- **29 baseline models** from the time series deep learning literature
- **4 foundation models** — Chronos, MOMENT, TimesFM, Moirai — with freeze/fine-tune support
- **Interactive web UI** (Streamlit) for dataset exploration, model training, and results analysis
- **REST API** (FastAPI) for programmatic dataset access and model evaluation
- **Python framework** (`pdm_framework`) for scripted experiments and custom model integration

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Supported Datasets](#supported-datasets)
3. [Supported Models](#supported-models)
4. [Web Application](#web-application)
5. [REST API](#rest-api)
6. [Python API (`pdm_framework`)](#python-api-pdm_framework)
7. [CLI Usage](#cli-usage)
8. [Acknowledgment](#acknowledgment)
9. [Contact](#contact)

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/zshuai8/PDMBenchmark
cd PDMBench
```

### 2. Install Requirements

We recommend Python 3.8+ and a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Prepare Datasets

Download and unzip the datasets into the `./dataset/` directory. Each dataset resides in a numbered subdirectory (e.g., `./dataset/01/` for Paderborn).

Datasets are available from the [HuggingFace collection](https://huggingface.co/collections/odysseywt/pdmlibrary-682504064fc74fcd889cc17f).

---

## Supported Datasets

PDMBench includes 14 datasets covering fault diagnosis, remaining useful life (RUL) prediction, and early failure detection.

| ID | Name | Description | Sampling Rate (Hz) | Channels | Design Target |
|----|------|-------------|--------------------:|:--------:|---------------|
| 01 | Paderborn | Paderborn University Bearing Dataset | 64,000 | 3 | Fault diagnosis |
| 02 | HUST | Huazhong Univ. of Science and Technology Bearing Dataset | 51,200 | 2 | Fault diagnosis |
| 03 | IMS | Intelligent Maintenance Systems Bearing Dataset | 20,480 | 1 | RUL prediction |
| 04 | CWRU | Case Western Reserve University Bearing Dataset | 12,000 | 1 | Fault diagnosis |
| 05 | XJTU | Xi'an Jiaotong University Bearing Dataset | 25,600 | 1 | RUL prediction & Fault diagnosis |
| 06 | MFPT | Mechanical Fault Prevention Technology Bearing Dataset | 97,600 | 1 | Fault diagnosis |
| 07 | FEMTO | FEMTO-ST Institute Bearing Dataset | 25,600 | 3 | RUL prediction |
| 09 | MAFAULDA | Machinery Fault Database | 51,200 | 4 | Fault diagnosis |
| 12 | Mendeley | Mendeley Bearing Dataset | 9,600 | 1 | Fault diagnosis |
| 13 | Planetary | Planetary Gearbox Dataset | 48,000 | 1 | Fault diagnosis |
| 16 | Azure | Microsoft Azure Predictive Maintenance Dataset | 1 | 4 | RUL prediction |
| 17 | Electric Motor Vibrations | Electric Motor Fault Dataset | 42,000 | 4 | Fault diagnosis |
| 18 | Rotor Broken Bar | Rotor Broken Bar Dataset | 50,000 | 1 | Fault diagnosis |
| 19 | Gear Box UoC | University of Connecticut Gearbox Dataset | 20,000 | 1 | Fault diagnosis |

---

## Supported Models

### Baseline Models (29)

All baseline models are adapted from the time series deep learning literature:

| Category | Models |
|----------|--------|
| Transformer variants | Transformer, PatchTST, iTransformer, Autoformer, Informer, FEDformer, Crossformer, ETSformer, Reformer, TimeXer |
| Advanced architectures | TimesNet, TimeMixer, TSMixer, FreTS, SegRNN, Pyraformer, MICN, FiLM, Koopa, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, WPMixer, MultiPatchFormer |
| Simple / Linear | DLinear, MLP, LightTS, TiDE |

### Foundation Models (4)

PDMBench supports four pre-trained foundation models for time series, available in the web UI and CLI:

| Model | Source | Available Sizes | Default | Description |
|-------|--------|-----------------|---------|-------------|
| **Chronos** | Amazon | tiny, mini, small, base, large | small | T5-based time series foundation model |
| **MOMENT** | CMU | small, base, large | base | Open-source time series foundation model |
| **TimesFM** | Google | 200m | 200m | Decoder-only time series foundation model |
| **Moirai** | Salesforce | small, base, large | small | Universal time series forecasting model |

**Foundation model configuration options:**

- **Model size** — select the pre-trained checkpoint size
- **Freeze backbone** — when enabled (default), freezes the pre-trained backbone and only trains a linear classification head (linear probing); when disabled, fine-tunes the entire model
- **Segment length** — input segment length passed to the foundation model (options: 24, 48, 96, 192)

---

## Web Application

PDMBench includes a Streamlit-based web application for interactive experimentation.

### Launch

```bash
streamlit run app_new.py
```

### Pages

The application has four main pages:

#### 1. Dataset Explorer & Processing

Five tabs for exploring and preprocessing datasets:

| Tab | Features |
|-----|----------|
| **Data Processing** | Normalization (standardization, minmax, robust, per-sample), smoothing (moving average, Savitzky-Golay, exponential), detrending, filtering (lowpass/highpass/bandpass/bandstop), outlier handling, resampling, sequence length configuration (64-4096). Live preview of raw vs. processed signals in time and frequency domains. |
| **Overview** | Sample counts, class distribution pie charts, imbalance ratio. |
| **Signal Viewer** | Per-sample time-series visualization, multi-feature selection, time-domain statistics (mean, std, RMS, peak, crest factor, kurtosis), FFT, spectral centroid, power spectral density. |
| **Distributions** | Per-feature value distributions by class, feature statistics table. |
| **Embeddings** | Dimensionality reduction (PCA, t-SNE, UMAP) with 2D scatter plots colored by class. |

#### 2. Model Training

Three tabs:

- **Training Configuration** — Select dataset, model (baseline or foundation), and task type (classification, early failure prediction, RUL estimation). Configure training parameters (batch size, learning rate, epochs, optimizer, LR scheduler, early stopping), architecture parameters (d_model, d_ff, n_heads, encoder/decoder layers, dropout), and model-specific parameters. Foundation models expose size selection, freeze backbone toggle, and segment length. Start training with real-time progress monitoring.
- **Cross-Condition Evaluation** — Evaluate generalization across operating conditions. Select source/target conditions for datasets with known operating regimes (CWRU, Paderborn, XJTU, IMS, MFPT).
- **Custom Model** — Upload a PyTorch model file (`.py`), auto-detect the model class, validate input/output shapes, and register for use in training.

#### 3. Results Analysis

- Leaderboard ranked by selected metric (accuracy, F1 micro/macro/weighted, ECE, Brier score)
- Bar chart comparisons across models
- Model-vs-dataset performance heatmaps
- Per-dataset breakdown tables

#### 4. Python API

Interactive documentation and code examples for the `pdm_framework` package.

---

## REST API

PDMBench provides a RESTful API built with FastAPI for programmatic access to datasets and evaluation services.

### Launch

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs are available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc` (ReDoc).

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API status, version, uptime, and number of available datasets |
| `GET` | `/datasets` | List all datasets with metadata |
| `GET` | `/datasets/{name}` | Get detailed info for a specific dataset (by name or ID) |
| `GET` | `/datasets/{name}/samples` | Get samples from a dataset split. Query params: `split` (train/val/test), `offset`, `limit` (max 100), `format` (json/numpy/csv) |
| `GET` | `/datasets/{name}/download/{split}` | Download a complete dataset split as an NPZ file |
| `GET` | `/datasets/{name}/stats` | Get dataset statistics (sample counts, class distributions, sequence lengths) |
| `POST` | `/evaluate` | Submit predictions for evaluation. Returns accuracy, F1 micro/macro/weighted |
| `GET` | `/leaderboard/{name}` | Get leaderboard for a dataset. Query params: `metric` (accuracy/f1_macro), `limit` |
| `GET` | `/models/baseline/{name}` | Get pre-computed baseline model results for a dataset |
| `GET` | `/client/python` | Get example Python client code |

### Authentication

Some endpoints support optional API key authentication via the `X-API-Key` header.

### Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# List datasets
response = requests.get(f"{BASE_URL}/datasets")
datasets = response.json()
print(f"Available datasets: {datasets['total']}")

# Get dataset info
response = requests.get(f"{BASE_URL}/datasets/Paderborn")
info = response.json()
print(f"Dataset: {info['name']}, Classes: {info['num_classes']}")

# Get samples
response = requests.get(
    f"{BASE_URL}/datasets/Paderborn/samples",
    params={"split": "train", "offset": 0, "limit": 10}
)
samples = response.json()
print(f"Returned {samples['returned_samples']} of {samples['total_samples']} samples")

# Download full split
response = requests.get(f"{BASE_URL}/datasets/Paderborn/download/train", stream=True)
with open("train_data.npz", "wb") as f:
    f.write(response.content)

# Submit predictions for evaluation
predictions = [0] * 100  # your model's predictions
response = requests.post(
    f"{BASE_URL}/evaluate",
    json={
        "dataset_name": "Paderborn",
        "model_name": "MyModel",
        "predictions": predictions
    }
)
result = response.json()
print(f"Accuracy: {result['accuracy']:.4f}, F1 macro: {result['f1_macro']:.4f}")
```

---

## Python API (`pdm_framework`)

The `pdm_framework` package provides a clean, programmatic interface for working with PDMBench datasets, models, training, and evaluation.

### Quick Start

```python
from pdm_framework import PDMDataset, Trainer, Evaluator, TrainerConfig
from pdm_framework.models import SimpleLSTM

# Load dataset by name
dataset = PDMDataset('Paderborn')
train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=32)

# Create model
model = SimpleLSTM(
    seq_len=dataset.seq_len,
    num_features=dataset.num_features,
    num_classes=dataset.num_classes,
    hidden_size=128,
    num_layers=2
)

# Train
config = TrainerConfig(optimizer='adamw', learning_rate=1e-3, scheduler='cosine', epochs=50, patience=5)
trainer = Trainer(model, train_loader, val_loader, config=config)
history = trainer.fit()

# Evaluate
evaluator = Evaluator(model, test_loader)
metrics = evaluator.evaluate()
print(metrics)
```

### Datasets

#### `list_datasets()`

Returns a list of all available dataset names.

```python
from pdm_framework import list_datasets
print(list_datasets())
# ['Paderborn', 'HUST', 'IMS', 'CWRU', 'XJTU', 'MFPT', 'FEMTO', 'MAFAULDA',
#  'Mendeley', 'Planetary', 'Azure', 'Electric Motor Vibrations', 'Rotor Broken Bar', 'Gear Box UoC']
```

#### `PDMDataset`

```python
PDMDataset(
    dataset_name: str,       # Dataset name (e.g., 'Paderborn', 'CWRU') or legacy ID ('01', '04')
    root_path: str = None,   # Custom path to dataset directory (default: auto-detect)
    seq_len: int = None,     # Truncate sequences to this length (default: full length)
    transform: Callable = None  # Transform applied to training samples
)
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `seq_len` | `int` | Sequence length of the data |
| `num_features` | `int` | Number of input channels/features |
| `num_classes` | `int` | Number of output classes |
| `train_data` | `(ndarray, ndarray)` | Raw training features and labels |
| `val_data` | `(ndarray, ndarray)` | Raw validation features and labels |
| `test_data` | `(ndarray, ndarray)` | Raw test features and labels |

**Methods:**

| Method | Description |
|--------|-------------|
| `get_loaders(batch_size=32, num_workers=4, shuffle_train=True)` | Returns `(train_loader, val_loader, test_loader)` as PyTorch DataLoaders |
| `get_class_distribution()` | Returns class counts per split as a dictionary |

### Models

#### Built-in Models

| Model | Description | Key Parameters |
|-------|-------------|----------------|
| `SimpleMLP` | Flattens input, passes through FC layers | `hidden_dims=[256, 128]`, `dropout=0.1` |
| `SimpleLSTM` | Bidirectional LSTM with last-hidden-state output | `hidden_size=128`, `num_layers=2`, `bidirectional=True` |
| `SimpleCNN` | 1D CNN with BatchNorm and MaxPool | `num_filters=[64, 128, 256]`, `kernel_size=3` |
| `SimpleTransformer` | Transformer encoder with global average pooling | `d_model=64`, `n_heads=4`, `num_layers=2` |

All models accept `(seq_len, num_features, num_classes)` as their first three arguments.

#### `PDMModel` (Base Class)

Inherit from `PDMModel` to create custom models compatible with the framework:

```python
from pdm_framework.models import PDMModel
import torch.nn as nn

class MyModel(PDMModel):
    def __init__(self, seq_len, num_features, num_classes):
        super().__init__(seq_len, num_features, num_classes)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * num_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [batch, seq_len, num_features]
        return self.net(x)
```

`PDMModel` also provides `predict(x)` (returns class indices) and `predict_proba(x)` (returns softmax probabilities).

#### `ModelWrapper`

Adapt external models with different input/output conventions:

```python
from pdm_framework.models import ModelWrapper

wrapped = ModelWrapper(
    model=my_external_model,
    seq_len=512, num_features=3, num_classes=5,
    input_transform=lambda x: x.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
)
```

#### `get_model(name, seq_len, num_features, num_classes, **kwargs)`

Factory function. `name` is one of `'mlp'`, `'lstm'`, `'cnn'`, `'transformer'`.

#### Utility Functions

- `count_parameters(model)` — returns the number of trainable parameters
- `get_model_summary(model)` — returns a formatted string summary of model layers

### Trainer

```python
from pdm_framework import Trainer, TrainerConfig

config = TrainerConfig(
    optimizer='adamw',        # 'adam', 'adamw', 'sgd', 'rmsprop'
    learning_rate=1e-3,
    weight_decay=1e-4,
    scheduler='cosine',       # 'cosine', 'step', 'plateau', 'exponential', 'none'
    epochs=50,
    patience=10,              # early stopping patience
    use_amp=False,            # mixed precision training
    verbose=True
)

trainer = Trainer(model, train_loader, val_loader, config=config)
history = trainer.fit()       # returns dict with 'train_loss', 'train_acc', 'val_loss', 'val_acc'
```

The `Trainer` handles:
- Optimizer and learning rate scheduler creation
- Early stopping with best-model restoration
- Optional mixed precision (AMP)
- Checkpointing via `save_checkpoint(path)` / `load_checkpoint(path)`

### Evaluator

```python
from pdm_framework import Evaluator

evaluator = Evaluator(model, test_loader)
metrics = evaluator.evaluate()  # returns EvaluationMetrics

print(metrics.accuracy)
print(metrics.f1)
print(metrics.confusion_matrix)
print(metrics.auc_roc)
print(metrics.summary())         # formatted string
print(metrics.to_dict())         # dictionary
```

**`EvaluationMetrics` fields:** `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`, `per_class_precision`, `per_class_recall`, `per_class_f1`, `predictions`, `true_labels`, `probabilities`, `auc_roc`.

**Additional evaluator methods:**

| Method | Description |
|--------|-------------|
| `error_analysis()` | Returns misclassified sample indices, top confusion pairs, confidence statistics |
| `get_worst_predictions(n=10)` | Returns the n most confidently wrong predictions |
| `class_performance_summary()` | Per-class accuracy, precision, recall, F1 |

#### Comparing Models

```python
from pdm_framework.evaluators import compare_models, print_comparison_table

results = compare_models(
    {'MLP': mlp_model, 'LSTM': lstm_model, 'CNN': cnn_model},
    test_loader
)
print_comparison_table(results)
```

### Transforms

Data augmentation transforms operate on individual samples (torch tensors). Use `Compose` to chain them.

```python
from pdm_framework.transforms import Compose, Normalize, Jitter, Scale, TimeWarp

augmentation = Compose([
    Normalize(),
    Jitter(sigma=0.03),
    Scale(sigma=0.1),
    TimeWarp(sigma=0.2),
])

dataset = PDMDataset('Paderborn', transform=augmentation)
```

Transforms are applied only to training samples (not validation/test).

**Available transforms:**

| Transform | Description | Key Parameters |
|-----------|-------------|----------------|
| `Normalize` | Min-max normalization to [0, 1] | `min_val=0.0`, `max_val=1.0` |
| `Standardize` | Zero mean, unit variance | `mean=None`, `std=None` |
| `Jitter` | Add Gaussian noise | `sigma=0.03` |
| `Scale` | Random magnitude scaling | `sigma=0.1` |
| `TimeWarp` | Smooth temporal warping | `sigma=0.2`, `num_knots=4` |
| `MagnitudeWarp` | Smooth magnitude warping | `sigma=0.2`, `num_knots=4` |
| `Cutout` | Zero out random segments | `n_holes=1`, `length_ratio=0.1` |
| `FrequencyMask` | Mask random frequency bands | `max_mask_ratio=0.2`, `n_masks=1` |
| `GaussianNoise` | Add noise at specified SNR | `snr_db=20.0` |
| `RandomCrop` | Crop and resize to original length | `crop_ratio_range=(0.8, 1.0)` |
| `ChannelDropout` | Randomly drop channels | `dropout_prob=0.1` |
| `RandomFlip` | Flip signal in time or value | `p=0.5`, `axis='time'` |
| `Mixup` | Mix two samples (batch-level) | `alpha=0.2` |
| `Compose` | Chain multiple transforms | `transforms=[...]` |

---

## CLI Usage

Train and evaluate models directly from the command line:

```bash
python run.py \
  --task_name classification \
  --model TimesNet \
  --data Paderborn \
  --is_training 1
```

Modify `--model`, `--data`, and other hyperparameters as needed. All 33 models (baseline + foundation) are available via the CLI.

---

## Acknowledgment

Our benchmark framework is built upon and extends the [Time Series Library (TSLib)](https://github.com/thuml/Time-Series-Library), an open-source project providing a unified codebase for evaluating deep learning models on time series tasks. We greatly appreciate the efforts of the TSLib team.

If you find our benchmark useful, please also consider citing TSLib:

```bibtex
@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Haixu Wu and Tengge Hu and Yong Liu and Hang Zhou and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Learning Representations},
  year={2023},
}
```

---

## Contact

For questions, please open an [issue](https://github.com/zshuai8/PDMBenchmark) or reach out via [email](mailto:zshuai8@vt.edu).
