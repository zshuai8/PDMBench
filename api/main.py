"""
PDMBench API Framework

A RESTful API for accessing PDM benchmark datasets and evaluation services.

Features:
- Dataset discovery and metadata
- Data download endpoints
- Model evaluation submission
- Leaderboard access
- API key authentication (optional)

Usage:
    uvicorn api.main:app --reload --port 8000

API Documentation available at:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import os
import json
import io
import hashlib
import time
from datetime import datetime
from enum import Enum

# Initialize FastAPI app
app = FastAPI(
    title="PDMBench API",
    description="""
    ## PDM Benchmark API

    Access predictive maintenance benchmark datasets and evaluation services.

    ### Features:
    - **Datasets**: Browse and download PDM datasets
    - **Evaluation**: Submit predictions for evaluation
    - **Leaderboard**: View model rankings

    ### Authentication:
    Some endpoints require an API key. Include it in the `X-API-Key` header.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATASET_DIR = './dataset/'
RESULTS_DIR = './results/'
API_KEYS_FILE = './api_keys.json'
SUBMISSIONS_DIR = './submissions/'

os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

# Internal mapping: dataset name -> directory index on disk
_NAME_TO_DIR = {
    'Paderborn':                '01',
    'HUST':                     '02',
    'IMS':                      '03',
    'CWRU':                     '04',
    'XJTU':                     '05',
    'MFPT':                     '06',
    'FEMTO':                    '07',
    'MAFAULDA':                 '09',
    'Mendeley':                 '12',
    'Planetary':                '13',
    'Azure':                    '16',
    'Electric Motor Vibrations':'17',
    'Rotor Broken Bar':         '18',
    'Gear Box UoC':             '19',
}
_DIR_TO_NAME = {v: k for k, v in _NAME_TO_DIR.items()}

# Dataset metadata keyed by dataset name
DATASET_INFO = {
    "Paderborn": {
        "name": "Paderborn",
        "description": "Paderborn University Bearing Dataset",
        "features": ["vibration", "temperature", "speed"],
        "sampling_rate": 64000,
        "num_classes": 3,
        "task": "fault_classification"
    },
    "HUST": {
        "name": "HUST",
        "description": "Huazhong University of Science and Technology Bearing Dataset",
        "features": ["vibration", "temperature"],
        "sampling_rate": 51200,
        "num_classes": 4,
        "task": "fault_classification"
    },
    "IMS": {
        "name": "IMS",
        "description": "Intelligent Maintenance Systems Bearing Dataset",
        "features": ["vibration"],
        "sampling_rate": 20480,
        "num_classes": 2,
        "task": "fault_classification"
    },
    "CWRU": {
        "name": "CWRU",
        "description": "Case Western Reserve University Bearing Dataset",
        "features": ["vibration"],
        "sampling_rate": 12000,
        "num_classes": 10,
        "task": "fault_classification"
    },
    "XJTU": {
        "name": "XJTU",
        "description": "Xi'an Jiaotong University Bearing Dataset",
        "features": ["vibration"],
        "sampling_rate": 25600,
        "num_classes": 5,
        "task": "fault_classification"
    },
    "MFPT": {
        "name": "MFPT",
        "description": "Machinery Failure Prevention Technology Bearing Dataset",
        "features": ["vibration"],
        "sampling_rate": 48828,
        "num_classes": 3,
        "task": "fault_classification"
    },
    "FEMTO": {
        "name": "FEMTO",
        "description": "FEMTO-ST Institute Bearing Dataset",
        "features": ["vibration_h", "vibration_v"],
        "sampling_rate": 25600,
        "num_classes": 3,
        "task": "rul_prediction"
    },
    "MAFAULDA": {
        "name": "MAFAULDA",
        "description": "Machinery Fault Database",
        "features": ["vibration", "microphone", "tachometer"],
        "sampling_rate": 50000,
        "num_classes": 5,
        "task": "fault_classification"
    },
    "Mendeley": {
        "name": "Mendeley",
        "description": "Mendeley Bearing Dataset",
        "features": ["vibration"],
        "sampling_rate": 9600,
        "num_classes": 4,
        "task": "fault_classification"
    },
    "Planetary": {
        "name": "Planetary",
        "description": "Wind Turbine Planetary Gearbox Dataset",
        "features": ["vibration"],
        "sampling_rate": 48000,
        "num_classes": 4,
        "task": "fault_classification"
    },
    "Azure": {
        "name": "Azure",
        "description": "Microsoft Azure Predictive Maintenance Dataset",
        "features": ["telemetry", "errors", "maintenance", "machines"],
        "sampling_rate": 1,
        "num_classes": 2,
        "task": "rul_prediction"
    },
    "Electric Motor Vibrations": {
        "name": "Electric Motor Vibrations",
        "description": "Electric Motor Fault Dataset",
        "features": ["current", "vibration", "temperature", "speed"],
        "sampling_rate": 42000,
        "num_classes": 4,
        "task": "fault_classification"
    },
    "Rotor Broken Bar": {
        "name": "Rotor Broken Bar",
        "description": "IEEE Rotor Broken Bar Dataset",
        "features": ["current"],
        "sampling_rate": 50000,
        "num_classes": 2,
        "task": "fault_classification"
    },
    "Gear Box UoC": {
        "name": "Gear Box UoC",
        "description": "University of Connecticut Gearbox Dataset",
        "features": ["vibration"],
        "sampling_rate": 20000,
        "num_classes": 9,
        "task": "fault_classification"
    },
}


def _resolve_dataset(name_or_id: str) -> str:
    """Resolve a dataset name or legacy index to canonical name. Raises HTTPException if not found."""
    if name_or_id in DATASET_INFO:
        return name_or_id
    if name_or_id in _DIR_TO_NAME:
        return _DIR_TO_NAME[name_or_id]
    raise HTTPException(status_code=404, detail=f"Dataset '{name_or_id}' not found. Available: {list(DATASET_INFO.keys())}")


# =============================================================================
# Pydantic Models
# =============================================================================

class DatasetSplit(str, Enum):
    train = "train"
    val = "val"
    test = "test"


class DataFormat(str, Enum):
    json = "json"
    numpy = "numpy"
    csv = "csv"


class DatasetInfo(BaseModel):
    id: str
    name: str
    description: str
    features: List[str]
    sampling_rate: int
    num_classes: int
    task: str
    available: bool = True


class DatasetListResponse(BaseModel):
    datasets: List[DatasetInfo]
    total: int


class DatasetSample(BaseModel):
    sample_id: int
    features: List[List[float]]
    label: int
    label_name: Optional[str] = None


class DatasetSamplesResponse(BaseModel):
    dataset_id: str
    split: str
    samples: List[DatasetSample]
    total_samples: int
    returned_samples: int


class PredictionSubmission(BaseModel):
    dataset_name: str = Field(..., description="Dataset name (e.g., 'Paderborn', 'CWRU')")
    model_name: str = Field(..., description="Name of your model")
    predictions: List[int] = Field(..., description="List of predicted class labels")
    model_description: Optional[str] = Field(None, description="Brief model description")


class EvaluationResult(BaseModel):
    submission_id: str
    dataset_id: str
    model_name: str
    accuracy: float
    f1_micro: float
    f1_macro: float
    f1_weighted: float
    timestamp: str
    rank: Optional[int] = None


class LeaderboardEntry(BaseModel):
    rank: int
    model_name: str
    accuracy: float
    f1_macro: float
    submission_date: str


class LeaderboardResponse(BaseModel):
    dataset_id: str
    dataset_name: str
    entries: List[LeaderboardEntry]
    last_updated: str


class APIStatus(BaseModel):
    status: str
    version: str
    datasets_available: int
    uptime: str


# =============================================================================
# API Key Authentication (Optional)
# =============================================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def load_api_keys() -> Dict[str, Any]:
    """Load API keys from file."""
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, 'r') as f:
            return json.load(f)
    return {}


async def verify_api_key(api_key: str = Depends(api_key_header)) -> Optional[str]:
    """Verify API key for protected endpoints."""
    if api_key is None:
        return None  # Allow unauthenticated access for some endpoints

    api_keys = load_api_keys()
    if api_key in api_keys:
        return api_keys[api_key].get('user', 'anonymous')
    return None


# =============================================================================
# Helper Functions
# =============================================================================

def get_available_datasets() -> List[str]:
    """Get list of available dataset names."""
    available = []
    for ds_name, ds_dir in _NAME_TO_DIR.items():
        dataset_path = os.path.join(DATASET_DIR, ds_dir)
        if os.path.exists(dataset_path):
            available.append(ds_name)
    return available


def load_dataset(dataset_name: str, split: str) -> tuple:
    """Load dataset split by name."""
    ds_dir = _NAME_TO_DIR.get(dataset_name, dataset_name)
    split_map = {
        'train': 'PdM_TRAIN.npz',
        'val': 'PdM_VAL.npz',
        'test': 'PdM_TEST.npz'
    }

    file_path = os.path.join(DATASET_DIR, ds_dir, split_map[split])

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    loaded = np.load(file_path, allow_pickle=True)
    df = pd.DataFrame(loaded['data'], columns=loaded['columns'])

    features = df['features'].tolist()
    labels = df['label'].values

    # Get class names if available
    class_names = loaded.get('class_names', None)
    if class_names is not None:
        class_names = class_names.tolist() if hasattr(class_names, 'tolist') else list(class_names)

    return features, labels, class_names


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_micro': float(f1_score(y_true, y_pred, average='micro')),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    }


# Track server start time
_start_time = time.time()


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_model=APIStatus)
async def root():
    """API status and information."""
    available = get_available_datasets()
    uptime_seconds = time.time() - _start_time
    hours, remainder = divmod(int(uptime_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    return APIStatus(
        status="online",
        version="1.0.0",
        datasets_available=len(available),
        uptime=f"{hours}h {minutes}m {seconds}s"
    )


@app.get("/datasets", response_model=DatasetListResponse)
async def list_datasets():
    """
    List all available PDM datasets.

    Returns metadata for all datasets including name, description,
    features, sampling rate, and number of classes.
    """
    available = get_available_datasets()

    datasets = []
    for ds_name, info in DATASET_INFO.items():
        datasets.append(DatasetInfo(
            id=ds_name,
            name=info['name'],
            description=info['description'],
            features=info['features'],
            sampling_rate=info['sampling_rate'],
            num_classes=info['num_classes'],
            task=info['task'],
            available=ds_name in available
        ))

    return DatasetListResponse(
        datasets=datasets,
        total=len(datasets)
    )


@app.get("/datasets/{dataset_name}", response_model=DatasetInfo)
async def get_dataset(dataset_name: str):
    """
    Get detailed information about a specific dataset.

    Use dataset names such as 'Paderborn', 'CWRU', 'XJTU', etc.
    """
    ds_name = _resolve_dataset(dataset_name)
    info = DATASET_INFO[ds_name]
    available = get_available_datasets()

    return DatasetInfo(
        id=ds_name,
        name=info['name'],
        description=info['description'],
        features=info['features'],
        sampling_rate=info['sampling_rate'],
        num_classes=info['num_classes'],
        task=info['task'],
        available=ds_name in available
    )


@app.get("/datasets/{dataset_name}/samples")
async def get_dataset_samples(
    dataset_name: str,
    split: DatasetSplit = Query(DatasetSplit.train, description="Dataset split"),
    offset: int = Query(0, ge=0, description="Starting sample index"),
    limit: int = Query(10, ge=1, le=100, description="Number of samples to return"),
    format: DataFormat = Query(DataFormat.json, description="Response format")
):
    """
    Get samples from a dataset.

    Use dataset names such as 'Paderborn', 'CWRU', 'XJTU', etc.
    Use pagination with `offset` and `limit` to retrieve samples.
    Maximum 100 samples per request.
    """
    ds_name = _resolve_dataset(dataset_name)

    try:
        features, labels, class_names = load_dataset(ds_name, split.value)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    total = len(features)

    if offset >= total:
        raise HTTPException(status_code=400, detail=f"Offset {offset} exceeds dataset size {total}")

    end_idx = min(offset + limit, total)

    if format == DataFormat.json:
        samples = []
        for i in range(offset, end_idx):
            feat = features[i]
            if isinstance(feat, np.ndarray):
                feat = feat.tolist()
            elif not isinstance(feat, list):
                feat = list(feat)

            sample = DatasetSample(
                sample_id=i,
                features=feat if isinstance(feat[0], list) else [feat],
                label=int(labels[i]),
                label_name=class_names[int(labels[i])] if class_names else None
            )
            samples.append(sample)

        return DatasetSamplesResponse(
            dataset_id=ds_name,
            split=split.value,
            samples=samples,
            total_samples=total,
            returned_samples=len(samples)
        )

    elif format == DataFormat.numpy:
        # Return as numpy binary
        subset_features = features[offset:end_idx]
        subset_labels = labels[offset:end_idx]

        buffer = io.BytesIO()
        np.savez(buffer, features=subset_features, labels=subset_labels)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={ds_name}_{split.value}_{offset}_{end_idx}.npz"}
        )

    elif format == DataFormat.csv:
        # Return as CSV (flattened features)
        data = []
        for i in range(offset, end_idx):
            feat = features[i]
            if isinstance(feat, np.ndarray):
                feat = feat.flatten().tolist()
            row = {'sample_id': i, 'label': int(labels[i])}
            for j, val in enumerate(feat[:100]):  # Limit features for CSV
                row[f'f{j}'] = val
            data.append(row)

        df = pd.DataFrame(data)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={ds_name}_{split.value}.csv"}
        )


@app.get("/datasets/{dataset_name}/download/{split}")
async def download_dataset(
    dataset_name: str,
    split: DatasetSplit
):
    """
    Download complete dataset split as NPZ file.

    Use dataset names such as 'Paderborn', 'CWRU', 'XJTU', etc.
    """
    ds_name = _resolve_dataset(dataset_name)
    ds_dir = _NAME_TO_DIR[ds_name]

    split_map = {
        'train': 'PdM_TRAIN.npz',
        'val': 'PdM_VAL.npz',
        'test': 'PdM_TEST.npz'
    }

    file_path = os.path.join(DATASET_DIR, ds_dir, split_map[split.value])

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Dataset file not found")

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=f"{ds_name}_{split.value}.npz"
    )


@app.get("/datasets/{dataset_name}/stats")
async def get_dataset_stats(dataset_name: str):
    """
    Get statistics about a dataset.

    Use dataset names such as 'Paderborn', 'CWRU', 'XJTU', etc.
    """
    ds_name = _resolve_dataset(dataset_name)

    stats = {}

    for split in ['train', 'val', 'test']:
        try:
            features, labels, class_names = load_dataset(ds_name, split)

            # Compute statistics
            seq_lengths = [len(f) if hasattr(f, '__len__') else 1 for f in features]

            unique, counts = np.unique(labels, return_counts=True)
            class_dist = {
                class_names[int(u)] if class_names else f"Class {u}": int(c)
                for u, c in zip(unique, counts)
            }

            stats[split] = {
                'num_samples': len(features),
                'num_classes': len(unique),
                'sequence_length': {
                    'min': int(min(seq_lengths)),
                    'max': int(max(seq_lengths)),
                    'mean': float(np.mean(seq_lengths))
                },
                'class_distribution': class_dist
            }
        except FileNotFoundError:
            stats[split] = None

    return {
        'dataset_name': ds_name,
        'splits': stats
    }


@app.post("/evaluate", response_model=EvaluationResult)
async def submit_predictions(
    submission: PredictionSubmission,
    background_tasks: BackgroundTasks,
    user: Optional[str] = Depends(verify_api_key)
):
    """
    Submit predictions for evaluation.

    Provide your model's predictions on the test set and receive
    evaluation metrics including accuracy and F1 scores.
    Use dataset names such as 'Paderborn', 'CWRU', 'XJTU', etc.
    """
    ds_name = _resolve_dataset(submission.dataset_name)

    # Load test labels
    try:
        _, labels, _ = load_dataset(ds_name, 'test')
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Test set not found for this dataset")

    # Validate predictions
    predictions = np.array(submission.predictions)

    if len(predictions) != len(labels):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(labels)} predictions, got {len(predictions)}"
        )

    # Compute metrics
    metrics = compute_metrics(labels, predictions)

    # Generate submission ID
    submission_id = hashlib.md5(
        f"{ds_name}{submission.model_name}{time.time()}".encode()
    ).hexdigest()[:12]

    timestamp = datetime.now().isoformat()

    # Save submission
    submission_data = {
        'submission_id': submission_id,
        'dataset_name': ds_name,
        'model_name': submission.model_name,
        'model_description': submission.model_description,
        'user': user or 'anonymous',
        'metrics': metrics,
        'timestamp': timestamp,
        'predictions': predictions.tolist()
    }

    submission_file = os.path.join(SUBMISSIONS_DIR, f"{submission_id}.json")
    with open(submission_file, 'w') as f:
        json.dump(submission_data, f, indent=2)

    return EvaluationResult(
        submission_id=submission_id,
        dataset_id=ds_name,
        model_name=submission.model_name,
        accuracy=metrics['accuracy'],
        f1_micro=metrics['f1_micro'],
        f1_macro=metrics['f1_macro'],
        f1_weighted=metrics['f1_weighted'],
        timestamp=timestamp
    )


@app.get("/leaderboard/{dataset_name}", response_model=LeaderboardResponse)
async def get_leaderboard(
    dataset_name: str,
    metric: str = Query("accuracy", description="Metric to rank by"),
    limit: int = Query(20, ge=1, le=100, description="Number of entries")
):
    """
    Get leaderboard for a dataset.

    Returns top models ranked by the specified metric.
    Use dataset names such as 'Paderborn', 'CWRU', 'XJTU', etc.
    """
    ds_name = _resolve_dataset(dataset_name)

    # Load all submissions for this dataset
    entries = []

    for filename in os.listdir(SUBMISSIONS_DIR):
        if filename.endswith('.json'):
            with open(os.path.join(SUBMISSIONS_DIR, filename), 'r') as f:
                submission = json.load(f)

            # Support both old 'dataset_id' and new 'dataset_name' keys in saved submissions
            sub_ds = submission.get('dataset_name', submission.get('dataset_id', ''))
            if sub_ds == ds_name or _DIR_TO_NAME.get(sub_ds) == ds_name:
                entries.append({
                    'model_name': submission['model_name'],
                    'accuracy': submission['metrics']['accuracy'],
                    'f1_macro': submission['metrics']['f1_macro'],
                    'submission_date': submission['timestamp']
                })

    # Sort by metric
    if metric in ['accuracy', 'f1_macro']:
        entries.sort(key=lambda x: x.get(metric, 0), reverse=True)

    # Add ranks
    leaderboard = []
    for i, entry in enumerate(entries[:limit]):
        leaderboard.append(LeaderboardEntry(
            rank=i + 1,
            model_name=entry['model_name'],
            accuracy=entry['accuracy'],
            f1_macro=entry['f1_macro'],
            submission_date=entry['submission_date']
        ))

    return LeaderboardResponse(
        dataset_id=ds_name,
        dataset_name=ds_name,
        entries=leaderboard,
        last_updated=datetime.now().isoformat()
    )


@app.get("/models/baseline/{dataset_name}")
async def get_baseline_results(dataset_name: str):
    """
    Get baseline model results for a dataset.

    Returns pre-computed results from baseline models.
    Use dataset names such as 'Paderborn', 'CWRU', 'XJTU', etc.
    """
    ds_name = _resolve_dataset(dataset_name)
    ds_dir = _NAME_TO_DIR[ds_name]

    # Load results from results directory
    baselines = []

    results_path = os.path.join(RESULTS_DIR, ds_dir)
    if os.path.exists(results_path):
        for model_dir in os.listdir(results_path):
            result_file = os.path.join(results_path, model_dir, 'result_classification.txt')
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    content = f.read()

                metrics = {}
                for line in content.split('\n'):
                    if ':' in line:
                        key, val = line.split(':', 1)
                        try:
                            metrics[key.strip().lower()] = float(val.strip())
                        except:
                            continue

                if metrics:
                    baselines.append({
                        'model': model_dir,
                        'metrics': metrics
                    })

    return {
        'dataset_name': ds_name,
        'baselines': baselines
    }


# =============================================================================
# Python Client Example (included in docs)
# =============================================================================

PYTHON_CLIENT_EXAMPLE = '''
"""
PDMBench Python Client Example

pip install requests numpy

"""
import requests
import numpy as np

BASE_URL = "http://localhost:8000"

# List datasets
response = requests.get(f"{BASE_URL}/datasets")
datasets = response.json()
print(f"Available datasets: {len(datasets['datasets'])}")

# Get dataset info by name
dataset_name = "Paderborn"
response = requests.get(f"{BASE_URL}/datasets/{dataset_name}")
info = response.json()
print(f"Dataset: {info['name']}")

# Get samples
response = requests.get(
    f"{BASE_URL}/datasets/{dataset_name}/samples",
    params={"split": "train", "offset": 0, "limit": 10}
)
samples = response.json()
print(f"Got {samples['returned_samples']} samples")

# Download full dataset
response = requests.get(
    f"{BASE_URL}/datasets/{dataset_name}/download/train",
    stream=True
)
with open("train_data.npz", "wb") as f:
    f.write(response.content)

# Load and use
data = np.load("train_data.npz", allow_pickle=True)
features = data['data']
print(f"Loaded {len(features)} samples")

# Submit predictions
predictions = [0] * 100  # Your model predictions
response = requests.post(
    f"{BASE_URL}/evaluate",
    json={
        "dataset_name": "Paderborn",
        "model_name": "MyModel",
        "predictions": predictions
    }
)
result = response.json()
print(f"Accuracy: {result['accuracy']:.4f}")
'''


@app.get("/client/python")
async def get_python_client():
    """
    Get example Python client code.
    """
    return {"language": "python", "code": PYTHON_CLIENT_EXAMPLE}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
