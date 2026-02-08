"""
PDM Datasets Module

Provides easy access to PDMBench datasets with clean interfaces.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict, Any, Callable


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

# Reverse mapping: directory index -> dataset name
_DIR_TO_NAME = {v: k for k, v in _NAME_TO_DIR.items()}

# Dataset metadata keyed by name
DATASET_INFO = {
    'Paderborn': {
        'name': 'Paderborn',
        'description': 'Paderborn University Bearing Dataset',
        'domain': 'Bearings',
        'num_classes': 3,
    },
    'HUST': {
        'name': 'HUST',
        'description': 'Huazhong University of Science and Technology Bearing Dataset',
        'domain': 'Bearings',
        'num_classes': 4,
    },
    'IMS': {
        'name': 'IMS',
        'description': 'Intelligent Maintenance Systems Bearing Dataset',
        'domain': 'Bearings',
        'num_classes': 2,
    },
    'CWRU': {
        'name': 'CWRU',
        'description': 'Case Western Reserve University Bearing Dataset',
        'domain': 'Bearings',
        'num_classes': 10,
    },
    'XJTU': {
        'name': 'XJTU',
        'description': "Xi'an Jiaotong University Bearing Dataset",
        'domain': 'Bearings',
        'num_classes': 5,
    },
    'MFPT': {
        'name': 'MFPT',
        'description': 'Machinery Failure Prevention Technology Bearing Dataset',
        'domain': 'Bearings',
        'num_classes': 3,
    },
    'FEMTO': {
        'name': 'FEMTO',
        'description': 'FEMTO-ST Institute Bearing Dataset',
        'domain': 'Bearings',
        'num_classes': 3,
    },
    'MAFAULDA': {
        'name': 'MAFAULDA',
        'description': 'Machinery Fault Database',
        'domain': 'Multi-Fault',
        'num_classes': 5,
    },
    'Mendeley': {
        'name': 'Mendeley',
        'description': 'Mendeley Bearing Dataset',
        'domain': 'Bearings',
        'num_classes': 4,
    },
    'Planetary': {
        'name': 'Planetary',
        'description': 'Wind Turbine Planetary Gearbox Dataset',
        'domain': 'Gearbox',
        'num_classes': 4,
    },
    'Azure': {
        'name': 'Azure',
        'description': 'Microsoft Azure Predictive Maintenance Dataset',
        'domain': 'Multi-Fault',
        'num_classes': 2,
    },
    'Electric Motor Vibrations': {
        'name': 'Electric Motor Vibrations',
        'description': 'Electric Motor Fault Dataset',
        'domain': 'Motor',
        'num_classes': 4,
    },
    'Rotor Broken Bar': {
        'name': 'Rotor Broken Bar',
        'description': 'IEEE Rotor Broken Bar Dataset',
        'domain': 'Motor',
        'num_classes': 2,
    },
    'Gear Box UoC': {
        'name': 'Gear Box UoC',
        'description': 'University of Connecticut Gearbox Dataset',
        'domain': 'Gearbox',
        'num_classes': 9,
    },
}


def _resolve_dataset_name(name_or_id: str) -> str:
    """Resolve a dataset name or legacy index to a canonical dataset name."""
    if name_or_id in DATASET_INFO:
        return name_or_id
    # Support legacy numeric indices for backward compatibility
    if name_or_id in _DIR_TO_NAME:
        return _DIR_TO_NAME[name_or_id]
    raise ValueError(
        f"Dataset '{name_or_id}' not found. Available datasets: {list(DATASET_INFO.keys())}"
    )


def list_datasets() -> List[str]:
    """List all available dataset names."""
    return list(DATASET_INFO.keys())


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get metadata for a specific dataset by name."""
    name = _resolve_dataset_name(dataset_name)
    return DATASET_INFO[name]


class PDMDatasetBase(Dataset):
    """Base PyTorch Dataset for PDM data."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        transform: Optional[Callable] = None
    ):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


class PDMDataset:
    """
    High-level interface for PDMBench datasets.

    Example:
        dataset = PDMDataset('Paderborn')
        train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=32)

        # Or access raw data
        X_train, y_train = dataset.train_data
        X_test, y_test = dataset.test_data
    """

    def __init__(
        self,
        dataset_name: str,
        root_path: Optional[str] = None,
        seq_len: Optional[int] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize PDM dataset.

        Args:
            dataset_name: Dataset name (e.g., 'Paderborn', 'CWRU', 'XJTU').
                          Legacy numeric indices ('01', '04') are also accepted.
            root_path: Root path to datasets. If None, uses default location.
            seq_len: Sequence length to use. If None, uses full length.
            transform: Optional transform to apply to samples.
        """
        self.dataset_name = _resolve_dataset_name(dataset_name)
        self.dataset_id = _NAME_TO_DIR[self.dataset_name]  # internal directory index
        self.info = DATASET_INFO[self.dataset_name]
        self.transform = transform

        # Determine root path
        if root_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            root_path = os.path.join(current_dir, '..', 'dataset', self.dataset_id)

        self.root_path = root_path

        # Load data
        self._load_data()

        # Apply sequence length truncation if specified
        if seq_len is not None:
            self._truncate_seq_len(seq_len)

    def _load_data(self):
        """Load train, validation, and test data."""
        train_path = os.path.join(self.root_path, 'PdM_TRAIN.npz')
        val_path = os.path.join(self.root_path, 'PdM_VAL.npz')
        test_path = os.path.join(self.root_path, 'PdM_TEST.npz')

        # Load training data
        if os.path.exists(train_path):
            train_data = np.load(train_path, allow_pickle=True)
            self.X_train = train_data['features']
            self.y_train = train_data['labels']
        else:
            raise FileNotFoundError(f"Training data not found: {train_path}")

        # Load validation data
        if os.path.exists(val_path):
            val_data = np.load(val_path, allow_pickle=True)
            self.X_val = val_data['features']
            self.y_val = val_data['labels']
        else:
            # If no validation set, split from training
            split_idx = int(len(self.X_train) * 0.8)
            self.X_val = self.X_train[split_idx:]
            self.y_val = self.y_train[split_idx:]
            self.X_train = self.X_train[:split_idx]
            self.y_train = self.y_train[:split_idx]

        # Load test data
        if os.path.exists(test_path):
            test_data = np.load(test_path, allow_pickle=True)
            self.X_test = test_data['features']
            self.y_test = test_data['labels']
        else:
            raise FileNotFoundError(f"Test data not found: {test_path}")

        # Ensure consistent shape [N, T, C]
        self.X_train = self._ensure_3d(self.X_train)
        self.X_val = self._ensure_3d(self.X_val)
        self.X_test = self._ensure_3d(self.X_test)

    def _ensure_3d(self, X: np.ndarray) -> np.ndarray:
        """Ensure data is 3D [N, T, C]."""
        if X.ndim == 2:
            return X[:, :, np.newaxis]
        return X

    def _truncate_seq_len(self, seq_len: int):
        """Truncate sequences to specified length."""
        self.X_train = self.X_train[:, :seq_len, :]
        self.X_val = self.X_val[:, :seq_len, :]
        self.X_test = self.X_test[:, :seq_len, :]

    @property
    def seq_len(self) -> int:
        """Return the sequence length."""
        return self.X_train.shape[1]

    @property
    def num_features(self) -> int:
        """Return the number of features/channels."""
        return self.X_train.shape[2]

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return len(np.unique(self.y_train))

    @property
    def train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return training data as (features, labels)."""
        return self.X_train, self.y_train

    @property
    def val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return validation data as (features, labels)."""
        return self.X_val, self.y_val

    @property
    def test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return test data as (features, labels)."""
        return self.X_test, self.y_test

    def get_loaders(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle_train: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get PyTorch DataLoaders for train, validation, and test sets.

        Args:
            batch_size: Batch size for all loaders
            num_workers: Number of workers for data loading
            shuffle_train: Whether to shuffle training data

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_dataset = PDMDatasetBase(self.X_train, self.y_train, self.transform)
        val_dataset = PDMDatasetBase(self.X_val, self.y_val)
        test_dataset = PDMDatasetBase(self.X_test, self.y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return train_loader, val_loader, test_loader

    def get_class_distribution(self) -> Dict[str, Dict[int, int]]:
        """Get class distribution for each split."""
        def count_classes(y):
            unique, counts = np.unique(y, return_counts=True)
            return dict(zip(unique.tolist(), counts.tolist()))

        return {
            'train': count_classes(self.y_train),
            'val': count_classes(self.y_val),
            'test': count_classes(self.y_test)
        }

    def __repr__(self) -> str:
        return (
            f"PDMDataset(name='{self.dataset_name}', "
            f"seq_len={self.seq_len}, features={self.num_features}, "
            f"classes={self.num_classes}, "
            f"train={len(self.y_train)}, val={len(self.y_val)}, test={len(self.y_test)})"
        )
