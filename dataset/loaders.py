import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

# Name-to-directory mapping for name-based dataset access
_NAME_TO_DIR = {
    'Paderborn': '01', 'HUST': '02', 'IMS': '03', 'CWRU': '04',
    'XJTU': '05', 'MFPT': '06', 'FEMTO': '07', 'MAFAULDA': '09',
    'Mendeley': '12', 'Planetary': '13', 'Azure': '16',
    'Electric Motor Vibrations': '17', 'Rotor Broken Bar': '18',
    'Gear Box UoC': '19',
}
_DIR_TO_NAME = {v: k for k, v in _NAME_TO_DIR.items()}


class PdMDataset(Dataset):
    """Dataset class for Predictive Maintenance data"""
    def __init__(self, data, labels=None, feature_df=None, class_names=None, max_seq_len=None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.feature_df = feature_df
        self.class_names = class_names
        self.max_seq_len = max_seq_len if max_seq_len is not None else data.shape[1]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]

class DataProvider:
    """Data provider for Predictive Maintenance tasks"""
    def __init__(self, args):
        self.args = args
        self.root_path = args.root_path
        self.data_path = args.data_path
        self.features = args.features
        self.target = args.target
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Dataset mapping with descriptions
        self.dataset_mapping = {
            "01": {  # Paderborn
                "name": "Paderborn",
                "description": "Paderborn University Bearing Dataset - Comprehensive bearing fault dataset with multiple sensor measurements",
                "features": ["vibration", "temperature", "speed"],
                "fault_types": ["normal", "inner_race", "outer_race", "cage"],
                "sampling_rate": 64000,
                "duration": 4,
                "channels": 3,
                "design_target": "Fault diagnosis",
                "property": "Multiple sensors"
            },
            "02": {  # HUST
                "name": "HUST",
                "description": "Huazhong University of Science and Technology Bearing Dataset - Vibration and temperature data for fault diagnosis",
                "features": ["vibration", "temperature"],
                "fault_types": ["normal", "inner_race", "outer_race", "ball"],
                "sampling_rate": 51200,
                "duration": 10,
                "channels": 2,
                "design_target": "Fault diagnosis",
                "property": "Vibration"
            },
            "03": {  # IMS
                "name": "IMS",
                "description": "Intelligent Maintenance Systems Bearing Dataset - Long-term bearing run-to-failure data for RUL prediction",
                "features": ["vibration"],
                "fault_types": ["normal", "failure"],
                "sampling_rate": 20480,
                "duration": 24,
                "channels": 1,
                "design_target": "RUL prediction",
                "property": "Vibration"
            },
            "04": {  # CWRU
                "name": "CWRU",
                "description": "Case Western Reserve University Bearing Dataset - A comprehensive bearing fault dataset with various fault types and operating conditions",
                "features": ["vibration"],
                "fault_types": ["normal", "inner_race", "outer_race", "ball"],
                "sampling_rate": 12000,
                "duration": 10,
                "channels": 1,
                "design_target": "Fault diagnosis",
                "property": "Vibration"
            },
            "05": {  # XJTU
                "name": "XJTU-SY",
                "description": "Xi'an Jiaotong University Bearing Dataset - Dataset for both RUL prediction and fault diagnosis with accelerated degradation tests",
                "features": ["vibration"],
                "fault_types": ["normal", "inner_race", "outer_race", "cage"],
                "sampling_rate": 25600,
                "duration": 8,
                "channels": 1,
                "design_target": "RUL prediction & Fault diagnosis",
                "property": "Vibration"
            },
            "06": {  # MFPT
                "name": "MFPT",
                "description": "Mechanical Fault Prevention Technology Bearing Dataset - High-frequency vibration data for fault diagnosis",
                "features": ["vibration"],
                "fault_types": ["normal", "inner_race", "outer_race", "ball"],
                "sampling_rate": 97600,
                "duration": 2,
                "channels": 1,
                "design_target": "Fault diagnosis",
                "property": "Vibration"
            },
            "07": {  # FEMTO
                "name": "FEMTO",
                "description": "FEMTO-ST Institute Bearing Dataset - Dataset for RUL prediction with multiple sensor measurements",
                "features": ["vibration", "temperature", "speed"],
                "fault_types": ["normal", "failure"],
                "sampling_rate": 25600,
                "duration": 6,
                "channels": 3,
                "design_target": "RUL prediction",
                "property": "Multiple sensors"
            },
            "09": {  # MAFAULDA
                "name": "MAFAULDA",
                "description": "Multiple Fault Dataset - Comprehensive dataset with multiple sensor measurements for various fault types",
                "features": ["vibration", "current", "temperature", "speed"],
                "fault_types": ["normal", "bearing", "rotor", "stator", "gear"],
                "sampling_rate": 51200,
                "duration": 12,
                "channels": 4,
                "design_target": "Fault diagnosis",
                "property": "Multiple sensors"
            },
            "12": {  # Mendeley
                "name": "Mendeley Bearing",
                "description": "Mendeley Bearing Dataset - Vibration data for bearing fault diagnosis",
                "features": ["vibration"],
                "fault_types": ["normal", "inner_race", "outer_race", "ball"],
                "sampling_rate": 9600,
                "duration": 3,
                "channels": 1,
                "design_target": "Fault diagnosis",
                "property": "Vibration"
            },
            "13": {  # Planetary
                "name": "WT Planetary Gearbox",
                "description": "Wind Turbine Planetary Gearbox Dataset - Vibration data for gearbox fault diagnosis",
                "features": ["vibration"],
                "fault_types": ["normal", "sun_gear", "planet_gear", "ring_gear"],
                "sampling_rate": 48000,
                "duration": 6,
                "channels": 1,
                "design_target": "Fault diagnosis",
                "property": "Vibration"
            },
            "16": {  # Azure
                "name": "Microsoft Azure",
                "description": "Microsoft Azure Predictive Maintenance Dataset - Telemetry and log data for RUL prediction",
                "features": ["telemetry", "errors", "maintenance", "machines"],
                "fault_types": ["normal", "failure"],
                "sampling_rate": 1,
                "duration": 365,
                "channels": 4,
                "design_target": "RUL prediction",
                "property": "Telemetry, Logs"
            },
            "17": {  # Electric Motor Vibrations
                "name": "Electric Motor",
                "description": "Electric Motor Fault Dataset - Multiple sensor measurements for motor fault diagnosis",
                "features": ["current", "vibration", "temperature", "speed"],
                "fault_types": ["normal", "bearing", "rotor", "stator"],
                "sampling_rate": 42000,
                "duration": 5,
                "channels": 4,
                "design_target": "Fault diagnosis",
                "property": "Multiple sensors"
            },
            "18": {  # Rotor Broken Bar
                "name": "Rotor Broken Bar",
                "description": "IEEE Rotor Broken Bar Dataset - Current measurements for rotor fault diagnosis",
                "features": ["current"],
                "fault_types": ["normal", "broken_bar"],
                "sampling_rate": 50000,
                "duration": 3,
                "channels": 1,
                "design_target": "Fault diagnosis",
                "property": "Current"
            },
            "19": {  # Gear Box UoC
                "name": "CQU Gearbox",
                "description": "Chongqing University Gearbox Dataset - Multiple sensor measurements for gearbox fault diagnosis",
                "features": ["vibration", "temperature", "speed"],
                "fault_types": ["normal", "gear_wear", "gear_break", "bearing"],
                "sampling_rate": 20000,
                "duration": 8,
                "channels": 3,
                "design_target": "Fault diagnosis",
                "property": "Multiple sensors"
            }
        }

    
    def _load_and_preprocess_data(self, flag):
        """Load and preprocess data for a specific split"""
        # Construct file path based on flag
        file_name = f'PdM_{flag}.npz'
        file_path = os.path.join(self.root_path, file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load data
        data = np.load(file_path, allow_pickle=True)
        
        # Extract features and labels
        x = data['x'] if 'x' in data else None
        y = data['y'] if 'y' in data else None
        
        if x is None:
            raise ValueError(f"No features found in {file_path}")
        
        # Get feature names and class names
        feature_names = self.dataset_mapping.get(self.args.data, {}).get('features', [f'feature_{i}' for i in range(x.shape[-1])])
        class_names = self.dataset_mapping.get(self.args.data, {}).get('fault_types', [f'class_{i}' for i in range(len(np.unique(y)))] if y is not None else [])
        
        # Create feature DataFrame
        feature_df = pd.DataFrame(columns=feature_names)
        
        # Normalize features if needed
        if self.args.normalize:
            if flag == 'TRAIN':
                x_reshaped = x.reshape(-1, x.shape[-1])
                self.scaler.fit(x_reshaped)
            x_reshaped = x.reshape(-1, x.shape[-1])
            x_normalized = self.scaler.transform(x_reshaped)
            x = x_normalized.reshape(x.shape)
        
        # Encode labels if needed
        if y is not None and flag == 'TRAIN':
            self.label_encoder.fit(y)
        if y is not None:
            y = self.label_encoder.transform(y)
        
        # Calculate max sequence length
        max_seq_len = x.shape[1]
        
        return x, y, feature_df, class_names, max_seq_len
    
    def get_data(self, flag):
        """Get data and dataloader for a specific split"""
        x, y, feature_df, class_names, max_seq_len = self._load_and_preprocess_data(flag)
        
        # Create dataset
        dataset = PdMDataset(
            data=x,
            labels=y,
            feature_df=feature_df,
            class_names=class_names,
            max_seq_len=max_seq_len
        )
        
        # Create dataloader
        dataloader = TorchDataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=(flag == 'TRAIN'),
            num_workers=self.args.num_workers,
            drop_last=(flag == 'TRAIN')
        )
        
        return dataset, dataloader

def data_provider(args, flag):
    """Data provider function used by experiment classes"""
    data_provider = DataProvider(args)
    return data_provider.get_data(flag)

def _resolve_dataset_id(name_or_id):
    """Resolve a dataset name or index to directory index."""
    if name_or_id in _NAME_TO_DIR:
        return _NAME_TO_DIR[name_or_id]
    if name_or_id in _DIR_TO_NAME:
        return name_or_id
    raise ValueError(f"Dataset '{name_or_id}' not found. Available: {list(_NAME_TO_DIR.keys())}")


def get_dataset_info(dataset_name, root_path='./dataset/'):
    """Get information about a specific dataset by name or index."""
    dataset_id = _resolve_dataset_id(dataset_name)
    class Args:
        def __init__(self):
            self.root_path = root_path
            self.data = dataset_id
            self.data_path = os.path.join(root_path, f'PdM_{dataset_id}')
            self.features = None
            self.target = None
            self.batch_size = 32
            self.num_workers = 0
            self.normalize = True

    provider = DataProvider(Args())
    return provider.dataset_mapping.get(dataset_id, {})

def get_available_datasets(root_path='./dataset/'):
    """Get list of available datasets as {name: name} dict."""
    class Args:
        def __init__(self):
            self.root_path = root_path
            self.data = None
            self.data_path = root_path
            self.features = None
            self.target = None
            self.batch_size = 32
            self.num_workers = 0
            self.normalize = True

    provider = DataProvider(Args())
    return {v['name']: v['name'] for k, v in provider.dataset_mapping.items()}

if __name__ == "__main__":
    # Test functionality
    class Args:
        def __init__(self):
            self.root_path = './dataset/'
            self.data = '01'
            self.data_path = './dataset/PdM_01'
            self.features = None
            self.target = None
            self.batch_size = 32
            self.num_workers = 0
            self.normalize = True
    
    args = Args()
    provider = DataProvider(args)
    
    # Test data loading
    train_dataset, train_loader = provider.get_data('TRAIN')
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Feature shape: {train_dataset.data.shape}")
    if train_dataset.labels is not None:
        print(f"Number of classes: {len(train_dataset.class_names)}")
    
    # Test dataset info
    print("\nDataset Info:")
    print(get_dataset_info('01'))
    
    # Test available datasets
    print("\nAvailable Datasets:")
    print(get_available_datasets())