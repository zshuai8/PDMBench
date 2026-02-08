"""
Enhanced PDMBenchmark Streamlit Application

Features:
- Improved hyperparameter configuration with presets
- Multiple optimizer and LR scheduler choices
- Advanced data processing options
- Comprehensive data augmentation controls
- Real-time training progress visualization
- Dataset Explorer with t-SNE, distributions, sample viewer
- Custom model upload support
- Learning curves and confusion matrices
- Experiment history and comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import sys
import json
import time
import tempfile
import importlib.util
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import threading
from io import StringIO
import uuid
import hashlib
import atexit

# Import existing modules
from utils.config_manager import ConfigManager, get_config_manager, create_args
from utils.training_tracker import TrainingTracker, get_tracker
from utils.tools import EarlyStopping, cal_accuracy, cal_f1, evaluate_calibration
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize

# Page configuration
st.set_page_config(
    page_title="PDM Benchmark",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .status-running {
        color: #1f77b4;
        font-weight: bold;
    }
    .status-completed {
        color: #2ca02c;
        font-weight: bold;
    }
    .status-failed {
        color: #d62728;
        font-weight: bold;
    }
    .augmentation-card {
        background-color: #e8f4f8;
        border-radius: 8px;
        padding: 15px;
        margin: 5px 0;
    }
    .custom-model-info {
        background-color: #fff3cd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
DATASET_DIR = './dataset/'
RESULTS_DIR = './results/'
CUSTOM_MODELS_DIR = './custom_models/'
SESSIONS_DIR = './sessions/'
SESSION_TIMEOUT_MINUTES = 60  # Session expires after 60 minutes of inactivity

# Ensure directories exist
os.makedirs(CUSTOM_MODELS_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

class SessionManager:
    """Manages user sessions with automatic expiration."""

    def __init__(self, sessions_dir: str = SESSIONS_DIR, timeout_minutes: int = SESSION_TIMEOUT_MINUTES):
        self.sessions_dir = sessions_dir
        self.timeout = timedelta(minutes=timeout_minutes)
        self._cleanup_expired_sessions()

    def _get_session_file(self, session_id: str) -> str:
        """Get the path to a session file."""
        return os.path.join(self.sessions_dir, f"{session_id}.json")

    def _cleanup_expired_sessions(self):
        """Remove expired session files."""
        if not os.path.exists(self.sessions_dir):
            return

        now = datetime.now()
        for filename in os.listdir(self.sessions_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.sessions_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        session_data = json.load(f)
                    last_activity = datetime.fromisoformat(session_data.get('last_activity', '2000-01-01'))
                    if now - last_activity > self.timeout:
                        os.remove(filepath)
                        # Clean up session-specific temp files
                        session_id = filename.replace('.json', '')
                        self._cleanup_session_files(session_id)
                except (json.JSONDecodeError, KeyError, ValueError):
                    os.remove(filepath)

    def _cleanup_session_files(self, session_id: str):
        """Clean up temporary files associated with a session."""
        # Remove session-specific directories
        session_temp_dir = os.path.join(tempfile.gettempdir(), f"pdm_session_{session_id}")
        if os.path.exists(session_temp_dir):
            import shutil
            shutil.rmtree(session_temp_dir, ignore_errors=True)

    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())[:8]
        session_data = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            'data': {}
        }
        with open(self._get_session_file(session_id), 'w') as f:
            json.dump(session_data, f)
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data if it exists and is not expired."""
        filepath = self._get_session_file(session_id)
        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r') as f:
                session_data = json.load(f)

            last_activity = datetime.fromisoformat(session_data.get('last_activity', '2000-01-01'))
            if datetime.now() - last_activity > self.timeout:
                self.delete_session(session_id)
                return None

            return session_data
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def update_session(self, session_id: str, data: Dict):
        """Update session data and refresh last activity time."""
        session = self.get_session(session_id)
        if session is None:
            return False

        session['last_activity'] = datetime.now().isoformat()
        session['data'].update(data)

        with open(self._get_session_file(session_id), 'w') as f:
            json.dump(session, f)
        return True

    def delete_session(self, session_id: str):
        """Delete a session and its associated files."""
        filepath = self._get_session_file(session_id)
        if os.path.exists(filepath):
            os.remove(filepath)
        self._cleanup_session_files(session_id)

    def refresh_session(self, session_id: str):
        """Refresh the last activity time for a session."""
        session = self.get_session(session_id)
        if session:
            session['last_activity'] = datetime.now().isoformat()
            with open(self._get_session_file(session_id), 'w') as f:
                json.dump(session, f)


def init_session():
    """Initialize or restore user session."""
    if 'session_id' not in st.session_state:
        # Create new session
        session_manager = SessionManager()
        st.session_state.session_id = session_manager.create_session()
        st.session_state.session_created = datetime.now()
        st.session_state.session_manager = session_manager
    else:
        # Refresh existing session
        if 'session_manager' not in st.session_state:
            st.session_state.session_manager = SessionManager()
        st.session_state.session_manager.refresh_session(st.session_state.session_id)

    return st.session_state.session_id


def get_session_temp_dir() -> str:
    """Get a session-specific temporary directory."""
    session_id = st.session_state.get('session_id', 'default')
    temp_dir = os.path.join(tempfile.gettempdir(), f"pdm_session_{session_id}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

# Dataset mapping - maps display names to folder IDs
# Display name -> {folder: actual folder name, description: dataset description}
dataset_mapping = {
    "Paderborn": {"folder": "01", "description": "Paderborn University bearing dataset"},
    "HUST": {"folder": "02", "description": "Huazhong University ball bearing fault data"},
    "IMS": {"folder": "03", "description": "NASA IMS bearing dataset"},
    "CWRU": {"folder": "04", "description": "Case Western Reserve University bearing dataset"},
    "XJTU": {"folder": "05", "description": "Xi'an Jiaotong University run-to-failure bearing data"},
    "MFPT": {"folder": "06", "description": "MFPT Society mechanical fault data"},
    "FEMTO": {"folder": "07", "description": "FEMTO-ST accelerated degradation data"},
    "MAFAULDA": {"folder": "09", "description": "Machinery Fault Database"},
    "Mendeley": {"folder": "12", "description": "Mendeley bearing dataset"},
    "Planetary": {"folder": "13", "description": "Planetary gearbox dataset"},
    "Azure": {"folder": "16", "description": "Microsoft Azure predictive maintenance data"},
    "Electric Motor Vibrations": {"folder": "17", "description": "Electric motor vibrations dataset"},
    "Rotor Broken Bar": {"folder": "18", "description": "Rotor broken bar dataset"},
    "Gear Box UoC": {"folder": "19", "description": "University of Connecticut gearbox data"},
}

# Reverse mapping: folder ID -> display name (for results display)
folder_to_name = {v["folder"]: k for k, v in dataset_mapping.items()}

# Model list - ordered by loading speed (fast/simple models first)
model_list = [
    # Fast/Simple models (recommended for quick experiments)
    'DLinear', 'MLP', 'LightTS', 'TiDE',
    # Transformer variants
    'Transformer', 'PatchTST', 'iTransformer', 'Autoformer', 'Informer',
    'FEDformer', 'Crossformer', 'ETSformer', 'Reformer',
    # Advanced models
    'TimesNet', 'TimeMixer', 'TSMixer', 'FreTS', 'SegRNN',
    'Pyraformer', 'MICN', 'FiLM', 'Koopa', 'MambaSimple',
    'TemporalFusionTransformer', 'SCINet', 'PAttn', 'TimeXer',
    'WPMixer', 'MultiPatchFormer',
    # Foundation Models (pre-trained time series models)
    'Chronos', 'MOMENT', 'TimesFM', 'Moirai'
]

# Foundation model info
FOUNDATION_MODELS = {
    'Chronos': {
        'description': 'Amazon Chronos - T5-based time series foundation model',
        'sizes': ['tiny', 'mini', 'small', 'base', 'large'],
        'default_size': 'small',
        'paper': 'https://arxiv.org/abs/2403.07815'
    },
    'MOMENT': {
        'description': 'CMU MOMENT - Open-source foundation model for time series',
        'sizes': ['small', 'base', 'large'],
        'default_size': 'base',
        'paper': 'https://arxiv.org/abs/2402.03885'
    },
    'TimesFM': {
        'description': 'Google TimesFM - Decoder-only foundation model',
        'sizes': ['200m'],
        'default_size': '200m',
        'paper': 'https://arxiv.org/abs/2310.10688'
    },
    'Moirai': {
        'description': 'Salesforce Moirai - Universal time series foundation model',
        'sizes': ['small', 'base', 'large'],
        'default_size': 'small',
        'paper': 'https://arxiv.org/abs/2402.02592'
    }
}

# Task types
TASK_TYPES = {
    'classification': 'Fault Classification - Classify equipment health state',
    'early_failure': 'Early Failure Prediction - Predict impending failures',
    'rul': 'RUL Estimation - Remaining Useful Life prediction'
}

# Optimizer options
OPTIMIZER_OPTIONS = {
    'RAdam': 'Rectified Adam - Default, robust to learning rate',
    'Adam': 'Adam - Classic adaptive learning rate',
    'AdamW': 'AdamW - Adam with decoupled weight decay',
    'SGD': 'SGD - Stochastic Gradient Descent with momentum',
    'RMSprop': 'RMSprop - Good for RNNs',
    'Adagrad': 'Adagrad - Adaptive gradient algorithm',
    'Adadelta': 'Adadelta - Extension of Adagrad',
}

# LR Scheduler options
SCHEDULER_OPTIONS = {
    'none': 'No scheduler - Constant learning rate',
    'step': 'StepLR - Decay by gamma every step_size epochs',
    'exponential': 'ExponentialLR - Decay by gamma every epoch',
    'cosine': 'CosineAnnealingLR - Cosine annealing',
    'cosine_warm': 'CosineAnnealingWarmRestarts - Cosine with warm restarts',
    'plateau': 'ReduceLROnPlateau - Reduce when metric plateaus',
    'onecycle': 'OneCycleLR - Super-convergence schedule',
}

# Normalization options
NORMALIZATION_OPTIONS = {
    'standardization': 'Z-score normalization (zero mean, unit variance)',
    'minmax': 'Min-Max scaling to [0, 1]',
    'per_sample_std': 'Per-sample standardization',
    'per_sample_minmax': 'Per-sample min-max scaling',
    'robust': 'Robust scaling using median and IQR',
    'none': 'No normalization',
}

# Augmentation method descriptions
AUGMENTATION_METHODS = {
    'jitter': {'name': 'Jitter', 'desc': 'Add Gaussian noise', 'params': {'sigma': (0.01, 0.1, 0.03)}},
    'scaling': {'name': 'Scaling', 'desc': 'Random magnitude scaling', 'params': {'sigma': (0.05, 0.2, 0.1)}},
    'rotation': {'name': 'Rotation', 'desc': 'Flip and permute channels', 'params': {}},
    'permutation': {'name': 'Permutation', 'desc': 'Segment reordering', 'params': {'max_segments': (2, 10, 5)}},
    'magwarp': {'name': 'Magnitude Warp', 'desc': 'Cubic spline magnitude warping', 'params': {'sigma': (0.1, 0.4, 0.2), 'knot': (2, 8, 4)}},
    'timewarp': {'name': 'Time Warp', 'desc': 'Temporal axis warping', 'params': {'sigma': (0.1, 0.4, 0.2), 'knot': (2, 8, 4)}},
    'windowslice': {'name': 'Window Slice', 'desc': 'Random window cropping', 'params': {'reduce_ratio': (0.7, 0.95, 0.9)}},
    'windowwarp': {'name': 'Window Warp', 'desc': 'Scale temporal windows', 'params': {'window_ratio': (0.05, 0.2, 0.1)}},
    'spawner': {'name': 'SPAWNER', 'desc': 'DTW-guided averaging', 'params': {'sigma': (0.01, 0.1, 0.05)}},
    'wdba': {'name': 'WDBA', 'desc': 'Weighted DTW Barycenter Averaging', 'params': {'batch_size': (3, 10, 6)}},
    'dtwwarp': {'name': 'DTW Warp', 'desc': 'Random guided DTW warping', 'params': {}},
    'discdtw': {'name': 'Disc. DTW', 'desc': 'Discriminative guided DTW warping', 'params': {'batch_size': (3, 10, 6)}},
}


def get_available_datasets() -> Dict[str, str]:
    """Get available datasets. Returns {display_name: display_name} for UI selection."""
    available = {}
    for display_name, info in dataset_mapping.items():
        folder_id = info['folder']
        dataset_path = os.path.join(DATASET_DIR, folder_id)
        if os.path.exists(dataset_path):
            available[display_name] = display_name
    return available if available else {k: k for k in dataset_mapping.keys()}


def get_dataset_folder(display_name: str) -> str:
    """Get folder ID from display name."""
    if display_name in dataset_mapping:
        return dataset_mapping[display_name]['folder']
    # Fallback: check if it's already a folder ID
    if display_name in folder_to_name:
        return display_name
    return display_name


def get_dataset_description(display_name: str) -> str:
    """Get dataset description from display name."""
    if display_name in dataset_mapping:
        return dataset_mapping[display_name]['description']
    return ""


# =============================================================================
# CUSTOM MODEL WRAPPER
# =============================================================================

class CustomModelWrapper(nn.Module):
    """
    Wrapper that adapts user-uploaded models to PDMBench interface.

    User model requirements:
    - Must be a PyTorch nn.Module
    - Must have forward(x) method accepting [B, T, C] input
    - Must output [B, num_classes] for classification
    """

    def __init__(self, user_model: nn.Module, configs):
        super().__init__()
        self.user_model = user_model
        self.task_name = configs.task_name
        self.num_class = configs.num_class
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Standard PDMBench forward interface."""
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None

    def classification(self, x_enc, x_mark_enc):
        """Classification task."""
        # x_enc: [B, T, C]
        output = self.user_model(x_enc)
        return output


def load_custom_model(file_content: str, class_name: str) -> Tuple[Optional[type], str]:
    """
    Load a custom model class from uploaded Python code.

    Returns:
        Tuple of (model_class, error_message)
    """
    try:
        # Create a temporary module
        spec = importlib.util.spec_from_loader("custom_model", loader=None)
        module = importlib.util.module_from_spec(spec)

        # Execute the code in the module's namespace
        exec(file_content, module.__dict__)

        # Find the model class
        if hasattr(module, class_name):
            model_class = getattr(module, class_name)
            if not issubclass(model_class, nn.Module):
                return None, f"'{class_name}' is not a subclass of nn.Module"
            return model_class, ""
        else:
            # Try to find any nn.Module subclass
            for name, obj in module.__dict__.items():
                if isinstance(obj, type) and issubclass(obj, nn.Module) and obj != nn.Module:
                    return obj, ""
            return None, f"No nn.Module subclass found. Specify class name or ensure your model inherits from nn.Module"

    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def validate_custom_model(model_class: type, seq_len: int = 100, n_features: int = 7, n_classes: int = 4) -> Tuple[bool, str]:
    """
    Validate that a custom model works with expected input/output shapes.

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        # Create dummy input
        dummy_input = torch.randn(2, seq_len, n_features)

        # Try to instantiate the model
        try:
            model = model_class(n_features, n_classes)
        except TypeError:
            try:
                model = model_class(n_features, seq_len, n_classes)
            except TypeError:
                try:
                    model = model_class()
                except Exception as e:
                    return False, f"Cannot instantiate model. Ensure __init__ accepts (n_features, n_classes) or no args: {e}"

        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        # Check output shape
        if output.shape[0] != 2:
            return False, f"Output batch size mismatch. Expected 2, got {output.shape[0]}"

        if len(output.shape) != 2:
            return False, f"Output should be 2D [B, num_classes], got shape {output.shape}"

        return True, f"Model validated successfully! Output shape: {output.shape}"

    except Exception as e:
        return False, f"Validation failed: {str(e)}"


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def apply_preprocessing(data: np.ndarray, options: Dict) -> np.ndarray:
    """Apply preprocessing options to data."""
    processed = data.copy()

    # Detrending
    if options.get('detrend', 'none') != 'none':
        from scipy import signal
        if options['detrend'] == 'linear':
            processed = signal.detrend(processed, axis=1, type='linear')
        elif options['detrend'] == 'constant':
            processed = signal.detrend(processed, axis=1, type='constant')

    # Smoothing
    if options.get('smoothing', 'none') != 'none':
        window_size = options.get('smoothing_window', 5)
        if options['smoothing'] == 'moving_avg':
            kernel = np.ones(window_size) / window_size
            for i in range(processed.shape[0]):
                for j in range(processed.shape[2]):
                    processed[i, :, j] = np.convolve(processed[i, :, j], kernel, mode='same')
        elif options['smoothing'] == 'savgol':
            from scipy.signal import savgol_filter
            polyorder = min(3, window_size - 1)
            for i in range(processed.shape[0]):
                for j in range(processed.shape[2]):
                    processed[i, :, j] = savgol_filter(processed[i, :, j], window_size, polyorder)

    # Outlier handling
    if options.get('outlier_method', 'none') != 'none':
        if options['outlier_method'] == 'clip':
            threshold = options.get('outlier_threshold', 3.0)
            mean = np.mean(processed, axis=1, keepdims=True)
            std = np.std(processed, axis=1, keepdims=True)
            processed = np.clip(processed, mean - threshold * std, mean + threshold * std)
        elif options['outlier_method'] == 'winsorize':
            from scipy.stats import mstats
            limits = options.get('winsorize_limits', 0.05)
            for i in range(processed.shape[0]):
                for j in range(processed.shape[2]):
                    processed[i, :, j] = mstats.winsorize(processed[i, :, j], limits=[limits, limits])

    return processed


def compute_features(data: np.ndarray, options: Dict) -> np.ndarray:
    """Compute additional features from time series data."""
    features_list = []

    if options.get('stat_features', False):
        # Statistical features per sample
        mean = np.mean(data, axis=1)
        std = np.std(data, axis=1)
        skew = np.zeros_like(mean)
        kurtosis = np.zeros_like(mean)
        try:
            from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis
            for i in range(data.shape[0]):
                for j in range(data.shape[2]):
                    skew[i, j] = scipy_skew(data[i, :, j])
                    kurtosis[i, j] = scipy_kurtosis(data[i, :, j])
        except:
            pass
        features_list.extend([mean, std, skew, kurtosis])

    if options.get('freq_features', False):
        # FFT features
        fft_data = np.abs(np.fft.rfft(data, axis=1))
        # Take first few frequency components
        n_freqs = min(10, fft_data.shape[1])
        features_list.append(fft_data[:, :n_freqs, :].reshape(data.shape[0], -1))

    if features_list:
        return np.concatenate([f.reshape(data.shape[0], -1) for f in features_list], axis=1)
    return None


# =============================================================================
# HYPERPARAMETER CONFIGURATION UI
# =============================================================================

def render_preset_selector() -> Optional[str]:
    """Render preset selection UI."""
    config_manager = get_config_manager()
    presets = config_manager.get_preset_names()

    # Group presets
    scenario_presets = ['quick', 'balanced', 'high_accuracy', 'memory_efficient']
    model_presets = [p for p in presets if p not in scenario_presets]

    st.subheader("Configuration Presets")

    col1, col2 = st.columns(2)

    with col1:
        scenario = st.selectbox(
            "Training Scenario",
            options=['custom'] + scenario_presets,
            format_func=lambda x: {
                'custom': '🔧 Custom Configuration',
                'quick': '⚡ Quick (Fast iteration)',
                'balanced': '⚖️ Balanced (Recommended)',
                'high_accuracy': '🎯 High Accuracy (Best results)',
                'memory_efficient': '💾 Memory Efficient'
            }.get(x, x),
            help="Select a pre-configured training scenario"
        )

    with col2:
        if scenario != 'custom':
            desc = config_manager.get_preset_description(scenario)
            st.info(f"**Description:** {desc}")

    return scenario if scenario != 'custom' else None


def render_hyperparameter_ui(preset: Optional[str] = None) -> Dict:
    """Render the hyperparameter configuration UI with all options."""
    config_manager = get_config_manager()

    # Get default values from preset or default config
    if preset:
        config = config_manager.get_preset(preset)
    else:
        config = config_manager.default_config

    training_cfg = config.get('training', {})
    model_cfg = config.get('model', {})

    st.subheader("Hyperparameter Configuration")

    # Create tabs for different parameter groups
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Training & Optimization", "Architecture", "Data Processing", "Augmentation", "Advanced"
    ])

    params = {}

    # =========================================================================
    # TAB 1: Training & Optimization
    # =========================================================================
    with tab1:
        st.markdown("### Basic Training")
        col1, col2, col3 = st.columns(3)

        with col1:
            params['batch_size'] = st.select_slider(
                "Batch Size",
                options=[8, 16, 32, 64, 128, 256, 512],
                value=training_cfg.get('batch_size', 32),
                help="Larger batch = faster training, more memory"
            )

            params['train_epochs'] = st.slider(
                "Training Epochs",
                min_value=5, max_value=200,
                value=training_cfg.get('train_epochs', 50),
                help="Number of training epochs"
            )

            params['seed'] = st.number_input(
                "Random Seed",
                min_value=0, max_value=9999,
                value=training_cfg.get('seed', 2021),
                help="For reproducibility"
            )

        with col2:
            params['learning_rate'] = st.select_slider(
                "Learning Rate",
                options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                value=training_cfg.get('learning_rate', 0.001),
                format_func=lambda x: f"{x:.5f}",
                help="Initial learning rate"
            )

            params['patience'] = st.slider(
                "Early Stopping Patience",
                min_value=3, max_value=50,
                value=training_cfg.get('patience', 7),
                help="Epochs to wait before early stopping"
            )

            params['gradient_clip'] = st.slider(
                "Gradient Clipping",
                min_value=0.0, max_value=10.0, step=0.5,
                value=4.0,
                help="Max gradient norm (0 = disabled)"
            )

        with col3:
            params['dropout'] = st.slider(
                "Dropout Rate",
                min_value=0.0, max_value=0.5, step=0.05,
                value=model_cfg.get('dropout', 0.1),
                help="Regularization dropout rate"
            )

            params['weight_decay'] = st.select_slider(
                "Weight Decay (L2)",
                options=[0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                value=0,
                format_func=lambda x: f"{x:.6f}" if x > 0 else "0 (disabled)",
                help="L2 regularization strength"
            )

        st.markdown("---")
        st.markdown("### Optimizer Selection")
        col1, col2 = st.columns(2)

        with col1:
            params['optimizer'] = st.selectbox(
                "Optimizer",
                options=list(OPTIMIZER_OPTIONS.keys()),
                index=0,
                format_func=lambda x: f"{x} - {OPTIMIZER_OPTIONS[x].split(' - ')[1]}",
                help="Choose optimization algorithm"
            )

            if params['optimizer'] == 'SGD':
                params['momentum'] = st.slider(
                    "Momentum",
                    min_value=0.0, max_value=0.99, step=0.01,
                    value=0.9,
                    help="SGD momentum"
                )
                params['nesterov'] = st.checkbox("Nesterov Momentum", value=True)
            else:
                params['momentum'] = 0.9
                params['nesterov'] = False

            if params['optimizer'] in ['Adam', 'AdamW', 'RAdam']:
                params['beta1'] = st.slider("Beta1", 0.8, 0.99, 0.9, 0.01)
                params['beta2'] = st.slider("Beta2", 0.9, 0.999, 0.999, 0.001)
            else:
                params['beta1'] = 0.9
                params['beta2'] = 0.999

        with col2:
            params['scheduler'] = st.selectbox(
                "LR Scheduler",
                options=list(SCHEDULER_OPTIONS.keys()),
                index=0,
                format_func=lambda x: SCHEDULER_OPTIONS[x].split(' - ')[0],
                help="Learning rate schedule"
            )

            if params['scheduler'] == 'step':
                params['lr_step_size'] = st.slider("Step Size (epochs)", 5, 50, 10)
                params['lr_gamma'] = st.slider("Gamma (decay factor)", 0.1, 0.9, 0.5, 0.1)
            elif params['scheduler'] == 'exponential':
                params['lr_gamma'] = st.slider("Gamma (decay factor)", 0.9, 0.99, 0.95, 0.01)
            elif params['scheduler'] == 'cosine':
                params['lr_t_max'] = st.slider("T_max (period)", 10, 100, 50)
                params['lr_eta_min'] = st.select_slider("Min LR", [0, 1e-7, 1e-6, 1e-5], 1e-6)
            elif params['scheduler'] == 'cosine_warm':
                params['lr_t_0'] = st.slider("T_0 (first restart)", 5, 50, 10)
                params['lr_t_mult'] = st.slider("T_mult (restart multiplier)", 1, 3, 2)
            elif params['scheduler'] == 'plateau':
                params['lr_factor'] = st.slider("Factor", 0.1, 0.9, 0.5, 0.1)
                params['lr_patience'] = st.slider("Patience", 2, 20, 5)
            elif params['scheduler'] == 'onecycle':
                params['lr_max_lr'] = params['learning_rate'] * 10
                params['lr_pct_start'] = st.slider("Warmup %", 0.1, 0.5, 0.3, 0.1)

    # =========================================================================
    # TAB 2: Architecture
    # =========================================================================
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            params['d_model'] = st.select_slider(
                "Model Dimension (d_model)",
                options=[32, 64, 128, 256, 512, 1024],
                value=model_cfg.get('d_model', 512),
                help="Hidden dimension of the model"
            )

            params['n_heads'] = st.select_slider(
                "Attention Heads",
                options=[1, 2, 4, 8, 16],
                value=model_cfg.get('n_heads', 8),
                help="Number of attention heads"
            )

            params['e_layers'] = st.slider(
                "Encoder Layers",
                min_value=1, max_value=6,
                value=model_cfg.get('e_layers', 2),
                help="Number of encoder layers"
            )

        with col2:
            params['d_ff'] = st.select_slider(
                "Feed-Forward Dimension",
                options=[64, 128, 256, 512, 1024, 2048, 4096],
                value=model_cfg.get('d_ff', 2048),
                help="Feed-forward network dimension"
            )

            params['d_layers'] = st.slider(
                "Decoder Layers",
                min_value=1, max_value=4,
                value=model_cfg.get('d_layers', 1),
                help="Number of decoder layers"
            )

            params['activation'] = st.selectbox(
                "Activation Function",
                options=['gelu', 'relu', 'swish', 'silu', 'tanh'],
                index=['gelu', 'relu', 'swish', 'silu', 'tanh'].index(model_cfg.get('activation', 'gelu')),
                help="Activation function type"
            )

        st.markdown("---")
        st.markdown("### Model-Specific Parameters")
        col1, col2 = st.columns(2)

        model_specific = config.get('model_specific', {})

        with col1:
            st.markdown("**TimesNet / Inception**")
            params['top_k'] = st.slider("Top-K Frequencies", 1, 10, model_specific.get('top_k', 5))
            params['num_kernels'] = st.slider("Num Kernels", 2, 12, model_specific.get('num_kernels', 6))

            st.markdown("**PatchTST**")
            params['patch_len'] = st.select_slider(
                "Patch Length",
                options=[8, 12, 16, 24, 32, 48, 64],
                value=model_specific.get('patch_len', 16)
            )

        with col2:
            st.markdown("**Mamba**")
            params['expand'] = st.slider("Expand Factor", 1, 4, model_specific.get('expand', 2))
            params['d_conv'] = st.slider("Conv Kernel Size", 2, 8, model_specific.get('d_conv', 4))

            st.markdown("**FEDformer / Decomposition**")
            params['moving_avg'] = st.slider("Moving Avg Window", 5, 51, model_specific.get('moving_avg', 25), step=2)

    # =========================================================================
    # TAB 3: Data Processing
    # =========================================================================
    with tab3:
        st.markdown("### Normalization")
        col1, col2 = st.columns(2)

        with col1:
            params['normalization'] = st.selectbox(
                "Normalization Method",
                options=list(NORMALIZATION_OPTIONS.keys()),
                index=0,
                format_func=lambda x: f"{x}: {NORMALIZATION_OPTIONS[x][:40]}...",
                help="Data normalization strategy"
            )

        with col2:
            params['norm_per_feature'] = st.checkbox(
                "Per-Feature Normalization",
                value=True,
                help="Normalize each feature independently"
            )

        st.markdown("---")
        st.markdown("### Signal Processing")
        col1, col2, col3 = st.columns(3)

        with col1:
            params['detrend'] = st.selectbox(
                "Detrending",
                options=['none', 'linear', 'constant'],
                help="Remove trends from data"
            )

        with col2:
            params['smoothing'] = st.selectbox(
                "Smoothing",
                options=['none', 'moving_avg', 'savgol'],
                help="Smooth noisy signals"
            )
            if params['smoothing'] != 'none':
                params['smoothing_window'] = st.slider("Window Size", 3, 21, 5, step=2)
            else:
                params['smoothing_window'] = 5

        with col3:
            params['outlier_method'] = st.selectbox(
                "Outlier Handling",
                options=['none', 'clip', 'winsorize'],
                help="Handle outliers in data"
            )
            if params['outlier_method'] == 'clip':
                params['outlier_threshold'] = st.slider("Threshold (std)", 2.0, 5.0, 3.0, 0.5)
            elif params['outlier_method'] == 'winsorize':
                params['winsorize_limits'] = st.slider("Winsorize %", 0.01, 0.1, 0.05, 0.01)
            else:
                params['outlier_threshold'] = 3.0
                params['winsorize_limits'] = 0.05

        st.markdown("---")
        st.markdown("### Feature Engineering")
        col1, col2 = st.columns(2)

        with col1:
            params['stat_features'] = st.checkbox(
                "Statistical Features",
                value=False,
                help="Add mean, std, skewness, kurtosis"
            )

        with col2:
            params['freq_features'] = st.checkbox(
                "Frequency Features",
                value=False,
                help="Add FFT-based frequency features"
            )

    # =========================================================================
    # TAB 4: Augmentation
    # =========================================================================
    with tab4:
        params['augmentation_enabled'] = st.checkbox(
            "Enable Data Augmentation",
            value=False,
            help="Apply data augmentation during training"
        )

        if params['augmentation_enabled']:
            col1, col2 = st.columns([1, 2])

            with col1:
                params['augmentation_ratio'] = st.slider(
                    "Augmentation Rounds",
                    1, 5, 2,
                    help="Number of augmentation passes per sample"
                )

                params['aug_probability'] = st.slider(
                    "Apply Probability",
                    0.1, 1.0, 0.5, 0.1,
                    help="Probability of applying each augmentation"
                )

            with col2:
                st.markdown("**Quick Select:**")
                aug_preset = st.radio(
                    "Preset",
                    ['Custom', 'Light', 'Medium', 'Heavy'],
                    horizontal=True,
                    label_visibility="collapsed"
                )

                if aug_preset == 'Light':
                    default_augs = ['jitter', 'scaling']
                elif aug_preset == 'Medium':
                    default_augs = ['jitter', 'scaling', 'permutation', 'magwarp']
                elif aug_preset == 'Heavy':
                    default_augs = ['jitter', 'scaling', 'permutation', 'magwarp', 'timewarp', 'windowwarp']
                else:
                    default_augs = []

            st.markdown("---")
            st.markdown("### Augmentation Methods")

            # Create columns for augmentation methods
            aug_cols = st.columns(3)
            aug_params = {}

            for idx, (aug_key, aug_info) in enumerate(AUGMENTATION_METHODS.items()):
                col_idx = idx % 3
                with aug_cols[col_idx]:
                    with st.expander(f"**{aug_info['name']}**", expanded=aug_key in default_augs):
                        params[aug_key] = st.checkbox(
                            f"Enable {aug_info['name']}",
                            value=aug_key in default_augs,
                            key=f"aug_{aug_key}"
                        )
                        st.caption(aug_info['desc'])

                        # Show parameter sliders if enabled
                        if params[aug_key] and aug_info['params']:
                            for param_name, (min_v, max_v, default_v) in aug_info['params'].items():
                                if isinstance(default_v, int):
                                    aug_params[f"{aug_key}_{param_name}"] = st.slider(
                                        param_name.title(),
                                        min_v, max_v, default_v,
                                        key=f"aug_{aug_key}_{param_name}"
                                    )
                                else:
                                    aug_params[f"{aug_key}_{param_name}"] = st.slider(
                                        param_name.title(),
                                        float(min_v), float(max_v), float(default_v),
                                        key=f"aug_{aug_key}_{param_name}"
                                    )

            params['aug_params'] = aug_params

            # Augmentation preview
            st.markdown("---")
            if st.checkbox("Show Augmentation Preview"):
                st.info("Augmentation preview will show sample transformations when data is loaded.")
        else:
            # Set all augmentation flags to False
            for aug_key in AUGMENTATION_METHODS.keys():
                params[aug_key] = False
            params['augmentation_ratio'] = 0
            params['aug_params'] = {}

    # =========================================================================
    # TAB 5: Advanced
    # =========================================================================
    with tab5:
        st.markdown("### Mixed Precision & Performance")
        col1, col2 = st.columns(2)

        with col1:
            params['use_amp'] = st.checkbox(
                "Mixed Precision (AMP)",
                value=False,
                help="Use automatic mixed precision for faster training"
            )

            params['num_workers'] = st.slider(
                "Data Loader Workers",
                0, 16, 4,
                help="Number of parallel data loading workers"
            )

        with col2:
            params['pin_memory'] = st.checkbox(
                "Pin Memory",
                value=True,
                help="Pin memory for faster GPU transfer"
            )

            params['prefetch_factor'] = st.slider(
                "Prefetch Factor",
                1, 4, 2,
                help="Batches to prefetch per worker"
            )

        st.markdown("---")
        st.markdown("### Loss Function")
        params['loss_function'] = st.selectbox(
            "Loss Function",
            options=['cross_entropy', 'focal', 'label_smoothing'],
            help="Classification loss function"
        )

        if params['loss_function'] == 'focal':
            params['focal_gamma'] = st.slider("Focal Gamma", 0.5, 5.0, 2.0, 0.5)
        elif params['loss_function'] == 'label_smoothing':
            params['label_smoothing'] = st.slider("Smoothing", 0.0, 0.3, 0.1, 0.05)

        st.markdown("---")
        st.markdown("### Logging & Checkpointing")
        col1, col2 = st.columns(2)

        with col1:
            params['save_best_only'] = st.checkbox("Save Best Model Only", value=True)
            params['log_every_n_steps'] = st.slider("Log Every N Steps", 1, 100, 10)

        with col2:
            params['use_wandb'] = st.checkbox("Use Weights & Biases", value=False)
            if params['use_wandb']:
                params['wandb_project'] = st.text_input("W&B Project", "pdmbench")

    return params


# =============================================================================
# TRAINING PROGRESS VISUALIZATION
# =============================================================================

def render_training_progress(progress: Dict):
    """Render real-time training progress."""
    if not progress:
        return

    # Progress bar
    progress_pct = progress.get('progress_pct', 0)
    current_epoch = progress.get('current_epoch', 0)
    total_epochs = progress.get('total_epochs', 1)

    st.progress(progress_pct / 100, text=f"Epoch {current_epoch}/{total_epochs}")

    # Current metrics
    epoch_history = progress.get('epoch_history', [])
    if epoch_history:
        latest = epoch_history[-1]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Train Loss", f"{latest.get('train_loss', 0):.4f}")
        with col2:
            st.metric("Train Acc", f"{latest.get('train_accuracy', 0):.2%}")
        with col3:
            st.metric("Val Loss", f"{latest.get('val_loss', 0):.4f}")
        with col4:
            st.metric(
                "Val Acc",
                f"{latest.get('val_accuracy', 0):.2%}",
                delta=f"Best: {progress.get('best_val_accuracy', 0):.2%}"
            )


def render_learning_curves(epoch_history: List[Dict], show_lr: bool = False):
    """Render learning curves visualization."""
    if not epoch_history or len(epoch_history) < 2:
        st.info("Training in progress... Learning curves will appear after the first epoch.")
        return

    df = pd.DataFrame(epoch_history)

    # Create subplots
    n_cols = 3 if show_lr and 'learning_rate' in df.columns else 2
    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=["Loss Curves", "Accuracy Curves"] + (["Learning Rate"] if n_cols == 3 else []),
        horizontal_spacing=0.1
    )

    # Loss curves
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['train_loss'], name='Train Loss',
                   line=dict(color='#1f77b4', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['val_loss'], name='Val Loss',
                   line=dict(color='#ff7f0e', width=2)),
        row=1, col=1
    )

    # Accuracy curves
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['train_accuracy'], name='Train Acc',
                   line=dict(color='#2ca02c', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['val_accuracy'], name='Val Acc',
                   line=dict(color='#d62728', width=2)),
        row=1, col=2
    )

    # Learning rate curve
    if n_cols == 3 and 'learning_rate' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['learning_rate'], name='LR',
                       line=dict(color='#9467bd', width=2)),
            row=1, col=3
        )
        fig.update_yaxes(title_text="Learning Rate", type="log", row=1, col=3)

    fig.update_layout(
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)


def render_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None):
    """Render confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]

    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>"
    ))

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=500,
        yaxis=dict(autorange="reversed")
    )

    st.plotly_chart(fig, use_container_width=True)


def render_roc_curves(y_true: np.ndarray, y_probs: np.ndarray, class_names: List[str] = None):
    """Render ROC curves for multi-class classification."""
    n_classes = y_probs.shape[1]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fig = go.Figure()

    colors = px.colors.qualitative.Set1

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)

        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'{class_names[i]} (AUC={roc_auc:.3f})',
            line=dict(color=colors[i % len(colors)], width=2)
        ))

    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random',
        line=dict(color='gray', dash='dash')
    ))

    fig.update_layout(
        title="ROC Curves (One-vs-Rest)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
        legend=dict(x=1.02, y=0.5)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None):
    """Render per-class precision, recall, F1 breakdown."""
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

    n_classes = len(precision)
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    fig = go.Figure()

    x = np.arange(n_classes)
    width = 0.25

    fig.add_trace(go.Bar(
        x=class_names, y=precision,
        name='Precision',
        marker_color='#1f77b4'
    ))
    fig.add_trace(go.Bar(
        x=class_names, y=recall,
        name='Recall',
        marker_color='#ff7f0e'
    ))
    fig.add_trace(go.Bar(
        x=class_names, y=f1,
        name='F1-Score',
        marker_color='#2ca02c'
    ))

    fig.update_layout(
        title="Per-Class Metrics",
        xaxis_title="Class",
        yaxis_title="Score",
        barmode='group',
        height=400,
        yaxis=dict(range=[0, 1])
    )

    st.plotly_chart(fig, use_container_width=True)

    # Also show support (sample counts)
    st.caption(f"Sample counts per class: {dict(zip(class_names, support.astype(int)))}")


def render_confidence_distribution(y_probs: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
    """Render prediction confidence distribution."""
    max_probs = np.max(y_probs, axis=1)
    correct = y_true == y_pred

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Confidence Distribution", "Confidence vs Accuracy"])

    # Histogram of confidence
    fig.add_trace(
        go.Histogram(x=max_probs[correct], name='Correct', opacity=0.7, marker_color='green'),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=max_probs[~correct], name='Incorrect', opacity=0.7, marker_color='red'),
        row=1, col=1
    )

    # Calibration-like plot
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accuracies = []
    bin_confidences = []

    for i in range(len(bins) - 1):
        mask = (max_probs >= bins[i]) & (max_probs < bins[i + 1])
        if mask.sum() > 0:
            bin_accuracies.append(correct[mask].mean())
            bin_confidences.append(max_probs[mask].mean())
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append(np.nan)

    fig.add_trace(
        go.Scatter(x=bin_centers, y=bin_accuracies, name='Accuracy', mode='lines+markers',
                   marker=dict(size=10), line=dict(color='blue', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], name='Perfect Calibration', mode='lines',
                   line=dict(color='gray', dash='dash')),
        row=1, col=2
    )

    fig.update_layout(height=400, barmode='overlay')
    fig.update_xaxes(title_text="Confidence", row=1, col=1)
    fig.update_xaxes(title_text="Confidence", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)


def render_metrics_radar(metrics: Dict):
    """Render radar chart of metrics."""
    categories = ['Accuracy', 'F1 Micro', 'F1 Macro', 'F1 Weighted', '1-ECE', '1-Brier']

    values = [
        metrics.get('accuracy', 0),
        metrics.get('f1_micro', 0),
        metrics.get('f1_macro', 0),
        metrics.get('f1_weighted', 0),
        1 - metrics.get('ece', 0),  # Invert so higher is better
        1 - metrics.get('brier', 0)  # Invert so higher is better
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Model Performance',
        line_color='#1f77b4'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        height=400,
        title="Performance Radar"
    )

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# DATASET EXPLORER PAGE
# =============================================================================

def dataset_explorer_page():
    """
    Comprehensive Dataset Explorer with data processing pipeline.
    This is the first step - configure data processing here before training.
    """
    st.title("Dataset Explorer & Processing")
    st.caption("Configure data processing pipeline here. Settings will be applied during training.")
    st.markdown("---")

    # Initialize session state for processing config
    if 'processing_config' not in st.session_state:
        st.session_state.processing_config = {
            'normalization': 'standardization',
            'smoothing': 'none',
            'smoothing_window': 5,
            'detrending': 'none',
            'filtering': 'none',
            'filter_cutoff_low': 0.1,
            'filter_cutoff_high': 0.4,
            'outlier_handling': 'none',
            'outlier_threshold': 3.0,
            'resampling': 'none',
            'resample_factor': 1.0,
            'seq_len': 512,
        }

    # Dataset selection
    datasets = get_available_datasets()

    col1, col2 = st.columns([1, 2])
    with col1:
        dataset_name = st.selectbox(
            "Select Dataset",
            options=list(datasets.keys()),
            format_func=lambda x: x  # Display name is already clean
        )
        # Store selected dataset in session (display name)
        st.session_state.selected_dataset = dataset_name
        # Get folder ID for file operations
        dataset_folder = get_dataset_folder(dataset_name)

    with col2:
        description = get_dataset_description(dataset_name)
        if description:
            st.info(f"**{dataset_name}**: {description}")

    # Load dataset
    try:
        dataset_path = os.path.join(DATASET_DIR, dataset_folder)
        train_file = os.path.join(dataset_path, 'PdM_TRAIN.npz')

        if not os.path.exists(train_file):
            st.warning(f"Dataset files not found at {dataset_path}")
            return

        loaded = np.load(train_file, allow_pickle=True)
        df_restored = pd.DataFrame(loaded['data'], columns=loaded['columns'])

        # Extract features and labels
        features_raw = df_restored['features'].tolist()
        labels = df_restored['label'].values

        # Convert features to numpy arrays
        features_list = []
        for f in features_raw:
            if isinstance(f, np.ndarray):
                features_list.append(f)
            elif isinstance(f, list):
                features_list.append(np.array(f))
            else:
                features_list.append(np.array(f))

        # Get class names if available
        class_names = loaded.get('class_names', [f"Class {i}" for i in range(len(np.unique(labels)))])
        if hasattr(class_names, 'tolist'):
            class_names = class_names.tolist()

        st.success(f"Loaded {len(features_list)} samples with {len(np.unique(labels))} classes")

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        import traceback
        st.code(traceback.format_exc())
        return

    # Helper function to get feature count
    def get_n_features(sample):
        if len(sample.shape) > 1:
            return sample.shape[1]
        return 1

    def get_seq_len(sample):
        if len(sample.shape) > 1:
            return sample.shape[0]
        return len(sample)

    # Create tabs - Data Processing first since it determines training behavior
    tab_processing, tab_overview, tab_viewer, tab_distributions, tab_dimred = st.tabs([
        "⚙️ Data Processing", "📊 Overview", "📈 Signal Viewer",
        "📉 Distributions", "🎯 Embeddings"
    ])

    # =========================================================================
    # TAB: Data Processing (FIRST - determines training behavior)
    # =========================================================================
    with tab_processing:
        st.markdown("### Data Processing Pipeline")
        st.caption("Configure preprocessing steps. These will be applied during training.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Normalization")
            normalization = st.selectbox(
                "Normalization Method",
                options=['none', 'standardization', 'minmax', 'robust', 'per_sample'],
                format_func=lambda x: {
                    'none': 'None - Raw data',
                    'standardization': 'Z-Score (zero mean, unit variance)',
                    'minmax': 'Min-Max (scale to [0, 1])',
                    'robust': 'Robust (median/IQR based)',
                    'per_sample': 'Per-Sample Standardization'
                }[x],
                index=['none', 'standardization', 'minmax', 'robust', 'per_sample'].index(
                    st.session_state.processing_config.get('normalization', 'standardization')
                )
            )
            st.session_state.processing_config['normalization'] = normalization

            st.markdown("#### Smoothing")
            smoothing = st.selectbox(
                "Smoothing Method",
                options=['none', 'moving_average', 'savgol', 'exponential'],
                format_func=lambda x: {
                    'none': 'None',
                    'moving_average': 'Moving Average',
                    'savgol': 'Savitzky-Golay Filter',
                    'exponential': 'Exponential Smoothing'
                }[x]
            )
            st.session_state.processing_config['smoothing'] = smoothing

            if smoothing != 'none':
                smoothing_window = st.slider("Window Size", 3, 51, 5, step=2)
                st.session_state.processing_config['smoothing_window'] = smoothing_window

            st.markdown("#### Detrending")
            detrending = st.selectbox(
                "Detrending Method",
                options=['none', 'linear', 'constant', 'polynomial'],
                format_func=lambda x: {
                    'none': 'None',
                    'linear': 'Linear Detrend',
                    'constant': 'Remove Mean (DC offset)',
                    'polynomial': 'Polynomial Detrend'
                }[x]
            )
            st.session_state.processing_config['detrending'] = detrending

        with col2:
            st.markdown("#### Filtering")
            filtering = st.selectbox(
                "Frequency Filter",
                options=['none', 'lowpass', 'highpass', 'bandpass', 'bandstop'],
                format_func=lambda x: {
                    'none': 'None',
                    'lowpass': 'Low-Pass (remove high frequencies)',
                    'highpass': 'High-Pass (remove low frequencies)',
                    'bandpass': 'Band-Pass (keep frequency range)',
                    'bandstop': 'Band-Stop (notch filter)'
                }[x]
            )
            st.session_state.processing_config['filtering'] = filtering

            if filtering in ['lowpass', 'bandpass', 'bandstop']:
                filter_high = st.slider("High Cutoff (normalized)", 0.01, 0.99, 0.4, 0.01)
                st.session_state.processing_config['filter_cutoff_high'] = filter_high

            if filtering in ['highpass', 'bandpass', 'bandstop']:
                filter_low = st.slider("Low Cutoff (normalized)", 0.01, 0.99, 0.1, 0.01)
                st.session_state.processing_config['filter_cutoff_low'] = filter_low

            st.markdown("#### Outlier Handling")
            outlier_handling = st.selectbox(
                "Outlier Method",
                options=['none', 'clip', 'winsorize', 'iqr_remove'],
                format_func=lambda x: {
                    'none': 'None',
                    'clip': 'Clip to threshold',
                    'winsorize': 'Winsorize (percentile-based)',
                    'iqr_remove': 'Remove using IQR'
                }[x]
            )
            st.session_state.processing_config['outlier_handling'] = outlier_handling

            if outlier_handling in ['clip', 'iqr_remove']:
                outlier_threshold = st.slider("Threshold (std deviations)", 1.0, 5.0, 3.0, 0.5)
                st.session_state.processing_config['outlier_threshold'] = outlier_threshold

            st.markdown("#### Resampling")
            resampling = st.selectbox(
                "Resampling",
                options=['none', 'downsample', 'upsample'],
                format_func=lambda x: {
                    'none': 'None - Keep original',
                    'downsample': 'Downsample (reduce points)',
                    'upsample': 'Upsample (interpolate)'
                }[x]
            )
            st.session_state.processing_config['resampling'] = resampling

            if resampling != 'none':
                resample_factor = st.slider(
                    "Resample Factor",
                    0.1 if resampling == 'downsample' else 1.0,
                    1.0 if resampling == 'downsample' else 4.0,
                    0.5 if resampling == 'downsample' else 2.0,
                    0.1
                )
                st.session_state.processing_config['resample_factor'] = resample_factor

        # Sequence length
        st.markdown("---")
        st.markdown("#### Sequence Length")
        seq_lengths = [get_seq_len(f) for f in features_list]
        max_seq = max(seq_lengths)
        target_seq_len = st.slider(
            "Target Sequence Length",
            min_value=64,
            max_value=min(4096, max_seq),
            value=min(512, max_seq),
            step=64,
            help="Sequences will be truncated or padded to this length"
        )
        st.session_state.processing_config['seq_len'] = target_seq_len

        # Processing summary
        st.markdown("---")
        st.markdown("### Processing Summary")
        active_steps = []
        if normalization != 'none':
            active_steps.append(f"Normalization: {normalization}")
        if smoothing != 'none':
            active_steps.append(f"Smoothing: {smoothing} (window={st.session_state.processing_config.get('smoothing_window', 5)})")
        if detrending != 'none':
            active_steps.append(f"Detrending: {detrending}")
        if filtering != 'none':
            active_steps.append(f"Filtering: {filtering}")
        if outlier_handling != 'none':
            active_steps.append(f"Outliers: {outlier_handling}")
        if resampling != 'none':
            active_steps.append(f"Resampling: {resampling} (factor={st.session_state.processing_config.get('resample_factor', 1.0)})")
        active_steps.append(f"Sequence Length: {target_seq_len}")

        if active_steps:
            for step in active_steps:
                st.text(f"  • {step}")
        else:
            st.text("  • No processing (raw data)")

        st.success("Processing configuration saved. Go to 'Model Training' to train with these settings.")

        # =====================================================================
        # Preview: Raw vs Processed Data Comparison
        # =====================================================================
        st.markdown("---")
        st.markdown("### Preview: Raw vs Processed")
        st.caption("See how processing affects your data before training")

        preview_col1, preview_col2 = st.columns([1, 3])

        with preview_col1:
            preview_class = st.selectbox(
                "Preview Class",
                options=list(range(len(np.unique(labels)))),
                format_func=lambda x: class_names[x] if x < len(class_names) else f"Class {x}",
                key="preview_class"
            )
            preview_indices = np.where(labels == preview_class)[0]
            preview_sample_idx = st.selectbox(
                "Preview Sample",
                options=list(range(min(10, len(preview_indices)))),
                format_func=lambda x: f"Sample {x + 1}",
                key="preview_sample"
            )

        # Get raw sample
        raw_sample = features_list[preview_indices[preview_sample_idx]]
        if len(raw_sample.shape) > 1:
            raw_signal = raw_sample[:, 0]  # First feature
        else:
            raw_signal = raw_sample.flatten()

        # Apply processing to get processed sample
        def apply_processing(signal, config):
            """Apply processing pipeline to a signal."""
            processed = signal.copy().astype(np.float64)

            # Normalization
            norm = config.get('normalization', 'none')
            if norm == 'standardization':
                mean, std = np.mean(processed), np.std(processed)
                if std > 0:
                    processed = (processed - mean) / std
            elif norm == 'minmax':
                min_val, max_val = np.min(processed), np.max(processed)
                if max_val > min_val:
                    processed = (processed - min_val) / (max_val - min_val)
            elif norm == 'robust':
                median = np.median(processed)
                q75, q25 = np.percentile(processed, [75, 25])
                iqr = q75 - q25
                if iqr > 0:
                    processed = (processed - median) / iqr

            # Smoothing
            smooth = config.get('smoothing', 'none')
            window = config.get('smoothing_window', 5)
            if smooth == 'moving_average' and len(processed) > window:
                kernel = np.ones(window) / window
                processed = np.convolve(processed, kernel, mode='same')
            elif smooth == 'savgol' and len(processed) > window:
                try:
                    from scipy.signal import savgol_filter
                    processed = savgol_filter(processed, window, 3)
                except:
                    pass
            elif smooth == 'exponential':
                alpha = 2 / (window + 1)
                result = np.zeros_like(processed)
                result[0] = processed[0]
                for i in range(1, len(processed)):
                    result[i] = alpha * processed[i] + (1 - alpha) * result[i-1]
                processed = result

            # Detrending
            detrend = config.get('detrending', 'none')
            if detrend == 'constant':
                processed = processed - np.mean(processed)
            elif detrend == 'linear':
                from scipy.signal import detrend as scipy_detrend
                processed = scipy_detrend(processed, type='linear')

            # Filtering
            filt = config.get('filtering', 'none')
            if filt != 'none' and len(processed) > 10:
                try:
                    from scipy.signal import butter, filtfilt
                    order = 4
                    if filt == 'lowpass':
                        b, a = butter(order, config.get('filter_cutoff_high', 0.4), btype='low')
                    elif filt == 'highpass':
                        b, a = butter(order, config.get('filter_cutoff_low', 0.1), btype='high')
                    elif filt == 'bandpass':
                        b, a = butter(order, [config.get('filter_cutoff_low', 0.1),
                                              config.get('filter_cutoff_high', 0.4)], btype='band')
                    elif filt == 'bandstop':
                        b, a = butter(order, [config.get('filter_cutoff_low', 0.1),
                                              config.get('filter_cutoff_high', 0.4)], btype='bandstop')
                    processed = filtfilt(b, a, processed)
                except:
                    pass

            # Outlier handling
            outlier = config.get('outlier_handling', 'none')
            threshold = config.get('outlier_threshold', 3.0)
            if outlier == 'clip':
                mean, std = np.mean(processed), np.std(processed)
                processed = np.clip(processed, mean - threshold * std, mean + threshold * std)
            elif outlier == 'winsorize':
                lower = np.percentile(processed, 5)
                upper = np.percentile(processed, 95)
                processed = np.clip(processed, lower, upper)

            # Truncate/pad to target length
            target_len = config.get('seq_len', 512)
            if len(processed) > target_len:
                processed = processed[:target_len]
            elif len(processed) < target_len:
                processed = np.pad(processed, (0, target_len - len(processed)), mode='constant')

            return processed

        processed_signal = apply_processing(raw_signal, st.session_state.processing_config)

        # Create comparison plots - Time domain
        st.markdown("#### Time Domain Comparison")
        time_fig = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Raw Signal", "Processed Signal"),
                                 horizontal_spacing=0.1)
        time_fig.add_trace(go.Scatter(y=raw_signal, mode='lines', name='Raw',
                                      line=dict(color='#636EFA')), row=1, col=1)
        time_fig.add_trace(go.Scatter(y=processed_signal, mode='lines', name='Processed',
                                      line=dict(color='#00CC96')), row=1, col=2)
        time_fig.update_layout(height=250, showlegend=False, margin=dict(t=30, b=30))
        time_fig.update_xaxes(title_text="Time Step")
        time_fig.update_yaxes(title_text="Amplitude")
        st.plotly_chart(time_fig, use_container_width=True)

        # Frequency domain
        st.markdown("#### Frequency Domain Comparison")
        raw_fft = np.abs(np.fft.rfft(raw_signal)) / len(raw_signal)
        raw_freqs = np.fft.rfftfreq(len(raw_signal))
        proc_fft = np.abs(np.fft.rfft(processed_signal)) / len(processed_signal)
        proc_freqs = np.fft.rfftfreq(len(processed_signal))

        freq_fig = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Raw Spectrum", "Processed Spectrum"),
                                 horizontal_spacing=0.1)
        freq_fig.add_trace(go.Scatter(x=raw_freqs, y=raw_fft, mode='lines', name='Raw FFT',
                                      line=dict(color='#636EFA'), fill='tozeroy'), row=1, col=1)
        freq_fig.add_trace(go.Scatter(x=proc_freqs, y=proc_fft, mode='lines', name='Processed FFT',
                                      line=dict(color='#00CC96'), fill='tozeroy'), row=1, col=2)
        freq_fig.update_layout(height=250, showlegend=False, margin=dict(t=30, b=30))
        freq_fig.update_xaxes(title_text="Normalized Frequency")
        freq_fig.update_yaxes(title_text="Magnitude")
        st.plotly_chart(freq_fig, use_container_width=True)

        # Statistics comparison in separate container
        st.markdown("#### Statistics Comparison")
        stats_cols = st.columns(2)
        with stats_cols[0]:
            st.markdown("**Raw Signal:**")
            st.text(f"Mean: {np.mean(raw_signal):.4f}  |  Std: {np.std(raw_signal):.4f}")
            st.text(f"Min: {np.min(raw_signal):.4f}  |  Max: {np.max(raw_signal):.4f}")
            st.text(f"Length: {len(raw_signal)}")

        with stats_cols[1]:
            st.markdown("**Processed Signal:**")
            st.text(f"Mean: {np.mean(processed_signal):.4f}  |  Std: {np.std(processed_signal):.4f}")
            st.text(f"Min: {np.min(processed_signal):.4f}  |  Max: {np.max(processed_signal):.4f}")
            st.text(f"Length: {len(processed_signal)}")

    # =========================================================================
    # TAB: Overview
    # =========================================================================
    with tab_overview:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Samples", len(features_list))
            st.metric("Unique Classes", len(np.unique(labels)))

        with col2:
            seq_lengths = [get_seq_len(f) for f in features_list]
            st.metric("Avg Sequence Length", f"{np.mean(seq_lengths):.0f}")
            st.metric("Sequence Length Range", f"{min(seq_lengths)} - {max(seq_lengths)}")

        with col3:
            n_features = get_n_features(features_list[0])
            st.metric("Number of Features", n_features)

        # Class distribution
        st.markdown("### Class Distribution")
        class_counts = pd.Series(labels).value_counts().sort_index()

        fig = px.pie(
            values=class_counts.values,
            names=[class_names[i] if i < len(class_names) else f"Class {i}" for i in class_counts.index],
            title="Class Distribution"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Show class balance statistics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Class Counts:**")
            for idx, count in class_counts.items():
                class_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
                pct = count / len(labels) * 100
                st.text(f"  {class_name}: {count} ({pct:.1f}%)")

        with col2:
            # Imbalance ratio
            max_count = class_counts.max()
            min_count = class_counts.min()
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            st.metric("Imbalance Ratio", f"{imbalance_ratio:.2f}x")

    # =========================================================================
    # TAB: Signal Viewer (Time Domain + Frequency Domain)
    # =========================================================================
    with tab_viewer:
        st.markdown("### Sample Time Series Viewer")

        col1, col2, col3 = st.columns(3)

        with col1:
            selected_class = st.selectbox(
                "Select Class",
                options=list(range(len(np.unique(labels)))),
                format_func=lambda x: class_names[x] if x < len(class_names) else f"Class {x}"
            )

        with col2:
            class_indices = np.where(labels == selected_class)[0]
            sample_idx = st.selectbox(
                "Select Sample",
                options=list(range(len(class_indices))),
                format_func=lambda x: f"Sample {x + 1} (idx: {class_indices[x]})"
            )

        with col3:
            n_features = get_n_features(features_list[0])
            if n_features > 1:
                feature_idx = st.selectbox(
                    "Select Feature",
                    options=['All'] + list(range(n_features)),
                    format_func=lambda x: 'All Features' if x == 'All' else f"Feature {x}"
                )
            else:
                feature_idx = 0

        # Plot time series
        sample = features_list[class_indices[sample_idx]]
        sample_n_features = get_n_features(sample)

        fig = go.Figure()

        if feature_idx == 'All' and sample_n_features > 1:
            for i in range(sample_n_features):
                fig.add_trace(go.Scatter(
                    y=sample[:, i],
                    name=f"Feature {i}",
                    mode='lines'
                ))
        else:
            if sample_n_features > 1:
                y_data = sample[:, feature_idx if feature_idx != 'All' else 0]
            else:
                y_data = sample.flatten() if len(sample.shape) > 1 else sample
            fig.add_trace(go.Scatter(
                y=y_data,
                name=f"Feature {feature_idx}",
                mode='lines',
                line=dict(color='#1f77b4')
            ))

        fig.update_layout(
            title=f"Time Series - {class_names[selected_class] if selected_class < len(class_names) else f'Class {selected_class}'} (Sample {sample_idx + 1})",
            xaxis_title="Time Step",
            yaxis_title="Value",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Multi-perspective view: Time Domain + Frequency Domain
        st.markdown("---")
        st.markdown("### Multi-Perspective Analysis")

        view_col1, view_col2 = st.columns(2)

        # Get the sample data for analysis
        if sample_n_features > 1 and feature_idx != 'All':
            analysis_signal = sample[:, feature_idx]
        elif sample_n_features > 1:
            analysis_signal = sample[:, 0]  # Use first feature
        else:
            analysis_signal = sample.flatten() if len(sample.shape) > 1 else sample

        with view_col1:
            st.markdown("#### Time Domain")
            # Time domain statistics
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                y=analysis_signal,
                mode='lines',
                name='Signal',
                line=dict(color='#1f77b4')
            ))

            # Add envelope if signal is long enough
            if len(analysis_signal) > 20:
                from scipy.signal import hilbert
                try:
                    analytic = hilbert(analysis_signal)
                    envelope = np.abs(analytic)
                    fig_time.add_trace(go.Scatter(
                        y=envelope,
                        mode='lines',
                        name='Envelope',
                        line=dict(color='#ff7f0e', dash='dash')
                    ))
                except:
                    pass

            fig_time.update_layout(
                title="Time Domain Signal",
                xaxis_title="Time Step",
                yaxis_title="Amplitude",
                height=300,
                showlegend=True
            )
            st.plotly_chart(fig_time, use_container_width=True)

            # Time domain statistics
            st.markdown("**Statistics:**")
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.text(f"Mean: {np.mean(analysis_signal):.4f}")
                st.text(f"Std: {np.std(analysis_signal):.4f}")
                st.text(f"RMS: {np.sqrt(np.mean(analysis_signal**2)):.4f}")
            with stats_col2:
                st.text(f"Peak: {np.max(np.abs(analysis_signal)):.4f}")
                st.text(f"Crest Factor: {np.max(np.abs(analysis_signal))/np.sqrt(np.mean(analysis_signal**2)):.4f}")
                st.text(f"Kurtosis: {float(pd.Series(analysis_signal).kurtosis()):.4f}")

        with view_col2:
            st.markdown("#### Frequency Domain")
            # FFT analysis
            n = len(analysis_signal)
            fft_vals = np.fft.rfft(analysis_signal)
            fft_freqs = np.fft.rfftfreq(n)
            fft_magnitude = np.abs(fft_vals) / n

            fig_freq = go.Figure()
            fig_freq.add_trace(go.Scatter(
                x=fft_freqs,
                y=fft_magnitude,
                mode='lines',
                name='Magnitude',
                fill='tozeroy',
                line=dict(color='#2ca02c')
            ))

            fig_freq.update_layout(
                title="Frequency Spectrum (FFT)",
                xaxis_title="Normalized Frequency",
                yaxis_title="Magnitude",
                height=300
            )
            st.plotly_chart(fig_freq, use_container_width=True)

            # Spectral statistics
            st.markdown("**Spectral Features:**")
            # Find dominant frequencies
            top_k = 5
            top_indices = np.argsort(fft_magnitude)[-top_k:][::-1]
            st.text("Top frequencies (normalized):")
            for idx in top_indices[:3]:
                st.text(f"  • {fft_freqs[idx]:.4f} (mag: {fft_magnitude[idx]:.4f})")

            # Spectral centroid
            spectral_centroid = np.sum(fft_freqs * fft_magnitude) / np.sum(fft_magnitude) if np.sum(fft_magnitude) > 0 else 0
            st.text(f"Spectral Centroid: {spectral_centroid:.4f}")

        # Power Spectral Density
        st.markdown("---")
        st.markdown("#### Power Spectral Density")
        try:
            from scipy.signal import welch
            freqs_psd, psd = welch(analysis_signal, nperseg=min(256, len(analysis_signal)//4))

            fig_psd = go.Figure()
            fig_psd.add_trace(go.Scatter(
                x=freqs_psd,
                y=10 * np.log10(psd + 1e-10),  # dB scale
                mode='lines',
                name='PSD',
                line=dict(color='#9467bd')
            ))
            fig_psd.update_layout(
                title="Power Spectral Density (Welch)",
                xaxis_title="Normalized Frequency",
                yaxis_title="Power (dB)",
                height=250
            )
            st.plotly_chart(fig_psd, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute PSD: {e}")

        st.markdown("---")

        # Show multiple samples comparison
        if st.checkbox("Compare Multiple Samples"):
            n_compare = st.slider("Number of samples to compare", 2, 10, 5)

            fig = make_subplots(rows=n_compare, cols=1, shared_xaxes=True,
                               vertical_spacing=0.02)

            for i in range(min(n_compare, len(class_indices))):
                sample = features_list[class_indices[i]]
                if get_n_features(sample) > 1:
                    y_data = sample[:, 0]  # First feature
                else:
                    y_data = sample.flatten() if len(sample.shape) > 1 else sample

                fig.add_trace(
                    go.Scatter(y=y_data, name=f"Sample {i+1}", mode='lines'),
                    row=i+1, col=1
                )

            fig.update_layout(height=150 * n_compare, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # TAB: Feature Distributions
    # =========================================================================
    with tab_distributions:
        st.markdown("### Feature Value Distributions")

        # Compute global statistics
        all_data = []
        for f in features_list[:1000]:  # Limit for performance
            f_arr = np.array(f) if not isinstance(f, np.ndarray) else f
            if len(f_arr.shape) > 1:
                all_data.append(f_arr)
            else:
                all_data.append(f_arr.reshape(-1, 1))

        # Stack all data
        try:
            all_data = np.vstack(all_data)
        except ValueError:
            # Handle variable-length sequences by padding
            max_len = max(d.shape[0] for d in all_data)
            n_feat = all_data[0].shape[1]
            padded = []
            for d in all_data:
                if d.shape[0] < max_len:
                    pad = np.zeros((max_len - d.shape[0], n_feat))
                    d = np.vstack([d, pad])
                padded.append(d)
            all_data = np.vstack(padded)

        n_features = all_data.shape[1]

        col1, col2 = st.columns([1, 3])

        with col1:
            selected_feature = st.selectbox(
                "Select Feature",
                options=list(range(n_features)),
                format_func=lambda x: f"Feature {x}"
            )

            show_by_class = st.checkbox("Show by Class", value=True)

        with col2:
            if show_by_class:
                # Compute per-class distributions
                fig = go.Figure()

                for cls in np.unique(labels)[:10]:  # Limit classes shown
                    cls_indices = np.where(labels == cls)[0][:100]  # Limit samples
                    cls_data = []
                    for idx in cls_indices:
                        f = features_list[idx]
                        f_arr = np.array(f) if not isinstance(f, np.ndarray) else f
                        if len(f_arr.shape) > 1 and f_arr.shape[1] > selected_feature:
                            cls_data.extend(f_arr[:, selected_feature].tolist())
                        else:
                            cls_data.extend(f_arr.flatten().tolist())

                    class_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
                    fig.add_trace(go.Histogram(
                        x=cls_data,
                        name=class_name,
                        opacity=0.6
                    ))

                fig.update_layout(
                    title=f"Feature {selected_feature} Distribution by Class",
                    barmode='overlay',
                    height=400
                )
            else:
                fig = go.Figure(data=[go.Histogram(
                    x=all_data[:, selected_feature],
                    name=f"Feature {selected_feature}"
                )])
                fig.update_layout(
                    title=f"Feature {selected_feature} Distribution",
                    height=400
                )

            st.plotly_chart(fig, use_container_width=True)

        # Statistics table
        st.markdown("### Feature Statistics")
        stats_data = []
        for i in range(n_features):
            stats_data.append({
                'Feature': f"Feature {i}",
                'Mean': f"{np.mean(all_data[:, i]):.4f}",
                'Std': f"{np.std(all_data[:, i]):.4f}",
                'Min': f"{np.min(all_data[:, i]):.4f}",
                'Max': f"{np.max(all_data[:, i]):.4f}",
                'Median': f"{np.median(all_data[:, i]):.4f}"
            })

        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB: Embeddings / Dimensionality Reduction
    # =========================================================================
    with tab_dimred:
        st.markdown("### Dimensionality Reduction")

        col1, col2 = st.columns(2)

        with col1:
            method = st.selectbox(
                "Method",
                options=['PCA', 't-SNE', 'UMAP'],
                help="Dimensionality reduction method"
            )

        with col2:
            n_samples = st.slider(
                "Number of samples",
                100, min(2000, len(features_list)), 500,
                help="More samples = better visualization but slower"
            )

        if st.button("Compute Embedding", type="primary"):
            with st.spinner(f"Computing {method} embedding..."):
                # Prepare data - use mean of time series as features
                sample_indices = np.random.choice(len(features_list), min(n_samples, len(features_list)), replace=False)

                X_flat = []
                y_subset = []
                for idx in sample_indices:
                    f = features_list[idx]
                    if len(f.shape) > 1:
                        # Use statistical features: mean, std, min, max per channel
                        feat = np.concatenate([
                            np.mean(f, axis=0),
                            np.std(f, axis=0),
                            np.min(f, axis=0),
                            np.max(f, axis=0)
                        ])
                    else:
                        feat = np.array([np.mean(f), np.std(f), np.min(f), np.max(f)])
                    X_flat.append(feat)
                    y_subset.append(labels[idx])

                X_flat = np.array(X_flat)
                y_subset = np.array(y_subset)

                # Apply dimensionality reduction
                try:
                    if method == 'PCA':
                        from sklearn.decomposition import PCA
                        reducer = PCA(n_components=2)
                        embedding = reducer.fit_transform(X_flat)
                        explained_var = reducer.explained_variance_ratio_
                        st.caption(f"Explained variance: {explained_var[0]:.2%} + {explained_var[1]:.2%} = {sum(explained_var):.2%}")

                    elif method == 't-SNE':
                        from sklearn.manifold import TSNE
                        perplexity = min(30, n_samples - 1)
                        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                        embedding = reducer.fit_transform(X_flat)

                    elif method == 'UMAP':
                        try:
                            import umap
                            reducer = umap.UMAP(n_components=2, random_state=42)
                            embedding = reducer.fit_transform(X_flat)
                        except ImportError:
                            st.error("UMAP not installed. Install with: pip install umap-learn")
                            return

                    # Create visualization
                    df_embed = pd.DataFrame({
                        'x': embedding[:, 0],
                        'y': embedding[:, 1],
                        'class': [class_names[c] if c < len(class_names) else f"Class {c}" for c in y_subset]
                    })

                    fig = px.scatter(
                        df_embed,
                        x='x', y='y',
                        color='class',
                        title=f"{method} Visualization",
                        height=600
                    )

                    fig.update_traces(marker=dict(size=8, opacity=0.7))
                    fig.update_layout(
                        xaxis_title=f"{method} 1",
                        yaxis_title=f"{method} 2"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error computing embedding: {e}")


# =============================================================================
# EXPERIMENT HISTORY
# =============================================================================

def render_experiment_history():
    """Render experiment history and comparison."""
    st.subheader("Experiment History")

    tracker = get_tracker()
    runs = tracker.get_run_summary()

    if not runs:
        st.info("No training runs recorded yet. Train a model to see history here.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(runs)

    # Format columns
    if 'best_val_accuracy' in df.columns:
        df['best_val_accuracy'] = df['best_val_accuracy'].apply(lambda x: f"{x:.2%}")

    # Display table
    st.dataframe(
        df[['run_id', 'model', 'dataset', 'status', 'best_val_accuracy', 'total_epochs']],
        use_container_width=True,
        hide_index=True
    )


# =============================================================================
# CUSTOM MODEL UPLOAD PAGE
# =============================================================================

def custom_model_page():
    """Page for uploading and managing custom models."""
    st.title("Custom Model Upload")
    st.markdown("---")

    st.markdown("""
    Upload your own PyTorch model to evaluate on PDMBench datasets.

    **Requirements:**
    - Python file containing a class that inherits from `torch.nn.Module`
    - Model must accept input of shape `[batch_size, sequence_length, n_features]`
    - Model must output shape `[batch_size, num_classes]` for classification
    """)

    # Model template
    with st.expander("📄 Model Template", expanded=False):
        template_code = '''import torch
import torch.nn as nn

class MyCustomModel(nn.Module):
    """
    Custom model template for PDMBench.

    Args:
        n_features: Number of input features (channels)
        n_classes: Number of output classes
    """

    def __init__(self, n_features, n_classes):
        super().__init__()

        # Example: Simple LSTM-based classifier
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_features]

        Returns:
            Output tensor of shape [batch_size, n_classes]
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        out = self.fc(h_n[-1])

        return out
'''
        st.code(template_code, language='python')

        # Download template
        st.download_button(
            "Download Template",
            template_code,
            file_name="custom_model_template.py",
            mime="text/x-python"
        )

    st.markdown("---")

    # File upload
    st.markdown("### Upload Model")

    uploaded_file = st.file_uploader(
        "Choose a Python file",
        type=['py'],
        help="Upload a .py file containing your model class"
    )

    if uploaded_file is not None:
        # Read file content
        file_content = uploaded_file.read().decode('utf-8')

        # Show file preview
        with st.expander("📄 File Preview", expanded=True):
            st.code(file_content, language='python')

        col1, col2 = st.columns(2)

        with col1:
            class_name = st.text_input(
                "Model Class Name (optional)",
                help="Leave empty to auto-detect"
            )

        with col2:
            model_name = st.text_input(
                "Model Name",
                value=uploaded_file.name.replace('.py', ''),
                help="Name to identify this model"
            )

        if st.button("Validate Model", type="primary"):
            with st.spinner("Validating model..."):
                # Try to load the model
                model_class, error = load_custom_model(file_content, class_name)

                if model_class is None:
                    st.error(f"Failed to load model: {error}")
                else:
                    # Validate with dummy data
                    is_valid, message = validate_custom_model(model_class)

                    if is_valid:
                        st.success(f"✅ {message}")

                        # Save model to custom models directory
                        save_path = os.path.join(CUSTOM_MODELS_DIR, f"{model_name}.py")
                        with open(save_path, 'w') as f:
                            f.write(file_content)

                        # Store in session state
                        if 'custom_models' not in st.session_state:
                            st.session_state.custom_models = {}

                        st.session_state.custom_models[model_name] = {
                            'class': model_class,
                            'file_path': save_path,
                            'class_name': model_class.__name__
                        }

                        st.success(f"Model saved as '{model_name}'. You can now select it in the Model Training page.")
                    else:
                        st.error(f"❌ {message}")

    st.markdown("---")

    # List saved custom models
    st.markdown("### Saved Custom Models")

    custom_models = st.session_state.get('custom_models', {})

    if custom_models:
        for name, info in custom_models.items():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.text(f"📦 {name} ({info['class_name']})")
            with col2:
                st.text(f"Path: {info['file_path']}")
            with col3:
                if st.button("Remove", key=f"remove_{name}"):
                    del st.session_state.custom_models[name]
                    st.rerun()
    else:
        st.info("No custom models saved yet. Upload a model above to get started.")


# =============================================================================
# MAIN MODEL TRAINING PAGE
# =============================================================================

def model_training_page():
    """Model training page with model-specific hyperparameters and custom model support."""
    st.title("Model Training")

    # Initialize session state
    if 'training_active' not in st.session_state:
        st.session_state.training_active = False
    if 'training_history' not in st.session_state:
        st.session_state.training_history = []

    # Get processing config from Dataset Explorer
    processing_config = st.session_state.get('processing_config', {})
    selected_dataset = st.session_state.get('selected_dataset', None)
    seq_len = processing_config.get('seq_len', 512)

    datasets = get_available_datasets()
    custom_models = st.session_state.get('custom_models', {})

    # Create tabs for configuration
    config_tab, generalization_tab, custom_tab = st.tabs([
        "Training Configuration",
        "Cross-Condition Evaluation",
        "Custom Model (Optional)"
    ])

    # =========================================================================
    # TAB 1: Training Configuration
    # =========================================================================
    with config_tab:
        # Top row: Dataset and Model selection
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Dataset")
            default_idx = 0
            if selected_dataset and selected_dataset in list(datasets.keys()):
                default_idx = list(datasets.keys()).index(selected_dataset)

            dataset_name = st.selectbox(
                "Select Dataset",
                options=list(datasets.keys()),
                format_func=lambda x: x,  # Display name is already clean
                index=default_idx
            )
            # Get folder ID for file operations
            dataset_folder = get_dataset_folder(dataset_name)

            # Show processing config
            if processing_config:
                active = []
                if processing_config.get('normalization', 'none') != 'none':
                    active.append(processing_config['normalization'])
                if processing_config.get('filtering', 'none') != 'none':
                    active.append(processing_config['filtering'])
                if active:
                    st.caption(f"Processing: {', '.join(active)} | Seq: {seq_len}")
                else:
                    st.caption(f"Sequence length: {seq_len}")

        with col2:
            st.subheader("Model")
            available_models = model_list.copy()
            if custom_models:
                available_models = model_list + ['---'] + list(custom_models.keys())

            model_type = st.selectbox(
                "Select Model",
                options=[m for m in available_models if m != '---'],
                index=0
            )
            is_custom_model = model_type in custom_models

            # Show foundation model info
            if model_type in FOUNDATION_MODELS:
                fm_info = FOUNDATION_MODELS[model_type]
                st.info(f"🔬 **Foundation Model**: {fm_info['description']}")

        # Task Type Selection
        st.markdown("---")
        task_col1, task_col2 = st.columns(2)

        with task_col1:
            task_type = st.selectbox(
                "Task Type",
                options=list(TASK_TYPES.keys()),
                format_func=lambda x: TASK_TYPES[x],
                index=0,
                help="Select the prediction task type"
            )

        with task_col2:
            if task_type == 'early_failure':
                failure_mode = st.selectbox(
                    "Failure Prediction Mode",
                    options=['classification', 'rul', 'hazard'],
                    format_func=lambda x: {'classification': 'Binary Classification', 'rul': 'RUL Regression', 'hazard': 'Hazard Probability'}[x],
                    help="How to predict failures"
                )
            else:
                failure_mode = 'classification'

        st.markdown("---")

        # Three columns for parameters
        train_col, arch_col, model_spec_col = st.columns(3)

        # Training Parameters
        with train_col:
            st.subheader("Training")
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64, 128], index=2)
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                value=1e-3,
                format_func=lambda x: f"{x:.0e}"
            )
            train_epochs = st.slider("Epochs", 5, 200, 50, 5)
            optimizer = st.selectbox("Optimizer", ['AdamW', 'Adam', 'RAdam', 'SGD'], index=0)
            scheduler = st.selectbox("LR Scheduler", ['cosine', 'step', 'plateau', 'none'], index=0)
            patience = st.slider("Early Stop Patience", 3, 20, 7)

        # Architecture Parameters
        with arch_col:
            st.subheader("Architecture")
            d_model = st.selectbox("Model Dim (d_model)", [32, 64, 128, 256, 512], index=2)
            d_ff = st.selectbox("FFN Dim (d_ff)", [64, 128, 256, 512, 1024, 2048], index=3)
            n_heads = st.selectbox("Attention Heads", [1, 2, 4, 8], index=2)
            e_layers = st.slider("Encoder Layers", 1, 6, 2)
            d_layers = st.slider("Decoder Layers", 1, 4, 1)
            dropout = st.slider("Dropout", 0.0, 0.5, 0.1, 0.05)

        # Model-Specific Parameters
        with model_spec_col:
            st.subheader("Model-Specific")

            # Initialize default values for all model-specific params
            top_k = 5
            num_kernels = 6
            patch_len = 16
            stride = 8
            moving_avg = 25
            factor = 1
            individual = False
            seg_len = 48
            expand = 2
            d_conv = 4

            # Show different params based on model type
            if model_type in ['TimesNet']:
                st.caption("TimesNet Parameters")
                top_k = st.slider("Top K (frequencies)", 1, 10, 5)
                num_kernels = st.slider("Num Kernels", 2, 10, 6)
            elif model_type in ['PatchTST', 'TimeXer']:
                st.caption("Patch Parameters")
                patch_len = st.selectbox("Patch Length", [8, 16, 32, 64], index=1)
                stride = st.selectbox("Stride", [4, 8, 16, 32], index=1)
            elif model_type in ['Autoformer', 'FEDformer']:
                st.caption("Decomposition Parameters")
                moving_avg = st.selectbox("Moving Avg Window", [13, 25, 49], index=1)
                factor = st.slider("Factor", 1, 5, 1)
            elif model_type in ['DLinear', 'MLP']:
                st.caption("Linear Parameters")
                individual = st.checkbox("Individual (channel-wise)", value=False)
            elif model_type in ['SegRNN']:
                st.caption("Segment Parameters")
            elif model_type in FOUNDATION_MODELS:
                st.caption(f"{model_type} Parameters")
                fm_info = FOUNDATION_MODELS[model_type]
                fm_size = st.selectbox(
                    "Model Size",
                    options=fm_info['sizes'],
                    index=fm_info['sizes'].index(fm_info['default_size'])
                )
                freeze_backbone = st.checkbox(
                    "Freeze Backbone",
                    value=True,
                    help="Freeze pre-trained weights (linear probing)"
                )
                seg_len = st.selectbox("Segment Length", [24, 48, 96, 192], index=1)
            elif model_type in ['MambaSimple', 'Mamba']:
                st.caption("Mamba Parameters")
                expand = st.slider("Expand Factor", 1, 4, 2)
                d_conv = st.selectbox("Conv Dimension", [2, 4, 8], index=1)
            elif model_type in ['Crossformer']:
                st.caption("Cross Parameters")
                seg_len = st.selectbox("Segment Length", [12, 24, 48], index=1)
            else:
                st.caption("Using default parameters")
                st.text("No model-specific params")

            st.markdown("---")
            st.caption("Compute")
            use_gpu = st.checkbox("Use GPU", value=False)
            if use_gpu and not torch.cuda.is_available():
                st.warning("CUDA unavailable")
                use_gpu = False
            gpu_id = 0
            if use_gpu and torch.cuda.is_available():
                gpu_id = st.selectbox("GPU ID", list(range(torch.cuda.device_count())))

    # =========================================================================
    # TAB 2: Cross-Condition Evaluation (Generalization)
    # =========================================================================
    with generalization_tab:
        st.markdown("### Cross-Condition Evaluation")
        st.markdown("""
        Test model generalization by training on one operating condition and evaluating on a different one.
        This simulates real-world scenarios where models must handle distribution shifts.
        """)

        # Operating conditions for known datasets (from official documentation)
        DATASET_CONDITIONS = {
            "CWRU": {
                "description": "Case Western Reserve University - Motor load conditions",
                "condition_type": "Motor Load & Speed",
                "conditions": {
                    "0HP_1797RPM": {"load_hp": 0, "speed_rpm": 1797, "label": "0 HP (1797 RPM)"},
                    "1HP_1772RPM": {"load_hp": 1, "speed_rpm": 1772, "label": "1 HP (1772 RPM)"},
                    "2HP_1750RPM": {"load_hp": 2, "speed_rpm": 1750, "label": "2 HP (1750 RPM)"},
                    "3HP_1730RPM": {"load_hp": 3, "speed_rpm": 1730, "label": "3 HP (1730 RPM)"},
                },
            },
            "Paderborn": {
                "description": "Paderborn University - Speed, torque, and radial force",
                "condition_type": "Speed/Torque/Force",
                "conditions": {
                    "N15_M07_F10": {"speed_rpm": 1500, "torque_nm": 0.7, "force_n": 1000, "label": "1500rpm, 0.7Nm, 1000N (Baseline)"},
                    "N09_M07_F10": {"speed_rpm": 900, "torque_nm": 0.7, "force_n": 1000, "label": "900rpm, 0.7Nm, 1000N"},
                    "N15_M01_F10": {"speed_rpm": 1500, "torque_nm": 0.1, "force_n": 1000, "label": "1500rpm, 0.1Nm, 1000N"},
                    "N15_M07_F04": {"speed_rpm": 1500, "torque_nm": 0.7, "force_n": 400, "label": "1500rpm, 0.7Nm, 400N"},
                },
            },
            "XJTU": {
                "description": "Xi'an Jiaotong University - Speed and radial load",
                "condition_type": "Speed/Load",
                "conditions": {
                    "Cond1_2100_12kN": {"speed_rpm": 2100, "load_kn": 12, "label": "2100rpm, 12kN"},
                    "Cond2_2250_11kN": {"speed_rpm": 2250, "load_kn": 11, "label": "2250rpm, 11kN"},
                    "Cond3_2400_10kN": {"speed_rpm": 2400, "load_kn": 10, "label": "2400rpm, 10kN"},
                },
            },
            "IMS": {
                "description": "NASA IMS - Constant speed with different bearings",
                "condition_type": "Bearing Set",
                "conditions": {
                    "Set1": {"label": "Bearing Set 1 (2000rpm, 6000lbs)"},
                    "Set2": {"label": "Bearing Set 2 (2000rpm, 6000lbs)"},
                    "Set3": {"label": "Bearing Set 3 (2000rpm, 6000lbs)"},
                },
            },
            "MFPT": {
                "description": "MFPT Society - Load conditions",
                "condition_type": "Load",
                "conditions": {
                    "270lbs": {"load_lbs": 270, "label": "270 lbs load"},
                    "25lbs": {"load_lbs": 25, "label": "25 lbs load"},
                    "50lbs": {"load_lbs": 50, "label": "50 lbs load"},
                },
            },
        }

        st.markdown("---")

        # Dataset selection
        gen_dataset = st.selectbox(
            "Select Dataset",
            options=[d for d in datasets.keys() if d in DATASET_CONDITIONS],
            key="gen_dataset",
            help="Datasets with known operating conditions"
        )

        if gen_dataset and gen_dataset in DATASET_CONDITIONS:
            dataset_info = DATASET_CONDITIONS[gen_dataset]

            st.info(f"**{gen_dataset}**: {dataset_info['description']}")
            st.caption(f"Condition Type: {dataset_info['condition_type']}")

            # Show condition details table
            with st.expander("View Operating Conditions Details", expanded=False):
                cond_data = []
                for cond_id, cond_info in dataset_info['conditions'].items():
                    row = {"Condition ID": cond_id, "Description": cond_info['label']}
                    row.update({k: v for k, v in cond_info.items() if k != 'label'})
                    cond_data.append(row)
                st.dataframe(pd.DataFrame(cond_data), use_container_width=True, hide_index=True)

            st.markdown("---")

            # Source and Target condition selection
            conditions = dataset_info['conditions']
            condition_options = [(cid, cinfo['label']) for cid, cinfo in conditions.items()]

            gen_col1, gen_col2 = st.columns(2)

            with gen_col1:
                st.markdown("#### Source Condition (Training)")
                source_conditions = st.multiselect(
                    "Train on these conditions",
                    options=[c[0] for c in condition_options],
                    format_func=lambda x: conditions[x]['label'],
                    default=[condition_options[0][0]] if condition_options else [],
                    key="source_conds"
                )

            with gen_col2:
                st.markdown("#### Target Condition (Testing)")
                available_targets = [c[0] for c in condition_options if c[0] not in source_conditions]
                target_conditions = st.multiselect(
                    "Test on these conditions",
                    options=available_targets,
                    format_func=lambda x: conditions[x]['label'],
                    default=available_targets[:1] if available_targets else [],
                    key="target_conds"
                )

            # Visualize condition differences
            if source_conditions and target_conditions:
                st.markdown("---")
                st.markdown("#### Condition Comparison")

                # Extract numeric values for comparison
                compare_data = []
                for cid in source_conditions + target_conditions:
                    cinfo = conditions[cid]
                    row = {
                        "Condition": cinfo['label'],
                        "Type": "Source (Train)" if cid in source_conditions else "Target (Test)"
                    }
                    # Add numeric parameters
                    for key, val in cinfo.items():
                        if key != 'label' and isinstance(val, (int, float)):
                            row[key] = val
                    compare_data.append(row)

                compare_df = pd.DataFrame(compare_data)
                st.dataframe(compare_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Model and experiment settings
            st.markdown("#### Experiment Settings")
            exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)

            with exp_col1:
                gen_model_type = st.selectbox("Model", options=model_list, key="gen_model")

            with exp_col2:
                gen_epochs = st.slider("Epochs", 10, 100, 30, key="gen_epochs")

            with exp_col3:
                num_runs = st.slider("Runs", 1, 5, 3, key="num_runs", help="Average over multiple runs")

            with exp_col4:
                gen_use_gpu = st.checkbox("GPU", value=False, key="gen_use_gpu")

            # Additional options
            adv_col1, adv_col2 = st.columns(2)
            with adv_col1:
                include_baseline = st.checkbox("Include in-domain baseline", value=True)
            with adv_col2:
                save_models = st.checkbox("Save trained models", value=False)

            # Run experiment
            st.markdown("---")
            if st.button("Run Cross-Condition Experiment", type="primary", key="run_gen"):
                if not source_conditions or not target_conditions:
                    st.error("Please select both source and target conditions")
                else:
                    st.success("Experiment Configuration Ready!")

                    # Show experiment summary
                    st.markdown("### Experiment Summary")
                    summary_col1, summary_col2 = st.columns(2)

                    with summary_col1:
                        st.markdown("**Training Setup:**")
                        st.text(f"Dataset: {gen_dataset}")
                        st.text(f"Model: {gen_model_type}")
                        st.text(f"Source: {[conditions[c]['label'] for c in source_conditions]}")

                    with summary_col2:
                        st.markdown("**Evaluation Setup:**")
                        st.text(f"Target: {[conditions[c]['label'] for c in target_conditions]}")
                        st.text(f"Runs: {num_runs}")
                        st.text(f"Epochs: {gen_epochs}")

                    st.markdown("---")
                    st.markdown("### Expected Results Format")

                    # Show expected results structure
                    example_results = pd.DataFrame({
                        'Evaluation': ['In-Domain (Source)', 'Cross-Domain (Target)', 'Accuracy Drop'],
                        'Accuracy': ['95.2%', '78.4%', '16.8%'],
                        'F1 Macro': ['0.943', '0.756', '0.187'],
                        'F1 Weighted': ['0.951', '0.774', '0.177'],
                    })
                    st.dataframe(example_results, use_container_width=True, hide_index=True)

                    # Visualization placeholder
                    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy Comparison", "Per-Condition Breakdown"))

                    # Bar chart for overall comparison
                    fig.add_trace(
                        go.Bar(name='Source', x=['Accuracy', 'F1'], y=[0.95, 0.94], marker_color='#636EFA'),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Bar(name='Target', x=['Accuracy', 'F1'], y=[0.78, 0.75], marker_color='#EF553B'),
                        row=1, col=1
                    )

                    # Per-condition breakdown
                    target_labels = [conditions[c]['label'][:15] for c in target_conditions]
                    fig.add_trace(
                        go.Bar(name='Target Acc', x=target_labels, y=[0.78 + i*0.02 for i in range(len(target_labels))]),
                        row=1, col=2
                    )

                    fig.update_layout(height=350, showlegend=True, barmode='group')
                    st.plotly_chart(fig, use_container_width=True)

                    st.info("""
                    **Interpretation:**
                    - **Accuracy Drop** measures generalization gap (lower = better generalization)
                    - Large drops indicate the model is sensitive to operating condition changes
                    - Consider domain adaptation techniques if drop > 15%
                    """)

        # =====================================================================
        # Existing Generalization Results
        # =====================================================================
        st.markdown("---")
        st.markdown("### Existing Generalization Results")

        gen_results_dir = './results/generalization'
        if os.path.exists(gen_results_dir):
            # Find latest results file
            csv_files = [f for f in os.listdir(gen_results_dir) if f.startswith('generalization_results') and f.endswith('.csv')]

            if csv_files:
                csv_files.sort(reverse=True)  # Latest first
                selected_file = st.selectbox("Select Results File", csv_files)

                if selected_file:
                    results_path = os.path.join(gen_results_dir, selected_file)
                    gen_df = pd.read_csv(results_path)

                    st.success(f"Loaded {len(gen_df)} experiment results")

                    # Filter options
                    filter_col1, filter_col2 = st.columns(2)
                    with filter_col1:
                        filter_datasets = st.multiselect(
                            "Filter by Dataset",
                            options=gen_df['dataset_name'].unique().tolist(),
                            default=gen_df['dataset_name'].unique().tolist()
                        )
                    with filter_col2:
                        filter_models = st.multiselect(
                            "Filter by Model",
                            options=gen_df['model'].unique().tolist(),
                            default=gen_df['model'].unique().tolist()
                        )

                    filtered_gen = gen_df[
                        (gen_df['dataset_name'].isin(filter_datasets)) &
                        (gen_df['model'].isin(filter_models))
                    ]

                    if len(filtered_gen) > 0:
                        # Summary statistics
                        st.markdown("#### Summary Statistics")
                        summary = filtered_gen.groupby(['dataset_name', 'model']).agg({
                            'source_accuracy': 'mean',
                            'target_accuracy': 'mean',
                            'accuracy_drop': 'mean'
                        }).round(4).reset_index()

                        st.dataframe(summary, use_container_width=True, hide_index=True)

                        # Visualization
                        st.markdown("#### Visualizations")
                        viz_tabs = st.tabs(["Accuracy Drop Heatmap", "Source vs Target", "Per-Condition"])

                        with viz_tabs[0]:
                            # Heatmap
                            pivot = filtered_gen.pivot_table(
                                values='accuracy_drop',
                                index='model',
                                columns='dataset_name',
                                aggfunc='mean'
                            )
                            fig = px.imshow(pivot, text_auto='.3f', color_continuous_scale='RdYlGn_r',
                                          title='Average Accuracy Drop (Lower = Better)')
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)

                        with viz_tabs[1]:
                            # Scatter plot
                            fig = px.scatter(filtered_gen, x='source_accuracy', y='target_accuracy',
                                           color='model', symbol='dataset_name',
                                           title='Source vs Target Accuracy')
                            fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
                                        line=dict(dash='dash', color='gray'))
                            fig.update_layout(height=450)
                            st.plotly_chart(fig, use_container_width=True)

                        with viz_tabs[2]:
                            # Per-condition breakdown
                            filtered_gen['transfer'] = filtered_gen['source_condition'] + ' → ' + filtered_gen['target_condition']
                            fig = px.bar(filtered_gen, x='transfer', y='accuracy_drop',
                                       color='model', barmode='group',
                                       title='Accuracy Drop by Condition Transfer')
                            fig.update_layout(height=400, xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)

                        # Download results
                        st.markdown("#### Download Results")
                        csv_data = filtered_gen.to_csv(index=False)
                        st.download_button(
                            label="Download Filtered Results (CSV)",
                            data=csv_data,
                            file_name="generalization_results_filtered.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No results match the selected filters")
            else:
                st.info("No generalization results found. Run experiments using the script:")
                st.code("python run_generalization_experiments.py --all --epochs 30 --runs 3 --visualize")
        else:
            st.info("No generalization experiments have been run yet.")
            st.markdown("""
            **To run experiments:**
            ```bash
            python run_generalization_experiments.py --all --epochs 30 --runs 3 --visualize
            ```
            """)

    # =========================================================================
    # TAB 3: Custom Model (Optional)
    # =========================================================================
    with custom_tab:
        st.markdown("### Upload Custom Model (Optional)")
        st.markdown("""
        Upload your own PyTorch model to benchmark against built-in models.

        **Requirements:**
        - Python file containing `nn.Module` subclass
        - Input shape: `[batch, seq_len, features]`
        - Output shape: `[batch, num_classes]`
        """)

        uploaded_file = st.file_uploader("Upload .py file", type=['py'])

        if uploaded_file:
            file_content = uploaded_file.read().decode('utf-8')

            with st.expander("Preview Code", expanded=False):
                st.code(file_content, language='python')

            col1, col2 = st.columns(2)
            with col1:
                class_name = st.text_input("Class Name (auto-detect if empty)")
            with col2:
                model_name = st.text_input("Model Name", value=uploaded_file.name.replace('.py', ''))

            if st.button("Validate & Register Model"):
                model_class, error = load_custom_model(file_content, class_name)
                if model_class is None:
                    st.error(f"Load failed: {error}")
                else:
                    is_valid, msg = validate_custom_model(model_class)
                    if is_valid:
                        st.success(f"✅ {msg}")
                        # Save and register
                        save_path = os.path.join(CUSTOM_MODELS_DIR, f"{model_name}.py")
                        with open(save_path, 'w') as f:
                            f.write(file_content)
                        if 'custom_models' not in st.session_state:
                            st.session_state.custom_models = {}
                        st.session_state.custom_models[model_name] = {
                            'file_path': save_path,
                            'class_name': class_name or 'auto'
                        }
                        st.info(f"Model '{model_name}' registered. Select it in Training Configuration.")
                    else:
                        st.error(f"Validation failed: {msg}")

        # Show registered custom models
        if custom_models:
            st.markdown("### Registered Custom Models")
            for name in custom_models:
                st.text(f"  • {name}")

    st.markdown("---")

    # =========================================================================
    # Model Architecture Visualization
    # =========================================================================
    with st.expander("Model Architecture Preview", expanded=False):
        st.markdown("### Model Architecture Visualization")
        st.caption("Preview model structure before training (uses torchview/torchinfo if available)")

        viz_col1, viz_col2 = st.columns([1, 2])

        with viz_col1:
            # Get dataset info for model dimensions
            try:
                dataset_path = os.path.join(DATASET_DIR, dataset_folder)
                train_file = os.path.join(dataset_path, 'PdM_TRAIN.npz')
                if os.path.exists(train_file):
                    loaded = np.load(train_file, allow_pickle=True)
                    df_restored = pd.DataFrame(loaded['data'], columns=loaded['columns'])
                    sample_features = df_restored['features'].iloc[0]
                    if isinstance(sample_features, np.ndarray):
                        if len(sample_features.shape) > 1:
                            enc_in = sample_features.shape[1]
                        else:
                            enc_in = 1
                    else:
                        enc_in = 1
                    num_classes = len(loaded.get('class_names', np.unique(df_restored['label'].values)))
                else:
                    enc_in = 7
                    num_classes = 3
            except:
                enc_in = 7
                num_classes = 3

            st.text(f"Input Features: {enc_in}")
            st.text(f"Sequence Length: {seq_len}")
            st.text(f"Output Classes: {num_classes}")
            st.text(f"d_model: {d_model}")
            st.text(f"d_ff: {d_ff}")
            st.text(f"Encoder Layers: {e_layers}")

        with viz_col2:
            if st.button("Generate Architecture Diagram"):
                try:
                    # Import model classes
                    from models import DLinear, MLP, Transformer, PatchTST, TimesNet, iTransformer

                    # Create a simple args namespace for model initialization
                    class ModelArgs:
                        pass

                    model_args = ModelArgs()
                    model_args.seq_len = seq_len
                    model_args.pred_len = 0
                    model_args.enc_in = enc_in
                    model_args.d_model = d_model
                    model_args.d_ff = d_ff
                    model_args.n_heads = n_heads
                    model_args.e_layers = e_layers
                    model_args.d_layers = d_layers
                    model_args.dropout = dropout
                    model_args.num_class = num_classes
                    model_args.embed = 'fixed'
                    model_args.freq = 'h'
                    model_args.factor = 1
                    model_args.activation = 'gelu'
                    model_args.output_attention = False
                    # Task settings
                    model_args.task_name = task_type
                    # Model-specific params (now always defined with defaults)
                    model_args.top_k = top_k
                    model_args.num_kernels = num_kernels
                    model_args.patch_len = patch_len
                    model_args.stride = stride
                    model_args.moving_avg = moving_avg
                    model_args.individual = individual
                    # Foundation model params
                    if model_type in FOUNDATION_MODELS:
                        model_args.chronos_size = fm_size if model_type == 'Chronos' else 'small'
                        model_args.moment_size = fm_size if model_type == 'MOMENT' else 'base'
                        model_args.freeze_backbone = freeze_backbone if 'freeze_backbone' in dir() else True

                    # Build model for selected type
                    model_map = {
                        'DLinear': DLinear,
                        'MLP': MLP,
                        'Transformer': Transformer,
                    }

                    if model_type in model_map:
                        model_instance = model_map[model_type].Model(model_args)

                        # Try to use torchview for visualization (preferred)
                        try:
                            from torchview import draw_graph
                            import tempfile

                            # Generate graph with torchview
                            model_graph = draw_graph(
                                model_instance,
                                input_size=(1, seq_len, enc_in),
                                expand_nested=True,
                                depth=3,
                                device='cpu'
                            )

                            # Save to temp file and display
                            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                                model_graph.visual_graph.render(tmp.name.replace('.png', ''), format='png', cleanup=True)
                                st.image(tmp.name, caption=f"{model_type} Architecture")

                            st.success("Architecture generated with torchview")

                        except Exception as viz_error:
                            # Fallback: show parameter count and layer info
                            st.info(f"Visual diagram unavailable ({str(viz_error)[:50]}...)")
                            st.markdown("**Model Layers:**")
                            for name, module in model_instance.named_modules():
                                if name and '.' not in name:  # Top-level modules only
                                    st.text(f"  {name}: {module.__class__.__name__}")

                        # Always show parameter counts
                        total_params = sum(p.numel() for p in model_instance.parameters())
                        trainable_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
                        param_cols = st.columns(2)
                        with param_cols[0]:
                            st.metric("Total Parameters", f"{total_params:,}")
                        with param_cols[1]:
                            st.metric("Trainable", f"{trainable_params:,}")

                        # Architecture description
                        st.markdown("**Architecture Flow:**")
                        arch_desc = {
                            'DLinear': "Input → Linear Decomposition → Trend/Seasonal → Linear Projection → Output",
                            'MLP': "Input → Flatten → Hidden Layers (ReLU) → Output",
                            'Transformer': "Input → Embedding → Positional Encoding → Encoder Blocks (Self-Attention + FFN) → Pooling → Output",
                            'PatchTST': "Input → Patching → Patch Embedding → Transformer Encoder → Flatten → Output",
                            'TimesNet': "Input → Embedding → TimesBlock (2D Conv over periods) × N → Output",
                            'iTransformer': "Input → Inverted Embedding → Transformer Encoder → Linear → Output",
                        }
                        st.info(arch_desc.get(model_type, "Input → Model Layers → Output"))

                    else:
                        st.info(f"Architecture preview not available for {model_type}. Model will be built during training.")

                except Exception as e:
                    st.warning(f"Could not generate architecture diagram: {str(e)}")
                    st.caption("Install torchview: `pip install torchview graphviz`")

    st.markdown("---")

    # Build params dict
    params = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_epochs': train_epochs,
        'd_model': d_model,
        'e_layers': e_layers,
        'd_layers': d_layers,
        'd_ff': d_ff,
        'n_heads': n_heads,
        'dropout': dropout,
        'patience': patience,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'weight_decay': 0.01 if optimizer == 'AdamW' else 0,
        'num_workers': 4,
    }

    # Add model-specific params
    if model_type in ['TimesNet']:
        params['top_k'] = top_k
        params['num_kernels'] = num_kernels
    elif model_type in ['PatchTST', 'TimeXer']:
        params['patch_len'] = patch_len
        params['stride'] = stride
    elif model_type in ['Autoformer', 'FEDformer']:
        params['moving_avg'] = moving_avg
        params['factor'] = factor
    elif model_type in ['DLinear', 'MLP']:
        params['individual'] = individual
    elif model_type in ['SegRNN', 'Crossformer']:
        params['seg_len'] = seg_len
    elif model_type in ['MambaSimple', 'Mamba']:
        params['expand'] = expand
        params['d_conv'] = d_conv
    elif model_type in FOUNDATION_MODELS:
        params['fm_size'] = fm_size if 'fm_size' in dir() else FOUNDATION_MODELS[model_type]['default_size']
        params['freeze_backbone'] = freeze_backbone if 'freeze_backbone' in dir() else True

    # Task settings
    params['task_type'] = task_type if 'task_type' in dir() else 'classification'
    params['failure_mode'] = failure_mode if 'failure_mode' in dir() else 'classification'

    # Training controls
    ctrl_col1, ctrl_col2, status_col = st.columns([1, 1, 2])

    with ctrl_col1:
        start_button = st.button(
            "▶️ Start Training",
            type="primary",
            disabled=st.session_state.training_active,
            use_container_width=True
        )

    with ctrl_col2:
        stop_button = st.button(
            "⏹️ Stop",
            disabled=not st.session_state.training_active,
            use_container_width=True
        )

    with status_col:
        if st.session_state.training_active:
            st.warning("Training in progress...")
        else:
            st.info(f"Ready to train {model_type} on {dataset_name}")

    # Training progress area
    progress_container = st.container()
    metrics_container = st.container()
    curves_container = st.container()

    if start_button and not st.session_state.training_active:
        st.session_state.training_active = True

        # =====================================================================
        # IMMEDIATE RESPONSE: Show configuration summary right away
        # =====================================================================
        with progress_container:
            st.subheader("Training Configuration Summary")

            # Show configuration immediately so users can verify settings
            config_cols = st.columns(3)
            with config_cols[0]:
                st.markdown("**Model & Data**")
                st.text(f"Model: {model_type}")
                st.text(f"Dataset: {dataset_name}")
                st.text(f"Sequence Length: {seq_len}")
                st.text(f"Device: {'GPU ' + str(gpu_id) if use_gpu else 'CPU'}")

            with config_cols[1]:
                st.markdown("**Training Parameters**")
                st.text(f"Batch Size: {batch_size}")
                st.text(f"Learning Rate: {learning_rate:.0e}")
                st.text(f"Epochs: {train_epochs}")
                st.text(f"Optimizer: {optimizer}")
                st.text(f"Scheduler: {scheduler}")
                st.text(f"Early Stop: {patience} epochs")

            with config_cols[2]:
                st.markdown("**Architecture**")
                st.text(f"d_model: {d_model}")
                st.text(f"d_ff: {d_ff}")
                st.text(f"Heads: {n_heads}")
                st.text(f"Encoder Layers: {e_layers}")
                st.text(f"Dropout: {dropout}")
                # Show model-specific params
                if model_type in ['TimesNet']:
                    st.text(f"Top K: {params.get('top_k', 5)}")
                elif model_type in ['PatchTST', 'TimeXer']:
                    st.text(f"Patch Len: {params.get('patch_len', 16)}")

            st.markdown("---")
            st.subheader("Training Progress")
            progress_bar = st.progress(0, text="Initializing experiment...")

            # Live metrics display with clearer labels
            metric_cols = st.columns(5)
            with metric_cols[0]:
                epoch_metric = st.empty()
            with metric_cols[1]:
                loss_metric = st.empty()
            with metric_cols[2]:
                acc_metric = st.empty()
            with metric_cols[3]:
                val_loss_metric = st.empty()
            with metric_cols[4]:
                val_acc_metric = st.empty()

            # Status text with log area
            status_text = st.empty()
            log_area = st.empty()

        with curves_container:
            st.subheader("Learning Curves")
            curves_placeholder = st.empty()

        # Build args from configuration
        config_manager = get_config_manager()
        args = config_manager.create_args(
            model=model_type if not is_custom_model else 'MLP',
            dataset=dataset_folder,  # Use folder ID for training
            overrides={'training': params, 'model': params}
        )
        # Store display name for results
        args.dataset_display_name = dataset_name

        # Override with UI selections
        args.model = model_type
        args.batch_size = params['batch_size']
        args.learning_rate = params['learning_rate']
        args.train_epochs = params['train_epochs']
        args.d_model = params['d_model']
        args.e_layers = params['e_layers']
        args.d_ff = params['d_ff']
        args.n_heads = params['n_heads']
        args.dropout = params['dropout']
        args.patience = params['patience']
        args.use_gpu = use_gpu
        args.gpu = gpu_id
        args.num_workers = params.get('num_workers', 4)
        args.seq_len = seq_len

        # Task settings
        args.task_name = task_type
        args.failure_prediction_mode = failure_mode

        # Foundation model settings
        if model_type in FOUNDATION_MODELS:
            args.chronos_size = params.get('fm_size', 'small') if model_type == 'Chronos' else 'small'
            args.moment_size = params.get('fm_size', 'base') if model_type == 'MOMENT' else 'base'
            args.freeze_backbone = params.get('freeze_backbone', True)

        # Optimizer/scheduler settings
        args.optimizer = params.get('optimizer', 'AdamW')
        args.weight_decay = params.get('weight_decay', 0)
        args.scheduler = params.get('scheduler', 'cosine')

        # Model-specific parameters
        if model_type in ['TimesNet']:
            args.top_k = params.get('top_k', 5)
            args.num_kernels = params.get('num_kernels', 6)
        elif model_type in ['PatchTST', 'TimeXer']:
            args.patch_len = params.get('patch_len', 16)
            args.stride = params.get('stride', 8)
        elif model_type in ['Autoformer', 'FEDformer']:
            args.moving_avg = params.get('moving_avg', 25)
            args.factor = params.get('factor', 1)
        elif model_type in ['SegRNN', 'Crossformer']:
            args.seg_len = params.get('seg_len', 48)
        elif model_type in ['MambaSimple', 'Mamba']:
            args.expand = params.get('expand', 2)
            args.d_conv = params.get('d_conv', 4)

        # Store processing config in args for use during data loading
        args.processing_config = processing_config

        status_text.info(f"Initializing {model_type} training...")

        try:
            # Import and run training
            from exp.exp_classification import Exp_Classification
            from exp.exp_early_failure import Exp_Early_Failure
            from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
                Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
                Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
                WPMixer, MultiPatchFormer, MLP

            # Import foundation models with error handling
            try:
                from models import Chronos
            except ImportError:
                Chronos = None
            try:
                from models import MOMENT
            except ImportError:
                MOMENT = None
            try:
                from models import TimesFM
            except ImportError:
                TimesFM = None
            try:
                from models import Moirai
            except ImportError:
                Moirai = None

            model_dict = {
                'TimesNet': TimesNet,
                'Autoformer': Autoformer,
                'Transformer': Transformer,
                'Nonstationary_Transformer': Nonstationary_Transformer,
                'DLinear': DLinear,
                'FEDformer': FEDformer,
                'Informer': Informer,
                'LightTS': LightTS,
                'Reformer': Reformer,
                'ETSformer': ETSformer,
                'PatchTST': PatchTST,
                'Pyraformer': Pyraformer,
                'MICN': MICN,
                'Crossformer': Crossformer,
                'FiLM': FiLM,
                'iTransformer': iTransformer,
                'Koopa': Koopa,
                'TiDE': TiDE,
                'FreTS': FreTS,
                'MambaSimple': MambaSimple,
                'TimeMixer': TimeMixer,
                'TSMixer': TSMixer,
                'SegRNN': SegRNN,
                'TemporalFusionTransformer': TemporalFusionTransformer,
                "SCINet": SCINet,
                'PAttn': PAttn,
                'TimeXer': TimeXer,
                'WPMixer': WPMixer,
                'MultiPatchFormer': MultiPatchFormer,
                'MLP': MLP,
                # Foundation Models
                'Chronos': Chronos,
                'MOMENT': MOMENT,
                'TimesFM': TimesFM,
                'Moirai': Moirai,
            }

            args.model_dict = model_dict

            # Initialize training
            progress_bar.progress(0.05, text="Creating experiment...")

            # Select experiment class based on task type
            if args.task_name == 'early_failure':
                experiment = Exp_Early_Failure(args)
            else:
                experiment = Exp_Classification(args)

            # Show model info immediately after creation
            progress_bar.progress(0.1, text="Loading data...")

            # Get model parameter count
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            try:
                # Build model to get parameter count
                model_instance = experiment._build_model()
                param_count = count_parameters(model_instance)
                if param_count >= 1e6:
                    param_str = f"{param_count/1e6:.2f}M"
                elif param_count >= 1e3:
                    param_str = f"{param_count/1e3:.1f}K"
                else:
                    param_str = str(param_count)
                log_area.info(f"Model initialized: {param_str} trainable parameters")
            except:
                log_area.info("Model initialized successfully")

            # Initialize live metrics display
            epoch_metric.metric("Epoch", "0 / " + str(args.train_epochs))
            loss_metric.metric("Train Loss", "-")
            acc_metric.metric("Train Acc", "-")
            val_loss_metric.metric("Val Loss", "-")
            val_acc_metric.metric("Val Acc", "-")

            progress_bar.progress(0.15, text="Starting training loop...")
            status_text.success("Ready! Training started...")

            # Run training
            training_start = time.time()
            model = experiment.train()

            progress_bar.progress(0.9, text="Running final evaluation...")
            experiment.test(load_model=True)

            training_time = time.time() - training_start
            st.session_state.training_active = False
            progress_bar.progress(1.0, text="Complete!")

            # Display final results
            st.success(f"Training completed in {training_time:.1f}s")

            st.markdown("---")
            st.subheader("Final Results")

            # Load and display results
            result_file = f'./results/{dataset_folder}/{model_type}/result_classification.txt'
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    results_text = f.read()

                metrics = {}
                for line in results_text.split('\n'):
                    if ':' in line:
                        key, val = line.split(':', 1)
                        try:
                            metrics[key.strip().lower()] = float(val.strip())
                        except:
                            continue

                # Large metrics display
                result_cols = st.columns(4)
                with result_cols[0]:
                    acc_val = metrics.get('accuracy', 0)
                    st.metric("Accuracy", f"{acc_val:.2%}", delta=None)
                with result_cols[1]:
                    f1_val = metrics.get('f1_weighted', metrics.get('f1_macro', 0))
                    st.metric("F1 Score", f"{f1_val:.4f}")
                with result_cols[2]:
                    prec_val = metrics.get('precision', 0)
                    st.metric("Precision", f"{prec_val:.4f}" if prec_val else "-")
                with result_cols[3]:
                    recall_val = metrics.get('recall', 0)
                    st.metric("Recall", f"{recall_val:.4f}" if recall_val else "-")

                # Additional metrics
                with st.expander("All Metrics", expanded=False):
                    metrics_df = pd.DataFrame([
                        {"Metric": k.title().replace('_', ' '), "Value": f"{v:.4f}"}
                        for k, v in metrics.items()
                    ])
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

                # Store in training history
                st.session_state.training_history.append({
                    'model': model_type,
                    'dataset': dataset_name,  # Use display name
                    'accuracy': metrics.get('accuracy', 0),
                    'f1': f1_val,
                    'time': training_time,
                    'timestamp': datetime.now().isoformat()
                })

                # Learning curves visualization
                st.markdown("---")
                st.subheader("Training Analysis")

                # Try to load training logs for curves
                log_file = f'./results/{dataset_folder}/{model_type}/training_log.json'
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        training_log = json.load(f)

                    # Plot learning curves
                    if 'train_loss' in training_log and 'val_loss' in training_log:
                        fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))

                        fig.add_trace(
                            go.Scatter(y=training_log['train_loss'], name='Train Loss', line=dict(color='blue')),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(y=training_log['val_loss'], name='Val Loss', line=dict(color='orange')),
                            row=1, col=1
                        )

                        if 'train_acc' in training_log:
                            fig.add_trace(
                                go.Scatter(y=training_log['train_acc'], name='Train Acc', line=dict(color='blue')),
                                row=1, col=2
                            )
                        if 'val_acc' in training_log:
                            fig.add_trace(
                                go.Scatter(y=training_log['val_acc'], name='Val Acc', line=dict(color='orange')),
                                row=1, col=2
                            )

                        fig.update_layout(height=350, showlegend=True)
                        curves_placeholder.plotly_chart(fig, use_container_width=True)

                # Confusion matrix if available
                cm_file = f'./results/{dataset_folder}/{model_type}/confusion_matrix.npy'
                if os.path.exists(cm_file):
                    cm = np.load(cm_file)
                    fig_cm = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="True", color="Count"),
                        color_continuous_scale='Blues',
                        title="Confusion Matrix"
                    )
                    fig_cm.update_layout(height=400)
                    st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.warning("Results file not found. Training may not have saved results properly.")

        except Exception as e:
            st.session_state.training_active = False
            st.error(f"Training failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    if stop_button:
        st.session_state.training_active = False
        tracker = get_tracker()
        tracker.request_stop()

    # Show experiment history
    st.divider()
    render_experiment_history()


# =============================================================================
# RESULTS ANALYSIS PAGE
# =============================================================================

def get_dataset_display_name(folder_id: str) -> str:
    """Get display name from folder ID (e.g., '01' -> 'Paderborn')."""
    return folder_to_name.get(folder_id, folder_id)


def is_clean_model_name(model_name: str) -> bool:
    """Check if model name is clean (not a long configuration string)."""
    # Clean model names are short and don't contain underscores with numbers
    if len(model_name) > 30:
        return False
    # Check for patterns like 'sl512_dm128_nh8' which indicate config strings
    if '_sl' in model_name.lower() or '_dm' in model_name.lower():
        return False
    if model_name.startswith('classification_'):
        return False
    return True


def results_analysis_page():
    """Enhanced results analysis page with better visualizations."""
    st.title("Results Analysis")
    st.markdown("---")

    # Load all results
    results = []
    if os.path.exists(RESULTS_DIR):
        for dataset_dir in os.listdir(RESULTS_DIR):
            dataset_path = os.path.join(RESULTS_DIR, dataset_dir)
            if os.path.isdir(dataset_path):
                for model_dir in os.listdir(dataset_path):
                    # Skip weirdly named models (long configuration strings)
                    if not is_clean_model_name(model_dir):
                        continue

                    result_file = os.path.join(dataset_path, model_dir, 'result_classification.txt')
                    if os.path.exists(result_file):
                        try:
                            with open(result_file, 'r') as f:
                                content = f.read()

                            # Get proper dataset name from folder ID
                            dataset_name = get_dataset_display_name(dataset_dir)

                            metrics = {
                                'dataset_folder': dataset_dir,
                                'dataset': dataset_name,
                                'model': model_dir
                            }
                            for line in content.split('\n'):
                                if ':' in line:
                                    key, val = line.split(':', 1)
                                    try:
                                        metrics[key.strip().lower()] = float(val.strip())
                                    except:
                                        continue
                            results.append(metrics)
                        except:
                            continue

    if not results:
        st.info("No results found. Train some models first to see analysis here.")
        return

    df = pd.DataFrame(results)

    # Sidebar-style filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_datasets = st.multiselect(
            "Select Datasets",
            options=df['dataset'].unique().tolist(),
            default=df['dataset'].unique().tolist(),
            help="Filter results by dataset"
        )

    with col2:
        selected_models = st.multiselect(
            "Select Models",
            options=sorted(df['model'].unique().tolist()),
            default=df['model'].unique().tolist(),
            help="Filter results by model"
        )

    with col3:
        metric_options = ['accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'ece', 'brier']
        available_metrics = [m for m in metric_options if m in df.columns]
        selected_metric = st.selectbox(
            "Primary Metric",
            options=available_metrics,
            index=0,
            help="Metric used for ranking"
        )

    # Filter dataframe
    filtered_df = df[
        (df['dataset'].isin(selected_datasets)) &
        (df['model'].isin(selected_models))
    ].copy()

    if filtered_df.empty:
        st.warning("No results match the selected filters.")
        return

    st.markdown("---")

    # ==========================================================================
    # LEADERBOARD SECTION
    # ==========================================================================
    st.subheader("Leaderboard")

    # Aggregate by model
    leaderboard = filtered_df.groupby('model')[available_metrics].mean().reset_index()

    # Sort by selected metric (descending for accuracy/f1, ascending for ece/brier)
    ascending = selected_metric in ['ece', 'brier', 'loss']
    leaderboard = leaderboard.sort_values(selected_metric, ascending=ascending).reset_index(drop=True)

    # Add rank column
    leaderboard.insert(0, 'Rank', range(1, len(leaderboard) + 1))

    # Display with medals for top 3
    def format_rank(rank):
        if rank == 1:
            return "🥇 1"
        elif rank == 2:
            return "🥈 2"
        elif rank == 3:
            return "🥉 3"
        return str(rank)

    leaderboard_display = leaderboard.copy()
    leaderboard_display['Rank'] = leaderboard_display['Rank'].apply(format_rank)

    # Format metrics as percentages where appropriate
    format_dict = {}
    for m in available_metrics:
        if m in ['accuracy', 'f1_micro', 'f1_macro', 'f1_weighted']:
            format_dict[m] = '{:.2%}'
        else:
            format_dict[m] = '{:.4f}'

    # Rename columns for display
    column_rename = {
        'model': 'Model',
        'accuracy': 'Accuracy',
        'f1_micro': 'F1 (Micro)',
        'f1_macro': 'F1 (Macro)',
        'f1_weighted': 'F1 (Weighted)',
        'ece': 'ECE',
        'brier': 'Brier Score'
    }
    leaderboard_display = leaderboard_display.rename(columns=column_rename)

    # Show leaderboard table
    st.dataframe(
        leaderboard_display.style.format(
            {column_rename.get(k, k): v for k, v in format_dict.items() if column_rename.get(k, k) in leaderboard_display.columns}
        ),
        use_container_width=True,
        hide_index=True,
        height=min(400, 50 + len(leaderboard) * 35)
    )

    # Leaderboard bar chart
    st.markdown("#### Visual Comparison")

    # Create bar chart with better styling
    metric_label = column_rename.get(selected_metric, selected_metric)

    fig = go.Figure()

    # Sort for chart
    chart_data = leaderboard.sort_values(selected_metric, ascending=not ascending)

    # Color gradient based on performance
    colors = px.colors.sequential.Blues_r[:len(chart_data)] if not ascending else px.colors.sequential.Reds[:len(chart_data)]
    if len(colors) < len(chart_data):
        colors = colors * (len(chart_data) // len(colors) + 1)

    fig.add_trace(go.Bar(
        x=chart_data['model'],
        y=chart_data[selected_metric],
        marker_color=colors[:len(chart_data)],
        text=[f"{v:.2%}" if selected_metric in ['accuracy', 'f1_micro', 'f1_macro', 'f1_weighted'] else f"{v:.4f}"
              for v in chart_data[selected_metric]],
        textposition='outside',
        textfont=dict(size=14)
    ))

    fig.update_layout(
        title=dict(
            text=f"{metric_label} by Model",
            font=dict(size=20)
        ),
        xaxis=dict(
            title="Model",
            tickfont=dict(size=14),
            tickangle=-45
        ),
        yaxis=dict(
            title=metric_label,
            tickfont=dict(size=14),
            tickformat='.0%' if selected_metric in ['accuracy', 'f1_micro', 'f1_macro', 'f1_weighted'] else '.3f'
        ),
        height=500,
        margin=dict(b=120),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ==========================================================================
    # DETAILED ANALYSIS
    # ==========================================================================
    st.subheader("Detailed Analysis")

    tab1, tab2, tab3 = st.tabs(["Per-Dataset Results", "Model Comparison Heatmap", "All Results Table"])

    with tab1:
        # Per-dataset performance
        st.markdown("#### Performance by Dataset")

        # Group by dataset
        for dataset in selected_datasets:
            dataset_results = filtered_df[filtered_df['dataset'] == dataset].copy()
            if dataset_results.empty:
                continue

            with st.expander(f"**{dataset}**", expanded=True):
                # Sort by metric
                dataset_results = dataset_results.sort_values(selected_metric, ascending=ascending)

                col1, col2 = st.columns([2, 1])

                with col1:
                    # Bar chart for this dataset
                    fig = px.bar(
                        dataset_results,
                        x='model',
                        y=selected_metric,
                        color=selected_metric,
                        color_continuous_scale='Blues' if not ascending else 'Reds_r',
                        text=selected_metric
                    )

                    fig.update_traces(
                        texttemplate='%{text:.2%}' if selected_metric in ['accuracy', 'f1_micro', 'f1_macro', 'f1_weighted'] else '%{text:.3f}',
                        textposition='outside',
                        textfont=dict(size=12)
                    )

                    fig.update_layout(
                        xaxis=dict(tickangle=-45, tickfont=dict(size=12)),
                        yaxis=dict(tickfont=dict(size=12)),
                        height=350,
                        margin=dict(b=100),
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Top 5 models for this dataset
                    st.markdown("**Top Models:**")
                    top_models = dataset_results.head(5)
                    for idx, row in top_models.iterrows():
                        rank = top_models.index.get_loc(idx) + 1
                        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
                        val = row[selected_metric]
                        if selected_metric in ['accuracy', 'f1_micro', 'f1_macro', 'f1_weighted']:
                            st.markdown(f"{medal} **{row['model']}**: {val:.2%}")
                        else:
                            st.markdown(f"{medal} **{row['model']}**: {val:.4f}")

    with tab2:
        # Heatmap
        st.markdown("#### Model vs Dataset Heatmap")

        pivot_df = filtered_df.pivot_table(
            index='model',
            columns='dataset',
            values=selected_metric,
            aggfunc='mean'
        )

        # Sort models by average performance
        pivot_df['avg'] = pivot_df.mean(axis=1)
        pivot_df = pivot_df.sort_values('avg', ascending=ascending)
        pivot_df = pivot_df.drop('avg', axis=1)

        fig = px.imshow(
            pivot_df,
            labels=dict(x="Dataset", y="Model", color=metric_label),
            aspect="auto",
            color_continuous_scale='Blues' if not ascending else 'Reds_r',
            text_auto='.2%' if selected_metric in ['accuracy', 'f1_micro', 'f1_macro', 'f1_weighted'] else '.3f'
        )

        fig.update_layout(
            height=max(400, len(pivot_df) * 30 + 100),
            xaxis=dict(tickfont=dict(size=12), tickangle=-45),
            yaxis=dict(tickfont=dict(size=12)),
            margin=dict(b=150)
        )

        fig.update_traces(textfont=dict(size=11))

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Full results table
        st.markdown("#### Complete Results")

        display_df = filtered_df[['dataset', 'model'] + available_metrics].copy()

        # Rename columns
        display_df = display_df.rename(columns=column_rename)

        # Sort
        sort_col = column_rename.get(selected_metric, selected_metric)
        display_df = display_df.sort_values(sort_col, ascending=ascending)

        st.dataframe(
            display_df.style.format(
                {column_rename.get(k, k): v for k, v in format_dict.items() if column_rename.get(k, k) in display_df.columns}
            ),
            use_container_width=True,
            hide_index=True,
            height=500
        )


# =============================================================================
# MODEL PLAYGROUND PAGE
# =============================================================================

def model_playground_page():
    """Interactive Model Playground for PDM experimentation."""
    st.title("Model Playground")
    st.markdown("---")

    st.markdown("""
    Interactive playground for experimenting with predictive maintenance models.
    Upload your model, test on sample data, and visualize predictions in real-time.
    """)

    # Initialize session state
    if 'playground_model' not in st.session_state:
        st.session_state.playground_model = None
    if 'playground_predictions' not in st.session_state:
        st.session_state.playground_predictions = None

    # Tabs for different playground features
    tab1, tab2, tab3, tab4 = st.tabs([
        "Quick Inference", "Batch Evaluation", "Model Comparison", "Signal Analysis"
    ])

    # ==========================================================================
    # TAB 1: Quick Inference
    # ==========================================================================
    with tab1:
        st.subheader("Quick Inference")
        st.markdown("Test a trained model on individual samples.")

        col1, col2 = st.columns([1, 1])

        with col1:
            # Model selection
            st.markdown("#### Select Model")

            model_source = st.radio(
                "Model Source",
                ["Pre-trained Baseline", "Custom Upload"],
                horizontal=True
            )

            if model_source == "Pre-trained Baseline":
                datasets = get_available_datasets()
                dataset_id = st.selectbox(
                    "Dataset",
                    options=list(datasets.keys()),
                    format_func=lambda x: f"{datasets[x]} ({x})"
                )

                # Check for available trained models
                results_path = os.path.join(RESULTS_DIR, dataset_id)
                available_models = []
                if os.path.exists(results_path):
                    for model_dir in os.listdir(results_path):
                        checkpoint_path = os.path.join(results_path, model_dir, 'checkpoint.pth')
                        if os.path.exists(checkpoint_path):
                            available_models.append(model_dir)

                if available_models:
                    selected_model = st.selectbox("Trained Model", available_models)
                else:
                    st.warning("No trained models found. Train a model first.")
                    selected_model = None

            else:
                # Custom model upload
                uploaded_model = st.file_uploader(
                    "Upload Model Checkpoint (.pth)",
                    type=['pth', 'pt']
                )
                selected_model = None
                if uploaded_model:
                    st.success(f"Uploaded: {uploaded_model.name}")

            # Dataset for testing
            st.markdown("#### Select Test Data")
            datasets = get_available_datasets()
            test_dataset = st.selectbox(
                "Test Dataset",
                options=list(datasets.keys()),
                format_func=lambda x: f"{datasets[x]} ({x})",
                key="test_dataset_select"
            )

        with col2:
            st.markdown("#### Sample Selection")

            # Load test data
            try:
                test_path = os.path.join(DATASET_DIR, test_dataset, 'PdM_TEST.npz')
                if os.path.exists(test_path):
                    loaded = np.load(test_path, allow_pickle=True)
                    df_restored = pd.DataFrame(loaded['data'], columns=loaded['columns'])
                    features_list = df_restored['features'].tolist()
                    labels = df_restored['label'].values

                    class_names = loaded.get('class_names', [f"Class {i}" for i in range(len(np.unique(labels)))])
                    if hasattr(class_names, 'tolist'):
                        class_names = class_names.tolist()

                    # Sample selector
                    sample_idx = st.slider(
                        "Sample Index",
                        0, len(features_list) - 1, 0
                    )

                    true_label = int(labels[sample_idx])
                    st.metric("True Label", f"{class_names[true_label]} ({true_label})")

                    # Show sample info
                    sample = features_list[sample_idx]
                    if isinstance(sample, list):
                        sample = np.array(sample)

                    st.caption(f"Sample shape: {sample.shape}")

                else:
                    st.error("Test data not found")
                    sample = None
                    sample_idx = 0

            except Exception as e:
                st.error(f"Error loading data: {e}")
                sample = None

        # Run inference button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            run_inference = st.button("Run Inference", type="primary", use_container_width=True)

        with col2:
            run_batch = st.button("Run on 10 Samples", use_container_width=True)

        # Visualization area
        if sample is not None:
            st.markdown("#### Sample Visualization")

            # Plot the time series
            fig = go.Figure()

            if len(sample.shape) == 1:
                fig.add_trace(go.Scatter(y=sample, mode='lines', name='Signal'))
            else:
                for i in range(min(sample.shape[1], 5)):
                    fig.add_trace(go.Scatter(y=sample[:, i], mode='lines', name=f'Channel {i}'))

            fig.update_layout(
                title=f"Sample {sample_idx} - True Label: {class_names[true_label]}",
                xaxis_title="Time Step",
                yaxis_title="Value",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # If inference was run, show predictions
            if run_inference and selected_model:
                with st.spinner("Running inference..."):
                    # Placeholder for actual inference
                    # In real implementation, load model and run prediction
                    st.info("Model inference would run here. Implement model loading and prediction.")

                    # Simulated prediction (replace with actual)
                    predicted_class = np.random.randint(0, len(class_names))
                    confidence = np.random.random()

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Label", f"{class_names[predicted_class]}")
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")

                    if predicted_class == true_label:
                        st.success("Correct prediction!")
                    else:
                        st.error("Incorrect prediction")

    # ==========================================================================
    # TAB 2: Batch Evaluation
    # ==========================================================================
    with tab2:
        st.subheader("Batch Evaluation")
        st.markdown("Evaluate a model on the entire test set.")

        col1, col2 = st.columns(2)

        with col1:
            datasets = get_available_datasets()
            eval_dataset = st.selectbox(
                "Dataset for Evaluation",
                options=list(datasets.keys()),
                format_func=lambda x: f"{datasets[x]} ({x})",
                key="eval_dataset"
            )

            # Check for available models
            results_path = os.path.join(RESULTS_DIR, eval_dataset)
            available_models = []
            if os.path.exists(results_path):
                for model_dir in os.listdir(results_path):
                    checkpoint_path = os.path.join(results_path, model_dir, 'checkpoint.pth')
                    if os.path.exists(checkpoint_path):
                        available_models.append(model_dir)

            if available_models:
                eval_model = st.selectbox("Model to Evaluate", available_models, key="eval_model")
            else:
                eval_model = None
                st.warning("No trained models available.")

        with col2:
            st.markdown("#### Evaluation Options")

            eval_split = st.selectbox("Split", ["test", "val"], key="eval_split")

            show_confusion = st.checkbox("Show Confusion Matrix", value=True)
            show_per_class = st.checkbox("Show Per-Class Metrics", value=True)
            show_errors = st.checkbox("Show Error Analysis", value=True)

        if st.button("Run Evaluation", type="primary", key="run_eval"):
            if eval_model:
                with st.spinner("Evaluating model..."):
                    # Load existing results if available
                    result_file = os.path.join(RESULTS_DIR, eval_dataset, eval_model, 'result_classification.txt')

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

                        st.markdown("#### Evaluation Results")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
                        with col2:
                            st.metric("F1 Macro", f"{metrics.get('f1_macro', 0):.4f}")
                        with col3:
                            st.metric("ECE", f"{metrics.get('ece', 0):.4f}")
                        with col4:
                            st.metric("Brier", f"{metrics.get('brier', 0):.4f}")

                        # Show full results
                        with st.expander("Full Results"):
                            st.text(content)

                    else:
                        st.info("No pre-computed results. Running live evaluation...")

    # ==========================================================================
    # TAB 3: Model Comparison
    # ==========================================================================
    with tab3:
        st.subheader("Model Comparison")
        st.markdown("Compare multiple models side-by-side on the same dataset.")

        datasets = get_available_datasets()
        compare_dataset = st.selectbox(
            "Dataset",
            options=list(datasets.keys()),
            format_func=lambda x: f"{datasets[x]} ({x})",
            key="compare_dataset"
        )

        # Find all available models for this dataset
        results_path = os.path.join(RESULTS_DIR, compare_dataset)
        models_with_results = []

        if os.path.exists(results_path):
            for model_dir in os.listdir(results_path):
                result_file = os.path.join(results_path, model_dir, 'result_classification.txt')
                if os.path.exists(result_file):
                    models_with_results.append(model_dir)

        if models_with_results:
            selected_models = st.multiselect(
                "Select Models to Compare",
                options=models_with_results,
                default=models_with_results[:min(5, len(models_with_results))]
            )

            if selected_models and len(selected_models) >= 2:
                # Load results for selected models
                comparison_data = []

                for model in selected_models:
                    result_file = os.path.join(results_path, model, 'result_classification.txt')
                    with open(result_file, 'r') as f:
                        content = f.read()

                    metrics = {'model': model}
                    for line in content.split('\n'):
                        if ':' in line:
                            key, val = line.split(':', 1)
                            try:
                                metrics[key.strip().lower()] = float(val.strip())
                            except:
                                continue

                    comparison_data.append(metrics)

                df_compare = pd.DataFrame(comparison_data)

                # Comparison bar chart
                st.markdown("#### Accuracy Comparison")

                fig = go.Figure()
                df_sorted = df_compare.sort_values('accuracy', ascending=True)

                fig.add_trace(go.Bar(
                    y=df_sorted['model'],
                    x=df_sorted['accuracy'],
                    orientation='h',
                    text=[f"{v:.2%}" for v in df_sorted['accuracy']],
                    textposition='outside',
                    marker_color='steelblue'
                ))

                fig.update_layout(
                    xaxis_title="Accuracy",
                    yaxis_title="Model",
                    height=max(300, len(selected_models) * 40),
                    xaxis=dict(tickformat='.0%', range=[0, 1.1])
                )

                st.plotly_chart(fig, use_container_width=True)

                # Multi-metric comparison
                st.markdown("#### Multi-Metric Comparison")

                metrics_to_compare = ['accuracy', 'f1_micro', 'f1_macro', 'f1_weighted']
                available_metrics = [m for m in metrics_to_compare if m in df_compare.columns]

                fig = go.Figure()

                for metric in available_metrics:
                    fig.add_trace(go.Bar(
                        name=metric.replace('_', ' ').title(),
                        x=df_compare['model'],
                        y=df_compare[metric],
                        text=[f"{v:.3f}" for v in df_compare[metric]],
                        textposition='outside'
                    ))

                fig.update_layout(
                    barmode='group',
                    xaxis_title="Model",
                    yaxis_title="Score",
                    height=450,
                    yaxis=dict(range=[0, 1.1])
                )

                st.plotly_chart(fig, use_container_width=True)

                # Comparison table
                st.markdown("#### Detailed Comparison")
                display_df = df_compare.set_index('model')[available_metrics]
                st.dataframe(
                    display_df.style.format("{:.4f}").highlight_max(axis=0),
                    use_container_width=True
                )

        else:
            st.info("No models with results found for this dataset. Train some models first.")

    # ==========================================================================
    # TAB 4: Signal Analysis
    # ==========================================================================
    with tab4:
        st.subheader("Signal Analysis")
        st.markdown("Analyze time series signals and their characteristics.")

        datasets = get_available_datasets()
        analysis_dataset = st.selectbox(
            "Dataset",
            options=list(datasets.keys()),
            format_func=lambda x: f"{datasets[x]} ({x})",
            key="analysis_dataset"
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Frequency Spectrum", "Statistical Analysis", "Wavelet Transform", "Envelope Analysis"]
            )

        with col2:
            sample_range = st.slider(
                "Sample Range",
                0, 100, (0, 10),
                help="Range of samples to analyze"
            )

        # Load data
        try:
            data_path = os.path.join(DATASET_DIR, analysis_dataset, 'PdM_TRAIN.npz')
            if os.path.exists(data_path):
                loaded = np.load(data_path, allow_pickle=True)
                df_restored = pd.DataFrame(loaded['data'], columns=loaded['columns'])
                features_list = df_restored['features'].tolist()
                labels = df_restored['label'].values

                # Get samples in range
                samples_to_analyze = features_list[sample_range[0]:sample_range[1]+1]
                sample_labels = labels[sample_range[0]:sample_range[1]+1]

                if samples_to_analyze:
                    sample = samples_to_analyze[0]
                    if isinstance(sample, list):
                        sample = np.array(sample)

                    if analysis_type == "Frequency Spectrum":
                        st.markdown("#### Frequency Spectrum Analysis")

                        # FFT analysis
                        if len(sample.shape) == 1:
                            signal = sample
                        else:
                            signal = sample[:, 0]

                        fft = np.fft.rfft(signal)
                        freqs = np.fft.rfftfreq(len(signal))
                        magnitude = np.abs(fft)

                        fig = make_subplots(rows=2, cols=1, subplot_titles=["Time Domain", "Frequency Domain"])

                        fig.add_trace(
                            go.Scatter(y=signal, mode='lines', name='Signal'),
                            row=1, col=1
                        )

                        fig.add_trace(
                            go.Scatter(x=freqs, y=magnitude, mode='lines', name='Magnitude'),
                            row=2, col=1
                        )

                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)

                        # Peak frequencies
                        peak_indices = np.argsort(magnitude)[-5:]
                        st.markdown("**Top 5 Peak Frequencies:**")
                        for idx in peak_indices[::-1]:
                            st.text(f"  Frequency: {freqs[idx]:.4f}, Magnitude: {magnitude[idx]:.2f}")

                    elif analysis_type == "Statistical Analysis":
                        st.markdown("#### Statistical Analysis")

                        if len(sample.shape) == 1:
                            signal = sample
                        else:
                            signal = sample[:, 0]

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Mean", f"{np.mean(signal):.4f}")
                            st.metric("Min", f"{np.min(signal):.4f}")

                        with col2:
                            st.metric("Std Dev", f"{np.std(signal):.4f}")
                            st.metric("Max", f"{np.max(signal):.4f}")

                        with col3:
                            st.metric("Median", f"{np.median(signal):.4f}")
                            st.metric("Range", f"{np.ptp(signal):.4f}")

                        with col4:
                            from scipy.stats import skew, kurtosis
                            st.metric("Skewness", f"{skew(signal):.4f}")
                            st.metric("Kurtosis", f"{kurtosis(signal):.4f}")

                        # Histogram
                        fig = px.histogram(signal, nbins=50, title="Value Distribution")
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)

                    elif analysis_type == "Wavelet Transform":
                        st.markdown("#### Wavelet Transform Analysis")
                        st.info("Wavelet analysis requires the PyWavelets package. Install with: pip install PyWavelets")

                    elif analysis_type == "Envelope Analysis":
                        st.markdown("#### Envelope Analysis")

                        if len(sample.shape) == 1:
                            signal = sample
                        else:
                            signal = sample[:, 0]

                        # Hilbert transform for envelope
                        from scipy.signal import hilbert

                        analytic_signal = hilbert(signal)
                        envelope = np.abs(analytic_signal)

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=signal, mode='lines', name='Signal', opacity=0.5))
                        fig.add_trace(go.Scatter(y=envelope, mode='lines', name='Envelope', line=dict(color='red')))
                        fig.add_trace(go.Scatter(y=-envelope, mode='lines', name='Envelope', line=dict(color='red'), showlegend=False))

                        fig.update_layout(
                            title="Signal Envelope",
                            xaxis_title="Time Step",
                            yaxis_title="Amplitude",
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("Dataset not found")

        except Exception as e:
            st.error(f"Error in analysis: {e}")


# =============================================================================
# API DOCUMENTATION PAGE
# =============================================================================

def python_api_page():
    """Python API Documentation page - shows how to use pdm_framework package."""
    st.title("Python API")
    st.markdown("Use PDMBench programmatically with our Python package `pdm_framework`.")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Quick Start", "Datasets", "Training", "Evaluation"
    ])

    # =========================================================================
    # TAB 1: Quick Start
    # =========================================================================
    with tab1:
        st.markdown("### Installation")
        st.code("""
# Install from the PDMBench directory
pip install -e ./PDMBench

# Or add to your Python path
import sys
sys.path.append('/path/to/PDMBench')
""", language='bash')

        st.markdown("### Quick Start Example")
        st.code('''
from pdm_framework import PDMDataset, Trainer, Evaluator
from pdm_framework.models import SimpleLSTM

# 1. Load a dataset
dataset = PDMDataset('01')  # Paderborn bearing dataset
print(dataset)  # Shows: samples, classes, seq_len, features

# 2. Get data loaders
train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=32)

# 3. Create a model
model = SimpleLSTM(
    seq_len=dataset.seq_len,
    num_features=dataset.num_features,
    num_classes=dataset.num_classes
)

# 4. Train
trainer = Trainer(model, train_loader, val_loader)
history = trainer.fit(epochs=50)

# 5. Evaluate
evaluator = Evaluator(model, test_loader)
metrics = evaluator.evaluate()
print(f"Accuracy: {metrics.accuracy:.4f}")
print(f"F1 Score: {metrics.f1:.4f}")
''', language='python')

        st.markdown("### Available Datasets")
        st.code('''
from pdm_framework import list_datasets, get_dataset_info

# List all datasets
print(list_datasets())
# ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

# Get dataset info
info = get_dataset_info('01')
print(info)
# {'name': 'Paderborn Bearing', 'description': '...', 'num_classes': 3}
''', language='python')

    # =========================================================================
    # TAB 2: Datasets
    # =========================================================================
    with tab2:
        st.markdown("### PDMDataset Class")

        st.code('''
from pdm_framework import PDMDataset

# Basic usage
dataset = PDMDataset('01')

# With custom root path
dataset = PDMDataset('01', root_path='/path/to/datasets/01')

# With sequence length truncation
dataset = PDMDataset('01', seq_len=256)

# With data augmentation
from pdm_framework.transforms import Compose, Jitter, Scale

transform = Compose([
    Jitter(sigma=0.03),
    Scale(sigma=0.1)
])
dataset = PDMDataset('01', transform=transform)
''', language='python')

        st.markdown("### Accessing Data")
        st.code('''
# Access raw numpy arrays
X_train, y_train = dataset.train_data
X_val, y_val = dataset.val_data
X_test, y_test = dataset.test_data

# Data shapes
print(f"Train: {X_train.shape}")  # (N, seq_len, features)
print(f"Labels: {y_train.shape}")  # (N,)

# Dataset properties
print(f"Sequence length: {dataset.seq_len}")
print(f"Number of features: {dataset.num_features}")
print(f"Number of classes: {dataset.num_classes}")

# Class distribution
dist = dataset.get_class_distribution()
print(dist)  # {'train': {0: 100, 1: 150, ...}, 'val': {...}, 'test': {...}}
''', language='python')

        st.markdown("### Data Loaders")
        st.code('''
# Get PyTorch DataLoaders
train_loader, val_loader, test_loader = dataset.get_loaders(
    batch_size=32,
    num_workers=4,
    shuffle_train=True
)

# Iterate over batches
for batch_x, batch_y in train_loader:
    print(batch_x.shape)  # torch.Size([32, seq_len, features])
    print(batch_y.shape)  # torch.Size([32])
    break
''', language='python')

    # =========================================================================
    # TAB 3: Training
    # =========================================================================
    with tab3:
        st.markdown("### Built-in Models")
        st.code('''
from pdm_framework.models import (
    SimpleMLP,
    SimpleLSTM,
    SimpleCNN,
    SimpleTransformer,
    get_model
)

# Create models directly
lstm = SimpleLSTM(seq_len=512, num_features=7, num_classes=3)
cnn = SimpleCNN(seq_len=512, num_features=7, num_classes=3)
transformer = SimpleTransformer(seq_len=512, num_features=7, num_classes=3)

# Or use factory function
model = get_model('lstm', seq_len=512, num_features=7, num_classes=3)
''', language='python')

        st.markdown("### Custom Models")
        st.code('''
from pdm_framework.models import PDMModel
import torch.nn as nn

class MyModel(PDMModel):
    """Custom model inheriting from PDMModel."""

    def __init__(self, seq_len, num_features, num_classes):
        super().__init__(seq_len, num_features, num_classes)

        self.encoder = nn.LSTM(num_features, 128, batch_first=True)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, features]
        _, (h, _) = self.encoder(x)
        return self.classifier(h[-1])

# Use with trainer
model = MyModel(dataset.seq_len, dataset.num_features, dataset.num_classes)
''', language='python')

        st.markdown("### Trainer Configuration")
        st.code('''
from pdm_framework import Trainer, TrainerConfig

# Configure training
config = TrainerConfig(
    optimizer='adamw',
    learning_rate=1e-3,
    weight_decay=1e-4,
    scheduler='cosine',
    epochs=100,
    patience=10,
    use_amp=True  # Mixed precision
)

# Create trainer
trainer = Trainer(model, train_loader, val_loader, config=config)

# Train and get history
history = trainer.fit()

# Access training history
print(history['train_loss'])  # Loss per epoch
print(history['val_acc'])     # Validation accuracy per epoch

# Save/load checkpoints
trainer.save_checkpoint('model.pt')
trainer.load_checkpoint('model.pt')
''', language='python')

    # =========================================================================
    # TAB 4: Evaluation
    # =========================================================================
    with tab4:
        st.markdown("### Evaluator Class")
        st.code('''
from pdm_framework import Evaluator

# Create evaluator
evaluator = Evaluator(model, test_loader)

# Run evaluation
metrics = evaluator.evaluate()

# Access metrics
print(f"Accuracy:  {metrics.accuracy:.4f}")
print(f"Precision: {metrics.precision:.4f}")
print(f"Recall:    {metrics.recall:.4f}")
print(f"F1:        {metrics.f1:.4f}")
print(f"AUC-ROC:   {metrics.auc_roc:.4f}")

# Per-class metrics
print(metrics.per_class_precision)
print(metrics.per_class_recall)
print(metrics.per_class_f1)

# Confusion matrix
print(metrics.confusion_matrix)

# Get predictions and probabilities
print(metrics.predictions)    # Predicted labels
print(metrics.probabilities)  # Class probabilities
''', language='python')

        st.markdown("### Error Analysis")
        st.code('''
# Detailed error analysis
analysis = evaluator.error_analysis()
print(f"Misclassified: {analysis['num_misclassified']}")
print(f"Error rate: {analysis['error_rate']:.4f}")
print(f"Avg correct confidence: {analysis['avg_correct_confidence']:.4f}")
print(f"Top confusions: {analysis['top_confusions']}")

# Get worst predictions (highest confidence errors)
worst = evaluator.get_worst_predictions(n=10)
for w in worst:
    print(f"Sample {w['sample_index']}: "
          f"true={w['true_label']}, pred={w['predicted_label']}, "
          f"conf={w['confidence']:.4f}")

# Per-class performance
class_perf = evaluator.class_performance_summary()
for cls, perf in class_perf.items():
    print(f"Class {cls}: acc={perf['accuracy']:.4f}, f1={perf['f1']:.4f}")
''', language='python')

        st.markdown("### Model Comparison")
        st.code('''
from pdm_framework import compare_models, print_comparison_table

# Compare multiple models
models = {
    'LSTM': lstm_model,
    'CNN': cnn_model,
    'Transformer': transformer_model
}

results = compare_models(models, test_loader)
print_comparison_table(results)

# Output:
# ======================================================================
# Model Comparison
# ======================================================================
# Model                Accuracy     Precision    Recall       F1
# ----------------------------------------------------------------------
# Transformer          0.9450       0.9423       0.9450       0.9435
# CNN                  0.9320       0.9298       0.9320       0.9308
# LSTM                 0.9180       0.9156       0.9180       0.9167
# ======================================================================
''', language='python')

    st.markdown("---")
    st.subheader("Data Transforms")

    st.code('''
from pdm_framework.transforms import (
    Compose, Normalize, Standardize,
    Jitter, Scale, TimeWarp, MagnitudeWarp,
    Cutout, FrequencyMask, GaussianNoise
)

# Create augmentation pipeline
augmentation = Compose([
    Normalize(),                    # Normalize to [0, 1]
    Jitter(sigma=0.03),            # Add noise
    Scale(sigma=0.1),              # Random scaling
    TimeWarp(sigma=0.2),           # Time warping
    Cutout(n_holes=1, length_ratio=0.1)  # Random masking
])

# Apply to dataset
dataset = PDMDataset('01', transform=augmentation)
''', language='python')

    st.markdown("---")
    st.info("""
    **Full Documentation**: See `pdm_framework/` directory for complete source code and examples.

    **Examples**: Check `pdm_framework/examples/` for working scripts:
    - `quick_start.py` - Basic usage
    - `compare_models.py` - Model comparison
    - `custom_model.py` - Custom model integration
    - `augmentation.py` - Data augmentation
    """)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point."""
    # Initialize session
    session_id = init_session()

    st.sidebar.title("PDM Benchmark")

    # Show session info
    with st.sidebar.expander("Session Info", expanded=False):
        st.text(f"Session ID: {session_id}")
        if 'session_created' in st.session_state:
            elapsed = datetime.now() - st.session_state.session_created
            st.text(f"Active: {int(elapsed.total_seconds() // 60)} min")
        st.text(f"Expires: {SESSION_TIMEOUT_MINUTES} min inactivity")

    st.sidebar.markdown("---")

    # Dataset Explorer first - determines data processing for the session
    page = st.sidebar.radio(
        "Navigate",
        options=[
            "Dataset Explorer",      # First - explore and configure data processing
            "Model Training",        # Second - train models (includes custom model option)
            "Results Analysis",      # Third - analyze results
            "Python API"             # Last - Python package documentation
        ]
    )

    if page == "Dataset Explorer":
        dataset_explorer_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Results Analysis":
        results_analysis_page()
    elif page == "Python API":
        python_api_page()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("PDM Benchmark v2.0")


if __name__ == "__main__":
    main()
