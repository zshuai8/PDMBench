"""
Configuration Manager for PDMBenchmark

Provides centralized configuration management with:
- YAML-based configuration files
- Preset configurations for different scenarios
- Model-specific default parameters
- Runtime configuration overrides
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import copy


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


@dataclass
class TrainingConfig:
    """Training-related configuration."""
    batch_size: int = 32
    learning_rate: float = 0.001
    train_epochs: int = 50
    patience: int = 7
    num_workers: int = 4
    use_amp: bool = False
    seed: int = 2021


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    d_model: int = 512
    d_ff: int = 2048
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    dropout: float = 0.1
    activation: str = "gelu"
    embed: str = "timeF"
    # Model-specific
    top_k: int = 5
    num_kernels: int = 6
    expand: int = 2
    d_conv: int = 4
    patch_len: int = 16
    stride: int = 8
    seg_len: int = 96
    moving_avg: int = 25


@dataclass
class DataConfig:
    """Data-related configuration."""
    seq_len: int = 512
    label_len: int = 48
    pred_len: int = 0
    features: str = "M"
    channel_independence: int = 1
    use_norm: int = 1


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    enabled: bool = False
    ratio: int = 0
    jitter: bool = False
    scaling: bool = False
    permutation: bool = False
    magwarp: bool = False
    timewarp: bool = False
    rotation: bool = False


@dataclass
class GPUConfig:
    """GPU configuration."""
    use_gpu: bool = False  # Default to CPU for stability
    gpu_id: int = 0
    gpu_type: str = "cuda"
    use_multi_gpu: bool = False
    device_ids: str = "0,1,2,3"


class ConfigManager:
    """
    Central configuration manager for PDMBenchmark.

    Usage:
        config_manager = ConfigManager()

        # Load a preset
        config = config_manager.get_preset('balanced')

        # Get model-specific config
        config = config_manager.get_model_config('TimesNet')

        # Create args namespace
        args = config_manager.create_args(preset='balanced', model='TimesNet')
    """

    def __init__(self, config_dir: str = None):
        """Initialize the configuration manager."""
        if config_dir is None:
            # Try to find configs directory relative to this file
            self.config_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'configs'
            )
        else:
            self.config_dir = config_dir

        self.default_config = self._load_default()
        self.presets = self._load_presets()

    def _load_default(self) -> Dict[str, Any]:
        """Load default configuration."""
        default_path = os.path.join(self.config_dir, 'default.yaml')
        if os.path.exists(default_path):
            return load_yaml(default_path)
        return {}

    def _load_presets(self) -> Dict[str, Any]:
        """Load preset configurations."""
        presets_path = os.path.join(self.config_dir, 'presets.yaml')
        if os.path.exists(presets_path):
            return load_yaml(presets_path)
        return {}

    def get_preset_names(self) -> list:
        """Get list of available preset names."""
        return list(self.presets.keys())

    def get_preset_description(self, preset_name: str) -> str:
        """Get description for a preset."""
        if preset_name in self.presets:
            return self.presets[preset_name].get('description', 'No description available')
        return 'Preset not found'

    def get_preset(self, preset_name: str) -> Dict[str, Any]:
        """Get a preset configuration merged with defaults."""
        if preset_name not in self.presets:
            raise ValueError(f"Preset '{preset_name}' not found. Available: {self.get_preset_names()}")

        preset = self.presets[preset_name]
        return deep_merge(self.default_config, preset)

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration optimized for a specific model."""
        model_lower = model_name.lower()

        # Check for model-specific preset
        if model_lower in self.presets:
            return self.get_preset(model_lower)

        # Return default config
        return copy.deepcopy(self.default_config)

    def create_args(
        self,
        preset: str = None,
        model: str = None,
        dataset: str = None,
        overrides: Dict[str, Any] = None
    ):
        """
        Create an args namespace for training.

        Args:
            preset: Name of preset to use (e.g., 'balanced', 'quick')
            model: Model name for model-specific defaults
            dataset: Dataset ID for dataset-specific defaults
            overrides: Dictionary of values to override

        Returns:
            argparse.Namespace-like object with all configuration
        """
        # Start with default config
        config = copy.deepcopy(self.default_config)

        # Apply preset if specified
        if preset:
            config = deep_merge(config, self.presets.get(preset, {}))

        # Apply model-specific config
        if model:
            model_lower = model.lower()
            if model_lower in self.presets:
                config = deep_merge(config, self.presets[model_lower])

        # Apply dataset-specific config
        if dataset:
            dataset_presets = {
                '01': 'paderborn',
                '02': 'cwru',
                '06': 'high_frequency',
            }
            if dataset in dataset_presets and dataset_presets[dataset] in self.presets:
                config = deep_merge(config, self.presets[dataset_presets[dataset]])

        # Apply overrides
        if overrides:
            config = deep_merge(config, overrides)

        # Convert to Args object
        return self._config_to_args(config, model, dataset)

    def _config_to_args(self, config: Dict, model: str = None, dataset: str = None):
        """Convert configuration dictionary to args namespace."""
        class Args:
            pass

        args = Args()

        # Basic settings
        args.task_name = 'classification'
        args.is_training = 1
        args.model = model or 'TimesNet'
        args.data = 'PDM'  # Data loader type
        args.dataset_id = dataset or '01'  # Dataset ID
        args.model_id = f"{args.model}_{args.dataset_id}"

        # Training config
        training = config.get('training', {})
        args.batch_size = training.get('batch_size', 32)
        args.learning_rate = training.get('learning_rate', 0.001)
        args.train_epochs = training.get('train_epochs', 50)
        args.patience = training.get('patience', 7)
        args.num_workers = training.get('num_workers', 4)
        args.use_amp = training.get('use_amp', False)
        args.seed = training.get('seed', 2021)
        args.itr = 1
        args.lradj = 'type1'
        args.loss = 'MSE'
        args.des = 'Exp'

        # Model config
        model_cfg = config.get('model', {})
        args.d_model = model_cfg.get('d_model', 512)
        args.d_ff = model_cfg.get('d_ff', 2048)
        args.n_heads = model_cfg.get('n_heads', 8)
        args.e_layers = model_cfg.get('e_layers', 2)
        args.d_layers = model_cfg.get('d_layers', 1)
        args.dropout = model_cfg.get('dropout', 0.1)
        args.activation = model_cfg.get('activation', 'gelu')
        args.embed = model_cfg.get('embed', 'timeF')
        args.factor = 1
        args.distil = True

        # Model-specific config
        model_specific = config.get('model_specific', {})
        args.top_k = model_specific.get('top_k', 5)
        args.num_kernels = model_specific.get('num_kernels', 6)
        args.expand = model_specific.get('expand', 2)
        args.d_conv = model_specific.get('d_conv', 4)
        args.patch_len = model_specific.get('patch_len', 16)
        args.seg_len = model_specific.get('seg_len', 96)
        args.moving_avg = model_specific.get('moving_avg', 25)
        args.project_input_shape = 96

        # Data config
        data_cfg = config.get('data', {})
        args.seq_len = data_cfg.get('seq_len', 512)
        args.label_len = data_cfg.get('label_len', 48)
        args.pred_len = data_cfg.get('pred_len', 0)
        args.features = data_cfg.get('features', 'M')
        args.channel_independence = data_cfg.get('channel_independence', 1)
        args.use_norm = data_cfg.get('use_norm', 1)
        args.target = 'OT'
        args.freq = 'h'
        args.seasonal_patterns = 'Monthly'
        args.inverse = False
        args.mask_rate = 0.25
        args.anomaly_ratio = 0.25
        args.decomp_method = 'moving_avg'
        args.down_sampling_layers = 0
        args.down_sampling_window = 1
        args.down_sampling_method = None

        # Augmentation config
        aug_cfg = config.get('augmentation', {})
        args.augmentation_ratio = aug_cfg.get('ratio', 0) if aug_cfg.get('enabled', False) else 0
        args.jitter = aug_cfg.get('jitter', False)
        args.scaling = aug_cfg.get('scaling', False)
        args.permutation = aug_cfg.get('permutation', False)
        args.randompermutation = False
        args.magwarp = aug_cfg.get('magwarp', False)
        args.timewarp = aug_cfg.get('timewarp', False)
        args.windowslice = False
        args.windowwarp = False
        args.rotation = aug_cfg.get('rotation', False)
        args.spawner = False
        args.dtwwarp = False
        args.shapedtwwarp = False
        args.wdba = False
        args.discdtw = False
        args.discsdtw = False
        args.extra_tag = ""

        # GPU config
        gpu_cfg = config.get('gpu', {})
        args.use_gpu = gpu_cfg.get('use_gpu', False)  # Default to CPU
        args.gpu = gpu_cfg.get('gpu_id', 0)
        args.gpu_type = gpu_cfg.get('gpu_type', 'cuda')
        args.use_multi_gpu = gpu_cfg.get('use_multi_gpu', False)
        args.devices = gpu_cfg.get('device_ids', '0,1,2,3')
        args.device_ids = [int(x) for x in args.devices.split(',')]

        # Logging config
        logging_cfg = config.get('logging', {})
        args.use_wandb = logging_cfg.get('use_wandb', False)
        args.checkpoints = logging_cfg.get('checkpoints_dir', './checkpoints/')

        # Path config
        args.root_path = f'./dataset/{args.dataset_id}/'
        args.file_list = ['PdM_TRAIN.npz', 'PdM_VAL.npz', 'PdM_TEST.npz']
        args.data_path = 'PdM.csv'

        # Additional
        args.enc_in = 7
        args.dec_in = 7
        args.c_out = 7
        args.use_dtw = False
        args.p_hidden_dims = [128, 128]
        args.p_hidden_layers = 2

        return args


# Singleton instance for easy access
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


# Convenience functions
def get_preset(name: str) -> Dict[str, Any]:
    """Get a preset configuration."""
    return get_config_manager().get_preset(name)


def get_preset_names() -> list:
    """Get list of available preset names."""
    return get_config_manager().get_preset_names()


def create_args(preset: str = None, model: str = None, dataset: str = None, **overrides):
    """Create training args with optional preset and overrides."""
    return get_config_manager().create_args(
        preset=preset,
        model=model,
        dataset=dataset,
        overrides=overrides if overrides else None
    )