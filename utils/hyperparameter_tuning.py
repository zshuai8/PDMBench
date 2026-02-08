"""
Hyperparameter Tuning Framework for PDMBench.

Provides GridSearchTuner, RandomSearchTuner, and SensitivityAnalyzer for
systematic hyperparameter optimization and analysis.
"""

import os
import json
import yaml
import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Optional, Union, Tuple
from dataclasses import dataclass, field
from copy import deepcopy
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


@dataclass
class SearchSpace:
    """
    Definition of a hyperparameter search space.

    Args:
        name: Parameter name
        distribution: Type of distribution ('values', 'uniform', 'log_uniform', 'int_uniform')
        values: List of discrete values (for 'values' distribution)
        low: Lower bound (for continuous distributions)
        high: Upper bound (for continuous distributions)
    """
    name: str
    distribution: str = 'values'
    values: Optional[List[Any]] = None
    low: Optional[float] = None
    high: Optional[float] = None

    def __post_init__(self):
        if self.distribution == 'values' and self.values is None:
            raise ValueError(f"SearchSpace '{self.name}' with distribution='values' requires 'values' list")
        if self.distribution in ['uniform', 'log_uniform', 'int_uniform']:
            if self.low is None or self.high is None:
                raise ValueError(f"SearchSpace '{self.name}' with distribution='{self.distribution}' "
                               f"requires 'low' and 'high' bounds")

    def sample(self, rng: Optional[np.random.Generator] = None) -> Any:
        """Sample a value from this search space."""
        if rng is None:
            rng = np.random.default_rng()

        if self.distribution == 'values':
            return rng.choice(self.values)
        elif self.distribution == 'uniform':
            return rng.uniform(self.low, self.high)
        elif self.distribution == 'log_uniform':
            log_low = np.log(self.low)
            log_high = np.log(self.high)
            return np.exp(rng.uniform(log_low, log_high))
        elif self.distribution == 'int_uniform':
            return int(rng.integers(int(self.low), int(self.high) + 1))
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def get_grid_values(self, n_points: int = 10) -> List[Any]:
        """Get grid values for this search space."""
        if self.distribution == 'values':
            return list(self.values)
        elif self.distribution == 'uniform':
            return np.linspace(self.low, self.high, n_points).tolist()
        elif self.distribution == 'log_uniform':
            return np.geomspace(self.low, self.high, n_points).tolist()
        elif self.distribution == 'int_uniform':
            step = max(1, (self.high - self.low) // (n_points - 1))
            return list(range(int(self.low), int(self.high) + 1, int(step)))
        else:
            return [self.low]


@dataclass
class TuningResult:
    """Result of a single tuning trial."""
    params: Dict[str, Any]
    metrics: Dict[str, float]
    trial_id: int
    duration_seconds: float = 0.0
    status: str = 'completed'  # completed, failed, timeout
    error_message: Optional[str] = None


class BaseTuner:
    """Base class for hyperparameter tuners."""

    def __init__(
        self,
        search_spaces: List[SearchSpace],
        train_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        metric: str = 'val_accuracy',
        direction: str = 'maximize',
        seed: int = 42,
        n_workers: int = 1,
        output_dir: str = './tuning_results'
    ):
        """
        Args:
            search_spaces: List of SearchSpace objects defining the search space
            train_fn: Function that takes params dict and returns metrics dict
            metric: Metric name to optimize
            direction: 'maximize' or 'minimize'
            seed: Random seed for reproducibility
            n_workers: Number of parallel workers
            output_dir: Directory to save results
        """
        self.search_spaces = {ss.name: ss for ss in search_spaces}
        self.train_fn = train_fn
        self.metric = metric
        self.direction = direction
        self.seed = seed
        self.n_workers = n_workers
        self.output_dir = output_dir
        self.rng = np.random.default_rng(seed)

        self.results: List[TuningResult] = []
        self.best_result: Optional[TuningResult] = None

        os.makedirs(output_dir, exist_ok=True)

    def search(self) -> Tuple[Dict[str, Any], List[TuningResult]]:
        """
        Run hyperparameter search.

        Returns:
            Tuple of (best_params, all_results)
        """
        raise NotImplementedError

    def _evaluate_params(self, params: Dict[str, Any], trial_id: int) -> TuningResult:
        """Evaluate a single parameter configuration."""
        import time
        start_time = time.time()

        try:
            metrics = self.train_fn(params)
            duration = time.time() - start_time
            result = TuningResult(
                params=params,
                metrics=metrics,
                trial_id=trial_id,
                duration_seconds=duration,
                status='completed'
            )
        except Exception as e:
            duration = time.time() - start_time
            result = TuningResult(
                params=params,
                metrics={self.metric: float('-inf') if self.direction == 'maximize' else float('inf')},
                trial_id=trial_id,
                duration_seconds=duration,
                status='failed',
                error_message=str(e)
            )

        return result

    def _is_better(self, new_result: TuningResult, current_best: Optional[TuningResult]) -> bool:
        """Check if new result is better than current best."""
        if current_best is None:
            return True

        new_value = new_result.metrics.get(self.metric, float('-inf') if self.direction == 'maximize' else float('inf'))
        best_value = current_best.metrics.get(self.metric, float('-inf') if self.direction == 'maximize' else float('inf'))

        if self.direction == 'maximize':
            return new_value > best_value
        else:
            return new_value < best_value

    def _save_results(self, filename: str = 'tuning_results.json'):
        """Save results to JSON file."""
        results_data = {
            'best_params': self.best_result.params if self.best_result else None,
            'best_metrics': self.best_result.metrics if self.best_result else None,
            'metric': self.metric,
            'direction': self.direction,
            'n_trials': len(self.results),
            'trials': [
                {
                    'trial_id': r.trial_id,
                    'params': r.params,
                    'metrics': r.metrics,
                    'duration_seconds': r.duration_seconds,
                    'status': r.status,
                    'error_message': r.error_message
                }
                for r in self.results
            ]
        }

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        return filepath


class GridSearchTuner(BaseTuner):
    """
    Grid search over all combinations of hyperparameters.
    """

    def __init__(
        self,
        search_spaces: List[SearchSpace],
        train_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        metric: str = 'val_accuracy',
        direction: str = 'maximize',
        seed: int = 42,
        n_workers: int = 1,
        output_dir: str = './tuning_results',
        max_combinations: int = 1000
    ):
        super().__init__(search_spaces, train_fn, metric, direction, seed, n_workers, output_dir)
        self.max_combinations = max_combinations

    def search(self) -> Tuple[Dict[str, Any], List[TuningResult]]:
        """Run grid search."""
        # Generate all parameter combinations
        param_names = list(self.search_spaces.keys())
        param_values = [self.search_spaces[name].get_grid_values() for name in param_names]

        combinations = list(itertools.product(*param_values))
        n_combinations = len(combinations)

        if n_combinations > self.max_combinations:
            warnings.warn(f"Grid search has {n_combinations} combinations, which exceeds "
                        f"max_combinations={self.max_combinations}. Consider using random search.")

        print(f"Starting grid search with {n_combinations} combinations...")

        self.results = []
        for trial_id, values in enumerate(combinations):
            params = dict(zip(param_names, values))
            print(f"Trial {trial_id + 1}/{n_combinations}: {params}")

            result = self._evaluate_params(params, trial_id)
            self.results.append(result)

            if self._is_better(result, self.best_result):
                self.best_result = result
                print(f"  New best! {self.metric}={result.metrics.get(self.metric, 'N/A')}")

        self._save_results('grid_search_results.json')
        print(f"\nBest params: {self.best_result.params}")
        print(f"Best {self.metric}: {self.best_result.metrics.get(self.metric, 'N/A')}")

        return self.best_result.params, self.results


class RandomSearchTuner(BaseTuner):
    """
    Random search sampling from hyperparameter distributions.
    """

    def __init__(
        self,
        search_spaces: List[SearchSpace],
        train_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        metric: str = 'val_accuracy',
        direction: str = 'maximize',
        seed: int = 42,
        n_workers: int = 1,
        output_dir: str = './tuning_results',
        n_trials: int = 50
    ):
        super().__init__(search_spaces, train_fn, metric, direction, seed, n_workers, output_dir)
        self.n_trials = n_trials

    def search(self) -> Tuple[Dict[str, Any], List[TuningResult]]:
        """Run random search."""
        print(f"Starting random search with {self.n_trials} trials...")

        self.results = []
        for trial_id in range(self.n_trials):
            # Sample random parameters
            params = {name: ss.sample(self.rng) for name, ss in self.search_spaces.items()}
            print(f"Trial {trial_id + 1}/{self.n_trials}: {params}")

            result = self._evaluate_params(params, trial_id)
            self.results.append(result)

            if self._is_better(result, self.best_result):
                self.best_result = result
                print(f"  New best! {self.metric}={result.metrics.get(self.metric, 'N/A')}")

        self._save_results('random_search_results.json')
        print(f"\nBest params: {self.best_result.params}")
        print(f"Best {self.metric}: {self.best_result.metrics.get(self.metric, 'N/A')}")

        return self.best_result.params, self.results


class SensitivityAnalyzer:
    """
    Analyze sensitivity of model performance to hyperparameters.

    Performs one-factor-at-a-time (OFAT) analysis and computes
    parameter importance scores.
    """

    def __init__(
        self,
        results: List[TuningResult],
        metric: str = 'val_accuracy',
        direction: str = 'maximize'
    ):
        """
        Args:
            results: List of TuningResult from tuning
            metric: Metric to analyze
            direction: 'maximize' or 'minimize'
        """
        self.results = results
        self.metric = metric
        self.direction = direction
        self.df = self._build_dataframe()

    def _build_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        data = []
        for r in self.results:
            row = {**r.params, self.metric: r.metrics.get(self.metric, np.nan)}
            data.append(row)
        return pd.DataFrame(data)

    def get_parameter_importance(self) -> Dict[str, float]:
        """
        Compute parameter importance using variance-based sensitivity.

        Returns:
            Dictionary mapping parameter names to importance scores (0-1)
        """
        param_columns = [c for c in self.df.columns if c != self.metric]

        importances = {}
        total_variance = self.df[self.metric].var()

        if total_variance == 0:
            return {p: 0.0 for p in param_columns}

        for param in param_columns:
            # Group by parameter value and compute variance reduction
            grouped = self.df.groupby(param)[self.metric]
            between_group_variance = grouped.mean().var()
            within_group_variance = grouped.var().mean()

            # Importance = fraction of variance explained by this parameter
            importance = between_group_variance / total_variance if total_variance > 0 else 0
            importances[param] = float(np.clip(importance, 0, 1))

        # Normalize to sum to 1
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}

        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    def get_parameter_statistics(self, param: str) -> Dict[str, Any]:
        """
        Get statistics for a specific parameter.

        Args:
            param: Parameter name

        Returns:
            Dictionary with statistics
        """
        if param not in self.df.columns:
            raise ValueError(f"Parameter '{param}' not found in results")

        grouped = self.df.groupby(param)[self.metric]

        stats = {
            'param': param,
            'unique_values': self.df[param].nunique(),
            'value_counts': self.df[param].value_counts().to_dict(),
            'mean_by_value': grouped.mean().to_dict(),
            'std_by_value': grouped.std().to_dict(),
            'best_value': grouped.mean().idxmax() if self.direction == 'maximize' else grouped.mean().idxmin(),
            'worst_value': grouped.mean().idxmin() if self.direction == 'maximize' else grouped.mean().idxmax(),
        }

        return stats

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Compute correlation matrix between parameters and metric.

        Returns:
            Correlation matrix DataFrame
        """
        # Convert categorical to numeric if needed
        df_numeric = self.df.copy()
        for col in df_numeric.columns:
            if df_numeric[col].dtype == 'object':
                df_numeric[col] = pd.Categorical(df_numeric[col]).codes

        return df_numeric.corr()

    def get_best_configs(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top N best configurations.

        Args:
            n: Number of top configurations to return

        Returns:
            List of parameter dictionaries
        """
        sorted_df = self.df.sort_values(
            self.metric,
            ascending=(self.direction == 'minimize')
        )

        configs = []
        for _, row in sorted_df.head(n).iterrows():
            config = {k: v for k, v in row.items() if k != self.metric}
            config['_metric_value'] = row[self.metric]
            configs.append(config)

        return configs

    def generate_report(self) -> str:
        """
        Generate a text report of the sensitivity analysis.

        Returns:
            Report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("HYPERPARAMETER SENSITIVITY ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append(f"\nMetric: {self.metric} (direction: {self.direction})")
        lines.append(f"Total trials analyzed: {len(self.results)}")

        # Parameter importance
        lines.append("\n" + "-" * 40)
        lines.append("PARAMETER IMPORTANCE")
        lines.append("-" * 40)
        importance = self.get_parameter_importance()
        for param, imp in importance.items():
            lines.append(f"  {param}: {imp:.4f} ({imp * 100:.1f}%)")

        # Best configurations
        lines.append("\n" + "-" * 40)
        lines.append("TOP 5 CONFIGURATIONS")
        lines.append("-" * 40)
        best_configs = self.get_best_configs(5)
        for i, config in enumerate(best_configs, 1):
            metric_val = config.pop('_metric_value')
            lines.append(f"\n  #{i} ({self.metric}={metric_val:.4f}):")
            for k, v in config.items():
                lines.append(f"    {k}: {v}")

        # Metric statistics
        lines.append("\n" + "-" * 40)
        lines.append("METRIC STATISTICS")
        lines.append("-" * 40)
        lines.append(f"  Mean: {self.df[self.metric].mean():.4f}")
        lines.append(f"  Std: {self.df[self.metric].std():.4f}")
        lines.append(f"  Min: {self.df[self.metric].min():.4f}")
        lines.append(f"  Max: {self.df[self.metric].max():.4f}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)


def load_search_space_from_yaml(yaml_path: str, model_name: Optional[str] = None) -> List[SearchSpace]:
    """
    Load search space definitions from YAML file.

    Args:
        yaml_path: Path to search_spaces.yaml
        model_name: Optional model name to load model-specific spaces

    Returns:
        List of SearchSpace objects
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    search_spaces = []

    # Helper to create SearchSpace from config
    def create_space(name: str, spec: Dict) -> SearchSpace:
        if 'values' in spec:
            return SearchSpace(name=name, distribution='values', values=spec['values'])
        elif 'distribution' in spec:
            return SearchSpace(
                name=name,
                distribution=spec['distribution'],
                low=spec.get('low'),
                high=spec.get('high')
            )
        else:
            raise ValueError(f"Invalid search space spec for {name}: {spec}")

    # Load common spaces
    if 'common' in config:
        for name, spec in config['common'].items():
            search_spaces.append(create_space(name, spec))

    # Load architecture spaces
    if 'architecture' in config:
        for name, spec in config['architecture'].items():
            search_spaces.append(create_space(name, spec))

    # Load model-specific spaces
    if model_name and 'model_specific' in config:
        if model_name in config['model_specific']:
            model_config = config['model_specific'][model_name]
            if 'include' in model_config:
                for category in model_config['include']:
                    if category in config:
                        for name, spec in config[category].items():
                            # Check if not already added
                            if not any(ss.name == name for ss in search_spaces):
                                search_spaces.append(create_space(name, spec))

    return search_spaces


def run_hyperparameter_tuning(
    args,
    train_fn: Callable,
    search_spaces_path: str = 'configs/search_spaces.yaml',
    tuning_mode: str = 'random',
    n_trials: int = 50,
    metric: str = 'val_accuracy',
    direction: str = 'maximize',
    output_dir: str = './tuning_results'
) -> Tuple[Dict[str, Any], List[TuningResult]]:
    """
    Main entry point for hyperparameter tuning.

    Args:
        args: Argument namespace with model configuration
        train_fn: Function that takes args and returns metrics dict
        search_spaces_path: Path to search spaces YAML
        tuning_mode: 'grid' or 'random'
        n_trials: Number of trials for random search
        metric: Metric to optimize
        direction: 'maximize' or 'minimize'
        output_dir: Directory to save results

    Returns:
        Tuple of (best_params, all_results)
    """
    # Load search spaces
    search_spaces = load_search_space_from_yaml(
        search_spaces_path,
        model_name=args.model
    )

    # Create wrapper function that updates args
    def wrapped_train_fn(params: Dict[str, Any]) -> Dict[str, float]:
        # Update args with params
        updated_args = deepcopy(args)
        for key, value in params.items():
            setattr(updated_args, key, value)
        return train_fn(updated_args)

    # Run tuning
    if tuning_mode == 'grid':
        tuner = GridSearchTuner(
            search_spaces=search_spaces,
            train_fn=wrapped_train_fn,
            metric=metric,
            direction=direction,
            output_dir=output_dir
        )
    elif tuning_mode == 'random':
        tuner = RandomSearchTuner(
            search_spaces=search_spaces,
            train_fn=wrapped_train_fn,
            metric=metric,
            direction=direction,
            n_trials=n_trials,
            output_dir=output_dir
        )
    else:
        raise ValueError(f"Unknown tuning_mode: {tuning_mode}")

    return tuner.search()
