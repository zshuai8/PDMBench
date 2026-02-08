"""
Normalization methods for time series data.

Extends the base Normalizer with robust, winsorized, and per-channel normalization options.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, Literal
import warnings

warnings.filterwarnings('ignore')


class ExtendedNormalizer:
    """
    Extended normalizer supporting multiple normalization strategies.

    Args:
        norm_type: Normalization method
            - 'standardization': Z-score normalization (default)
            - 'minmax': Min-max normalization to [0, 1]
            - 'robust': Robust normalization using median and IQR
            - 'winsorized': Winsorized normalization with outlier clipping
            - 'per_channel': Per-channel/feature standardization
            - 'per_sample_std': Per-sample standardization
            - 'per_sample_minmax': Per-sample min-max normalization
            - 'log': Log transformation (for positive values)
            - 'none': No normalization
        clip_range: Tuple of (min, max) for clipping after normalization
        winsorize_limits: Tuple of (lower, upper) percentiles for winsorization (default: (5, 95))
    """

    SUPPORTED_METHODS = [
        'standardization', 'minmax', 'robust', 'winsorized',
        'per_channel', 'per_sample_std', 'per_sample_minmax',
        'log', 'none'
    ]

    def __init__(
        self,
        norm_type: str = 'standardization',
        clip_range: Optional[Tuple[float, float]] = None,
        winsorize_limits: Tuple[float, float] = (5, 95)
    ):
        if norm_type not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unknown normalization method: {norm_type}. "
                           f"Supported methods: {self.SUPPORTED_METHODS}")

        self.norm_type = norm_type
        self.clip_range = clip_range
        self.winsorize_limits = winsorize_limits

        # Store fitted parameters
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None
        self.median_ = None
        self.iqr_ = None
        self.q1_ = None
        self.q3_ = None
        self.winsorize_low_ = None
        self.winsorize_high_ = None
        self._fitted = False

    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> 'ExtendedNormalizer':
        """
        Fit normalizer parameters on training data.

        Args:
            data: Training data (samples * time_steps, features) for DataFrame
                  or (samples, time_steps, features) for array

        Returns:
            self
        """
        if isinstance(data, pd.DataFrame):
            values = data.values
        else:
            values = data

        # Flatten to 2D if 3D
        if values.ndim == 3:
            values = values.reshape(-1, values.shape[-1])

        # Compute statistics
        self.mean_ = np.nanmean(values, axis=0)
        self.std_ = np.nanstd(values, axis=0)
        self.min_ = np.nanmin(values, axis=0)
        self.max_ = np.nanmax(values, axis=0)
        self.median_ = np.nanmedian(values, axis=0)

        # IQR for robust normalization
        self.q1_ = np.nanpercentile(values, 25, axis=0)
        self.q3_ = np.nanpercentile(values, 75, axis=0)
        self.iqr_ = self.q3_ - self.q1_

        # Winsorization bounds
        self.winsorize_low_ = np.nanpercentile(values, self.winsorize_limits[0], axis=0)
        self.winsorize_high_ = np.nanpercentile(values, self.winsorize_limits[1], axis=0)

        self._fitted = True
        return self

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Apply normalization to data.

        Args:
            data: Data to normalize

        Returns:
            Normalized data in same format as input
        """
        return_df = isinstance(data, pd.DataFrame)
        if return_df:
            columns = data.columns
            index = data.index
            values = data.values.copy()
        else:
            values = data.copy()

        original_shape = values.shape
        is_3d = values.ndim == 3

        if is_3d:
            n_samples, n_steps, n_features = values.shape
            values = values.reshape(-1, n_features)

        # Apply normalization
        if self.norm_type == 'standardization':
            values = self._standardize(values)
        elif self.norm_type == 'minmax':
            values = self._minmax(values)
        elif self.norm_type == 'robust':
            values = self._robust(values)
        elif self.norm_type == 'winsorized':
            values = self._winsorized(values)
        elif self.norm_type == 'per_channel':
            values = self._per_channel(values)
        elif self.norm_type == 'per_sample_std':
            if is_3d:
                values = values.reshape(original_shape)
                values = self._per_sample_std_3d(values)
            else:
                values = self._per_sample_std(values, index if return_df else None)
        elif self.norm_type == 'per_sample_minmax':
            if is_3d:
                values = values.reshape(original_shape)
                values = self._per_sample_minmax_3d(values)
            else:
                values = self._per_sample_minmax(values, index if return_df else None)
        elif self.norm_type == 'log':
            values = self._log_transform(values)
        elif self.norm_type == 'none':
            pass

        # Reshape back if was 3D
        if is_3d and self.norm_type not in ['per_sample_std', 'per_sample_minmax']:
            values = values.reshape(original_shape)

        # Apply clipping if specified
        if self.clip_range is not None:
            values = np.clip(values, self.clip_range[0], self.clip_range[1])

        if return_df:
            return pd.DataFrame(values.reshape(original_shape[0] if not is_3d else -1, original_shape[-1]),
                              columns=columns, index=index)
        return values

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)

    def inverse_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Reverse normalization.

        Args:
            data: Normalized data

        Returns:
            Original-scale data
        """
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        return_df = isinstance(data, pd.DataFrame)
        if return_df:
            columns = data.columns
            index = data.index
            values = data.values.copy()
        else:
            values = data.copy()

        original_shape = values.shape
        is_3d = values.ndim == 3

        if is_3d:
            values = values.reshape(-1, values.shape[-1])

        # Reverse normalization
        if self.norm_type == 'standardization':
            values = values * (self.std_ + np.finfo(float).eps) + self.mean_
        elif self.norm_type == 'minmax':
            values = values * (self.max_ - self.min_ + np.finfo(float).eps) + self.min_
        elif self.norm_type == 'robust':
            values = values * (self.iqr_ + np.finfo(float).eps) + self.median_
        elif self.norm_type == 'per_channel':
            values = values * (self.std_ + np.finfo(float).eps) + self.mean_
        elif self.norm_type == 'log':
            values = np.exp(values) - 1
        # Note: per_sample and winsorized cannot be exactly reversed

        if is_3d:
            values = values.reshape(original_shape)

        if return_df:
            return pd.DataFrame(values.reshape(original_shape[0] if not is_3d else -1, original_shape[-1]),
                              columns=columns, index=index)
        return values

    def _standardize(self, values: np.ndarray) -> np.ndarray:
        """Z-score normalization."""
        if self.mean_ is None:
            mean = np.nanmean(values, axis=0)
            std = np.nanstd(values, axis=0)
        else:
            mean = self.mean_
            std = self.std_
        return (values - mean) / (std + np.finfo(float).eps)

    def _minmax(self, values: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]."""
        if self.min_ is None:
            min_val = np.nanmin(values, axis=0)
            max_val = np.nanmax(values, axis=0)
        else:
            min_val = self.min_
            max_val = self.max_
        return (values - min_val) / (max_val - min_val + np.finfo(float).eps)

    def _robust(self, values: np.ndarray) -> np.ndarray:
        """Robust normalization using median and IQR."""
        if self.median_ is None:
            median = np.nanmedian(values, axis=0)
            q1 = np.nanpercentile(values, 25, axis=0)
            q3 = np.nanpercentile(values, 75, axis=0)
            iqr = q3 - q1
        else:
            median = self.median_
            iqr = self.iqr_
        return (values - median) / (iqr + np.finfo(float).eps)

    def _winsorized(self, values: np.ndarray) -> np.ndarray:
        """Winsorized normalization: clip outliers then standardize."""
        if self.winsorize_low_ is None:
            low = np.nanpercentile(values, self.winsorize_limits[0], axis=0)
            high = np.nanpercentile(values, self.winsorize_limits[1], axis=0)
        else:
            low = self.winsorize_low_
            high = self.winsorize_high_

        # Clip values
        clipped = np.clip(values, low, high)

        # Standardize clipped values
        mean = np.nanmean(clipped, axis=0)
        std = np.nanstd(clipped, axis=0)
        return (clipped - mean) / (std + np.finfo(float).eps)

    def _per_channel(self, values: np.ndarray) -> np.ndarray:
        """Per-channel/feature standardization (same as standardization but clearer naming)."""
        return self._standardize(values)

    def _per_sample_std(self, values: np.ndarray, index: Optional[pd.Index]) -> np.ndarray:
        """Per-sample standardization for 2D data with index."""
        if index is not None:
            df = pd.DataFrame(values, index=index)
            grouped = df.groupby(level=0)
            result = (df - grouped.transform('mean')) / (grouped.transform('std') + np.finfo(float).eps)
            return result.values
        else:
            # Treat entire array as one sample
            mean = np.nanmean(values)
            std = np.nanstd(values)
            return (values - mean) / (std + np.finfo(float).eps)

    def _per_sample_std_3d(self, values: np.ndarray) -> np.ndarray:
        """Per-sample standardization for 3D data."""
        result = np.zeros_like(values)
        for i in range(values.shape[0]):
            sample = values[i]
            mean = np.nanmean(sample)
            std = np.nanstd(sample)
            result[i] = (sample - mean) / (std + np.finfo(float).eps)
        return result

    def _per_sample_minmax(self, values: np.ndarray, index: Optional[pd.Index]) -> np.ndarray:
        """Per-sample min-max normalization for 2D data."""
        if index is not None:
            df = pd.DataFrame(values, index=index)
            grouped = df.groupby(level=0)
            min_vals = grouped.transform('min')
            max_vals = grouped.transform('max')
            return ((df - min_vals) / (max_vals - min_vals + np.finfo(float).eps)).values
        else:
            min_val = np.nanmin(values)
            max_val = np.nanmax(values)
            return (values - min_val) / (max_val - min_val + np.finfo(float).eps)

    def _per_sample_minmax_3d(self, values: np.ndarray) -> np.ndarray:
        """Per-sample min-max normalization for 3D data."""
        result = np.zeros_like(values)
        for i in range(values.shape[0]):
            sample = values[i]
            min_val = np.nanmin(sample)
            max_val = np.nanmax(sample)
            result[i] = (sample - min_val) / (max_val - min_val + np.finfo(float).eps)
        return result

    def _log_transform(self, values: np.ndarray) -> np.ndarray:
        """Log transformation (handles negative/zero values)."""
        # Shift to positive if needed
        min_val = np.nanmin(values)
        if min_val <= 0:
            values = values - min_val + 1
        return np.log1p(values)


def normalize(
    data: Union[np.ndarray, pd.DataFrame],
    method: str = 'standardization',
    **kwargs
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Convenience function for one-shot normalization.

    Args:
        data: Input data
        method: Normalization method
        **kwargs: Additional arguments passed to ExtendedNormalizer

    Returns:
        Normalized data
    """
    normalizer = ExtendedNormalizer(norm_type=method, **kwargs)
    return normalizer.fit_transform(data)


class ChannelWiseNormalizer:
    """
    Applies different normalization methods to different channels/features.

    Args:
        channel_methods: Dictionary mapping channel indices to normalization methods
                        e.g., {0: 'standardization', 1: 'robust', 2: 'minmax'}
        default_method: Default method for channels not specified
    """

    def __init__(
        self,
        channel_methods: dict,
        default_method: str = 'standardization'
    ):
        self.channel_methods = channel_methods
        self.default_method = default_method
        self.normalizers_ = {}
        self._fitted = False

    def fit(self, data: np.ndarray) -> 'ChannelWiseNormalizer':
        """Fit normalizer for each channel."""
        if data.ndim == 3:
            n_channels = data.shape[-1]
            flat_data = data.reshape(-1, n_channels)
        else:
            n_channels = data.shape[-1]
            flat_data = data

        for i in range(n_channels):
            method = self.channel_methods.get(i, self.default_method)
            normalizer = ExtendedNormalizer(norm_type=method)
            normalizer.fit(flat_data[:, i:i+1])
            self.normalizers_[i] = normalizer

        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform each channel with its normalizer."""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        original_shape = data.shape
        is_3d = data.ndim == 3

        if is_3d:
            n_channels = data.shape[-1]
            flat_data = data.reshape(-1, n_channels)
        else:
            n_channels = data.shape[-1]
            flat_data = data.copy()

        result = np.zeros_like(flat_data)
        for i in range(n_channels):
            result[:, i] = self.normalizers_[i].transform(flat_data[:, i:i+1]).flatten()

        if is_3d:
            return result.reshape(original_shape)
        return result

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)


class AdaptiveNormalizer:
    """
    Automatically selects normalization method based on data characteristics.

    Analyzes data distribution and selects appropriate normalization:
    - High skewness -> log or robust
    - Many outliers -> winsorized or robust
    - Normal-like -> standardization
    - Bounded data -> minmax
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.selected_method_ = None
        self.normalizer_ = None

    def fit(self, data: np.ndarray) -> 'AdaptiveNormalizer':
        """Analyze data and select normalization method."""
        flat_data = data.reshape(-1, data.shape[-1]) if data.ndim == 3 else data

        # Compute statistics
        skewness = self._compute_skewness(flat_data)
        outlier_ratio = self._compute_outlier_ratio(flat_data)
        is_bounded = self._check_bounded(flat_data)

        # Select method based on characteristics
        if is_bounded:
            self.selected_method_ = 'minmax'
        elif outlier_ratio > 0.1:
            self.selected_method_ = 'winsorized'
        elif abs(skewness) > 1.0:
            self.selected_method_ = 'robust'
        else:
            self.selected_method_ = 'standardization'

        if self.verbose:
            print(f"Selected normalization method: {self.selected_method_}")
            print(f"  Skewness: {skewness:.2f}, Outlier ratio: {outlier_ratio:.2%}")

        self.normalizer_ = ExtendedNormalizer(norm_type=self.selected_method_)
        self.normalizer_.fit(data)

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply selected normalization."""
        if self.normalizer_ is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return self.normalizer_.transform(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)

    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute average skewness across features."""
        from scipy import stats
        skewnesses = []
        for j in range(data.shape[1]):
            col = data[:, j]
            col = col[~np.isnan(col)]
            if len(col) > 2:
                skewnesses.append(stats.skew(col))
        return np.mean(skewnesses) if skewnesses else 0.0

    def _compute_outlier_ratio(self, data: np.ndarray) -> float:
        """Compute ratio of outliers using IQR method."""
        outlier_counts = 0
        total_counts = 0

        for j in range(data.shape[1]):
            col = data[:, j]
            col = col[~np.isnan(col)]
            if len(col) > 0:
                q1, q3 = np.percentile(col, [25, 75])
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_counts += np.sum((col < lower) | (col > upper))
                total_counts += len(col)

        return outlier_counts / total_counts if total_counts > 0 else 0.0

    def _check_bounded(self, data: np.ndarray) -> bool:
        """Check if data appears to be bounded (e.g., percentages, ratios)."""
        min_vals = np.nanmin(data, axis=0)
        max_vals = np.nanmax(data, axis=0)

        # Check if all features are in [0, 1] or [0, 100]
        in_unit = np.all((min_vals >= 0) & (max_vals <= 1))
        in_percent = np.all((min_vals >= 0) & (max_vals <= 100))

        return in_unit or (in_percent and np.all(max_vals > 1))
