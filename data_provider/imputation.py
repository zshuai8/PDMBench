"""
Imputation methods for handling missing data in time series.

Supports: linear, spline, knn, mice, forward-fill methods.
"""

import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.impute import KNNImputer
from typing import Optional, Union, Literal
import warnings

warnings.filterwarnings('ignore')


class Imputer:
    """
    Imputation handler for time series data with missing values.

    Args:
        method: Imputation method to use
            - 'linear': Linear interpolation (default)
            - 'spline': Cubic spline interpolation
            - 'knn': K-nearest neighbors imputation
            - 'mice': Multiple Imputation by Chained Equations
            - 'ffill': Forward fill (last observation carried forward)
            - 'bfill': Backward fill
            - 'mean': Fill with column mean
            - 'median': Fill with column median
            - 'zero': Fill with zeros
        n_neighbors: Number of neighbors for KNN imputation (default: 5)
        max_iter: Maximum iterations for MICE imputation (default: 10)
        spline_order: Order of spline for spline interpolation (default: 3)
    """

    SUPPORTED_METHODS = ['linear', 'spline', 'knn', 'mice', 'ffill', 'bfill', 'mean', 'median', 'zero']

    def __init__(
        self,
        method: str = 'linear',
        n_neighbors: int = 5,
        max_iter: int = 10,
        spline_order: int = 3
    ):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unknown imputation method: {method}. "
                           f"Supported methods: {self.SUPPORTED_METHODS}")

        self.method = method
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.spline_order = spline_order

        # Store fitted parameters for transform
        self._fitted_means = None
        self._fitted_medians = None
        self._knn_imputer = None

    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> 'Imputer':
        """
        Fit the imputer on training data.

        Args:
            data: Input data (samples, time_steps, features) or DataFrame

        Returns:
            self
        """
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data

        # Compute statistics ignoring NaN
        if data_array.ndim == 3:
            # Flatten to (samples * time_steps, features) for statistics
            flat_data = data_array.reshape(-1, data_array.shape[-1])
        else:
            flat_data = data_array

        self._fitted_means = np.nanmean(flat_data, axis=0)
        self._fitted_medians = np.nanmedian(flat_data, axis=0)

        # Fit KNN imputer if needed
        if self.method == 'knn':
            self._knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
            # Fit on non-NaN samples if possible
            valid_mask = ~np.any(np.isnan(flat_data), axis=1)
            if np.sum(valid_mask) > self.n_neighbors:
                self._knn_imputer.fit(flat_data[valid_mask])
            else:
                self._knn_imputer.fit(flat_data)

        return self

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Apply imputation to data.

        Args:
            data: Input data with missing values

        Returns:
            Imputed data in the same format as input
        """
        return_df = isinstance(data, pd.DataFrame)
        if return_df:
            columns = data.columns
            index = data.index
            data_array = data.values.copy()
        else:
            data_array = data.copy()

        # Handle different array dimensions
        original_shape = data_array.shape
        is_3d = data_array.ndim == 3

        if is_3d:
            # Process each sample independently for interpolation methods
            imputed = self._impute_3d(data_array)
        else:
            imputed = self._impute_2d(data_array)

        if return_df:
            return pd.DataFrame(imputed, columns=columns, index=index)
        return imputed

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)

    def _impute_2d(self, data: np.ndarray) -> np.ndarray:
        """Impute 2D array (time_steps, features) or (samples, features)."""
        if not np.any(np.isnan(data)):
            return data

        if self.method == 'linear':
            return self._interpolate_linear(data)
        elif self.method == 'spline':
            return self._interpolate_spline(data)
        elif self.method == 'knn':
            return self._impute_knn(data)
        elif self.method == 'mice':
            return self._impute_mice(data)
        elif self.method == 'ffill':
            return self._fill_forward(data)
        elif self.method == 'bfill':
            return self._fill_backward(data)
        elif self.method == 'mean':
            return self._fill_mean(data)
        elif self.method == 'median':
            return self._fill_median(data)
        elif self.method == 'zero':
            return np.nan_to_num(data, nan=0.0)
        else:
            return data

    def _impute_3d(self, data: np.ndarray) -> np.ndarray:
        """Impute 3D array (samples, time_steps, features)."""
        imputed = np.zeros_like(data)
        for i in range(data.shape[0]):
            imputed[i] = self._impute_2d(data[i])
        return imputed

    def _interpolate_linear(self, data: np.ndarray) -> np.ndarray:
        """Linear interpolation along time axis."""
        result = data.copy()
        n_features = data.shape[1] if data.ndim > 1 else 1

        if data.ndim == 1:
            data = data.reshape(-1, 1)
            result = result.reshape(-1, 1)

        for j in range(n_features):
            col = result[:, j]
            nan_mask = np.isnan(col)
            if np.any(nan_mask) and not np.all(nan_mask):
                valid_idx = np.where(~nan_mask)[0]
                invalid_idx = np.where(nan_mask)[0]

                # Interpolate
                interp_func = interpolate.interp1d(
                    valid_idx, col[valid_idx],
                    kind='linear',
                    bounds_error=False,
                    fill_value=(col[valid_idx[0]], col[valid_idx[-1]])
                )
                result[invalid_idx, j] = interp_func(invalid_idx)

        return result.squeeze() if n_features == 1 and data.ndim == 1 else result

    def _interpolate_spline(self, data: np.ndarray) -> np.ndarray:
        """Cubic spline interpolation along time axis."""
        result = data.copy()
        n_features = data.shape[1] if data.ndim > 1 else 1

        if data.ndim == 1:
            data = data.reshape(-1, 1)
            result = result.reshape(-1, 1)

        for j in range(n_features):
            col = result[:, j]
            nan_mask = np.isnan(col)
            if np.any(nan_mask) and not np.all(nan_mask):
                valid_idx = np.where(~nan_mask)[0]
                invalid_idx = np.where(nan_mask)[0]

                # Need at least k+1 points for spline of order k
                if len(valid_idx) > self.spline_order:
                    try:
                        spline = interpolate.UnivariateSpline(
                            valid_idx, col[valid_idx],
                            k=min(self.spline_order, len(valid_idx) - 1),
                            s=0
                        )
                        result[invalid_idx, j] = spline(invalid_idx)
                    except Exception:
                        # Fall back to linear if spline fails
                        result[:, j] = self._interpolate_linear(col.reshape(-1, 1)).flatten()
                else:
                    # Not enough points, use linear
                    result[:, j] = self._interpolate_linear(col.reshape(-1, 1)).flatten()

        return result.squeeze() if n_features == 1 and data.ndim == 1 else result

    def _impute_knn(self, data: np.ndarray) -> np.ndarray:
        """K-nearest neighbors imputation."""
        if self._knn_imputer is None:
            self._knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
            self._knn_imputer.fit(data)
        return self._knn_imputer.transform(data)

    def _impute_mice(self, data: np.ndarray) -> np.ndarray:
        """Multiple Imputation by Chained Equations (iterative imputation)."""
        try:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer

            imputer = IterativeImputer(max_iter=self.max_iter, random_state=42)
            return imputer.fit_transform(data)
        except ImportError:
            warnings.warn("IterativeImputer not available, falling back to KNN")
            return self._impute_knn(data)

    def _fill_forward(self, data: np.ndarray) -> np.ndarray:
        """Forward fill (last observation carried forward)."""
        df = pd.DataFrame(data)
        return df.ffill().bfill().values  # bfill for leading NaNs

    def _fill_backward(self, data: np.ndarray) -> np.ndarray:
        """Backward fill."""
        df = pd.DataFrame(data)
        return df.bfill().ffill().values  # ffill for trailing NaNs

    def _fill_mean(self, data: np.ndarray) -> np.ndarray:
        """Fill with column mean."""
        result = data.copy()
        if self._fitted_means is None:
            means = np.nanmean(data, axis=0)
        else:
            means = self._fitted_means

        for j in range(data.shape[1] if data.ndim > 1 else 1):
            if data.ndim > 1:
                nan_mask = np.isnan(result[:, j])
                result[nan_mask, j] = means[j]
            else:
                nan_mask = np.isnan(result)
                result[nan_mask] = means[0] if hasattr(means, '__len__') else means
        return result

    def _fill_median(self, data: np.ndarray) -> np.ndarray:
        """Fill with column median."""
        result = data.copy()
        if self._fitted_medians is None:
            medians = np.nanmedian(data, axis=0)
        else:
            medians = self._fitted_medians

        for j in range(data.shape[1] if data.ndim > 1 else 1):
            if data.ndim > 1:
                nan_mask = np.isnan(result[:, j])
                result[nan_mask, j] = medians[j]
            else:
                nan_mask = np.isnan(result)
                result[nan_mask] = medians[0] if hasattr(medians, '__len__') else medians
        return result


def impute_missing(
    data: Union[np.ndarray, pd.DataFrame, pd.Series],
    method: str = 'linear',
    **kwargs
) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
    """
    Convenience function for one-shot imputation.

    Args:
        data: Input data with missing values
        method: Imputation method ('linear', 'spline', 'knn', 'mice', 'ffill', 'bfill', 'mean', 'median', 'zero')
        **kwargs: Additional arguments passed to Imputer

    Returns:
        Imputed data in the same format as input
    """
    is_series = isinstance(data, pd.Series)
    if is_series:
        data = data.to_frame()

    imputer = Imputer(method=method, **kwargs)
    result = imputer.fit_transform(data)

    if is_series:
        return result.iloc[:, 0]
    return result


def detect_missing_patterns(data: np.ndarray) -> dict:
    """
    Analyze missing data patterns in a dataset.

    Args:
        data: Input array (samples, time_steps, features) or (time_steps, features)

    Returns:
        Dictionary with missing data statistics
    """
    nan_mask = np.isnan(data)

    stats = {
        'total_missing': int(np.sum(nan_mask)),
        'total_values': int(data.size),
        'missing_ratio': float(np.sum(nan_mask) / data.size),
        'samples_with_missing': 0,
        'features_with_missing': [],
        'max_consecutive_missing': 0
    }

    if data.ndim == 3:
        # Per-sample statistics
        sample_missing = np.any(nan_mask, axis=(1, 2))
        stats['samples_with_missing'] = int(np.sum(sample_missing))

        # Per-feature statistics
        feature_missing = np.any(nan_mask, axis=(0, 1))
        stats['features_with_missing'] = np.where(feature_missing)[0].tolist()

    elif data.ndim == 2:
        # Per-feature statistics
        feature_missing = np.any(nan_mask, axis=0)
        stats['features_with_missing'] = np.where(feature_missing)[0].tolist()

    # Find maximum consecutive missing values
    for idx in np.ndindex(data.shape[:-1] if data.ndim > 1 else (1,)):
        if data.ndim > 1:
            col = nan_mask[idx]
        else:
            col = nan_mask
        max_consec = _max_consecutive_true(col if data.ndim == 1 else col.flatten())
        stats['max_consecutive_missing'] = max(stats['max_consecutive_missing'], max_consec)

    return stats


def _max_consecutive_true(arr: np.ndarray) -> int:
    """Find maximum consecutive True values in boolean array."""
    if not np.any(arr):
        return 0

    max_count = 0
    current_count = 0

    for val in arr:
        if val:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0

    return max_count
