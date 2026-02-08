"""
Resampling methods for handling irregular sampling in time series.

Supports: timestamp-aware resampling, anti-aliasing, and various interpolation methods.
"""

import numpy as np
import pandas as pd
from scipy import signal, interpolate
from typing import Optional, Union, Tuple, List, Literal
import warnings

warnings.filterwarnings('ignore')


class Resampler:
    """
    Resampler for time series data with irregular sampling.

    Args:
        target_length: Target sequence length after resampling
        method: Resampling method to use
            - 'linear': Linear interpolation (default)
            - 'cubic': Cubic spline interpolation
            - 'nearest': Nearest neighbor interpolation
            - 'zero': Zero-order hold (step function)
            - 'subsample': Simple subsampling with optional anti-aliasing
            - 'average': Averaging-based downsampling
        anti_alias: Whether to apply anti-aliasing filter before downsampling
        anti_alias_order: Order of the anti-aliasing filter (default: 8)
        preserve_endpoints: Whether to preserve first and last values exactly
    """

    SUPPORTED_METHODS = ['linear', 'cubic', 'nearest', 'zero', 'subsample', 'average']

    def __init__(
        self,
        target_length: Optional[int] = None,
        method: str = 'linear',
        anti_alias: bool = True,
        anti_alias_order: int = 8,
        preserve_endpoints: bool = True
    ):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unknown resampling method: {method}. "
                           f"Supported methods: {self.SUPPORTED_METHODS}")

        self.target_length = target_length
        self.method = method
        self.anti_alias = anti_alias
        self.anti_alias_order = anti_alias_order
        self.preserve_endpoints = preserve_endpoints

    def resample(
        self,
        data: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        target_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample time series to target length.

        Args:
            data: Input data (time_steps, features) or (samples, time_steps, features)
            timestamps: Optional timestamps for each time step
            target_length: Target length (overrides constructor value)

        Returns:
            Tuple of (resampled_data, new_timestamps)
        """
        target_len = target_length or self.target_length
        if target_len is None:
            raise ValueError("target_length must be specified")

        is_3d = data.ndim == 3
        if is_3d:
            return self._resample_3d(data, timestamps, target_len)
        else:
            return self._resample_2d(data, timestamps, target_len)

    def _resample_2d(
        self,
        data: np.ndarray,
        timestamps: Optional[np.ndarray],
        target_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample 2D array (time_steps, features)."""
        current_length = data.shape[0]
        n_features = data.shape[1] if data.ndim > 1 else 1

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Create timestamps if not provided
        if timestamps is None:
            timestamps = np.linspace(0, 1, current_length)

        # Create target timestamps
        new_timestamps = np.linspace(timestamps[0], timestamps[-1], target_length)

        # Determine if upsampling or downsampling
        is_downsampling = target_length < current_length

        # Apply anti-aliasing for downsampling
        if is_downsampling and self.anti_alias and self.method != 'average':
            data = self._apply_anti_alias(data, current_length / target_length)

        # Resample each feature
        resampled = np.zeros((target_length, n_features))

        for j in range(n_features):
            resampled[:, j] = self._resample_1d(
                data[:, j], timestamps, new_timestamps
            )

        # Preserve endpoints if requested
        if self.preserve_endpoints:
            resampled[0, :] = data[0, :]
            resampled[-1, :] = data[-1, :]

        return resampled.squeeze() if n_features == 1 else resampled, new_timestamps

    def _resample_3d(
        self,
        data: np.ndarray,
        timestamps: Optional[np.ndarray],
        target_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample 3D array (samples, time_steps, features)."""
        n_samples = data.shape[0]
        n_features = data.shape[2]

        resampled = np.zeros((n_samples, target_length, n_features))
        new_timestamps = None

        for i in range(n_samples):
            sample_timestamps = timestamps[i] if timestamps is not None and timestamps.ndim > 1 else timestamps
            resampled[i], new_timestamps = self._resample_2d(
                data[i], sample_timestamps, target_length
            )

        return resampled, new_timestamps

    def _resample_1d(
        self,
        data: np.ndarray,
        old_timestamps: np.ndarray,
        new_timestamps: np.ndarray
    ) -> np.ndarray:
        """Resample a single 1D time series."""
        if self.method == 'linear':
            interp_func = interpolate.interp1d(
                old_timestamps, data, kind='linear',
                bounds_error=False, fill_value='extrapolate'
            )
            return interp_func(new_timestamps)

        elif self.method == 'cubic':
            # Ensure enough points for cubic interpolation
            if len(data) >= 4:
                try:
                    interp_func = interpolate.interp1d(
                        old_timestamps, data, kind='cubic',
                        bounds_error=False, fill_value='extrapolate'
                    )
                    return interp_func(new_timestamps)
                except Exception:
                    pass
            # Fall back to linear
            interp_func = interpolate.interp1d(
                old_timestamps, data, kind='linear',
                bounds_error=False, fill_value='extrapolate'
            )
            return interp_func(new_timestamps)

        elif self.method == 'nearest':
            interp_func = interpolate.interp1d(
                old_timestamps, data, kind='nearest',
                bounds_error=False, fill_value='extrapolate'
            )
            return interp_func(new_timestamps)

        elif self.method == 'zero':
            interp_func = interpolate.interp1d(
                old_timestamps, data, kind='zero',
                bounds_error=False, fill_value='extrapolate'
            )
            return interp_func(new_timestamps)

        elif self.method == 'subsample':
            return self._subsample(data, len(new_timestamps))

        elif self.method == 'average':
            return self._resample_average(data, old_timestamps, new_timestamps)

        else:
            return data

    def _apply_anti_alias(self, data: np.ndarray, downsample_factor: float) -> np.ndarray:
        """Apply low-pass anti-aliasing filter before downsampling."""
        # Butterworth low-pass filter
        nyquist = 0.5
        cutoff = nyquist / downsample_factor * 0.8  # 80% of Nyquist for safety

        # Clamp cutoff to valid range
        cutoff = max(0.01, min(cutoff, 0.99))

        try:
            b, a = signal.butter(self.anti_alias_order, cutoff, btype='low')
            # Apply filter forward and backward to avoid phase shift
            return signal.filtfilt(b, a, data, axis=0)
        except Exception:
            # Return original data if filtering fails
            return data

    def _subsample(self, data: np.ndarray, target_length: int) -> np.ndarray:
        """Simple subsampling with uniform spacing."""
        indices = np.linspace(0, len(data) - 1, target_length, dtype=int)
        return data[indices]

    def _resample_average(
        self,
        data: np.ndarray,
        old_timestamps: np.ndarray,
        new_timestamps: np.ndarray
    ) -> np.ndarray:
        """Resample by averaging values in each new bin."""
        result = np.zeros(len(new_timestamps))

        # Create bins around new timestamps
        bin_edges = np.zeros(len(new_timestamps) + 1)
        bin_edges[0] = new_timestamps[0] - (new_timestamps[1] - new_timestamps[0]) / 2
        bin_edges[-1] = new_timestamps[-1] + (new_timestamps[-1] - new_timestamps[-2]) / 2
        bin_edges[1:-1] = (new_timestamps[:-1] + new_timestamps[1:]) / 2

        for i in range(len(new_timestamps)):
            mask = (old_timestamps >= bin_edges[i]) & (old_timestamps < bin_edges[i + 1])
            if np.any(mask):
                result[i] = np.mean(data[mask])
            else:
                # No data in this bin, interpolate
                result[i] = np.interp(new_timestamps[i], old_timestamps, data)

        return result


class TimestampAwareResampler:
    """
    Resampler that handles irregular timestamps intelligently.

    Args:
        target_frequency: Target frequency as string ('1s', '1ms', '10ms', '1min', etc.)
                         or as float (samples per second)
        method: Interpolation method ('linear', 'cubic', 'nearest')
        handle_gaps: How to handle large gaps in timestamps
            - 'interpolate': Interpolate across gaps
            - 'nan': Fill gaps with NaN
            - 'hold': Hold last value across gaps
        max_gap_ratio: Maximum gap ratio before treating as a gap (default: 5.0)
    """

    def __init__(
        self,
        target_frequency: Union[str, float] = '1s',
        method: str = 'linear',
        handle_gaps: str = 'interpolate',
        max_gap_ratio: float = 5.0
    ):
        self.target_frequency = target_frequency
        self.method = method
        self.handle_gaps = handle_gaps
        self.max_gap_ratio = max_gap_ratio

    def resample(
        self,
        data: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample irregular time series to regular timestamps.

        Args:
            data: Input data (time_steps, features) or (time_steps,)
            timestamps: Timestamps for each time step (Unix timestamps or datetime)

        Returns:
            Tuple of (resampled_data, new_timestamps)
        """
        # Convert timestamps to numeric if needed
        if isinstance(timestamps[0], (pd.Timestamp, np.datetime64)):
            timestamps = pd.DatetimeIndex(timestamps).astype(np.int64) / 1e9

        timestamps = np.array(timestamps, dtype=float)

        # Calculate target interval
        target_interval = self._parse_frequency(self.target_frequency, timestamps)

        # Detect gaps
        intervals = np.diff(timestamps)
        median_interval = np.median(intervals)
        gap_mask = intervals > median_interval * self.max_gap_ratio

        # Create new regular timestamps
        new_timestamps = np.arange(
            timestamps[0],
            timestamps[-1] + target_interval,
            target_interval
        )

        # Ensure original shape
        is_1d = data.ndim == 1
        if is_1d:
            data = data.reshape(-1, 1)

        n_features = data.shape[1]
        resampled = np.zeros((len(new_timestamps), n_features))

        # Resample each feature
        for j in range(n_features):
            resampled[:, j] = self._resample_with_gaps(
                data[:, j], timestamps, new_timestamps, gap_mask
            )

        return resampled.squeeze() if is_1d else resampled, new_timestamps

    def _parse_frequency(self, freq: Union[str, float], timestamps: np.ndarray) -> float:
        """Parse frequency specification to interval in seconds."""
        if isinstance(freq, (int, float)):
            return 1.0 / freq

        freq = freq.lower().strip()

        # Parse common frequency strings
        multipliers = {
            'ns': 1e-9, 'us': 1e-6, 'ms': 1e-3, 's': 1.0,
            'min': 60.0, 'h': 3600.0, 'd': 86400.0
        }

        for unit, mult in multipliers.items():
            if freq.endswith(unit):
                try:
                    value = float(freq[:-len(unit)]) if freq[:-len(unit)] else 1.0
                    return value * mult
                except ValueError:
                    pass

        # Default to median interval
        return np.median(np.diff(timestamps))

    def _resample_with_gaps(
        self,
        data: np.ndarray,
        old_timestamps: np.ndarray,
        new_timestamps: np.ndarray,
        gap_mask: np.ndarray
    ) -> np.ndarray:
        """Resample handling gaps according to configuration."""
        result = np.zeros(len(new_timestamps))

        # Basic interpolation
        interp_func = interpolate.interp1d(
            old_timestamps, data, kind=self.method,
            bounds_error=False, fill_value='extrapolate'
        )
        result = interp_func(new_timestamps)

        # Handle gaps
        if self.handle_gaps != 'interpolate' and np.any(gap_mask):
            gap_starts = old_timestamps[:-1][gap_mask]
            gap_ends = old_timestamps[1:][gap_mask]

            for gap_start, gap_end in zip(gap_starts, gap_ends):
                gap_new_mask = (new_timestamps > gap_start) & (new_timestamps < gap_end)

                if self.handle_gaps == 'nan':
                    result[gap_new_mask] = np.nan
                elif self.handle_gaps == 'hold':
                    # Hold last value before gap
                    last_value = data[old_timestamps <= gap_start][-1]
                    result[gap_new_mask] = last_value

        return result


def resample_to_length(
    data: Union[np.ndarray, pd.DataFrame],
    target_length: int,
    method: str = 'linear',
    anti_alias: bool = True
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Convenience function to resample time series to a target length.

    Args:
        data: Input time series
        target_length: Target sequence length
        method: Resampling method ('linear', 'cubic', 'nearest', 'subsample', 'average')
        anti_alias: Whether to apply anti-aliasing for downsampling

    Returns:
        Resampled data
    """
    return_df = isinstance(data, pd.DataFrame)
    if return_df:
        columns = data.columns
        data_array = data.values
    else:
        data_array = data

    resampler = Resampler(
        target_length=target_length,
        method=method,
        anti_alias=anti_alias
    )

    resampled, _ = resampler.resample(data_array)

    if return_df:
        return pd.DataFrame(resampled, columns=columns)
    return resampled


def detect_irregular_sampling(timestamps: np.ndarray, tolerance: float = 0.1) -> dict:
    """
    Detect irregular sampling in timestamps.

    Args:
        timestamps: Array of timestamps
        tolerance: Relative tolerance for considering intervals irregular

    Returns:
        Dictionary with sampling statistics
    """
    if len(timestamps) < 2:
        return {'is_regular': True, 'n_samples': len(timestamps)}

    intervals = np.diff(timestamps)
    median_interval = np.median(intervals)
    std_interval = np.std(intervals)

    # Check for regularity
    is_regular = std_interval / median_interval < tolerance if median_interval > 0 else True

    # Find gaps
    gap_threshold = median_interval * 3
    n_gaps = np.sum(intervals > gap_threshold)

    return {
        'is_regular': is_regular,
        'n_samples': len(timestamps),
        'median_interval': float(median_interval),
        'std_interval': float(std_interval),
        'min_interval': float(np.min(intervals)),
        'max_interval': float(np.max(intervals)),
        'n_gaps': int(n_gaps),
        'coefficient_of_variation': float(std_interval / median_interval) if median_interval > 0 else 0
    }


def align_multiple_series(
    series_list: List[np.ndarray],
    timestamps_list: List[np.ndarray],
    target_timestamps: Optional[np.ndarray] = None,
    method: str = 'linear'
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Align multiple time series to common timestamps.

    Args:
        series_list: List of time series arrays
        timestamps_list: List of corresponding timestamps
        target_timestamps: Target timestamps (if None, use union of all timestamps)
        method: Interpolation method

    Returns:
        Tuple of (aligned_series_list, common_timestamps)
    """
    if target_timestamps is None:
        # Create union of all timestamps
        all_timestamps = np.concatenate(timestamps_list)
        target_timestamps = np.unique(np.sort(all_timestamps))

    aligned = []
    for data, ts in zip(series_list, timestamps_list):
        interp_func = interpolate.interp1d(
            ts, data, axis=0, kind=method,
            bounds_error=False, fill_value='extrapolate'
        )
        aligned.append(interp_func(target_timestamps))

    return aligned, target_timestamps
