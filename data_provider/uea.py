import os
import numpy as np
import pandas as pd
import torch
from typing import Optional, Union

# Import extended preprocessing modules
from data_provider.imputation import Imputer, impute_missing
from data_provider.resampling import Resampler, resample_to_length
from data_provider.normalization import ExtendedNormalizer


def collate_fn(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)

    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets, padding_masks


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.

    Extended to support robust, winsorized, and per-channel normalization methods.
    """

    SUPPORTED_METHODS = [
        'standardization', 'minmax', 'per_sample_std', 'per_sample_minmax',
        'robust', 'winsorized', 'per_channel', 'none'
    ]

    def __init__(self, norm_type='standardization', mean=None, std=None, min_val=None, max_val=None,
                 winsorize_limits=(5, 95)):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
                "robust": robust normalization using median and IQR (resistant to outliers)
                "winsorized": winsorized normalization with outlier clipping
                "per_channel": per-channel/feature standardization
                "none": no normalization
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
            winsorize_limits: tuple of (lower, upper) percentiles for winsorization
        """

        if norm_type not in self.SUPPORTED_METHODS:
            raise ValueError(f'Unknown normalization method: {norm_type}. '
                           f'Supported methods: {self.SUPPORTED_METHODS}')

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
        self.median = None
        self.iqr = None
        self.q1 = None
        self.q3 = None
        self.winsorize_limits = winsorize_limits
        self.winsorize_low = None
        self.winsorize_high = None

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / (grouped.transform('std') + np.finfo(float).eps)

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        elif self.norm_type == "robust":
            if self.median is None:
                self.median = df.median()
                self.q1 = df.quantile(0.25)
                self.q3 = df.quantile(0.75)
                self.iqr = self.q3 - self.q1
            return (df - self.median) / (self.iqr + np.finfo(float).eps)

        elif self.norm_type == "winsorized":
            if self.winsorize_low is None:
                self.winsorize_low = df.quantile(self.winsorize_limits[0] / 100)
                self.winsorize_high = df.quantile(self.winsorize_limits[1] / 100)
            # Clip values
            clipped = df.clip(lower=self.winsorize_low, upper=self.winsorize_high, axis=1)
            # Standardize clipped values
            if self.mean is None:
                self.mean = clipped.mean()
                self.std = clipped.std()
            return (clipped - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "per_channel":
            # Same as standardization but with clearer naming
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "none":
            return df

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))

    def inverse_normalize(self, df):
        """
        Inverse normalization to recover original scale.

        Args:
            df: normalized dataframe
        Returns:
            df: denormalized dataframe
        """
        if self.norm_type == "standardization" or self.norm_type == "per_channel":
            return df * (self.std + np.finfo(float).eps) + self.mean
        elif self.norm_type == "minmax":
            return df * (self.max_val - self.min_val + np.finfo(float).eps) + self.min_val
        elif self.norm_type == "robust":
            return df * (self.iqr + np.finfo(float).eps) + self.median
        elif self.norm_type == "winsorized":
            return df * (self.std + np.finfo(float).eps) + self.mean
        elif self.norm_type == "none":
            return df
        else:
            # Cannot inverse per_sample normalizations without original values
            return df


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y
