"""
PDM Transforms Module

Data transformations and augmentations for time series data.
"""

import numpy as np
import torch
from typing import List, Optional, Union, Tuple


class Compose:
    """Compose multiple transforms together."""

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class Normalize:
    """Min-max normalization to [0, 1]."""

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_min = x.min()
        x_max = x.max()
        if x_max - x_min > 0:
            x = (x - x_min) / (x_max - x_min)
            x = x * (self.max_val - self.min_val) + self.min_val
        return x


class Standardize:
    """Standardize to zero mean and unit variance."""

    def __init__(self, mean: Optional[float] = None, std: Optional[float] = None):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean if self.mean is not None else x.mean()
        std = self.std if self.std is not None else x.std()
        if std > 0:
            x = (x - mean) / std
        return x


class Jitter:
    """Add random Gaussian noise."""

    def __init__(self, sigma: float = 0.03):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.sigma
        return x + noise


class Scale:
    """Random scaling of the signal."""

    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        factor = torch.randn(1) * self.sigma + 1.0
        return x * factor


class TimeWarp:
    """Time warping augmentation using smooth random curves."""

    def __init__(self, sigma: float = 0.2, num_knots: int = 4):
        self.sigma = sigma
        self.num_knots = num_knots

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Simple time warping implementation
        orig_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        seq_len = x.shape[0]

        # Generate random warp path
        random_warps = torch.randn(self.num_knots + 2) * self.sigma
        random_warps[0] = 0
        random_warps[-1] = 0

        # Interpolate to full sequence length
        warp_steps = torch.linspace(0, seq_len - 1, self.num_knots + 2)
        orig_steps = torch.arange(seq_len).float()

        # Create warped time steps
        warped_steps = orig_steps.clone()
        for i in range(len(warp_steps) - 1):
            mask = (orig_steps >= warp_steps[i]) & (orig_steps < warp_steps[i + 1])
            t = (orig_steps[mask] - warp_steps[i]) / (warp_steps[i + 1] - warp_steps[i])
            warped_steps[mask] = orig_steps[mask] + (1 - t) * random_warps[i] + t * random_warps[i + 1]

        # Clamp to valid range
        warped_steps = torch.clamp(warped_steps, 0, seq_len - 1)

        # Interpolate
        warped_indices = warped_steps.long()
        warped_indices = torch.clamp(warped_indices, 0, seq_len - 1)
        x_warped = x[warped_indices]

        if len(orig_shape) == 1:
            x_warped = x_warped.squeeze(-1)

        return x_warped


class MagnitudeWarp:
    """Magnitude warping using smooth random curves."""

    def __init__(self, sigma: float = 0.2, num_knots: int = 4):
        self.sigma = sigma
        self.num_knots = num_knots

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[0] if x.dim() > 0 else 1

        # Generate smooth warp curve
        knots = torch.randn(self.num_knots) * self.sigma + 1.0
        knot_positions = torch.linspace(0, seq_len - 1, self.num_knots)

        # Interpolate to full length
        warp_curve = torch.ones(seq_len)
        for i in range(self.num_knots - 1):
            start_idx = int(knot_positions[i])
            end_idx = int(knot_positions[i + 1])
            if end_idx > start_idx:
                t = torch.linspace(0, 1, end_idx - start_idx)
                warp_curve[start_idx:end_idx] = knots[i] * (1 - t) + knots[i + 1] * t

        if x.dim() > 1:
            warp_curve = warp_curve.unsqueeze(-1)

        return x * warp_curve


class Mixup:
    """
    Mixup augmentation - requires a second sample.
    This transform is typically applied at the batch level.
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(
        self,
        x1: torch.Tensor,
        y1: torch.Tensor,
        x2: torch.Tensor,
        y2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mix two samples with their labels."""
        lam = np.random.beta(self.alpha, self.alpha)
        x_mixed = lam * x1 + (1 - lam) * x2
        y_mixed = lam * y1 + (1 - lam) * y2
        return x_mixed, y_mixed

    @staticmethod
    def batch_mixup(
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup to a batch by shuffling and mixing."""
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size)

        x_mixed = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]

        return x_mixed, y_a, y_b, lam


class Cutout:
    """Randomly zero out segments of the time series."""

    def __init__(self, n_holes: int = 1, length_ratio: float = 0.1):
        self.n_holes = n_holes
        self.length_ratio = length_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        seq_len = x.shape[0]
        hole_length = int(seq_len * self.length_ratio)

        for _ in range(self.n_holes):
            start = np.random.randint(0, max(1, seq_len - hole_length))
            end = min(start + hole_length, seq_len)
            x[start:end] = 0

        return x


class FrequencyMask:
    """Mask random frequency bands in the Fourier domain."""

    def __init__(self, max_mask_ratio: float = 0.2, n_masks: int = 1):
        self.max_mask_ratio = max_mask_ratio
        self.n_masks = n_masks

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Apply FFT
        x_fft = torch.fft.rfft(x, dim=0)
        freq_len = x_fft.shape[0]
        mask_len = int(freq_len * self.max_mask_ratio)

        for _ in range(self.n_masks):
            start = np.random.randint(0, max(1, freq_len - mask_len))
            end = min(start + mask_len, freq_len)
            x_fft[start:end] = 0

        # Apply inverse FFT
        x_masked = torch.fft.irfft(x_fft, n=x.shape[0], dim=0)
        return x_masked


class GaussianNoise:
    """Add Gaussian noise with specified SNR."""

    def __init__(self, snr_db: float = 20.0):
        self.snr_db = snr_db

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        signal_power = torch.mean(x ** 2)
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        return x + noise


class RandomCrop:
    """Randomly crop and resize to original length."""

    def __init__(self, crop_ratio_range: Tuple[float, float] = (0.8, 1.0)):
        self.crop_ratio_range = crop_ratio_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[0]
        crop_ratio = np.random.uniform(*self.crop_ratio_range)
        crop_len = int(seq_len * crop_ratio)

        start = np.random.randint(0, seq_len - crop_len + 1)
        x_cropped = x[start:start + crop_len]

        # Resize back to original length using interpolation
        x_cropped = x_cropped.unsqueeze(0).unsqueeze(0)  # [1, 1, crop_len, C]
        if x_cropped.dim() == 3:
            x_resized = torch.nn.functional.interpolate(
                x_cropped.permute(0, 2, 1),
                size=seq_len,
                mode='linear',
                align_corners=True
            ).permute(0, 2, 1)
        else:
            x_resized = torch.nn.functional.interpolate(
                x_cropped,
                size=seq_len,
                mode='linear',
                align_corners=True
            )
        return x_resized.squeeze(0).squeeze(0)


class ChannelDropout:
    """Randomly drop channels (for multi-channel data)."""

    def __init__(self, dropout_prob: float = 0.1):
        self.dropout_prob = dropout_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            return x

        num_channels = x.shape[-1]
        mask = torch.rand(num_channels) > self.dropout_prob
        mask = mask.float()

        # Ensure at least one channel remains
        if mask.sum() == 0:
            mask[np.random.randint(num_channels)] = 1

        return x * mask


class RandomFlip:
    """Randomly flip the time series."""

    def __init__(self, p: float = 0.5, axis: str = 'time'):
        self.p = p
        self.axis = axis

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() < self.p:
            if self.axis == 'time':
                x = torch.flip(x, dims=[0])
            elif self.axis == 'value':
                x = -x
        return x
