#!/usr/bin/env python3
"""
Audio and Spectrogram Augmentations for Triplet Learning

This module provides augmentations that work on AST features to reduce overfitting
in triplet learning scenarios. Designed to integrate seamlessly with the existing
preprocessing and training pipeline.
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SpectrogramAugmentations:
    """
    Spectrogram-level augmentations for AST features.
    Operates on the input_values tensor from AST feature extractor.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize augmentation pipeline.

        Args:
            config: Dictionary with augmentation parameters
        """
        if config is None:
            config = {}

        # Time masking parameters
        self.time_mask_prob = config.get('time_mask_prob', 0.3)
        self.time_mask_max_length = config.get('time_mask_max_length', 20)  # ~10% of 200 frames

        # Frequency masking parameters
        self.freq_mask_prob = config.get('freq_mask_prob', 0.3)
        self.freq_mask_max_length = config.get('freq_mask_max_length', 16)  # ~10% of 128 mel bins

        # Gaussian noise parameters
        self.noise_prob = config.get('noise_prob', 0.5)
        self.noise_std = config.get('noise_std', 0.05)

        # Mixup parameters (for same-subgenre mixing)
        self.mixup_prob = config.get('mixup_prob', 0.2)
        self.mixup_alpha = config.get('mixup_alpha', 0.3)

        # Volume scaling
        self.volume_prob = config.get('volume_prob', 0.4)
        self.volume_range = config.get('volume_range', (0.8, 1.2))

        # Temporal shifts (small random time offsets)
        self.temporal_shift_prob = config.get('temporal_shift_prob', 0.3)
        self.temporal_shift_max = config.get('temporal_shift_max', 5)  # frames

        self.enabled = config.get('enabled', True)

        if self.enabled:
            logger.info(f"SpectrogramAugmentations initialized: "
                       f"time_mask={self.time_mask_prob}, freq_mask={self.freq_mask_prob}, "
                       f"noise={self.noise_prob}, mixup={self.mixup_prob}")

    def __call__(self, features: torch.Tensor,
                 subgenre: Optional[str] = None,
                 training: bool = True) -> torch.Tensor:
        """
        Apply augmentations to AST features.

        Args:
            features: AST input_values tensor of shape [batch_size, sequence_length]
                     or [sequence_length] for single sample
            subgenre: Subgenre label (for mixup logic)
            training: Whether model is in training mode

        Returns:
            Augmented features tensor
        """
        if not self.enabled or not training:
            return features

        # Ensure batch dimension
        original_shape = features.shape
        if features.dim() == 1:
            features = features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, seq_len = features.shape
        device = features.device

        # Apply augmentations
        augmented_features = features.clone()

        # 1. Gaussian noise
        if random.random() < self.noise_prob:
            noise = torch.randn_like(augmented_features) * self.noise_std
            augmented_features = augmented_features + noise

        # 2. Volume scaling
        if random.random() < self.volume_prob:
            volume_factor = random.uniform(*self.volume_range)
            augmented_features = augmented_features * volume_factor

        # 3. Time masking (horizontal masking)
        if random.random() < self.time_mask_prob:
            augmented_features = self._apply_time_mask(augmented_features)

        # 4. Temporal shift (small time offset)
        if random.random() < self.temporal_shift_prob:
            augmented_features = self._apply_temporal_shift(augmented_features)

        # Restore original shape if needed
        if squeeze_output:
            augmented_features = augmented_features.squeeze(0)

        return augmented_features

    def _apply_time_mask(self, features: torch.Tensor) -> torch.Tensor:
        """Apply time masking to features."""
        batch_size, seq_len = features.shape

        for i in range(batch_size):
            # Random mask length
            mask_length = random.randint(1, min(self.time_mask_max_length, seq_len // 4))

            # Random start position
            start_pos = random.randint(0, max(0, seq_len - mask_length))

            # Apply mask (set to mean value of the sequence)
            sequence_mean = features[i].mean()
            features[i, start_pos:start_pos + mask_length] = sequence_mean

        return features

    def _apply_temporal_shift(self, features: torch.Tensor) -> torch.Tensor:
        """Apply small temporal shifts."""
        batch_size, seq_len = features.shape

        for i in range(batch_size):
            # Random shift amount
            shift = random.randint(-self.temporal_shift_max, self.temporal_shift_max)

            if shift == 0:
                continue

            # Apply circular shift
            features[i] = torch.roll(features[i], shift, dims=0)

        return features


class AST2DSpectrogramAugmentations:
    """
    2D spectrogram augmentations for AST features when reshaped to 2D.
    AST uses mel spectrograms internally, so we can apply 2D augmentations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize 2D augmentation pipeline.

        Args:
            config: Dictionary with augmentation parameters
        """
        if config is None:
            config = {}

        # Assuming AST typical dimensions: 128 mel bins x ~200 time frames
        self.mel_bins = config.get('mel_bins', 128)
        self.time_frames = config.get('time_frames', 200)

        # SpecAugment parameters
        self.freq_mask_prob = config.get('freq_mask_prob', 0.4)
        self.freq_mask_max = config.get('freq_mask_max', 16)  # max mel bins to mask

        self.time_mask_prob = config.get('time_mask_prob', 0.4)
        self.time_mask_max = config.get('time_mask_max', 20)   # max time frames to mask

        # Number of masks to apply
        self.num_freq_masks = config.get('num_freq_masks', 1)
        self.num_time_masks = config.get('num_time_masks', 1)

        self.enabled = config.get('enabled', True)

        if self.enabled:
            logger.info(f"AST2DSpectrogramAugmentations initialized: "
                       f"freq_mask={self.freq_mask_prob}, time_mask={self.time_mask_prob}")

    def __call__(self, features: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Apply 2D spectrogram augmentations to AST features.

        Args:
            features: AST input_values tensor [batch_size, sequence_length]
            training: Whether model is in training mode

        Returns:
            Augmented features tensor
        """
        if not self.enabled or not training:
            return features

        # Reshape to 2D spectrogram format
        original_shape = features.shape
        if features.dim() == 1:
            features = features.unsqueeze(0)

        batch_size, seq_len = features.shape

        # Reshape to approximate 2D spectrogram
        # AST typically uses sequence length of ~1024 for 10s audio
        # This roughly corresponds to 128 mel bins x ~8 time frames per "super-frame"
        try:
            # Attempt to reshape to 2D
            if seq_len == 1024:  # Standard AST input
                features_2d = features.view(batch_size, self.mel_bins, -1)  # [B, 128, 8]
            else:
                # Fallback: try to make it roughly square
                sqrt_len = int(np.sqrt(seq_len))
                remainder = seq_len - sqrt_len * sqrt_len
                if remainder == 0:
                    features_2d = features.view(batch_size, sqrt_len, sqrt_len)
                else:
                    # Pad to make it divisible
                    pad_len = sqrt_len * (sqrt_len + 1) - seq_len
                    features_padded = F.pad(features, (0, pad_len))
                    features_2d = features_padded.view(batch_size, sqrt_len, sqrt_len + 1)

        except RuntimeError:
            # If reshaping fails, fall back to 1D augmentations
            logger.warning(f"Could not reshape features of size {seq_len} to 2D, skipping 2D augmentations")
            return features.view(original_shape)

        batch_size, freq_dim, time_dim = features_2d.shape

        # Apply SpecAugment-style masking
        for i in range(batch_size):
            # Frequency masking
            if random.random() < self.freq_mask_prob:
                for _ in range(self.num_freq_masks):
                    mask_size = random.randint(1, min(self.freq_mask_max, freq_dim // 4))
                    mask_start = random.randint(0, max(0, freq_dim - mask_size))

                    # Set masked frequencies to mean value
                    mask_value = features_2d[i].mean()
                    features_2d[i, mask_start:mask_start + mask_size, :] = mask_value

            # Time masking
            if random.random() < self.time_mask_prob:
                for _ in range(self.num_time_masks):
                    mask_size = random.randint(1, min(self.time_mask_max, time_dim // 4))
                    mask_start = random.randint(0, max(0, time_dim - mask_size))

                    # Set masked time frames to mean value
                    mask_value = features_2d[i].mean()
                    features_2d[i, :, mask_start:mask_start + mask_size] = mask_value

        # Reshape back to original format
        augmented_features = features_2d.view(batch_size, -1)

        # Trim to original sequence length if we padded
        if augmented_features.shape[1] > seq_len:
            augmented_features = augmented_features[:, :seq_len]

        return augmented_features.view(original_shape)


class AugmentationPipeline:
    """
    Combined augmentation pipeline that applies multiple augmentation strategies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the augmentation pipeline.

        Args:
            config: Configuration dictionary with augmentation settings
        """
        if config is None:
            config = {}

        self.config = config
        self.enabled = config.get('enabled', True)

        # Initialize individual augmentation modules
        self.spec_aug = SpectrogramAugmentations(config.get('spectrogram', {}))
        self.spec_2d_aug = AST2DSpectrogramAugmentations(config.get('spectrogram_2d', {}))

        # Pipeline settings
        self.use_2d_augmentations = config.get('use_2d_augmentations', True)
        self.augmentation_probability = config.get('augmentation_probability', 0.8)

        if self.enabled:
            logger.info(f"AugmentationPipeline initialized: "
                       f"enabled={self.enabled}, 2d_aug={self.use_2d_augmentations}, "
                       f"prob={self.augmentation_probability}")

    def __call__(self, features: torch.Tensor,
                 subgenre: Optional[str] = None,
                 training: bool = True) -> torch.Tensor:
        """
        Apply the full augmentation pipeline.

        Args:
            features: AST input_values tensor
            subgenre: Subgenre label for context-aware augmentations
            training: Whether model is in training mode

        Returns:
            Augmented features tensor
        """
        if not self.enabled or not training:
            return features

        # Randomly decide whether to apply augmentations
        if random.random() > self.augmentation_probability:
            return features

        augmented_features = features

        # Apply 1D augmentations
        augmented_features = self.spec_aug(augmented_features, subgenre=subgenre, training=training)

        # Apply 2D augmentations if enabled
        if self.use_2d_augmentations:
            augmented_features = self.spec_2d_aug(augmented_features, training=training)

        return augmented_features


def get_default_augmentation_config() -> Dict[str, Any]:
    """
    Get default augmentation configuration optimized for triplet learning.

    Returns:
        Default configuration dictionary
    """
    return {
        'enabled': True,
        'augmentation_probability': 0.7,  # Apply augmentations to 70% of samples
        'use_2d_augmentations': True,

        'spectrogram': {
            'enabled': True,
            'time_mask_prob': 0.3,
            'time_mask_max_length': 20,
            'freq_mask_prob': 0.3,
            'freq_mask_max_length': 16,
            'noise_prob': 0.5,
            'noise_std': 0.05,
            'mixup_prob': 0.2,
            'mixup_alpha': 0.3,
            'volume_prob': 0.4,
            'volume_range': (0.8, 1.2),
            'temporal_shift_prob': 0.3,
            'temporal_shift_max': 5
        },

        'spectrogram_2d': {
            'enabled': True,
            'mel_bins': 128,
            'time_frames': 200,
            'freq_mask_prob': 0.4,
            'freq_mask_max': 16,
            'time_mask_prob': 0.4,
            'time_mask_max': 20,
            'num_freq_masks': 1,
            'num_time_masks': 1
        }
    }


def create_augmentation_pipeline(config: Optional[Dict[str, Any]] = None) -> AugmentationPipeline:
    """
    Factory function to create an augmentation pipeline.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured AugmentationPipeline instance
    """
    if config is None:
        config = get_default_augmentation_config()

    return AugmentationPipeline(config)