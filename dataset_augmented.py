#!/usr/bin/env python3
"""
Augmented Triplet Dataset

This module provides an augmented version of the ImprovedTripletFeatureDataset
that integrates audio augmentations during training to reduce overfitting.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
from torch.utils.data import Dataset as TorchDataset

from augmentations import AugmentationPipeline, get_default_augmentation_config
from config import ExperimentConfig

logger = logging.getLogger(__name__)


class AugmentedTripletFeatureDataset(TorchDataset):
    """
    Triplet dataset with integrated audio augmentations for overfitting reduction.

    This dataset extends the base functionality by applying augmentations to the
    AST features during training, which helps prevent memorization of specific
    spectral patterns.
    """

    def __init__(self,
                 base_dataset: TorchDataset,
                 config: ExperimentConfig,
                 augmentation_config: Optional[Dict[str, Any]] = None,
                 split_name: str = "train"):
        """
        Initialize augmented dataset.

        Args:
            base_dataset: Base ImprovedTripletFeatureDataset instance
            config: Main experiment configuration
            augmentation_config: Configuration for augmentations
            split_name: Split name ('train', 'test', etc.)
        """
        self.base_dataset = base_dataset
        self.config = config
        self.split_name = split_name

        # Initialize augmentation pipeline
        if augmentation_config is None:
            augmentation_config = get_default_augmentation_config()

        # Only enable augmentations for training split
        if split_name != "train":
            augmentation_config['enabled'] = False
            logger.info(f"Augmentations disabled for {split_name} split")

        self.augmentation_pipeline = AugmentationPipeline(augmentation_config)

        # Training mode flag (affects augmentation application)
        self.training = True

        logger.info(f"AugmentedTripletFeatureDataset initialized for {split_name} split: "
                   f"augmentations_enabled={self.augmentation_pipeline.enabled}, "
                   f"base_dataset_size={len(self.base_dataset)}")

    def train(self):
        """Set dataset to training mode (enables augmentations)."""
        self.training = True
        if hasattr(self.base_dataset, 'train'):
            self.base_dataset.train()

    def eval(self):
        """Set dataset to evaluation mode (disables augmentations)."""
        self.training = False
        if hasattr(self.base_dataset, 'eval'):
            self.base_dataset.eval()

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get triplet with augmentations applied during training.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing augmented triplet data
        """
        # Get base triplet data
        batch = self.base_dataset[idx]

        # Extract components
        anchor_input = batch["anchor_input"]
        positive_input = batch["positive_input"]
        negative_input = batch["negative_input"]
        anchor_subgenre = batch.get("anchor_subgenre", "unknown")
        negative_subgenre = batch.get("negative_subgenre", "unknown")

        # Apply augmentations during training
        if self.training and self.augmentation_pipeline.enabled:
            # Augment anchor
            anchor_input = self._augment_input(anchor_input, anchor_subgenre)

            # Augment positive (same subgenre as anchor)
            positive_input = self._augment_input(positive_input, anchor_subgenre)

            # Augment negative (different subgenre)
            negative_input = self._augment_input(negative_input, negative_subgenre)

        # Return augmented batch
        return {
            "anchor_input": anchor_input,
            "positive_input": positive_input,
            "negative_input": negative_input,
            "labels": batch.get("labels", 0),
            "anchor_subgenre": anchor_subgenre,
            "negative_subgenre": negative_subgenre
        }

    def _augment_input(self, input_dict: Dict[str, torch.Tensor],
                      subgenre: str) -> Dict[str, torch.Tensor]:
        """
        Apply augmentations to a single input dictionary.

        Args:
            input_dict: Dictionary containing AST features (e.g., {"input_values": tensor})
            subgenre: Subgenre label for context-aware augmentations

        Returns:
            Dictionary with augmented features
        """
        augmented_dict = {}

        for key, tensor in input_dict.items():
            if key == "input_values":
                # Apply augmentations to the main input tensor
                augmented_tensor = self.augmentation_pipeline(
                    tensor,
                    subgenre=subgenre,
                    training=self.training
                )
                augmented_dict[key] = augmented_tensor
            else:
                # Pass through other keys unchanged
                augmented_dict[key] = tensor

        return augmented_dict

    # Delegate other methods to base dataset
    def __getattr__(self, name):
        """Delegate attribute access to base dataset."""
        return getattr(self.base_dataset, name)


def create_augmented_dataset(base_dataset: TorchDataset,
                           config: ExperimentConfig,
                           split_name: str = "train",
                           augmentation_config: Optional[Dict[str, Any]] = None) -> AugmentedTripletFeatureDataset:
    """
    Factory function to create an augmented dataset.

    Args:
        base_dataset: Base dataset instance
        config: Experiment configuration
        split_name: Split name
        augmentation_config: Optional augmentation configuration

    Returns:
        AugmentedTripletFeatureDataset instance
    """
    return AugmentedTripletFeatureDataset(
        base_dataset=base_dataset,
        config=config,
        augmentation_config=augmentation_config,
        split_name=split_name
    )


def get_conservative_augmentation_config() -> Dict[str, Any]:
    """
    Get a conservative augmentation configuration for initial testing.

    Returns:
        Conservative augmentation configuration
    """
    return {
        'enabled': True,
        'augmentation_probability': 0.5,  # Apply to 50% of samples
        'use_2d_augmentations': True,

        'spectrogram': {
            'enabled': True,
            'time_mask_prob': 0.2,
            'time_mask_max_length': 15,
            'freq_mask_prob': 0.2,
            'freq_mask_max_length': 12,
            'noise_prob': 0.3,
            'noise_std': 0.03,
            'mixup_prob': 0.0,  # Disable mixup initially
            'mixup_alpha': 0.0,
            'volume_prob': 0.3,
            'volume_range': (0.9, 1.1),
            'temporal_shift_prob': 0.2,
            'temporal_shift_max': 3
        },

        'spectrogram_2d': {
            'enabled': True,
            'mel_bins': 128,
            'time_frames': 200,
            'freq_mask_prob': 0.3,
            'freq_mask_max': 12,
            'time_mask_prob': 0.3,
            'time_mask_max': 15,
            'num_freq_masks': 1,
            'num_time_masks': 1
        }
    }


def get_aggressive_augmentation_config() -> Dict[str, Any]:
    """
    Get an aggressive augmentation configuration for heavy regularization.

    Returns:
        Aggressive augmentation configuration
    """
    return {
        'enabled': True,
        'augmentation_probability': 0.8,  # Apply to 80% of samples
        'use_2d_augmentations': True,

        'spectrogram': {
            'enabled': True,
            'time_mask_prob': 0.5,
            'time_mask_max_length': 25,
            'freq_mask_prob': 0.5,
            'freq_mask_max_length': 20,
            'noise_prob': 0.6,
            'noise_std': 0.07,
            'mixup_prob': 0.3,
            'mixup_alpha': 0.4,
            'volume_prob': 0.5,
            'volume_range': (0.7, 1.3),
            'temporal_shift_prob': 0.4,
            'temporal_shift_max': 7
        },

        'spectrogram_2d': {
            'enabled': True,
            'mel_bins': 128,
            'time_frames': 200,
            'freq_mask_prob': 0.6,
            'freq_mask_max': 20,
            'time_mask_prob': 0.6,
            'time_mask_max': 25,
            'num_freq_masks': 2,
            'num_time_masks': 2
        }
    }