#!/usr/bin/env python3
"""
Training Integration Patch for Audio Augmentations

This module provides a patch to integrate audio augmentations into your existing
training pipeline with minimal changes to the main training script.
"""

import logging
from typing import Dict, Any, Optional

from dataset_augmented import (
    AugmentedTripletFeatureDataset,
    create_augmented_dataset,
    get_conservative_augmentation_config,
    get_aggressive_augmentation_config
)
from config import ExperimentConfig

logger = logging.getLogger(__name__)


def wrap_dataset_with_augmentations(base_dataset,
                                  config: ExperimentConfig,
                                  split_name: str = "train") -> AugmentedTripletFeatureDataset:
    """
    Wrap a base dataset with augmentations based on configuration.

    Args:
        base_dataset: Base ImprovedTripletFeatureDataset instance
        config: Experiment configuration
        split_name: Split name ('train', 'test', etc.)

    Returns:
        AugmentedTripletFeatureDataset instance
    """
    # Check if augmentation config exists in the experiment config
    augmentation_config = None

    if hasattr(config, 'augmentation') and config.augmentation:
        # Convert config object to dictionary
        augmentation_config = config.augmentation
        if hasattr(augmentation_config, '__dict__'):
            augmentation_config = vars(augmentation_config)
        logger.info("Using augmentation configuration from experiment config")
    else:
        # Use default conservative augmentation
        if split_name == "train":
            augmentation_config = get_conservative_augmentation_config()
            logger.info("Using default conservative augmentation configuration")
        else:
            augmentation_config = {'enabled': False}
            logger.info(f"Augmentations disabled for {split_name} split")

    return create_augmented_dataset(
        base_dataset=base_dataset,
        config=config,
        split_name=split_name,
        augmentation_config=augmentation_config
    )


def create_datasets_with_augmentations(train_data, test_data, config: ExperimentConfig):
    """
    Create both training and test datasets with appropriate augmentation settings.

    Args:
        train_data: Training triplet data
        test_data: Test triplet data
        config: Experiment configuration

    Returns:
        Tuple of (augmented_train_dataset, augmented_test_dataset)
    """
    # Import the base dataset class (avoid circular imports)
    from AST_Triplet_training import ImprovedTripletFeatureDataset

    # Create base datasets
    logger.info("Creating datasets...")
    base_train_dataset = ImprovedTripletFeatureDataset(train_data, config, "train")
    base_test_dataset = ImprovedTripletFeatureDataset(test_data, config, "test")

    # Wrap with augmentations
    augmented_train_dataset = wrap_dataset_with_augmentations(base_train_dataset, config, "train")
    augmented_test_dataset = wrap_dataset_with_augmentations(base_test_dataset, config, "test")

    return augmented_train_dataset, augmented_test_dataset


def log_augmentation_status(dataset, split_name: str):
    """Log augmentation status for a dataset."""
    if hasattr(dataset, 'augmentation_pipeline'):
        pipeline = dataset.augmentation_pipeline
        if pipeline.enabled:
            logger.info(f"{split_name} dataset: augmentations ENABLED "
                       f"(prob={pipeline.augmentation_probability:.1%}, "
                       f"2d_aug={pipeline.use_2d_augmentations})")
        else:
            logger.info(f"{split_name} dataset: augmentations DISABLED")
    else:
        logger.info(f"{split_name} dataset: no augmentation pipeline (base dataset)")


# Example usage functions for different augmentation levels

def get_minimal_augmentation_config() -> Dict[str, Any]:
    """Get minimal augmentation configuration for testing."""
    return {
        'enabled': True,
        'augmentation_probability': 0.3,
        'use_2d_augmentations': False,
        'spectrogram': {
            'enabled': True,
            'time_mask_prob': 0.1,
            'time_mask_max_length': 10,
            'freq_mask_prob': 0.1,
            'freq_mask_max_length': 8,
            'noise_prob': 0.2,
            'noise_std': 0.02,
            'mixup_prob': 0.0,
            'volume_prob': 0.2,
            'volume_range': (0.95, 1.05),
            'temporal_shift_prob': 0.1,
            'temporal_shift_max': 2
        },
        'spectrogram_2d': {'enabled': False}
    }


def create_training_datasets(train_data, test_data, config: ExperimentConfig, augmentation_level: str = "conservative"):
    """
    Create training datasets with specified augmentation level.

    Args:
        train_data: Training triplet data
        test_data: Test triplet data
        config: Experiment configuration
        augmentation_level: 'none', 'minimal', 'conservative', 'aggressive'

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    from AST_Triplet_training import ImprovedTripletFeatureDataset

    # Create base datasets
    base_train_dataset = ImprovedTripletFeatureDataset(train_data, config, "train")
    base_test_dataset = ImprovedTripletFeatureDataset(test_data, config, "test")

    # Select augmentation configuration
    if augmentation_level == "none":
        augmentation_config = {'enabled': False}
    elif augmentation_level == "minimal":
        augmentation_config = get_minimal_augmentation_config()
    elif augmentation_level == "conservative":
        augmentation_config = get_conservative_augmentation_config()
    elif augmentation_level == "aggressive":
        augmentation_config = get_aggressive_augmentation_config()
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation_level}")

    logger.info(f"Creating datasets with {augmentation_level} augmentation level")

    # Create augmented datasets
    train_dataset = create_augmented_dataset(base_train_dataset, config, "train", augmentation_config)
    test_dataset = create_augmented_dataset(base_test_dataset, config, "test", {'enabled': False})

    # Log status
    log_augmentation_status(train_dataset, "Train")
    log_augmentation_status(test_dataset, "Test")

    return train_dataset, test_dataset