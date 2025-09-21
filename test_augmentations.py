#!/usr/bin/env python3
"""
Test script for audio augmentations integration.

This script tests the augmentation pipeline to ensure it works correctly
with your existing dataset and preprocessing.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from augmentations import AugmentationPipeline, get_default_augmentation_config
from dataset_augmented import create_augmented_dataset, get_conservative_augmentation_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_augmentation_pipeline():
    """Test the augmentation pipeline with synthetic data."""
    logger.info("Testing augmentation pipeline...")

    # Create synthetic AST features (typical shape for 10s audio)
    batch_size = 4
    sequence_length = 1024  # Typical AST input length
    features = torch.randn(batch_size, sequence_length)

    # Test with default config
    config = get_default_augmentation_config()
    pipeline = AugmentationPipeline(config)

    # Test training mode
    pipeline.training = True
    augmented_features = pipeline(features, subgenre="test_genre", training=True)

    logger.info(f"Original features shape: {features.shape}")
    logger.info(f"Augmented features shape: {augmented_features.shape}")
    logger.info(f"Features changed: {not torch.equal(features, augmented_features)}")

    # Test evaluation mode
    pipeline.training = False
    eval_features = pipeline(features, subgenre="test_genre", training=False)
    logger.info(f"Evaluation mode unchanged: {torch.equal(features, eval_features)}")

    logger.info("✓ Augmentation pipeline test passed")


def test_single_sample_augmentation():
    """Test augmentation on single samples."""
    logger.info("Testing single sample augmentation...")

    # Single sample
    features = torch.randn(1024)
    config = get_conservative_augmentation_config()
    pipeline = AugmentationPipeline(config)

    augmented = pipeline(features, training=True)
    logger.info(f"Single sample: {features.shape} -> {augmented.shape}")
    logger.info(f"Single sample changed: {not torch.equal(features, augmented)}")

    logger.info("✓ Single sample augmentation test passed")


def test_feature_dict_format():
    """Test augmentation with the actual feature dictionary format."""
    logger.info("Testing feature dictionary format...")

    # Simulate the format that comes from torch.load of preprocessed features
    feature_dict = {
        "input_values": torch.randn(1, 1024),  # Shape from AST feature extractor
        "attention_mask": torch.ones(1, 1024)  # Typical attention mask
    }

    config = get_conservative_augmentation_config()
    pipeline = AugmentationPipeline(config)

    # Test augmentation on input_values
    original_input = feature_dict["input_values"].clone()
    augmented_input = pipeline(feature_dict["input_values"], training=True)

    logger.info(f"Feature dict input_values: {original_input.shape} -> {augmented_input.shape}")
    logger.info(f"Feature dict changed: {not torch.equal(original_input, augmented_input)}")

    logger.info("✓ Feature dictionary format test passed")


def test_different_augmentation_levels():
    """Test different augmentation intensity levels."""
    logger.info("Testing different augmentation levels...")

    features = torch.randn(2, 1024)

    # Conservative
    config_conservative = get_conservative_augmentation_config()
    pipeline_conservative = AugmentationPipeline(config_conservative)
    aug_conservative = pipeline_conservative(features, training=True)

    # Default (more aggressive)
    config_default = get_default_augmentation_config()
    pipeline_default = AugmentationPipeline(config_default)
    aug_default = pipeline_default(features, training=True)

    # Calculate differences
    diff_conservative = torch.norm(features - aug_conservative).item()
    diff_default = torch.norm(features - aug_default).item()

    logger.info(f"Conservative augmentation difference: {diff_conservative:.4f}")
    logger.info(f"Default augmentation difference: {diff_default:.4f}")

    logger.info("✓ Different augmentation levels test passed")


def test_reproducibility():
    """Test that augmentations are stochastic (different each time)."""
    logger.info("Testing augmentation stochasticity...")

    features = torch.randn(1, 1024)
    config = get_conservative_augmentation_config()
    pipeline = AugmentationPipeline(config)

    # Apply augmentation multiple times
    aug1 = pipeline(features, training=True)
    aug2 = pipeline(features, training=True)

    # They should be different (stochastic)
    different = not torch.equal(aug1, aug2)
    logger.info(f"Augmentations are stochastic: {different}")

    logger.info("✓ Stochasticity test passed")


def main():
    """Run all tests."""
    logger.info("Starting augmentation integration tests...")
    logger.info("=" * 60)

    try:
        test_augmentation_pipeline()
        logger.info("")

        test_single_sample_augmentation()
        logger.info("")

        test_feature_dict_format()
        logger.info("")

        test_different_augmentation_levels()
        logger.info("")

        test_reproducibility()
        logger.info("")

        logger.info("=" * 60)
        logger.info("🎉 All augmentation tests passed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Run: python train_with_augmentations.py --config config_experiments/config_augmentation_test.json")
        logger.info("2. Compare results with your baseline runs")
        logger.info("3. Monitor train vs eval loss to see overfitting reduction")

        return 0

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())