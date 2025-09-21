#!/usr/bin/env python3
"""
Improved Configuration Management for AST Triplet Training

This module provides a well-structured, type-safe configuration system with:
- Clear hierarchical organization
- Comprehensive validation
- Better default values
- Support for augmentations
- Backward compatibility
"""

import os
import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """High-level experiment metadata and identification."""
    name: str = "AST Triplet Training"
    description: str = "Audio Spectrogram Transformer triplet learning for music similarity"
    random_seed: int = 42
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class DatasetConfig:
    """Dataset and data loading configuration."""
    # Data source
    preprocessed_features_path: str = "preprocessed_features"

    # Dataset generation
    dataset_generation: Dict[str, Any] = field(default_factory=lambda: {
        "max_positive_tracks_per_anchor": 4,
        "triplets_per_positive_track": 1
    })

    # Train/test splitting
    test_split_ratio: float = 0.2
    min_files_per_subgenre: int = 5

    # Data resampling (for reducing overfitting)
    resample_train_samples: bool = False
    resample_start_epoch: int = 1
    resample_cadence: int = 1
    resample_fraction: float = 0.3
    resample_schedule_override: Optional[Dict[int, float]] = None

    # Negative mining
    negative_mining: str = "none"  # "none", "semi_hard", "hard"
    negative_mining_start_epoch: int = 1

    # Stratified batching
    stratified_batching: bool = True
    min_samples_per_subgenre: int = 2

    # Quick testing
    quick_test_mode: bool = False
    quick_test_fraction: float = 0.05


@dataclass
class ModelArchitectureConfig:
    """Model architecture and layer configuration."""
    # Base model
    base_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"

    # Projection head architecture
    projection_head: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_dim": 512,
        "output_dim": 128,
        "activation": "relu",
        "dropout": 0.15
    })

    # Triplet loss configuration
    triplet_margin: float = 0.5
    margin_scheduling: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "start_margin": 0.5,
        "end_margin": 0.8,
        "schedule_epochs": 1
    })


@dataclass
class TrainingHyperparametersConfig:
    """Training hyperparameters and optimization settings."""
    # Basic training parameters
    epochs: int = 10
    batch_size: int = 24
    gradient_accumulation_steps: int = 3

    # Optimization
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0

    # Learning rate scheduling
    scheduler: Dict[str, Any] = field(default_factory=lambda: {
        "type": "linear",  # "linear", "cosine", "polynomial"
        "warmup_ratio": 0.1,
        "min_lr": 1e-7
    })

    # Advanced optimization
    optimizer: Dict[str, Any] = field(default_factory=lambda: {
        "type": "adamw",
        "betas": [0.9, 0.999],
        "eps": 1e-8
    })

    # Dual-group learning rates (for AST base vs projection head)
    dual_group_lr: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "head_lr_multiplier": 1.0,
        "multiplier_converge_epoch": 1
    })


@dataclass
class RegularizationConfig:
    """Regularization and overfitting prevention settings."""
    # Early stopping
    early_stopping: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "patience": 3,
        "min_delta": 0.01,
        "post_resample_grace_epochs": 2
    })

    # Data augmentation
    augmentation: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "augmentation_probability": 0.5,
        "use_2d_augmentations": True,
        "spectrogram": {
            "enabled": True,
            "time_mask_prob": 0.2,
            "time_mask_max_length": 15,
            "freq_mask_prob": 0.2,
            "freq_mask_max_length": 12,
            "noise_prob": 0.3,
            "noise_std": 0.03,
            "volume_prob": 0.3,
            "volume_range": [0.9, 1.1],
            "temporal_shift_prob": 0.2,
            "temporal_shift_max": 3
        },
        "spectrogram_2d": {
            "enabled": True,
            "freq_mask_prob": 0.3,
            "freq_mask_max": 12,
            "time_mask_prob": 0.3,
            "time_mask_max": 15,
            "num_freq_masks": 1,
            "num_time_masks": 1
        }
    })


@dataclass
class ComputeConfig:
    """Compute and hardware configuration."""
    # Hardware settings
    device: str = "auto"  # "auto", "cuda", "cpu"
    force_single_gpu: bool = False

    # Precision and performance
    mixed_precision: Dict[str, Any] = field(default_factory=lambda: {
        "bf16": False,
        "fp16": False,
        "tf32": False
    })

    # PyTorch backends
    backends: Dict[str, Any] = field(default_factory=lambda: {
        "allow_tf32_matmul": False,
        "allow_tf32_cudnn": False,
        "set_float32_matmul_precision": None
    })

    # Data loading
    data_loading: Dict[str, Any] = field(default_factory=lambda: {
        "num_workers": 2,
        "pin_memory": False,
        "prefetch_factor": 2,
        "persistent_workers": True
    })


@dataclass
class LoggingConfig:
    """Logging, evaluation, and output configuration."""
    # Logging frequency
    logging_strategy: str = "epoch"  # "steps", "epoch"
    logging_steps: int = 100

    # Evaluation strategy
    eval_strategy: str = "epoch"  # "steps", "epoch", "no"
    eval_steps: Optional[int] = None

    # Saving strategy
    save_strategy: str = "epoch"  # "steps", "epoch", "no"
    save_steps: Optional[int] = None
    save_total_limit: int = 2

    # Output settings
    output_dir: str = "training_output"
    run_name: Optional[str] = None

    # Progress display
    disable_tqdm: bool = False
    report_to: Optional[List[str]] = None  # ["wandb", "tensorboard"]


@dataclass
class MasterConfig:
    """Master configuration combining all components."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    training: TrainingHyperparametersConfig = field(default_factory=TrainingHyperparametersConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._apply_experiment_seed()
        self._setup_output_directories()
        self._validate_configuration()

    def _apply_experiment_seed(self):
        """Apply random seed consistently."""
        import random
        import numpy as np
        import torch

        seed = self.experiment.random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _setup_output_directories(self):
        """Create output directories if they don't exist."""
        if self.logging.run_name is None:
            self.logging.run_name = f"run_{self.experiment.name.replace(' ', '_').lower()}"

        output_path = Path(self.logging.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    def _validate_configuration(self):
        """Validate configuration consistency and requirements."""
        # Validate data paths
        if not Path(self.data.preprocessed_features_path).exists():
            logger.warning(f"Preprocessed features path does not exist: {self.data.preprocessed_features_path}")

        # Validate batch size for stratified batching
        if self.data.stratified_batching:
            min_batch_size = 7 * self.data.min_samples_per_subgenre  # Assume 7 subgenres
            if self.training.batch_size < min_batch_size:
                logger.warning(f"Batch size ({self.training.batch_size}) may be too small for stratified batching")

        # Validate early stopping configuration
        if (self.regularization.early_stopping["enabled"] and
            self.regularization.early_stopping["patience"] < 1):
            raise ValueError("Early stopping patience must be >= 1")

        # Validate learning rate
        if self.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        config_dict = asdict(self)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, separators=(',', ': '))

        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'MasterConfig':
        """Load configuration from JSON file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        # Reconstruct nested dataclasses
        experiment = ExperimentConfig(**config_dict.get('experiment', {}))
        data = DatasetConfig(**config_dict.get('data', {}))
        model = ModelArchitectureConfig(**config_dict.get('model', {}))
        training = TrainingHyperparametersConfig(**config_dict.get('training', {}))
        regularization = RegularizationConfig(**config_dict.get('regularization', {}))
        compute = ComputeConfig(**config_dict.get('compute', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))

        config = cls(
            experiment=experiment,
            data=data,
            model=model,
            training=training,
            regularization=regularization,
            compute=compute,
            logging=logging_config
        )

        logger.info(f"Configuration loaded from {filepath}")
        return config

    def to_legacy_format(self) -> 'ExperimentConfig':
        """Convert to legacy ExperimentConfig format for backward compatibility."""
        from config import ExperimentConfig, ModelConfig, TrainingConfig, DataConfig

        # Convert model config
        model_config = ModelConfig(
            pretrained_model=self.model.base_model,
            projection_hidden_layers=[self.model.projection_head["hidden_dim"]],
            projection_activation=self.model.projection_head["activation"],
            projection_dropout_rate=self.model.projection_head["dropout"],
            output_dim=self.model.projection_head["output_dim"]
        )

        # Convert training config
        training_config = TrainingConfig(
            batch_size=self.training.batch_size,
            epochs=self.training.epochs,
            learning_rate=self.training.learning_rate,
            weight_decay=self.training.weight_decay,
            gradient_accumulation_steps=self.training.gradient_accumulation_steps,
            triplet_margin=self.model.triplet_margin,
            enable_early_stopping=self.regularization.early_stopping["enabled"],
            early_stopping_patience=self.regularization.early_stopping["patience"],
            early_stopping_min_delta=self.regularization.early_stopping["min_delta"],
            post_resample_grace_epochs=self.regularization.early_stopping["post_resample_grace_epochs"],
            seed=self.experiment.random_seed
        )

        # Convert data config
        data_config = DataConfig(
            chunks_dir=self.data.preprocessed_features_path,
            test_split_ratio=self.data.test_split_ratio,
            max_positive_tracks=self.data.dataset_generation["max_positive_tracks_per_anchor"],
            triplets_per_positive_track=self.data.dataset_generation["triplets_per_positive_track"],
            resample_train_samples=self.data.resample_train_samples,
            stratified_batching=self.data.stratified_batching,
            min_per_subgenre=self.data.min_samples_per_subgenre
        )

        return ExperimentConfig(
            model=model_config,
            training=training_config,
            data=data_config,
            experiment_name=self.experiment.name,
            description=self.experiment.description
        )


# Factory functions for common configurations

def create_baseline_config() -> MasterConfig:
    """Create baseline configuration without augmentations."""
    config = MasterConfig()
    config.experiment.name = "Baseline Training"
    config.experiment.description = "Baseline triplet training without augmentations"
    return config


def create_augmentation_test_config() -> MasterConfig:
    """Create configuration for testing augmentations."""
    config = MasterConfig()
    config.experiment.name = "Augmentation Test"
    config.experiment.description = "Testing conservative audio augmentations"

    # Enable conservative augmentations
    config.regularization.augmentation["enabled"] = True
    config.regularization.augmentation["augmentation_probability"] = 0.5

    return config


def create_overfitting_mitigation_config() -> MasterConfig:
    """Create configuration focused on reducing overfitting."""
    config = MasterConfig()
    config.experiment.name = "Overfitting Mitigation"
    config.experiment.description = "Aggressive regularization to combat overfitting"

    # Enable aggressive augmentations
    config.regularization.augmentation["enabled"] = True
    config.regularization.augmentation["augmentation_probability"] = 0.8

    # Stronger regularization
    config.training.weight_decay = 5e-4
    config.training.learning_rate = 5e-6
    config.model.projection_head["dropout"] = 0.5
    config.model.loss["margin_start"] = 0.8

    # More aggressive early stopping
    config.regularization.early_stopping["patience"] = 5
    config.regularization.early_stopping["min_delta"] = 0.005

    return config


def create_cluster_optimized_config() -> MasterConfig:
    """Create configuration optimized for cluster training."""
    config = MasterConfig()
    config.experiment.name = "Cluster Training"
    config.experiment.description = "Optimized for high-performance cluster execution"

    # Cluster-optimized compute settings
    config.compute.mixed_precision["bf16"] = True
    config.compute.mixed_precision["tf32"] = True
    config.compute.backends["allow_tf32_matmul"] = True
    config.compute.backends["allow_tf32_cudnn"] = True
    config.compute.backends["set_float32_matmul_precision"] = "high"

    # Optimized data loading
    config.compute.data_loading["num_workers"] = 12
    config.compute.data_loading["pin_memory"] = True
    config.compute.data_loading["prefetch_factor"] = 8

    return config


def load_or_create_config(config_path: Optional[str] = None,
                         preset: str = "baseline",
                         test_run: bool = False) -> MasterConfig:
    """
    Load configuration from file or create with preset.

    Args:
        config_path: Path to configuration file
        preset: Preset configuration ("baseline", "augmentation", "overfitting", "cluster")
        test_run: Whether to create test run configuration

    Returns:
        MasterConfig instance
    """
    if config_path and Path(config_path).exists():
        return MasterConfig.load(config_path)

    # Create preset configuration
    preset_factories = {
        "baseline": create_baseline_config,
        "augmentation": create_augmentation_test_config,
        "overfitting": create_overfitting_mitigation_config,
        "cluster": create_cluster_optimized_config
    }

    factory = preset_factories.get(preset, create_baseline_config)
    config = factory()

    if test_run:
        # Modify for quick testing
        config.experiment.name += " (Test Run)"
        config.training.epochs = 2
        config.training.batch_size = 4
        config.data.quick_test_mode = True
        config.data.dataset_generation["max_positive_tracks_per_anchor"] = 2

    return config


# Backward compatibility function
def convert_legacy_config(legacy_config_path: str) -> MasterConfig:
    """Convert old configuration format to new structured format."""
    from config import ExperimentConfig

    legacy_config = ExperimentConfig.load(legacy_config_path)

    # Create new config and populate from legacy
    config = MasterConfig()

    # Experiment info
    config.experiment.name = legacy_config.experiment_name
    config.experiment.description = legacy_config.description
    config.experiment.random_seed = legacy_config.training.seed

    # Model configuration
    config.model.base_model = legacy_config.model.pretrained_model
    config.model.projection_head = {
        "hidden_dim": legacy_config.model.projection_hidden_layers[0] if legacy_config.model.projection_hidden_layers else 512,
        "output_dim": legacy_config.model.output_dim,
        "activation": legacy_config.model.projection_activation,
        "dropout": legacy_config.model.projection_dropout_rate
    }

    # Training configuration
    config.training.batch_size = legacy_config.training.batch_size
    config.training.epochs = legacy_config.training.epochs
    config.training.learning_rate = legacy_config.training.learning_rate
    config.training.weight_decay = legacy_config.training.weight_decay
    config.training.gradient_accumulation_steps = legacy_config.training.gradient_accumulation_steps

    # Data configuration
    config.data.preprocessed_features_path = legacy_config.data.chunks_dir
    config.data.test_split_ratio = legacy_config.data.test_split_ratio
    config.data.dataset_generation = {
        "max_positive_tracks_per_anchor": legacy_config.data.max_positive_tracks,
        "triplets_per_positive_track": legacy_config.data.triplets_per_positive_track
    }

    return config