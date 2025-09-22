"""
Configuration management for AST Triplet Training.
Provides type-safe configuration with validation.
"""

import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    pretrained_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"

    # MLP Projection Head Configuration
    # Architecture: AST(768D) -> hidden_layers -> output_dim
    # Example: [512] creates: 768->512->128 (if output_dim=128)
    projection_hidden_layers: List[int] = None
    projection_activation: str = "relu"
    projection_dropout_rate: float = 0.0
    output_dim: int = 128  # Final embedding dimension

    # Legacy support (will be deprecated)
    hidden_sizes: List[int] = None  # For backward compatibility
    activation: str = "relu"        # For backward compatibility
    dropout_rate: float = 0.0       # For backward compatibility

    def __post_init__(self):
        # Handle legacy configuration first
        if self.hidden_sizes is not None:
            # Convert legacy format: [512, 128] -> projection_hidden_layers=[512]
            # The last element in hidden_sizes was never actually used as hidden layer
            if len(self.hidden_sizes) > 1:
                legacy_hidden_layers = self.hidden_sizes[:-1]
            else:
                legacy_hidden_layers = self.hidden_sizes

            # Only override if new format not explicitly set
            if self.projection_hidden_layers is None:
                self.projection_hidden_layers = legacy_hidden_layers
            if self.projection_activation == "relu" and self.activation != "relu":
                self.projection_activation = self.activation
            if self.projection_dropout_rate == 0.0 and self.dropout_rate != 0.0:
                self.projection_dropout_rate = self.dropout_rate

        # Set defaults if still not specified
        if self.projection_hidden_layers is None:
            self.projection_hidden_layers = [512]  # Default: 768->512->128


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 2
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 6
    triplet_margin: float = 0.3
    margin_schedule_end_epoch: int = 10  # Epoch where margin reaches max value
    margin_schedule_max: float = 0.8    # Maximum margin value
    gradient_clip_norm: float = 1.0     # Gradient clipping norm
    
    # Learning rate scheduler configuration
    use_custom_lr: bool = True  # Use custom dual-group LR scheduler
    head_lr_multiplier: float = 10.0  # LR multiplier for projection head vs transformer base
    multiplier_converge_epoch: int = 15  # Epoch after which multiplier becomes 1.0
    warmup_steps_pct: float = 0.04  # Warmup percentage of total steps (3-5% typical)
    min_lr: float = 1e-6  # Minimum learning rate floor for both groups
    use_cosine_scheduler: bool = True  # Use cosine annealing scheduler
    
    # Data loading
    num_workers: int = 2
    pin_memory: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    # Logging, Eval and safe strategy
    logging_strategy: str = "epoch"
    logging_steps: int = 50
    eval_strategy: str = "epoch"
    save_strategy: str = "no"
    save_steps: Optional[int] = None
    eval_steps: Optional[int] = None
    
    # Reproducibility
    seed: int = 42
    
    # Progress bars
    disable_tqdm: bool = True  # Disable tqdm progress bars (useful for cluster/batch jobs)
    
    # Early stopping configuration
    enable_early_stopping: bool = False  # Enable early stopping based on eval_accuracy
    early_stopping_patience: int = 5  # Number of epochs without improvement before stopping
    early_stopping_min_delta: float = 0.001  # Minimum improvement threshold to reset patience
    post_resample_grace_epochs: int = 2  # Grace period after resampling before early stopping can trigger

    # GPU configuration
    force_single_gpu: bool = False  # Force single GPU mode even with multiple GPUs available

    # Mixed precision & math mode (defaults are safe for GTX 1080/1080 Ti)
    bf16: bool = False                 # Ampere+ only; leave False on GTX 1080
    fp16: bool = False                 # Not recommended on GTX 1080; keep False
    tf32: bool = False                 # Ampere+ only; keep False on GTX 1080

    # Low-level PyTorch backend switches
    allow_tf32_matmul: bool = False    # torch.backends.cuda.matmul.allow_tf32
    allow_tf32_cudnn: bool = False     # torch.backends.cudnn.allow_tf32
    set_float32_matmul_precision: Optional[str] = None  # "high"/"medium"/"highest" or None
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.triplet_margin < 0:
            raise ValueError("Triplet margin must be non-negative")


@dataclass
class DataConfig:
    """Data processing configuration."""
    chunks_dir: str = "Precomputed_AST_7G"
    split_file_train: Optional[str] = None
    split_file_test: Optional[str] = None
    test_split_ratio: float = 0.1
    min_files_per_genre: int = 2
    
    # Triplet generation parameters for cluster training
    max_positive_tracks: int = 12  # Maximum positive tracks per anchor (balanced for clusters)
    triplets_per_positive_track: int = 2  # Triplets per positive track (prevents overfitting)
    
    resample_train_samples: bool = False   # If True, regenerate fresh triplets at the start of each epoch
    resample_fraction: float = 1.0      # Fraction of train dataset to resample (0.0-1.0)
    resample_cadence: int = 1           # Resample every N epochs (e.g., 3 = every 3rd epoch starting from epoch 3)
    resample_start_epoch: int = 1       # Epoch to start resampling (e.g., 3 = start resampling from epoch 3)
    resample_schedule_override: Optional[Dict[int, float]] = None  # Override resampling for specific epochs {epoch: fraction}
    negative_mining: str = "none"       # Choose mining strategy: "none", "semi_hard", or "hard"
    negative_mining_start_epoch: int = 1  # Epoch to start negative mining (0 = from beginning)
    
    # Stratified batching configuration  
    stratified_batching: bool = False      # Enable balanced subgenre representation per batch
    min_per_subgenre: int = 2             # Minimum samples per subgenre per batch
    
    # Quick testing configuration
    quick_test_run: bool = False      # If True, use only subset of data for quick error detection
    quick_test_fraction: float = 0.05 # Fraction of data to use for quick test runs (default 5%)
    
    # For backward compatibility with old split format
    use_legacy_splits: bool = False
    legacy_split_dir: Optional[str] = None

    # Audio augmentation configuration
    enable_augmentations: bool = False  # Global augmentation enable/disable
    augmentation_probability: float = 0.5  # Probability of applying augmentations
    use_2d_augmentations: bool = True  # Enable 2D spectrogram augmentations

    # Spectrogram augmentation parameters
    aug_time_mask_prob: float = 0.2
    aug_time_mask_max_length: int = 15
    aug_freq_mask_prob: float = 0.2
    aug_freq_mask_max_length: int = 12
    aug_noise_prob: float = 0.3
    aug_noise_std: float = 0.03
    aug_volume_prob: float = 0.3
    aug_volume_range: tuple = (0.9, 1.1)
    aug_temporal_shift_prob: float = 0.2
    aug_temporal_shift_max: int = 3

    # 2D spectrogram augmentation parameters
    aug_2d_freq_mask_prob: float = 0.3
    aug_2d_freq_mask_max: int = 12
    aug_2d_time_mask_prob: float = 0.3
    aug_2d_time_mask_max: int = 15
    aug_2d_num_freq_masks: int = 1
    aug_2d_num_time_masks: int = 1

    # K-Fold cross-validation configuration
    enable_kfold: bool = False  # Enable K-fold cross-validation
    kfold_k: int = 5  # Number of folds (default: 5)
    kfold_partitions_file: Optional[str] = None  # Pre-computed partitions file
    kfold_current_fold: Optional[int] = None  # Current fold index (0 to k-1) when running single fold


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    
    # Experiment metadata
    experiment_name: str = "ast_triplet_training"
    description: str = ""
    
    def __init__(self,
                 model: Optional[ModelConfig] = None,
                 training: Optional[TrainingConfig] = None,
                 data: Optional[DataConfig] = None,
                 **kwargs):
        """Initialize with optional component configs."""
        self.model = model or ModelConfig()
        self.training = training or TrainingConfig()
        self.data = data or DataConfig()
        
        # Set any additional attributes
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct nested dataclasses
        model = ModelConfig(**config_dict.pop('model', {}))
        training = TrainingConfig(**config_dict.pop('training', {}))
        data = DataConfig(**config_dict.pop('data', {}))
        # Remove paths field if present (backward compatibility)
        config_dict.pop('paths', None)

        return cls(model=model, training=training, data=data, **config_dict)
    
    def validate(self) -> None:
        """Validate the complete configuration."""
        # Check that required paths exist
        if not Path(self.data.chunks_dir).exists():
            raise ValueError(f"Chunks directory does not exist: {self.data.chunks_dir}")
        
        # Validate split files if specified
        if self.data.split_file_train and not Path(self.data.split_file_train).exists():
            raise ValueError(f"Train split file does not exist: {self.data.split_file_train}")
        
        if self.data.split_file_test and not Path(self.data.split_file_test).exists():
            raise ValueError(f"Test split file does not exist: {self.data.split_file_test}")
        
        # Validate stratified batching configuration
        if self.data.stratified_batching:
            if self.data.min_per_subgenre <= 0:
                raise ValueError("min_per_subgenre must be positive")
            
            # Conservative estimate: assume 8 subgenres
            estimated_min_batch_size = 8 * self.data.min_per_subgenre
            if self.training.batch_size < estimated_min_batch_size:
                raise ValueError(
                    f"Batch size ({self.training.batch_size}) too small for stratified batching. "
                    f"Need at least {estimated_min_batch_size} (8 subgenres × {self.data.min_per_subgenre} min_per_subgenre)"
                )


def create_default_config(test_run: bool = False) -> ExperimentConfig:
    """Create default configuration with optional test run settings."""
    config = ExperimentConfig()
    
    if test_run:
        # Reduce parameters for quick testing
        config.training.epochs = 2
        config.training.batch_size = 1
        config.data.max_positive_tracks = 3  # Minimal for testing
        config.data.triplets_per_positive_track = 1  # Minimal for testing
        config.experiment_name = "ast_triplet_test"
        config.description = "Quick test run with reduced parameters"
    else:
        # Cluster-optimized defaults for full training
        config.experiment_name = "ast_triplet_cluster"
        config.description = "Cluster training with optimized triplet generation"
    
    return config


def load_or_create_config(config_path: Optional[str] = None, 
                         test_run: bool = False) -> ExperimentConfig:
    """Load config from file or create default."""
    if config_path and Path(config_path).exists():
        return ExperimentConfig.load(config_path)
    else:
        return create_default_config(test_run=test_run)