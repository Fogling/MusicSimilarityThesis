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
    hidden_sizes: List[int] = None
    activation: str = "relu"
    dropout_rate: float = 0.0
    output_dim: int = 128
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [512, 128]


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
    
    # Training strategy
    eval_strategy: str = "epoch"
    save_strategy: str = "no"
    logging_steps: int = 50
    save_steps: Optional[int] = None
    eval_steps: Optional[int] = None
    
    # Reproducibility
    seed: int = 42
    
    # Progress bars
    disable_tqdm: bool = True  # Disable tqdm progress bars (useful for cluster/batch jobs)
    
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
    chunks_dir: str = "WAV"
    split_file_train: Optional[str] = None
    split_file_test: Optional[str] = None
    test_split_ratio: float = 0.1
    min_files_per_genre: int = 2
    
    # Triplet generation parameters for cluster training
    max_positive_tracks: int = 12  # Maximum positive tracks per anchor (balanced for clusters)
    triplets_per_positive_track: int = 2  # Triplets per positive track (prevents overfitting)
    
    resample_each_epoch: bool = False   # If True, regenerate fresh triplets at the start of each epoch
    negative_mining: str = "none"       # Choose mining strategy: "none", "semi_hard", or "hard"
    
    # Stratified batching configuration  
    stratified_batching: bool = False      # Enable balanced subgenre representation per batch
    min_per_subgenre: int = 2             # Minimum samples per subgenre per batch
    
    # Quick testing configuration
    quick_test_run: bool = False      # If True, use only subset of data for quick error detection
    quick_test_fraction: float = 0.05 # Fraction of data to use for quick test runs (default 5%)
    
    # For backward compatibility with old split format
    use_legacy_splits: bool = False
    legacy_split_dir: Optional[str] = None


@dataclass
class PathConfig:
    """Path configuration."""
    model_output_dir: str = "ast_triplet_output"
    log_dir: str = "logs"
    checkpoint_dir: Optional[str] = None
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [self.model_output_dir, self.log_dir]:
            if dir_path:
                Path(dir_path).mkdir(exist_ok=True)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    paths: PathConfig
    
    # Experiment metadata
    experiment_name: str = "ast_triplet_training"
    description: str = ""
    
    def __init__(self, 
                 model: Optional[ModelConfig] = None,
                 training: Optional[TrainingConfig] = None,
                 data: Optional[DataConfig] = None,
                 paths: Optional[PathConfig] = None,
                 **kwargs):
        """Initialize with optional component configs."""
        self.model = model or ModelConfig()
        self.training = training or TrainingConfig()
        self.data = data or DataConfig()
        self.paths = paths or PathConfig()
        
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
        paths = PathConfig(**config_dict.pop('paths', {}))
        
        return cls(model=model, training=training, data=data, paths=paths, **config_dict)
    
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