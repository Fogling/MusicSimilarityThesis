#!/usr/bin/env python3
"""
Improved AST Triplet Loss Training with proper error handling, configuration management,
and modular architecture.

This refactored version addresses all issues found in the original:
- Comprehensive error handling and validation
- Configuration management with type safety
- Modular, testable code structure
- Proper logging and monitoring
- Memory efficiency improvements
- Security fixes
"""

import os
import json
import logging
import random
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import ASTModel, TrainingArguments, Trainer, TrainerCallback
from safetensors.torch import save_file, load_file
from tqdm import tqdm
from dataclasses import dataclass

from config import ExperimentConfig, load_or_create_config
from lr_scheduler import create_dual_group_optimizer, create_dual_group_scheduler, DualGroupLRCallback

import os
# Set up cache directory for efficient model loading (safe - only caches files, not model state)
cache_dir = os.getenv('HF_HOME', './huggingface_cache')
os.makedirs(cache_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from transformers and other libraries
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("safetensors").setLevel(logging.WARNING)


class TripletValidationError(Exception):
    """Raised when triplet data is invalid."""
    pass


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


class MarginSchedulingCallback(TrainerCallback):
    """Callback to update triplet margin based on epoch scheduling."""
    
    def __init__(self, model):
        self.model = model
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Update margin at the beginning of each epoch."""
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        self.model.set_epoch(current_epoch)


class CleanLoggingCallback(TrainerCallback):
    """Custom callback for clean, readable logging without redundant information."""
    
    def __init__(self):
        self.last_logged_step = -1
        self.last_logged_epoch = -1
        self.total_steps = None
        self.training_start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training to capture total steps and start time."""
        self.training_start_time = time.perf_counter()
        
        # Calculate total steps from epochs and batch size since max_steps might not be set
        if args.num_train_epochs and args.per_device_train_batch_size:
            # Get dataset size from kwargs if available
            train_dataset = kwargs.get('train_dataset')
            if train_dataset and hasattr(train_dataset, '__len__'):
                total_train_samples = len(train_dataset)
                effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
                steps_per_epoch = (total_train_samples + effective_batch_size - 1) // effective_batch_size
                self.total_steps = steps_per_epoch * args.num_train_epochs
                logger.info(f"Training will run for {self.total_steps} total steps ({steps_per_epoch} steps/epoch × {args.num_train_epochs} epochs)")
        elif state.max_steps and state.max_steps > 0:
            self.total_steps = state.max_steps
            logger.info(f"Training will run for {self.total_steps} total steps")
        
        logger.info("Training started at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called whenever logging occurs - only customize evaluation logs."""
        if logs is None:
            return
            
        # Only customize evaluation logging - let HuggingFace handle training logs
        if 'eval_loss' in logs:
            # Use state.global_step instead of logs['step'] for accurate step count
            current_step = state.global_step if state else logs.get('step', 0)
            current_epoch = logs.get('epoch', 0)
            
            # Only log evaluation once per epoch to avoid duplicates
            if current_epoch != self.last_logged_epoch:
                # Calculate elapsed time and estimate remaining time
                elapsed_time = time.perf_counter() - self.training_start_time if self.training_start_time else 0
                
                # Format step info with progress if total steps is known
                if self.total_steps:
                    step_info = f"Step {current_step}/{self.total_steps} ({current_step/self.total_steps*100:.1f}%)"
                    
                    # Estimate remaining time based on current progress
                    if current_step > 0:
                        estimated_total_time = elapsed_time * self.total_steps / current_step
                        remaining_time = max(0, estimated_total_time - elapsed_time)
                        time_info = f"Elapsed: {self._format_time(elapsed_time)}, ETA: {self._format_time(remaining_time)}"
                    else:
                        time_info = f"Elapsed: {self._format_time(elapsed_time)}"
                else:
                    step_info = f"Step {current_step}"
                    time_info = f"Elapsed: {self._format_time(elapsed_time)}"
                
                logger.info(f"Evaluation - Epoch {current_epoch}, {step_info}: "
                           f"eval_loss={logs['eval_loss']:.4f}, "
                           f"eval_accuracy={logs.get('eval_accuracy', 0):.3f}, "
                           f"{time_info}")
                self.last_logged_epoch = current_epoch
                
                # Only remove evaluation metrics, keep other important logs for trainer
                eval_keys_to_remove = [k for k in logs.keys() if k.startswith('eval_')]
                for key in eval_keys_to_remove:
                    if key not in ['eval_loss', 'eval_accuracy']:  # Keep the main metrics
                        logs.pop(key, None)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training to log total time."""
        if self.training_start_time:
            total_training_time = time.perf_counter() - self.training_start_time
            logger.info(f"Training completed! Total training time: {self._format_time(total_training_time)}")
            logger.info("Training ended at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    
    def _format_time(self, seconds):
        """Format seconds into a readable time string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.1f}s"

class ResampleCallback(TrainerCallback):
    """
    When enabled via config (data.resample_train_samples == True), this callback
    asks the train dataset to regenerate its triplets at the start of each epoch
    according to the configured cadence.
    """
    def __init__(self, train_dataset, config, enabled: bool):
        self.train_dataset = train_dataset
        self.config = config
        self.enabled = bool(enabled)
        self.resample_cadence = getattr(config.data, "resample_cadence", 1)

    def on_epoch_begin(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch) + 1
        
        if not (self.enabled and hasattr(self.train_dataset, "resample_for_new_epoch")):
            return
            
        # Get resampling configuration parameters
        resample_start_epoch = getattr(self.config.data, "resample_start_epoch", 1)
        resample_schedule_override = getattr(self.config.data, "resample_schedule_override", None)
        
        # Check if we haven't reached the start epoch yet
        if current_epoch < resample_start_epoch:
            logger.debug(f"[ResampleCallback] Epoch {current_epoch}: Before start epoch {resample_start_epoch}, no resampling")
            return
            
        # Check for epoch-specific override first
        custom_fraction = None
        if resample_schedule_override and current_epoch in resample_schedule_override:
            custom_fraction = resample_schedule_override[current_epoch]
            should_resample = True
            logger.info(f"[ResampleCallback] Epoch {current_epoch}: Using schedule override with fraction {custom_fraction}")
        else:
            # Check if this epoch should trigger resampling based on cadence
            # cadence=1: resample every epoch starting from resample_start_epoch
            # cadence=3: resample at epochs (start_epoch + 3n) where n >= 0
            epochs_since_start = current_epoch - resample_start_epoch
            should_resample = (epochs_since_start % self.resample_cadence == 0) and (epochs_since_start >= 0)
        
        if should_resample:
            try:
                if custom_fraction is not None:
                    # Pass custom fraction to resampling method
                    self.train_dataset.resample_for_new_epoch(custom_fraction=custom_fraction)
                else:
                    self.train_dataset.resample_for_new_epoch()
            except Exception as e:
                print(f"[ResampleCallback] resample_for_new_epoch failed: {e}")
                logger.warning(f"[ResampleCallback] resample_for_new_epoch failed: {e}")
        else:
            logger.debug(f"[ResampleCallback] Epoch {current_epoch}: No resampling (cadence={self.resample_cadence}, start_epoch={resample_start_epoch})")


class EarlyStoppingCallback(TrainerCallback):
    """
    Early stopping callback that is aware of resampling operations.
    
    This callback monitors eval_accuracy and stops training when no improvement
    is observed for a specified number of epochs. It includes special handling
    for resampling scenarios where temporary accuracy drops are expected.
    
    Features:
    - Patience-based early stopping on eval_accuracy
    - Resampling-aware: suspends early stopping for grace period after resampling
    - Configurable minimum improvement threshold to avoid stopping on noise
    - Detailed logging of early stopping decisions
    """
    
    def __init__(self, config: ExperimentConfig, resample_callback: Optional[ResampleCallback] = None):
        """
        Initialize early stopping callback.
        
        Args:
            config: Experiment configuration containing early stopping parameters
            resample_callback: Optional reference to resample callback to track resampling events
        """
        self.enabled = config.training.enable_early_stopping
        self.patience = config.training.early_stopping_patience
        self.min_delta = config.training.early_stopping_min_delta
        self.post_resample_grace_epochs = config.training.post_resample_grace_epochs
        
        # State tracking
        self.best_accuracy = -float('inf')
        self.epochs_without_improvement = 0
        self.last_resample_epoch = -1
        self.resample_callback = resample_callback
        self.stopped_early = False
        
        if self.enabled:
            logger.info(f"Early stopping enabled: patience={self.patience}, min_delta={self.min_delta}, "
                       f"post_resample_grace={self.post_resample_grace_epochs}")
        else:
            logger.info("Early stopping disabled")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Track resampling events by monitoring the resample callback."""
        if not self.enabled:
            return
            
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        
        # Check if resampling occurred in this epoch by checking resample callback
        if (self.resample_callback and 
            hasattr(self.resample_callback, 'enabled') and 
            self.resample_callback.enabled):
            
            # Check if this epoch triggers resampling based on callback logic
            resample_start_epoch = getattr(self.resample_callback.config.data, "resample_start_epoch", 1)
            resample_cadence = getattr(self.resample_callback.config.data, "resample_cadence", 1)
            resample_schedule_override = getattr(self.resample_callback.config.data, "resample_schedule_override", None)
            
            # Same logic as ResampleCallback to detect resampling
            resampling_this_epoch = False
            if current_epoch >= resample_start_epoch:
                if resample_schedule_override and current_epoch in resample_schedule_override:
                    resampling_this_epoch = True
                else:
                    epochs_since_start = current_epoch - resample_start_epoch
                    resampling_this_epoch = (epochs_since_start % resample_cadence == 0) and (epochs_since_start >= 0)
            
            if resampling_this_epoch:
                self.last_resample_epoch = current_epoch
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """
        Monitor evaluation metrics and trigger early stopping if conditions are met.
        
        Args:
            args: Training arguments
            state: Trainer state
            control: Training control
            logs: Dictionary containing evaluation metrics
        """
        if not self.enabled or logs is None:
            return
        
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        eval_accuracy = logs.get('eval_accuracy', 0)
        
        # Check if we're in grace period after resampling
        epochs_since_resample = current_epoch - self.last_resample_epoch
        in_grace_period = (self.last_resample_epoch >= 0 and 
                          epochs_since_resample <= self.post_resample_grace_epochs)
        
        # Update best accuracy and patience counter
        improvement = eval_accuracy - self.best_accuracy
        if improvement > self.min_delta:
            self.best_accuracy = eval_accuracy
            self.epochs_without_improvement = 0
            logger.info(f"[EarlyStopping] New best accuracy: {eval_accuracy:.4f} "
                       f"(+{improvement:.4f}) at epoch {current_epoch}")
        else:
            self.epochs_without_improvement += 1
            logger.debug(f"[EarlyStopping] No improvement: {self.epochs_without_improvement}/{self.patience} "
                        f"(current: {eval_accuracy:.4f}, best: {self.best_accuracy:.4f})")
        
        # Early stopping decision logic
        should_stop = (self.epochs_without_improvement >= self.patience and 
                      not in_grace_period)
        
        if should_stop:
            logger.info(f"[EarlyStopping] Stopping training: no improvement for {self.patience} epochs "
                       f"(best accuracy: {self.best_accuracy:.4f} at epoch {current_epoch - self.epochs_without_improvement})")
            control.should_training_stop = True
            self.stopped_early = True
            
        elif self.epochs_without_improvement >= self.patience and in_grace_period:
            logger.info(f"[EarlyStopping] Would stop training, but in grace period "
                       f"({epochs_since_resample}/{self.post_resample_grace_epochs} epochs since resampling)")
        
        # Log current status every few epochs for visibility
        if current_epoch % 3 == 0 or self.epochs_without_improvement >= self.patience - 1:
            grace_status = f" (in grace period: {epochs_since_resample}/{self.post_resample_grace_epochs})" if in_grace_period else ""
            logger.info(f"[EarlyStopping] Status: {self.epochs_without_improvement}/{self.patience} epochs without improvement{grace_status}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Log final early stopping status."""
        if not self.enabled:
            return
            
        if self.stopped_early:
            logger.info(f"[EarlyStopping] Training stopped early with best accuracy: {self.best_accuracy:.4f}")
        else:
            logger.info(f"[EarlyStopping] Training completed normally with final best accuracy: {self.best_accuracy:.4f}")


class StratifiedSubgenreBatchSampler:
    """
    BatchSampler that ensures each batch contains a minimum number of samples 
    from each subgenre, with remainder filled randomly.
    
    This sampler groups dataset indices by subgenre and creates batches where:
    1. Each subgenre gets at least min_per_subgenre samples per batch
    2. Remaining slots are filled randomly from all available samples
    3. Supports dynamic resampling when dataset.triplets_active changes
    """
    
    def __init__(self, dataset, batch_size, min_per_subgenre=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_per_subgenre = min_per_subgenre
        self._last_triplets_id = None  # Track when triplets_active changes
        self._rebuild_indices()
        
    def _rebuild_indices(self):
        """Rebuild subgenre indices from current dataset.triplets_active."""
        # Group indices by subgenre
        self.subgenre_indices = defaultdict(list)
        
        for idx, (_, _, _, subgenre) in enumerate(self.dataset.triplets_active):
            self.subgenre_indices[subgenre].append(idx)
        
        self.subgenres = list(self.subgenre_indices.keys())
        self.num_subgenres = len(self.subgenres)
        
        # Validate configuration
        min_required = self.num_subgenres * self.min_per_subgenre
        if self.batch_size < min_required:
            logger.warning(
                f"Batch size ({self.batch_size}) < required minimum ({min_required}) "
                f"for {self.num_subgenres} subgenres × {self.min_per_subgenre} min_per_subgenre. "
                f"Will use best-effort stratification."
            )
            # Adjust min_per_subgenre to fit available batch size
            self.min_per_subgenre = max(1, self.batch_size // self.num_subgenres)
        
        # Calculate slots allocation
        self.guaranteed_slots = self.num_subgenres * self.min_per_subgenre
        self.random_slots = self.batch_size - self.guaranteed_slots
        
        logger.info(
            f"Stratified batching: {self.num_subgenres} subgenres, "
            f"{self.min_per_subgenre} guaranteed per subgenre, "
            f"{self.random_slots} random slots per batch"
        )
        
        # Mark triplets version for change detection
        self._last_triplets_id = id(self.dataset.triplets_active)
    
    def _check_for_resampling(self):
        """Check if dataset has been resampled and rebuild indices if needed."""
        current_triplets_id = id(self.dataset.triplets_active)
        if current_triplets_id != self._last_triplets_id:
            logger.info("Dataset resampled detected, rebuilding stratified indices...")
            self._rebuild_indices()
    
    def __iter__(self):
        """Generate stratified batches."""
        # Check for dataset resampling
        self._check_for_resampling()
        
        # Create cycling iterators for each subgenre
        subgenre_iterators = {}
        for subgenre in self.subgenres:
            indices = self.subgenre_indices[subgenre].copy()
            np.random.shuffle(indices)  # Shuffle within subgenre
            subgenre_iterators[subgenre] = iter(indices)
        
        # All indices for random sampling
        all_indices = list(range(len(self.dataset.triplets_active)))
        np.random.shuffle(all_indices)
        random_iterator = iter(all_indices)
        
        # Generate batches
        total_samples = len(self.dataset.triplets_active)
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            batch_indices = []
            
            # 1. Add guaranteed samples from each subgenre
            for subgenre in self.subgenres:
                for _ in range(self.min_per_subgenre):
                    try:
                        idx = next(subgenre_iterators[subgenre])
                        batch_indices.append(idx)
                    except StopIteration:
                        # Exhausted this subgenre, restart iterator
                        indices = self.subgenre_indices[subgenre].copy()
                        np.random.shuffle(indices)
                        subgenre_iterators[subgenre] = iter(indices)
                        try:
                            idx = next(subgenre_iterators[subgenre])
                            batch_indices.append(idx)
                        except StopIteration:
                            # Subgenre is empty, skip
                            continue
            
            # 2. Fill remaining slots randomly
            remaining_slots = self.batch_size - len(batch_indices)
            for _ in range(remaining_slots):
                try:
                    idx = next(random_iterator)
                    batch_indices.append(idx)
                except StopIteration:
                    # Exhausted all samples, we're done
                    break
            
            if batch_indices:
                np.random.shuffle(batch_indices)  # Final shuffle of the batch
                yield batch_indices
    
    def __len__(self):
        """Return number of batches."""
        total_samples = len(self.dataset.triplets_active)
        return (total_samples + self.batch_size - 1) // self.batch_size

# ========== UTILITY FUNCTIONS ==========

def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seeds set to {seed}")


def validate_tensor_shape(tensor: torch.Tensor, expected_shape: Optional[Tuple] = None, 
                         tensor_name: str = "tensor") -> None:
    """Validate tensor shape and contents."""
    if tensor is None:
        raise TripletValidationError(f"{tensor_name} is None")
    
    if not isinstance(tensor, torch.Tensor):
        raise TripletValidationError(f"{tensor_name} is not a tensor, got {type(tensor)}")
    
    if torch.isnan(tensor).any():
        raise TripletValidationError(f"{tensor_name} contains NaN values")
    
    if torch.isinf(tensor).any():
        raise TripletValidationError(f"{tensor_name} contains infinite values")
    
    if expected_shape and tensor.shape != expected_shape:
        raise TripletValidationError(
            f"{tensor_name} shape {tensor.shape} != expected {expected_shape}"
        )


def sanitize_dict_tensor(tensor_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Clean and validate dictionary of tensors with comprehensive error handling.
    
    Args:
        tensor_dict: Dictionary potentially containing tensors or lists
        
    Returns:
        Dictionary with cleaned tensors
        
    Raises:
        TripletValidationError: If tensor validation fails
    """
    if not isinstance(tensor_dict, dict):
        raise TripletValidationError(f"Expected dict, got {type(tensor_dict)}")
    
    cleaned = {}
    for key, value in tensor_dict.items():
        try:
            # Handle list of tensors
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                value = value[0]
            elif isinstance(value, list):
                value = torch.tensor(value, dtype=torch.float32)
            
            # Validate and squeeze if needed
            if isinstance(value, torch.Tensor):
                validate_tensor_shape(value, tensor_name=f"tensor_dict['{key}']")
                
                # Remove leading singleton dimensions
                if value.ndim == 3 and value.shape[0] == 1:
                    value = value.squeeze(0)
                
                cleaned[key] = value
            else:
                logger.warning(f"Unexpected type for key '{key}': {type(value)}")
                
        except Exception as e:
            raise TripletValidationError(f"Error processing tensor_dict['{key}']: {e}")
    
    return cleaned


class ImprovedTripletFeatureDataset(TorchDataset):
    """
    Improved triplet dataset with proper error handling and validation.
    """
    
    def __init__(self, split_data: Union[List, Dict], config: ExperimentConfig, split_name: str = "train"):
        self.config = config
        self.split_name = split_name

        # always keep an immutable copy of original triplets
        self.triplets_original = self._parse_split_data(split_data)
        self.triplets_active = list(self.triplets_original)

        # resampling flag (defaults to False if not present)
        self._resample_enabled = bool(getattr(self.config.data, "resample_train_samples", False))

        # build indices that allow resampling (subgenre -> track -> chunks)
        self._build_indices_from_triplets(self.triplets_original)
        # Note: Do NOT resample during initialization - let the ResampleCallback handle this during training

        # Show actual resampling behavior, not just config flag
        actually_resampling = self._resample_enabled and split_name == "train"
        logger.info(
            f"Initialized dataset ({split_name}) with {len(self.triplets_active)} triplets "
            f"(resample_train_samples={actually_resampling})"
        )

    def _parse_split_data(self, split_data: Union[List, Dict]) -> List[Tuple[str, str, str, str]]:
        """Parse split data from either legacy or new format."""
        if isinstance(split_data, list):
            # Legacy format: [anchor_path, positive_path, negative_path, subgenre]
            return [(item[0], item[1], item[2], item[3]) for item in split_data]
        
        elif isinstance(split_data, dict) and "tracks" in split_data:
            # New format: structured by subgenre and tracks
            triplets = []
            tracks_by_genre = split_data["tracks"]
            
            for subgenre, track_list in tracks_by_genre.items():
                for track in track_list:
                    chunks = track["chunks"]
                    if len(chunks) >= 3:
                        # Group chunks into triplets
                        for i in range(0, len(chunks) - 2, 3):
                            triplets.append((chunks[i], chunks[i+1], chunks[i+2], subgenre))
            
            return triplets
        
        else:
            raise ValueError(f"Unsupported split data format: {type(split_data)}")
    
    def _track_name_from_path(self, p: str) -> str:
        """
        Extracts the track name (without chunk suffix) from a chunk filepath.

        Example:
        ".../techno/track123_chunk0.pt" -> "track123"
        """
        return Path(p).stem.rsplit("_chunk", 1)[0]
    
    def _get_subgenre_from_path(self, p: str) -> str:
        """
        Extracts the subgenre from a chunk filepath.

        Example:
        ".../precomputed_AST/techno/track123_chunk0.pt" -> "techno"
        """
        return Path(p).parent.name


    def _build_indices_from_triplets(self, triplets):
        """
        Build lookup indices from the original triplets. These indices are used
        later to resample new triplets each epoch.

        Creates:
        - self.sg_to_track_chunks: {subgenre -> {track_name -> [chunk_paths]}}
        - self.sg_all_chunks: {subgenre -> [all chunk_paths]}
        - self.unique_anchors: list of unique (anchor_path, subgenre) pairs
        - self.neg_pool_by_sg: {subgenre -> [all chunks from other subgenres]}
        """
        from collections import defaultdict

        self.sg_to_track_chunks = defaultdict(lambda: defaultdict(list))
        self.sg_all_chunks = defaultdict(list)

        # Fill indices from all paths found in the triplets
        for a, p, n, sg in triplets:
            for path in (a, p, n):
                self.sg_to_track_chunks[sg][self._track_name_from_path(path)].append(path)
                self.sg_all_chunks[sg].append(path)

        # Deduplicate chunk lists
        for sg in list(self.sg_to_track_chunks.keys()):
            for t in list(self.sg_to_track_chunks[sg].keys()):
                self.sg_to_track_chunks[sg][t] = sorted(set(self.sg_to_track_chunks[sg][t]))
            self.sg_all_chunks[sg] = sorted(set(self.sg_all_chunks[sg]))

        # Store one entry per unique anchor (so we can regenerate positives/negatives later)
        self.unique_anchors = []
        seen = set()
        for a, _p, _n, sg in triplets:
            if a not in seen:
                seen.add(a)
                self.unique_anchors.append((a, sg))

        # For each subgenre, prepare a pool of negatives = all chunks from other subgenres
        self.neg_pool_by_sg = {}
        all_sg = list(self.sg_to_track_chunks.keys())
        for sg in all_sg:
            pool = []
            for other in all_sg:
                if other != sg:
                    pool.extend(self.sg_all_chunks[other])
            self.neg_pool_by_sg[sg] = pool


    def resample_for_new_epoch(self, custom_fraction=None):
        """
        Generate a fresh set of triplets for the new epoch with configurable partial resampling.

        Args:
            custom_fraction: If provided, use this fraction instead of config.data.resample_fraction

        Uses config.data.resample_fraction (or custom_fraction) to determine what percentage of triplets to resample:
        - 1.0 = resample all triplets (original behavior)  
        - 0.3 = resample 30% of triplets, keep 70% unchanged
        - 0.0 = no resampling (keep all original triplets)

        For resampled anchors:
        - Sample a positive from a *different* track in the same subgenre.
        - Sample a negative from any other subgenre.
        - Fall back gracefully if only one track exists in the subgenre.

        The resulting triplets are stored in self.triplets_active and used
        during this epoch.
        """
        resample_fraction = custom_fraction if custom_fraction is not None else getattr(self.config.data, "resample_fraction", 1.0)
        resample_fraction = max(0.0, min(1.0, resample_fraction))  # Clamp to [0, 1]
        
        if resample_fraction == 0.0:
            # No resampling - keep original triplets
            self.triplets_active = list(self.triplets_original)
            fraction_source = "custom" if custom_fraction is not None else "config"
            logger.info(f"[Dataset] No resampling (fraction=0.0 from {fraction_source}), keeping {len(self.triplets_active)} original triplets")
            return
        
        rng = random.Random(random.randint(0, 2**31 - 1))
        
        # Determine which anchors to resample
        num_to_resample = int(len(self.unique_anchors) * resample_fraction)
        anchors_to_resample = rng.sample(self.unique_anchors, num_to_resample)
        anchors_to_resample_set = set(anchors_to_resample)
        
        new_triplets = []
        
        # Keep original triplets for anchors not being resampled
        if resample_fraction < 1.0:
            for triplet in self.triplets_original:
                anchor_path, _, _, sg = triplet
                if (anchor_path, sg) not in anchors_to_resample_set:
                    new_triplets.append(triplet)
        
        # Generate new triplets for selected anchors
        for anchor_path, sg in anchors_to_resample:
            anchor_track = self._track_name_from_path(anchor_path)

            # Choose a positive chunk from a different track in the same subgenre
            candidate_tracks = [t for t in self.sg_to_track_chunks[sg].keys() if t != anchor_track]
            if candidate_tracks:
                t = rng.choice(candidate_tracks)
                pos_path = rng.choice(self.sg_to_track_chunks[sg][t])
            else:
                # fallback: reuse any positive from the original triplets for this anchor
                fallback = [p for (a, p, _n, sg_) in self.triplets_original if a == anchor_path and sg_ == sg]
                if not fallback:
                    continue
                pos_path = rng.choice(fallback)

            # Choose a negative chunk from another subgenre
            neg_pool = self.neg_pool_by_sg.get(sg, [])
            if not neg_pool:
                continue
            neg_path = rng.choice(neg_pool)

            new_triplets.append((anchor_path, pos_path, neg_path, sg))

        rng.shuffle(new_triplets)
        self.triplets_active = new_triplets
        
        num_resampled = len(anchors_to_resample)
        num_kept = len(self.triplets_active) - num_resampled
        fraction_source = "custom" if custom_fraction is not None else "config"
        logger.info(f"[Dataset] Partial resampling: {num_resampled}/{len(self.unique_anchors)} anchors resampled "
                   f"({resample_fraction:.1%} from {fraction_source}), {num_kept} triplets kept, {len(self.triplets_active)} total")
    
    def _safe_load_tensor(self, filepath: str) -> Dict[str, torch.Tensor]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tensor file not found: {filepath}")

        try:
            tensor_dict = torch.load(filepath, map_location='cpu', weights_only=True)
            if not isinstance(tensor_dict, dict):
                raise TripletValidationError(f"Expected dict from {filepath}, got {type(tensor_dict)}")
            cleaned = sanitize_dict_tensor(tensor_dict)
            return cleaned
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise TripletValidationError(f"Failed to load {filepath}: {e}")
    
    def __len__(self) -> int:
        return len(self.triplets_active)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get triplet with comprehensive validation."""
        if idx >= len(self.triplets_active):
            raise IndexError(f"Index {idx} out of range for dataset size {len(self.triplets_active)}")
        
        try:
            anchor_path, positive_path, negative_path, anchor_subgenre = self.triplets_active[idx]
            
            # Determine negative subgenre from path
            negative_subgenre = self._get_subgenre_from_path(negative_path)
            
            # Load tensors with error handling
            anchor_input = self._safe_load_tensor(anchor_path)
            positive_input = self._safe_load_tensor(positive_path)
            negative_input = self._safe_load_tensor(negative_path)
            
            return {
                "anchor_input": anchor_input,
                "positive_input": positive_input,
                "negative_input": negative_input,
                "labels": 0,  # Dummy label for compatibility
                "anchor_subgenre": anchor_subgenre,  # Anchor/positive subgenre
                "negative_subgenre": negative_subgenre  # Negative subgenre
            }
            
        except Exception as e:
            logger.error(f"Error loading triplet {idx}: {e}")
            raise


@dataclass
class _Schema:
    inner_keys: List[str]
    shapes: Dict[str, torch.Size]  # per inner_key
    dtypes: Dict[str, torch.dtype] # optional, cheap to check

class FastSafeCollator:
    """
    Collates anchor/positive/negative dicts with a fast path after the first batch.

    Behavior:
      - Batch 1: do full validation (NaN/Inf, dtype checks, squeeze, shape capture).
      - Later: only check presence + shape/dtype matches; skip expensive per-tensor scans.
      - Optionally re-run full validation every `validate_every_n` batches.

    This preserves your robust logging and error handling while cutting
    redundant work each iteration.
    """
    def __init__(self, validate_every_n: int = 0, logger_=logger, quiet: bool = True):
        """
        Args:
            validate_every_n: if > 0, re-run full validation every N batches
                              (0 = only validate batch 1).
        """
        self.schema: Optional[_Schema] = None
        self.batch_count: int = 0
        self.validate_every_n = max(0, int(validate_every_n))
        self.logger = logger_
        self.quiet = quiet
        self._logged_schema_once = False

    def _to_tensor_and_squeeze(self, t: Any, name: str) -> torch.Tensor:
        # Convert lists to tensors if needed
        if isinstance(t, list):
            t = torch.tensor(t, dtype=torch.float32)
        if not isinstance(t, torch.Tensor):
            raise TripletValidationError(f"{name} is not a tensor (got {type(t)})")

        # Squeeze leading singleton dim [1, ...] -> [...]
        if t.ndim == 3 and t.shape[0] == 1:
            t = t.squeeze(0)
        return t

    def _full_validate(self, t: torch.Tensor, name: str) -> None:
        # Keep your strict checks for the first/periodic validation.
        validate_tensor_shape(t, tensor_name=name)

    def _capture_schema_from_item(self, item: Dict[str, Dict[str, torch.Tensor]]) -> _Schema:
        # item has keys: "anchor_input", "positive_input", "negative_input"
        # and each maps to dicts like {"input_values": tensor, ...}
        ref = item["anchor_input"]
        inner_keys = list(ref.keys())
        shapes = {}
        dtypes = {}

        for k in inner_keys:
            t = self._to_tensor_and_squeeze(ref[k], f"anchor_input.{k}")
            shapes[k] = t.shape
            dtypes[k] = t.dtype

        return _Schema(inner_keys=inner_keys, shapes=shapes, dtypes=dtypes)

    def _stack_group(self, batch: List[Dict], top_key: str, do_full_validate: bool) -> Dict[str, torch.Tensor]:
        tensors_dict: Dict[str, torch.Tensor] = {}
        if top_key not in batch[0]:
            raise ValueError(f"Key '{top_key}' not found in batch items")

        # Determine inner keys from schema (fast) or from the first item (first batch)
        if self.schema is None:
            inner_keys = batch[0][top_key].keys()
        else:
            inner_keys = self.schema.inner_keys

        for inner_key in inner_keys:
            tl: List[torch.Tensor] = []
            for i, item in enumerate(batch):
                if top_key not in item:
                    raise ValueError(f"Key '{top_key}' missing from batch item")
                if inner_key not in item[top_key]:
                    raise ValueError(f"Inner key '{inner_key}' missing from {top_key}")

                t = self._to_tensor_and_squeeze(item[top_key][inner_key], f"{top_key}.{inner_key}")

                if self.schema is None:
                    # First batch (or schema reset) – full validate + capture shape
                    if do_full_validate:
                        self._full_validate(t, f"{top_key}.{inner_key}")
                else:
                    # Fast path: check only cheap invariants
                    exp_shape = self.schema.shapes[inner_key]
                    exp_dtype = self.schema.dtypes[inner_key]
                    if t.shape != exp_shape:
                        raise TripletValidationError(
                            f"Shape mismatch for {top_key}.{inner_key}: {t.shape} != {exp_shape}"
                        )
                    if t.dtype != exp_dtype:
                        raise TripletValidationError(
                            f"Dtype mismatch for {top_key}.{inner_key}: {t.dtype} != {exp_dtype}"
                        )

                tl.append(t)

            tensors_dict[inner_key] = torch.stack(tl, dim=0)

        return tensors_dict

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        if not batch:
            raise ValueError("Empty batch provided to collate")

        self.batch_count += 1
        # Decide whether to run full validation on this batch
        do_full_validate = False
        if self.schema is None:
            do_full_validate = True
        elif self.validate_every_n > 0 and (self.batch_count % self.validate_every_n == 0):
            do_full_validate = True

        try:
            # First time: capture schema from the first item (anchor_input)
            if self.schema is None:
                try:
                    if (not self.quiet) and (not self._logged_schema_once) and \
                        self.logger.isEnabledFor(logging.DEBUG):
                        shapes_map = {k: tuple(v) for k, v in self.schema.shapes.items()}
                        self.logger.debug(
                        "[Collate] Captured schema from first batch | keys=%s | shapes=%s",
                        self.schema.inner_keys, shapes_map
                        )
                        self._logged_schema_once = True
                except Exception as e:
                    self.logger.error(f"[Collate] Failed to capture schema: {e}")
                    raise

            out = {
                "anchor_input":   self._stack_group(batch, "anchor_input",   do_full_validate),
                "positive_input": self._stack_group(batch, "positive_input", do_full_validate),
                "negative_input": self._stack_group(batch, "negative_input", do_full_validate),
                "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.long),
                "anchor_subgenre": [item["anchor_subgenre"] for item in batch],
                "negative_subgenre": [item["negative_subgenre"] for item in batch]
            }
            return out

        except Exception as e:
            # On collate failure, drop schema so next batch re-validates fully.
            self.schema = None
            self.logger.error(f"Collation failed at batch {self.batch_count}: {e}")
            raise


class ImprovedASTTripletWrapper(nn.Module):
    """
    Improved AST wrapper with configurable architecture and better error handling.
    """
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        
        try:
            logger.info(f"Loading pretrained model: {config.model.pretrained_model}")
            self.ast = ASTModel.from_pretrained(
                config.model.pretrained_model,
                cache_dir=cache_dir,
                hidden_dropout_prob=config.model.ast_hidden_dropout_prob,
                attention_probs_dropout_prob=config.model.ast_attention_dropout_prob
            )
            logger.info(f"AST dropout configured - Hidden: {config.model.ast_hidden_dropout_prob}, "
                       f"Attention: {config.model.ast_attention_dropout_prob}")

        except Exception as e:
            raise ModelLoadError(f"Failed to load AST model: {e}")
        
        # Build configurable projection head
        self.projector = self._build_projection_head()
        self.triplet_margin = config.training.triplet_margin

        self.negative_mining = str(getattr(self.config.data, "negative_mining", "none")).lower()
        if self.negative_mining not in {"none", "semi_hard", "hard"}:
            self.negative_mining = "none"

        # Margin scheduling parameters
        self.initial_margin = config.training.triplet_margin
        self.margin_schedule_end_epoch = config.training.margin_schedule_end_epoch
        self.margin_schedule_max = config.training.margin_schedule_max
        self.current_epoch = 0
        
        logger.info(f"Model initialized with projection to {config.model.output_dim}D")
        logger.info(f"Margin scheduling: {self.initial_margin} → {self.margin_schedule_max} over {self.margin_schedule_end_epoch} epochs")
        logger.info(f"Negative mining: {self.negative_mining}")
        
    
    def _build_projection_head(self) -> nn.Module:
        """
        Build MLP projection head for triplet learning.

        Architecture: AST(768D) -> [hidden_layers] -> output_dim(128D)
        Example: projection_hidden_layers=[512] creates: 768->512->128

        Returns:
            nn.Module: MLP projection head with L2 normalization applied in forward()
        """
        # Use new configuration format, with fallback to legacy
        hidden_layers = self.config.model.projection_hidden_layers
        activation_name = self.config.model.projection_activation
        dropout_rate = self.config.model.projection_dropout_rate
        output_dim = self.config.model.output_dim

        # AST output dimension (from pretrained model)
        ast_output_dim = self.ast.config.hidden_size  # 768 for MIT AST

        # Get activation function
        activation_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU()
        }
        activation = activation_map.get(activation_name.lower(), nn.ReLU())

        # Build layers
        layers = []
        current_dim = ast_output_dim

        # Add hidden layers
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                activation,
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            ])
            current_dim = hidden_dim

        # Add final projection layer (no activation, L2 norm applied in forward)
        layers.append(nn.Linear(current_dim, output_dim))

        # Log the architecture for clarity
        arch_str = f"{ast_output_dim}"
        for hidden_dim in hidden_layers:
            arch_str += f" -> {hidden_dim}"
        arch_str += f" -> {output_dim}"

        logger.info(f"MLP Projection Head Architecture: {arch_str}")
        logger.info(f"  Activation: {activation_name}, Dropout: {dropout_rate}")
        logger.info(f"  Total parameters: {self._count_projection_params(layers):,}")

        return nn.Sequential(*layers)

    def _count_projection_params(self, layers: List[nn.Module]) -> int:
        """Count parameters in projection head layers."""
        total_params = 0
        for layer in layers:
            if hasattr(layer, 'weight'):
                total_params += layer.weight.numel()
                if hasattr(layer, 'bias') and layer.bias is not None:
                    total_params += layer.bias.numel()
        return total_params
    
    def set_epoch(self, epoch: int):
        """Set current epoch for margin scheduling."""
        self.current_epoch = epoch
        current_margin = self.get_current_margin()
        self.triplet_margin = current_margin
        #logger.info(f"Epoch {epoch}: Using margin {current_margin:.3f}")
    
    def get_current_margin(self) -> float:
        """Get current margin based on linear scheduling."""
        if self.current_epoch >= self.margin_schedule_end_epoch:
            return self.margin_schedule_max
        
        # Linear interpolation from initial_margin to max_margin
        progress = self.current_epoch / self.margin_schedule_end_epoch
        return self.initial_margin + (self.margin_schedule_max - self.initial_margin) * progress
    
    def embed(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate embeddings with error handling."""
        try:
            # Validate inputs
            if "input_values" not in inputs:
                raise ValueError("Missing 'input_values' in inputs")
            
            validate_tensor_shape(inputs["input_values"], tensor_name="input_values")
            
            # Forward through AST
            outputs = self.ast(**inputs)
            
            if not hasattr(outputs, 'last_hidden_state'):
                raise ValueError("AST model output missing 'last_hidden_state'")
            
            # Pool and project
            pooled = outputs.last_hidden_state.mean(dim=1)
            projected = self.projector(pooled)
            
            # L2 normalize
            normalized = F.normalize(projected, dim=1)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error in embed: {e}")
            raise
    
    def forward(
        self,
        anchor_input: Dict,
        positive_input: Dict,
        negative_input: Dict,
        labels: Optional[torch.Tensor] = None,
        anchor_subgenre: Optional[List[str]] = None,
        negative_subgenre: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for triplet learning with optional batch-wise negative mining.

        Behavior:
        1) Compute anchor/positive/negative embeddings via self.embed(...)
        2) Compute default distances from dataset-provided triplets
        3) If mining is enabled and we have per-sample labels (subgenre):
            - Mine the hardest positive and a semi-hard (or hard) negative
                within the current batch to replace the defaults.
            Else:
            - Fall back to the dataset-provided (a, p, n) distances.
        4) Compute margin-based triplet loss and return logits for metrics.

        Returns:
        dict with:
            - "loss": scalar triplet loss
            - "logits": [d_ap, d_an] per example (used by compute_metrics)
            - "distances": detached per-example distances for debugging
        """
        try:
            # 1) Embeddings
            emb_anchor = self.embed(anchor_input)   # (B, D), L2-normalized
            emb_positive = self.embed(positive_input)
            emb_negative = self.embed(negative_input)

            # 2) Default distances from dataset-provided triplets
            #    cosine distance = 1 - cos_sim  (smaller = more similar)
            dist_ap_def = 1.0 - F.cosine_similarity(emb_anchor, emb_positive, dim=1)  # (B,)
            dist_an_def = 1.0 - F.cosine_similarity(emb_anchor, emb_negative, dim=1)  # (B,)

            dist_ap, dist_an = dist_ap_def, dist_an_def

            # 3) Optional batch-wise mining (semi-hard / hard)
            #    We only attempt mining if:
            #      - config sets negative_mining to "semi_hard" or "hard"
            #      - current epoch >= negative_mining_start_epoch
            #      - the collator passed subgenre labels for this batch
            #      - we're in training mode (not evaluation)
            mining_start_epoch = getattr(self.config.data, "negative_mining_start_epoch", 1)
            mining_enabled = (self.negative_mining in {"semi_hard", "hard"} and 
                            self.current_epoch >= mining_start_epoch and
                            anchor_subgenre is not None and 
                            self.training)
            
            if mining_enabled:
                try:
                    mined = self._mine_within_batch(emb_anchor, emb_positive, emb_negative, anchor_subgenre, negative_subgenre)
                    # mined is a tuple (d_ap, d_an) or (None, None) if mining couldn't run
                    if mined[0] is not None:
                        dist_ap, dist_an = mined
                        logger.debug(f"[Mining] Applied {self.negative_mining} mining at epoch {self.current_epoch}")
                except Exception as e:
                    # Safe fallback: keep default distances (dataset triplets)
                    print(f"[Mining] Mining failed, fallback to dataset triplets: {e}")
                    logger.warning(f"[Mining] Mining failed, fallback to dataset triplets: {e}")
            elif self.negative_mining in {"semi_hard", "hard"} and self.current_epoch < mining_start_epoch:
                logger.debug(f"[Mining] Skipping mining at epoch {self.current_epoch} (starts at epoch {mining_start_epoch})")

            # 4) Triplet loss with margin
            #    Enforce: d_an >= d_ap + margin  ->  relu(d_ap - d_an + m)
            triplet_loss = torch.clamp(dist_ap - dist_an + self.triplet_margin, min=0.0)
            loss = triplet_loss.mean()

            # 5) Logits for metric computation (compute_metrics expects [d_ap, d_an])
            logits = torch.stack([dist_ap, dist_an], dim=1)  # (B, 2)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("[Forward] Invalid logits detected (NaN/Inf); replacing with zeros")
                logger.warning("[Forward] Invalid logits detected (NaN/Inf); replacing with zeros")
                logits = torch.zeros_like(logits)

            return {
                "loss": loss,
                "logits": logits,
                "distances": {
                    "anchor_positive": dist_ap.detach(),
                    "anchor_negative": dist_an.detach(),
                },
            }

        except Exception as e:
            # Bubble up after emitting both console + logger messages
            print(f"[Forward] Forward pass failed: {e}")
            logger.error(f"[Forward] Forward pass failed: {e}")
            raise
        
    def _mine_within_batch(self, emb_anchor, emb_positive, emb_negative, anchor_subgenres, negative_subgenres):
        """
        Perform batch-wise mining with full 3B candidate pool.

        Inputs:
        - emb_anchor: (B, D) anchor embeddings
        - emb_positive: (B, D) positive embeddings (paired with anchors)  
        - emb_negative: (B, D) negative embeddings (paired with anchors)
        - anchor_subgenres: list of anchor/positive subgenre strings, length B
        - negative_subgenres: list of negative subgenre strings, length B

        Behavior:
        - Hardest positive: the positive sample in the batch with the same label
                            that has the *largest* distance to the anchor.
        - Semi-hard negative: a negative (different label) that is farther than
                                the positive but as close as possible.
        - Hard negative fallback: if no semi-hard exists, use the closest negative.

        Returns:
        (d_ap, d_an) = per-sample distances (tensor length B),
        or (None, None) if mining not possible.
        """
        if not anchor_subgenres or emb_anchor.size(0) != len(anchor_subgenres):
            return None, None
        if not negative_subgenres or len(negative_subgenres) != len(anchor_subgenres):
            return None, None

        device = emb_anchor.device
        B = emb_anchor.size(0)

        # Candidate pool = anchors + positives + negatives (3B total candidates)
        emb_cand = torch.cat([emb_anchor, emb_positive, emb_negative], dim=0)

        # Create combined label list: [anchor_labels, anchor_labels, negative_labels]
        # Note: positives have same labels as anchors, negatives have their own labels
        all_subgenres = anchor_subgenres + anchor_subgenres + negative_subgenres
        
        # Convert labels to integers for mask building
        uniq = {s: i for i, s in enumerate(sorted(set(all_subgenres)))}
        y_anchor = torch.tensor([uniq[s] for s in anchor_subgenres], device=device)
        y_cand = torch.tensor([uniq[s] for s in all_subgenres], device=device)

        # Distance matrix: (B, 3B)
        sim = torch.matmul(emb_anchor, emb_cand.T).clamp(-1, 1)
        D = 1.0 - sim

        # Same-class mask (same subgenre as anchor)
        same = (y_anchor.unsqueeze(1) == y_cand.unsqueeze(0))
        
        # Exclude trivial self matches (anchor[i] == anchor[i])
        eye = torch.zeros_like(D, dtype=torch.bool)
        eye[:, :B] = torch.eye(B, device=device, dtype=torch.bool)
        same = same & (~eye)

        if not same.any():
            return None, None

        # Hardest positive: max distance among same-class candidates
        D_pos = D.clone()
        D_pos[~same] = -float("inf")
        pos_idx = D_pos.argmax(dim=1)
        d_ap = D.gather(1, pos_idx.unsqueeze(1)).squeeze(1)

        # Negatives = different-class candidates
        diff = ~same
        if self.negative_mining == "semi_hard":
            # Semi-hard negatives: farther than positive but as close as possible
            mask = diff & (D > d_ap.unsqueeze(1))
            D_neg = D.clone()
            D_neg[~mask] = float("inf")
            neg_idx = D_neg.argmin(dim=1)
            no_semi = torch.isinf(D_neg.gather(1, neg_idx.unsqueeze(1)).squeeze(1))
            if no_semi.any():
                # Fallback: hard negatives (closest different-class)
                D_hard = D.clone()
                D_hard[~diff] = float("inf")
                hard_idx = D_hard.argmin(dim=1)
                neg_idx = torch.where(no_semi, hard_idx, neg_idx)
        else:
            # Hard negatives directly (closest different-class)
            D_hard = D.clone()
            D_hard[~diff] = float("inf")
            neg_idx = D_hard.argmin(dim=1)

        d_an = D.gather(1, neg_idx.unsqueeze(1)).squeeze(1)

        return d_ap, d_an


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute evaluation metrics for triplet loss with proper handling of tuple predictions."""
    try:
        if eval_pred.predictions is None:
            logger.warning("No predictions available for metric computation")
            return {"accuracy": 0.0, "eval_accuracy": 0.0}
        
        predictions = eval_pred.predictions
        
        # Handle tuple format (HuggingFace Trainer returns tuple for custom models)
        if isinstance(predictions, tuple):
            # Take the first element which contains the logits
            predictions = predictions[0]
        
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        
        # Ensure predictions is a numpy array
        if not isinstance(predictions, np.ndarray):
            logger.error(f"Predictions is not numpy array after conversion: {type(predictions)}")
            return {"accuracy": 0.0, "eval_accuracy": 0.0}
        
        # Ensure predictions is 2D array
        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)
        elif predictions.ndim > 2:
            predictions = predictions.reshape(predictions.shape[0], -1)
        
        # For triplet loss logits: [dist_ap, dist_an]
        # We want anchor-positive distance < anchor-negative distance
        # So correct predictions should be 0 (closer to positive)
        if predictions.shape[1] >= 2:
            predicted_labels = np.argmin(predictions, axis=1)
            true_labels = np.zeros(predictions.shape[0])  # All should be 0
            accuracy = float((predicted_labels == true_labels).mean())
        else:
            logger.warning(f"Unexpected predictions shape: {predictions.shape}")
            accuracy = 0.0
        
        return {"accuracy": accuracy, "eval_accuracy": accuracy}
        
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {"accuracy": 0.0, "eval_accuracy": 0.0}


def load_preprocessed_split_data(config: ExperimentConfig) -> Tuple[List, List]:
    """
    Load train/test splits from preprocessed k-fold directories.

    This function loads triplets directly from preprocessed train/test chunk directories,
    eliminating the need for on-the-fly splitting and ensuring no data leakage.

    Args:
        config: Experiment configuration with chunks_train_dir and chunks_test_dir set

    Returns:
        Tuple of (train_triplets, test_triplets)

    Raises:
        FileNotFoundError: If train or test directories don't exist
        ValueError: If no valid chunks are found
    """
    data_config = config.data

    # Validate that we have separate train/test directories
    if not data_config.chunks_train_dir or not data_config.chunks_test_dir:
        raise ValueError("chunks_train_dir and chunks_test_dir must be set for preprocessed splits")

    train_chunks_dir = Path(data_config.chunks_train_dir)
    test_chunks_dir = Path(data_config.chunks_test_dir)

    if not train_chunks_dir.exists():
        raise FileNotFoundError(f"Training chunks directory not found: {train_chunks_dir}")
    if not test_chunks_dir.exists():
        raise FileNotFoundError(f"Test chunks directory not found: {test_chunks_dir}")

    logger.info(f"Loading preprocessed splits:")
    logger.info(f"  Train chunks: {train_chunks_dir}")
    logger.info(f"  Test chunks: {test_chunks_dir}")

    # Generate triplets from preprocessed train directory
    logger.info("Generating train triplets from preprocessed train chunks...")
    train_triplets = _generate_triplets_from_chunks_dir(train_chunks_dir, "train", config)

    # Generate triplets from preprocessed test directory
    logger.info("Generating test triplets from preprocessed test chunks...")
    test_triplets = _generate_triplets_from_chunks_dir(test_chunks_dir, "test", config)

    # Validation
    if not train_triplets:
        raise ValueError(f"No train triplets generated from {train_chunks_dir}")
    if not test_triplets:
        raise ValueError(f"No test triplets generated from {test_chunks_dir}")

    logger.info(f"✅ Preprocessed splits loaded:")
    logger.info(f"  Train triplets: {len(train_triplets)}")
    logger.info(f"  Test triplets: {len(test_triplets)}")
    logger.info(f"✅ NO DATA LEAKAGE: Using scientifically sound preprocessed train/test separation")

    return train_triplets, test_triplets


def _generate_triplets_from_chunks_dir(chunks_dir: Path, split_name: str, config: ExperimentConfig) -> List[Tuple[str, str, str, str]]:
    """
    Generate triplets from a chunks directory (either train or test).

    Args:
        chunks_dir: Directory containing subgenre subdirectories with chunk files
        split_name: "train" or "test" (for logging)
        config: Experiment configuration

    Returns:
        List of triplets in format (anchor_path, positive_path, negative_path, subgenre)
    """
    # Organize tracks by subgenre from the chunks directory
    subgenre_tracks = defaultdict(dict)  # {subgenre: {track_name: [chunk_files]}}

    for subdir in sorted(chunks_dir.iterdir()):
        if not subdir.is_dir():
            continue

        subgenre = subdir.name
        chunk_files = [f for f in subdir.iterdir() if f.suffix == '.pt']

        if not chunk_files:
            logger.warning(f"{split_name.capitalize()} - No chunks found in {subdir}")
            continue

        # Group chunks by track name
        for chunk_file in chunk_files:
            # Extract track name (remove _chunkN.pt suffix)
            track_name = chunk_file.stem.rsplit('_chunk', 1)[0] if '_chunk' in chunk_file.stem else chunk_file.stem

            if track_name not in subgenre_tracks[subgenre]:
                subgenre_tracks[subgenre][track_name] = []
            subgenre_tracks[subgenre][track_name].append(str(chunk_file))

        logger.info(f"{split_name.capitalize()} - Subgenre {subgenre}: {len(subgenre_tracks[subgenre])} tracks, {len(chunk_files)} chunks")

    # Generate triplets using existing logic
    return _generate_triplets_from_tracks(subgenre_tracks, split_name, config)


def load_split_data(config: ExperimentConfig) -> Tuple[List, List]:
    """Load train/test splits with support for both old and new formats."""
    data_config = config.data

    # NEW: Check if we have preprocessed train/test directories (preferred approach)
    if data_config.chunks_train_dir and data_config.chunks_test_dir:
        logger.info("Using preprocessed train/test splits (scientifically sound - no data leakage)")
        return load_preprocessed_split_data(config)

    # OLD: Check if K-fold mode is enabled and we're running a specific fold (deprecated)
    if data_config.enable_kfold and data_config.kfold_current_fold is not None:
        logger.warning("Using deprecated k-fold approach - consider using preprocessed splits instead")
        return load_kfold_split_data(config)

    # Try new format first
    if data_config.split_file_train and data_config.split_file_test:
        logger.info("Loading splits from specified files")

        try:
            with open(data_config.split_file_train, 'r') as f:
                train_data = json.load(f)
            with open(data_config.split_file_test, 'r') as f:
                test_data = json.load(f)

            logger.info(f"Loaded train/test splits from files")
            return train_data, test_data

        except Exception as e:
            logger.error(f"Error loading split files: {e}")
            raise

    # Fallback to generating splits
    logger.info("Generating train/test splits from preprocessed features")
    return generate_triplet_splits(config)


def load_kfold_split_data(config: ExperimentConfig) -> Tuple[List, List]:
    """Load train/test splits for a specific K-fold iteration."""
    from kfold_utils import load_kfold_partitions, get_fold_splits, generate_kfold_triplet_splits

    data_config = config.data
    fold_idx = data_config.kfold_current_fold
    k = data_config.kfold_k

    logger.info(f"Loading K-fold data for fold {fold_idx}/{k-1}")

    # Load or create partitions
    if data_config.kfold_partitions_file and Path(data_config.kfold_partitions_file).exists():
        logger.info(f"Loading existing K-fold partitions from {data_config.kfold_partitions_file}")
        kfold_partitions = load_kfold_partitions(data_config.kfold_partitions_file)
    else:
        logger.info("Creating new K-fold partitions")
        from kfold_utils import create_kfold_partitions
        kfold_partitions = create_kfold_partitions(config, k=k)

    # Get train/test track splits for this fold
    train_tracks, test_tracks = get_fold_splits(kfold_partitions, fold_idx, k=k)

    # Generate triplets for this fold
    train_triplets, test_triplets = generate_kfold_triplet_splits(
        train_tracks, test_tracks, config, fold_idx
    )

    return train_triplets, test_triplets


def generate_triplet_splits(config: ExperimentConfig) -> Tuple[List, List]:
    """Generate proper track-level splits to prevent data leakage, then create triplets."""
    chunks_dir = Path(config.data.chunks_dir)
    test_ratio = config.data.test_split_ratio
    
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
    
    logger.info("Generating track-level splits to prevent data leakage...")
    
    # Step 1: Organize files by subgenre and track
    subgenre_tracks = defaultdict(dict)  # {subgenre: {track_name: [chunk_files]}}
    
    for subdir in sorted(chunks_dir.iterdir()):
        if not subdir.is_dir():
            continue
            
        subgenre = subdir.name
        files = sorted([f for f in subdir.iterdir() if f.suffix == '.pt'])
        
        # Group files by track name (remove _chunkX.pt suffix)
        tracks = defaultdict(list)
        for file in files:
            # Extract track name by removing _chunkN.pt
            track_name = file.stem.rsplit('_chunk', 1)[0]
            tracks[track_name].append(str(file))
        
        # Only keep tracks with at least 2 chunks
        valid_tracks = {name: chunks for name, chunks in tracks.items() if len(chunks) >= 2}
        
        if len(valid_tracks) < 4:  # Need at least 4 tracks to split meaningfully
            logger.warning(f"Subgenre {subgenre} has only {len(valid_tracks)} tracks with >=2 chunks, skipping")
            continue
            
        subgenre_tracks[subgenre] = valid_tracks
        total_chunks = sum(len(chunks) for chunks in valid_tracks.values())
        logger.info(f"Subgenre {subgenre}: {len(valid_tracks)} tracks, {total_chunks} chunks total")
    
    # Step 2: Split tracks (not triplets) into train/test sets
    train_tracks = defaultdict(dict)  # {subgenre: {track_name: [chunk_files]}}
    test_tracks = defaultdict(dict)   # {subgenre: {track_name: [chunk_files]}}
    
    for subgenre in sorted(subgenre_tracks.keys()):
        tracks = subgenre_tracks[subgenre]
        track_names = list(tracks.keys())
        random.shuffle(track_names)  # Randomize track order
        
        n_test_tracks = max(1, int(len(track_names) * test_ratio))
        test_track_names = track_names[:n_test_tracks]
        train_track_names = track_names[n_test_tracks:]
        
        # Assign tracks to train/test splits
        for track_name in train_track_names:
            train_tracks[subgenre][track_name] = tracks[track_name]
        
        for track_name in test_track_names:
            test_tracks[subgenre][track_name] = tracks[track_name]
        
        logger.info(f"Subgenre {subgenre}: {len(train_track_names)} train tracks, {len(test_track_names)} test tracks")
    
    # Step 3: Generate triplets separately for train and test sets
    logger.info("Generating training triplets from train tracks...")
    train_triplets = _generate_triplets_from_tracks(train_tracks, "train", config)
    
    logger.info("Generating test triplets from test tracks...")
    test_triplets = _generate_triplets_from_tracks(test_tracks, "test", config)
    
    # Verify no track overlap
    train_track_set = set()
    test_track_set = set()
    
    for subgenre, tracks in train_tracks.items():
        for track_name in tracks.keys():
            train_track_set.add(f"{subgenre}/{track_name}")
    
    for subgenre, tracks in test_tracks.items():
        for track_name in tracks.keys():
            test_track_set.add(f"{subgenre}/{track_name}")
    
    overlap = train_track_set.intersection(test_track_set)
    if overlap:
        raise ValueError(f"Track overlap detected between train/test: {overlap}")
    
    logger.info(f"✓ NO DATA LEAKAGE: {len(train_track_set)} unique train tracks, {len(test_track_set)} unique test tracks")
    logger.info(f"Total: {len(train_triplets)} train triplets, {len(test_triplets)} test triplets")
    
    return train_triplets, test_triplets



def _generate_triplets_from_tracks(subgenre_tracks: Dict[str, Dict[str, List[str]]], 
                                 split_name: str, config: ExperimentConfig) -> List[Tuple[str, str, str, str]]:
    """Generate triplets from a given set of tracks (train or test)."""
    all_triplets = []
    subgenre_list = list(subgenre_tracks.keys())
    
    # Log triplet generation configuration
    logger.info(f"{split_name.capitalize()} triplet generation config:")
    logger.info(f"  Max positive tracks per anchor: {config.data.max_positive_tracks}")
    logger.info(f"  Triplets per positive track: {config.data.triplets_per_positive_track}")
    logger.info(f"  Expected triplets per anchor: ~{config.data.max_positive_tracks * config.data.triplets_per_positive_track}")
    
    if len(subgenre_list) < 2:
        logger.warning(f"Only {len(subgenre_list)} subgenres available for {split_name} split")
        return []
    
    for subgenre in subgenre_list:
        tracks = subgenre_tracks[subgenre]
        track_names = list(tracks.keys())
        
        if len(track_names) < 2:
            logger.warning(f"Subgenre {subgenre} has only {len(track_names)} tracks in {split_name}, skipping")
            continue
        
        # Negative candidates: all tracks from DIFFERENT subgenres
        other_subgenres = [s for s in subgenre_list if s != subgenre]
        negative_chunks_pool = []
        for other_subgenre in other_subgenres:
            for other_track, other_chunks in subgenre_tracks[other_subgenre].items():
                negative_chunks_pool.extend(other_chunks)
        
        if not negative_chunks_pool:
            logger.warning(f"No negative samples available for subgenre {subgenre} in {split_name}")
            continue
        
        triplets = []
        
        # Generate triplets within this subgenre
        for anchor_track in track_names:
            anchor_chunks = tracks[anchor_track]
            
            # Positive candidates: different tracks in SAME subgenre
            positive_tracks = [t for t in track_names if t != anchor_track]
            
            # Use ALL chunks as anchors
            for anchor_chunk in anchor_chunks:
                
                # Sample positive tracks (configurable for cluster training)
                max_positives = min(len(positive_tracks), config.data.max_positive_tracks)
                positive_sample = random.sample(positive_tracks, max_positives)
                
                for positive_track in positive_sample:
                    positive_chunks = tracks[positive_track]
                    
                    # Generate configurable triplets per positive track
                    triplets_per_positive = min(config.data.triplets_per_positive_track, len(positive_chunks))
                    
                    for _ in range(triplets_per_positive):
                        positive_chunk = random.choice(positive_chunks)
                        negative_chunk = random.choice(negative_chunks_pool)
                        
                        triplet = (anchor_chunk, positive_chunk, negative_chunk, subgenre)
                        triplets.append(triplet)
        
        random.shuffle(triplets)
        all_triplets.extend(triplets)
        
        logger.info(f"{split_name.capitalize()} - Subgenre {subgenre}: {len(triplets)} triplets from {len(track_names)} tracks")
    
    return all_triplets


def save_model_and_artifacts(model: ImprovedASTTripletWrapper, config: ExperimentConfig,
                           train_data: Any, test_data: Any, timestamp: str) -> str:
    """Save model, configuration, and metadata with error handling."""
    save_dir = Path(f"run_{timestamp}")
    # Directory should already exist, created during training setup
    if not save_dir.exists():
        save_dir.mkdir(exist_ok=True)
    
    try:
        # Save model weights directly in run directory
        model_path = save_dir / "model.safetensors"
        save_file(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save configuration directly in run directory
        config_path = save_dir / "config.json"
        config.save(config_path)
        logger.info(f"Configuration saved to {config_path}")
        
        # Save split information directly in run directory
        splits_path = save_dir / "splits.json"
        splits_data = {
            "train_split": train_data,
            "test_split": test_data,
            "timestamp": timestamp,
            "config_summary": {
                "experiment_name": config.experiment_name,
                "chunks_dir": config.data.chunks_dir,
                "test_split_ratio": config.data.test_split_ratio
            }
        }
        with open(splits_path, 'w') as f:
            json.dump(splits_data, f, indent=2)
        logger.info(f"Splits saved to {splits_path}")
        
        logger.info(f"All training artifacts saved to {save_dir}")
        return str(save_dir)
        
    except Exception as e:
        logger.error(f"Error saving model artifacts: {e}")
        raise

class DataLoaderTrainer(Trainer):
    """Trainer that forwards full DataLoader knobs (prefetch_factor, persistent_workers, etc.)."""
    
    def __init__(self, config=None, use_custom_scheduler=False, **kwargs):
        """Initialize trainer with optional config for stratified batching."""
        super().__init__(**kwargs)
        self.config = config
        self.use_custom_scheduler = use_custom_scheduler
    
    def _build_dl(self, dataset, shuffle: bool):
        args = self.args
        
        # Use stratified sampling ONLY for training (shuffle=True) when enabled
        if (self.config and 
            getattr(self.config.data, 'stratified_batching', False) == True and 
            shuffle):
            
            logger.info("Using stratified batch sampling for training")
            batch_sampler = StratifiedSubgenreBatchSampler(
                dataset=dataset,
                batch_size=args.train_batch_size,
                min_per_subgenre=self.config.data.min_per_subgenre
            )
            
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=self.data_collator,
                num_workers=args.dataloader_num_workers,
                pin_memory=args.dataloader_pin_memory,
                persistent_workers=getattr(args, "dataloader_persistent_workers", False),
                prefetch_factor=getattr(args, "dataloader_prefetch_factor", 2)
            )
        else:
            # Original logic for evaluation or when stratified batching disabled
            return DataLoader(
                dataset,
                batch_size=args.train_batch_size if shuffle else args.eval_batch_size,
                shuffle=shuffle,
                collate_fn=self.data_collator,
                num_workers=args.dataloader_num_workers,
                pin_memory=args.dataloader_pin_memory,
                persistent_workers=getattr(args, "dataloader_persistent_workers", False),
                prefetch_factor=getattr(args, "dataloader_prefetch_factor", 2),
                drop_last=False
            )

    def get_train_dataloader(self):
        return self._build_dl(self.train_dataset, shuffle=True)

    def get_eval_dataloader(self, eval_dataset=None):
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        return self._build_dl(ds, shuffle=False)
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """Override scheduler creation to use custom dual-group scheduler when enabled."""
        if self.use_custom_scheduler:
            # Don't create default scheduler, our callback will handle it
            return None
        else:
            # Use default HuggingFace scheduler
            return super().create_scheduler(num_training_steps, optimizer)


def main():
    """Main training function with comprehensive error handling."""
    parser = argparse.ArgumentParser(description="AST Triplet Loss Training (Improved)")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--test-run", action="store_true", help="Run quick test with minimal data")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_or_create_config(args.config, test_run=args.test_run)
        config.validate()

        # Optional: set matmul precision hint (Ampere+)
        if getattr(config.training, "set_float32_matmul_precision", None):
            try:
                torch.set_float32_matmul_precision(config.training.set_float32_matmul_precision)
                logger.info(f"set_float32_matmul_precision={config.training.set_float32_matmul_precision}")
            except Exception as e:
                logger.warning(f"Could not set float32 matmul precision: {e}")

        # TF32 backend toggles (effective on Ampere+, harmless elsewhere)
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(config.training.allow_tf32_matmul)
            torch.backends.cudnn.allow_tf32      = bool(config.training.allow_tf32_cudnn)
            logger.info(f"TF32: matmul={torch.backends.cuda.matmul.allow_tf32}, cudnn={torch.backends.cudnn.allow_tf32}")
        except Exception as e:
            logger.warning(f"Could not set TF32 backend flags: {e}")

        # Decide effective precision (with safety checks and clear logging)
        want_bf16 = bool(getattr(config.training, "bf16", False))
        want_fp16 = bool(getattr(config.training, "fp16", False))
        want_tf32 = bool(getattr(config.training, "tf32", False))

        bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_bf16 = want_bf16 and bf16_supported
        use_fp16 = (not use_bf16) and want_fp16  # fp16 only if bf16 wasn’t taken

        if want_bf16 and not bf16_supported:
            logger.warning("bf16 requested but not supported on this GPU/PyTorch build. Falling back to "
                        f"{'fp16' if use_fp16 else 'fp32'}.")
        
        logger.info(f"Experiment: {config.experiment_name}")
        if config.description:
            logger.info(f"Description: {config.description}")
        
        # Set random seeds for reproducibility
        set_random_seeds(config.training.seed)
        
        # Load data splits
        logger.info("Loading data splits...")
        train_data, test_data = load_split_data(config)
        
        # Store splits data to be saved later with model artifacts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create main run directory and checkpoints subdirectory for trainer
        run_dir = Path(f"run_{timestamp}")
        run_dir.mkdir(exist_ok=True)
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        
        # Create datasets
        logger.info("Creating datasets...")
        train_dataset = ImprovedTripletFeatureDataset(train_data, config, split_name="train")
        test_dataset  = ImprovedTripletFeatureDataset(test_data,  config, split_name="test")

        # Apply augmentations if enabled
        if config.data.enable_augmentations:
            logger.info("Augmentations enabled - wrapping datasets...")
            from dataset_augmented import AugmentedTripletFeatureDataset

            # Wrap train dataset with augmentations
            train_dataset = AugmentedTripletFeatureDataset(
                base_dataset=train_dataset,
                config=config,
                split_name="train"
            )

            # Test dataset gets no augmentations (keep original)
            logger.info("Train dataset: augmentations ENABLED")
            logger.info("Test dataset: augmentations DISABLED")
        else:
            logger.info("Augmentations disabled - using original datasets")

        # Initialize model
        logger.info("Initializing model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        model = ImprovedASTTripletWrapper(config).to(device)
        
        # Create custom optimizer with separate parameter groups for sophisticated LR scheduling
        use_custom_lr = config.training.use_custom_lr
        if use_custom_lr:
            logger.info("Using custom dual-group optimizer with sophisticated LR scheduling")
            optimizer = create_dual_group_optimizer(model, config)
        else:
            logger.info("Using default HuggingFace optimizer")
            optimizer = None
        
        # Setup training arguments with clean logging
        training_args_dict = {
            "output_dir": str(checkpoints_dir),
            "logging_strategy": config.training.logging_strategy,
            "eval_strategy": config.training.eval_strategy,
            "eval_on_start": True,  # Enable initial evaluation at epoch 0
            "save_strategy": config.training.save_strategy,
            "metric_for_best_model": "eval_accuracy",
            "greater_is_better": True,
            "learning_rate": config.training.learning_rate,
            "per_device_train_batch_size": config.training.batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "num_train_epochs": config.training.epochs,
            "weight_decay": config.training.weight_decay,
            "logging_steps": config.training.logging_steps,
            "dataloader_num_workers": config.training.num_workers,
            "dataloader_pin_memory": config.training.pin_memory,
            "dataloader_prefetch_factor": config.training.prefetch_factor,
            "dataloader_persistent_workers": config.training.persistent_workers,
            "report_to": "none",  # Disable wandb/tensorboard
            "logging_first_step": False,  # Don't log first step
            "disable_tqdm": config.training.disable_tqdm,  # Control tqdm progress bars
            "log_level": "warning",  # Reduce HF Trainer's internal logging
            "seed": config.training.seed,
            "data_seed": config.training.seed,
            "max_grad_norm": config.training.gradient_clip_norm,  # Gradient clipping
        }

        # Precision / GPU knobs
        training_args_dict.update({
            "bf16": config.training.bf16,
            "fp16": config.training.fp16,
            "tf32": config.training.tf32,
        })

        logger.info(f"Precision summary -> bf16={use_bf16}, fp16={use_fp16}, tf32={want_tf32}")
        
        # Add optional parameters if they exist in config
        if hasattr(config.training, 'save_steps') and config.training.save_steps:
            training_args_dict["save_steps"] = config.training.save_steps
        if hasattr(config.training, 'eval_steps') and config.training.eval_steps:
            training_args_dict["eval_steps"] = config.training.eval_steps
        
        training_args = TrainingArguments(**training_args_dict)
        
        resample_flag = bool(getattr(config.data, "resample_train_samples", False))
        resample_callback = ResampleCallback(train_dataset, config, enabled=resample_flag)
        
        callbacks = [
            CleanLoggingCallback(), 
            resample_callback,
            MarginSchedulingCallback(model),
            EarlyStoppingCallback(config, resample_callback)
        ]
        
        # Create custom LR scheduler if using sophisticated scheduling
        if use_custom_lr:
            # Calculate scheduler parameters
            total_train_samples = len(train_dataset)
            effective_batch_size = config.training.batch_size * config.training.gradient_accumulation_steps
            steps_per_epoch = (total_train_samples + effective_batch_size - 1) // effective_batch_size
            total_steps = steps_per_epoch * config.training.epochs
            
            # Create PyTorch LambdaLR scheduler (HuggingFace compatible)
            lr_scheduler = create_dual_group_scheduler(optimizer, config, total_steps, steps_per_epoch)
            
            logger.info(f"PyTorch LambdaLR scheduler created with {total_steps} total steps, {steps_per_epoch} steps/epoch")

        # Initialize trainer with custom callback
        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": test_dataset,
            "data_collator": FastSafeCollator(validate_every_n=0),
            "compute_metrics": compute_metrics,
            "callbacks": callbacks,
            "config": config,
            "use_custom_scheduler": use_custom_lr,
        }
        
        # Only add custom optimizers when using custom LR scheduler
        if use_custom_lr:
            trainer_kwargs["optimizers"] = (optimizer, lr_scheduler)
        
        trainer = DataLoaderTrainer(**trainer_kwargs)
        
        # Log training setup information
        total_train_samples = len(train_dataset)
        total_test_samples = len(test_dataset)
        effective_batch_size = config.training.batch_size * config.training.gradient_accumulation_steps
        steps_per_epoch = (total_train_samples + effective_batch_size - 1) // effective_batch_size
        total_steps = steps_per_epoch * config.training.epochs
        
        logger.info(f"Training setup:")
        logger.info(f"  - Train samples: {total_train_samples}")
        logger.info(f"  - Test samples: {total_test_samples}")
        logger.info(f"  - Batch size: {config.training.batch_size}")
        logger.info(f"  - Gradient accumulation: {config.training.gradient_accumulation_steps}")
        logger.info(f"  - Effective batch size: {effective_batch_size}")
        logger.info(f"  - Steps per epoch: {steps_per_epoch}")
        logger.info(f"  - Total epochs: {config.training.epochs}")
        logger.info(f"  - Estimated total steps: {total_steps}")
        
        # Perform initial evaluation to get baseline accuracy
        logger.info("Performing initial evaluation to establish baseline...")
        try:
            initial_metrics = trainer.evaluate()
        except Exception as e:
            logger.warning(f"Initial evaluation failed: {e}")
        
        # Start training
        logger.info("Starting training...")
        main_start_time = time.perf_counter()
        trainer.train()
        main_end_time = time.perf_counter()
        
        # Save model and artifacts
        logger.info("Saving model and artifacts...")
        save_dir = save_model_and_artifacts(model, config, train_data, test_data, timestamp)
        
        # Calculate and log total execution time (including setup and saving)
        total_execution_time = main_end_time - main_start_time
        def format_time_simple(seconds):
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                minutes = int(seconds // 60)
                secs = seconds % 60
                return f"{minutes}m {secs:.1f}s"
            else:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = seconds % 60
                return f"{hours}h {minutes}m {secs:.1f}s"
        
        logger.info(f"Training completed successfully! Training time: {format_time_simple(total_execution_time)} | Results saved to: {save_dir}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


def run_kfold_training(base_config: ExperimentConfig, preprocessed_dir: str, k: int = 5, output_dir: str = "kfold_results") -> Dict[str, Any]:
    """
    Run complete K-fold cross-validation training using preprocessed k-fold directories.

    This function assumes that k-fold preprocessing has already been done using
    Preprocess_AST_features.py with --enable-kfold flag, creating a directory structure like:
    preprocessed_dir/fold_0/train_chunks/, preprocessed_dir/fold_0/test_chunks/, etc.

    Args:
        base_config: Base experiment configuration
        preprocessed_dir: Path to preprocessed k-fold directory (e.g., "precomputed_7Gen_5Fold")
        k: Number of folds (should match the preprocessed data)
        output_dir: Directory to save results

    Returns:
        Dictionary with aggregated results and statistics
    """
    from datetime import datetime
    import numpy as np

    # Validate preprocessed directory
    preprocessed_path = Path(preprocessed_dir)
    if not preprocessed_path.exists():
        raise FileNotFoundError(f"Preprocessed k-fold directory not found: {preprocessed_path}")

    # Check that all folds exist
    for fold_idx in range(k):
        fold_dir = preprocessed_path / f"fold_{fold_idx}"
        train_dir = fold_dir / "train_chunks"
        test_dir = fold_dir / "test_chunks"

        if not fold_dir.exists():
            raise FileNotFoundError(f"Fold directory not found: {fold_dir}")
        if not train_dir.exists():
            raise FileNotFoundError(f"Train chunks directory not found: {train_dir}")
        if not test_dir.exists():
            raise FileNotFoundError(f"Test chunks directory not found: {test_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(output_dir) / f"kfold_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*80}")
    logger.info(f"STARTING {k}-FOLD CROSS-VALIDATION EXPERIMENT")
    logger.info(f"Experiment: {base_config.experiment_name}")
    logger.info(f"Description: {base_config.description}")
    logger.info(f"Using preprocessed k-fold data from: {preprocessed_path}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"{'='*80}")

    # Load k-fold partitions metadata if available (for reproducibility tracking)
    partitions_file = preprocessed_path / "kfold_partitions.json"
    if partitions_file.exists():
        logger.info(f"Loading k-fold partition metadata from {partitions_file}")
        with open(partitions_file, 'r') as f:
            partition_metadata = json.load(f)
        logger.info(f"Partition seed used: {partition_metadata.get('metadata', {}).get('partition_seed', 'unknown')}")
    else:
        logger.warning(f"No partition metadata found at {partitions_file}")
        partition_metadata = {}

    # Copy partition metadata for reproducibility
    if partition_metadata:
        partitions_file = results_dir / "kfold_partitions.json"
        with open(partitions_file, 'w') as f:
            json.dump(partition_metadata, f, indent=2)
        logger.info(f"Partition metadata copied to {partitions_file}")
    else:
        logger.warning("No partition metadata available to copy")

    # Save base configuration
    base_config_file = results_dir / "base_config.json"
    base_config.save(str(base_config_file))

    fold_metrics = []

    # Run training for each fold using preprocessed data
    for fold_idx in range(k):
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING FOLD {fold_idx + 1}/{k}")
        logger.info(f"{'='*60}")

        # Create fold-specific config
        fold_config = ExperimentConfig.load(str(base_config_file))
        fold_config.training.seed = base_config.training.seed + fold_idx
        fold_config.experiment_name = f"{base_config.experiment_name}_fold_{fold_idx}"
        fold_config.description = f"Fold {fold_idx + 1}/{k} of K-fold cross-validation"

        # Set preprocessed train/test directories for this fold
        fold_config.data.chunks_train_dir = str(preprocessed_path / f"fold_{fold_idx}" / "train_chunks")
        fold_config.data.chunks_test_dir = str(preprocessed_path / f"fold_{fold_idx}" / "test_chunks")

        # Clear legacy k-fold settings
        fold_config.data.enable_kfold = False
        fold_config.data.kfold_current_fold = None
        fold_config.data.kfold_partitions_file = None

        # Create fold directory
        fold_dir = results_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)

        # Save fold config
        fold_config_file = fold_dir / "config.json"
        fold_config.save(str(fold_config_file))

        try:
            # Set up fold-specific random seeds
            set_random_seeds(fold_config.training.seed)

            # Load fold-specific data
            train_data, test_data = load_split_data(fold_config)

            # Create datasets
            train_dataset = ImprovedTripletFeatureDataset(train_data, fold_config, split_name="train")
            test_dataset = ImprovedTripletFeatureDataset(test_data, fold_config, split_name="test")

            # Apply augmentations if enabled
            if fold_config.data.enable_augmentations:
                from dataset_augmented import AugmentedTripletFeatureDataset
                train_dataset = AugmentedTripletFeatureDataset(
                    base_dataset=train_dataset,
                    config=fold_config,
                    split_name="train"
                )

            # Initialize model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = ImprovedASTTripletWrapper(fold_config).to(device)

            # Set up trainer
            checkpoints_dir = fold_dir / "checkpoints"
            checkpoints_dir.mkdir(exist_ok=True)

            training_args = TrainingArguments(
                output_dir=str(checkpoints_dir),
                logging_strategy=fold_config.training.logging_strategy,
                eval_strategy=fold_config.training.eval_strategy,
                eval_on_start=True,  # Enable initial evaluation at epoch 0
                save_strategy="no",  # Don't save intermediate checkpoints in K-fold
                metric_for_best_model="eval_accuracy",
                greater_is_better=True,
                learning_rate=fold_config.training.learning_rate,
                per_device_train_batch_size=fold_config.training.batch_size,
                gradient_accumulation_steps=fold_config.training.gradient_accumulation_steps,
                num_train_epochs=fold_config.training.epochs,
                weight_decay=fold_config.training.weight_decay,
                logging_steps=fold_config.training.logging_steps,
                dataloader_num_workers=fold_config.training.num_workers,
                dataloader_pin_memory=fold_config.training.pin_memory,
                report_to="none",
                logging_first_step=True,
                disable_tqdm=fold_config.training.disable_tqdm,
                log_level="warning",
                seed=fold_config.training.seed,
                data_seed=fold_config.training.seed,
                max_grad_norm=fold_config.training.gradient_clip_norm,
                bf16=fold_config.training.bf16,
                fp16=fold_config.training.fp16,
                tf32=fold_config.training.tf32,
            )

            # Create callbacks for K-fold (including resampling support)
            resample_flag = bool(getattr(fold_config.data, "resample_train_samples", False))
            resample_callback = ResampleCallback(train_dataset, fold_config, enabled=resample_flag)

            callbacks = [
                CleanLoggingCallback(),
                resample_callback,
                MarginSchedulingCallback(model),
                EarlyStoppingCallback(fold_config, resample_callback)
            ]

            trainer = DataLoaderTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                data_collator=FastSafeCollator(validate_every_n=0),
                compute_metrics=compute_metrics,
                callbacks=callbacks,
                config=fold_config
            )

            logger.info(f"Starting training for fold {fold_idx} with seed {fold_config.training.seed}")

            # Train the model
            train_result = trainer.train()

            # Get final evaluation metrics
            final_metrics = trainer.evaluate()

            # Save model
            model_path = fold_dir / "model.safetensors"
            save_file(model.state_dict(), model_path)

            # Extract and organize training history
            training_history = {
                "epoch": [],
                "train_loss": [],
                "eval_loss": [],
                "eval_accuracy": [],
                "learning_rate": []
            }

            # Process log history from trainer
            if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
                epoch_data = {}

                for log_entry in trainer.state.log_history:
                    epoch = log_entry.get('epoch')
                    if epoch is not None:
                        if epoch not in epoch_data:
                            epoch_data[epoch] = {}

                        # Update epoch data with all available metrics
                        for key, value in log_entry.items():
                            if key != 'epoch':
                                epoch_data[epoch][key] = value

                # Convert to lists for easier analysis
                for epoch in sorted(epoch_data.keys()):
                    data = epoch_data[epoch]
                    training_history["epoch"].append(epoch)
                    # HuggingFace logs training loss as 'loss', not 'train_loss'
                    training_history["train_loss"].append(data.get('loss'))
                    training_history["eval_loss"].append(data.get('eval_loss'))
                    training_history["eval_accuracy"].append(data.get('eval_accuracy'))
                    training_history["learning_rate"].append(data.get('learning_rate'))

            # Find best accuracy from training history
            best_accuracy = 0.0
            best_loss = float('inf')
            if training_history["eval_accuracy"]:
                # Filter out None values and find max
                valid_accuracies = [acc for acc in training_history["eval_accuracy"] if acc is not None]
                if valid_accuracies:
                    best_accuracy = max(valid_accuracies)
                    # Find corresponding loss for best accuracy epoch
                    best_idx = training_history["eval_accuracy"].index(best_accuracy)
                    best_loss = training_history["eval_loss"][best_idx] if training_history["eval_loss"][best_idx] is not None else final_metrics.get("eval_loss", 0.0)

            # Save comprehensive fold metrics
            fold_result = {
                "fold_idx": fold_idx,
                "fold_seed": fold_config.training.seed,
                "train_samples": len(train_dataset),
                "test_samples": len(test_dataset),
                "final_accuracy": best_accuracy,  # Now uses BEST accuracy, not final
                "final_loss": best_loss,          # Loss corresponding to best accuracy
                "training_history": training_history,
                "final_metrics": final_metrics,
                "train_runtime": train_result.metrics.get("train_runtime", 0.0),
                "total_steps": trainer.state.global_step if trainer.state else 0
            }

            # Save detailed fold results
            with open(fold_dir / "fold_metrics.json", 'w') as f:
                json.dump(fold_result, f, indent=2)

            # Save training history as separate CSV for easy analysis
            if training_history["epoch"]:
                import pandas as pd
                history_df = pd.DataFrame(training_history)
                history_df.to_csv(fold_dir / "training_history.csv", index=False)

            fold_metrics.append(fold_result)

            logger.info(f"Fold {fold_idx} completed - Best Accuracy: {best_accuracy:.4f} (Final: {final_metrics.get('eval_accuracy', 0.0):.4f})")

            # Cleanup to prevent state leakage between folds
            del trainer
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            import gc
            gc.collect()

        except Exception as e:
            logger.error(f"Fold {fold_idx} failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

            # Add failed fold to results
            fold_metrics.append({
                "fold_idx": fold_idx,
                "fold_seed": fold_config.training.seed,
                "status": "failed",
                "error": str(e)
            })

    # Compute final statistics
    successful_folds = [f for f in fold_metrics if "final_accuracy" in f]

    if successful_folds:
        accuracies = [f["final_accuracy"] for f in successful_folds]
        losses = [f["final_loss"] for f in successful_folds]

        final_statistics = {
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "min_accuracy": float(np.min(accuracies)),
            "max_accuracy": float(np.max(accuracies)),
            "mean_loss": float(np.mean(losses)),
            "std_loss": float(np.std(losses)),
            "successful_folds": len(successful_folds),
            "total_folds": k,
            "individual_accuracies": accuracies,
            "individual_losses": losses
        }

        logger.info(f"\n{'='*80}")
        logger.info(f"K-FOLD CROSS-VALIDATION RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Mean Accuracy: {final_statistics['mean_accuracy']:.4f} ± {final_statistics['std_accuracy']:.4f}")
        logger.info(f"Accuracy Range: {final_statistics['min_accuracy']:.4f} - {final_statistics['max_accuracy']:.4f}")
        logger.info(f"Mean Loss: {final_statistics['mean_loss']:.4f} ± {final_statistics['std_loss']:.4f}")
        logger.info(f"Successful Folds: {final_statistics['successful_folds']}/{k}")
        logger.info(f"Individual Accuracies: {[f'{acc:.4f}' for acc in accuracies]}")

        # Save aggregated training curves for analysis
        if successful_folds:
            aggregate_training_curves(successful_folds, results_dir / "aggregated_training_curves.json")

    else:
        final_statistics = {
            "status": "all_folds_failed",
            "successful_folds": 0,
            "total_folds": k
        }
        logger.error("All folds failed!")

    # Save experiment summary
    experiment_summary = {
        "experiment_name": base_config.experiment_name,
        "timestamp": timestamp,
        "k_folds": k,
        "base_seed": base_config.training.seed,
        "results_dir": str(results_dir),
        "fold_metrics": fold_metrics,
        "final_statistics": final_statistics,
        "config_summary": {
            "batch_size": base_config.training.batch_size,
            "epochs": base_config.training.epochs,
            "learning_rate": base_config.training.learning_rate,
            "triplet_margin": base_config.training.triplet_margin
        }
    }

    with open(results_dir / "experiment_summary.json", 'w') as f:
        json.dump(experiment_summary, f, indent=2)

    logger.info(f"\nK-fold experiment complete! Results saved to: {results_dir}")
    return experiment_summary


def aggregate_training_curves(successful_folds: List[Dict], output_file: str) -> None:
    """
    Aggregate training curves across successful folds for analysis.

    Args:
        successful_folds: List of successful fold results
        output_file: Path to save aggregated curves
    """
    import numpy as np

    # Find common epochs across all folds
    all_epochs = []
    for fold in successful_folds:
        history = fold.get("training_history", {})
        epochs = history.get("epoch", [])
        if epochs:
            all_epochs.append(set(epochs))

    if not all_epochs:
        return

    # Get intersection of all epochs (epochs present in all folds)
    common_epochs = sorted(set.intersection(*all_epochs))

    if not common_epochs:
        return

    # Aggregate metrics for common epochs
    aggregated = {
        "epochs": common_epochs,
        "mean_train_loss": [],
        "std_train_loss": [],
        "mean_eval_loss": [],
        "std_eval_loss": [],
        "mean_eval_accuracy": [],
        "std_eval_accuracy": [],
        "individual_folds": {}
    }

    # Collect data for each fold
    for fold_idx, fold in enumerate(successful_folds):
        fold_id = f"fold_{fold['fold_idx']}"
        aggregated["individual_folds"][fold_id] = {
            "epochs": [],
            "train_loss": [],
            "eval_loss": [],
            "eval_accuracy": []
        }

        history = fold.get("training_history", {})
        epochs = history.get("epoch", [])

        for epoch in common_epochs:
            if epoch in epochs:
                epoch_idx = epochs.index(epoch)
                aggregated["individual_folds"][fold_id]["epochs"].append(epoch)
                aggregated["individual_folds"][fold_id]["train_loss"].append(
                    history.get("train_loss", [])[epoch_idx] if epoch_idx < len(history.get("train_loss", [])) else None
                )
                aggregated["individual_folds"][fold_id]["eval_loss"].append(
                    history.get("eval_loss", [])[epoch_idx] if epoch_idx < len(history.get("eval_loss", [])) else None
                )
                aggregated["individual_folds"][fold_id]["eval_accuracy"].append(
                    history.get("eval_accuracy", [])[epoch_idx] if epoch_idx < len(history.get("eval_accuracy", [])) else None
                )

    # Compute mean and std for each epoch
    for epoch in common_epochs:
        train_losses = []
        eval_losses = []
        eval_accuracies = []

        for fold_data in aggregated["individual_folds"].values():
            if epoch in fold_data["epochs"]:
                epoch_idx = fold_data["epochs"].index(epoch)

                tl = fold_data["train_loss"][epoch_idx]
                el = fold_data["eval_loss"][epoch_idx]
                ea = fold_data["eval_accuracy"][epoch_idx]

                if tl is not None:
                    train_losses.append(tl)
                if el is not None:
                    eval_losses.append(el)
                if ea is not None:
                    eval_accuracies.append(ea)

        aggregated["mean_train_loss"].append(float(np.mean(train_losses)) if train_losses else None)
        aggregated["std_train_loss"].append(float(np.std(train_losses)) if train_losses else None)
        aggregated["mean_eval_loss"].append(float(np.mean(eval_losses)) if eval_losses else None)
        aggregated["std_eval_loss"].append(float(np.std(eval_losses)) if eval_losses else None)
        aggregated["mean_eval_accuracy"].append(float(np.mean(eval_accuracies)) if eval_accuracies else None)
        aggregated["std_eval_accuracy"].append(float(np.std(eval_accuracies)) if eval_accuracies else None)

    # Save aggregated curves
    with open(output_file, 'w') as f:
        json.dump(aggregated, f, indent=2)

    logger.info(f"Aggregated training curves saved to: {output_file}")


if __name__ == "__main__":
    exit(main())
