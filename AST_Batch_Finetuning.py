#!/usr/bin/env python3
"""
Cluster-Optimized AST Triplet Loss Training with Direct WAV Processing

This version processes WAV files directly without requiring preprocessed features,
making it ideal for cluster environments where storage efficiency is important.
It maintains identical training logic to AST_Triplet_training.py while enabling
on-the-fly audio processing.

Key differences from AST_Triplet_training.py:
- Processes audio files directly from WAV/ directory
- On-the-fly chunking and feature extraction
- Memory-efficient audio loading
- Cluster-optimized batching
- Maintains identical training statistics and behavior
"""

import os
import json
import logging
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Generator
from collections import defaultdict
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import ASTModel, ASTFeatureExtractor, TrainingArguments, Trainer, TrainerCallback
from safetensors.torch import save_file, load_file

from config import ExperimentConfig, load_or_create_config

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


class AudioProcessingError(Exception):
    """Raised when audio processing fails."""
    pass


class CleanLoggingCallback(TrainerCallback):
    """Custom callback for clean, readable logging without redundant information."""
    
    def __init__(self):
        self.last_logged_step = -1
        self.last_logged_epoch = -1
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called whenever logging occurs - filters and formats logs cleanly."""
        if logs is None:
            return
            
        # Remove unwanted fields completely
        unwanted_fields = [
            'grad_norm', 'eval_runtime', 'eval_samples_per_second', 
            'eval_steps_per_second', 'train_runtime', 'train_samples_per_second',
            'train_steps_per_second', 'total_flos', 'train_steps_per_second'
        ]
        
        for field in unwanted_fields:
            logs.pop(field, None)
        
        # Only log meaningful progress updates (avoid duplicates)
        current_step = logs.get('step', 0)
        current_epoch = logs.get('epoch', 0)
        
        if 'train_loss' in logs and 'eval_loss' not in logs:
            # Training step - only log every N steps to reduce noise
            if current_step != self.last_logged_step and current_step % 50 == 0:
                logger.info(f"Training - Step {current_step}: "
                           f"loss={logs['train_loss']:.4f}, "
                           f"lr={logs.get('learning_rate', 0):.2e}")
                self.last_logged_step = current_step
                
        elif 'eval_loss' in logs:
            # Evaluation step - always log these as they're less frequent
            if current_epoch != self.last_logged_epoch:
                logger.info(f"Evaluation - Epoch {current_epoch}: "
                           f"train_loss={logs.get('train_loss', 0):.4f}, "
                           f"eval_loss={logs['eval_loss']:.4f}, "
                           f"eval_accuracy={logs.get('eval_accuracy', 0):.3f}")
                self.last_logged_epoch = current_epoch

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


# ========== AUDIO PROCESSING FUNCTIONS ==========

# Cache for resampler objects
_resampler_cache = {}

def get_resampler(orig_freq: int, new_freq: int) -> torchaudio.transforms.Resample:
    """Get cached resampler to avoid recreating objects."""
    key = (orig_freq, new_freq)
    if key not in _resampler_cache:
        _resampler_cache[key] = torchaudio.transforms.Resample(
            orig_freq=orig_freq, new_freq=new_freq
        )
    return _resampler_cache[key]


def load_and_resample_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Load audio file and convert to mono with target sample rate.
    
    Args:
        path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        Mono audio tensor at target sample rate
        
    Raises:
        AudioProcessingError: If audio loading or processing fails
    """
    try:
        # Load audio
        waveform, original_sr = torchaudio.load(path)
        
        # Validate audio
        duration = waveform.shape[-1] / original_sr
        if duration < 1.0:
            raise AudioProcessingError(f"Audio too short: {duration:.2f}s < 1.0s")
        
        # Convert to mono by averaging channels
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if necessary
        if original_sr != target_sr:
            resampler = get_resampler(original_sr, target_sr)
            waveform = resampler(waveform)
        
        # Remove channel dimension and validate
        result = waveform.squeeze(0)
        
        if torch.isnan(result).any():
            raise AudioProcessingError("Audio contains NaN values")
        if torch.isinf(result).any():
            raise AudioProcessingError("Audio contains infinite values")
        
        return result
        
    except Exception as e:
        if isinstance(e, AudioProcessingError):
            raise
        raise AudioProcessingError(f"Failed to load audio from {path}: {e}")


def chunk_waveform_random(waveform: torch.Tensor, sr: int, duration_s: int = 10, 
                         max_chunks: int = 3, seed: Optional[int] = None) -> List[torch.Tensor]:
    """
    Extract random chunks from waveform, matching Preprocess_AST_features.py behavior.
    
    Args:
        waveform: Input audio waveform
        sr: Sample rate
        duration_s: Duration of each chunk in seconds
        max_chunks: Maximum number of chunks to generate
        seed: Random seed for reproducible sampling
        
    Returns:
        List of audio chunks as tensors
    """
    if seed is not None:
        np.random.seed(seed)
    
    chunk_size = duration_s * sr
    total_length = waveform.shape[0]
    
    if chunk_size > total_length:
        logger.warning(f"Chunk size ({chunk_size}) larger than audio length ({total_length})")
        return [waveform]
    
    # Random sampling: select random positions throughout the audio
    max_start = total_length - chunk_size
    if max_start <= 0:
        return [waveform]
    
    # Generate random positions ensuring no overlap
    positions = []
    attempts = 0
    max_attempts = max_chunks * 10
    
    while len(positions) < max_chunks and attempts < max_attempts:
        pos = np.random.randint(0, max_start + 1)
        
        # Check for overlap with existing positions
        min_distance = chunk_size
        too_close = any(abs(pos - existing) < min_distance for existing in positions)
        if not too_close:
            positions.append(pos)
        
        attempts += 1
    
    # Extract chunks
    chunks = []
    for start in sorted(positions):
        end = start + chunk_size
        chunk = waveform[start:end]
        
        # Validate chunk
        if torch.isnan(chunk).any() or torch.isinf(chunk).any():
            logger.warning(f"Skipping chunk due to invalid values")
            continue
        
        chunks.append(chunk)
    
    return chunks


def extract_ast_features(audio_chunk: torch.Tensor, feature_extractor: ASTFeatureExtractor,
                        target_sr: int = 16000) -> Dict[str, torch.Tensor]:
    """
    Extract AST features from audio chunk, matching preprocessing output format.
    
    Args:
        audio_chunk: Audio tensor
        feature_extractor: AST feature extractor
        target_sr: Target sample rate
        
    Returns:
        Dictionary with 'input_values' key containing features
    """
    try:
        # Convert to numpy for feature extractor
        audio_np = audio_chunk.numpy()
        
        # Extract features
        inputs = feature_extractor(
            audio_np, 
            sampling_rate=target_sr, 
            return_tensors="pt",
            padding="max_length",
            max_length=1024  # AST standard input length
        )
        
        # Validate output
        if "input_values" not in inputs:
            raise AudioProcessingError("Feature extractor did not return 'input_values'")
        
        input_values = inputs["input_values"]
        validate_tensor_shape(input_values, tensor_name="input_values")
        
        # Squeeze batch dimension if present
        if input_values.ndim == 3 and input_values.shape[0] == 1:
            input_values = input_values.squeeze(0)
        
        return {"input_values": input_values}
        
    except Exception as e:
        raise AudioProcessingError(f"Feature extraction failed: {e}")


# ========== DATASET NORMALIZATION ==========

class OnlineStatsCalculator:
    """
    Memory-efficient online statistics calculation using Welford's algorithm.
    Identical to the one in Preprocess_AST_features.py.
    """
    
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared deviations
    
    def update(self, value: float) -> None:
        """Update statistics with a new value."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2
    
    def update_batch(self, values: torch.Tensor) -> None:
        """Update statistics with a batch of values."""
        if values.numel() == 0:
            return
            
        batch_mean = values.mean().item()
        batch_std = values.std(unbiased=False).item()
        
        # Update with batch statistics (simplified)
        self.update(batch_mean)
    
    @property
    def variance(self) -> float:
        """Get current variance estimate."""
        return self.M2 / self.count if self.count > 1 else 0.0
    
    @property
    def std(self) -> float:
        """Get current standard deviation estimate."""
        return np.sqrt(self.variance)


def compute_dataset_normalization_stats(wav_files: List[str], config: ExperimentConfig) -> Tuple[float, float]:
    """
    Compute dataset-wide normalization statistics from WAV files.
    
    Args:
        wav_files: List of WAV file paths
        config: Experiment configuration
        
    Returns:
        Tuple of (mean, std) values for normalization
    """
    logger.info("🧮 Computing dataset-wide normalization statistics...")
    
    # Initialize extractor without normalization
    extractor = ASTFeatureExtractor.from_pretrained(config.model.pretrained_model)
    if hasattr(extractor, "do_normalize"):
        extractor.do_normalize = False
    else:
        logger.warning("Extractor does not have do_normalize attribute")
    
    # Initialize online stats calculator
    stats_calc = OnlineStatsCalculator()
    target_sr = 16000
    chunk_duration = 10
    
    # Process ALL files for statistics (ensuring maximum accuracy)
    sample_files = wav_files  # Use entire dataset for statistics
    
    successful_files = 0
    total_chunks = 0
    
    logger.info("Processing files for normalization statistics (no progress bar to avoid disk I/O)...")
    
    for i, filepath in enumerate(sample_files):
        try:
            # Load and resample audio
            waveform = load_and_resample_audio(filepath, target_sr)
            
            # Extract random chunks (same as in preprocessing)
            chunks = chunk_waveform_random(
                waveform, target_sr, duration_s=chunk_duration, max_chunks=3, seed=42
            )
            
            file_chunks = 0
            for chunk in chunks:
                try:
                    # Extract features without normalization
                    audio_np = chunk.numpy()
                    inputs = extractor(
                        audio_np,
                        sampling_rate=target_sr,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=1024
                    )
                    
                    if "input_values" not in inputs:
                        continue
                    
                    # Update statistics
                    features = inputs["input_values"][0].float()
                    stats_calc.update_batch(features)
                    
                    file_chunks += 1
                    total_chunks += 1
                    
                except Exception as e:
                    if file_chunks == 0:  # Only log first error per file
                        logger.warning(f"Error processing chunk from {filepath}: {e}")
                    continue
            
            if file_chunks > 0:
                successful_files += 1
                
                # Log progress every 100 files (no frequent disk writes)
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(sample_files)} files, "
                               f"chunks: {total_chunks}, "
                               f"mean: {stats_calc.mean:.4f}, "
                               f"std: {stats_calc.std:.4f}")
            
        except Exception as e:
            logger.warning(f"Error processing {filepath}: {e}")
            continue
    
    # Validate results
    if stats_calc.count == 0:
        logger.warning("No valid chunks found for statistics computation, using defaults")
        return 0.0, 1.0
    
    mean_val = float(stats_calc.mean)
    std_val = float(stats_calc.std) if stats_calc.std > 0 else 1.0
    
    logger.info(f"✅ Normalization stats computed from {successful_files} files, {total_chunks} chunks")
    logger.info(f"📊 Dataset statistics — mean: {mean_val:.6f}, std: {std_val:.6f}")
    
    return mean_val, std_val


# ========== CACHING SYSTEM ==========

class ClusterFeatureCache:
    """
    High-performance feature cache optimized for cluster environments with abundant RAM.
    """
    
    def __init__(self, config: ExperimentConfig, split_name: str = "unknown", 
                 normalization_stats: Optional[Tuple[float, float]] = None):
        self.config = config
        self.split_name = split_name
        self.cache = {}  # filepath -> features
        self.access_count = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_size_bytes = int(config.data.max_cache_size_gb * 1024**3)
        self.current_size_bytes = 0
        
        # Feature extraction components
        self.target_sr = 16000
        self.chunk_duration = 10
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(
            config.model.pretrained_model
        )
        
        # Apply dataset-wide normalization statistics
        if normalization_stats:
            mean_val, std_val = normalization_stats
            self.feature_extractor.mean = mean_val
            self.feature_extractor.std = std_val
            logger.info(f"Applied dataset normalization: mean={mean_val:.6f}, std={std_val:.6f}")
        else:
            logger.warning("No normalization statistics provided - using default normalization")
        
        logger.info(f"Initialized {split_name} cache with {config.data.max_cache_size_gb}GB limit")
    
    def _estimate_feature_size(self, features: Dict[str, torch.Tensor]) -> int:
        """Estimate memory size of cached features in bytes."""
        total_bytes = 0
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                total_bytes += tensor.nelement() * tensor.element_size()
        return total_bytes
    
    def _should_cache(self, filepath: str) -> bool:
        """Determine if file should be cached based on cache settings."""
        if not self.config.data.enable_feature_caching:
            return False
        
        if self.split_name == "train" and not self.config.data.cache_train_dataset:
            return False
        
        if self.split_name == "test" and not self.config.data.cache_test_dataset:
            return False
        
        return True
    
    def get_features(self, filepath: str) -> Dict[str, torch.Tensor]:
        """Get features from cache or compute them."""
        if self._should_cache(filepath) and filepath in self.cache:
            self.cache_hits += 1
            self.access_count[filepath] += 1
            return self.cache[filepath]
        
        # Cache miss - compute features
        self.cache_misses += 1
        features = self._compute_features(filepath)
        
        # Cache if enabled and we have space
        if self._should_cache(filepath):
            feature_size = self._estimate_feature_size(features)
            
            if self.current_size_bytes + feature_size <= self.max_size_bytes:
                self.cache[filepath] = features
                self.current_size_bytes += feature_size
                self.access_count[filepath] += 1
                logger.debug(f"Cached {filepath} ({feature_size/1024/1024:.1f}MB)")
            else:
                logger.warning(f"Cache full, cannot cache {filepath}")
        
        return features
    
    def _compute_features(self, filepath: str) -> Dict[str, torch.Tensor]:
        """Compute AST features from audio file."""
        try:
            # Load and resample audio
            waveform = load_and_resample_audio(filepath, self.target_sr)
            
            # Extract random chunk
            chunks = chunk_waveform_random(
                waveform, 
                self.target_sr, 
                duration_s=self.chunk_duration,
                max_chunks=1,
                seed=None
            )
            
            if not chunks:
                raise AudioProcessingError(f"No valid chunks extracted from {filepath}")
            
            # Use first chunk and extract AST features
            chunk = chunks[0]
            features = extract_ast_features(chunk, self.feature_extractor, self.target_sr)
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing features for {filepath}: {e}")
            raise AudioProcessingError(f"Failed to process {filepath}: {e}")
    
    def preload_files(self, filepaths: List[str]) -> None:
        """Preload features for multiple files - pure RAM caching, no disk I/O."""
        if not self.config.data.preload_chunks:
            return
        
        logger.info(f"Starting pure RAM preloading of {len(filepaths)} files for {self.split_name} dataset...")
        
        successful = 0
        failed = 0
        
        # Pure memory operations - no progress bars or frequent logging
        for filepath in filepaths:
            try:
                self.get_features(filepath)
                successful += 1
            except Exception as e:
                failed += 1
                # Only log critical errors, not warnings
                if failed <= 5:  # Log first 5 failures only
                    logger.error(f"Failed to preload {filepath}: {e}")
                elif failed == 6:
                    logger.error("... (suppressing further preload error messages)")
        
        # Single summary log at the end
        cache_size_gb = self.current_size_bytes / 1024 / 1024 / 1024
        
        logger.info(f"✅ {self.split_name.capitalize()} RAM preloading complete:")
        logger.info(f"  Successfully cached: {successful}/{len(filepaths)} files")
        logger.info(f"  Failed: {failed} files")
        logger.info(f"  Cache size: {cache_size_gb:.2f}GB")
        logger.info(f"  Pure RAM caching - no disk I/O performed")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        hit_rate = self.cache_hits / max(1, self.cache_hits + self.cache_misses) * 100
        return {
            "cached_files": len(self.cache),
            "cache_size_gb": self.current_size_bytes / 1024 / 1024 / 1024,
            "hit_rate": hit_rate,
            "total_accesses": self.cache_hits + self.cache_misses
        }


# ========== DATASET CLASSES ==========

class DirectWAVTripletDataset(TorchDataset):
    """
    Triplet dataset that processes WAV files directly with intelligent caching.
    Optimized for cluster environments with abundant RAM.
    """
    
    def __init__(self, split_data: Union[List, Dict], config: ExperimentConfig, 
                 split_name: str = "unknown", normalization_stats: Optional[Tuple[float, float]] = None):
        """
        Initialize dataset with WAV file paths and caching system.
        
        Args:
            split_data: Triplet data with WAV file paths
            config: Experiment configuration
            split_name: Name of the split (train/test) for cache management
            normalization_stats: Optional (mean, std) for dataset-wide normalization
        """
        self.config = config
        self.split_name = split_name
        self.triplets = self._parse_split_data(split_data)
        
        # Initialize high-performance cache with normalization
        self.cache = ClusterFeatureCache(config, split_name, normalization_stats)
        
        logger.info(f"Initialized {split_name} WAV dataset with {len(self.triplets)} triplets")
        
        # Preload all unique files if caching is enabled
        if config.data.preload_chunks:
            unique_files = set()
            for triplet in self.triplets:
                unique_files.update(triplet[:3])  # anchor, positive, negative paths
            
            self.cache.preload_files(list(unique_files))
    
    def _parse_split_data(self, split_data: Union[List, Dict]) -> List[Tuple[str, str, str, str]]:
        """Parse split data containing WAV file paths."""
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
    
    def _process_audio_file(self, filepath: str) -> Dict[str, torch.Tensor]:
        """
        Process single audio file using cache system for optimal performance.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Dictionary with AST features
        """
        try:
            return self.cache.get_features(filepath)
        except Exception as e:
            logger.error(f"Error processing audio file {filepath}: {e}")
            raise AudioProcessingError(f"Failed to process {filepath}: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        return self.cache.get_cache_stats()
    
    def __len__(self) -> int:
        return len(self.triplets)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get triplet with on-the-fly audio processing."""
        logger.debug(f"Dataset __getitem__ called with idx: {idx}, dataset size: {len(self.triplets)}")
        
        if idx >= len(self.triplets):
            raise IndexError(f"Index {idx} out of range for dataset size {len(self.triplets)}")
        
        try:
            anchor_path, positive_path, negative_path, subgenre = self.triplets[idx]
            logger.debug(f"Processing triplet {idx}: {anchor_path[:50]}...")
            
            # Process audio files with error handling
            anchor_input = self._process_audio_file(anchor_path)
            positive_input = self._process_audio_file(positive_path)
            negative_input = self._process_audio_file(negative_path)
            
            result = {
                "anchor_input": anchor_input,
                "positive_input": positive_input,
                "negative_input": negative_input,
                "labels": 0,  # Dummy label for compatibility
                "subgenre": subgenre
            }
            
            logger.debug(f"Successfully created triplet item {idx} with keys: {list(result.keys())}")
            return result
            
        except Exception as e:
            logger.error(f"Error loading triplet {idx}: {e}")
            logger.error(f"Triplet paths: anchor={anchor_path}, positive={positive_path}, negative={negative_path}")
            raise


def safe_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Improved collation function with error handling and validation.
    """
    if not batch:
        raise ValueError("Empty batch provided to collate_fn")
    
    # Debug: Print actual keys in batch to understand the structure
    if len(batch) > 0:
        logger.debug(f"Batch keys: {list(batch[0].keys())}")
        for key in ['anchor_input', 'positive_input', 'negative_input']:
            if key in batch[0]:
                logger.debug(f"{key} type: {type(batch[0][key])}")
                if isinstance(batch[0][key], dict):
                    logger.debug(f"{key} inner keys: {list(batch[0][key].keys())}")
    
    def stack_tensors_safely(key: str) -> torch.Tensor:
        """Stack tensors with validation."""
        try:
            tensors_dict = {}
            
            # Get all inner keys from the first item
            if key not in batch[0]:
                raise ValueError(f"Key '{key}' not found in batch items")
            
            inner_keys = batch[0][key].keys()
            
            for inner_key in inner_keys:
                tensor_list = []
                
                for item in batch:
                    if key not in item:
                        raise ValueError(f"Key '{key}' missing from batch item")
                    
                    if inner_key not in item[key]:
                        raise ValueError(f"Inner key '{inner_key}' missing from {key}")
                    
                    tensor = item[key][inner_key]
                    
                    # Convert lists to tensors if needed
                    if isinstance(tensor, list):
                        tensor = torch.tensor(tensor, dtype=torch.float32)
                    
                    # Validate tensor
                    validate_tensor_shape(tensor, tensor_name=f"{key}.{inner_key}")
                    
                    # Squeeze leading dimensions
                    if tensor.ndim == 3 and tensor.shape[0] == 1:
                        tensor = tensor.squeeze(0)
                    
                    tensor_list.append(tensor)
                
                # Stack tensors
                stacked = torch.stack(tensor_list, dim=0)
                tensors_dict[inner_key] = stacked
            
            return tensors_dict
            
        except Exception as e:
            logger.error(f"Error stacking tensors for key '{key}': {e}")
            raise ValueError(f"Failed to collate {key}: {e}")
    
    try:
        return {
            "anchor_input": stack_tensors_safely("anchor_input"),
            "positive_input": stack_tensors_safely("positive_input"),
            "negative_input": stack_tensors_safely("negative_input"),
            "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        }
    except Exception as e:
        logger.error(f"Collation failed: {e}")
        raise


class ImprovedASTTripletWrapper(nn.Module):
    """
    Improved AST wrapper with configurable architecture and better error handling.
    Identical to the one in AST_Triplet_training.py.
    """
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        
        try:
            logger.info(f"Loading pretrained model: {config.model.pretrained_model}")
            self.ast = ASTModel.from_pretrained(config.model.pretrained_model)
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load AST model: {e}")
        
        # Build configurable projection head
        self.projector = self._build_projection_head()
        self.triplet_margin = config.training.triplet_margin
        
        logger.info(f"Model initialized with projection to {config.model.output_dim}D")
        logger.info(f"Triplet margin: {self.triplet_margin}")
    
    def _build_projection_head(self) -> nn.Module:
        """Build configurable projection head."""
        hidden_sizes = self.config.model.hidden_sizes
        output_dim = self.config.model.output_dim
        dropout_rate = self.config.model.dropout_rate
        
        # Get activation function
        activation_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU()
        }
        activation = activation_map.get(
            self.config.model.activation.lower(), 
            nn.ReLU()
        )
        
        layers = []
        input_size = self.ast.config.hidden_size
        
        for hidden_size in hidden_sizes[:-1]:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                activation,
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            ])
            input_size = hidden_size
        
        # Final layer
        layers.append(nn.Linear(input_size, output_dim))
        
        return nn.Sequential(*layers)
    
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
    
    def forward(self, anchor_input: Dict, positive_input: Dict, 
                negative_input: Dict, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with comprehensive error handling."""
        try:
            # Generate embeddings
            emb_anchor = self.embed(anchor_input)
            emb_positive = self.embed(positive_input)
            emb_negative = self.embed(negative_input)
            
            # Compute distances
            dist_ap = 1 - F.cosine_similarity(emb_anchor, emb_positive, dim=1)
            dist_an = 1 - F.cosine_similarity(emb_anchor, emb_negative, dim=1)
            
            # Triplet loss
            triplet_loss = torch.clamp(dist_ap - dist_an + self.triplet_margin, min=0.0)
            loss = triplet_loss.mean()
            
            # Create logits for compatibility with Trainer
            # Stack distances as [dist_ap, dist_an] for each sample
            logits = torch.stack([dist_ap, dist_an], dim=1)
            
            # Ensure logits are finite and well-formed
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logger.warning("Invalid logits detected, replacing with zeros")
                logits = torch.zeros_like(logits)
            
            return {
                "loss": loss,
                "logits": logits,
                "distances": {
                    "anchor_positive": dist_ap.detach(),
                    "anchor_negative": dist_an.detach()
                }
            }
            
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise


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


# ========== SPLITTING FUNCTIONS ==========

def load_split_data(config: ExperimentConfig) -> Tuple[List, List]:
    """Load train/test splits with support for both old and new formats."""
    data_config = config.data
    
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
    logger.info("Generating train/test splits from WAV directory")
    return generate_wav_triplet_splits(config)


def generate_wav_triplet_splits(config: ExperimentConfig) -> Tuple[List, List]:
    """Generate proper track-level splits from WAV directory, then create triplets."""
    wav_dir = Path(config.data.chunks_dir)
    test_ratio = config.data.test_split_ratio
    
    if not wav_dir.exists():
        raise FileNotFoundError(f"WAV directory not found: {wav_dir}")
    
    logger.info("Generating track-level splits from WAV files to prevent data leakage...")
    
    # Step 1: Organize files by subgenre and track (handle nested directory structure)
    subgenre_tracks = defaultdict(dict)  # {subgenre: {track_name: [audio_files]}}
    audio_extensions = (".mp3", ".wav", ".flac", ".ogg")
    
    # Recursively search for audio files in nested directory structure
    for genre_dir in sorted(wav_dir.iterdir()):
        if not genre_dir.is_dir():
            continue
            
        logger.info(f"Scanning genre directory: {genre_dir.name}")
        
        # Check for subgenre directories within genre directory
        for subgenre_dir in sorted(genre_dir.iterdir()):
            if not subgenre_dir.is_dir():
                continue
                
            subgenre = f"{genre_dir.name}/{subgenre_dir.name}"  # Full path as subgenre name
            logger.info(f"  Scanning subgenre: {subgenre}")
            
            files = [f for f in subgenre_dir.iterdir() if f.suffix.lower() in audio_extensions]
            
            if not files:
                logger.warning(f"    No audio files found in {subgenre}")
                continue
            
            logger.info(f"    Found {len(files)} audio files")
            
            # Group files by track name (assuming one file per track)
            tracks = {}
            for file in files:
                track_name = file.stem  # Use filename without extension as track name
                tracks[track_name] = [str(file)]  # List for consistency with preprocessing
            
            # Only keep tracks with valid files
            valid_tracks = {name: files for name, files in tracks.items() if files}
            
            if len(valid_tracks) < 4:  # Need at least 4 tracks to split meaningfully
                logger.warning(f"    Subgenre {subgenre} has only {len(valid_tracks)} tracks, skipping")
                continue
                
            subgenre_tracks[subgenre] = valid_tracks
            logger.info(f"    ✅ Subgenre {subgenre}: {len(valid_tracks)} tracks")
    
    # Step 2: Split tracks (not triplets) into train/test sets
    train_tracks = defaultdict(dict)  # {subgenre: {track_name: [audio_files]}}
    test_tracks = defaultdict(dict)   # {subgenre: {track_name: [audio_files]}}
    
    for subgenre, tracks in subgenre_tracks.items():
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
    train_triplets = _generate_triplets_from_wav_tracks(train_tracks, "train", config)
    
    logger.info("Generating test triplets from test tracks...")
    test_triplets = _generate_triplets_from_wav_tracks(test_tracks, "test", config)
    
    # Quick test run: use only 5% of data for rapid error detection
    if config.data.quick_test_run:
        logger.info("🧪 QUICK TEST MODE: Using only 5% of data for rapid error detection")
        train_sample_size = max(10, len(train_triplets) // 20)  # 5% or minimum 10
        test_sample_size = max(5, len(test_triplets) // 20)     # 5% or minimum 5
        
        train_triplets = train_triplets[:train_sample_size]
        test_triplets = test_triplets[:test_sample_size]
        
        logger.info(f"  Reduced to {len(train_triplets)} train triplets, {len(test_triplets)} test triplets")
    
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


def _generate_triplets_from_wav_tracks(subgenre_tracks: Dict[str, Dict[str, List[str]]], 
                                      split_name: str, config: ExperimentConfig) -> List[Tuple[str, str, str, str]]:
    """Generate triplets from a given set of WAV tracks (train or test)."""
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
        negative_files_pool = []
        for other_subgenre in other_subgenres:
            for other_track, other_files in subgenre_tracks[other_subgenre].items():
                negative_files_pool.extend(other_files)
        
        if not negative_files_pool:
            logger.warning(f"No negative samples available for subgenre {subgenre} in {split_name}")
            continue
        
        triplets = []
        
        # Generate triplets within this subgenre
        for anchor_track in track_names:
            anchor_files = tracks[anchor_track]
            
            # Positive candidates: different tracks in SAME subgenre
            positive_tracks = [t for t in track_names if t != anchor_track]
            
            # Use ALL files as anchors (for WAV, typically one file per track)
            for anchor_file in anchor_files:
                
                # Sample positive tracks (limit to avoid explosion)
                max_positives = min(len(positive_tracks), config.data.max_positive_tracks)
                positive_sample = random.sample(positive_tracks, max_positives)
                
                for positive_track in positive_sample:
                    positive_files = tracks[positive_track]
                    
                    # Generate configurable triplets per positive track
                    triplets_per_positive = min(config.data.triplets_per_positive_track, len(positive_files))
                    
                    for _ in range(triplets_per_positive):
                        positive_file = random.choice(positive_files)
                        negative_file = random.choice(negative_files_pool)
                        
                        triplet = (anchor_file, positive_file, negative_file, subgenre)
                        triplets.append(triplet)
        
        random.shuffle(triplets)
        all_triplets.extend(triplets)
        
        logger.info(f"{split_name.capitalize()} - Subgenre {subgenre}: {len(triplets)} triplets from {len(track_names)} tracks")
    
    return all_triplets


def estimate_total_ram_requirements(config: ExperimentConfig, num_train_tracks: int, 
                                   num_test_tracks: int, gpu_count: int = 1) -> Dict[str, float]:
    """
    Comprehensive RAM estimation for full training run on cluster.
    Returns generous estimates (better too high than too low).
    
    Args:
        config: Experiment configuration
        num_train_tracks: Number of training tracks
        num_test_tracks: Number of test tracks
        
    Returns:
        Dictionary with RAM estimates in GB
    """
    
    # Base memory per track (AST features: 1024 x 768 x 4 bytes per chunk, 3 chunks per track)
    ast_features_per_track_mb = (1024 * 768 * 4 * 3) / (1024 * 1024)  # ~9.3 MB per track
    
    # Dataset caching
    train_cache_gb = (num_train_tracks * ast_features_per_track_mb) / 1024
    test_cache_gb = (num_test_tracks * ast_features_per_track_mb) / 1024
    total_dataset_cache_gb = train_cache_gb + test_cache_gb
    
    # Model memory (AST + projection head) - multiplied by GPU count for DataParallel
    # AST model: ~87M parameters, projection head: ~1M parameters 
    model_params = 88_000_000  # Conservative estimate
    model_memory_gb = (model_params * 4 * 2 * gpu_count) / (1024**3)  # 4 bytes per param, 2x for gradients, Nx for GPUs
    
    # Training batch memory (worst case scenario)
    batch_size = config.training.batch_size
    gradient_accumulation = config.training.gradient_accumulation_steps
    effective_batch_size = batch_size * gradient_accumulation
    
    # Memory per batch item: 3 triplet members × AST features + intermediate activations
    batch_item_memory_mb = 3 * ast_features_per_track_mb * 2  # 2x for intermediate activations
    batch_memory_gb = (effective_batch_size * batch_item_memory_mb) / 1024
    
    # Optimizer state (AdamW: 2x model params for momentum + variance)
    optimizer_memory_gb = model_memory_gb * 2
    
    # PyTorch overhead and caching
    pytorch_overhead_gb = 2.0  # Conservative estimate for PyTorch internal caching
    
    # HuggingFace Trainer overhead
    trainer_overhead_gb = 1.0  # Logging, checkpointing, etc.
    
    # Operating system and other processes
    system_overhead_gb = 4.0  # OS, monitoring, etc.
    
    # CUDA context and kernels (if using GPU)
    cuda_overhead_gb = 2.0 if torch.cuda.is_available() else 0.0
    
    # Feature extraction overhead during preloading
    feature_extraction_overhead_gb = 1.0  # Temporary buffers during preprocessing
    
    # Safety margin (25% of total)
    base_total = (total_dataset_cache_gb + model_memory_gb + batch_memory_gb + 
                 optimizer_memory_gb + pytorch_overhead_gb + trainer_overhead_gb + 
                 system_overhead_gb + cuda_overhead_gb + feature_extraction_overhead_gb)
    
    safety_margin_gb = base_total * 0.25
    
    total_estimated_gb = base_total + safety_margin_gb
    
    return {
        "dataset_cache": total_dataset_cache_gb,
        "train_cache": train_cache_gb,
        "test_cache": test_cache_gb,
        "model_memory": model_memory_gb,
        "batch_memory": batch_memory_gb,
        "optimizer_memory": optimizer_memory_gb,
        "pytorch_overhead": pytorch_overhead_gb,
        "trainer_overhead": trainer_overhead_gb,
        "system_overhead": system_overhead_gb,
        "cuda_overhead": cuda_overhead_gb,
        "feature_extraction_overhead": feature_extraction_overhead_gb,
        "safety_margin": safety_margin_gb,
        "total_estimated": total_estimated_gb,
        "recommended_request": max(total_estimated_gb * 1.1, 32.0)  # At least 32GB, 10% extra buffer
    }


def save_model_and_artifacts(model: ImprovedASTTripletWrapper, config: ExperimentConfig,
                           train_data: Any, test_data: Any) -> str:
    """Save model, configuration, and metadata with error handling."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"fully_trained_models_{timestamp}")
    save_dir.mkdir(exist_ok=True)
    
    try:
        # Save model weights
        model_path = save_dir / "model.safetensors"
        save_file(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save configuration
        config_path = save_dir / "config.json"
        config.save(config_path)
        logger.info(f"Configuration saved to {config_path}")
        
        # Save split information
        splits_path = save_dir / "splits.json"
        splits_data = {
            "train_split": train_data,
            "test_split": test_data,
            "timestamp": timestamp
        }
        with open(splits_path, 'w') as f:
            json.dump(splits_data, f, indent=2)
        
        logger.info(f"Training artifacts saved to {save_dir}")
        return str(save_dir)
        
    except Exception as e:
        logger.error(f"Error saving model artifacts: {e}")
        raise


def main():
    """Main training function with comprehensive error handling."""
    parser = argparse.ArgumentParser(description="AST Triplet Loss Training - Cluster Optimized WAV Processing")
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
        
        # Enable debug logging for quick test runs
        if hasattr(config.data, 'quick_test_run') and config.data.quick_test_run:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("🧪 Quick test run detected - enabling debug logging")
        config.validate()
        
        logger.info(f"Experiment: {config.experiment_name}")
        if config.description:
            logger.info(f"Description: {config.description}")
        
        logger.info("🎵 CLUSTER-OPTIMIZED WAV PROCESSING MODE")
        logger.info("Processing audio files directly from WAV directory")
        
        # Set random seeds for reproducibility
        set_random_seeds(config.training.seed)
        
        # Load data splits
        logger.info("Loading data splits...")
        train_data, test_data = load_split_data(config)
        
        # Estimate RAM requirements based on actual data and GPU setup
        num_train_tracks = len(set(triplet[0] for triplet in train_data))  # Count unique anchor files
        num_test_tracks = len(set(triplet[0] for triplet in test_data))
        expected_gpu_count = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        
        ram_estimates = estimate_total_ram_requirements(config, num_train_tracks, num_test_tracks, expected_gpu_count)
        
        logger.info("🧠 CLUSTER RAM REQUIREMENTS ESTIMATION:")
        logger.info(f"  Dataset cache: {ram_estimates['dataset_cache']:.1f}GB")
        logger.info(f"    - Train cache: {ram_estimates['train_cache']:.1f}GB ({num_train_tracks} tracks)")
        logger.info(f"    - Test cache: {ram_estimates['test_cache']:.1f}GB ({num_test_tracks} tracks)")
        logger.info(f"  Model + gradients: {ram_estimates['model_memory']:.1f}GB")
        logger.info(f"  Batch processing: {ram_estimates['batch_memory']:.1f}GB")
        logger.info(f"  Optimizer state: {ram_estimates['optimizer_memory']:.1f}GB")
        logger.info(f"  System overhead: {ram_estimates['system_overhead']:.1f}GB")
        logger.info(f"  Safety margin: {ram_estimates['safety_margin']:.1f}GB")
        logger.info(f"  📊 TOTAL ESTIMATED: {ram_estimates['total_estimated']:.1f}GB")
        logger.info(f"  🎯 RECOMMENDED REQUEST: {ram_estimates['recommended_request']:.0f}GB")
        logger.info("")
        
        # Save splits immediately to preserve split information
        logger.info("Saving train/test splits...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        splits_dir = Path(f"splits_{timestamp}")
        splits_dir.mkdir(exist_ok=True)
        
        splits_path = splits_dir / "splits.json"
        splits_data = {
            "train_split": train_data,
            "test_split": test_data,
            "timestamp": timestamp,
            "config_summary": {
                "experiment_name": config.experiment_name,
                "wav_processing": True,
                "test_split_ratio": config.data.test_split_ratio
            }
        }
        with open(splits_path, 'w') as f:
            json.dump(splits_data, f, indent=2)
        logger.info(f"Splits saved to {splits_path}")
        
        # Compute dataset-wide normalization statistics
        logger.info("Computing dataset-wide normalization statistics...")
        all_wav_files = set()
        for triplet in train_data:
            all_wav_files.update(triplet[:3])  # anchor, positive, negative paths
        for triplet in test_data:
            all_wav_files.update(triplet[:3])
        
        logger.info(f"Found {len(all_wav_files)} unique audio files for normalization")
        normalization_stats = compute_dataset_normalization_stats(list(all_wav_files), config)
        
        # Create datasets with caching and normalization
        logger.info("Creating WAV datasets with intelligent caching and dataset normalization...")
        train_dataset = DirectWAVTripletDataset(train_data, config, "train", normalization_stats)
        test_dataset = DirectWAVTripletDataset(test_data, config, "test", normalization_stats)
        
        # Log cache performance
        train_stats = train_dataset.get_cache_stats()
        test_stats = test_dataset.get_cache_stats()
        
        total_cache_gb = train_stats["cache_size_gb"] + test_stats["cache_size_gb"]
        logger.info(f"📊 Cache Performance Summary:")
        logger.info(f"  Train cache: {train_stats['cached_files']} files, {train_stats['cache_size_gb']:.2f}GB")
        logger.info(f"  Test cache: {test_stats['cached_files']} files, {test_stats['cache_size_gb']:.2f}GB")
        logger.info(f"  Total cached: {total_cache_gb:.2f}GB")
        
        # Initialize model with multi-GPU support
        logger.info("Initializing model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        logger.info(f"Using device: {device}")
        logger.info(f"Available GPUs: {gpu_count}")
        
        if gpu_count > 1:
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        model = ImprovedASTTripletWrapper(config).to(device)
        
        # Enable DataParallel for multi-GPU training (unless forced to single GPU)
        if gpu_count > 1 and not getattr(config.training, 'force_single_gpu', False):
            logger.info(f"🚀 Enabling DataParallel training across {gpu_count} GPUs")
            model = torch.nn.DataParallel(model)
            logger.info(f"  Effective batch size will be: {config.training.batch_size} × {gpu_count} = {config.training.batch_size * gpu_count}")
        else:
            if getattr(config.training, 'force_single_gpu', False) and gpu_count > 1:
                logger.info("🔄 Single GPU training mode (forced by config)")
                # Optionally restrict to GPU 0 only
                device = torch.device("cuda:0")
                model = model.to(device)
            else:
                logger.info("🔄 Single GPU training mode")
        
        # Setup training arguments with clean logging
        training_args_dict = {
            "output_dir": config.paths.model_output_dir,
            "eval_strategy": config.training.eval_strategy,
            "save_strategy": config.training.save_strategy,
            "learning_rate": config.training.learning_rate,
            "per_device_train_batch_size": config.training.batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "num_train_epochs": config.training.epochs,
            "weight_decay": config.training.weight_decay,
            "warmup_ratio": config.training.warmup_ratio,
            "logging_steps": config.training.logging_steps,
            "dataloader_num_workers": config.training.num_workers,
            "dataloader_pin_memory": config.training.pin_memory,
            "report_to": "none",  # Disable wandb/tensorboard
            "logging_first_step": False,  # Don't log first step
            "disable_tqdm": True,  # No Progress bars
            "log_level": "warning",  # Reduce HF Trainer's internal logging
            "seed": config.training.seed,
            "data_seed": config.training.seed,
        }
        
        # Add optional parameters if they exist in config
        if hasattr(config.training, 'save_steps') and config.training.save_steps:
            training_args_dict["save_steps"] = config.training.save_steps
        if hasattr(config.training, 'eval_steps') and config.training.eval_steps:
            training_args_dict["eval_steps"] = config.training.eval_steps
        
        training_args = TrainingArguments(**training_args_dict)
        
        # Initialize trainer with custom callback
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=safe_collate_fn,
            compute_metrics=compute_metrics,
            callbacks=[CleanLoggingCallback()],
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save model and artifacts
        logger.info("Saving model and artifacts...")
        save_dir = save_model_and_artifacts(model, config, train_data, test_data)
        
        logger.info(f"Training completed successfully! Results saved to: {save_dir}")
        
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


if __name__ == "__main__":
    exit(main())