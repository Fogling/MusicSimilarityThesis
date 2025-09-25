#!/usr/bin/env python3
"""
Improved AST Feature Preprocessing with proper error handling and configuration management.

This refactored version addresses all issues found in the original:
- Full dataset statistics computation for accurate normalization
- Comprehensive error handling and validation
- Configuration management with type safety
- Proper logging and progress tracking
- Input validation and robustness
- Performance optimizations
"""

import os
import re
import json
import logging
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Generator, Set
from dataclasses import dataclass
from collections import defaultdict
import time

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import ASTFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Raised when preprocessing fails."""
    pass


class AudioLoadError(Exception):
    """Raised when audio loading fails."""
    pass


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing."""
    # Input/Output
    wav_dir: str = "WAV"
    output_dir: str = "precomputed_AST"

    # Audio processing
    chunk_duration: float = 10.24  # seconds - optimized for AST max_length 1024
    max_chunks_per_file: int = 3  # Reduced from 9 to avoid pseudo-duplicates
    target_sample_rate: int = 16000
    audio_extensions: Tuple[str, ...] = (".mp3", ".wav", ".flac", ".ogg")

    # Chunk sampling strategy
    chunk_strategy: str = "random"  # "sequential", "random", "spaced" - random is best for diversity
    random_seed: Optional[int] = 42  # For reproducible random sampling
    single_chunk: bool = False  # Average 3 chunks into 1 chunk

    # Feature extraction
    extractor_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    padding_strategy: str = "max_length"

    # Processing
    max_filename_length: int = 100
    batch_processing: bool = False
    num_workers: int = 1

    # Validation
    validate_audio: bool = True
    min_audio_length: float = 1.0  # seconds
    max_audio_length: float = 600.0  # seconds

    # K-Fold cross-validation
    enable_kfold: bool = False  # Enable k-fold preprocessing
    k_folds: int = 5  # Number of folds
    kfold_partition_seed: int = 42  # Seed for reproducible partitioning
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.chunk_duration <= 0:
            raise ValueError("Chunk duration must be positive")
        if self.target_sample_rate <= 0:
            raise ValueError("Target sample rate must be positive")
        if self.max_chunks_per_file <= 0:
            raise ValueError("Max chunks per file must be positive")
        if self.chunk_strategy not in ["sequential", "random", "spaced"]:
            raise ValueError(f"Invalid chunk strategy: {self.chunk_strategy}. Must be 'sequential', 'random', or 'spaced'")
        if not Path(self.wav_dir).exists():
            raise ValueError(f"WAV directory does not exist: {self.wav_dir}")
        if self.enable_kfold and self.k_folds < 2:
            raise ValueError("K-folds must be at least 2 for cross-validation")
        if self.enable_kfold and self.k_folds > 10:
            raise ValueError("K-folds should not exceed 10 for practical purposes")

def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """
    Sanitize filename by removing invalid characters and limiting length.
    
    Args:
        filename: Original filename
        max_length: Maximum length for the sanitized filename
        
    Returns:
        Sanitized filename
    """
    if not filename:
        raise ValueError("Filename cannot be empty")
    
    # Remove invalid characters and replace whitespace
    sanitized = re.sub(r'[\\/:*?"<>|\s]+', '_', filename.strip())
    
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    if not sanitized:
        sanitized = "unnamed_file"
    
    return sanitized[:max_length]

def list_audio_files(root: str, config: PreprocessingConfig) -> List[Tuple[str, str, str]]:
    """
    Returns list of (subgenre, filepath, basename) for all audio files
    under <root>/<genre>/<subgenre>/*.{exts}
    
    Args:
        root: Root directory path
        config: Preprocessing configuration
        
    Returns:
        List of (subgenre, filepath, basename) tuples
    """
    if not os.path.exists(root):
        raise FileNotFoundError(f"Root directory not found: {root}")
    
    items = []
    exts = config.audio_extensions
    
    try:
        for genre in sorted(os.listdir(root)):
            genre_path = os.path.join(root, genre)
            if not os.path.isdir(genre_path):
                continue
                
            logger.info(f"Processing genre: {genre}")
            
            for subgenre in sorted(os.listdir(genre_path)):
                subgenre_path = os.path.join(genre_path, subgenre)
                if not os.path.isdir(subgenre_path):
                    continue
                
                # Count files in subgenre
                audio_files = [
                    f for f in os.listdir(subgenre_path) 
                    if f.lower().endswith(exts)
                ]
                
                if not audio_files:
                    logger.warning(f"No audio files found in {subgenre_path}")
                    continue
                
                logger.info(f"  Subgenre '{subgenre}': {len(audio_files)} files")
                
                for audio_file in sorted(audio_files):
                    full_path = os.path.join(subgenre_path, audio_file)
                    basename = os.path.splitext(audio_file)[0]
                    items.append((subgenre, full_path, basename))
                    
    except Exception as e:
        raise PreprocessingError(f"Error scanning audio files: {e}")
    
    if not items:
        raise PreprocessingError(f"No audio files found in {root}")
    
    logger.info(f"Found {len(items)} total audio files across all subgenres")
    return items

def create_track_based_kfold_partitions(items: List[Tuple[str, str, str]],
                                       k: int, seed: int) -> Dict[str, List[List[str]]]:
    """
    Create K-fold partitions at the track level for each subgenre.

    Args:
        items: List of (subgenre, filepath, basename) tuples
        k: Number of folds
        seed: Random seed for reproducible partitioning

    Returns:
        Dictionary mapping subgenre -> list of K track partitions
        Format: {subgenre: [[tracks_fold0], [tracks_fold1], ..., [tracks_fold(k-1)]]}
    """
    logger.info(f"Creating {k}-fold partitions for track-level cross-validation")

    # Set seed for reproducible partitioning
    random.seed(seed)

    # Organize unique tracks by subgenre
    subgenre_tracks = defaultdict(set)  # {subgenre: {track_names}}

    for subgenre, filepath, basename in items:
        # Extract track name (remove any existing chunk suffix if present)
        track_name = re.sub(r'_chunk\d+$', '', basename)
        subgenre_tracks[subgenre].add(track_name)

    # Create K partitions for each subgenre
    kfold_partitions = {}

    for subgenre, track_set in subgenre_tracks.items():
        tracks = sorted(list(track_set))  # Convert to sorted list for reproducibility

        if len(tracks) < k:
            logger.warning(f"Subgenre {subgenre} has only {len(tracks)} tracks, "
                          f"less than K={k} folds. Some folds will be empty.")

        # Shuffle tracks with partition seed for reproducibility
        shuffled_tracks = tracks.copy()
        random.shuffle(shuffled_tracks)

        # Split into K approximately equal partitions
        partitions = [[] for _ in range(k)]
        for i, track in enumerate(shuffled_tracks):
            fold_idx = i % k
            partitions[fold_idx].append(track)

        kfold_partitions[subgenre] = partitions

        # Log partition sizes
        partition_sizes = [len(p) for p in partitions]
        logger.info(f"Subgenre {subgenre}: {len(tracks)} tracks -> K-fold partitions: {partition_sizes}")

    return kfold_partitions

def get_fold_track_splits(kfold_partitions: Dict[str, List[List[str]]],
                         fold_idx: int, k: int) -> Tuple[Set[str], Set[str]]:
    """
    Get train/test track sets for a specific fold across all subgenres.

    Args:
        kfold_partitions: Output from create_track_based_kfold_partitions()
        fold_idx: Current fold index (0 to k-1)
        k: Number of folds

    Returns:
        Tuple of (train_tracks, test_tracks) where each is a set of track names
    """
    if not (0 <= fold_idx < k):
        raise ValueError(f"fold_idx must be between 0 and {k-1}, got {fold_idx}")

    train_tracks = set()
    test_tracks = set()

    for subgenre, partitions in kfold_partitions.items():
        # Test set: partition at fold_idx
        test_tracks.update(partitions[fold_idx])

        # Train set: all other partitions combined
        for i in range(k):
            if i != fold_idx:
                train_tracks.update(partitions[i])

    logger.info(f"Fold {fold_idx}: {len(train_tracks)} train tracks, {len(test_tracks)} test tracks")
    return train_tracks, test_tracks

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

def clear_resampler_cache():
    """Clear the global resampler cache to free memory."""
    global _resampler_cache
    _resampler_cache.clear()

def load_and_resample(path: str, target_sr: int, config: PreprocessingConfig) -> torch.Tensor:
    """
    Load audio file and convert to mono with target sample rate.
    
    Args:
        path: Path to audio file
        target_sr: Target sample rate
        config: Preprocessing configuration
        
    Returns:
        Mono audio tensor at target sample rate
        
    Raises:
        AudioLoadError: If audio loading or processing fails
    """
    try:
        # Load audio
        waveform, original_sr = torchaudio.load(path)
        
        # Validate audio
        if config.validate_audio:
            duration = waveform.shape[-1] / original_sr
            if duration < config.min_audio_length:
                raise AudioLoadError(f"Audio too short: {duration:.2f}s < {config.min_audio_length}s")
            if duration > config.max_audio_length:
                logger.warning(f"Audio very long: {duration:.2f}s, consider splitting")
        
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
            raise AudioLoadError("Audio contains NaN values")
        if torch.isinf(result).any():
            raise AudioLoadError("Audio contains infinite values")
        
        return result
        
    except Exception as e:
        if isinstance(e, AudioLoadError):
            raise
        raise AudioLoadError(f"Failed to load audio from {path}: {e}")

def chunk_waveform(waveform: torch.Tensor, sr: int, duration_s: int,
                  max_chunks: int = 9, overlap: float = 0.0,
                  strategy: str = "sequential", seed: Optional[int] = None) -> Generator[torch.Tensor, None, None]:
    """
    Yield up to max_chunks chunks using different sampling strategies.
    Only generates chunks that actually fit within the available audio duration.

    Args:
        waveform: Input audio waveform
        sr: Sample rate
        duration_s: Duration of each chunk in seconds
        max_chunks: Maximum number of chunks to generate
        overlap: Overlap fraction between chunks (0.0 = no overlap, 0.5 = 50% overlap)
        strategy: Sampling strategy - "sequential", "random", or "spaced"
        seed: Random seed for reproducible sampling

    Yields:
        Audio chunks as tensors (only chunks that actually exist)
    """
    if duration_s <= 0:
        raise ValueError("Chunk duration must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("Overlap must be between 0 and 1 (exclusive)")

    chunk_size = int(duration_s * sr)
    total_length = waveform.shape[0]

    # Check if we have enough space for any chunks
    if total_length < chunk_size:
        audio_duration_s = total_length / sr
        min_required_s = chunk_size / sr
        logger.warning(f"Audio too short for chunking: {audio_duration_s:.1f}s < {min_required_s:.1f}s required. Skipping.")
        return

    # Calculate maximum possible non-overlapping chunks given total length
    max_possible_chunks = max(1, int(total_length / chunk_size))
    actual_max_chunks = min(max_chunks, max_possible_chunks)

    if actual_max_chunks < max_chunks:
        logger.info(f"Audio allows only {actual_max_chunks} chunks instead of requested {max_chunks}")

    # Get chunk start positions based on strategy
    start_positions = _get_chunk_positions(
        total_length, chunk_size, actual_max_chunks, overlap, strategy, seed
    )

    chunks_generated = 0
    for start in start_positions:
        if chunks_generated >= actual_max_chunks:
            break

        end = start + chunk_size
        if end > total_length:
            continue

        chunk = waveform[start:end]

        # Validate chunk
        if torch.isnan(chunk).any() or torch.isinf(chunk).any():
            logger.warning(f"Skipping chunk {chunks_generated} due to invalid values")
            continue

        yield chunk
        chunks_generated += 1


def _get_chunk_positions(total_length: int, chunk_size: int, max_chunks: int,
                        overlap: float, strategy: str, seed: Optional[int] = None) -> List[int]:
    """
    Get random, non-overlapping chunk start positions from the full audio duration.

    Args:
        total_length: Total audio length in samples
        chunk_size: Chunk size in samples
        max_chunks: Maximum number of chunks
        overlap: Overlap fraction (ignored)
        strategy: Sampling strategy (only "random" supported)
        seed: Random seed

    Returns:
        List of start positions for non-overlapping chunks
    """
    if strategy != "random":
        raise ValueError(f"Only 'random' strategy is supported, got: {strategy}")

    # Calculate valid positions where a chunk can start (ensuring chunk fits)
    max_start_pos = total_length - chunk_size
    if max_start_pos < 0:
        return []

    if seed is not None:
        np.random.seed(seed)

    # Generate non-overlapping random positions
    positions = []
    used_ranges = []  # (start, end) tuples for already placed chunks

    for _ in range(max_chunks):
        # Find valid positions that don't overlap with existing chunks
        valid_positions = []
        for pos in range(0, max_start_pos + 1):
            chunk_end = pos + chunk_size
            # Check if this position overlaps with any existing chunk
            overlaps = any(not (chunk_end <= used_start or pos >= used_end)
                          for used_start, used_end in used_ranges)
            if not overlaps:
                valid_positions.append(pos)

        if not valid_positions:
            break  # No more non-overlapping positions available

        # Randomly select from valid positions
        chosen_pos = np.random.choice(valid_positions)
        positions.append(chosen_pos)
        used_ranges.append((chosen_pos, chosen_pos + chunk_size))

    return sorted(positions)

class FullDatasetStatsCollector:
    """
    Collects all feature values from the entire dataset to compute exact statistics.

    This class stores every feature value from every chunk to enable precise
    mean and standard deviation computation for dataset normalization.
    """

    def __init__(self):
        self.all_values = []

    def update_batch(self, values: torch.Tensor) -> None:
        """Collect all values from a batch."""
        if values.numel() == 0:
            return
        # Flatten and convert to list for collection
        self.all_values.extend(values.flatten().tolist())

    def compute_stats(self) -> Tuple[float, float]:
        """Compute final mean and std from all collected values."""
        if not self.all_values:
            raise ValueError("No values collected for statistics computation")

        # Convert to numpy for efficient computation
        all_values_array = np.array(self.all_values)
        mean_val = float(np.mean(all_values_array))
        std_val = float(np.std(all_values_array))

        return mean_val, std_val

    @property
    def count(self) -> int:
        """Get total number of values collected."""
        return len(self.all_values)

    def get_current_count(self) -> int:
        """Get current number of values collected (for progress display)."""
        return len(self.all_values)


def compute_stats_pass_for_tracks(config: PreprocessingConfig, train_tracks: Set[str]) -> Tuple[float, float]:
    """
    Pass 1: compute mean/std over *pre-normalized* AST inputs using only training tracks.

    Args:
        config: Preprocessing configuration
        train_tracks: Set of track names to use for statistics computation (train partition only)

    Returns:
        Tuple of (mean, std) values

    Raises:
        PreprocessingError: If statistics computation fails
    """
    logger.info(f"Pass 1/2: Computing dataset statistics from {len(train_tracks)} training tracks...")

    try:
        # Initialize extractor without normalization
        extractor = ASTFeatureExtractor.from_pretrained(config.extractor_model)
        if hasattr(extractor, "do_normalize"):
            extractor.do_normalize = False
        else:
            logger.warning("Extractor does not have do_normalize attribute")

        # Initialize full dataset stats collector
        stats_collector = FullDatasetStatsCollector()

        # Get all audio files
        items = list_audio_files(config.wav_dir, config)

        # Filter items to only include training tracks
        train_items = []
        for subgenre, filepath, basename in items:
            # Extract track name (remove any existing chunk suffix if present)
            track_name = re.sub(r'_chunk\d+$', '', basename)
            if track_name in train_tracks:
                train_items.append((subgenre, filepath, basename))

        logger.info(f"Computing statistics from {len(train_items)} training files out of {len(items)} total files")

        # Process files with progress bar
        successful_files = 0
        total_chunks = 0

        with tqdm(train_items, desc="Computing statistics (train only)") as pbar:
            for subgenre, filepath, basename in pbar:
                try:
                    # Load and resample audio
                    waveform = load_and_resample(filepath, config.target_sample_rate, config)

                    # Process chunks
                    file_chunks = 0
                    for chunk in chunk_waveform(
                        waveform,
                        config.target_sample_rate,
                        config.chunk_duration,
                        config.max_chunks_per_file,
                        strategy=config.chunk_strategy,
                        seed=config.random_seed
                    ):
                        try:
                            # Extract features
                            inputs = extractor(
                                chunk.numpy(),
                                sampling_rate=config.target_sample_rate,
                                return_tensors="pt",
                                padding=config.padding_strategy
                            )

                            if "input_values" not in inputs:
                                logger.warning(f"No input_values in extractor output for {filepath}")
                                continue

                            # Update statistics
                            features = inputs["input_values"][0].float()
                            stats_collector.update_batch(features)

                            file_chunks += 1
                            total_chunks += 1

                        except Exception as e:
                            logger.warning(f"Error processing chunk from {filepath}: {e}")
                            continue

                    if file_chunks > 0:
                        successful_files += 1
                        pbar.set_postfix({
                            'files': successful_files,
                            'chunks': total_chunks,
                            'values': f'{stats_collector.count:,}'
                        })

                except AudioLoadError as e:
                    logger.warning(f"Skipping {filepath}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error processing {filepath}: {e}")
                    continue

        # Validate results
        if stats_collector.count == 0:
            raise PreprocessingError("No valid audio chunks found for statistics computation")

        # Compute final statistics from all collected values
        mean_val, std_val = stats_collector.compute_stats()
        std_val = std_val if std_val > 0 else 1.0

        logger.info(f"Statistics computed from {successful_files} training files, {total_chunks} chunks")
        logger.info(f"Total values collected: {stats_collector.count:,}")
        logger.info(f"Training dataset statistics — mean: {mean_val:.6f}, std: {std_val:.6f}")

        return mean_val, std_val

    except Exception as e:
        if isinstance(e, PreprocessingError):
            raise
        raise PreprocessingError(f"Statistics computation failed: {e}")

def compute_stats_pass(config: PreprocessingConfig) -> Tuple[float, float]:
    """
    Pass 1: compute mean/std over *pre-normalized* AST inputs using full dataset collection.

    Args:
        config: Preprocessing configuration

    Returns:
        Tuple of (mean, std) values

    Raises:
        PreprocessingError: If statistics computation fails
    """
    logger.info("Pass 1/2: Computing dataset statistics (full dataset pass)...")
    
    try:
        # Initialize extractor without normalization
        extractor = ASTFeatureExtractor.from_pretrained(config.extractor_model)
        if hasattr(extractor, "do_normalize"):
            extractor.do_normalize = False
        else:
            logger.warning("Extractor does not have do_normalize attribute")
        
        # Initialize full dataset stats collector
        stats_collector = FullDatasetStatsCollector()
        
        # Get all audio files
        items = list_audio_files(config.wav_dir, config)
        
        # Process files with progress bar
        successful_files = 0
        total_chunks = 0
        
        with tqdm(items, desc="Computing statistics") as pbar:
            for subgenre, filepath, basename in pbar:
                try:
                    # Load and resample audio
                    waveform = load_and_resample(filepath, config.target_sample_rate, config)
                    
                    # Process chunks
                    file_chunks = 0
                    for chunk in chunk_waveform(
                        waveform, 
                        config.target_sample_rate, 
                        config.chunk_duration, 
                        config.max_chunks_per_file,
                        strategy=config.chunk_strategy,
                        seed=config.random_seed
                    ):
                        try:
                            # Extract features
                            inputs = extractor(
                                chunk.numpy(),
                                sampling_rate=config.target_sample_rate,
                                return_tensors="pt",
                                padding=config.padding_strategy
                            )
                            
                            if "input_values" not in inputs:
                                logger.warning(f"No input_values in extractor output for {filepath}")
                                continue
                            
                            # Update statistics
                            features = inputs["input_values"][0].float()
                            stats_collector.update_batch(features)
                            
                            file_chunks += 1
                            total_chunks += 1
                            
                        except Exception as e:
                            logger.warning(f"Error processing chunk from {filepath}: {e}")
                            continue
                    
                    if file_chunks > 0:
                        successful_files += 1
                        pbar.set_postfix({
                            'files': successful_files,
                            'chunks': total_chunks,
                            'values': f'{stats_collector.count:,}'
                        })
                    
                except AudioLoadError as e:
                    logger.warning(f"Skipping {filepath}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error processing {filepath}: {e}")
                    continue
        
        # Validate results
        if stats_collector.count == 0:
            raise PreprocessingError("No valid audio chunks found for statistics computation")

        # Compute final statistics from all collected values
        mean_val, std_val = stats_collector.compute_stats()
        std_val = std_val if std_val > 0 else 1.0
        
        logger.info(f"Statistics computed from {successful_files} files, {total_chunks} chunks")
        logger.info(f"Total values collected: {stats_collector.count:,}")
        logger.info(f"Dataset statistics — mean: {mean_val:.6f}, std: {std_val:.6f}")
        
        return mean_val, std_val
        
    except Exception as e:
        if isinstance(e, PreprocessingError):
            raise
        raise PreprocessingError(f"Statistics computation failed: {e}")

def average_chunks_features(chunk_features_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Average multiple chunk features into a single representation.

    Args:
        chunk_features_list: List of feature tensors from chunks

    Returns:
        Averaged feature tensor
    """
    if not chunk_features_list:
        raise ValueError("Cannot average empty chunk list")

    if len(chunk_features_list) == 1:
        return chunk_features_list[0]

    # Stack and average along the batch dimension
    stacked_features = torch.stack(chunk_features_list, dim=0)
    averaged_features = torch.mean(stacked_features, dim=0, keepdim=True)

    return averaged_features


def extract_and_save_features_with_split(config: PreprocessingConfig, mean_val: float, std_val: float,
                                        train_tracks: Set[str], test_tracks: Set[str],
                                        fold_idx: int) -> Dict[str, Any]:
    """
    Pass 2: Extract and save normalized features with train/test separation for a specific fold.

    Args:
        config: Preprocessing configuration
        mean_val: Dataset mean for normalization (computed from train tracks only)
        std_val: Dataset standard deviation for normalization (computed from train tracks only)
        train_tracks: Set of track names in the training partition
        test_tracks: Set of track names in the test partition
        fold_idx: Current fold index

    Returns:
        Dictionary with extraction statistics

    Raises:
        PreprocessingError: If feature extraction fails
    """
    logger.info(f"Pass 2/2: Extracting and saving normalized features for fold {fold_idx}...")

    try:
        # Create and configure extractor with dataset stats
        feature_extractor = ASTFeatureExtractor.from_pretrained(config.extractor_model)
        feature_extractor.mean = mean_val
        feature_extractor.std = std_val
        if hasattr(feature_extractor, "do_normalize"):
            feature_extractor.do_normalize = True

        # Get all audio files
        items = list_audio_files(config.wav_dir, config)

        # Create fold directory structure
        fold_dir = Path(config.output_dir) / f"fold_{fold_idx}"
        train_dir = fold_dir / "train_chunks"
        test_dir = fold_dir / "test_chunks"

        # Create directories
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        # Track statistics
        stats = {
            "fold_idx": fold_idx,
            "total_files": len(items),
            "successful_train_files": 0,
            "successful_test_files": 0,
            "failed_files": 0,
            "total_train_chunks": 0,
            "total_test_chunks": 0,
            "train_subgenres": {},
            "test_subgenres": {},
            "start_time": time.time()
        }

        # Process files
        with tqdm(items, desc=f"Extracting features (fold {fold_idx})") as pbar:
            for subgenre, filepath, basename in pbar:
                try:
                    # Extract track name and determine partition
                    track_name = re.sub(r'_chunk\d+$', '', basename)

                    if track_name in train_tracks:
                        partition = "train"
                        output_base_dir = train_dir
                        stats_key = "train_subgenres"
                        file_count_key = "successful_train_files"
                        chunk_count_key = "total_train_chunks"
                    elif track_name in test_tracks:
                        partition = "test"
                        output_base_dir = test_dir
                        stats_key = "test_subgenres"
                        file_count_key = "successful_test_files"
                        chunk_count_key = "total_test_chunks"
                    else:
                        # Track not in this fold's partitions, skip
                        continue

                    # Create output directory for this subgenre in the appropriate partition
                    subgenre_dir = output_base_dir / subgenre
                    subgenre_dir.mkdir(parents=True, exist_ok=True)

                    # Load and resample audio
                    waveform = load_and_resample(filepath, config.target_sample_rate, config)

                    # Process chunks
                    file_chunks = 0

                    if config.single_chunk:
                        # Collect features from multiple chunks and average them
                        chunk_features = []
                        chunk_idx = 1

                        for chunk in chunk_waveform(
                            waveform,
                            config.target_sample_rate,
                            config.chunk_duration,
                            config.max_chunks_per_file,
                            strategy=config.chunk_strategy,
                            seed=config.random_seed
                        ):
                            try:
                                # Extract features
                                inputs = feature_extractor(
                                    chunk.numpy(),
                                    sampling_rate=config.target_sample_rate,
                                    return_tensors="pt",
                                    padding=config.padding_strategy
                                )

                                # Validate features
                                if "input_values" not in inputs:
                                    logger.warning(f"No input_values in features for {filepath}, chunk {chunk_idx}")
                                    chunk_idx += 1
                                    continue

                                features = inputs["input_values"]
                                if torch.isnan(features).any() or torch.isinf(features).any():
                                    logger.warning(f"Invalid features in {filepath}, chunk {chunk_idx}")
                                    chunk_idx += 1
                                    continue

                                chunk_features.append(features)
                                chunk_idx += 1

                            except Exception as e:
                                logger.warning(f"Error processing chunk {chunk_idx} from {filepath}: {e}")
                                chunk_idx += 1
                                continue

                        # Average chunks and save single file
                        if chunk_features:
                            try:
                                averaged_features = average_chunks_features(chunk_features)

                                # Create averaged inputs dict
                                averaged_inputs = {
                                    "input_values": averaged_features
                                }

                                # Save averaged features
                                sanitized_name = sanitize_filename(basename, config.max_filename_length)
                                output_filename = f"{sanitized_name}_chunk1.pt"
                                output_path = subgenre_dir / output_filename

                                torch.save(averaged_inputs, output_path)
                                file_chunks = 1  # One averaged chunk per file

                            except Exception as e:
                                logger.warning(f"Error averaging chunks from {filepath}: {e}")
                    else:
                        # Original behavior: save individual chunks
                        chunk_idx = 1

                        for chunk in chunk_waveform(
                            waveform,
                            config.target_sample_rate,
                            config.chunk_duration,
                            config.max_chunks_per_file,
                            strategy=config.chunk_strategy,
                            seed=config.random_seed
                        ):
                            try:
                                # Extract features
                                inputs = feature_extractor(
                                    chunk.numpy(),
                                    sampling_rate=config.target_sample_rate,
                                    return_tensors="pt",
                                    padding=config.padding_strategy
                                )

                                # Validate features
                                if "input_values" not in inputs:
                                    logger.warning(f"No input_values in features for {filepath}, chunk {chunk_idx}")
                                    continue

                                features = inputs["input_values"]
                                if torch.isnan(features).any() or torch.isinf(features).any():
                                    logger.warning(f"Invalid features in {filepath}, chunk {chunk_idx}")
                                    continue

                                # Save features
                                sanitized_name = sanitize_filename(basename, config.max_filename_length)
                                output_filename = f"{sanitized_name}_chunk{chunk_idx}.pt"
                                output_path = subgenre_dir / output_filename

                                torch.save(dict(inputs), output_path)

                                file_chunks += 1
                                chunk_idx += 1

                            except Exception as e:
                                logger.warning(f"Error processing chunk {chunk_idx} from {filepath}: {e}")
                                continue

                    if file_chunks > 0:
                        stats[file_count_key] += 1
                        stats[chunk_count_key] += file_chunks

                        # Track subgenre statistics
                        if subgenre not in stats[stats_key]:
                            stats[stats_key][subgenre] = {"files": 0, "chunks": 0}
                        stats[stats_key][subgenre]["files"] += 1
                        stats[stats_key][subgenre]["chunks"] += file_chunks
                    else:
                        stats["failed_files"] += 1
                        logger.warning(f"No valid chunks extracted from {filepath}")

                    # Update progress bar
                    pbar.set_postfix({
                        'fold': fold_idx,
                        'train': stats["successful_train_files"],
                        'test': stats["successful_test_files"],
                        'failed': stats["failed_files"],
                        'chunks': stats["total_train_chunks"] + stats["total_test_chunks"]
                    })

                except AudioLoadError as e:
                    stats["failed_files"] += 1
                    logger.warning(f"Skipping {filepath}: {e}")
                    continue
                except Exception as e:
                    stats["failed_files"] += 1
                    logger.error(f"Unexpected error processing {filepath}: {e}")
                    continue

        stats["end_time"] = time.time()
        stats["duration"] = stats["end_time"] - stats["start_time"]

        # Log final statistics
        logger.info(f"Fold {fold_idx} feature extraction completed in {stats['duration']:.2f} seconds")
        logger.info(f"Train: {stats['successful_train_files']} files, {stats['total_train_chunks']} chunks")
        logger.info(f"Test: {stats['successful_test_files']} files, {stats['total_test_chunks']} chunks")
        logger.info(f"Failed: {stats['failed_files']} files")

        for partition, subgenre_stats in [("train", stats["train_subgenres"]), ("test", stats["test_subgenres"])]:
            for subgenre, subgenre_data in subgenre_stats.items():
                logger.info(f"  {partition.capitalize()} {subgenre}: {subgenre_data['files']} files, {subgenre_data['chunks']} chunks")

        return stats

    except Exception as e:
        if isinstance(e, PreprocessingError):
            raise
        raise PreprocessingError(f"Feature extraction failed: {e}")

def extract_and_save_features(config: PreprocessingConfig, mean_val: float, std_val: float) -> Dict[str, Any]:
    """
    Pass 2: Extract and save normalized features with comprehensive error handling.
    
    Args:
        config: Preprocessing configuration
        mean_val: Dataset mean for normalization
        std_val: Dataset standard deviation for normalization
        
    Returns:
        Dictionary with extraction statistics
        
    Raises:
        PreprocessingError: If feature extraction fails
    """
    logger.info("Pass 2/2: Extracting and saving normalized features...")
    
    try:
        # Create and configure extractor with dataset stats
        feature_extractor = ASTFeatureExtractor.from_pretrained(config.extractor_model)
        feature_extractor.mean = mean_val
        feature_extractor.std = std_val
        if hasattr(feature_extractor, "do_normalize"):
            feature_extractor.do_normalize = True
        
        # Get all audio files
        items = list_audio_files(config.wav_dir, config)
        
        # Track statistics
        stats = {
            "total_files": len(items),
            "successful_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "subgenres": {},
            "start_time": time.time()
        }
        
        # Process files
        with tqdm(items, desc="Extracting features") as pbar:
            for subgenre, filepath, basename in pbar:
                try:
                    # Create output directory
                    subgenre_dir = Path(config.output_dir) / subgenre
                    subgenre_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Load and resample audio
                    waveform = load_and_resample(filepath, config.target_sample_rate, config)
                    
                    # Process chunks
                    file_chunks = 0

                    if config.single_chunk:
                        # Collect features from multiple chunks and average them
                        chunk_features = []
                        chunk_idx = 1

                        for chunk in chunk_waveform(
                            waveform,
                            config.target_sample_rate,
                            config.chunk_duration,
                            config.max_chunks_per_file,
                            strategy=config.chunk_strategy,
                            seed=config.random_seed
                        ):
                            try:
                                # Extract features
                                inputs = feature_extractor(
                                    chunk.numpy(),
                                    sampling_rate=config.target_sample_rate,
                                    return_tensors="pt",
                                    padding=config.padding_strategy
                                )

                                # Validate features
                                if "input_values" not in inputs:
                                    logger.warning(f"No input_values in features for {filepath}, chunk {chunk_idx}")
                                    chunk_idx += 1
                                    continue

                                features = inputs["input_values"]
                                if torch.isnan(features).any() or torch.isinf(features).any():
                                    logger.warning(f"Invalid features in {filepath}, chunk {chunk_idx}")
                                    chunk_idx += 1
                                    continue

                                chunk_features.append(features)
                                chunk_idx += 1

                            except Exception as e:
                                logger.warning(f"Error processing chunk {chunk_idx} from {filepath}: {e}")
                                chunk_idx += 1
                                continue

                        # Average chunks and save single file
                        if chunk_features:
                            try:
                                averaged_features = average_chunks_features(chunk_features)

                                # Create averaged inputs dict
                                averaged_inputs = {
                                    "input_values": averaged_features
                                }

                                # Save averaged features
                                sanitized_name = sanitize_filename(basename, config.max_filename_length)
                                output_filename = f"{sanitized_name}_chunk1.pt"
                                output_path = subgenre_dir / output_filename

                                torch.save(averaged_inputs, output_path)
                                file_chunks = 1  # One averaged chunk per file

                            except Exception as e:
                                logger.warning(f"Error averaging chunks from {filepath}: {e}")
                    else:
                        # Original behavior: save individual chunks
                        chunk_idx = 1

                        for chunk in chunk_waveform(
                            waveform,
                            config.target_sample_rate,
                            config.chunk_duration,
                            config.max_chunks_per_file,
                            strategy=config.chunk_strategy,
                            seed=config.random_seed
                        ):
                            try:
                                # Extract features
                                inputs = feature_extractor(
                                    chunk.numpy(),
                                    sampling_rate=config.target_sample_rate,
                                    return_tensors="pt",
                                    padding=config.padding_strategy
                                )

                                # Validate features
                                if "input_values" not in inputs:
                                    logger.warning(f"No input_values in features for {filepath}, chunk {chunk_idx}")
                                    continue

                                features = inputs["input_values"]
                                if torch.isnan(features).any() or torch.isinf(features).any():
                                    logger.warning(f"Invalid features in {filepath}, chunk {chunk_idx}")
                                    continue

                                # Save features
                                sanitized_name = sanitize_filename(basename, config.max_filename_length)
                                output_filename = f"{sanitized_name}_chunk{chunk_idx}.pt"
                                output_path = subgenre_dir / output_filename

                                torch.save(dict(inputs), output_path)

                                file_chunks += 1
                                chunk_idx += 1

                            except Exception as e:
                                logger.warning(f"Error processing chunk {chunk_idx} from {filepath}: {e}")
                                continue
                    
                    if file_chunks > 0:
                        stats["successful_files"] += 1
                        stats["total_chunks"] += file_chunks
                        
                        # Track subgenre statistics
                        if subgenre not in stats["subgenres"]:
                            stats["subgenres"][subgenre] = {"files": 0, "chunks": 0}
                        stats["subgenres"][subgenre]["files"] += 1
                        stats["subgenres"][subgenre]["chunks"] += file_chunks
                    else:
                        stats["failed_files"] += 1
                        logger.warning(f"No valid chunks extracted from {filepath}")
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'success': stats["successful_files"],
                        'failed': stats["failed_files"],
                        'chunks': stats["total_chunks"]
                    })
                    
                except AudioLoadError as e:
                    stats["failed_files"] += 1
                    logger.warning(f"Skipping {filepath}: {e}")
                    continue
                except Exception as e:
                    stats["failed_files"] += 1
                    logger.error(f"Unexpected error processing {filepath}: {e}")
                    continue
        
        stats["end_time"] = time.time()
        stats["duration"] = stats["end_time"] - stats["start_time"]
        
        # Log final statistics
        logger.info(f"Feature extraction completed in {stats['duration']:.2f} seconds")
        logger.info(f"Successful: {stats['successful_files']}/{stats['total_files']} files")
        logger.info(f"Total chunks extracted: {stats['total_chunks']}")
        
        for subgenre, subgenre_stats in stats["subgenres"].items():
            logger.info(f"  {subgenre}: {subgenre_stats['files']} files, {subgenre_stats['chunks']} chunks")
        
        if stats["failed_files"] > 0:
            logger.warning(f"Failed to process {stats['failed_files']} files")
        
        return stats
        
    except Exception as e:
        if isinstance(e, PreprocessingError):
            raise
        raise PreprocessingError(f"Feature extraction failed: {e}")


def preprocess_and_extract_features_kfold(config: PreprocessingConfig) -> None:
    """
    Main K-fold preprocessing function that creates separate train/test partitions for each fold.

    Args:
        config: Preprocessing configuration with enable_kfold=True

    Raises:
        PreprocessingError: If preprocessing fails
    """
    try:
        # Validate configuration
        config.validate()
        logger.info(f"Starting K-fold preprocessing with configuration: {config}")

        # Create output directory
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_path.absolute()}")

        # Get all audio files
        items = list_audio_files(config.wav_dir, config)

        # Create K-fold partitions at the track level
        logger.info("Creating K-fold track partitions...")
        kfold_partitions = create_track_based_kfold_partitions(
            items, config.k_folds, config.kfold_partition_seed
        )

        # Save partitions for reproducibility
        partitions_file = output_path / "kfold_partitions.json"
        partition_data = {
            "kfold_partitions": kfold_partitions,
            "metadata": {
                "k_folds": config.k_folds,
                "partition_seed": config.kfold_partition_seed,
                "wav_dir": config.wav_dir,
                "total_tracks": sum(len(set(track for partition in partitions for track in partition))
                                  for partitions in kfold_partitions.values()),
                "total_subgenres": len(kfold_partitions),
                "chunk_strategy": config.chunk_strategy,
                "max_chunks_per_file": config.max_chunks_per_file,
                "chunk_duration": config.chunk_duration
            }
        }

        with open(partitions_file, "w") as f:
            json.dump(partition_data, f, indent=2)
        logger.info(f"K-fold partitions saved to {partitions_file}")

        # Process each fold
        all_fold_stats = []

        for fold_idx in range(config.k_folds):
            logger.info(f"\n{'='*80}")
            logger.info(f"PROCESSING FOLD {fold_idx + 1}/{config.k_folds}")
            logger.info(f"{'='*80}")

            # Get train/test track splits for this fold
            train_tracks, test_tracks = get_fold_track_splits(
                kfold_partitions, fold_idx, config.k_folds
            )

            # Pass 1: Compute dataset statistics using only training tracks
            logger.info(f"Computing normalization statistics from training tracks only...")
            mean_val, std_val = compute_stats_pass_for_tracks(config, train_tracks)

            # Pass 2: Extract and save features for both train and test partitions
            logger.info(f"Extracting features with train/test separation...")
            fold_stats = extract_and_save_features_with_split(
                config, mean_val, std_val, train_tracks, test_tracks, fold_idx
            )

            # Save fold-specific statistics and metadata
            fold_dir = output_path / f"fold_{fold_idx}"

            # Save normalization statistics for this fold
            fold_stats_data = {
                "fold_idx": fold_idx,
                "normalization_stats": {
                    "mean": mean_val,
                    "std": std_val,
                    "computed_from_train_only": True
                },
                "extraction_stats": fold_stats,
                "train_tracks_count": len(train_tracks),
                "test_tracks_count": len(test_tracks),
                "config": {
                    "target_sample_rate": config.target_sample_rate,
                    "chunk_duration": config.chunk_duration,
                    "max_chunks_per_file": config.max_chunks_per_file,
                    "extractor_model": config.extractor_model,
                    "chunk_strategy": config.chunk_strategy,
                    "random_seed": config.random_seed
                },
                "timestamp": time.time()
            }

            fold_stats_file = fold_dir / "fold_statistics.json"
            with open(fold_stats_file, "w") as f:
                json.dump(fold_stats_data, f, indent=2, default=str)
            logger.info(f"Fold {fold_idx} statistics saved to {fold_stats_file}")

            all_fold_stats.append(fold_stats_data)

            # Memory cleanup after each fold to prevent accumulation
            import gc
            clear_resampler_cache()  # Clear resampler cache
            gc.collect()  # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear CUDA cache
            logger.info(f"Memory cleanup completed after fold {fold_idx}")

        # Save overall experiment statistics
        experiment_stats = {
            "k_folds": config.k_folds,
            "partition_seed": config.kfold_partition_seed,
            "total_subgenres": len(kfold_partitions),
            "fold_results": all_fold_stats,
            "config_summary": {
                "wav_dir": config.wav_dir,
                "output_dir": config.output_dir,
                "chunk_strategy": config.chunk_strategy,
                "max_chunks_per_file": config.max_chunks_per_file,
                "chunk_duration": config.chunk_duration,
                "extractor_model": config.extractor_model
            },
            "timestamp": time.time()
        }

        experiment_stats_file = output_path / "kfold_experiment_summary.json"
        with open(experiment_stats_file, "w") as f:
            json.dump(experiment_stats, f, indent=2, default=str)
        logger.info(f"K-fold experiment summary saved to {experiment_stats_file}")

        logger.info(f"\n{'='*80}")
        logger.info(f"K-FOLD PREPROCESSING COMPLETED SUCCESSFULLY!")
        logger.info(f"{'='*80}")
        logger.info(f"Processed {config.k_folds} folds with scientifically sound train/test separation")
        logger.info(f"Output directory: {output_path.absolute()}")

    except Exception as e:
        if isinstance(e, (PreprocessingError, AudioLoadError)):
            raise
        raise PreprocessingError(f"K-fold preprocessing failed: {e}")

def preprocess_and_extract_features(config: PreprocessingConfig) -> None:
    """
    Main preprocessing function with comprehensive error handling.
    Automatically chooses between standard and K-fold preprocessing based on config.

    Args:
        config: Preprocessing configuration

    Raises:
        PreprocessingError: If preprocessing fails
    """
    if config.enable_kfold:
        logger.info("K-fold preprocessing enabled - using scientifically sound train/test separation")
        preprocess_and_extract_features_kfold(config)
    else:
        # Original single-pass preprocessing (with data leakage warning)
        logger.warning("Standard preprocessing enabled - normalization will use ALL data (includes test leakage!)")
        logger.warning("For scientific validity, consider using --enable-kfold flag")

        try:
            # Validate configuration
            config.validate()
            logger.info(f"Starting preprocessing with configuration: {config}")

            # Create output directory
            output_path = Path(config.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {output_path.absolute()}")

            # Pass 1: Compute dataset statistics
            logger.info("Starting Pass 1: Computing dataset statistics...")
            mean_val, std_val = compute_stats_pass(config)

            # Save statistics
            stats_file = output_path / "feature_stats.json"
            stats_data = {
                "mean": mean_val,
                "std": std_val,
                "target_sample_rate": config.target_sample_rate,
                "chunk_duration": config.chunk_duration,
                "max_chunks_per_file": config.max_chunks_per_file,
                "extractor_model": config.extractor_model,
                "timestamp": time.time(),
                "warning": "Statistics computed from ALL data - includes train/test leakage!"
            }

            with open(stats_file, "w") as f:
                json.dump(stats_data, f, indent=2)
            logger.info(f"Statistics saved to {stats_file}")

            # Pass 2: Extract and save features
            logger.info("Starting Pass 2: Feature extraction and saving...")
            extraction_stats = extract_and_save_features(config, mean_val, std_val)

            # Save extraction statistics
            extraction_stats_file = output_path / "extraction_stats.json"
            with open(extraction_stats_file, "w") as f:
                json.dump(extraction_stats, f, indent=2, default=str)
            logger.info(f"Extraction statistics saved to {extraction_stats_file}")

            logger.info("Preprocessing completed successfully!")

        except Exception as e:
            if isinstance(e, (PreprocessingError, AudioLoadError)):
                raise
            raise PreprocessingError(f"Preprocessing failed: {e}")

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Improved AST Feature Preprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument("--wav-dir", default="WAV", help="Directory containing WAV files")
    parser.add_argument("--output-dir", default="preprocessed_features", help="Output directory")
    
    # Audio processing arguments
    parser.add_argument("--chunk-duration", type=int, default=10, help="Chunk duration in seconds")
    parser.add_argument("--max-chunks", type=int, default=3, help="Maximum chunks per file (default: 3 for better generalization)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate")
    
    # Chunk sampling arguments
    parser.add_argument("--chunk-strategy", default="random", 
                       choices=["sequential", "random", "spaced"],
                       help="Chunk sampling strategy (default: random for best diversity)")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducible sampling")
    
    # Feature extraction arguments
    parser.add_argument("--extractor-model", default="MIT/ast-finetuned-audioset-10-10-0.4593",
                       help="AST model for feature extraction")
    parser.add_argument("--padding", default="max_length", help="Padding strategy")
    
    # Processing arguments
    parser.add_argument("--max-filename-length", type=int, default=100, help="Max filename length")
    parser.add_argument("--no-validation", action="store_true", help="Skip audio validation")
    parser.add_argument("--min-length", type=float, default=1.0, help="Minimum audio length (seconds)")
    parser.add_argument("--max-length", type=float, default=700.0, help="Maximum audio length (seconds)")
    
    # Chunking strategy arguments
    parser.add_argument("--single-chunk", action="store_true",
                       help="Average 3 chunks into 1 single chunk per file")

    # K-fold cross-validation arguments
    parser.add_argument("--enable-kfold", action="store_true",
                       help="Enable K-fold preprocessing with scientifically sound train/test separation")
    parser.add_argument("--k-folds", type=int, default=5,
                       help="Number of folds for K-fold cross-validation (default: 5)")
    parser.add_argument("--kfold-seed", type=int, default=42,
                       help="Seed for reproducible K-fold partitioning (default: 42)")

    # Other arguments
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--config", type=str, help="Load configuration from JSON file")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        if args.config and os.path.exists(args.config):
            logger.info(f"Loading configuration from {args.config}")
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
            config = PreprocessingConfig(**config_dict)
        else:
            # Create configuration from command line arguments
            config = PreprocessingConfig(
                wav_dir=args.wav_dir,
                output_dir=args.output_dir,
                chunk_duration=args.chunk_duration,
                max_chunks_per_file=args.max_chunks,
                target_sample_rate=args.sample_rate,
                chunk_strategy=args.chunk_strategy,
                random_seed=args.random_seed,
                single_chunk=args.single_chunk,
                extractor_model=args.extractor_model,
                padding_strategy=args.padding,
                max_filename_length=args.max_filename_length,
                validate_audio=not args.no_validation,
                min_audio_length=args.min_length,
                max_audio_length=args.max_length,
                enable_kfold=args.enable_kfold,
                k_folds=args.k_folds,
                kfold_partition_seed=args.kfold_seed
            )
        
        # Run preprocessing
        preprocess_and_extract_features(config)
        
        logger.info("Preprocessing completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Preprocessing interrupted by user")
        return 1
    except (PreprocessingError, AudioLoadError) as e:
        logger.error(f"Preprocessing failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
