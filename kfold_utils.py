"""
K-Fold Cross-Validation utilities for music similarity learning.

This module provides track-level K-Fold splitting to ensure no data leakage
between training and test sets while enabling robust statistical evaluation.
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

from config import ExperimentConfig

logger = logging.getLogger(__name__)


def create_kfold_partitions(config: ExperimentConfig, k: int = 5) -> Dict[str, List[List[str]]]:
    """
    DEPRECATED: Create K-Fold partitions at the track level for each subgenre.

    This function is deprecated in favor of using preprocessed k-fold splits created by
    Preprocess_AST_features.py with --enable-kfold flag, which eliminates data leakage
    by computing normalization statistics only from training data.

    Use the new approach instead:
    1. python Preprocess_AST_features.py --enable-kfold --k-folds 5 --wav-dir WAV --output-dir precomputed_5fold
    2. python AST_Triplet_kfold.py --preprocessed-dir precomputed_5fold --k 5
    """
    import warnings
    warnings.warn(
        "create_kfold_partitions is deprecated. Use preprocessed k-fold splits instead to avoid data leakage.",
        DeprecationWarning,
        stacklevel=2
    )
    """
    Create K-Fold partitions at the track level for each subgenre.

    Args:
        config: Experiment configuration
        k: Number of folds (default: 5)

    Returns:
        Dictionary mapping subgenre -> list of K track partitions
        Format: {subgenre: [[tracks_fold0], [tracks_fold1], ..., [tracks_fold4]]}
    """
    chunks_dir = Path(config.data.chunks_dir)
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

    logger.info(f"Creating {k}-fold partitions for track-level cross-validation")

    # Set seed for reproducible partitioning
    partition_seed = config.training.seed
    random.seed(partition_seed)

    # Step 1: Organize tracks by subgenre
    subgenre_tracks = defaultdict(list)  # {subgenre: [track_names]}

    for subdir in sorted(chunks_dir.iterdir()):
        if not subdir.is_dir():
            continue

        subgenre = subdir.name
        files = sorted([f for f in subdir.iterdir() if f.suffix == '.pt'])

        # Extract unique track names (remove _chunkN.pt suffix)
        track_names = set()
        for file in files:
            track_name = file.stem.rsplit('_chunk', 1)[0]
            track_names.add(track_name)

        # Only keep tracks with at least 2 chunks
        valid_tracks = []
        for track_name in track_names:
            track_chunks = [f for f in files if f.stem.startswith(track_name + '_chunk')]
            if len(track_chunks) >= 2:
                valid_tracks.append(track_name)

        if len(valid_tracks) < k:
            logger.warning(f"Subgenre {subgenre} has only {len(valid_tracks)} tracks, "
                          f"less than K={k} folds. Some folds will be empty.")

        subgenre_tracks[subgenre] = sorted(valid_tracks)
        logger.info(f"Subgenre {subgenre}: {len(valid_tracks)} tracks available for {k}-fold CV")

    # Step 2: Create K partitions for each subgenre
    kfold_partitions = {}

    for subgenre, tracks in subgenre_tracks.items():
        if not tracks:
            logger.warning(f"No valid tracks found for subgenre {subgenre}")
            kfold_partitions[subgenre] = [[] for _ in range(k)]
            continue

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
        logger.info(f"Subgenre {subgenre} K-fold partitions: {partition_sizes}")

    logger.info(f"Created {k}-fold partitions for {len(kfold_partitions)} subgenres")
    return kfold_partitions


def get_fold_splits(kfold_partitions: Dict[str, List[List[str]]],
                   fold_idx: int, k: int = 5) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Get train/test track splits for a specific fold.

    Args:
        kfold_partitions: Output from create_kfold_partitions()
        fold_idx: Current fold index (0 to k-1)
        k: Number of folds

    Returns:
        Tuple of (train_tracks, test_tracks) where each is {subgenre: [track_names]}
    """
    if not (0 <= fold_idx < k):
        raise ValueError(f"fold_idx must be between 0 and {k-1}, got {fold_idx}")

    train_tracks = defaultdict(list)
    test_tracks = defaultdict(list)

    for subgenre, partitions in kfold_partitions.items():
        # Test set: partition at fold_idx
        test_tracks[subgenre] = partitions[fold_idx].copy()

        # Train set: all other partitions combined
        for i in range(k):
            if i != fold_idx:
                train_tracks[subgenre].extend(partitions[i])

    # Convert to regular dicts
    train_tracks = dict(train_tracks)
    test_tracks = dict(test_tracks)

    # Log split sizes
    total_train = sum(len(tracks) for tracks in train_tracks.values())
    total_test = sum(len(tracks) for tracks in test_tracks.values())
    logger.info(f"Fold {fold_idx}: {total_train} train tracks, {total_test} test tracks")

    return train_tracks, test_tracks


def generate_kfold_triplet_splits(train_tracks: Dict[str, List[str]],
                                 test_tracks: Dict[str, List[str]],
                                 config: ExperimentConfig,
                                 fold_idx: int) -> Tuple[List, List]:
    """
    DEPRECATED: Generate triplets from K-fold train/test track splits.

    This function is deprecated in favor of using preprocessed k-fold splits which
    eliminate data leakage by computing normalization statistics only from training data.

    Use the new approach with AST_Triplet_kfold.py --preprocessed-dir instead.
    """
    import warnings
    warnings.warn(
        "generate_kfold_triplet_splits is deprecated. Use preprocessed k-fold splits instead to avoid data leakage.",
        DeprecationWarning,
        stacklevel=2
    )
    """
    Generate triplets from K-fold train/test track splits.

    Args:
        train_tracks: Training tracks by subgenre
        test_tracks: Test tracks by subgenre
        config: Experiment configuration
        fold_idx: Current fold index (for seeding)

    Returns:
        Tuple of (train_triplets, test_triplets)
    """
    # Set fold-specific seed for triplet generation
    fold_seed = config.training.seed + fold_idx
    random.seed(fold_seed)

    logger.info(f"Generating triplets for fold {fold_idx} with seed {fold_seed}")

    # Convert track lists to track->chunks mapping (same format as existing code)
    chunks_dir = Path(config.data.chunks_dir)

    def tracks_to_chunks_mapping(tracks_by_subgenre: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
        """Convert track names to track->chunks mapping."""
        result = defaultdict(dict)

        for subgenre, track_names in tracks_by_subgenre.items():
            subdir = chunks_dir / subgenre
            if not subdir.exists():
                continue

            files = sorted([f for f in subdir.iterdir() if f.suffix == '.pt'])

            for track_name in track_names:
                track_chunks = [str(f) for f in files if f.stem.startswith(track_name + '_chunk')]
                if track_chunks:
                    result[subgenre][track_name] = sorted(track_chunks)

        return dict(result)

    # Convert to chunks mapping format
    train_chunks_mapping = tracks_to_chunks_mapping(train_tracks)
    test_chunks_mapping = tracks_to_chunks_mapping(test_tracks)

    # Import the existing triplet generation function
    from AST_Triplet_training import _generate_triplets_from_tracks

    # Generate triplets using existing logic
    logger.info(f"Generating training triplets for fold {fold_idx}...")
    train_triplets = _generate_triplets_from_tracks(train_chunks_mapping, f"fold_{fold_idx}_train", config)

    logger.info(f"Generating test triplets for fold {fold_idx}...")
    test_triplets = _generate_triplets_from_tracks(test_chunks_mapping, f"fold_{fold_idx}_test", config)

    return train_triplets, test_triplets


def save_kfold_partitions(kfold_partitions: Dict[str, List[List[str]]],
                         filepath: str,
                         config: ExperimentConfig) -> None:
    """
    Save K-fold partitions to file for reproducibility.

    Args:
        kfold_partitions: Partition data from create_kfold_partitions()
        filepath: Path to save partitions
        config: Experiment configuration (for metadata)
    """
    partition_data = {
        "kfold_partitions": kfold_partitions,
        "metadata": {
            "seed": config.training.seed,
            "chunks_dir": config.data.chunks_dir,
            "k_folds": len(next(iter(kfold_partitions.values()))) if kfold_partitions else 0,
            "total_subgenres": len(kfold_partitions),
            "experiment_name": config.experiment_name
        }
    }

    with open(filepath, 'w') as f:
        json.dump(partition_data, f, indent=2)

    logger.info(f"K-fold partitions saved to {filepath}")


def load_kfold_partitions(filepath: str) -> Dict[str, List[List[str]]]:
    """
    Load K-fold partitions from file.

    Args:
        filepath: Path to partitions file

    Returns:
        K-fold partitions dictionary
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    logger.info(f"Loaded K-fold partitions from {filepath}")
    return data["kfold_partitions"]


def run_kfold_experiment(config: ExperimentConfig, k: int = 5,
                        output_dir: str = "kfold_results") -> Dict[str, Any]:
    """
    Run complete K-fold cross-validation experiment.

    Args:
        config: Experiment configuration
        k: Number of folds
        output_dir: Directory to save results

    Returns:
        Dictionary with results summary and fold-wise metrics
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(output_dir) / f"kfold_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting {k}-fold cross-validation experiment")
    logger.info(f"Results will be saved to: {results_dir}")

    # Create K-fold partitions
    kfold_partitions = create_kfold_partitions(config, k=k)

    # Save partitions for reproducibility
    partitions_file = results_dir / "kfold_partitions.json"
    save_kfold_partitions(kfold_partitions, str(partitions_file), config)

    # Save base configuration
    base_config_file = results_dir / "base_config.json"
    config.save(str(base_config_file))

    fold_results = []

    # Run training for each fold
    for fold_idx in range(k):
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING FOLD {fold_idx + 1}/{k}")
        logger.info(f"{'='*60}")

        # Get train/test splits for this fold
        train_tracks, test_tracks = get_fold_splits(kfold_partitions, fold_idx, k=k)

        # Generate triplets for this fold
        train_triplets, test_triplets = generate_kfold_triplet_splits(
            train_tracks, test_tracks, config, fold_idx
        )

        # Create fold-specific output directory
        fold_dir = results_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)

        # Save fold-specific splits
        fold_splits = {
            "train_triplets": train_triplets,
            "test_triplets": test_triplets,
            "train_tracks": train_tracks,
            "test_tracks": test_tracks,
            "fold_idx": fold_idx,
            "fold_seed": config.training.seed + fold_idx
        }

        with open(fold_dir / "fold_splits.json", 'w') as f:
            json.dump(fold_splits, f, indent=2)

        # Create fold-specific config with updated seed and output directory
        fold_config = ExperimentConfig.load(str(base_config_file))
        fold_config.training.seed = config.training.seed + fold_idx
        fold_config.experiment_name = f"{config.experiment_name}_fold_{fold_idx}"
        fold_config.description = f"Fold {fold_idx + 1}/{k} of K-fold cross-validation"

        # Save fold config
        fold_config_file = fold_dir / "config.json"
        fold_config.save(str(fold_config_file))

        logger.info(f"Fold {fold_idx} setup complete:")
        logger.info(f"  Train triplets: {len(train_triplets)}")
        logger.info(f"  Test triplets: {len(test_triplets)}")
        logger.info(f"  Fold seed: {fold_config.training.seed}")
        logger.info(f"  Results dir: {fold_dir}")

        # Note: Actual training would happen here
        # For now, we just prepare the data and configurations
        fold_results.append({
            "fold_idx": fold_idx,
            "train_samples": len(train_triplets),
            "test_samples": len(test_triplets),
            "fold_dir": str(fold_dir),
            "fold_seed": fold_config.training.seed
        })

    # Save experiment summary
    experiment_summary = {
        "experiment_name": config.experiment_name,
        "timestamp": timestamp,
        "k_folds": k,
        "base_seed": config.training.seed,
        "results_dir": str(results_dir),
        "fold_results": fold_results,
        "total_subgenres": len(kfold_partitions),
        "config_summary": {
            "batch_size": config.training.batch_size,
            "epochs": config.training.epochs,
            "learning_rate": config.training.learning_rate,
            "triplet_margin": config.training.triplet_margin
        }
    }

    with open(results_dir / "experiment_summary.json", 'w') as f:
        json.dump(experiment_summary, f, indent=2)

    logger.info(f"\nK-fold experiment setup complete!")
    logger.info(f"Experiment summary saved to: {results_dir / 'experiment_summary.json'}")
    logger.info(f"Ready to run training for {k} folds")

    return experiment_summary