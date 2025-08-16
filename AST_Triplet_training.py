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
        
        # No fake train_accuracy calculation - only real eval_accuracy matters
        
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
    Improved triplet dataset with proper error handling, validation, and caching.
    """
    
    def __init__(self, split_data: Union[List, Dict], config: ExperimentConfig):
        """
        Initialize dataset with improved data format support.
        
        Args:
            split_data: Either legacy format (list of triplets) or new format (dict)
            config: Experiment configuration
        """
        self.config = config
        self.triplets = self._parse_split_data(split_data)
        self.cache = {} if len(self.triplets) < 1000 else None  # Cache for small datasets
        
        logger.info(f"Initialized dataset with {len(self.triplets)} triplets")
        logger.info(f"Caching {'enabled' if self.cache is not None else 'disabled'}")
    
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
    
    def _safe_load_tensor(self, filepath: str) -> Dict[str, torch.Tensor]:
        """Safely load tensor with comprehensive error handling."""
        if self.cache is not None and filepath in self.cache:
            return self.cache[filepath]
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tensor file not found: {filepath}")
        
        try:
            # Use weights_only=True for security
            tensor_dict = torch.load(filepath, map_location='cpu', weights_only=True)
            
            if not isinstance(tensor_dict, dict):
                raise TripletValidationError(f"Expected dict from {filepath}, got {type(tensor_dict)}")
            
            # Validate and clean
            cleaned = sanitize_dict_tensor(tensor_dict)
            
            # Cache if enabled
            if self.cache is not None:
                self.cache[filepath] = cleaned
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise TripletValidationError(f"Failed to load {filepath}: {e}")
    
    def __len__(self) -> int:
        return len(self.triplets)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get triplet with comprehensive validation."""
        if idx >= len(self.triplets):
            raise IndexError(f"Index {idx} out of range for dataset size {len(self.triplets)}")
        
        try:
            anchor_path, positive_path, negative_path, subgenre = self.triplets[idx]
            
            # Load tensors with error handling
            anchor_input = self._safe_load_tensor(anchor_path)
            positive_input = self._safe_load_tensor(positive_path)
            negative_input = self._safe_load_tensor(negative_path)
            
            return {
                "anchor_input": anchor_input,
                "positive_input": positive_input,
                "negative_input": negative_input,
                "labels": 0,  # Dummy label for compatibility
                "subgenre": subgenre
            }
            
        except Exception as e:
            logger.error(f"Error loading triplet {idx}: {e}")
            raise


def safe_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Improved collation function with error handling and validation.
    """
    if not batch:
        raise ValueError("Empty batch provided to collate_fn")
    
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
    logger.info("Generating train/test splits from preprocessed features")
    return generate_triplet_splits(config)


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
        files = [f for f in subdir.iterdir() if f.suffix == '.pt']
        
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
    train_triplets = _generate_triplets_from_tracks(train_tracks, "train")
    
    logger.info("Generating test triplets from test tracks...")
    test_triplets = _generate_triplets_from_tracks(test_tracks, "test")
    
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
                                 split_name: str) -> List[Tuple[str, str, str, str]]:
    """Generate triplets from a given set of tracks (train or test)."""
    all_triplets = []
    subgenre_list = list(subgenre_tracks.keys())
    
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
                
                # Sample positive tracks (limit to avoid explosion)
                max_positives = min(len(positive_tracks), 6)
                positive_sample = random.sample(positive_tracks, max_positives)
                
                for positive_track in positive_sample:
                    positive_chunks = tracks[positive_track]
                    
                    # Generate 1-2 triplets per positive track
                    triplets_per_positive = min(2, len(positive_chunks))
                    
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
        
        logger.info(f"Experiment: {config.experiment_name}")
        if config.description:
            logger.info(f"Description: {config.description}")
        
        # Set random seeds for reproducibility
        set_random_seeds(config.training.seed)
        
        # Load data splits
        logger.info("Loading data splits...")
        train_data, test_data = load_split_data(config)
        
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
                "chunks_dir": config.data.chunks_dir,
                "test_split_ratio": config.data.test_split_ratio
            }
        }
        with open(splits_path, 'w') as f:
            json.dump(splits_data, f, indent=2)
        logger.info(f"Splits saved to {splits_path}")
        
        # Create datasets
        logger.info("Creating datasets...")
        train_dataset = ImprovedTripletFeatureDataset(train_data, config)
        test_dataset = ImprovedTripletFeatureDataset(test_data, config)
        
        # Initialize model
        logger.info("Initializing model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        model = ImprovedASTTripletWrapper(config).to(device)
        
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
            "disable_tqdm": False,  # Enable tqdm progress bars
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
