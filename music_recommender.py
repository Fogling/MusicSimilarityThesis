#!/usr/bin/env python3
"""
Music Recommendation System using trained AST Triplet Loss models.

This system generates music recommendations by:
1. Computing embeddings for all tracks (averaging 3 chunks per track)
2. Normalizing embeddings to unit length
3. Computing cosine similarity with query tracks
4. Ranking tracks by similarity score and returning top-k recommendations

Features:
- Single track recommendations
- Playlist generation from seed tracks
- Batch processing for efficiency
- Comprehensive similarity metrics
- Export capabilities for playlists
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTModel
from safetensors.torch import load_file
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from config import ExperimentConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedASTTripletWrapper(nn.Module):
    """AST wrapper identical to the one used in training"""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config

        try:
            logger.info(f"Loading pretrained model: {config.model.pretrained_model}")
            self.ast = ASTModel.from_pretrained(config.model.pretrained_model)
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            raise

        # Projection head - handle both legacy and new config formats
        projection_layers = []
        input_dim = self.ast.config.hidden_size  # 768 for AST

        if hasattr(config.model, 'projection_hidden_layers') and config.model.projection_hidden_layers:
            hidden_sizes = config.model.projection_hidden_layers
            activation = config.model.projection_activation
            dropout_rate = config.model.projection_dropout_rate
        else:
            hidden_sizes = getattr(config.model, 'hidden_sizes', [512])
            activation = getattr(config.model, 'activation', 'relu')
            dropout_rate = getattr(config.model, 'dropout_rate', 0.0)

        # Build projection layers
        for hidden_size in hidden_sizes:
            projection_layers.append(nn.Linear(input_dim, hidden_size))
            if activation.lower() == 'relu':
                projection_layers.append(nn.ReLU())
            elif activation.lower() == 'gelu':
                projection_layers.append(nn.GELU())
            if dropout_rate > 0:
                projection_layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_size

        # Final projection to output dimension
        projection_layers.append(nn.Linear(input_dim, config.model.output_dim))

        self.projection_head = nn.Sequential(*projection_layers)

        logger.info(f"Model initialized with projection: {self.ast.config.hidden_size}D -> {config.model.output_dim}D")

    def forward(self, input_values):
        try:
            outputs = self.ast(input_values=input_values)
            last_hidden_state = outputs.last_hidden_state
            pooled = last_hidden_state.mean(dim=1)
            projected = self.projection_head(pooled)
            return F.normalize(projected, p=2, dim=1)
        except Exception as e:
            logger.error(f"Forward pass error: {e}")
            raise


class MusicRecommender:
    """
    Music recommendation system using trained AST triplet loss models.
    """

    def __init__(self, model_dir: str, device: Optional[str] = None):
        """
        Initialize the recommender with a trained model.

        Args:
            model_dir: Path to directory containing model.safetensors and config.json
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

        logger.info(f"Initializing MusicRecommender with device: {self.device}")

        # Load model and config
        self.model, self.config = self._load_model()

        # Storage for embeddings and metadata
        self.track_embeddings = {}  # {track_id: embedding}
        self.track_metadata = {}    # {track_id: {subgenre, original_filename, chunks_used}}
        self.subgenre_to_tracks = defaultdict(list)  # {subgenre: [track_ids]}

        # Precomputed similarity matrices
        self._similarity_matrix = None
        self._track_id_to_index = None
        self._index_to_track_id = None

    def _load_model(self) -> Tuple[ImprovedASTTripletWrapper, ExperimentConfig]:
        """Load trained model and configuration"""
        config_path = self.model_dir / "config.json"
        model_path = self.model_dir / "model.safetensors"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        # Load config
        config = ExperimentConfig.load(str(config_path))

        # Initialize model
        model = ImprovedASTTripletWrapper(config)

        # Load weights
        state_dict = load_file(str(model_path))

        # Handle legacy layer naming compatibility
        # Old models saved projection layers as "projector.X"
        # New models expect "projection_head.X"
        legacy_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('projector.'):
                # Map old naming to new naming
                new_key = key.replace('projector.', 'projection_head.')
                legacy_state_dict[new_key] = value
                logger.debug(f"Mapped legacy key: {key} -> {new_key}")
            else:
                legacy_state_dict[key] = value

        model.load_state_dict(legacy_state_dict)

        model.to(self.device)
        model.eval()

        logger.info(f"Model loaded from {self.model_dir}")
        return model, config

    def _extract_track_info(self, file_path: str) -> Tuple[str, str, int]:
        """Extract subgenre, track name, and chunk number from file path"""
        parts = file_path.split('/')
        if len(parts) < 3:
            raise ValueError(f"Invalid file path format: {file_path}")

        subgenre = parts[1]
        filename = parts[2]

        if filename.endswith('.pt'):
            filename = filename[:-3]

        # Extract chunk number
        if filename.endswith('_chunk1'):
            track_name = filename[:-7]
            chunk_num = 1
        elif filename.endswith('_chunk2'):
            track_name = filename[:-7]
            chunk_num = 2
        elif filename.endswith('_chunk3'):
            track_name = filename[:-7]
            chunk_num = 3
        else:
            # Fallback: assume no chunk number
            track_name = filename
            chunk_num = 1

        return subgenre, track_name, chunk_num

    def load_embeddings_from_splits(self, splits_file: Optional[str] = None,
                                  chunks_dir: Optional[str] = None,
                                  splits_type: str = "both") -> None:
        """
        Load and compute embeddings from train/test splits.

        Args:
            splits_file: Path to splits.json file (default: model_dir/splits.json)
            chunks_dir: Path to chunks directory (default: from config)
            splits_type: Which splits to use - 'train', 'test', or 'both'
        """
        if splits_file is None:
            splits_file = self.model_dir / "splits.json"

        if chunks_dir is None:
            chunks_dir = Path(self.config.data.chunks_dir)
        else:
            chunks_dir = Path(chunks_dir)

        logger.info(f"Loading splits from {splits_file}")

        with open(splits_file, 'r') as f:
            splits = json.load(f)

        # Determine which splits to process
        splits_to_process = []
        if splits_type in ["train", "both"]:
            splits_to_process.extend(splits["train_split"])
        if splits_type in ["test", "both"]:
            splits_to_process.extend(splits["test_split"])

        logger.info(f"Processing {len(splits_to_process)} triplets from {splits_type} split(s)")

        # Organize tracks by chunks
        track_chunks = defaultdict(lambda: defaultdict(lambda: [None, None, None]))

        for triplet in splits_to_process:
            if len(triplet) < 4:
                continue

            anchor_path, pos_path, neg_path, label = triplet

            for path in [anchor_path, pos_path, neg_path]:
                try:
                    subgenre, track_name, chunk_num = self._extract_track_info(path)
                    full_path = chunks_dir.parent / path

                    if full_path.exists():
                        track_chunks[subgenre][track_name][chunk_num - 1] = str(full_path)
                except Exception as e:
                    logger.warning(f"Failed to process path {path}: {e}")
                    continue

        # Compute embeddings for each track
        self._compute_track_embeddings(track_chunks)

        logger.info(f"Loaded embeddings for {len(self.track_embeddings)} tracks across {len(self.subgenre_to_tracks)} subgenres")

    def load_embeddings_from_fold(self, fold_dir: str, data_split: str = "test") -> None:
        """
        Load and compute embeddings from 5-fold precomputed data structure.

        Args:
            fold_dir: Path to fold directory (e.g., precomp_5Fold_7Gen_3Chunk/fold_0/)
            data_split: Which split to use - 'train' or 'test'
        """
        fold_path = Path(fold_dir)
        chunks_path = fold_path / f"{data_split}_chunks"

        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks directory not found: {chunks_path}")

        logger.info(f"Loading embeddings from {chunks_path}")

        # Organize tracks by chunks - scan subgenre directories
        track_chunks = defaultdict(lambda: defaultdict(lambda: [None, None, None]))

        for subgenre_dir in chunks_path.iterdir():
            if not subgenre_dir.is_dir():
                continue

            subgenre = subgenre_dir.name
            logger.info(f"Processing {subgenre}")

            # Find all chunk files in this subgenre
            for chunk_file in subgenre_dir.glob("*_chunk*.pt"):
                try:
                    filename = chunk_file.stem  # Remove .pt extension

                    # Extract track name and chunk number
                    if filename.endswith("_chunk1"):
                        track_name = filename[:-7]  # Remove _chunk1
                        chunk_idx = 0
                    elif filename.endswith("_chunk2"):
                        track_name = filename[:-7]  # Remove _chunk2
                        chunk_idx = 1
                    elif filename.endswith("_chunk3"):
                        track_name = filename[:-7]  # Remove _chunk3
                        chunk_idx = 2
                    else:
                        logger.warning(f"Unexpected chunk file format: {chunk_file}")
                        continue

                    track_chunks[subgenre][track_name][chunk_idx] = str(chunk_file)

                except Exception as e:
                    logger.warning(f"Failed to process chunk file {chunk_file}: {e}")
                    continue

        # Compute embeddings for each track
        self._compute_track_embeddings(track_chunks)

        logger.info(f"Loaded embeddings for {len(self.track_embeddings)} tracks across {len(self.subgenre_to_tracks)} subgenres")

    def _compute_track_embeddings(self, track_chunks: Dict[str, Dict[str, List[str]]]) -> None:
        """Compute embeddings for all tracks by averaging their chunks"""

        logger.info("Computing track embeddings...")

        with torch.no_grad():
            for subgenre, tracks in track_chunks.items():
                logger.info(f"Processing {subgenre} ({len(tracks)} tracks)")

                for track_name, chunk_paths in tqdm(tracks.items(), desc=f"Processing {subgenre}"):
                    chunk_embeddings = []
                    chunks_used = 0

                    # Load and process each chunk
                    for chunk_path in chunk_paths:
                        if chunk_path is None:
                            continue

                        try:
                            # Load chunk features (same way as training script)
                            chunk_data = torch.load(chunk_path, map_location='cpu', weights_only=True)

                            # Move to device
                            if isinstance(chunk_data, dict):
                                chunk_features = {k: v.to(self.device) for k, v in chunk_data.items()}
                            else:
                                chunk_features = chunk_data.to(self.device)

                            # Extract input_values tensor from dict (like training script)
                            if isinstance(chunk_features, dict):
                                input_values = chunk_features['input_values']
                            else:
                                input_values = chunk_features

                            # Get embedding from model
                            embedding = self.model(input_values)
                            chunk_embeddings.append(embedding.cpu().numpy())
                            chunks_used += 1

                        except Exception as e:
                            logger.warning(f"Failed to process chunk {chunk_path}: {e}")
                            continue

                    if chunk_embeddings:
                        # Average embeddings across chunks
                        track_embedding = np.mean(chunk_embeddings, axis=0)
                        # Normalize to unit length
                        track_embedding = normalize(track_embedding.reshape(1, -1))[0]

                        # Create unique track ID
                        track_id = f"{subgenre}/{track_name}"

                        # Store embedding and metadata
                        self.track_embeddings[track_id] = track_embedding
                        self.track_metadata[track_id] = {
                            'subgenre': subgenre,
                            'track_name': track_name,
                            'chunks_used': chunks_used
                        }
                        self.subgenre_to_tracks[subgenre].append(track_id)

    def precompute_similarity_matrix(self) -> None:
        """Precompute similarity matrix for faster recommendations"""
        if not self.track_embeddings:
            raise ValueError("No embeddings loaded. Call load_embeddings_from_splits() or load_embeddings_from_fold() first.")

        logger.info("Precomputing similarity matrix...")

        # Create ordered lists
        self._track_id_to_index = {track_id: i for i, track_id in enumerate(self.track_embeddings.keys())}
        self._index_to_track_id = {i: track_id for track_id, i in self._track_id_to_index.items()}

        # Stack embeddings
        embeddings_matrix = np.stack([self.track_embeddings[track_id]
                                    for track_id in self._track_id_to_index.keys()])

        # Compute cosine similarity matrix
        self._similarity_matrix = cosine_similarity(embeddings_matrix)

        logger.info(f"Similarity matrix computed: {self._similarity_matrix.shape}")

    def get_recommendations(self, query_track_id: str, k: int = 10,
                          exclude_same_track: bool = True,
                          exclude_same_subgenre: bool = False) -> List[Tuple[str, float]]:
        """
        Get top-k recommendations for a query track.

        Args:
            query_track_id: Track ID in format "subgenre/track_name"
            k: Number of recommendations to return
            exclude_same_track: Whether to exclude the query track itself
            exclude_same_subgenre: Whether to exclude tracks from same subgenre

        Returns:
            List of (track_id, similarity_score) tuples, sorted by similarity (descending)
        """
        if query_track_id not in self.track_embeddings:
            raise ValueError(f"Track {query_track_id} not found in embeddings")

        query_embedding = self.track_embeddings[query_track_id]
        query_subgenre = self.track_metadata[query_track_id]['subgenre']

        # Compute similarities
        if self._similarity_matrix is not None:
            # Use precomputed matrix
            query_index = self._track_id_to_index[query_track_id]
            similarities = self._similarity_matrix[query_index]

            # Create list of (track_id, similarity) pairs
            candidates = [(self._index_to_track_id[i], sim) for i, sim in enumerate(similarities)]
        else:
            # Compute on-the-fly
            candidates = []
            for track_id, embedding in self.track_embeddings.items():
                similarity = np.dot(query_embedding, embedding)  # Already normalized
                candidates.append((track_id, similarity))

        # Filter candidates
        filtered_candidates = []
        for track_id, similarity in candidates:
            # Exclude same track
            if exclude_same_track and track_id == query_track_id:
                continue

            # Exclude same subgenre
            if exclude_same_subgenre:
                track_subgenre = self.track_metadata[track_id]['subgenre']
                if track_subgenre == query_subgenre:
                    continue

            filtered_candidates.append((track_id, similarity))

        # Sort by similarity (descending) and return top-k
        filtered_candidates.sort(key=lambda x: x[1], reverse=True)

        return filtered_candidates[:k]

    def generate_playlist(self, seed_tracks: List[str], playlist_length: int = 20,
                         diversity_weight: float = 0.1) -> List[str]:
        """
        Generate a playlist starting from seed tracks.

        Args:
            seed_tracks: List of track IDs to use as seeds
            playlist_length: Total length of generated playlist (including seeds)
            diversity_weight: Weight for diversity vs similarity (0 = pure similarity, 1 = pure diversity)

        Returns:
            List of track IDs representing the generated playlist
        """
        if not seed_tracks:
            raise ValueError("At least one seed track must be provided")

        for track_id in seed_tracks:
            if track_id not in self.track_embeddings:
                raise ValueError(f"Seed track {track_id} not found in embeddings")

        playlist = seed_tracks.copy()
        remaining_length = playlist_length - len(seed_tracks)

        if remaining_length <= 0:
            return playlist[:playlist_length]

        logger.info(f"Generating playlist of length {playlist_length} from {len(seed_tracks)} seed(s)")

        for _ in range(remaining_length):
            # Compute average embedding of current playlist
            playlist_embeddings = [self.track_embeddings[track_id] for track_id in playlist]
            playlist_centroid = np.mean(playlist_embeddings, axis=0)
            playlist_centroid = normalize(playlist_centroid.reshape(1, -1))[0]

            # Find candidates
            candidates = []
            for track_id, embedding in self.track_embeddings.items():
                if track_id in playlist:
                    continue

                # Similarity to playlist centroid
                similarity = np.dot(playlist_centroid, embedding)

                # Diversity score (minimum distance to any track in playlist)
                if diversity_weight > 0:
                    min_similarity_to_playlist = min(
                        np.dot(embedding, self.track_embeddings[pl_track])
                        for pl_track in playlist
                    )
                    diversity = 1 - min_similarity_to_playlist
                else:
                    diversity = 0

                # Combined score
                score = (1 - diversity_weight) * similarity + diversity_weight * diversity
                candidates.append((track_id, score))

            if not candidates:
                break

            # Select track with highest score
            candidates.sort(key=lambda x: x[1], reverse=True)
            next_track = candidates[0][0]
            playlist.append(next_track)

        logger.info(f"Generated playlist with {len(playlist)} tracks")
        return playlist

    def get_similar_tracks_batch(self, query_tracks: List[str], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get recommendations for multiple query tracks efficiently.

        Args:
            query_tracks: List of track IDs
            k: Number of recommendations per query

        Returns:
            Dictionary mapping query_track_id -> list of (track_id, similarity) pairs
        """
        results = {}

        for query_track in query_tracks:
            try:
                recommendations = self.get_recommendations(query_track, k=k)
                results[query_track] = recommendations
            except Exception as e:
                logger.warning(f"Failed to get recommendations for {query_track}: {e}")
                results[query_track] = []

        return results

    def export_playlist(self, playlist: List[str], output_file: str, format: str = 'json') -> None:
        """
        Export playlist to file.

        Args:
            playlist: List of track IDs
            output_file: Output file path
            format: Export format ('json', 'csv', 'txt')
        """
        output_path = Path(output_file)

        if format.lower() == 'json':
            playlist_data = {
                'playlist': playlist,
                'metadata': {track_id: self.track_metadata[track_id] for track_id in playlist if track_id in self.track_metadata},
                'generated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                'total_tracks': len(playlist)
            }

            with open(output_path, 'w') as f:
                json.dump(playlist_data, f, indent=2)

        elif format.lower() == 'csv':
            playlist_df = pd.DataFrame([
                {
                    'track_id': track_id,
                    'subgenre': self.track_metadata.get(track_id, {}).get('subgenre', 'Unknown'),
                    'track_name': self.track_metadata.get(track_id, {}).get('track_name', 'Unknown'),
                    'position': i + 1
                }
                for i, track_id in enumerate(playlist)
            ])
            playlist_df.to_csv(output_path, index=False)

        elif format.lower() == 'txt':
            with open(output_path, 'w') as f:
                f.write(f"Generated Playlist ({len(playlist)} tracks)\n")
                f.write("=" * 50 + "\n\n")
                for i, track_id in enumerate(playlist, 1):
                    metadata = self.track_metadata.get(track_id, {})
                    f.write(f"{i:2d}. {metadata.get('subgenre', 'Unknown')} - {metadata.get('track_name', track_id)}\n")

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Playlist exported to {output_path}")

    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about loaded embeddings"""
        stats = {
            'total_tracks': len(self.track_embeddings),
            'total_subgenres': len(self.subgenre_to_tracks),
            'embedding_dimension': len(next(iter(self.track_embeddings.values()))) if self.track_embeddings else 0,
            'tracks_per_subgenre': {subgenre: len(tracks) for subgenre, tracks in self.subgenre_to_tracks.items()},
            'has_precomputed_similarities': self._similarity_matrix is not None
        }
        return stats

    def save_embeddings(self, output_file: str) -> None:
        """Save computed embeddings to file for later use"""
        data = {
            'track_embeddings': self.track_embeddings,
            'track_metadata': self.track_metadata,
            'subgenre_to_tracks': dict(self.subgenre_to_tracks),
            'model_dir': str(self.model_dir),
            'embedding_dimension': len(next(iter(self.track_embeddings.values()))) if self.track_embeddings else 0
        }

        with open(output_file, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Embeddings saved to {output_file}")

    def load_embeddings(self, input_file: str) -> None:
        """Load precomputed embeddings from file"""
        with open(input_file, 'rb') as f:
            data = pickle.load(f)

        self.track_embeddings = data['track_embeddings']
        self.track_metadata = data['track_metadata']
        self.subgenre_to_tracks = defaultdict(list, data['subgenre_to_tracks'])

        logger.info(f"Loaded embeddings from {input_file}: {len(self.track_embeddings)} tracks")


def main():
    """Command-line interface for the music recommender"""
    parser = argparse.ArgumentParser(description="Music Recommendation System using AST Triplet Loss")

    parser.add_argument("--model_dir", type=str, required=True,
                       help="Path to directory containing trained model")
    parser.add_argument("--splits_file", type=str, default=None,
                       help="Path to splits.json file (default: model_dir/splits.json)")
    parser.add_argument("--chunks_dir", type=str, default=None,
                       help="Path to chunks directory (default: from config)")
    parser.add_argument("--splits_type", type=str, default="test", choices=["train", "test", "both"],
                       help="Which splits to use for embeddings")
    # New 5-fold data arguments
    parser.add_argument("--fold_dir", type=str, default=None,
                       help="Path to fold directory (e.g., precomp_5Fold_7Gen_3Chunk/fold_0/)")
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "test"],
                       help="Which data split to use from fold ('train' or 'test')")

    # Recommendation options
    parser.add_argument("--query_track", type=str, default=None,
                       help="Track ID to get recommendations for (format: subgenre/track_name)")
    parser.add_argument("--k", type=int, default=10,
                       help="Number of recommendations to return")
    parser.add_argument("--exclude_same_subgenre", action="store_true",
                       help="Exclude tracks from same subgenre in recommendations")

    # Playlist generation
    parser.add_argument("--generate_playlist", action="store_true",
                       help="Generate a playlist from seed tracks")
    parser.add_argument("--seed_tracks", type=str, nargs="+", default=None,
                       help="Seed tracks for playlist generation")
    parser.add_argument("--playlist_length", type=int, default=20,
                       help="Length of generated playlist")
    parser.add_argument("--diversity_weight", type=float, default=0.1,
                       help="Weight for diversity vs similarity in playlist generation")

    # Output options
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for results")
    parser.add_argument("--format", type=str, default="json", choices=["json", "csv", "txt"],
                       help="Output format for playlists")
    parser.add_argument("--save_embeddings", type=str, default=None,
                       help="Save computed embeddings to file")
    parser.add_argument("--load_embeddings", type=str, default=None,
                       help="Load precomputed embeddings from file")

    # Performance options
    parser.add_argument("--precompute_similarities", action="store_true",
                       help="Precompute similarity matrix for faster recommendations")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"],
                       help="Device to use for inference")

    args = parser.parse_args()

    # Initialize recommender
    recommender = MusicRecommender(args.model_dir, device=args.device)

    # Load embeddings
    if args.load_embeddings:
        recommender.load_embeddings(args.load_embeddings)
    elif args.fold_dir:
        # Use new 5-fold data structure
        recommender.load_embeddings_from_fold(
            fold_dir=args.fold_dir,
            data_split=args.data_split
        )
    else:
        # Use legacy splits.json structure
        recommender.load_embeddings_from_splits(
            splits_file=args.splits_file,
            chunks_dir=args.chunks_dir,
            splits_type=args.splits_type
        )

    # Precompute similarities if requested
    if args.precompute_similarities:
        recommender.precompute_similarity_matrix()

    # Save embeddings if requested
    if args.save_embeddings:
        recommender.save_embeddings(args.save_embeddings)

    # Print statistics
    stats = recommender.get_statistics()
    print(f"\nLoaded {stats['total_tracks']} tracks across {stats['total_subgenres']} subgenres")
    print(f"Embedding dimension: {stats['embedding_dimension']}")

    # Get recommendations for single track
    if args.query_track:
        print(f"\nTop {args.k} recommendations for '{args.query_track}':")
        try:
            recommendations = recommender.get_recommendations(
                args.query_track,
                k=args.k,
                exclude_same_subgenre=args.exclude_same_subgenre
            )

            for i, (track_id, similarity) in enumerate(recommendations, 1):
                metadata = recommender.track_metadata.get(track_id, {})
                print(f"{i:2d}. {track_id} (similarity: {similarity:.3f}) - {metadata.get('subgenre', 'Unknown')}")

        except Exception as e:
            print(f"Error getting recommendations: {e}")

    # Generate playlist
    if args.generate_playlist and args.seed_tracks:
        print(f"\nGenerating playlist from seed tracks: {args.seed_tracks}")
        try:
            playlist = recommender.generate_playlist(
                args.seed_tracks,
                playlist_length=args.playlist_length,
                diversity_weight=args.diversity_weight
            )

            print(f"\nGenerated playlist ({len(playlist)} tracks):")
            for i, track_id in enumerate(playlist, 1):
                metadata = recommender.track_metadata.get(track_id, {})
                marker = "🌱" if track_id in args.seed_tracks else "  "
                print(f"{marker} {i:2d}. {track_id} - {metadata.get('subgenre', 'Unknown')}")

            # Export playlist if requested
            if args.output_file:
                recommender.export_playlist(playlist, args.output_file, args.format)

        except Exception as e:
            print(f"Error generating playlist: {e}")


if __name__ == "__main__":
    main()