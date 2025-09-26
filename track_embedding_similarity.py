#!/usr/bin/env python3
"""
Track-level embedding extraction and similarity computation.

This script:
1. Loads WAV files from a directory
2. Extracts multiple chunks per track using random sampling
3. Processes chunks through a trained AST triplet model
4. Averages chunk embeddings to get track-level representations
5. Computes cosine similarities with provided subgenre centroids
6. Outputs similarity matrix as CSV
"""

import os
import json
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import ASTModel, ASTFeatureExtractor
from safetensors.torch import load_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CHUNK_DURATION = 10.24  # seconds - optimal for AST
TARGET_SAMPLE_RATE = 16000  # Hz - required by AST
RANDOM_SEED = 42  # For reproducible chunk sampling


class ImprovedASTTripletWrapper(nn.Module):
    """
    AST wrapper for triplet learning (inference only).
    Simplified version without training-specific features.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # Load pretrained AST model
        model_config = config.get('model', {})
        pretrained_model = model_config.get('pretrained_model', 'MIT/ast-finetuned-audioset-10-10-0.4593')

        logger.info(f"Loading pretrained AST model: {pretrained_model}")
        self.ast = ASTModel.from_pretrained(
            pretrained_model,
            hidden_dropout_prob=0.0,  # No dropout in eval mode
            attention_probs_dropout_prob=0.0
        )

        # Build projection head based on config
        self.projector = self._build_projection_head(model_config)
        logger.info(f"Model initialized with projection to {model_config.get('output_dim', 128)}D")

    def _build_projection_head(self, model_config: Dict[str, Any]) -> nn.Module:
        """Build MLP projection head for embeddings to match training architecture."""
        # Get architecture parameters from config
        projection_hidden_layers = model_config.get('projection_hidden_layers')
        if projection_hidden_layers is None:
            # Fallback to legacy format
            hidden_sizes = model_config.get('hidden_sizes')
            if hidden_sizes and len(hidden_sizes) > 1:
                projection_hidden_layers = hidden_sizes[:-1]
            else:
                projection_hidden_layers = [512]  # Default

        output_dim = model_config.get('output_dim', 128)
        activation_name = model_config.get('projection_activation', 'relu')
        dropout_rate = model_config.get('projection_dropout_rate', 0.15)

        # AST output dimension
        ast_output_dim = self.ast.config.hidden_size  # 768 for MIT AST

        # Get activation function
        activation_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU()
        }
        activation = activation_map.get(activation_name.lower(), nn.ReLU())

        # Build layers to match training architecture
        layers = []
        current_dim = ast_output_dim

        # Add hidden layers with ReLU and Dropout
        for hidden_dim in projection_hidden_layers:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),  # index 0, 4, 8, etc.
                activation,                          # index 1, 5, 9, etc.
                nn.Dropout(dropout_rate)            # index 2, 6, 10, etc.
            ])
            current_dim = hidden_dim

        # Add final projection layer (index 3, 7, 11, etc.)
        layers.append(nn.Linear(current_dim, output_dim))

        # Log architecture
        arch_str = f"{ast_output_dim}"
        for hidden_dim in projection_hidden_layers:
            arch_str += f" -> {hidden_dim} (ReLU + Dropout)"
        arch_str += f" -> {output_dim}"
        logger.info(f"Projection head: {arch_str}")

        return nn.Sequential(*layers)

    def embed(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate L2-normalized embeddings."""
        # Forward through AST
        outputs = self.ast(**inputs)

        # Pool and project
        pooled = outputs.last_hidden_state.mean(dim=1)
        projected = self.projector(pooled)

        # L2 normalize
        normalized = F.normalize(projected, dim=1)

        return normalized


def load_audio_file(filepath: str, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[torch.Tensor, int]:
    """
    Load and resample audio file.

    Returns:
        Tuple of (waveform, sample_rate)
    """
    try:
        waveform, sr = torchaudio.load(filepath)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
            sr = target_sr

        return waveform.squeeze(0), sr
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        raise


def extract_random_chunks(waveform: torch.Tensor, sr: int, num_chunks: int,
                         chunk_duration: float = CHUNK_DURATION) -> List[torch.Tensor]:
    """
    Extract random non-overlapping chunks from waveform.

    Args:
        waveform: Audio tensor
        sr: Sample rate
        num_chunks: Number of chunks to extract
        chunk_duration: Duration of each chunk in seconds

    Returns:
        List of chunk tensors
    """
    chunk_samples = int(chunk_duration * sr)
    total_samples = waveform.shape[0]

    # Check if audio is long enough
    if total_samples < chunk_samples:
        logger.warning(f"Audio too short ({total_samples} < {chunk_samples} samples)")
        return [waveform]  # Return full audio as single chunk

    # Calculate valid start positions
    max_start = total_samples - chunk_samples

    if max_start < num_chunks:
        # Not enough room for non-overlapping chunks, allow overlap
        starts = [random.randint(0, max_start) for _ in range(num_chunks)]
    else:
        # Sample non-overlapping chunks
        segment_size = max_start // num_chunks
        starts = []
        for i in range(num_chunks):
            segment_start = i * segment_size
            segment_end = min((i + 1) * segment_size, max_start)
            starts.append(random.randint(segment_start, segment_end))

    # Extract chunks
    chunks = []
    for start in starts:
        end = start + chunk_samples
        chunk = waveform[start:end]
        chunks.append(chunk)

    return chunks


def process_audio_chunks(chunks: List[torch.Tensor], extractor: ASTFeatureExtractor,
                        norm_mean: float, norm_std: float) -> List[Dict[str, torch.Tensor]]:
    """
    Process audio chunks through AST feature extractor and normalize.

    Args:
        chunks: List of audio chunk tensors
        extractor: AST feature extractor
        norm_mean: Normalization mean
        norm_std: Normalization standard deviation

    Returns:
        List of input dictionaries ready for model
    """
    processed = []

    for chunk in chunks:
        # Extract features
        inputs = extractor(
            chunk.numpy(),
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=1024
        )

        # Apply normalization
        normalized_values = (inputs['input_values'] - norm_mean) / norm_std

        processed.append({
            'input_values': normalized_values
        })

    return processed


def load_model(model_path: str, config_path: str, device: torch.device) -> ImprovedASTTripletWrapper:
    """
    Load trained model from checkpoint.

    Args:
        model_path: Path to model.safetensors file
        config_path: Path to config.json file
        device: Target device (cuda/cpu)

    Returns:
        Model in eval mode
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Initialize model
    model = ImprovedASTTripletWrapper(config)

    # Load weights
    logger.info(f"Loading model weights from {model_path}")
    state_dict = load_file(model_path)

    # Try to load with strict=True first, fall back to strict=False if there are issues
    try:
        model.load_state_dict(state_dict, strict=True)
        logger.info("Model loaded with strict=True")
    except RuntimeError as e:
        logger.warning(f"Strict loading failed: {e}")
        logger.info("Attempting to load with strict=False...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
        logger.info("Model loaded with strict=False")

    # Move to device and set eval mode
    model = model.to(device)
    model.eval()

    return model


def load_centroids(centroids_path: str, device: torch.device) -> Tuple[torch.Tensor, List[str]]:
    """
    Load subgenre centroid vectors from CSV.

    Expected format:
    - First column: subgenre labels
    - Remaining columns: embedding dimensions

    Args:
        centroids_path: Path to centroids CSV file
        device: Target device

    Returns:
        Tuple of (centroid_tensor, subgenre_labels)
    """
    df = pd.read_csv(centroids_path)

    # Extract labels from first column
    labels = df.iloc[:, 0].tolist()

    # Extract embeddings from remaining columns
    embeddings = df.iloc[:, 1:].values

    # Convert to tensor and normalize
    centroids = torch.tensor(embeddings, dtype=torch.float32, device=device)
    centroids = F.normalize(centroids, dim=1)  # L2 normalize

    logger.info(f"Loaded {len(labels)} centroids with dimension {centroids.shape[1]}")

    return centroids, labels


def compute_similarities(embeddings: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarities between embeddings and centroids.

    Args:
        embeddings: Track embeddings (N x D)
        centroids: Subgenre centroids (M x D)

    Returns:
        Similarity matrix (N x M)
    """
    # Both should already be L2-normalized
    similarities = torch.mm(embeddings, centroids.t())
    return similarities


def create_similarity_playlists(track_embeddings: List[torch.Tensor],
                               track_names: List[str],
                               output_dir: str,
                               num_playlists: int = 3,
                               playlist_size: int = 10,
                               random_seed: int = 42) -> None:
    """
    Create playlists by finding nearest neighbors to randomly selected seed tracks.

    Args:
        track_embeddings: List of L2-normalized track embeddings
        track_names: List of track names corresponding to embeddings
        output_dir: Directory to save playlist files
        num_playlists: Number of playlists to create
        playlist_size: Number of tracks per playlist (including seed)
        random_seed: Random seed for reproducible playlist generation
    """
    import os
    from pathlib import Path

    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Convert to numpy for easier computation
    embeddings_matrix = torch.stack(track_embeddings).cpu().numpy()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Creating {num_playlists} playlists with {playlist_size} tracks each...")

    # Track which songs have already been used as seeds to avoid duplicates
    used_seed_indices = set()

    for playlist_idx in range(num_playlists):
        # Select a random seed track that hasn't been used yet
        available_indices = [i for i in range(len(track_names)) if i not in used_seed_indices]

        if not available_indices:
            logger.warning(f"No more unused tracks available for playlist {playlist_idx + 1}")
            break

        seed_idx = random.choice(available_indices)
        used_seed_indices.add(seed_idx)

        seed_track = track_names[seed_idx]
        seed_embedding = embeddings_matrix[seed_idx]

        # Compute cosine similarities with all other tracks
        similarities = np.dot(embeddings_matrix, seed_embedding)

        # Get indices of most similar tracks (including the seed itself)
        # argsort returns indices in ascending order, so we take the last playlist_size
        most_similar_indices = np.argsort(similarities)[-playlist_size:][::-1]  # Reverse for descending order

        # Create playlist
        playlist_tracks = []
        playlist_similarities = []

        for idx in most_similar_indices:
            playlist_tracks.append(track_names[idx])
            playlist_similarities.append(similarities[idx])

        # Save playlist to file
        playlist_name = f"playlist_{playlist_idx + 1}_seed_{Path(seed_track).stem}"
        playlist_file = Path(output_dir) / f"{playlist_name}.txt"

        with open(playlist_file, 'w', encoding='utf-8') as f:
            f.write(f"PLAYLIST {playlist_idx + 1}: {playlist_name}\n")
            f.write(f"Seed Track: {seed_track}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            for i, (track, similarity) in enumerate(zip(playlist_tracks, playlist_similarities)):
                marker = "🎵 [SEED]" if i == 0 else f"   {i:2d}."
                f.write(f"{marker} {track:<60} (similarity: {similarity:.4f})\n")

        logger.info(f"Playlist {playlist_idx + 1} saved: {playlist_file}")
        logger.info(f"  Seed: {seed_track} (similarity: {playlist_similarities[0]:.4f})")
        logger.info(f"  Most similar: {playlist_tracks[1]} (similarity: {playlist_similarities[1]:.4f})")
        logger.info(f"  Least similar: {playlist_tracks[-1]} (similarity: {playlist_similarities[-1]:.4f})")

    # Create summary file
    summary_file = Path(output_dir) / "playlists_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("SIMILARITY-BASED PLAYLISTS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total tracks processed: {len(track_names)}\n")
        f.write(f"Number of playlists: {len(used_seed_indices)}\n")
        f.write(f"Tracks per playlist: {playlist_size}\n")
        f.write(f"Random seed: {random_seed}\n\n")

        f.write("PLAYLIST OVERVIEW:\n")
        for i, seed_idx in enumerate(sorted(used_seed_indices)):
            seed_name = track_names[seed_idx]
            f.write(f"  {i+1}. {Path(seed_name).stem}\n")

    logger.info(f"Playlists summary saved: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract track embeddings and compute centroid similarities")
    parser.add_argument('--wav-dir', type=str, required=True,
                       help='Directory containing WAV files')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model.safetensors')
    parser.add_argument('--config-path', type=str, required=True,
                       help='Path to model config.json')
    parser.add_argument('--norm-stats', type=str, required=True,
                       help='Path to normalization_stats.json')
    parser.add_argument('--centroids', type=str, required=True,
                       help='Path to centroids CSV file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for similarities CSV')
    parser.add_argument('--num-chunks', type=int, default=3,
                       help='Number of chunks to extract per track')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for model inference')
    parser.add_argument('--create-playlists', action='store_true',
                       help='Create similarity-based playlists')
    parser.add_argument('--playlist-dir', type=str, default='playlists',
                       help='Directory to save playlist files')
    parser.add_argument('--num-playlists', type=int, default=3,
                       help='Number of playlists to create')
    parser.add_argument('--playlist-size', type=int, default=11,
                       help='Number of tracks per playlist (including seed)')
    parser.add_argument('--playlist-seed', type=int, default=42,
                       help='Random seed for playlist generation')

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load normalization stats
    with open(args.norm_stats, 'r') as f:
        norm_stats = json.load(f)

    # Handle both nested and flat formats
    if 'norm_stats' in norm_stats:
        # Nested format from fold preprocessing
        norm_mean = norm_stats['norm_stats']['mean']
        norm_std = norm_stats['norm_stats']['std']
    else:
        # Flat format
        norm_mean = norm_stats['mean']
        norm_std = norm_stats['std']

    logger.info(f"Loaded normalization stats: mean={norm_mean:.6f}, std={norm_std:.6f}")

    # Initialize AST feature extractor
    extractor = ASTFeatureExtractor.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')

    # Load model
    model = load_model(args.model_path, args.config_path, device)

    # Load centroids
    centroids, centroid_labels = load_centroids(args.centroids, device)

    # Find all audio files
    audio_extensions = ('.mp3', '.wav', '.flac', '.ogg')
    audio_files = []
    wav_dir = Path(args.wav_dir)

    for ext in audio_extensions:
        audio_files.extend(wav_dir.glob(f'**/*{ext}'))

    logger.info(f"Found {len(audio_files)} audio files")

    # Process each file
    track_embeddings = []
    track_names = []
    failed_files = []

    with torch.no_grad():
        for audio_path in tqdm(audio_files, desc="Processing tracks"):
            try:
                # Load audio
                waveform, sr = load_audio_file(str(audio_path))

                # Extract chunks
                chunks = extract_random_chunks(waveform, sr, args.num_chunks)

                # Process chunks
                chunk_inputs = process_audio_chunks(chunks, extractor, norm_mean, norm_std)

                # Get embeddings for each chunk
                chunk_embeddings = []
                for inputs in chunk_inputs:
                    # Move to device
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    # Get embedding
                    embedding = model.embed(inputs)
                    chunk_embeddings.append(embedding)

                # Average chunk embeddings
                track_embedding = torch.stack(chunk_embeddings).mean(dim=0)

                # Re-normalize after averaging
                track_embedding = F.normalize(track_embedding, dim=1)

                track_embeddings.append(track_embedding.cpu().squeeze())  # Remove batch dimension
                track_names.append(str(audio_path))  # Keep full path for playlists

            except Exception as e:
                logger.warning(f"Failed to process {audio_path}: {e}")
                failed_files.append(str(audio_path))
                continue

    if not track_embeddings:
        logger.error("No tracks successfully processed")
        return

    logger.info(f"Extracted embeddings for {len(track_names)} tracks")

    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files")

    # Create playlists if requested
    if args.create_playlists:
        logger.info("Creating similarity-based playlists...")
        create_similarity_playlists(
            track_embeddings=track_embeddings,
            track_names=track_names,
            output_dir=args.playlist_dir,
            num_playlists=args.num_playlists,
            playlist_size=args.playlist_size,
            random_seed=args.playlist_seed
        )

    # Compute centroid similarities if centroids provided
    if centroids is not None:
        # Stack all embeddings for centroid similarity computation
        all_embeddings = torch.stack(track_embeddings).to(device)
        similarities = compute_similarities(all_embeddings, centroids)

        # Convert to DataFrame (use just filenames for cleaner output)
        track_basenames = [Path(name).stem for name in track_names]
        similarity_df = pd.DataFrame(
            similarities.cpu().numpy(),
            index=track_basenames,
            columns=centroid_labels
        )

        # Save to CSV
        similarity_df.to_csv(args.output)
        logger.info(f"Saved similarity matrix to {args.output}")
        logger.info(f"Shape: {similarity_df.shape}")

    logger.info("Processing completed successfully!")


if __name__ == "__main__":
    main()