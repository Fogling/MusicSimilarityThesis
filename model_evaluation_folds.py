#!/usr/bin/env python3
"""
Model Evaluation Script for AST Triplet Loss Training with Fold Support

This script evaluates a trained model using preprocessed fold data by:
1. Loading a specific fold's train/test data from preprocessed directories
2. Computing embeddings for all tracks by averaging 3 chunks per track
3. Computing cluster centroids for each subgenre using TRAINING data
4. Computing cosine similarity matrices
5. Evaluating classification accuracy on TEST data
6. Saving all results (CSVs, plots) to the model directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTModel
from safetensors.torch import load_file
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps as mpl_cmaps
import matplotlib
from matplotlib.lines import Line2D
import umap
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score

from config import ExperimentConfig, load_or_create_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

matplotlib.use("Agg")  # non-interactive backend


class ImprovedASTTripletWrapper(nn.Module):
    """
    AST wrapper identical to the one used in training
    """

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

        # Use projection_hidden_layers if available, otherwise fall back to hidden_sizes
        if hasattr(config.model, 'projection_hidden_layers') and config.model.projection_hidden_layers:
            hidden_sizes = config.model.projection_hidden_layers
            activation = config.model.projection_activation
            dropout_rate = config.model.projection_dropout_rate
        else:
            # Legacy format - exclude last element as it was the output_dim
            hidden_sizes = config.model.hidden_sizes[:-1] if len(config.model.hidden_sizes) > 1 else config.model.hidden_sizes
            activation = config.model.activation
            dropout_rate = config.model.dropout_rate

        for i, hidden_size in enumerate(hidden_sizes):
            # Format activation name properly
            activation_class = activation.lower()
            if activation_class == 'relu':
                activation_layer = nn.ReLU()
            elif activation_class == 'gelu':
                activation_layer = nn.GELU()
            elif activation_class == 'tanh':
                activation_layer = nn.Tanh()
            else:
                activation_layer = getattr(nn, activation.capitalize())()

            projection_layers.extend([
                nn.Linear(input_dim, hidden_size),
                activation_layer,
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_size

        # Final output layer
        projection_layers.append(nn.Linear(input_dim, config.model.output_dim))

        self.projector = nn.Sequential(*projection_layers)

        logger.info(f"Model architecture: {self.ast.config.hidden_size} -> "
                   f"{' -> '.join(map(str, hidden_sizes))} -> {config.model.output_dim}")

    def forward(self, input_values):
        try:
            outputs = self.ast(input_values=input_values)
            last_hidden_state = outputs.last_hidden_state
            pooled = last_hidden_state.mean(dim=1)
            projected = self.projector(pooled)
            return F.normalize(projected, p=2, dim=1)
        except Exception as e:
            logger.error(f"Forward pass error: {e}")
            raise


def load_model(model_dir: Path, device: torch.device) -> tuple[nn.Module, ExperimentConfig]:
    """Load the trained model from the given directory"""
    config_path = model_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config using ExperimentConfig.load method
    config = ExperimentConfig.load(str(config_path))

    # Initialize trained model and load weights
    model_path = model_dir / "model.safetensors"

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = ImprovedASTTripletWrapper(config)

    # Load weights
    state_dict = load_file(str(model_path))
    model.load_state_dict(state_dict)

    logger.info(f"Trained model loaded from {model_dir}")

    model.to(device)
    model.eval()

    return model, config


def load_fold_data(preprocessed_dir: Path, fold_idx: int) -> Tuple[Dict, Dict]:
    """
    Load train and test data from preprocessed fold directory

    Returns:
        Tuple of (train_data, test_data) where each is:
        {subgenre: {track_name: [chunk_paths]}}
    """
    fold_dir = preprocessed_dir / f"fold_{fold_idx}"
    train_dir = fold_dir / "train_chunks"
    test_dir = fold_dir / "test_chunks"

    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

    def load_partition(partition_dir: Path) -> Dict[str, Dict[str, List[str]]]:
        """Load all chunks from a partition directory"""
        data = defaultdict(lambda: defaultdict(list))

        for subgenre_dir in partition_dir.iterdir():
            if not subgenre_dir.is_dir():
                continue

            subgenre = subgenre_dir.name

            # Group chunks by track
            track_chunks = defaultdict(dict)

            for chunk_file in subgenre_dir.glob("*.pt"):
                # Extract track name and chunk number
                filename = chunk_file.stem

                # Handle different chunk naming patterns
                if '_chunk' in filename:
                    parts = filename.rsplit('_chunk', 1)
                    if len(parts) == 2:
                        track_name = parts[0]
                        try:
                            chunk_num = int(parts[1])
                        except ValueError:
                            continue
                    else:
                        continue
                else:
                    continue

                track_chunks[track_name][chunk_num] = str(chunk_file)

            # Convert to list format (ensuring order)
            for track_name, chunks_dict in track_chunks.items():
                # Get maximum chunk number to determine array size
                max_chunk = max(chunks_dict.keys())
                chunks_list = [None] * max_chunk

                for chunk_num, path in chunks_dict.items():
                    chunks_list[chunk_num - 1] = path

                # Only include tracks with all chunks present
                if all(chunk is not None for chunk in chunks_list):
                    data[subgenre][track_name] = chunks_list

        return dict(data)

    logger.info(f"Loading fold {fold_idx} data from {fold_dir}")

    train_data = load_partition(train_dir)
    test_data = load_partition(test_dir)

    # Log statistics
    train_tracks_total = sum(len(tracks) for tracks in train_data.values())
    test_tracks_total = sum(len(tracks) for tracks in test_data.values())

    logger.info(f"Loaded fold {fold_idx}: {train_tracks_total} train tracks, {test_tracks_total} test tracks")

    for subgenre in train_data:
        logger.info(f"  Train - {subgenre}: {len(train_data[subgenre])} tracks")
    for subgenre in test_data:
        logger.info(f"  Test - {subgenre}: {len(test_data[subgenre])} tracks")

    return train_data, test_data


def compute_track_embeddings(model: nn.Module,
                            track_data: Dict[str, Dict[str, List[str]]],
                            device: torch.device,
                            batch_size: int = 4) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute embeddings for all tracks by averaging their chunks
    """
    track_embeddings = defaultdict(dict)

    logger.info("Computing track embeddings...")

    with torch.no_grad():
        for subgenre, tracks in track_data.items():
            logger.info(f"Processing {subgenre} ({len(tracks)} tracks)")

            for track_name, chunk_paths in tqdm(tracks.items(), desc=f"Processing {subgenre}"):
                chunk_embeddings = []

                # Load and process each chunk
                for chunk_path in chunk_paths:
                    if chunk_path is None:
                        continue

                    try:
                        # Load chunk data
                        chunk_data = torch.load(chunk_path, map_location='cpu', weights_only=True)
                        input_values = chunk_data['input_values']

                        # Ensure proper tensor format and move to device
                        if len(input_values.shape) == 2:
                            input_values = input_values.unsqueeze(0)  # Add batch dimension

                        input_values = input_values.to(device)

                        # Get embedding
                        embedding = model(input_values)
                        chunk_embeddings.append(embedding.cpu().numpy())

                    except Exception as e:
                        logger.error(f"Error processing {chunk_path}: {e}")
                        continue

                if chunk_embeddings:
                    # Average the chunk embeddings
                    avg_embedding = np.mean(chunk_embeddings, axis=0).squeeze()
                    # Re-normalize after averaging
                    avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-12)
                    track_embeddings[subgenre][track_name] = avg_embedding
                else:
                    logger.warning(f"Track {track_name} has no valid chunks")

    return dict(track_embeddings)


def compute_subgenre_centroids(embeddings: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Compute cluster centroids for each subgenre
    """
    centroids = {}

    logger.info("Computing subgenre centroids from training data...")

    for subgenre, tracks in embeddings.items():
        if tracks:
            track_embeddings = list(tracks.values())
            centroid = np.mean(track_embeddings, axis=0)
            # L2 normalize the centroid
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            centroids[subgenre] = centroid
            logger.info(f"  {subgenre}: centroid computed from {len(tracks)} tracks")
        else:
            logger.warning(f"  {subgenre}: no tracks available for centroid computation")

    return centroids


def compute_cosine_similarity_matrix(centroids: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute cosine similarity matrix between centroids
    """
    subgenres = sorted(list(centroids.keys()))
    n = len(subgenres)
    similarity_matrix = np.zeros((n, n))

    for i, subgenre1 in enumerate(subgenres):
        for j, subgenre2 in enumerate(subgenres):
            # Both centroids are L2-normalized, so dot product = cosine similarity
            similarity = np.dot(centroids[subgenre1], centroids[subgenre2])
            similarity_matrix[i, j] = similarity

    return similarity_matrix, subgenres


def evaluate_classification(test_embeddings: Dict[str, Dict[str, np.ndarray]],
                           centroids: Dict[str, np.ndarray]) -> Dict[str, any]:
    """
    Evaluate classification accuracy by checking which centroid each test track is closest to
    """
    results = {
        'total_tracks': 0,
        'correct_predictions': 0,
        'accuracy': 0.0,
        'per_subgenre': {},
        'predictions': []
    }

    logger.info("Evaluating classification accuracy on test data...")

    for true_subgenre, tracks in test_embeddings.items():
        subgenre_correct = 0
        subgenre_total = len(tracks)
        subgenre_predictions = []

        for track_name, track_embedding in tracks.items():
            # Compute similarities with all centroids
            similarities = {}
            for centroid_subgenre, centroid in centroids.items():
                similarity = np.dot(track_embedding, centroid)
                similarities[centroid_subgenre] = similarity

            # Find the closest centroid
            predicted_subgenre = max(similarities, key=similarities.get)
            max_similarity = similarities[predicted_subgenre]

            # Check if prediction is correct
            is_correct = predicted_subgenre == true_subgenre
            if is_correct:
                subgenre_correct += 1

            prediction = {
                'track_name': track_name,
                'true_subgenre': true_subgenre,
                'predicted_subgenre': predicted_subgenre,
                'max_similarity': max_similarity,
                'all_similarities': similarities.copy(),
                'correct': is_correct
            }
            subgenre_predictions.append(prediction)
            results['predictions'].append(prediction)

        # Store per-subgenre results
        subgenre_accuracy = subgenre_correct / subgenre_total if subgenre_total > 0 else 0
        results['per_subgenre'][true_subgenre] = {
            'correct': subgenre_correct,
            'total': subgenre_total,
            'accuracy': subgenre_accuracy,
            'predictions': subgenre_predictions
        }

        results['total_tracks'] += subgenre_total
        results['correct_predictions'] += subgenre_correct

    # Overall accuracy
    results['accuracy'] = results['correct_predictions'] / results['total_tracks'] if results['total_tracks'] > 0 else 0

    return results


def flatten_embeddings_to_matrix(embeddings_dict: Dict[str, Dict[str, np.ndarray]]):
    """
    Convert nested dict {subgenre: {track_name: emb}} to:
      X: (n_samples, d) np.ndarray
      y_true: (n_samples,) np.ndarray of integer labels
      y_names: list[str] subgenre names per sample
      track_ids: list[str] "subgenre/track_name" identifiers
      label_encoder: fitted LabelEncoder
    """
    X_list, y_names, track_ids = [], [], []
    for sg, tracks in embeddings_dict.items():
        for tname, emb in tracks.items():
            X_list.append(emb)
            y_names.append(sg)
            track_ids.append(f"{sg}/{tname}")

    if len(X_list) == 0:
        return None, None, None, None, None

    X = np.vstack(X_list).astype(np.float32)
    # Safety: re-normalize to unit length
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms

    le = LabelEncoder()
    y_true = le.fit_transform(y_names)
    return X, y_true, y_names, track_ids, le


def run_kmeans_and_metrics(
    X: np.ndarray,
    y_true: Optional[np.ndarray],
    n_clusters: int,
    random_state: int = 42,
    silhouette_metric: str = "cosine"
) -> Dict[str, any]:
    """
    Run Euclidean k-means on L2-normalized X and compute metrics
    """
    if X is None or len(X) == 0:
        raise ValueError("No embeddings to cluster.")

    if n_clusters < 2:
        raise ValueError("Need at least 2 clusters for k-means/silhouette.")

    # sklearn KMeans uses Euclidean; with unit-norm X this approximates spherical k-means
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    y_pred = km.fit_predict(X)

    # Silhouette score
    sil = None
    try:
        if len(np.unique(y_pred)) > 1 and X.shape[0] > n_clusters:
            sil = silhouette_score(X, y_pred, metric=silhouette_metric)
    except Exception as e:
        logger.warning(f"Silhouette computation failed: {e}")

    nmi = ari = None
    if y_true is not None:
        try:
            nmi = normalized_mutual_info_score(y_true, y_pred)
            ari = adjusted_rand_score(y_true, y_pred)
        except Exception as e:
            logger.warning(f"NMI/ARI computation failed: {e}")

    # Normalize k-means centroids for later use
    km_centroids = km.cluster_centers_.copy()
    km_centroids /= (np.linalg.norm(km_centroids, axis=1, keepdims=True) + 1e-12)

    return {
        "kmeans_model": km,
        "kmeans_centroids": km_centroids,
        "labels_pred": y_pred,
        "silhouette_cosine": sil,
        "nmi": nmi,
        "ari": ari
    }


def create_enhanced_umap_visualization(embeddings: Dict[str, Dict[str, np.ndarray]],
                                     subgenre_centroids: Dict[str, np.ndarray],
                                     kmeans_results: Dict[str, any],
                                     output_dir: Path,
                                     title_suffix: str = "") -> None:
    """
    Create enhanced UMAP visualizations with improved styling and k-means centroids
    """
    logger.info("Creating enhanced UMAP visualizations...")

    # Flatten embeddings for tracks only
    X_list, y_names, track_ids = [], [], []
    for sg, tracks in embeddings.items():
        for track_name, emb in tracks.items():
            X_list.append(emb)
            y_names.append(sg)
            track_ids.append(f"{sg}/{track_name}")

    if len(X_list) == 0:
        logger.warning("No embeddings available for UMAP.")
        return

    X_tracks = np.vstack(X_list).astype(np.float32)
    # Safety: re-normalize to unit length
    norms = np.linalg.norm(X_tracks, axis=1, keepdims=True) + 1e-12
    X_tracks = X_tracks / norms

    # Subgenre order for consistent colors
    subgenres = sorted(list(set(y_names)))

    # Color mapping with more distinct colors
    distinct_colors = [
        '#8B00FF',  # Electric Purple
        '#FF1493',  # Deep Pink
        '#DC143C',  # Crimson Red
        '#00CED1',  # Dark Turquoise
        '#32CD32',  # Lime Green
        '#FF8C00',  # Dark Orange
        '#4169E1',  # Royal Blue
        '#FFD700',  # Gold
        '#FF6347',  # Tomato
        '#9370DB'   # Medium Purple
    ]
    color_map = {sg: distinct_colors[i % len(distinct_colors)] for i, sg in enumerate(subgenres)}

    # Compute subgenre centroids in embedding space
    true_centroids = []
    true_centroid_labels = []
    for sg in subgenres:
        if sg in subgenre_centroids:
            true_centroids.append(subgenre_centroids[sg])
            true_centroid_labels.append(sg)

    true_centroids_array = np.vstack(true_centroids) if true_centroids else None

    # Get k-means centroids (already normalized in run_kmeans_and_metrics)
    kmeans_centroids = kmeans_results.get("kmeans_centroids") if kmeans_results else None

    # UMAP projection with all data points
    all_points_for_umap = [X_tracks]
    if true_centroids_array is not None:
        all_points_for_umap.append(true_centroids_array)
    if kmeans_centroids is not None:
        all_points_for_umap.append(kmeans_centroids)

    X_all = np.vstack(all_points_for_umap)

    # UMAP fitting with optimized parameters for cluster separation
    n_samples = X_all.shape[0]
    # Use higher n_neighbors for better global structure and cluster separation
    n_neighbors = max(2, min(30, n_samples - 1))

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.3,  # Higher min_dist for more spread out clusters
        spread=3.0,    # Much higher spread for better separation
        metric="cosine",
        random_state=42,
        n_epochs=500,  # More epochs for better convergence
        learning_rate=1.0,
        repulsion_strength=2.0  # Higher repulsion to push clusters apart
    )
    X_all_2d = reducer.fit_transform(X_all)

    # Split back the projected points
    track_start = 0
    track_end = len(X_tracks)
    X_tracks_2d = X_all_2d[track_start:track_end]

    current_idx = track_end
    if true_centroids_array is not None:
        true_start = current_idx
        true_end = current_idx + len(true_centroids_array)
        true_centroids_2d = X_all_2d[true_start:true_end]
        current_idx = true_end
    else:
        true_centroids_2d = None

    if kmeans_centroids is not None:
        km_start = current_idx
        km_end = current_idx + len(kmeans_centroids)
        kmeans_centroids_2d = X_all_2d[km_start:km_end]
    else:
        kmeans_centroids_2d = None

    def create_base_plot(ax):
        """Create base scatter plot of tracks"""
        for sg in subgenres:
            idx = [i for i, name in enumerate(y_names) if name == sg]
            if idx:
                ax.scatter(
                    X_tracks_2d[idx, 0], X_tracks_2d[idx, 1],
                    s=40,  # Slightly larger for better visibility
                    marker="o",
                    c=[color_map[sg]],
                    edgecolors="black",  # Dark border as requested
                    linewidths=0.8,     # Thicker border for visibility
                    alpha=0.8,
                    label=f"{sg}"
                )
        ax.set_xlabel("UMAP Dimension 1", fontsize=12)
        ax.set_ylabel("UMAP Dimension 2", fontsize=12)
        ax.grid(True, alpha=0.2)

    def create_legend(ax, show_subgenre_centroids=False, show_kmeans_centroids=False):
        """Create legend with proper ordering"""
        handles = []

        # Add track handles
        for sg in subgenres:
            if any(name == sg for name in y_names):
                handles.append(Line2D([0], [0], marker='o', linestyle='',
                                    markersize=6, markerfacecolor=color_map[sg],
                                    markeredgecolor='black', markeredgewidth=0.8,
                                    label=f"{sg}"))

        # Add centroid handles
        if show_subgenre_centroids:
            handles.append(Line2D([0], [0], marker='*', linestyle='',
                                markersize=12, markerfacecolor='none',
                                markeredgecolor='black', markeredgewidth=1.0,
                                label="Subgenre centroid"))

        if show_kmeans_centroids:
            handles.append(Line2D([0], [0], marker='X', linestyle='',
                                markersize=10, markerfacecolor='black',
                                markeredgecolor='white', markeredgewidth=1.0,
                                label="K-means centroid"))

        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
                 frameon=True, borderaxespad=0.0)

    # Plot 1: Tracks + Subgenre centroids
    fig, ax = plt.subplots(figsize=(12, 8))
    create_base_plot(ax)

    if true_centroids_2d is not None:
        # Draw subgenre centroids with matching colors
        for i, (sg, (cx, cy)) in enumerate(zip(true_centroid_labels, true_centroids_2d)):
            ax.scatter([cx], [cy], marker='*', s=300, linewidths=1.2,
                      edgecolors='black', facecolors=color_map[sg], alpha=0.95)

    ax.set_title(f"UMAP: Tracks + Subgenre Centroids{title_suffix}", fontsize=14, fontweight="bold")
    create_legend(ax, show_subgenre_centroids=True, show_kmeans_centroids=False)

    output_path = output_dir / "umap_tracks_subgenre_centroids.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"UMAP (tracks + subgenre centroids) saved to: {output_path}")

    # Plot 2: Tracks + K-means centroids
    if kmeans_centroids_2d is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        create_base_plot(ax)

        # Draw k-means centroids as distinctive X markers
        ax.scatter(kmeans_centroids_2d[:, 0], kmeans_centroids_2d[:, 1],
                  marker='X', s=200, linewidths=1.0,
                  edgecolors='white', facecolors='black', alpha=0.95)

        ax.set_title(f"UMAP: Tracks + K-means Centroids{title_suffix}", fontsize=14, fontweight="bold")
        create_legend(ax, show_subgenre_centroids=False, show_kmeans_centroids=True)

        output_path = output_dir / "umap_tracks_kmeans_centroids.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"UMAP (tracks + k-means centroids) saved to: {output_path}")

    # Plot 3: Tracks + Both centroid types
    if true_centroids_2d is not None and kmeans_centroids_2d is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        create_base_plot(ax)

        # Draw subgenre centroids
        for i, (sg, (cx, cy)) in enumerate(zip(true_centroid_labels, true_centroids_2d)):
            ax.scatter([cx], [cy], marker='*', s=300, linewidths=1.2,
                      edgecolors='black', facecolors=color_map[sg], alpha=0.95)

        # Draw k-means centroids
        ax.scatter(kmeans_centroids_2d[:, 0], kmeans_centroids_2d[:, 1],
                  marker='X', s=200, linewidths=1.0,
                  edgecolors='white', facecolors='black', alpha=0.95)

        ax.set_title(f"UMAP: Tracks + All Centroids{title_suffix}", fontsize=14, fontweight="bold")
        create_legend(ax, show_subgenre_centroids=True, show_kmeans_centroids=True)

        output_path = output_dir / "umap_tracks_all_centroids.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"UMAP (tracks + all centroids) saved to: {output_path}")


def analyze_cluster_assignments(embeddings: Dict[str, Dict[str, np.ndarray]],
                               subgenre_centroids: Dict[str, np.ndarray],
                               kmeans_results: Dict[str, any]) -> Dict[str, any]:
    """
    Analyze and compare cluster assignments between subgenre centroids and k-means centroids
    """
    logger.info("Analyzing cluster assignments...")

    # Flatten embeddings
    X_list, y_names, track_ids = [], [], []
    for sg, tracks in embeddings.items():
        for track_name, emb in tracks.items():
            X_list.append(emb)
            y_names.append(sg)
            track_ids.append(f"{sg}/{track_name}")

    if len(X_list) == 0:
        return {}

    X = np.vstack(X_list).astype(np.float32)
    # Re-normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms

    # Get k-means assignments
    kmeans_labels = kmeans_results.get("labels_pred", [])
    if len(kmeans_labels) == 0:
        return {}

    # Compute subgenre centroid assignments
    subgenre_assignments = []
    subgenre_list = sorted(list(subgenre_centroids.keys()))

    for emb in X:
        similarities = {}
        for sg, centroid in subgenre_centroids.items():
            similarity = np.dot(emb, centroid)
            similarities[sg] = similarity

        # Assign to closest subgenre centroid
        closest_sg = max(similarities, key=similarities.get)
        subgenre_assignments.append(subgenre_list.index(closest_sg))

    # Compare assignments
    subgenre_assignments = np.array(subgenre_assignments)

    # Create assignment comparison DataFrame
    assignment_comparison = pd.DataFrame({
        'track_id': track_ids,
        'true_subgenre': y_names,
        'subgenre_centroid_assignment': [subgenre_list[i] for i in subgenre_assignments],
        'kmeans_cluster_assignment': kmeans_labels
    })

    # Compute agreement between subgenre and k-means assignments
    agreement_matrix = confusion_matrix(subgenre_assignments, kmeans_labels)

    # Calculate cluster purity for both assignment methods
    from sklearn.metrics import adjusted_mutual_info_score

    # True labels encoded
    le = LabelEncoder()
    true_labels_encoded = le.fit_transform(y_names)

    # Metrics
    subgenre_ami = adjusted_mutual_info_score(true_labels_encoded, subgenre_assignments)
    kmeans_ami = adjusted_mutual_info_score(true_labels_encoded, kmeans_labels)
    assignment_agreement = adjusted_mutual_info_score(subgenre_assignments, kmeans_labels)

    results = {
        'assignment_comparison': assignment_comparison,
        'agreement_matrix': agreement_matrix,
        'subgenre_centroid_ami': subgenre_ami,
        'kmeans_ami': kmeans_ami,
        'assignment_agreement_ami': assignment_agreement,
        'subgenre_list': subgenre_list
    }

    return results


def save_results(model_dir: Path, fold_idx: int, results: Dict[str, any]) -> None:
    """
    Save all evaluation results to CSV files in the model directory
    """
    output_dir = model_dir / f"evaluation_fold{fold_idx}"
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Saving results to {output_dir}")

    # Save similarity matrix
    if 'similarity_matrix' in results:
        sim_df = pd.DataFrame(
            results['similarity_matrix'],
            index=results['subgenres'],
            columns=results['subgenres']
        )
        sim_df.to_csv(output_dir / "centroid_similarity_matrix.csv")
        logger.info(f"  Saved centroid similarity matrix")

    # Save classification results
    if 'classification' in results:
        class_results = results['classification']

        # Overall accuracy
        with open(output_dir / "classification_accuracy.txt", 'w') as f:
            f.write(f"Overall Accuracy: {class_results['accuracy']:.4f}\n")
            f.write(f"Correct: {class_results['correct_predictions']}/{class_results['total_tracks']}\n\n")

            f.write("Per-Subgenre Accuracy:\n")
            for sg, data in class_results['per_subgenre'].items():
                f.write(f"  {sg}: {data['accuracy']:.4f} ({data['correct']}/{data['total']})\n")

        # Detailed predictions
        predictions_data = []
        for pred in class_results['predictions']:
            predictions_data.append({
                'track': pred['track_name'],
                'true_subgenre': pred['true_subgenre'],
                'predicted_subgenre': pred['predicted_subgenre'],
                'similarity': pred['max_similarity'],
                'correct': pred['correct']
            })

        pred_df = pd.DataFrame(predictions_data)
        pred_df.to_csv(output_dir / "classification_predictions.csv", index=False)
        logger.info(f"  Saved classification predictions")

    # Save centroids
    if 'centroids' in results:
        centroids_data = []
        for sg, centroid in results['centroids'].items():
            row = {'subgenre': sg}
            for i, val in enumerate(centroid):
                row[f'dim_{i}'] = val
            centroids_data.append(row)

        cent_df = pd.DataFrame(centroids_data)
        cent_df.to_csv(output_dir / "subgenre_centroids.csv", index=False)
        logger.info(f"  Saved subgenre centroids")

    # Save k-means results if available
    if 'kmeans' in results:
        km_results = results['kmeans']

        # K-means metrics
        with open(output_dir / "kmeans_metrics.txt", 'w') as f:
            f.write(f"Silhouette Score (cosine): {km_results.get('silhouette_cosine', 'N/A'):.4f}\n")
            f.write(f"NMI: {km_results.get('nmi', 'N/A'):.4f}\n")
            f.write(f"ARI: {km_results.get('ari', 'N/A'):.4f}\n")

        # K-means centroids
        if 'kmeans_centroids' in km_results:
            km_cent_data = []
            for i, centroid in enumerate(km_results['kmeans_centroids']):
                row = {'cluster_id': i}
                for j, val in enumerate(centroid):
                    row[f'dim_{j}'] = val
                km_cent_data.append(row)

            km_cent_df = pd.DataFrame(km_cent_data)
            km_cent_df.to_csv(output_dir / "kmeans_centroids.csv", index=False)

        logger.info(f"  Saved k-means results")

    # Save cluster assignment analysis if available
    if 'cluster_assignments' in results:
        cluster_analysis = results['cluster_assignments']

        # Save assignment comparison
        if 'assignment_comparison' in cluster_analysis:
            cluster_analysis['assignment_comparison'].to_csv(
                output_dir / "cluster_assignment_comparison.csv", index=False
            )

        # Save agreement matrix
        if 'agreement_matrix' in cluster_analysis:
            agreement_df = pd.DataFrame(
                cluster_analysis['agreement_matrix'],
                index=[f"SG_{i}" for i in range(cluster_analysis['agreement_matrix'].shape[0])],
                columns=[f"KM_{i}" for i in range(cluster_analysis['agreement_matrix'].shape[1])]
            )
            agreement_df.to_csv(output_dir / "centroid_kmeans_agreement_matrix.csv")

        # Save assignment metrics
        with open(output_dir / "cluster_assignment_metrics.txt", 'w') as f:
            subgenre_ami = cluster_analysis.get('subgenre_ami', None)
            kmeans_ami = cluster_analysis.get('kmeans_ami', None)
            agreement_ami = cluster_analysis.get('assignment_agreement_ami', None)

            subgenre_str = f"{subgenre_ami:.4f}" if subgenre_ami is not None else "N/A"
            kmeans_str = f"{kmeans_ami:.4f}" if kmeans_ami is not None else "N/A"
            agreement_str = f"{agreement_ami:.4f}" if agreement_ami is not None else "N/A"

            f.write(f"Subgenre Centroid AMI: {subgenre_str}\n")
            f.write(f"K-means AMI: {kmeans_str}\n")
            f.write(f"Assignment Agreement AMI: {agreement_str}\n")

        logger.info(f"  Saved cluster assignment analysis")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AST Triplet Model with Fold Support")
    parser.add_argument("model_dir", type=str,
                       help="Path to the trained model directory containing model.safetensors and config.json")
    parser.add_argument("--preprocessed_dir", type=str, default="precomputed_AST",
                       help="Path to preprocessed data directory containing fold_N subdirectories")
    parser.add_argument("--fold", type=int, required=True,
                       help="Fold index to evaluate (e.g., 0, 1, 2...)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for embedding computation")
    parser.add_argument("--no_umap", action="store_true",
                       help="Skip UMAP visualization generation")

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup paths
    model_dir = Path(args.model_dir)
    preprocessed_dir = Path(args.preprocessed_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if not preprocessed_dir.exists():
        raise FileNotFoundError(f"Preprocessed data directory not found: {preprocessed_dir}")

    try:
        # 1. Load model
        logger.info("Loading trained model...")
        model, config = load_model(model_dir, device)

        # 2. Load fold data
        logger.info(f"Loading fold {args.fold} data...")
        train_data, test_data = load_fold_data(preprocessed_dir, args.fold)

        # 3. Compute embeddings
        logger.info("Computing training embeddings...")
        train_embeddings = compute_track_embeddings(model, train_data, device, args.batch_size)

        logger.info("Computing test embeddings...")
        test_embeddings = compute_track_embeddings(model, test_data, device, args.batch_size)

        # 4. Compute centroids from TRAINING data
        logger.info("Computing subgenre centroids from training data...")
        centroids = compute_subgenre_centroids(train_embeddings)

        # 5. Compute similarity matrix
        logger.info("Computing centroid similarity matrix...")
        similarity_matrix, subgenres = compute_cosine_similarity_matrix(centroids)

        # 6. Evaluate classification on TEST data
        logger.info("Evaluating classification on test data...")
        classification_results = evaluate_classification(test_embeddings, centroids)

        # 7. Clustering metrics on test embeddings
        logger.info("Computing clustering metrics...")
        X_test, y_test, y_names_test, track_ids_test, le = flatten_embeddings_to_matrix(test_embeddings)

        kmeans_results = None
        if X_test is not None and len(centroids) > 1:
            kmeans_results = run_kmeans_and_metrics(
                X_test, y_test,
                n_clusters=len(centroids),
                random_state=42
            )

        # 8. Cluster assignment analysis
        cluster_assignment_analysis = None
        if kmeans_results:
            cluster_assignment_analysis = analyze_cluster_assignments(
                test_embeddings, centroids, kmeans_results
            )

        # 9. Prepare results dictionary
        results = {
            'similarity_matrix': similarity_matrix,
            'subgenres': subgenres,
            'classification': classification_results,
            'centroids': centroids,
            'kmeans': kmeans_results,
            'cluster_assignments': cluster_assignment_analysis
        }

        # 10. Save all results
        save_results(model_dir, args.fold, results)

        # 11. Print summary
        print("\n" + "="*80)
        print(f"EVALUATION RESULTS - FOLD {args.fold}")
        print("="*80)

        print("\n1. CLASSIFICATION ACCURACY:")
        print(f"  Overall: {classification_results['accuracy']:.4f} "
              f"({classification_results['correct_predictions']}/{classification_results['total_tracks']})")

        print("\n  Per-subgenre:")
        for sg, data in classification_results['per_subgenre'].items():
            print(f"    {sg}: {data['accuracy']:.4f} ({data['correct']}/{data['total']})")

        if kmeans_results:
            print("\n2. CLUSTERING METRICS:")
            sil_score = kmeans_results.get('silhouette_cosine', None)
            nmi_score = kmeans_results.get('nmi', None)
            ari_score = kmeans_results.get('ari', None)

            sil_str = f"{sil_score:.4f}" if sil_score is not None else "N/A"
            nmi_str = f"{nmi_score:.4f}" if nmi_score is not None else "N/A"
            ari_str = f"{ari_score:.4f}" if ari_score is not None else "N/A"

            print(f"  Silhouette (cosine): {sil_str}")
            print(f"  NMI: {nmi_str}")
            print(f"  ARI: {ari_str}")

        if cluster_assignment_analysis:
            print("\n3. CLUSTER ASSIGNMENT ANALYSIS:")
            subgenre_ami = cluster_assignment_analysis.get('subgenre_ami', None)
            kmeans_ami = cluster_assignment_analysis.get('kmeans_ami', None)
            agreement_ami = cluster_assignment_analysis.get('assignment_agreement_ami', None)

            subgenre_str = f"{subgenre_ami:.4f}" if subgenre_ami is not None else "N/A"
            kmeans_str = f"{kmeans_ami:.4f}" if kmeans_ami is not None else "N/A"
            agreement_str = f"{agreement_ami:.4f}" if agreement_ami is not None else "N/A"

            print(f"  Subgenre Centroid AMI: {subgenre_str}")
            print(f"  K-means AMI: {kmeans_str}")
            print(f"  Assignment Agreement AMI: {agreement_str}")

        # 12. Generate enhanced UMAP visualizations
        if not args.no_umap:
            logger.info("Generating enhanced UMAP visualizations...")

            # Create output directory for evaluation results
            eval_output_dir = model_dir / f"evaluation_fold{args.fold}"

            # Test embeddings with enhanced visualization
            create_enhanced_umap_visualization(
                test_embeddings,
                centroids,
                kmeans_results if kmeans_results else {},
                eval_output_dir,
                title_suffix=f" - Test Data (Fold {args.fold})"
            )

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()