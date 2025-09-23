#!/usr/bin/env python3
"""
Model Evaluation Script for AST Triplet Loss Training

This script evaluates a trained model by:
1. Loading train/test splits from the model folder
2. Computing embeddings for all tracks by averaging 3 chunks per track
3. Computing cluster centroids for each subgenre
4. Computing cosine similarity matrices
5. Evaluating classification accuracy
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
from matplotlib import colormaps as mpl_cmaps  # modern API
import matplotlib
from matplotlib.lines import Line2D
import umap
import seaborn as sns
import umap
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


matplotlib.use("Agg")  # non-interactive; prevents blocking windows


class BaselineASTWrapper(nn.Module):
    """
    Baseline AST model wrapper that uses only the pretrained AST without projection head.
    Used for baseline comparisons.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config

        try:
            logger.info(f"Loading baseline pretrained model: {config.model.pretrained_model}")
            self.ast = ASTModel.from_pretrained(config.model.pretrained_model)
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            raise

        logger.info(f"Baseline model initialized with {self.ast.config.hidden_size}D embeddings (no projection)")

    def forward(self, input_values):
        try:
            outputs = self.ast(input_values=input_values)
            last_hidden_state = outputs.last_hidden_state
            pooled = last_hidden_state.mean(dim=1)
            # Return L2-normalized embeddings without projection
            return F.normalize(pooled, p=2, dim=1)
        except Exception as e:
            logger.error(f"Forward pass error: {e}")
            raise


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
            # Format activation name properly (relu -> ReLU)
            activation_class = activation.lower()
            if activation_class == 'relu':
                activation_layer = nn.ReLU()
            elif activation_class == 'gelu':
                activation_layer = nn.GELU()
            elif activation_class == 'tanh':
                activation_layer = nn.Tanh()
            else:
                # Fallback to getattr with proper casing
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


def load_model(model_dir: Path, device: torch.device, use_baseline: bool = False) -> tuple[nn.Module, ExperimentConfig]:
    """Load the trained model or baseline from the given directory"""
    config_path = model_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config using ExperimentConfig.load method
    config = ExperimentConfig.load(str(config_path))

    if use_baseline:
        # Initialize baseline model (no trained weights)
        model = BaselineASTWrapper(config)
        logger.info(f"Baseline model loaded (pretrained AST only)")
    else:
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


def load_splits(model_dir: Path) -> Tuple[List, List]:
    """Load train/test splits from the model directory"""
    splits_path = model_dir / "splits.json"
    
    if not splits_path.exists():
        raise FileNotFoundError(f"splits.json not found in {model_dir}")
    
    logger.info(f"Loading splits from {splits_path}")
    
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    return splits["train_split"], splits["test_split"]


def extract_track_info(file_path: str) -> Tuple[str, str, int]:
    """Extract subgenre, track name, and chunk number from file path"""
    # Expected format: precomputed_AST/Subgenre/track_name_chunkN.pt
    parts = file_path.split('/')
    if len(parts) < 3:
        raise ValueError(f"Invalid file path format: {file_path}")
    
    subgenre = parts[1]
    filename = parts[2]
    
    # Extract chunk number
    if filename.endswith('.pt'):
        filename = filename[:-3]  # Remove .pt extension
    
    if filename.endswith('_chunk1'):
        track_name = filename[:-7]  # Remove _chunk1
        chunk_num = 1
    elif filename.endswith('_chunk2'):
        track_name = filename[:-7]  # Remove _chunk2
        chunk_num = 2
    elif filename.endswith('_chunk3'):
        track_name = filename[:-7]  # Remove _chunk3
        chunk_num = 3
    else:
        raise ValueError(f"Invalid chunk format in filename: {filename}")
    
    return subgenre, track_name, chunk_num


def load_and_process_data(splits: List, chunks_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load all chunks and organize by track.
    Returns: {subgenre: {track_name: [chunk1_path, chunk2_path, chunk3_path]}}
    """
    track_data = defaultdict(lambda: defaultdict(lambda: [None, None, None]))
    
    logger.info("Organizing track chunks...")
    
    for triplet in splits:
        if len(triplet) < 4:
            continue
        
        anchor_path, pos_path, neg_path, label = triplet
        
        # Process all three paths
        for path in [anchor_path, pos_path, neg_path]:
            try:
                subgenre, track_name, chunk_num = extract_track_info(path)
                full_path = chunks_dir.parent / path
                
                if full_path.exists():
                    track_data[subgenre][track_name][chunk_num - 1] = str(full_path)
                else:
                    logger.warning(f"File not found: {full_path}")
            except Exception as e:
                logger.warning(f"Error processing path {path}: {e}")
    
    # Convert to regular dict and filter complete tracks (all 3 chunks)
    result = {}
    for subgenre, tracks in track_data.items():
        result[subgenre] = {}
        for track_name, chunks in tracks.items():
            if all(chunk is not None for chunk in chunks):
                result[subgenre][track_name] = chunks
            else:
                logger.warning(f"Incomplete track {subgenre}/{track_name}: {chunks}")
    
    # Log statistics
    for subgenre, tracks in result.items():
        logger.info(f"{subgenre}: {len(tracks)} complete tracks")
    
    return result


def compute_track_embeddings(model: ImprovedASTTripletWrapper, 
                           track_data: Dict[str, Dict[str, List[str]]], 
                           device: torch.device,
                           batch_size: int = 4) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute embeddings for all tracks by averaging their 3 chunks
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
                    try:
                        # Load chunk data
                        chunk_data = torch.load(chunk_path, map_location='cpu')
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
                
                if len(chunk_embeddings) == 3:
                    # Average the 3 chunk embeddings
                    avg_embedding = np.mean(chunk_embeddings, axis=0).squeeze()
                    track_embeddings[subgenre][track_name] = avg_embedding
                else:
                    logger.warning(f"Track {track_name} missing chunks: {len(chunk_embeddings)}/3")
    
    return dict(track_embeddings)

def flatten_embeddings_to_matrix(embeddings_dict: Dict[str, Dict[str, np.ndarray]]):
    """
    Convert nested dict {subgenre: {track_name: emb}} to:
      X: (n_samples, d) np.ndarray
      y_true: (n_samples,) np.ndarray of integer labels
      y_names: list[str] subgenre names per sample (parallel to y_true)
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
    # safety: re-normalize to unit length
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms

    le = LabelEncoder()
    y_true = le.fit_transform(y_names)
    return X, y_true, y_names, track_ids, le


def compute_subgenre_centroids(embeddings: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Compute cluster centroids for each subgenre
    """
    centroids = {}
    
    logger.info("Computing subgenre centroids...")
    
    for subgenre, tracks in embeddings.items():
        if tracks:
            track_embeddings = list(tracks.values())
            centroid = np.mean(track_embeddings, axis=0)
            # L2 normalize the centroid
            centroid = centroid / np.linalg.norm(centroid)
            centroids[subgenre] = centroid
            logger.info(f"{subgenre}: centroid computed from {len(tracks)} tracks")
        else:
            logger.warning(f"{subgenre}: no tracks available for centroid computation")
    
    return centroids

def run_kmeans_and_metrics(
    X: np.ndarray,
    y_true: Optional[np.ndarray],
    n_clusters: int,
    random_state: int = 42,
    silhouette_metric: str = "cosine"
) -> Dict[str, any]:
    """
    Run Euclidean k-means on L2-normalized X and compute:
      - silhouette (cosine)
      - NMI, ARI (if y_true is provided)
    Returns dict with metrics and arrays.
    """
    if X is None or len(X) == 0:
        raise ValueError("No embeddings to cluster.")

    if n_clusters < 2:
        raise ValueError("Need at least 2 clusters for k-means/silhouette.")

    # sklearn KMeans uses Euclidean; with unit-norm X this approximates spherical k-means
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    y_pred = km.fit_predict(X)

    # Silhouette needs at least 2 clusters and #samples > #clusters
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

    return {
        "kmeans_model": km,
        "labels_pred": y_pred,
        "silhouette_cosine": sil,
        "nmi": nmi,
        "ari": ari
    }


def compute_cosine_similarity_matrix(centroids: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute cosine similarity matrix between centroids
    """
    subgenres = list(centroids.keys())
    n = len(subgenres)
    similarity_matrix = np.zeros((n, n))
    
    for i, subgenre1 in enumerate(subgenres):
        for j, subgenre2 in enumerate(subgenres):
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
    
    logger.info("Evaluating classification accuracy...")
    
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

def compute_triplet_distance_stats(model: nn.Module,
                                   splits: List,
                                   device: torch.device,
                                   chunks_dir: Path,
                                   batch_size: int = 16) -> Dict[str, any]:
    """
    Compute mean/std of triplet distances over (anchor, positive, negative) from a split.
    Returns:
      {
        'count': int,
        'mean_d_ap': float, 'std_d_ap': float,
        'mean_d_an': float, 'std_d_an': float,
        'gap': float,
        'per_subgenre': { sg: {... same keys ...} }
      }
    """
    model.eval()
    cache: Dict[str, np.ndarray] = {}

    def _embed_chunk(chunk_rel_path: str) -> Optional[np.ndarray]:
        # same path logic as load_and_process_data()
        full_path = chunks_dir.parent / chunk_rel_path
        if not full_path.exists():
            logger.warning(f"[dist] Missing chunk: {full_path}")
            return None
        if chunk_rel_path in cache:
            return cache[chunk_rel_path]
        try:
            td = torch.load(str(full_path), map_location='cpu')
            x = td['input_values']
            if x.ndim == 2:
                x = x.unsqueeze(0)
            with torch.no_grad():
                emb = model(x.to(device)).detach().cpu().numpy().squeeze()
            cache[chunk_rel_path] = emb
            return emb
        except Exception as e:
            logger.warning(f"[dist] Failed to embed {full_path}: {e}")
            return None

    d_ap_all, d_an_all = [], []
    per_sg = defaultdict(lambda: {'d_ap': [], 'd_an': []})

    for triplet in splits:
        if len(triplet) < 4:
            continue
        a_path, p_path, n_path, sg = triplet

        a = _embed_chunk(a_path)
        p = _embed_chunk(p_path)
        n = _embed_chunk(n_path)
        if a is None or p is None or n is None:
            continue

        # cosine distance since outputs are L2-normalized
        d_ap = 1.0 - float(np.dot(a, p))
        d_an = 1.0 - float(np.dot(a, n))

        d_ap_all.append(d_ap)
        d_an_all.append(d_an)
        per_sg[sg]['d_ap'].append(d_ap)
        per_sg[sg]['d_an'].append(d_an)

    def _summary(vals):
        if len(vals) == 0:
            return (float('nan'), float('nan'))
        return (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

    mean_d_ap, std_d_ap = _summary(d_ap_all)
    mean_d_an, std_d_an = _summary(d_an_all)

    out = {
        'count': len(d_ap_all),
        'mean_d_ap': mean_d_ap, 'std_d_ap': std_d_ap,
        'mean_d_an': mean_d_an, 'std_d_an': std_d_an,
        'gap': (mean_d_an - mean_d_ap) if (not np.isnan(mean_d_an) and not np.isnan(mean_d_ap)) else float('nan'),
        'per_subgenre': {}
    }

    for sg, d in per_sg.items():
        m_ap, s_ap = _summary(d['d_ap'])
        m_an, s_an = _summary(d['d_an'])
        out['per_subgenre'][sg] = {
            'count': len(d['d_ap']),
            'mean_d_ap': m_ap, 'std_d_ap': s_ap,
            'mean_d_an': m_an, 'std_d_an': s_an,
            'gap': (m_an - m_ap) if (not np.isnan(m_an) and not np.isnan(m_ap)) else float('nan'),
        }
    return out


def create_umap_visualization(test_embeddings: Dict[str, Dict[str, np.ndarray]],
                              centroids: Dict[str, np.ndarray],
                              model_dir: Path,
                              save_plots: bool = True,
                              label_centroids: bool = False,
                              filename_suffix: str = "",
                              # ---- NEW, all defaulted so existing calls keep working ----
                              centroids_only: bool = False,
                              umap_n_neighbors: int = 8,
                              umap_min_dist: float = 0.6,
                              umap_spread: float = 1.6,
                              umap_repulsion: float = 1.8,
                              umap_neg_sample_rate: int = 10,
                              umap_init: str = "pca",
                              jitter: float = 0.0) -> None:
    """
    UMAP visualization with clearer styling:
      - Distinct colors (tab20)
      - Larger track dots with thin white outline
      - Centroid stars with thinner stroke + subtle halo
      - Robust legend (no legendHandles access)

    The new parameters allow you to control layout & visibility:
      centroids_only: if True, compute UMAP only from centroids and plot only stars
      umap_*:         expose key UMAP levers (neighbors, min_dist, spread, etc.)
      jitter:         small noise added to track points after projection to unstack overlaps
    """
    logger.info("Creating UMAP visualization...")

    # ---------- 1) Flatten data ----------
    all_embeddings = []
    labels = []
    kinds = []  # 'track' or 'centroid'

    if not centroids_only:
        for sg, tracks in test_embeddings.items():
            for _, emb in tracks.items():
                all_embeddings.append(emb)
                labels.append(sg)
                kinds.append("track")

    for sg, emb in centroids.items():
        all_embeddings.append(emb)
        labels.append(sg)
        kinds.append("centroid")

    if len(all_embeddings) == 0:
        logger.warning("No embeddings available for UMAP.")
        return

    X = np.vstack(all_embeddings)
    kinds = np.array(kinds)
    labels = np.array(labels)

    # ---------- 2) UMAP (robust to small sample sizes) ----------
    n_samples = X.shape[0]
    # UMAP requires n_neighbors < n_samples; keep it >=2 for stability
    n_neighbors = max(2, min(umap_n_neighbors, n_samples - 1))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=umap_min_dist,
        spread=umap_spread,
        repulsion_strength=umap_repulsion,
        negative_sample_rate=umap_neg_sample_rate,
        metric="cosine",
        init=umap_init,
        random_state=42
    )
    X2 = reducer.fit_transform(X)

    # Optional tiny jitter to unstack coincident points (tracks only)
    if jitter > 0:
        rng = np.random.default_rng(42)
        J = rng.normal(scale=jitter, size=(X2.shape[0], 2))
        track_mask_for_jitter = (kinds == "track")
        X2[track_mask_for_jitter] += J[track_mask_for_jitter]

    track_mask = kinds == "track"
    cent_mask  = kinds == "centroid"
    X_track = X2[track_mask]
    X_cent  = X2[cent_mask]
    L_track = labels[track_mask]
    L_cent  = labels[cent_mask]

    # ---------- 3) Styling ----------

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Color palette (distinct + CB-friendly)
    subgenres = list(dict.fromkeys(labels.tolist()))  # preserve order
    base = mpl_cmaps.get_cmap("tab20").resampled(max(2, len(subgenres)))
    color_map = {sg: base(i / max(1, len(subgenres) - 1)) for i, sg in enumerate(subgenres)}

    # Nicer axes
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#777")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    # ---------- 4) Plot tracks: larger dots + white stroke ----------
    if not centroids_only:
        for sg in subgenres:
            m = (L_track == sg)
            if np.any(m):
                ax.scatter(
                    X_track[m, 0], X_track[m, 1],
                    s=64, marker="o",
                    c=[color_map[sg]],
                    edgecolors="white", linewidths=0.8,
                    alpha=0.85, label=f"{sg} (test tracks)"
                )

    # ---------- 5) Plot centroids: thin-edged stars + subtle halo ----------
    for sg in subgenres:
        m = (L_cent == sg)
        if np.any(m):
            x, y = X_cent[m, 0], X_cent[m, 1]

            # halo underlay improves contrast without thick black borders
            ax.scatter(x, y, s=380, marker="*", c=[color_map[sg]],
                       edgecolors="none", alpha=0.20, zorder=2.5)

            # main star, thinner edge than before
            ax.scatter(x, y, s=260, marker="*",
                       c=[color_map[sg]], edgecolors="black",
                       linewidths=0.9, alpha=0.95, zorder=3,
                       label=f"{sg} (centroid)")

            if label_centroids:
                for xi, yi in zip(x, y):
                    ax.text(xi + 0.12, yi + 0.08, sg,
                            fontsize=9, weight="semibold",
                            color="black", alpha=0.85)

    # ---------- 6) Title/labels/legend ----------
    title = "UMAP Visualization of Subgenre Centroids" if centroids_only \
            else "UMAP Visualization of Test Track Embeddings and Subgenre Centroids"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)

    # Robust legend: dedupe labels and avoid touching internals
    handles, labels_ = ax.get_legend_handles_labels()
    seen = set(); H = []; L = []
    for h, l in zip(handles, labels_):
        if l not in seen:
            seen.add(l); H.append(h); L.append(l)
    ax.legend(H, L, bbox_to_anchor=(1.02, 1), loc="upper left",
              frameon=False, borderaxespad=0.0, ncol=1)

    plt.tight_layout()

    # ---------- 7) Save/show ----------
    if save_plots:
        fname = "umap_centroids_only" if centroids_only else "umap_visualization"
        out = model_dir / f"{fname}{filename_suffix}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"UMAP visualization saved to: {out}")
    plt.show()

    # ---------- 8) Log small stats ----------
    logger.info("UMAP Statistics:")
    logger.info(f"  Total points: {X2.shape[0]}")
    logger.info(f"  Subgenres: {len(subgenres)}")
    logger.info(f"  n_neighbors used: {n_neighbors}")


def build_paired_label_order(labels):
    """
    Put paired subgenres next to each other. Any other labels are appended alphabetically.
    Pairs: (Chill House, Party House), (Chiller vibe goa, Party Goa), (Dark Techno, Emotional Techno)
    """
    labels = list(labels)
    want_pairs = [
        ("Chill House", "Party House"),
        ("Chiller vibe goa", "Party Goa"),
        ("Dark Techno", "Emotional Techno"),
    ]
    order = []
    used = set()
    for a, b in want_pairs:
        if a in labels:
            order.append(a); used.add(a)
        if b in labels:
            order.append(b); used.add(b)
    # add everything else, stable alphabetic
    for lab in sorted([x for x in labels if x not in used]):
        order.append(lab)
    return order


def compute_label_centroids(X, y_names, label_order):
    """
    Compute L2-normalized centroids per label in the given order.
    Returns centroids [len(label_order), d], and the list of labels actually present.
    """
    cents = []
    labels_present = []
    for lab in label_order:
        idx = [i for i, n in enumerate(y_names) if n == lab]
        if len(idx) == 0:
            continue
        c = X[idx].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        cents.append(c)
        labels_present.append(lab)
    return (np.vstack(cents) if cents else None), labels_present


def make_color_dict(label_order, cmap_name="tab10"):
    """Deterministic label -> color mapping."""
    cmap = plt.get_cmap(cmap_name)
    return {lab: cmap(i % 10) for i, lab in enumerate(label_order)}


def make_umap_reducer():
    """
    Tuned UMAP that works well for unit-norm cosine embeddings:
      - cosine metric (matches training geometry)
      - slightly denser clusters (min_dist=0.05)
      - neighbors=20 balances local vs global structure
    """
    return umap.UMAP(
        n_neighbors=20,
        min_dist=0.05,
        metric="cosine",
        random_state=42,
        n_components=2,
    )


def plot_umap_variants(
    X2,  # (n,2) precomputed UMAP coords for tracks
    y_names,
    label_order,
    color_dict,
    reducer,  # fitted umap reducer
    out_path_tracks_and_true=None,
    out_path_tracks_and_kmeans=None,
    out_path_all=None,
    true_centroids=None,       # (k,d) in embedding space (L2-normalized)
    kmeans_centroids=None,     # (k,d) in embedding space (L2-normalized)
):
    """
    Create up to three plots with consistent style:
      1) tracks + subgenre centroids
      2) tracks + k-means centroids
      3) tracks + both centroid types
    """
    # Precompute projected centroids if given
    tc2 = reducer.transform(true_centroids) if true_centroids is not None else None
    km2 = reducer.transform(kmeans_centroids) if kmeans_centroids is not None else None

    def _base(ax):
        # scatter test tracks per label
        for lab in label_order:
            idx = [i for i, n in enumerate(y_names) if n == lab]
            if not idx:
                continue
            ax.scatter(X2[idx, 0], X2[idx, 1], s=22, alpha=0.75,
                       color=color_dict[lab], label=lab, edgecolors='none')
        ax.set_xlabel("UMAP Dimension 1"); ax.set_ylabel("UMAP Dimension 2")
        ax.grid(True, alpha=0.15)

    def _legend(ax, show_true=False, show_km=False):
        # Custom legend: ordered labels + generic centroid handles at the end
        handles = [Line2D([0], [0], marker='o', linestyle='', markersize=6,
                          markerfacecolor=color_dict[lab], markeredgecolor='none', label=lab)
                   for lab in label_order if any(n == lab for n in y_names)]
        if show_true:
            handles.append(Line2D([0], [0], marker='*', linestyle='', markersize=12,
                                  markerfacecolor='none', markeredgecolor='black',
                                  label="Subgenre centroid"))
        if show_km:
            handles.append(Line2D([0], [0], marker='X', linestyle='', markersize=9,
                                  markerfacecolor='black', markeredgecolor='white',
                                  label="k-means centroid"))
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    # 1) tracks + subgenre centroids
    if out_path_tracks_and_true and tc2 is not None:
        fig, ax = plt.subplots(figsize=(9.0, 6.0))
        _base(ax)
        # subgenre centroids: star with colored face + black edge
        # match order to label_order subset that exists in tc2
        if true_centroids is not None:
            # We don't know exact label alignment after possible missing labels,
            # so recompute the projected centroids in label_order order:
            # (compute_label_centroids ensures the same order used to build tc2)
            pass
        ax.scatter(tc2[:, 0], tc2[:, 1], marker='*', s=260, linewidths=0.9,
                   edgecolors='black', facecolors='none', alpha=0.95)
        ax.set_title("UMAP: Test tracks + subgenre centroids")
        _legend(ax, show_true=True, show_km=False)
        plt.tight_layout()
        plt.savefig(out_path_tracks_and_true, dpi=220)
        plt.close()

    # 2) tracks + k-means centroids
    if out_path_tracks_and_kmeans and km2 is not None:
        fig, ax = plt.subplots(figsize=(9.0, 6.0))
        _base(ax)
        ax.scatter(km2[:, 0], km2[:, 1], marker='X', s=300, linewidths=0.9,
                   edgecolors='white', facecolors='black', alpha=0.95)
        ax.set_title("UMAP: Test tracks + k-means centroids")
        _legend(ax, show_true=False, show_km=True)
        plt.tight_layout()
        plt.savefig(out_path_tracks_and_kmeans, dpi=220)
        plt.close()

    # 3) tracks + both centroid types
    if out_path_all and (tc2 is not None or km2 is not None):
        fig, ax = plt.subplots(figsize=(9.0, 6.0))
        _base(ax)
        if tc2 is not None:
            ax.scatter(tc2[:, 0], tc2[:, 1], marker='*', s=260, linewidths=0.9,
                       edgecolors='black', facecolors='none', alpha=0.95)
        if km2 is not None:
            ax.scatter(km2[:, 0], km2[:, 1], marker='X', s=300, linewidths=0.9,
                       edgecolors='white', facecolors='black', alpha=0.95)
        ax.set_title("UMAP: Test tracks + subgenre & k-means centroids")
        _legend(ax, show_true=tc2 is not None, show_km=km2 is not None)
        plt.tight_layout()
        plt.savefig(out_path_all, dpi=220)
        plt.close()




def print_results(similarity_matrix: np.ndarray,
                 subgenres: List[str],
                 evaluation_results: Dict[str, any],
                 is_baseline: bool = False,
                 train_triplet_stats: Optional[Dict[str, any]] = None,
                 test_triplet_stats: Optional[Dict[str, any]] = None):
    """
    Print all evaluation results
    """
    print("\n" + "="*80)
    if is_baseline:
        print("BASELINE MODEL EVALUATION RESULTS (Pretrained AST Only)")
    else:
        print("TRAINED MODEL EVALUATION RESULTS")
    print("="*80)
    
    # 1. Cosine similarity matrix between centroids
    print("\n1. COSINE SIMILARITY MATRIX BETWEEN SUBGENRE CENTROIDS:")
    print("-" * 60)
    
    # Create DataFrame for better formatting
    df_sim = pd.DataFrame(similarity_matrix, index=subgenres, columns=subgenres)
    print(df_sim.round(4))
    
    # 2. Classification results for each test track
    print("\n2. TEST TRACK CLASSIFICATION RESULTS:")
    print("-" * 60)
    
    for subgenre, data in evaluation_results['per_subgenre'].items():
        print(f"\n{subgenre.upper()} ({data['total']} tracks, {data['accuracy']:.3f} accuracy):")
        print(f"{'Track Name':<50} {'Predicted':<20} {'Max Sim':<8} {'Correct'}")
        print("-" * 85)
        
        for pred in data['predictions']:
            correct_mark = "✓" if pred['correct'] else "✗"
            predicted_with_mark = f"{pred['predicted_subgenre']} {correct_mark}"
            print(f"{pred['track_name']:<50} {predicted_with_mark:<20} {pred['max_similarity']:.4f}   {pred['correct']}")
    
    # 3. Overall accuracy
    print("\n3. OVERALL EVALUATION METRICS:")
    print("-" * 60)
    print(f"Total test tracks: {evaluation_results['total_tracks']}")
    print(f"Correct predictions: {evaluation_results['correct_predictions']}")
    print(f"Overall accuracy: {evaluation_results['accuracy']:.4f} ({evaluation_results['accuracy']*100:.2f}%)")
    
    # Per-subgenre accuracy summary
    print("\nPer-subgenre accuracy:")
    for subgenre, data in evaluation_results['per_subgenre'].items():
        print(f"  {subgenre}: {data['accuracy']:.4f} ({data['correct']}/{data['total']})")

    # 4. Triplet intra- vs inter-class distances
    print("\n4. INTRA- VS. INTER-CLASS DISTANCE (cosine distance; lower is closer):")
    print("-" * 60)
    def _fmt(block_name, stats):
        if not stats or np.isnan(stats['mean_d_ap']) or np.isnan(stats['mean_d_an']):
            print(f"{block_name}: n/a")
            return
        print(f"{block_name}: N={stats['count']}")
        print(f"  mean d(a,p) = {stats['mean_d_ap']:.4f} ± {stats['std_d_ap']:.4f}")
        print(f"  mean d(a,n) = {stats['mean_d_an']:.4f} ± {stats['std_d_an']:.4f}")
        print(f"  gap Δ = mean d(a,n) - mean d(a,p) = {stats['gap']:.4f}")
        # per-subgenre brief
        for sg, d in sorted(stats['per_subgenre'].items()):
            print(f"    {sg:>16}: Δ={d['gap']:.4f} (ap={d['mean_d_ap']:.4f}, an={d['mean_d_an']:.4f}, N={d['count']})")

    _fmt("TRAIN", train_triplet_stats)
    _fmt("TEST ", test_triplet_stats)


def main():
    parser = argparse.ArgumentParser(description="Evaluate AST Triplet Loss Model")
    parser.add_argument("model_folder", type=str,
                       help="Path to the trained model output directory containing model.safetensors, splits.json, and config.json")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for embedding computation (default: 4)")
    parser.add_argument("--no_umap", action="store_true",
                       help="Skip UMAP visualization generation")
    parser.add_argument("--no_save_plots", action="store_true",
                       help="Don't save UMAP plots to disk (still display them)")
    parser.add_argument("--use_baseline", action="store_true",
                       help="Use base pretrained AST model instead of trained weights (for baseline comparison)")
    parser.add_argument("--cluster_on", choices=["test", "train", "all"], default="test",
                    help="Which embeddings to cluster for metrics (default: test)")

    args = parser.parse_args()
    
    # Setup device (auto-detect)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Setup paths
    model_dir = Path(args.model_folder)
    chunks_dir = Path("./precomputed_AST_7G")  # Default chunks directory
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model folder not found: {model_dir}")
    
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
    
    try:
        # 1. Load model
        if args.use_baseline:
            logger.info("Loading baseline model (pretrained AST only)...")
        else:
            logger.info("Loading trained model...")
        model, config = load_model(model_dir, device, use_baseline=args.use_baseline)
        
        # 2. Load splits
        logger.info("Loading train/test splits...")
        train_splits, test_splits = load_splits(model_dir)
        
        # 3. Process data
        logger.info("Processing training data...")
        train_track_data = load_and_process_data(train_splits, chunks_dir)
        
        logger.info("Processing test data...")
        test_track_data = load_and_process_data(test_splits, chunks_dir)
        
        # 4. Compute embeddings
        logger.info("Computing training embeddings...")
        train_embeddings = compute_track_embeddings(model, train_track_data, device, args.batch_size)
        
        logger.info("Computing test embeddings...")
        test_embeddings = compute_track_embeddings(model, test_track_data, device, args.batch_size)
        
        # 5. Compute centroids
        logger.info("Computing subgenre centroids...")
        centroids = compute_subgenre_centroids(train_embeddings)
        
        # 6. Compute similarity matrices
        logger.info("Computing cosine similarity matrices...")
        similarity_matrix, subgenres = compute_cosine_similarity_matrix(centroids)
        
        # 7. Evaluate classification
        logger.info("Evaluating classification accuracy...")
        evaluation_results = evaluate_classification(test_embeddings, centroids)
        logger.info("Computing triplet distance statistics (train/test)...")
        train_triplet_stats = compute_triplet_distance_stats(
            model=model, splits=train_splits, device=device, chunks_dir=chunks_dir, batch_size=args.batch_size
        )
        test_triplet_stats = compute_triplet_distance_stats(
            model=model, splits=test_splits, device=device, chunks_dir=chunks_dir, batch_size=args.batch_size
        )
        
        # 8. Print results
        print_results(similarity_matrix, subgenres, evaluation_results,
                      is_baseline=args.use_baseline,
                      train_triplet_stats=train_triplet_stats,
                      test_triplet_stats=test_triplet_stats)

        # 9. Generate UMAP visualization
        if not args.no_umap:
            logger.info("Generating UMAP visualization...")
            try:
                # Add suffix to filename if baseline
                filename_suffix = "_baseline" if args.use_baseline else ""
                create_umap_visualization(
                    test_embeddings=test_embeddings,
                    centroids=centroids,
                    model_dir=model_dir,
                    save_plots=not args.no_save_plots,
                    filename_suffix=filename_suffix
                )
            except Exception as e:
                logger.warning(f"UMAP visualization failed: {e}")
                logger.warning("Continuing without visualization...")

        
        # 10. Clustering quality metrics (k-means + silhouette/NMI/ARI + extras)
        if args.cluster_on == "train":
            emb_source = train_embeddings
            source_name = "TRAIN"
        elif args.cluster_on == "all":
            emb_source = {}
            for d in (train_embeddings, test_embeddings):
                for sg, tracks in d.items():
                    emb_source.setdefault(sg, {}).update(tracks)
            source_name = "TRAIN+TEST"
        else:
            emb_source = test_embeddings
            source_name = "TEST"

        X, y_true, y_names, track_ids, le = flatten_embeddings_to_matrix(emb_source)
        if X is None:
            logger.warning("No embeddings available for clustering metrics; skipping.")
        else:
            n_clusters = len(centroids)
            try:
                cm = run_kmeans_and_metrics(X, y_true, n_clusters=n_clusters)

                print("\n" + "="*80)
                print(f"CLUSTERING QUALITY METRICS on {source_name} embeddings (k = {n_clusters})")
                print("="*80)
                print(f"Silhouette (cosine): {cm['silhouette_cosine']:.4f}" if cm['silhouette_cosine'] is not None else "Silhouette: n/a")
                print(f"NMI: {cm['nmi']:.4f}" if cm['nmi'] is not None else "NMI: n/a")
                print(f"ARI: {cm['ari']:.4f}" if cm['ari'] is not None else "ARI: n/a")

                # --- Confusion matrix (true subgenre vs predicted cluster) ---
                conf = confusion_matrix(y_true, cm["labels_pred"])
                conf_df = pd.DataFrame(conf, index=le.classes_, columns=[f"C{j}" for j in range(n_clusters)])
                print("\nConfusion matrix (rows = true subgenres, cols = cluster IDs):")
                print(conf_df)
                out_csv = model_dir / f"confusion_matrix_{source_name.lower()}.csv"
                conf_df.to_csv(out_csv)
                logger.info(f"Confusion matrix saved to: {out_csv}")

                # --- Cluster purity ---
                purities = []
                for j in range(n_clusters):
                    members = (cm["labels_pred"] == j)
                    if members.sum() == 0:
                        purities.append(0.0)
                        continue
                    majority = np.bincount(y_true[members]).max()
                    purity = majority / members.sum()
                    purities.append(purity)
                avg_purity = np.mean(purities)
                print("\nCluster purities:", [f"{p:.2f}" for p in purities])
                print(f"Average purity: {avg_purity:.4f}")

                # --- UMAP visualizations (match old style; colored centroids; slimmer X) ---

                # 1) Legend order: keep pairs adjacent
                labels_all = sorted(set(y_names))
                pair_order = [
                    ("Chill House", "Party House"),
                    ("Chiller vibe goa", "Party Goa"),
                    ("Dark Techno", "Emotional Techno"),
                ]
                ordered = []
                used = set()
                for a, b in pair_order:
                    if a in labels_all:
                        ordered.append(a); used.add(a)
                    if b in labels_all:
                        ordered.append(b); used.add(b)
                for lab in labels_all:
                    if lab not in used:
                        ordered.append(lab)

                # 2) Color mapping (same palette across plots)
                cmap = plt.get_cmap("tab10")
                color_of = {lab: cmap(i % 10) for i, lab in enumerate(ordered)}

                # 3) Compute subgenre centroids (L2-normalized) per label
                true_centroids = {}
                for lab in ordered:
                    idx = [i for i, n in enumerate(y_names) if n == lab]
                    if not idx:
                        continue
                    c = X[idx].mean(axis=0)
                    c = c / (np.linalg.norm(c) + 1e-12)
                    true_centroids[lab] = c
                Ctrue = (np.stack(list(true_centroids.values())).astype(np.float32)
                         if true_centroids else None)
                Ctrue_labels = list(true_centroids.keys())

                # 4) Normalize k-means centroids (means -> unit sphere)
                Ckm = cm["kmeans_model"].cluster_centers_.astype(np.float32)
                Ckm /= (np.linalg.norm(Ckm, axis=1, keepdims=True) + 1e-12)

                # 5) Use "old-style" UMAP params for nicer spread (match your previous look)
                reducer = umap.UMAP(
                    n_neighbors=15,
                    min_dist=0.10,
                    metric="cosine",
                    random_state=42,
                    n_components=2,
                )
                X_2d = reducer.fit_transform(X)
                Ctrue_2d = reducer.transform(Ctrue) if Ctrue is not None else None
                Ckm_2d = reducer.transform(Ckm) if Ckm is not None else None

                # Small helpers
                def _scatter_tracks(ax):
                    for lab in ordered:
                        idx = [i for i, n in enumerate(y_names) if n == lab]
                        if not idx:
                            continue
                        ax.scatter(
                            X_2d[idx, 0], X_2d[idx, 1],
                            s=36, alpha=0.80, edgecolors='none',
                            color=color_of[lab], label=f"{lab}"
                        )
                    ax.set_xlabel("UMAP Dimension 1")
                    ax.set_ylabel("UMAP Dimension 2")
                    ax.grid(True, alpha=0.15)

                def _legend(ax, add_true=False, add_km=False):
                    # Label handles in the requested order
                    handles = []
                    for lab in ordered:
                        if any(n == lab for n in y_names):
                            handles.append(Line2D([0],[0], marker='o', linestyle='',
                                                  markersize=6,
                                                  markerfacecolor=color_of[lab],
                                                  markeredgecolor='none',
                                                  label=f"{lab}"))
                    if add_true:
                        handles.append(Line2D([0],[0], marker='*', linestyle='',
                                              markersize=10,
                                              markerfacecolor='white',
                                              markeredgecolor='black',
                                              label="Subgenre centroid"))
                    if add_km:
                        handles.append(Line2D([0],[0], marker='X', linestyle='',
                                              markersize=9,  # slimmer + smaller X
                                              markerfacecolor='black',
                                              markeredgecolor='white',
                                              label="k-means centroid"))
                    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1),
                              loc="upper left", frameon=True)

                # 6) Plot A: tracks + subgenre centroids (★ filled with same color as tracks)
                fig, ax = plt.subplots(figsize=(10, 6.5))
                _scatter_tracks(ax)
                if Ctrue_2d is not None:
                    # draw per-label so star face matches that label's color
                    for lab, (cx, cy) in zip(Ctrue_labels, Ctrue_2d):
                        ax.scatter([cx], [cy],
                                   marker='*', s=220, linewidths=1.0,
                                   edgecolors='black', facecolors=color_of[lab], alpha=0.95)
                ax.set_title("UMAP: Test tracks + subgenre centroids")
                _legend(ax, add_true=True, add_km=False)
                out_true = model_dir / f"umap_tracks_truecentroids_{source_name.lower()}.png"
                plt.tight_layout(); plt.savefig(out_true, dpi=220); plt.close()
                logger.info(f"UMAP (tracks+true centroids) saved to: {out_true}")

                # 7) Plot B: tracks + k-means centroids (smaller, thinner 'X')
                fig, ax = plt.subplots(figsize=(10, 6.5))
                _scatter_tracks(ax)
                if Ckm_2d is not None:
                    ax.scatter(Ckm_2d[:, 0], Ckm_2d[:, 1],
                               marker='X', s=160, linewidths=0.8,  # << slimmer, smaller
                               edgecolors='white', facecolors='black', alpha=0.95)
                ax.set_title("UMAP: Test tracks + k-means centroids")
                _legend(ax, add_true=False, add_km=True)
                out_km = model_dir / f"umap_tracks_kmeans_{source_name.lower()}.png"
                plt.tight_layout(); plt.savefig(out_km, dpi=220); plt.close()
                logger.info(f"UMAP (tracks+k-means centroids) saved to: {out_km}")

                # 8) Plot C: tracks + both centroid types (stars colored per label + slim X)
                fig, ax = plt.subplots(figsize=(10, 6.5))
                _scatter_tracks(ax)
                if Ctrue_2d is not None:
                    for lab, (cx, cy) in zip(Ctrue_labels, Ctrue_2d):
                        ax.scatter([cx], [cy],
                                   marker='*', s=220, linewidths=1.0,
                                   edgecolors='black', facecolors=color_of[lab], alpha=0.95)
                if Ckm_2d is not None:
                    ax.scatter(Ckm_2d[:, 0], Ckm_2d[:, 1],
                               marker='X', s=160, linewidths=0.8,
                               edgecolors='white', facecolors='black', alpha=0.95)
                ax.set_title("UMAP: Test tracks + subgenre & k-means centroids")
                _legend(ax, add_true=True, add_km=True)
                out_all = model_dir / f"umap_tracks_truecentroids_kmeans_{source_name.lower()}.png"
                plt.tight_layout(); plt.savefig(out_all, dpi=220); plt.close()
                logger.info(f"UMAP (tracks+true+k-means centroids) saved to: {out_all}")

                # 7) Plot B: tracks + k-means centroids
                fig, ax = plt.subplots(figsize=(9, 6))
                _scatter_tracks(ax)
                ax.scatter(Ckm_2d[:, 0], Ckm_2d[:, 1],
                           marker='X', s=300, linewidths=0.9,
                           edgecolors='white', facecolors='black',
                           label="k-means centroid", alpha=0.95)
                ax.set_title("UMAP: Test tracks + k-means centroids")
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
                out_km = model_dir / f"umap_tracks_kmeans_{source_name.lower()}.png"
                plt.tight_layout(); plt.savefig(out_km, dpi=220); plt.close()
                logger.info(f"UMAP (tracks+k-means centroids) saved to: {out_km}")

                # 8) Plot C: tracks + both centroid types
                fig, ax = plt.subplots(figsize=(9, 6))
                _scatter_tracks(ax)
                if Ctrue_2d is not None:
                    ax.scatter(Ctrue_2d[:, 0], Ctrue_2d[:, 1],
                               marker='*', s=260, linewidths=0.9,
                               edgecolors='black', facecolors='none',
                               label="Subgenre centroid", alpha=0.95)
                ax.scatter(Ckm_2d[:, 0], Ckm_2d[:, 1],
                           marker='X', s=300, linewidths=0.9,
                           edgecolors='white', facecolors='black',
                           label="k-means centroid", alpha=0.95)
                ax.set_title("UMAP: Test tracks + subgenre & k-means centroids")
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
                out_all = model_dir / f"umap_tracks_truecentroids_kmeans_{source_name.lower()}.png"
                plt.tight_layout(); plt.savefig(out_all, dpi=220); plt.close()
                logger.info(f"UMAP (tracks+true+k-means centroids) saved to: {out_all}")

                # Save detailed assignments
                assign_df = pd.DataFrame({
                    "track_id": track_ids,
                    "true_subgenre": y_names,
                    "cluster_id": cm["labels_pred"]
                })
                out_csv = model_dir / f"kmeans_assignments_{source_name.lower()}.csv"
                assign_df.to_csv(out_csv, index=False)
                logger.info(f"k-means assignments saved to: {out_csv}")

            except Exception as e:
                logger.warning(f"Clustering metrics failed: {e}")

        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()