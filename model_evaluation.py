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
import seaborn as sns
import umap
from sklearn.preprocessing import LabelEncoder

from config import ExperimentConfig, load_or_create_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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




def print_results(similarity_matrix: np.ndarray,
                 subgenres: List[str],
                 evaluation_results: Dict[str, any],
                 is_baseline: bool = False):
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
        
        # 8. Print results
        print_results(similarity_matrix, subgenres, evaluation_results, is_baseline=args.use_baseline)

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

        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()