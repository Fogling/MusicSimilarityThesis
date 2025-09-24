#!/usr/bin/env python3
"""
K-Fold Cross-Validation wrapper for AST Triplet Training.

This script runs K-fold cross-validation using preprocessed k-fold directories
created by Preprocess_AST_features.py with --enable-kfold flag.

Usage:
    # First, preprocess your data with k-fold splitting:
    python Preprocess_AST_features.py --enable-kfold --k-folds 5 --wav-dir WAV --output-dir precomputed_7Gen_5Fold

    # Then run k-fold training:
    python AST_Triplet_kfold.py --config train_from_precomputed --preprocessed-dir precomputed_7Gen_5Fold --k 5
"""

import argparse
import os
from pathlib import Path
from AST_Triplet_training import run_kfold_training
from config import load_or_create_config

def main():
    """Main K-fold training function using preprocessed k-fold directories."""
    parser = argparse.ArgumentParser(
        description="AST Triplet K-Fold Training with Preprocessed Splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # 1. First preprocess with k-fold splitting:
    python Preprocess_AST_features.py --enable-kfold --k-folds 5 --wav-dir WAV --output-dir precomputed_7Gen_5Fold

    # 2. Then run k-fold training:
    python AST_Triplet_kfold.py --config train_from_precomputed --preprocessed-dir precomputed_7Gen_5Fold --k 5
        """
    )
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--preprocessed-dir", type=str, required=True,
                       help="Path to preprocessed k-fold directory (e.g., 'precomputed_7Gen_5Fold')")
    parser.add_argument("--test-run", action="store_true", help="Run quick test with minimal data")
    parser.add_argument("--k", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--output-dir", type=str, default="kfold_results",
                       help="Output directory for results (default: 'kfold_results')")

    args = parser.parse_args()

    try:
        # Validate preprocessed directory
        preprocessed_path = Path(args.preprocessed_dir)
        if not preprocessed_path.exists():
            raise FileNotFoundError(f"Preprocessed directory not found: {preprocessed_path}")

        # Check that k-fold structure exists
        missing_folds = []
        for fold_idx in range(args.k):
            fold_dir = preprocessed_path / f"fold_{fold_idx}"
            if not fold_dir.exists():
                missing_folds.append(f"fold_{fold_idx}")

        if missing_folds:
            raise FileNotFoundError(f"Missing fold directories: {missing_folds}. "
                                   f"Did you preprocess with --k-folds {args.k}?")

        # Load configuration
        config = load_or_create_config(args.config, test_run=args.test_run)
        config.validate()

        print(f"Using preprocessed k-fold data from: {preprocessed_path}")
        print(f"Running {args.k}-fold cross-validation...")

        # Run K-fold training with preprocessed data
        results = run_kfold_training(
            base_config=config,
            preprocessed_dir=str(preprocessed_path),
            k=args.k,
            output_dir=args.output_dir
        )

        # Print final summary
        if "final_statistics" in results:
            stats = results["final_statistics"]
            print(f"\n{'='*60}")
            print(f"K-FOLD TRAINING COMPLETED!")
            print(f"{'='*60}")
            print(f"Mean Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
            print(f"Results saved to: {results['results_dir']}")
        else:
            print(f"\nK-fold training completed. Results saved to: {results.get('results_dir', 'unknown')}")

        return 0

    except Exception as e:
        print(f"K-fold training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())