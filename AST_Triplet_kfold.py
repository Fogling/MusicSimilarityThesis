#!/usr/bin/env python3
"""
K-Fold Cross-Validation wrapper for AST Triplet Training.

This script runs K-fold cross-validation using the same arguments
and config system as AST_Triplet_training.py.

Usage (same as AST_Triplet_training.py):
    python AST_Triplet_kfold.py --config train_from_precomputed
    python AST_Triplet_kfold.py --test-run
"""

import argparse
from AST_Triplet_training import run_kfold_training
from config import load_or_create_config

def main():
    """Main K-fold training function with same interface as AST_Triplet_training.py"""
    parser = argparse.ArgumentParser(description="AST Triplet K-Fold Training")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--test-run", action="store_true", help="Run quick test with minimal data")
    parser.add_argument("--k", type=int, default=5, help="Number of folds (default: 5)")

    args = parser.parse_args()

    try:
        # Load configuration (same as original)
        config = load_or_create_config(args.config, test_run=args.test_run)
        config.validate()

        # Run K-fold training
        results = run_kfold_training(config, k=args.k)

        # Print final summary
        if "final_statistics" in results:
            stats = results["final_statistics"]
            print(f"\nK-FOLD RESULTS:")
            print(f"Mean Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
            print(f"Results saved to: {results['results_dir']}")

        return 0

    except Exception as e:
        print(f"K-fold training failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())