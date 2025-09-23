# Final Experiments for Music Similarity Thesis

## Overview
Based on the best performing **6x2 configuration with aggressive augmentations** (K-fold result: **0.8434 ± 0.0171**, max: 0.8742), these experiments systematically explore the key research questions for the thesis.

## Current Best Configuration
- **Dataset**: 6 positive tracks × 2 triplets = ~12 triplets per anchor
- **Augmentations**: Aggressive (prob=0.7, freq/time masks, noise, volume, etc.)
- **Architecture**: 768→512→128 projection head with 0.15 dropout
- **Training**: 15 epochs, LR=2e-05, margin=0.5, bf16 precision

## Experiment Categories

### 1. Chunking Strategies (`chunking_strategies/`)
**Research Question**: How do different audio chunking approaches affect performance?

- **`config_1chunk_averaged.json`**: Single averaged chunk (uses 3-chunk archive)
- **`config_2chunks.json`**: Current best (baseline for comparison)
- **`config_3chunks.json`**: Traditional 3-chunk approach (4×3=12 triplets, comparable to 6×2)

**Expected Outcome**: 2-chunk should outperform others due to optimal diversity/similarity balance.

### 2. Augmentation Comparison (`augmentation_comparison/`)
**Research Question**: What is the impact of audio augmentations on model performance?

- **`config_no_augmentations.json`**: Pure model performance without augmentations
- **`config_with_augmentations.json`**: Best performing config with aggressive augmentations

**Expected Outcome**: Augmentations should provide significant improvement (~5-10% accuracy gain).

### 3. Hyperparameter Tuning (`hyperparameter_tuning/`)
**Research Question**: Can we fine-tune the current best config for even better performance?

- **`config_higher_lr.json`**: 2× learning rate (4e-05 vs 2e-05)
- **`config_larger_margin.json`**: Larger triplet margin (0.8 vs 0.5)
- **`config_lower_weight_decay.json`**: More model flexibility (0.01 vs 0.05)

**Expected Outcome**: One of these may push performance closer to 0.85-0.86 range.

### 4. Negative Mining Ablation (`negative_mining_ablation/`)
**Research Question**: Can negative mining push the model beyond the ~0.87 accuracy plateau?

- **`config_no_mining_baseline.json`**: Current approach (baseline reproduction)
- **`config_semi_hard_mining_early.json`**: Semi-hard mining from epoch 3
- **`config_semi_hard_mining_mid.json`**: Semi-hard mining from epoch 7
- **`config_hard_mining_late.json`**: Hard mining from epoch 10

**Expected Outcome**: Strategic negative mining timing should break through the 0.87 plateau and potentially reach 0.88-0.90.

## Execution Strategy

### Priority Order (Limited Time):
1. **Negative Mining Ablation** (highest potential for breakthrough)
2. **Chunking Strategies** (core thesis comparison)
3. **Augmentation Comparison** (fundamental technique validation)
4. **Hyperparameter Tuning** (incremental improvements)

### K-Fold Validation:
Run each significant config with K-fold cross-validation using:
```bash
python AST_Triplet_kfold.py --config final_experiments/[category]/[config].json --k 5
```

## Expected Results Summary

| Experiment | Expected Accuracy | Key Insight |
|------------|-------------------|-------------|
| Current Best (6x2 + Aug) | 0.8434 ± 0.0171 | Baseline |
| 1-chunk averaged | ~0.81-0.82 | Information loss |
| 3-chunks | ~0.83-0.84 | Similar to 2-chunk |
| No augmentations | ~0.78-0.80 | Overfitting issues |
| Higher LR | ~0.84-0.85 | Faster convergence |
| Larger margin | ~0.84-0.85 | Stronger separation |
| Semi-hard mining | **~0.86-0.88** | **Breakthrough potential** |
| Hard mining | **~0.87-0.90** | **Maximum performance** |

## Statistical Significance
All experiments will report **mean ± standard deviation** from 5-fold cross-validation, enabling proper statistical comparison and thesis-quality results reporting.