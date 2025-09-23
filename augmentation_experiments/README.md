# Augmentation Experiment Configurations

This directory contains configuration files for testing different dataset sizes and augmentation strategies to combat overfitting in AST triplet training.

## Configuration Overview

All configs use:
- **Dataset**: `precomputed_7G_2Chunk` (2 chunks per track)
- **Base Settings**: 15 epochs, early stopping (patience=4, delta=0.01)
- **Cluster Settings**: GPU optimized (bf16=true, tf32=true, batch_size=24)

## Dataset Size Testing

### Small Datasets (Slower Overfitting)
- **config_aug_1x5.json**: 1 positive track × 5 triplets = ~5 triplets per anchor
  - **Mild augmentations** (prob=0.3) - small datasets overfit slower
  - Conservative approach for smallest dataset

- **config_aug_2x3.json**: 2 positive tracks × 3 triplets = ~6 triplets per anchor
  - **Moderate augmentations** (prob=0.5)
  - Reference dataset size from original experiments

### Medium to Large Datasets (Faster Overfitting)
- **config_aug_2x5.json**: 2 positive tracks × 5 triplets = ~10 triplets per anchor
  - **Moderate augmentations** (prob=0.5)

- **config_aug_4x4.json**: 4 positive tracks × 4 triplets = ~16 triplets per anchor
  - **Strong augmentations** (prob=0.6) - larger datasets overfit faster

- **config_aug_6x2.json**: 6 positive tracks × 2 triplets = ~12 triplets per anchor
  - **Aggressive augmentations** (prob=0.7) - combat rapid overfitting

## Augmentation Intensity Testing

### Conservative (Subtle Effects)
- **config_aug_conservative.json**: 2x3 dataset with mild augmentations
  - Low probability (0.3), gentle parameters
  - Tests if minimal augmentation helps

### Aggressive (Strong Effects)
- **config_aug_aggressive.json**: 1x5 dataset with strong augmentations
  - High probability (0.7), aggressive masking/noise
  - Tests maximum overfitting prevention

## Expected Results

**Larger datasets (4x4, 6x2)** should benefit most from strong augmentations, showing:
- Slower train loss decrease (preventing rapid overfitting)
- Better eval loss convergence
- Smaller train/eval gap by epoch 3-5

**Smaller datasets (1x5, 2x3)** may show:
- Modest improvement with mild augmentations
- Risk of degradation if augmentations are too aggressive

## Usage

```bash
# Run on cluster
sbatch train_precomputed.sbatch augmentation_experiments/config_aug_2x3.json

# Compare with baseline (no augmentations)
sbatch train_precomputed.sbatch config_experiments/config_1.json
```

## Key Metrics to Monitor

1. **Train vs Eval Loss Gap**: Should be smaller with augmentations
2. **Early Stopping Trigger**: Should happen later (more stable training)
3. **Final Eval Accuracy**: Should improve for small datasets
4. **Training Stability**: Less volatile loss curves