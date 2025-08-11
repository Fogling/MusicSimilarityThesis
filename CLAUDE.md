# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**Python Environment**: Use the `thesis-env` virtual environment located in the `thesis-env/` directory.

**Activation**:
```bash
# Windows
thesis-env\Scripts\activate

# Unix/Mac
source thesis-env/bin/activate
```

**Dependencies**: The project uses PyTorch, Transformers (Hugging Face), torchaudio, and other ML libraries as specified in `environment.yml`.

## High-Level Architecture

This is a **music genre classification and similarity learning** research project using Audio Spectrogram Transformer (AST) models with triplet/quadruplet loss functions.

### Core Components

1. **Audio Preprocessing Pipeline** (`Preprocess_AST_features.py`):
   - Loads audio files from `WAV/` directory organized by genre/subgenre
   - Chunks audio into 10-second segments (3 chunks per track using random sampling)
   - Extracts AST features using `MIT/ast-finetuned-audioset-10-10-0.4593` model
   - Uses memory-efficient online statistics computation
   - Supports multiple chunk sampling strategies: random (default), sequential, spaced
   - Saves preprocessed features to `preprocessed_features/`

2. **Triplet Loss Training** (`AST_Triplet_training.py`):
   - Main training script using triplet loss (anchor-positive-negative)
   - Uses `ASTTripletWrapper` class that wraps pretrained AST with projection head
   - Implements stratified train/test split by subgenre
   - Saves trained models to timestamped directories (`fully_trained_models_YYYYMMDD_HHMMSS/`)
   - Also saves intermediate checkpoints to `ast_triplet_output/`

3. **Custom Loss Functions**:
   - `MyQuadrupletLoss.py`: Implements quadruplet loss with "trash" samples
   - Triplet loss in main training uses cosine similarity with margin=0.3

4. **Dataset Handling** (`datasets/TripletAudioDataset.py`):
   - Custom PyTorch Dataset for triplet sampling
   - Loads preprocessed `.pt` files containing AST feature inputs

5. **Evaluation & Visualization** (`evaluate_triplet_embeddings.py`):
   - Computes embeddings for trained model
   - Creates UMAP visualizations of learned representations
   - Analyzes subgenre clustering

### Data Organization

- **Input**: `WAV/[genre]/[subgenre]/*.{mp3,wav}` - Raw audio files
- **Preprocessed**: `preprocessed_features/[subgenre]/*_chunk[N].pt` - AST feature tensors
- **Models**: `fully_trained_models_*/` and `ast_triplet_output/` - Saved model weights
- **Splits**: Train/test splits saved as JSON files in model output directories

## Common Development Commands

**Preprocess audio data**:
```bash
# Uses optimal defaults: 3 random chunks per song
python Preprocess_AST_features.py

# Or with custom chunk strategy
python Preprocess_AST_features.py --chunk-strategy sequential
python Preprocess_AST_features.py --chunk-strategy spaced
```

**Train triplet model**:
```bash
# Full training
python AST_Triplet_training.py

# Quick test run with small subset
python AST_Triplet_training.py --test_run
```

**Evaluate trained model**:
```bash
python evaluate_triplet_embeddings.py
```

**Extract embeddings for analysis**:
```bash
python extract_test_embeddings.py
```

## Key Configuration Parameters

- **Chunk Duration**: 10 seconds (configurable in preprocessing)
- **Sample Rate**: 16kHz for AST model compatibility
- **Batch Size**: 2 (optimized for memory constraints)
- **Learning Rate**: 1e-4
- **Epochs**: 30
- **Triplet Margin**: 0.3
- **Max Chunks per Song**: 3 (optimized to avoid pseudo-duplicates)
- **Chunk Sampling**: Random sampling by default for maximum diversity

## Model Architecture Details

- **Base Model**: MIT/ast-finetuned-audioset-10-10-0.4593 (pretrained AST)
- **Projection Head**: Linear(768) → ReLU → Linear(512) → ReLU → Linear(128)
- **Output**: L2-normalized 128-dimensional embeddings
- **Loss**: Cosine similarity-based triplet loss with margin

## Important Notes

- The project uses CUDA when available, falls back to CPU
- Feature extraction includes dataset-specific normalization computed in two passes
- Model checkpoints are saved in SafeTensors format
- Train/test splits are stratified by subgenre to ensure balanced evaluation

## Important Context
- It is my wish to iteratively refactor and improve the main files AST_Triplet_Training.py, Preprocess_AST_features.py and TripletAudioDataset.py. This means replacing unclear or unneccessary Code with concise best practice Code that is easy to read and comprehend. Also I want to remove any unneccessary or unused parts of the code.
