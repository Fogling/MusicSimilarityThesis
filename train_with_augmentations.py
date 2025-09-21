#!/usr/bin/env python3
"""
AST Triplet Training with Audio Augmentations

Modified version of AST_Triplet_training.py that integrates audio augmentations
for overfitting reduction. This script uses the same core logic but wraps
datasets with augmentation capabilities.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the current directory to Python path to ensure imports work
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import all the existing training infrastructure
from AST_Triplet_training import (
    generate_train_test_splits,
    ASTTripletWrapper,
    CleanLoggingCallback,
    EarlyStoppingCallback,
    MarginSchedulingCallback,
    logger
)
from training_integration_patch import create_training_datasets, log_augmentation_status
from config import load_or_create_config
from lr_scheduler import create_dual_group_optimizer, create_dual_group_scheduler, DualGroupLRCallback

# Import training infrastructure
import torch
from transformers import TrainingArguments, Trainer


def main():
    """Main training function with augmentations."""
    parser = argparse.ArgumentParser(description="AST Triplet Training with Audio Augmentations")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--augmentation-level", type=str, default="conservative",
                       choices=["none", "minimal", "conservative", "aggressive"],
                       help="Augmentation level to apply")
    parser.add_argument("--test-run", action="store_true", help="Run quick test with minimal data")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_or_create_config(args.config, test_run=args.test_run)

        # Set random seeds
        import random
        import numpy as np

        seed = config.experiment.random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seeds set to {seed}")

        # Set precision optimizations
        torch.set_float32_matmul_precision('high')
        logger.info("set_float32_matmul_precision=high")

        # Enable TF32 for better performance on Ampere GPUs
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32: matmul=True, cudnn=True")

        # Set cache directory for remote execution
        cache = os.environ.get('SLURM_JOB_TMP', '/tmp')
        features_path = os.path.join(cache, config.data.preprocessed_features_path)

        if not os.path.exists(features_path):
            features_path = config.data.preprocessed_features_path
            if not os.path.exists(features_path):
                raise FileNotFoundError(f"Preprocessed features not found at {features_path}")

        logger.info(f"Loading data splits...")
        train_data, test_data = generate_train_test_splits(features_path, config)

        # Create datasets with augmentations
        logger.info("Creating datasets...")
        train_dataset, test_dataset = create_training_datasets(
            train_data, test_data, config, augmentation_level=args.augmentation_level
        )

        # Initialize model
        logger.info("Initializing model...")
        logger.info("Using device: cuda" if torch.cuda.is_available() else "Using device: cpu")

        model = ASTTripletWrapper(config)

        # Create training arguments
        output_dir = f"run_{config.experiment.name.replace(' ', '_')}_{args.augmentation_level}"
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.training.epochs,
            per_device_train_batch_size=config.training.batch_size,
            per_device_eval_batch_size=config.training.batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            report_to=None,  # Disable wandb/tensorboard
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            bf16=torch.cuda.is_available(),
            tf32=torch.cuda.is_available(),
            dataloader_num_workers=0,
        )

        # Set up optimizers and scheduler
        optimizer = create_dual_group_optimizer(model, config)
        scheduler = create_dual_group_scheduler(optimizer, config, len(train_dataset), training_args)

        # Create callbacks
        callbacks = [
            CleanLoggingCallback(),
            MarginSchedulingCallback(model),
            DualGroupLRCallback()
        ]

        # Add early stopping if enabled
        if config.training.early_stopping.enabled:
            callbacks.append(EarlyStoppingCallback(config, None))

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            optimizers=(optimizer, scheduler),
            callbacks=callbacks,
        )

        # Log training setup
        effective_batch_size = (
            config.training.batch_size *
            config.training.gradient_accumulation_steps
        )
        steps_per_epoch = len(train_dataset) // effective_batch_size
        total_steps = steps_per_epoch * config.training.epochs

        logger.info("Training setup:")
        logger.info(f"  - Train samples: {len(train_dataset)}")
        logger.info(f"  - Test samples: {len(test_dataset)}")
        logger.info(f"  - Batch size: {config.training.batch_size}")
        logger.info(f"  - Gradient accumulation: {config.training.gradient_accumulation_steps}")
        logger.info(f"  - Effective batch size: {effective_batch_size}")
        logger.info(f"  - Steps per epoch: {steps_per_epoch}")
        logger.info(f"  - Total epochs: {config.training.epochs}")
        logger.info(f"  - Estimated total steps: {total_steps}")
        logger.info(f"  - Augmentation level: {args.augmentation_level}")

        # Log augmentation status
        log_augmentation_status(train_dataset, "Training")
        log_augmentation_status(test_dataset, "Evaluation")

        # Perform initial evaluation
        logger.info("Performing initial evaluation to establish baseline...")
        initial_metrics = trainer.evaluate()
        logger.info(f"Evaluation - Epoch 0, Step 0: eval_loss={initial_metrics['eval_loss']:.4f}, "
                   f"eval_accuracy={initial_metrics['eval_accuracy']:.3f}, Elapsed: 0.0s")

        # Start training
        logger.info("Starting training...")
        logger.info(f"Training started at {trainer.state.log_history}")

        train_result = trainer.train()

        # Log final results
        logger.info("Training completed!")
        logger.info(f"Final training loss: {train_result.training_loss:.6f}")

        # Final evaluation
        final_metrics = trainer.evaluate()
        logger.info(f"Final evaluation: eval_loss={final_metrics['eval_loss']:.4f}, "
                   f"eval_accuracy={final_metrics['eval_accuracy']:.3f}")

        # Save model and artifacts
        logger.info("Saving model and artifacts...")
        trainer.save_model()

        # Save configuration with augmentation info
        config_save_path = Path(output_dir) / "training_config.json"
        config_dict = vars(config) if hasattr(config, '__dict__') else config
        config_dict['augmentation_level'] = args.augmentation_level

        import json
        with open(config_save_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

        logger.info(f"Training completed successfully! Results saved to: {output_dir}")

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())