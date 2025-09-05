"""
Sophisticated Learning Rate Scheduler for AST Triplet Training.

Provides different learning rates for transformer base and projection head,
with warmup, cosine annealing, and epoch-based multiplier convergence.
"""

import math
import logging
from typing import List, Dict, Any, Optional
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class DualGroupLRScheduler:
    """
    Learning rate scheduler with separate groups for base transformer and projection head.
    
    Features:
    - Different LR multipliers for base vs head (head typically gets higher LR)
    - Warmup period (3-5% of total steps)
    - Cosine annealing decay
    - Minimum LR floor for both groups
    - Epoch-based multiplier convergence (multiplier becomes 1.0 after specified epoch)
    """
    
    def __init__(self, 
                 optimizer: Optimizer,
                 base_lr: float,
                 head_lr_multiplier: float,
                 total_steps: int,
                 warmup_steps: int,
                 multiplier_converge_epoch: int,
                 steps_per_epoch: int,
                 min_lr: float = 1e-6,
                 use_cosine: bool = True):
        """
        Args:
            optimizer: PyTorch optimizer with two parameter groups [base, head]
            base_lr: Base learning rate for transformer
            head_lr_multiplier: Multiplier for head LR vs base LR
            total_steps: Total training steps
            warmup_steps: Number of warmup steps
            multiplier_converge_epoch: Epoch after which multiplier becomes 1.0
            steps_per_epoch: Steps per epoch (for epoch-based convergence)
            min_lr: Minimum learning rate floor
            use_cosine: Whether to use cosine annealing (else linear decay)
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.head_lr_multiplier = head_lr_multiplier
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.multiplier_converge_epoch = multiplier_converge_epoch
        self.converge_step = multiplier_converge_epoch * steps_per_epoch
        self.min_lr = min_lr
        self.use_cosine = use_cosine
        
        # Validate parameter groups
        if len(optimizer.param_groups) != 2:
            raise ValueError(f"Optimizer must have exactly 2 parameter groups, got {len(optimizer.param_groups)}. "
                           f"Groups: {[len(g['params']) for g in optimizer.param_groups]}")
        
        # Log scheduler configuration
        logger.info(f"DualGroupLRScheduler initialized:")
        logger.info(f"  Base LR: {base_lr:.2e}")
        logger.info(f"  Head LR multiplier: {head_lr_multiplier}x")
        logger.info(f"  Multiplier converges to 1.0 at epoch {multiplier_converge_epoch}")
        logger.info(f"  Warmup steps: {warmup_steps} ({warmup_steps/total_steps*100:.1f}%)")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Min LR: {min_lr:.2e}")
        logger.info(f"  Scheduler type: {'Cosine' if use_cosine else 'Linear'}")
        
    def _get_multiplier_at_step(self, step: int) -> float:
        """Get the current head LR multiplier based on step/epoch."""
        if step >= self.converge_step:
            return 1.0
        else:
            # Linear interpolation from head_lr_multiplier to 1.0
            progress = step / self.converge_step
            return self.head_lr_multiplier * (1 - progress) + 1.0 * progress
    
    def _get_warmup_factor(self, step: int) -> float:
        """Get warmup factor (linear warmup)."""
        if step >= self.warmup_steps:
            return 1.0
        else:
            return step / self.warmup_steps
    
    def _get_cosine_decay_factor(self, step: int) -> float:
        """Get cosine decay factor after warmup."""
        if step <= self.warmup_steps:
            return 1.0
        
        # Cosine decay from warmup_steps to total_steps
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)  # Clamp to [0, 1]
        
        if self.use_cosine:
            return 0.5 * (1 + math.cos(math.pi * progress))
        else:
            # Linear decay
            return 1.0 - progress
    
    def step(self, current_step: int = None):
        """Update learning rates for both parameter groups."""
        if current_step is None:
            # If called by HuggingFace without step number, we can't update properly
            logger.warning("DualGroupLRScheduler.step() called without current_step - skipping update")
            return
            
        warmup_factor = self._get_warmup_factor(current_step)
        decay_factor = self._get_cosine_decay_factor(current_step)
        multiplier = self._get_multiplier_at_step(current_step)
        
        # Base transformer group (index 0)
        base_lr = self.base_lr * warmup_factor * decay_factor
        base_lr = max(base_lr, self.min_lr)
        self.optimizer.param_groups[0]['lr'] = base_lr
        
        # Projection head group (index 1) 
        head_lr = self.base_lr * multiplier * warmup_factor * decay_factor
        head_lr = max(head_lr, self.min_lr)
        self.optimizer.param_groups[1]['lr'] = head_lr
        
        # Log periodically (every 50 steps)
        if current_step % 50 == 0:
            logger.debug(f"Step {current_step}: base_lr={base_lr:.2e}, head_lr={head_lr:.2e}, "
                        f"multiplier={multiplier:.2f}, warmup={warmup_factor:.3f}, decay={decay_factor:.3f}")
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rates for compatibility with Trainer."""
        return [group['lr'] for group in self.optimizer.param_groups]


class DualGroupLRCallback(TrainerCallback):
    """
    TrainerCallback that manages the DualGroupLRScheduler and adds LR info to HF logging.
    """
    
    def __init__(self, scheduler: DualGroupLRScheduler):
        self.scheduler = scheduler
        self.logged_lr_once = False
    
    def on_step_end(self, args, state, control, **kwargs):
        """Update learning rates at the end of each step."""
        current_step = state.global_step if state.global_step is not None else 0
        if self.scheduler is not None:
            self.scheduler.step(current_step)
        else:
            logger.error("Scheduler is None in callback!")
        
        # Log LR periodically for debugging (less frequent)
        if self.scheduler is not None and (not self.logged_lr_once or current_step % 500 == 0):
            lrs = self.scheduler.get_last_lr()
            logger.info(f"Step {current_step}: base_lr={lrs[0]:.2e}, head_lr={lrs[1]:.2e}")
            self.logged_lr_once = True
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Add learning rates and training loss to HuggingFace's standard logging."""
        if logs is None:
            return
            
        # Only add to training logs (not evaluation logs)
        if ('train_loss' in logs or 'loss' in logs) and self.scheduler is not None:
            lrs = self.scheduler.get_last_lr()
            
            # Add base and head learning rates to the logs
            logs['base_lr'] = lrs[0]
            logs['head_lr'] = lrs[1] 
            
            # Remove the confusing single 'learning_rate' entry since we have separate rates
            logs.pop('learning_rate', None)
            
            # Ensure training_loss is in logs (sometimes it's just 'loss')
            if 'loss' in logs and 'train_loss' not in logs:
                logs['train_loss'] = logs['loss']


def create_dual_group_optimizer(model, config) -> Optimizer:
    """
    Create optimizer with separate parameter groups for base transformer and projection head.
    
    Args:
        model: ImprovedASTTripletWrapper model
        config: ExperimentConfig with training parameters
        
    Returns:
        Optimizer with two parameter groups: [base_params, head_params]
    """
    # Separate base transformer parameters from projection head parameters
    base_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'projector' in name:
            head_params.append(param)
        else:
            base_params.append(param)
    
    logger.info(f"Parameter groups created:")
    logger.info(f"  Base parameters: {len(base_params)} tensors")
    logger.info(f"  Head parameters: {len(head_params)} tensors")
    
    # Validate that both groups have parameters
    if len(base_params) == 0:
        raise ValueError("No base transformer parameters found! Check model parameter names.")
    if len(head_params) == 0:
        raise ValueError("No projection head parameters found! Check that 'projector' is in parameter names.")
    
    # Create parameter groups with different learning rates
    base_lr = config.training.learning_rate
    head_lr = base_lr * config.training.head_lr_multiplier
    
    param_groups = [
        {
            'params': base_params,
            'lr': base_lr,
            'weight_decay': config.training.weight_decay,
            'name': 'base_transformer'
        },
        {
            'params': head_params, 
            'lr': head_lr,
            'weight_decay': config.training.weight_decay,
            'name': 'projection_head'
        }
    ]
    
    # Create optimizer (using AdamW)
    optimizer = torch.optim.AdamW(param_groups)
    
    logger.info(f"Dual group optimizer created:")
    logger.info(f"  Base LR: {base_lr:.2e}")
    logger.info(f"  Head LR: {head_lr:.2e} ({config.training.head_lr_multiplier}x multiplier)")
    
    return optimizer


def create_dual_group_scheduler(optimizer, config, total_steps: int, steps_per_epoch: int):
    """
    Create PyTorch LambdaLR scheduler with dual-group sophisticated scheduling.
    
    Args:
        optimizer: Optimizer with two parameter groups
        config: ExperimentConfig with training parameters
        total_steps: Total training steps
        steps_per_epoch: Steps per epoch
        
    Returns:
        PyTorch LambdaLR scheduler compatible with HuggingFace Trainer
    """
    from torch.optim.lr_scheduler import LambdaLR
    
    warmup_steps = int(total_steps * config.training.warmup_steps_pct)
    converge_step = config.training.multiplier_converge_epoch * steps_per_epoch
    base_lr = config.training.learning_rate
    head_lr_multiplier = config.training.head_lr_multiplier
    min_lr = config.training.min_lr
    use_cosine = config.training.use_cosine_scheduler
    
    def get_multiplier_at_step(step: int) -> float:
        """Get the current head LR multiplier based on step/epoch."""
        if step >= converge_step:
            return 1.0
        else:
            # Linear interpolation from head_lr_multiplier to 1.0
            progress = step / converge_step
            return head_lr_multiplier * (1 - progress) + 1.0 * progress
    
    def get_warmup_factor(step: int) -> float:
        """Get warmup factor (linear warmup)."""
        if step >= warmup_steps:
            return 1.0
        else:
            return step / warmup_steps
    
    def get_cosine_decay_factor(step: int) -> float:
        """Get cosine decay factor after warmup."""
        if step <= warmup_steps:
            return 1.0
        
        # Cosine decay from warmup_steps to total_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = min(progress, 1.0)  # Clamp to [0, 1]
        
        if use_cosine:
            return 0.5 * (1 + math.cos(math.pi * progress))
        else:
            # Linear decay
            return 1.0 - progress
    
    def base_lr_lambda(current_step: int) -> float:
        """LR lambda for base transformer parameters."""
        warmup_factor = get_warmup_factor(current_step)
        decay_factor = get_cosine_decay_factor(current_step)
        lr_factor = warmup_factor * decay_factor
        
        # Apply min_lr floor
        min_factor = min_lr / base_lr
        return max(lr_factor, min_factor)
    
    def head_lr_lambda(current_step: int) -> float:
        """LR lambda for projection head parameters."""
        multiplier = get_multiplier_at_step(current_step)
        warmup_factor = get_warmup_factor(current_step)
        decay_factor = get_cosine_decay_factor(current_step)
        lr_factor = multiplier * warmup_factor * decay_factor
        
        # Apply min_lr floor
        min_factor = min_lr / base_lr
        return max(lr_factor, min_factor)
    
    # Create LambdaLR with separate lambda functions for each parameter group
    scheduler = LambdaLR(optimizer, [base_lr_lambda, head_lr_lambda])
    
    logger.info(f"PyTorch LambdaLR scheduler created:")
    logger.info(f"  Base LR: {base_lr:.2e}")
    logger.info(f"  Head LR multiplier: {head_lr_multiplier}x")
    logger.info(f"  Multiplier converges to 1.0 at epoch {config.training.multiplier_converge_epoch}")
    logger.info(f"  Warmup steps: {warmup_steps} ({warmup_steps/total_steps*100:.1f}%)")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Min LR: {min_lr:.2e}")
    logger.info(f"  Scheduler type: {'Cosine' if use_cosine else 'Linear'}")
    
    return scheduler