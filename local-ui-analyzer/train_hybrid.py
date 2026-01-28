"""
EML-NET Hybrid Training Script
Trains saliency model on combined Silicon + Ueyes datasets.

Optimized for:
- NVIDIA RTX 5060 Ti (8GB VRAM)
- BFloat16 mixed precision
- Gradient accumulation (effective batch size 32)

Usage:
    python train_hybrid.py
    python train_hybrid.py --epochs 5 --dry-run

Author: Generated for EML-NET Hybrid Training Pipeline
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from eml_net_model import EMLNet
from hybrid_loader import HybridDataset, get_balanced_sampler


# =============================================================================
# Configuration
# =============================================================================

BATCH_SIZE = 8
ACCUMULATION_STEPS = 4  # Effective batch size = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Paths (relative to project root)
SILICON_ROOT = "models/Datasets/Silicon"
UEYES_ROOT = "models/Datasets/Ueyes"
MODEL_SAVE_PATH = "models/eml_net_hybrid.pth"

# Device configuration
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    USE_BFLOAT16 = torch.cuda.is_bf16_supported()
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"BFloat16 supported: {USE_BFLOAT16}")
else:
    DEVICE = torch.device("cpu")
    USE_BFLOAT16 = False
    print(f"Device: {DEVICE} (CPU fallback)")



# =============================================================================
# Loss Functions
# =============================================================================

def cc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Correlation Coefficient (CC) Loss.
    
    Measures the linear correlation between predicted and ground truth saliency maps.
    Returns negative CC so minimizing loss = maximizing correlation.
    
    Args:
        pred: Predicted saliency map (B, 1, H, W)
        target: Ground truth saliency map (B, 1, H, W)
    
    Returns:
        Scalar loss (negative CC, averaged over batch)
    """
    # Flatten spatial dimensions
    pred_flat = pred.flatten(start_dim=1)  # (B, H*W)
    target_flat = target.flatten(start_dim=1)  # (B, H*W)
    
    # Center the data (subtract mean)
    pred_mean = pred_flat.mean(dim=1, keepdim=True)
    target_mean = target_flat.mean(dim=1, keepdim=True)
    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean
    
    # Compute covariance (numerator)
    covariance = (pred_centered * target_centered).sum(dim=1)
    
    # Compute standard deviations
    pred_std = pred_centered.std(dim=1) + 1e-8
    target_std = target_centered.std(dim=1) + 1e-8
    
    # Correlation coefficient
    cc = covariance / (pred_std * target_std * pred_flat.size(1))
    
    # Return negative mean (we want to maximize CC, so minimize -CC)
    return -cc.mean()


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Combined loss: CC + MSE for stable training.
    
    CC is scale-invariant but can be unstable early in training.
    MSE provides gradient stability.
    """
    cc = cc_loss(pred, target)
    mse = nn.functional.mse_loss(pred, target)
    return cc + 0.1 * mse


# =============================================================================
# Training Logic
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
    accumulation_steps: int = 4,
    use_bf16: bool = True
) -> float:
    """
    Train for one epoch with gradient accumulation.
    
    Args:
        model: EMLNet model
        dataloader: Training data loader
        optimizer: Optimizer
        scaler: GradScaler for mixed precision (not needed for bf16)
        epoch: Current epoch number
        accumulation_steps: Number of steps to accumulate gradients
        use_bf16: Whether to use BFloat16
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=True)
    
    for batch_idx, (images, maps, _labels) in enumerate(pbar):
        # Move to device
        images = images.to(DEVICE, non_blocking=True)
        maps = maps.to(DEVICE, non_blocking=True)
        
        # Forward pass with autocast
        if use_bf16 and DEVICE.type == 'cuda':
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(images)
                # Apply sigmoid to get [0, 1] range
                outputs = torch.sigmoid(outputs)
                loss = combined_loss(outputs, maps)
                loss = loss / accumulation_steps  # Scale for accumulation
        else:
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            loss = combined_loss(outputs, maps)
            loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Track loss (unscaled)
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
        
        # Update progress bar
        avg_loss = total_loss / num_batches
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    # Handle remaining gradients
    if num_batches % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / max(num_batches, 1)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    use_bf16: bool = True
) -> float:
    """
    Validate the model.
    
    Args:
        model: EMLNet model
        dataloader: Validation data loader
        use_bf16: Whether to use BFloat16
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, maps, _labels in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(DEVICE, non_blocking=True)
            maps = maps.to(DEVICE, non_blocking=True)
            
            if use_bf16 and DEVICE.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(images)
                    outputs = torch.sigmoid(outputs)
                    loss = combined_loss(outputs, maps)
            else:
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                loss = combined_loss(outputs, maps)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


# =============================================================================
# Main Training Pipeline
# =============================================================================

def main(args):
    """Main training function."""
    print("=" * 60)
    print("EML-NET Hybrid Saliency Model Training")
    print("=" * 60)
    
    # Device info
    print(f"\nDevice: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"BFloat16 Support: {USE_BFLOAT16}")
    
    # ==========================================================================
    # Load Datasets
    # ==========================================================================
    print(f"\n[1/4] Loading datasets...")
    
    train_dataset = HybridDataset(
        silicon_root=SILICON_ROOT,
        ueyes_root=UEYES_ROOT,
        split='train'
    )
    
    val_dataset = HybridDataset(
        silicon_root=SILICON_ROOT,
        ueyes_root=UEYES_ROOT,
        split='val'
    )
    
    if len(train_dataset) == 0:
        print("ERROR: No training samples found!")
        sys.exit(1)
    
    # Create balanced sampler
    train_sampler = get_balanced_sampler(train_dataset)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    ) if len(val_dataset) > 0 else None
    
    # ==========================================================================
    # Initialize Model
    # ==========================================================================
    print(f"\n[2/4] Initializing EMLNet model...")
    
    model = EMLNet()
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ==========================================================================
    # Setup Training
    # ==========================================================================
    print(f"\n[3/4] Setting up training...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Accumulation steps: {ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {args.epochs}")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=LEARNING_RATE * 0.01
    )
    
    # GradScaler not needed for BFloat16 (native support)
    scaler = None
    
    # Tracking best model
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Ensure model directory exists
    Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # Training Loop
    # ==========================================================================
    print(f"\n[4/4] Starting training...")
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            accumulation_steps=ACCUMULATION_STEPS,
            use_bf16=USE_BFLOAT16
        )
        
        # Validate
        if val_loader is not None:
            val_loss = validate(model, val_loader, use_bf16=USE_BFLOAT16)
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, MODEL_SAVE_PATH)
                print(f"  -> Saved best model (val_loss: {val_loss:.4f})")
        else:
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}")
            # Save latest model if no validation
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, MODEL_SAVE_PATH)
        
        # Step scheduler
        scheduler.step()
        
        # Dry run: exit after first epoch
        if args.dry_run:
            print("\n[DRY RUN] Stopping after 1 epoch.")
            break
    
    # ==========================================================================
    # Training Complete
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    if val_loader is not None:
        print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EML-NET Hybrid Saliency Model")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help=f"Number of epochs (default: {EPOCHS})")
    parser.add_argument('--dry-run', action='store_true', help="Run only 1 epoch for testing")
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    
    args = parser.parse_args()
    
    # Override batch size if specified
    if args.batch_size != BATCH_SIZE:
        BATCH_SIZE = args.batch_size
    
    main(args)
