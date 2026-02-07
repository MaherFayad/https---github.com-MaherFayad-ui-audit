"""
EML-NET Hybrid Training Script v2
Trains saliency model on combined datasets with improved loss functions.

Improvements:
- Hybrid multi-objective loss: KL_Div + CC + NSS + MSE
- Multi-scale training support
- Automatic Mixed Precision (AMP) with GradScaler

Usage:
    python train_hybrid.py
    python train_hybrid.py --epochs 5 --dry-run

Author: Refactored for UX-Heatmap v2
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
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

# Loss weights (tuned for UI saliency)
ALPHA_KL = 1.0    # KL Divergence - distribution alignment (primary)
BETA_CC = 0.3     # Correlation Coefficient - global pattern
GAMMA_NSS = 0.5   # Normalized Scanpath Saliency - fixation accuracy
DELTA_MSE = 0.1   # Mean Squared Error - pixel smoothness

# Paths (relative to project root)
SILICON_ROOT = "models/Datasets/Silicon"
UEYES_ROOT = "models/Datasets/Ueyes"
MASSVIS_ROOT = "models/Datasets/massvis"
FIWI_ROOT = "models/Datasets/mobile ui salency"
SALCHART_ROOT = "models/Datasets/SALchart QA"
MODEL_SAVE_PATH = "models/eml_net_hybrid.pth"

# Device configuration
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    USE_AMP = True  # Use Automatic Mixed Precision
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"AMP Enabled: {USE_AMP}")
else:
    DEVICE = torch.device("cpu")
    USE_AMP = False
    print(f"Device: {DEVICE} (CPU fallback)")


# =============================================================================
# Loss Functions
# =============================================================================

def kl_divergence_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    KL Divergence Loss for saliency maps.
    
    Treats both maps as probability distributions and measures how the predicted
    distribution diverges from the ground truth.
    
    Formula: KL(P||Q) = Σ P(x) * log(P(x) / Q(x))
    
    Args:
        pred: Predicted saliency map (B, 1, H, W), should be in [0, 1]
        target: Ground truth saliency map (B, 1, H, W)
        eps: Small value for numerical stability
        
    Returns:
        Scalar KL divergence loss (averaged over batch)
    """
    # Flatten spatial dimensions
    pred_flat = pred.flatten(start_dim=1)    # (B, H*W)
    target_flat = target.flatten(start_dim=1)  # (B, H*W)
    
    # Normalize to probability distributions (sum to 1)
    pred_dist = pred_flat / (pred_flat.sum(dim=1, keepdim=True) + eps)
    target_dist = target_flat / (target_flat.sum(dim=1, keepdim=True) + eps)
    
    # KL Divergence: P * log(P / Q) = P * (log(P) - log(Q))
    # We compute: target * log(target / pred)
    kl = target_dist * (torch.log(target_dist + eps) - torch.log(pred_dist + eps))
    
    # Sum over spatial dimension, mean over batch
    return kl.sum(dim=1).mean()


def cc_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Correlation Coefficient (CC) Loss.
    
    Measures linear correlation between predicted and ground truth saliency.
    Returns 1 - CC so that minimizing loss = maximizing correlation.
    
    Formula: CC = cov(P, Q) / (σ_P * σ_Q)
    
    Args:
        pred: Predicted saliency map (B, 1, H, W)
        target: Ground truth saliency map (B, 1, H, W)
        
    Returns:
        Scalar loss (1 - CC, averaged over batch)
    """
    # Flatten spatial dimensions
    pred_flat = pred.flatten(start_dim=1)
    target_flat = target.flatten(start_dim=1)
    
    # Center the data
    pred_mean = pred_flat.mean(dim=1, keepdim=True)
    target_mean = target_flat.mean(dim=1, keepdim=True)
    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean
    
    # Compute covariance and standard deviations
    n = pred_flat.size(1)
    covariance = (pred_centered * target_centered).sum(dim=1)
    pred_std = torch.sqrt((pred_centered ** 2).sum(dim=1) + eps)
    target_std = torch.sqrt((target_centered ** 2).sum(dim=1) + eps)
    
    # Correlation coefficient
    cc = covariance / (pred_std * target_std + eps)
    
    # Return 1 - CC (want to maximize CC)
    return (1 - cc).mean()


def nss_loss(pred: torch.Tensor, target: torch.Tensor, 
             threshold: float = 0.5, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalized Scanpath Saliency (NSS) Loss.
    
    Measures saliency values at fixation locations (high values in ground truth).
    Higher NSS = better prediction at actual fixation points.
    
    Formula: NSS = mean((P[fixations] - μ_P) / σ_P)
    
    Args:
        pred: Predicted saliency map (B, 1, H, W)
        target: Ground truth map (B, 1, H, W), values > threshold are fixations
        threshold: Threshold to identify fixation locations
        
    Returns:
        Scalar loss (-NSS, averaged over batch)
    """
    # Flatten
    pred_flat = pred.flatten(start_dim=1)
    target_flat = target.flatten(start_dim=1)
    
    # Normalize predictions (z-score)
    pred_mean = pred_flat.mean(dim=1, keepdim=True)
    pred_std = pred_flat.std(dim=1, keepdim=True) + eps
    pred_norm = (pred_flat - pred_mean) / pred_std
    
    # Create fixation mask (binary: where target > threshold)
    fixation_mask = (target_flat > threshold).float()
    
    # Compute NSS: mean of normalized prediction at fixation locations
    # Weighted mean using fixation mask
    nss = (pred_norm * fixation_mask).sum(dim=1) / (fixation_mask.sum(dim=1) + eps)
    
    # Return negative (minimize -NSS = maximize NSS)
    return -nss.mean()


def hybrid_loss(pred: torch.Tensor, target: torch.Tensor,
                alpha: float = ALPHA_KL, beta: float = BETA_CC,
                gamma: float = GAMMA_NSS, delta: float = DELTA_MSE) -> Tuple[torch.Tensor, dict]:
    """
    Combined hybrid loss for saliency prediction.
    
    L = α*KL_Div + β*(1-CC) + γ*(-NSS) + δ*MSE
    
    Args:
        pred: Predicted saliency (B, 1, H, W), should be sigmoid-activated
        target: Ground truth saliency (B, 1, H, W)
        alpha, beta, gamma, delta: Loss weights
        
    Returns:
        Tuple of (total_loss, dict of individual losses for logging)
    """
    loss_kl = kl_divergence_loss(pred, target)
    loss_cc = cc_loss(pred, target)
    loss_nss = nss_loss(pred, target)
    loss_mse = F.mse_loss(pred, target)
    
    total = alpha * loss_kl + beta * loss_cc + gamma * loss_nss + delta * loss_mse
    
    return total, {
        'kl': loss_kl.item(),
        'cc': loss_cc.item(),
        'nss': loss_nss.item(),
        'mse': loss_mse.item(),
        'total': total.item()
    }


# =============================================================================
# Training Logic
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: Optional[GradScaler],
    epoch: int,
    accumulation_steps: int = 4,
    use_amp: bool = True
) -> Tuple[float, dict]:
    """
    Train for one epoch with gradient accumulation and AMP.
    
    Returns:
        Tuple of (average loss, dict of individual losses)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    loss_accum = {'kl': 0, 'cc': 0, 'nss': 0, 'mse': 0}
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=True)
    
    for batch_idx, (images, maps, _labels) in enumerate(pbar):
        images = images.to(DEVICE, non_blocking=True)
        maps = maps.to(DEVICE, non_blocking=True)
        
        # Forward pass with AMP
        if use_amp and DEVICE.type == 'cuda':
            with autocast('cuda'):
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                loss, loss_dict = hybrid_loss(outputs, maps)
                loss = loss / accumulation_steps
        else:
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            loss, loss_dict = hybrid_loss(outputs, maps)
            loss = loss / accumulation_steps
        
        # Backward pass with scaler
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
        
        # Track losses
        total_loss += loss_dict['total']
        for k in loss_accum:
            loss_accum[k] += loss_dict[k]
        num_batches += 1
        
        # Update progress bar
        avg_loss = total_loss / num_batches
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'kl': f'{loss_accum["kl"]/num_batches:.3f}',
            'cc': f'{loss_accum["cc"]/num_batches:.3f}'
        })
    
    # Handle remaining gradients
    if num_batches % accumulation_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()
    
    avg_losses = {k: v / max(num_batches, 1) for k, v in loss_accum.items()}
    return total_loss / max(num_batches, 1), avg_losses


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    use_amp: bool = True
) -> Tuple[float, dict]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    loss_accum = {'kl': 0, 'cc': 0, 'nss': 0, 'mse': 0}
    
    with torch.no_grad():
        for images, maps, _labels in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(DEVICE, non_blocking=True)
            maps = maps.to(DEVICE, non_blocking=True)
            
            if use_amp and DEVICE.type == 'cuda':
                with autocast('cuda'):
                    outputs = model(images)
                    outputs = torch.sigmoid(outputs)
                    loss, loss_dict = hybrid_loss(outputs, maps)
            else:
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                loss, loss_dict = hybrid_loss(outputs, maps)
            
            total_loss += loss_dict['total']
            for k in loss_accum:
                loss_accum[k] += loss_dict[k]
            num_batches += 1
    
    avg_losses = {k: v / max(num_batches, 1) for k, v in loss_accum.items()}
    return total_loss / max(num_batches, 1), avg_losses


# =============================================================================
# Main Training Pipeline
# =============================================================================

def main(args):
    """Main training function."""
    print("=" * 60)
    print("EML-NET v2 Hybrid Saliency Model Training")
    print("=" * 60)
    
    # Device info
    print(f"\nDevice: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"AMP Enabled: {USE_AMP}")
    
    print(f"\nLoss Weights: KL={ALPHA_KL}, CC={BETA_CC}, NSS={GAMMA_NSS}, MSE={DELTA_MSE}")
    
    # ==========================================================================
    # Load Datasets
    # ==========================================================================
    print(f"\n[1/4] Loading Datasets...")
    
    # Filter roots based on --dataset flag
    s_root = SILICON_ROOT if args.dataset in ['all', 'silicon'] else None
    u_root = UEYES_ROOT if args.dataset in ['all', 'ueyes'] else None
    m_root = MASSVIS_ROOT if args.dataset in ['all', 'massvis'] else None
    f_root = FIWI_ROOT if args.dataset in ['all', 'mobile'] else None
    sc_root = SALCHART_ROOT if args.dataset in ['all', 'salchart'] else None

    train_dataset = HybridDataset(
        silicon_root=s_root,
        ueyes_root=u_root,
        massvis_root=m_root,
        fiwi_root=f_root,
        salchart_root=sc_root,
        split='train'
    )
    
    val_dataset = HybridDataset(
        silicon_root=s_root,
        ueyes_root=u_root,
        massvis_root=m_root,
        fiwi_root=f_root,
        salchart_root=sc_root,
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
    print(f"\n[2/4] Initializing EMLNet v2 (EfficientNet-V2 + FPN)...")
    
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
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=LEARNING_RATE * 0.01
    )
    
    # GradScaler for AMP
    scaler = GradScaler() if USE_AMP else None
    
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
        train_loss, train_losses = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            accumulation_steps=ACCUMULATION_STEPS,
            use_amp=USE_AMP
        )
        
        # Validate
        if val_loader is not None:
            val_loss, val_losses = validate(model, val_loader, use_amp=USE_AMP)
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"  Train - Total: {train_loss:.4f} | KL: {train_losses['kl']:.4f} | "
                  f"CC: {train_losses['cc']:.4f} | NSS: {train_losses['nss']:.4f}")
            print(f"  Val   - Total: {val_loss:.4f} | KL: {val_losses['kl']:.4f} | "
                  f"CC: {val_losses['cc']:.4f} | NSS: {val_losses['nss']:.4f}")
            
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
                    'loss_weights': {
                        'alpha_kl': ALPHA_KL,
                        'beta_cc': BETA_CC,
                        'gamma_nss': GAMMA_NSS,
                        'delta_mse': DELTA_MSE
                    }
                }, MODEL_SAVE_PATH)
                print(f"  -> Saved best model (val_loss: {val_loss:.4f})")
        else:
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}")
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
    parser = argparse.ArgumentParser(description="Train EML-NET v2 Hybrid Saliency Model")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help=f"Number of epochs (default: {EPOCHS})")
    parser.add_argument('--dry-run', action='store_true', help="Run only 1 epoch for testing")
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument('--dataset', type=str, default='all', 
                        choices=['all', 'silicon', 'ueyes', 'massvis', 'mobile', 'salchart'], 
                        help="Specific dataset to train on")
    
    args = parser.parse_args()
    
    # Override batch size if specified
    if args.batch_size != BATCH_SIZE:
        BATCH_SIZE = args.batch_size
    
    main(args)
