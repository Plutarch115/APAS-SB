"""
Training script for ΔΔG prediction with Pearl.

Tests the ΔΔG prediction capability on synthetic data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

from pearl.models.mock_pearl import MockPearl
from pearl.models.ddg_predictor import PearlWithDDG
from pearl.training.ddg_losses import DDGLoss, DDGMetrics
from pearl.data.ddg_dataset import SyntheticDDGDataset, collate_ddg_batch, create_train_val_split


def train_ddg_model(
    model: PearlWithDDG,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir: str = './results/ddg_training'
):
    """
    Train the ΔΔG prediction model.
    
    Args:
        model: PearlWithDDG model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save results
    """
    print(f"Training on device: {device}")
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = DDGLoss()
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': [],
        'val_pearson_r': [],
        'val_rmse': []
    }
    
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        train_maes = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                wt_protein_features=batch['wt_protein_features'],
                wt_ligand_features=batch['wt_ligand_features'],
                mut_protein_features=batch['mut_protein_features'],
                mut_ligand_features=batch['mut_ligand_features'],
                protein_mask=batch['protein_mask'],
                ligand_mask=batch['ligand_mask']
            )
            
            # Compute loss
            loss_dict = criterion(
                pred_ddg=outputs['ddg'],
                true_ddg=batch['ddg_true'],
                ddg_confidence=outputs['ddg_confidence'],
                residue_contrib=outputs['residue_contrib'],
                data_weight=batch['weight']
            )
            
            # Backward pass
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss_dict['total'].item())
            train_maes.append(loss_dict['mae'].item())
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total'].item():.4f}",
                'mae': f"{loss_dict['mae'].item():.4f}"
            })
        
        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_trues = []
        val_confidences = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in pbar:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                outputs = model(
                    wt_protein_features=batch['wt_protein_features'],
                    wt_ligand_features=batch['wt_ligand_features'],
                    mut_protein_features=batch['mut_protein_features'],
                    mut_ligand_features=batch['mut_ligand_features'],
                    protein_mask=batch['protein_mask'],
                    ligand_mask=batch['ligand_mask']
                )
                
                loss_dict = criterion(
                    pred_ddg=outputs['ddg'],
                    true_ddg=batch['ddg_true'],
                    ddg_confidence=outputs['ddg_confidence'],
                    residue_contrib=outputs['residue_contrib']
                )
                
                val_losses.append(loss_dict['total'].item())
                val_preds.extend(outputs['ddg'].cpu().numpy())
                val_trues.extend(batch['ddg_true'].cpu().numpy())
                val_confidences.extend(outputs['ddg_confidence'].cpu().numpy())
        
        # Compute metrics
        val_preds = torch.tensor(val_preds)
        val_trues = torch.tensor(val_trues)
        val_confidences = torch.tensor(val_confidences)
        
        metrics = DDGMetrics.compute_metrics(val_preds, val_trues, val_confidences)
        
        # Update history
        history['train_loss'].append(np.mean(train_losses))
        history['train_mae'].append(np.mean(train_maes))
        history['val_loss'].append(np.mean(val_losses))
        history['val_mae'].append(metrics['mae'])
        history['val_pearson_r'].append(metrics['pearson_r'])
        history['val_rmse'].append(metrics['rmse'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {history['train_loss'][-1]:.4f}, Train MAE: {history['train_mae'][-1]:.4f}")
        print(f"  Val Loss: {history['val_loss'][-1]:.4f}, Val MAE: {metrics['mae']:.4f}")
        print(f"  Val Pearson R: {metrics['pearson_r']:.4f}, Val RMSE: {metrics['rmse']:.4f}")
        print(f"  Val Calibration Error: {metrics['calibration_error']:.4f}")
        print(f"  Within CI: {metrics['within_ci']:.2%}")
        print("=" * 80)
        
        # Save best model
        if history['val_loss'][-1] < best_val_loss:
            best_val_loss = history['val_loss'][-1]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'metrics': metrics
            }, save_dir / 'best_ddg_model.pt')
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Learning rate schedule
        scheduler.step()
    
    # Save training history
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved to: {save_dir}")
    
    return history


def main():
    """Main training function"""
    print("=" * 80)
    print("ΔΔG Prediction Training with Pearl")
    print("=" * 80)
    
    # Configuration
    config = {
        'num_samples': 2000,
        'n_protein': 100,
        'n_ligand': 20,
        'protein_feature_dim': 64,
        'ligand_feature_dim': 64,
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'val_fraction': 0.2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    full_dataset = SyntheticDDGDataset(
        num_samples=config['num_samples'],
        n_protein=config['n_protein'],
        n_ligand=config['n_ligand'],
        protein_feature_dim=config['protein_feature_dim'],
        ligand_feature_dim=config['ligand_feature_dim']
    )
    
    # Split into train/val
    train_dataset, val_dataset = create_train_val_split(
        full_dataset, val_fraction=config['val_fraction']
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_ddg_batch,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_ddg_batch,
        num_workers=0
    )
    
    # Create base Pearl model
    print("\nCreating Pearl model...")
    base_pearl = MockPearl(
        protein_feature_dim=config['protein_feature_dim'],
        ligand_feature_dim=config['ligand_feature_dim'],
        pair_dim=128,
        trunk_blocks=4,
        trunk_heads=8
    )
    
    # Create ΔΔG model
    print("Creating ΔΔG prediction model...")
    model = PearlWithDDG(base_pearl, freeze_pearl=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Train model
    history = train_ddg_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        device=config['device']
    )
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()

