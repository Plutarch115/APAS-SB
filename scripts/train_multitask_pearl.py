"""
Training script for multi-task PEARL model.

Trains PEARL to predict multiple biochemical properties:
- Binding affinity (PDBbind)
- ΔΔG for protein-protein interactions (SKEMPI 2.0)
- Catalytic activity kcat (BRENDA)
- Fitness scores (ProteinGym)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

# Import PEARL components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pearl.models.mock_pearl import MockPearl
from pearl.models.multitask_pearl import MultiTaskPEARL
from pearl.data.multitask_datasets import (
    PDBBindDataset,
    SKEMPI2Dataset,
    BRENDADataset,
    ProteinGymDataset,
    create_multitask_dataset
)
from pearl.training.ddg_losses import DDGMetrics


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function with task-specific weighting.
    """
    
    def __init__(
        self,
        task_weights: Optional[Dict[str, float]] = None,
        use_uncertainty: bool = True
    ):
        super().__init__()
        
        # Default task weights
        self.task_weights = task_weights or {
            'binding_affinity': 1.0,
            'ddg_ppi': 1.0,
            'kcat': 1.0,
            'fitness': 1.0
        }
        
        self.use_uncertainty = use_uncertainty
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        task: str,
        data_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for a specific task.
        
        Args:
            predictions: Dictionary with task-specific predictions
            targets: Ground truth values [batch]
            task: Task name
            data_weights: Per-sample weights [batch]
        
        Returns:
            Dictionary with loss components
        """
        # Get predicted values and confidence
        if task == 'binding_affinity':
            pred_values = predictions['affinity']
            confidence = predictions['confidence']
        elif task == 'ddg_ppi':
            pred_values = predictions['ddg']
            confidence = predictions['ddg_confidence']
        elif task == 'kcat':
            pred_values = predictions['log_kcat']
            confidence = predictions['confidence']
        elif task == 'fitness':
            pred_values = predictions['fitness']
            confidence = predictions['confidence']
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # MSE loss
        mse_loss = F.mse_loss(pred_values, targets, reduction='none')
        
        # Apply data weights if provided
        if data_weights is not None:
            mse_loss = mse_loss * data_weights
        
        mse_loss = mse_loss.mean()
        
        # Uncertainty-aware loss (negative log-likelihood)
        if self.use_uncertainty:
            nll_loss = 0.5 * (
                torch.log(2 * np.pi * confidence**2) +
                ((pred_values - targets)**2) / (confidence**2)
            )
            
            if data_weights is not None:
                nll_loss = nll_loss * data_weights
            
            nll_loss = nll_loss.mean()
        else:
            nll_loss = torch.tensor(0.0, device=pred_values.device)
        
        # Total loss with task weighting
        task_weight = self.task_weights.get(task, 1.0)
        total_loss = task_weight * (mse_loss + 0.5 * nll_loss)
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'nll_loss': nll_loss,
            'task_weight': task_weight
        }


def collate_multitask_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for multi-task batches.
    
    Handles different input formats for different tasks.
    """
    # Determine task from first item
    task = batch[0]['task']
    
    if task == 'binding_affinity':
        return {
            'protein_features': torch.stack([item['protein_features'] for item in batch]),
            'ligand_features': torch.stack([item['ligand_features'] for item in batch]),
            'target': torch.stack([item['target'] for item in batch]),
            'weight': torch.stack([item['weight'] for item in batch]),
            'task': task
        }
    
    elif task in ['ddg_ppi', 'fitness']:
        return {
            'wt_protein_features': torch.stack([item['wt_protein_features'] for item in batch]),
            'mut_protein_features': torch.stack([item['mut_protein_features'] for item in batch]),
            'target': torch.stack([item['target'] for item in batch]),
            'weight': torch.stack([item['weight'] for item in batch]),
            'task': task
        }
    
    elif task == 'kcat':
        return {
            'enzyme_features': torch.stack([item['enzyme_features'] for item in batch]),
            'substrate_features': torch.stack([item['substrate_features'] for item in batch]),
            'target': torch.stack([item['target'] for item in batch]),
            'weight': torch.stack([item['weight'] for item in batch]),
            'task': task
        }
    
    else:
        raise ValueError(f"Unknown task: {task}")


def train_multitask_model(
    model: MultiTaskPEARL,
    train_loaders: Dict[str, DataLoader],
    val_loaders: Dict[str, DataLoader],
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    device: str = 'cpu',
    save_dir: str = 'results/multitask_training'
):
    """
    Train multi-task PEARL model.
    
    Args:
        model: MultiTaskPEARL model
        train_loaders: Dictionary of training dataloaders for each task
        val_loaders: Dictionary of validation dataloaders for each task
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save results
    """
    # Setup
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = MultiTaskLoss()
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'task_losses': {task: [] for task in train_loaders.keys()},
        'task_metrics': {task: [] for task in train_loaders.keys()}
    }
    
    best_val_loss = float('inf')
    
    print(f"Starting multi-task training on {device}")
    print(f"Tasks: {list(train_loaders.keys())}")
    print(f"Training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = {task: [] for task in train_loaders.keys()}
        
        # Iterate through all tasks
        for task_name, train_loader in train_loaders.items():
            for batch in train_loader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                predictions = model(batch, task=batch['task'])
                
                # Compute loss
                loss_dict = criterion(
                    predictions,
                    batch['target'],
                    task=batch['task'],
                    data_weights=batch.get('weight')
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss_dict['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Record loss
                train_losses[task_name].append(loss_dict['total_loss'].item())
        
        # Validation phase
        model.eval()
        val_losses = {task: [] for task in val_loaders.keys()}
        val_metrics = {task: {'predictions': [], 'targets': []} for task in val_loaders.keys()}
        
        with torch.no_grad():
            for task_name, val_loader in val_loaders.items():
                for batch in val_loader:
                    # Move batch to device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Forward pass
                    predictions = model(batch, task=batch['task'])
                    
                    # Compute loss
                    loss_dict = criterion(
                        predictions,
                        batch['target'],
                        task=batch['task'],
                        data_weights=batch.get('weight')
                    )
                    
                    # Record loss and predictions
                    val_losses[task_name].append(loss_dict['total_loss'].item())
                    
                    # Get predicted values based on task
                    if task_name == 'binding_affinity':
                        pred_values = predictions['affinity']
                    elif task_name == 'ddg_ppi':
                        pred_values = predictions['ddg']
                    elif task_name == 'kcat':
                        pred_values = predictions['log_kcat']
                    elif task_name == 'fitness':
                        pred_values = predictions['fitness']
                    
                    val_metrics[task_name]['predictions'].extend(pred_values.cpu().numpy())
                    val_metrics[task_name]['targets'].extend(batch['target'].cpu().numpy())
        
        # Compute epoch statistics
        avg_train_loss = np.mean([np.mean(losses) for losses in train_losses.values()])
        avg_val_loss = np.mean([np.mean(losses) for losses in val_losses.values()])
        
        # Compute metrics for each task
        task_metrics_summary = {}
        for task_name in val_metrics.keys():
            preds = np.array(val_metrics[task_name]['predictions'])
            targets = np.array(val_metrics[task_name]['targets'])
            
            # Compute Pearson R and MAE
            pearson_r = np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else 0.0
            mae = np.mean(np.abs(preds - targets))
            
            task_metrics_summary[task_name] = {
                'pearson_r': pearson_r,
                'mae': mae
            }
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        for task_name in train_loaders.keys():
            history['task_losses'][task_name].append(np.mean(train_losses[task_name]))
            history['task_metrics'][task_name].append(task_metrics_summary[task_name])
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        for task_name, metrics in task_metrics_summary.items():
            print(f"  {task_name}: R={metrics['pearson_r']:.3f}, MAE={metrics['mae']:.3f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, os.path.join(save_dir, 'best_multitask_model.pt'))
            print(f"  ✓ Saved best model (val_loss={avg_val_loss:.4f})")
        
        # Update learning rate
        scheduler.step()
    
    # Save training history
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        history_serializable = {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'task_losses': {
                task: [float(x) for x in losses]
                for task, losses in history['task_losses'].items()
            },
            'task_metrics': {
                task: [
                    {k: float(v) for k, v in metrics.items()}
                    for metrics in metrics_list
                ]
                for task, metrics_list in history['task_metrics'].items()
            }
        }
        json.dump(history_serializable, f, indent=2)
    
    print(f"\n✓ Training complete! Results saved to {save_dir}")
    
    return history


if __name__ == '__main__':
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create synthetic datasets for testing
    data_dirs = {
        'pdbind': 'data/pdbind',
        'skempi2': 'data/skempi2',
        'brenda': 'data/brenda',
        'proteingym': 'data/proteingym'
    }
    
    # Create datasets
    print("Creating datasets...")
    train_datasets = {
        'binding_affinity': PDBBindDataset(data_dirs['pdbind'], split='train', use_synthetic=True),
        'ddg_ppi': SKEMPI2Dataset(data_dirs['skempi2'], split='train', use_synthetic=True),
        'kcat': BRENDADataset(data_dirs['brenda'], split='train', use_synthetic=True),
        'fitness': ProteinGymDataset(data_dirs['proteingym'], split='train', use_synthetic=True)
    }
    
    val_datasets = {
        'binding_affinity': PDBBindDataset(data_dirs['pdbind'], split='val', use_synthetic=True),
        'ddg_ppi': SKEMPI2Dataset(data_dirs['skempi2'], split='val', use_synthetic=True),
        'kcat': BRENDADataset(data_dirs['brenda'], split='val', use_synthetic=True),
        'fitness': ProteinGymDataset(data_dirs['proteingym'], split='val', use_synthetic=True)
    }
    
    # Create dataloaders
    train_loaders = {
        task: DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_multitask_batch)
        for task, dataset in train_datasets.items()
    }
    
    val_loaders = {
        task: DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_multitask_batch)
        for task, dataset in val_datasets.items()
    }
    
    print(f"Dataset sizes:")
    for task, dataset in train_datasets.items():
        print(f"  {task}: {len(dataset)} train, {len(val_datasets[task])} val")
    
    # Create model
    print("\nCreating model...")
    base_pearl = MockPearl(
        protein_feature_dim=64,
        ligand_feature_dim=64,
        pair_dim=128,
        trunk_blocks=4,
        trunk_heads=8
    )
    
    model = MultiTaskPEARL(
        base_pearl=base_pearl,
        pair_dim=128,
        hidden_dim=512,
        num_heads=8,
        freeze_pearl=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train model
    print("\nStarting training...")
    history = train_multitask_model(
        model=model,
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        num_epochs=20,
        learning_rate=1e-4,
        device=device,
        save_dir='results/multitask_training'
    )
    
    print("\n✓ All done!")

