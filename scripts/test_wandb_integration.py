"""
Quick test script for W&B integration.

Tests:
1. W&B initialization and configuration
2. Dataset loading and statistics logging
3. Model creation and architecture logging
4. Training loop with metrics logging
5. Checkpoint saving and artifact upload

Usage:
    # Test with synthetic data (fast)
    python scripts/test_wandb_integration.py
    
    # Test with custom config
    python scripts/test_wandb_integration.py --config scripts/wandb_config.yaml
"""

import os
import sys
import argparse
from pathlib import Path
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Weights & Biases
import wandb

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pearl.data.multitask_datasets import create_multitask_dataset
from pearl.training.boltz2_losses import CombinedBoltz2Loss
from pearl.models.multitask_pearl import MultiTaskPEARL
from pearl.models.mock_pearl import MockPearl


def test_wandb_init():
    """Test W&B initialization."""
    print("\n" + "="*60)
    print("TEST 1: W&B Initialization")
    print("="*60)
    
    run = wandb.init(
        project="apas-sb-test",
        name="test_run",
        config={
            'test': True,
            'batch_size': 2,
            'learning_rate': 1e-4,
        },
        tags=['test'],
        mode='online',  # Use 'offline' if you don't want to sync
    )
    
    print(f"✅ W&B initialized successfully!")
    print(f"   Run ID: {run.id}")
    print(f"   Run URL: {run.url}")
    
    return run


def test_dataset_loading(run):
    """Test dataset loading and logging."""
    print("\n" + "="*60)
    print("TEST 2: Dataset Loading & Statistics")
    print("="*60)
    
    # Create small synthetic dataset
    data_dirs = {
        'chembl': Path('./data/chembl'),
        'bindingdb': Path('./data/bindingdb'),
    }
    
    dataset = create_multitask_dataset(
        data_dirs,
        split='train',
        use_synthetic=True
    )
    
    print(f"✅ Dataset created: {len(dataset)} samples")
    
    # Log dataset statistics
    wandb.log({
        'dataset/total_size': len(dataset),
        'dataset/num_datasets': len(data_dirs),
    })
    
    # Sample a few examples
    sample = dataset[0]
    print(f"   Sample keys: {sample.keys()}")
    print(f"   Target: {sample['target']}")
    print(f"   Weight: {sample['weight']}")
    print(f"   Task: {sample['task']}")
    
    return dataset


def test_model_creation(run):
    """Test model creation and logging."""
    print("\n" + "="*60)
    print("TEST 3: Model Creation & Architecture Logging")
    print("="*60)

    # Create base PEARL model
    base_pearl = MockPearl(
        protein_feature_dim=64,
        ligand_feature_dim=64,
        pair_dim=128,
        trunk_blocks=4,
        trunk_heads=4,
    )

    # Create multi-task model
    model = MultiTaskPEARL(
        base_pearl=base_pearl,
        pair_dim=128,
        hidden_dim=256,  # Smaller for testing
        num_heads=4,
        freeze_pearl=False  # Allow training for testing
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / (1024**2):.2f} MB")
    
    # Log to W&B
    wandb.config.update({
        'model/total_parameters': total_params,
        'model/trainable_parameters': trainable_params,
        'model/size_mb': total_params * 4 / (1024**2),
    })
    
    return model


def test_training_loop(model, dataset, run):
    """Test training loop with logging."""
    print("\n" + "="*60)
    print("TEST 4: Training Loop & Metrics Logging")
    print("="*60)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Create loss and optimizer
    criterion = CombinedBoltz2Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"   Device: {device}")
    print(f"   Training for 3 epochs with {len(dataloader)} batches each...")
    
    # Training loop
    for epoch in range(3):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            optimizer.zero_grad()
            
            try:
                # Get task type (assume binding_affinity for now)
                task = batch['task'][0] if isinstance(batch['task'], (list, tuple)) else 'binding_affinity'

                # Forward pass through model
                outputs_dict = model(batch, task=task)

                # Extract predictions
                outputs = outputs_dict['affinity']

                # Handle targets - ensure they're 1D
                targets = batch['target']
                if targets.dim() > 1:
                    targets = targets.squeeze()

                task_types = batch['task']
                weights = batch.get('weight', None)
                if weights is not None and weights.dim() > 1:
                    weights = weights.squeeze()

                loss, loss_dict = criterion(outputs, targets, task_types, weights)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Log to W&B
                global_step = epoch * len(dataloader) + batch_idx
                wandb.log({
                    'train/loss': loss.item(),
                    'train/loss_huber': loss_dict.get('huber', 0.0),
                    'train/loss_ranking': loss_dict.get('ranking', 0.0),
                    'train/loss_focal': loss_dict.get('focal', 0.0),
                    'train/epoch': epoch,
                    'train/batch': batch_idx,
                }, step=global_step)
                
            except Exception as e:
                print(f"   ⚠️  Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"   Epoch {epoch}: loss={avg_loss:.4f}")
        
        # Log epoch summary
        wandb.log({
            'epoch/train_loss': avg_loss,
            'epoch/number': epoch,
        }, step=epoch)
    
    print(f"✅ Training loop completed successfully!")
    
    return model


def test_checkpoint_saving(model, run):
    """Test checkpoint saving and artifact upload."""
    print("\n" + "="*60)
    print("TEST 5: Checkpoint Saving & Artifact Upload")
    print("="*60)
    
    # Create checkpoint directory
    checkpoint_dir = Path('./test_checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / 'test_checkpoint.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'test': True,
    }, checkpoint_path)
    
    print(f"✅ Checkpoint saved to {checkpoint_path}")
    
    # Upload as W&B artifact
    artifact = wandb.Artifact(
        name='test-model-checkpoint',
        type='model',
        description='Test checkpoint for W&B integration'
    )
    artifact.add_file(str(checkpoint_path))
    wandb.log_artifact(artifact)
    
    print(f"✅ Artifact uploaded to W&B")
    
    # Cleanup
    checkpoint_path.unlink()
    checkpoint_dir.rmdir()


def main():
    parser = argparse.ArgumentParser(description='Test W&B integration')
    parser.add_argument('--config', type=str, default='scripts/wandb_config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("APAS-SB Weights & Biases Integration Test")
    print("="*60)
    
    # Run tests
    try:
        run = test_wandb_init()
        dataset = test_dataset_loading(run)
        model = test_model_creation(run)
        model = test_training_loop(model, dataset, run)
        test_checkpoint_saving(model, run)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print(f"\nView your run at: {run.url}")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    
    finally:
        # Finish W&B run
        wandb.finish()
        print("\nW&B run finished.")


if __name__ == '__main__':
    main()

