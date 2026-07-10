"""
Training script with Weights & Biases integration for APAS-SB.

Features:
- Real-time metrics logging (loss, accuracy, learning rate)
- Model architecture visualization
- Gradient tracking and histograms
- Dataset statistics and sample visualization
- Hyperparameter tracking
- System metrics (GPU utilization, memory)
- Checkpoint management with W&B artifacts

Usage:
    # Local testing (single GPU)
    python scripts/train_with_wandb.py --config scripts/wandb_config.yaml --test
    
    # Full training (single node)
    python scripts/train_with_wandb.py --config scripts/wandb_config.yaml --phase 2a
    
    # Distributed training (multi-node)
    torchrun --nproc_per_node=8 --nnodes=8 \
        scripts/train_with_wandb.py --config scripts/wandb_config.yaml --phase 2c
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Optional, List
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

# Weights & Biases
import wandb

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pearl.data.multitask_datasets import create_multitask_dataset
from pearl.data.mdcath_loader import mdCATHDataset
from pearl.data.atlas_loader import ATLASDataset
from pearl.training.boltz2_losses import CombinedBoltz2Loss
from pearl.models.multitask_pearl import MultiTaskPEARL
from pearl.models.mock_pearl import MockPearl


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Single GPU or CPU
        return 0, 1, 0


def setup_wandb(config: Dict, rank: int, world_size: int, args):
    """Initialize Weights & Biases logging."""
    # Only initialize W&B on rank 0
    if rank == 0:
        # Create run name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        phase = args.phase if hasattr(args, 'phase') else 'test'
        run_name = f"apas-sb_{phase}_{timestamp}"
        
        # Initialize W&B
        wandb.init(
            project=config.get('wandb', {}).get('project', 'apas-sb'),
            entity=config.get('wandb', {}).get('entity', None),
            name=run_name,
            config={
                **config,
                'world_size': world_size,
                'phase': phase,
                'timestamp': timestamp,
            },
            tags=[phase, f"gpus_{world_size}"],
            notes=f"Training phase {phase} on {world_size} GPUs",
            resume='allow',  # Allow resuming if run crashes
        )
        
        # Log system info
        wandb.config.update({
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__,
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'num_gpus': torch.cuda.device_count(),
        })
        
        return wandb.run
    return None


def log_dataset_stats(datasets: Dict, run):
    """Log dataset statistics to W&B."""
    if run is None:
        return
    
    stats = {}
    for name, dataset in datasets.items():
        stats[f'dataset/{name}/size'] = len(dataset)
        
        # Sample a few examples to get statistics
        if len(dataset) > 0:
            sample = dataset[0]
            if 'target' in sample:
                stats[f'dataset/{name}/target_mean'] = float(sample['target'])
            if 'weight' in sample:
                stats[f'dataset/{name}/weight'] = float(sample['weight'])
    
    wandb.log(stats, step=0)


def log_model_architecture(model: nn.Module, run):
    """Log model architecture to W&B."""
    if run is None:
        return
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    wandb.config.update({
        'model/total_parameters': total_params,
        'model/trainable_parameters': trainable_params,
        'model/parameter_size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
    })
    
    # Log model architecture as text
    wandb.run.summary['model_architecture'] = str(model)


# Datasets that currently have a real (non-synthetic) data loader implemented.
REAL_CAPABLE_DATASETS = {'bindingdb'}


def resolve_phase_key(config: Dict, phase: str) -> str:
    """Normalize a phase identifier (e.g. '2a') to its config key ('phase_2a')."""
    training = config['training']
    if phase in training:
        return phase
    if f'phase_{phase}' in training:
        return f'phase_{phase}'
    raise KeyError(
        f"Phase '{phase}' not found in config['training'] "
        f"(available: {sorted(training.keys())})"
    )


def get_feature_config(config: Dict, use_synthetic: bool) -> Dict:
    """
    Resolve the featurization settings and the resulting protein/ligand feature
    dimensions. Real ESM2 + MolFormer embeddings are used on real data; the
    fast deterministic 'hash' features (dim 64) are used in synthetic/test mode.
    """
    from pearl.data.featurizers import ESM2_DIMS, MOLFORMER_DIMS

    feat = dict(config.get('features', {}))
    default_mode = 'hash' if use_synthetic else 'esm2_molformer'
    mode = feat.get('mode', default_mode)
    # Synthetic/test data has no real sequences/SMILES to encode -> force hash.
    if use_synthetic:
        mode = 'hash'

    esm2_model = feat.get('esm2_model', 'esm2_t33_650M_UR50D')
    molformer_model = feat.get('molformer_model', 'ibm/MoLFormer-XL-both-10pct')

    if mode == 'esm2_molformer':
        protein_dim = ESM2_DIMS[esm2_model]
        ligand_dim = MOLFORMER_DIMS.get(molformer_model, 768)
    else:
        protein_dim = ligand_dim = 64

    return {
        'mode': mode,
        'esm2_model': esm2_model,
        'molformer_model': molformer_model,
        'max_protein_len': feat.get('max_protein_len', 200),
        'max_ligand_len': feat.get('max_ligand_len', 64),
        'emb_cache_dir': feat.get('emb_cache_dir'),
        'precompute': feat.get('precompute', True),
        'protein_dim': protein_dim,
        'ligand_dim': ligand_dim,
    }


def precompute_embeddings(datasets: Dict, rank: int, world_size: int, batch_size: int = 8):
    """Warm the ESM2/MolFormer embedding cache on rank 0, then sync all ranks."""
    from pearl.data.multitask_datasets import BindingDBDataset

    if rank == 0:
        for ds in datasets.values():
            members = getattr(ds, 'datasets', [ds])  # unwrap ConcatDataset
            for member in members:
                if isinstance(member, BindingDBDataset) and \
                        getattr(member, 'featurizer', 'hash') == 'esm2_molformer':
                    print("Precomputing ESM2/MolFormer embeddings (this may take a while)...")
                    member.precompute_embeddings(batch_size=batch_size)

    if world_size > 1:
        dist.barrier()  # ensure cache is populated before other ranks read it


def create_datasets(config: Dict, phase: str, rank: int, test_mode: bool = False):
    """Create datasets for training."""
    phase_config = config['training'][resolve_phase_key(config, phase)]
    dataset_names = phase_config['datasets']

    # Use synthetic data only in --test mode or if explicitly configured.
    use_synthetic = test_mode or config.get('use_synthetic', False)

    # Data directories
    data_root = Path(config.get('data_root', './data'))

    datasets = {}
    all_datasets = []

    # Multi-task datasets
    multitask_names = [
        'pdbind', 'skempi2', 'brenda', 'proteingym',
        'chembl', 'bindingdb', 'pubchem_hts', 'pubchem_small',
        'cemm', 'midas', 'synthetic_decoys'
    ]

    included_multitask = [name for name in multitask_names if name in dataset_names]

    # When running on real data, only keep datasets that have a real loader.
    # The others are synthetic-only and would raise NotImplementedError, so we
    # skip them (with a warning) to guarantee no synthetic data leaks in.
    if not use_synthetic:
        skipped = [n for n in included_multitask if n not in REAL_CAPABLE_DATASETS]
        included_multitask = [n for n in included_multitask if n in REAL_CAPABLE_DATASETS]
        if skipped and rank == 0:
            print(f"[real-data mode] Skipping synthetic-only datasets: {skipped}")
        if not included_multitask:
            raise ValueError(
                "No real-data-capable datasets selected for this phase. "
                f"Add one of {sorted(REAL_CAPABLE_DATASETS)} to the phase 'datasets' list."
            )

    # Per-dataset extra kwargs (e.g. BindingDB paths / sample caps).
    bindingdb_cfg = config.get('bindingdb', {})
    bindingdb_kwargs = {
        'tsv_path': bindingdb_cfg.get('tsv_path'),
        'processed_csv': bindingdb_cfg.get('processed_csv'),
        'max_samples': (1000 if test_mode else bindingdb_cfg.get('max_samples')),
    }

    # Featurization mode. Real ESM2 + MolFormer embeddings by default on real
    # data; deterministic 'hash' placeholders in --test / synthetic mode (so the
    # test path stays fast and does not require loading the encoders).
    feat_cfg = get_feature_config(config, use_synthetic)
    if feat_cfg['mode'] == 'esm2_molformer':
        bindingdb_kwargs.update({
            'featurizer': 'esm2_molformer',
            'esm2_model': feat_cfg['esm2_model'],
            'molformer_model': feat_cfg['molformer_model'],
            'max_protein_len': feat_cfg['max_protein_len'],
            'max_ligand_len': feat_cfg['max_ligand_len'],
            'emb_cache_dir': feat_cfg['emb_cache_dir'],
        })

    dataset_kwargs = {'bindingdb': bindingdb_kwargs}

    if included_multitask:
        if rank == 0:
            print(f"Loading multi-task datasets: {included_multitask} "
                  f"(use_synthetic={use_synthetic})")

        data_dirs = {name: data_root / name for name in included_multitask}
        multitask_dataset = create_multitask_dataset(
            data_dirs, split='train', use_synthetic=use_synthetic,
            dataset_kwargs=dataset_kwargs,
        )
        datasets['multitask'] = multitask_dataset
        all_datasets.append(multitask_dataset)

    # MD datasets (only in phase 2c)
    if 'mdcath' in dataset_names:
        if rank == 0:
            print("Loading mdCATH dataset...")
        mdcath_dataset = mdCATHDataset(
            data_dir=data_root / 'mdcath',
            temperature=350,  # Default temperature
            split='train',
            use_synthetic=use_synthetic,
            max_samples=1000 if test_mode else None
        )
        datasets['mdcath'] = mdcath_dataset
        all_datasets.append(mdcath_dataset)

    if 'atlas' in dataset_names:
        if rank == 0:
            print("Loading ATLAS dataset...")
        atlas_dataset = ATLASDataset(
            data_dir=data_root / 'atlas',
            split='train',
            use_synthetic=use_synthetic,
            max_samples=500 if test_mode else None
        )
        datasets['atlas'] = atlas_dataset
        all_datasets.append(atlas_dataset)

    # Combine all datasets
    combined_dataset = ConcatDataset(all_datasets)

    if rank == 0:
        print(f"Total dataset size: {len(combined_dataset)} samples")

    return combined_dataset, datasets


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    rank: int,
    world_size: int,
    run,
    config: Dict
):
    """Train for one epoch with W&B logging."""
    model.train()

    total_loss = 0.0
    num_batches = 0
    log_interval = config.get('log_interval', 10)

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        device = next(model.parameters()).device

        # Handle different batch formats
        if isinstance(batch, dict):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
        else:
            batch = batch.to(device)

        # Forward pass
        optimizer.zero_grad()

        try:
            # Get predictions from model
            if isinstance(batch, dict) and 'protein_features' in batch:
                # Get task type (assume binding_affinity for now)
                task = batch['task'][0] if isinstance(batch['task'], list) else 'binding_affinity'

                # Forward pass
                outputs_dict = model(batch, task=task)

                # Extract predictions
                outputs = outputs_dict['affinity']
                targets = batch['target']
                task_types = batch.get('task', ['binding_affinity'] * len(targets))
                weights = batch.get('weight', None)

                # Compute loss
                loss, loss_dict = criterion(outputs, targets, task_types, weights)
            else:
                # Fallback for simple batches
                task = 'binding_affinity'
                outputs_dict = model(batch, task=task)
                outputs = outputs_dict['affinity']
                loss = criterion(outputs, batch)
                loss_dict = {'total': loss.item()}

        except Exception as e:
            if rank == 0:
                print(f"Error in batch {batch_idx}: {e}")
            continue

        # Backward pass
        loss.backward()

        # Gradient clipping
        max_grad_norm = config.get('max_grad_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1

        # Log to W&B
        if rank == 0 and batch_idx % log_interval == 0:
            global_step = epoch * len(dataloader) + batch_idx

            log_dict = {
                'train/loss': loss.item(),
                'train/epoch': epoch,
                'train/batch': batch_idx,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
            }

            # Add individual loss components
            for loss_name, loss_value in loss_dict.items():
                log_dict[f'train/loss_{loss_name}'] = loss_value

            # Log gradient norms
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            log_dict['train/grad_norm'] = total_norm

            # Log GPU memory
            if torch.cuda.is_available():
                log_dict['system/gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
                log_dict['system/gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / 1e9

            wandb.log(log_dict, step=global_step)

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    rank: int,
    run
):
    """Validate model with W&B logging."""
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            device = next(model.parameters()).device

            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            else:
                batch = batch.to(device)

            try:
                # Get predictions
                if isinstance(batch, dict) and 'protein_features' in batch:
                    # Get task type
                    task = batch['task'][0] if isinstance(batch['task'], list) else 'binding_affinity'

                    # Forward pass
                    outputs_dict = model(batch, task=task)
                    outputs = outputs_dict['affinity']

                    targets = batch['target']
                    task_types = batch.get('task', ['binding_affinity'] * len(targets))
                    weights = batch.get('weight', None)

                    loss, _ = criterion(outputs, targets, task_types, weights)
                else:
                    task = 'binding_affinity'
                    outputs_dict = model(batch, task=task)
                    outputs = outputs_dict['affinity']
                    loss = criterion(outputs, batch)

                total_loss += loss.item()
                num_batches += 1

            except Exception as e:
                if rank == 0:
                    print(f"Validation error in batch {batch_idx}: {e}")
                continue

    avg_loss = total_loss / max(num_batches, 1)

    # Log validation metrics
    if rank == 0 and run is not None:
        wandb.log({
            'val/loss': avg_loss,
            'val/epoch': epoch,
        }, step=epoch)

    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train APAS-SB with W&B tracking')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--phase', type=str, default='2a', choices=['2a', '2b', '2c'],
                       help='Training phase')
    parser.add_argument('--test', action='store_true', help='Test mode with small dataset')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print(f"Starting training on {world_size} GPU(s)")
        print(f"Phase: {args.phase}")
        print(f"Test mode: {args.test}")

    # Setup W&B
    run = setup_wandb(config, rank, world_size, args)

    # Resolve featurization (real ESM2 + MolFormer vs. hash placeholders).
    use_synthetic = args.test or config.get('use_synthetic', False)
    feat_cfg = get_feature_config(config, use_synthetic)
    if rank == 0:
        print(f"Featurizer: {feat_cfg['mode']} "
              f"(protein_dim={feat_cfg['protein_dim']}, ligand_dim={feat_cfg['ligand_dim']})")

    # Create datasets
    train_dataset, dataset_dict = create_datasets(
        config, args.phase, rank, test_mode=args.test
    )

    # Warm the embedding cache before spawning DataLoader workers.
    if feat_cfg['mode'] == 'esm2_molformer' and feat_cfg['precompute']:
        precompute_embeddings(dataset_dict, rank, world_size)

    # Log dataset statistics
    if rank == 0:
        log_dataset_stats(dataset_dict, run)

    # Create dataloader
    phase_config = config['training'][resolve_phase_key(config, args.phase)]
    batch_size = phase_config.get('batch_size_per_gpu', 4)

    if world_size > 1:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
    else:
        dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    # Create model
    model_config = config.get('model', {})

    # Create base PEARL model. Input feature dims match the featurizer
    # (ESM2 protein dim / MolFormer ligand dim, or 64 for hash placeholders).
    base_pearl = MockPearl(
        protein_feature_dim=feat_cfg['protein_dim'],
        ligand_feature_dim=feat_cfg['ligand_dim'],
        pair_dim=model_config.get('pair_dim', 128),
        trunk_blocks=model_config.get('num_layers', 12),
        trunk_heads=model_config.get('num_heads', 8),
    )

    # Create multi-task model
    model = MultiTaskPEARL(
        base_pearl=base_pearl,
        pair_dim=model_config.get('pair_dim', 128),
        hidden_dim=model_config.get('hidden_dim', 512),
        num_heads=model_config.get('num_heads', 8),
        dropout=model_config.get('dropout', 0.1),
        freeze_pearl=False  # Allow training
    )

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Wrap with DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Log model architecture
    if rank == 0:
        log_model_architecture(model, run)

    # Create loss and optimizer
    criterion = CombinedBoltz2Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=phase_config.get('learning_rate', 1e-4),
        weight_decay=phase_config.get('weight_decay', 0.01)
    )

    # Training loop
    num_epochs = 10 if args.test else phase_config.get('num_epochs', 100)

    if rank == 0:
        print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)

        # Train
        train_loss = train_epoch(
            model, dataloader, criterion, optimizer,
            epoch, rank, world_size, run, config
        )

        if rank == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

            # Log epoch summary
            wandb.log({
                'epoch/train_loss': train_loss,
                'epoch/number': epoch,
            }, step=epoch)

            # Save checkpoint
            if (epoch + 1) % config.get('checkpoint_interval', 10) == 0:
                checkpoint_path = Path(config.get('checkpoint_dir', './checkpoints'))
                checkpoint_path.mkdir(parents=True, exist_ok=True)

                checkpoint_file = checkpoint_path / f'checkpoint_epoch_{epoch}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }, checkpoint_file)

                # Log checkpoint as W&B artifact
                artifact = wandb.Artifact(
                    name=f'model-checkpoint-epoch-{epoch}',
                    type='model',
                    description=f'Model checkpoint at epoch {epoch}'
                )
                artifact.add_file(str(checkpoint_file))
                wandb.log_artifact(artifact)

    # Finish W&B run
    if rank == 0 and run is not None:
        wandb.finish()

    # Cleanup distributed
    if world_size > 1:
        dist.destroy_process_group()

    if rank == 0:
        print("\nTraining complete!")


if __name__ == '__main__':
    main()

