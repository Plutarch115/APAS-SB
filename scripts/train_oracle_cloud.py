"""
Oracle Cloud Training Script for APAS-SB.

Distributed training on 64 H100 GPUs with:
- Multi-task learning (11 datasets)
- Boltz-2 loss functions
- MD trajectory integration (mdCATH + ATLAS)
- Density-aware training

Aligned with APAS-SB_Development_Roadmap.md Phase 2 (Days 13-62).

Usage:
    # Single node (8 GPUs)
    python scripts/train_oracle_cloud.py --config scripts/oracle_cloud_config.yaml --phase 2a
    
    # Multi-node (64 GPUs)
    torchrun --nproc_per_node=8 --nnodes=8 --node_rank=$NODE_RANK \\
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \\
        scripts/train_oracle_cloud.py --config scripts/oracle_cloud_config.yaml --phase 2c
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pearl.data.multitask_datasets import create_multitask_dataset
from pearl.data.mdcath_loader import mdCATHDataset
from pearl.data.atlas_loader import ATLASDataset
from pearl.training.boltz2_losses import CombinedBoltz2Loss


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed training")
        return 0, 1, 0
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_datasets(config: Dict, phase: str, rank: int):
    """Create training datasets based on phase"""
    phase_config = config['training'][f'phase_{phase}']
    dataset_names = phase_config['datasets']
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Creating datasets for Phase {phase.upper()}")
        print(f"{'='*60}")
        print(f"Datasets: {', '.join(dataset_names)}")
    
    # Create data directories dict
    data_dirs = {name: f"{config['infrastructure']['storage']['data_dir']}/{name}" 
                 for name in dataset_names if name not in ['mdcath', 'atlas']}
    
    # Create multi-task dataset (binding affinity, ddg, etc.)
    datasets = []
    
    if data_dirs:
        multitask_dataset = create_multitask_dataset(
            data_dirs=data_dirs,
            split='train',
            use_synthetic=True  # Use synthetic for testing
        )
        datasets.append(multitask_dataset)
        
        if rank == 0:
            print(f"✓ Multi-task dataset: {len(multitask_dataset)} samples")
    
    # Add MD trajectory datasets if in phase 2c
    if 'mdcath' in dataset_names:
        md_config = phase_config.get('md_config', {})
        mdcath_dataset = mdCATHDataset(
            data_dir=f"{config['infrastructure']['storage']['data_dir']}/mdcath",
            temperature=md_config.get('mdcath_temperature', 320),
            max_frames=md_config.get('mdcath_max_frames', 100),
            stride=md_config.get('mdcath_stride', 10),
            compute_density=md_config.get('compute_density', True),
            use_synthetic=True
        )
        datasets.append(mdcath_dataset)
        
        if rank == 0:
            print(f"✓ mdCATH dataset: {len(mdcath_dataset)} trajectories")
    
    if 'atlas' in dataset_names:
        md_config = phase_config.get('md_config', {})
        atlas_dataset = ATLASDataset(
            data_dir=f"{config['infrastructure']['storage']['data_dir']}/atlas",
            max_frames=md_config.get('atlas_max_frames', 100),
            stride=md_config.get('atlas_stride', 10),
            compute_density=md_config.get('compute_density', True),
            use_synthetic=True
        )
        datasets.append(atlas_dataset)
        
        if rank == 0:
            print(f"✓ ATLAS dataset: {len(atlas_dataset)} trajectories")
    
    # Combine all datasets
    if len(datasets) == 1:
        combined_dataset = datasets[0]
    else:
        combined_dataset = ConcatDataset(datasets)
    
    if rank == 0:
        print(f"\n✓ Total dataset size: {len(combined_dataset)} samples")
    
    return combined_dataset


def create_dataloader(dataset, config: Dict, phase: str, rank: int, world_size: int):
    """Create distributed dataloader"""
    phase_config = config['training'][f'phase_{phase}']
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create dataloader
    # Note: batch_size=1 due to variable-size samples
    # Actual batching handled by gradient accumulation
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader, sampler


def create_model(config: Dict):
    """Create PEARL model with multi-task heads"""
    # TODO: Implement actual PEARL model
    # This is a placeholder
    model_config = config['model']
    
    # Placeholder: simple model for testing
    model = torch.nn.Sequential(
        torch.nn.Linear(100, model_config['hidden_dim']),
        torch.nn.ReLU(),
        torch.nn.Linear(model_config['hidden_dim'], 1)
    )
    
    return model


def train_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    config: Dict,
    phase: str,
    epoch: int,
    rank: int
):
    """Train for one epoch"""
    model.train()
    phase_config = config['training'][f'phase_{phase}']
    grad_accum_steps = phase_config['gradient_accumulation_steps']

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # TODO: Implement actual forward pass
        # This is a placeholder

        # Dummy loss for testing
        loss = torch.tensor(0.1, requires_grad=True)

        # Backward pass with gradient accumulation
        loss = loss / grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['optimization']['grad_clip']
            )

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1

        # Logging
        if rank == 0 and batch_idx % config['monitoring']['log_interval'] == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Oracle Cloud Training for APAS-SB')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--phase', type=str, required=True,
                        choices=['2a', '2b', '2c'],
                        help='Training phase (2a, 2b, or 2c)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print("\n" + "🚀" * 40)
        print("APAS-SB ORACLE CLOUD TRAINING")
        print(f"Phase {args.phase.upper()}")
        print("🚀" * 40)
        print(f"\nDistributed Training:")
        print(f"  World size: {world_size}")
        print(f"  Rank: {rank}")
        print(f"  Local rank: {local_rank}")

    # Load configuration
    config = load_config(args.config)
    phase_config = config['training'][f'phase_{args.phase}']

    if rank == 0:
        print(f"\nPhase Configuration:")
        print(f"  Name: {phase_config['name']}")
        print(f"  GPUs: {phase_config['gpus']}")
        print(f"  Duration: {phase_config['duration_days']} days")
        print(f"  Effective batch size: {phase_config['effective_batch_size']}")
        print(f"  Learning rate: {phase_config['learning_rate']}")

    # Create datasets
    dataset = create_datasets(config, args.phase, rank)

    # Create dataloader
    dataloader, sampler = create_dataloader(dataset, config, args.phase, rank, world_size)

    if rank == 0:
        print(f"\nDataLoader created:")
        print(f"  Batches per epoch: {len(dataloader)}")

    # Create model
    model = create_model(config)

    # Move to device
    if torch.cuda.is_available():
        model = model.to(local_rank)

    # Wrap with DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=phase_config['learning_rate'],
        betas=config['optimization']['betas'],
        weight_decay=config['optimization']['weight_decay']
    )

    # Create loss function
    loss_fn = CombinedBoltz2Loss()

    if rank == 0:
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}")

    # Training loop (simplified for testing)
    num_epochs = 2  # Just test with 2 epochs

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)

        avg_loss = train_epoch(
            model, dataloader, loss_fn, optimizer,
            config, args.phase, epoch, rank
        )

        if rank == 0:
            print(f"\nEpoch {epoch} completed. Average loss: {avg_loss:.4f}")

    if rank == 0:
        print(f"\n{'='*60}")
        print("✅ Training completed successfully!")
        print(f"{'='*60}")

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
