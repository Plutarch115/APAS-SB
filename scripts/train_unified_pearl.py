#!/usr/bin/env python3
"""
Train Unified Pearl Model: Protein-Ligand + Protein-Protein

This script trains Pearl on both:
1. Protein-ligand cofolding (small molecule drug discovery)
2. Protein-protein interaction prediction (biologics, antibodies)

Features:
- Multi-task learning
- Curriculum learning
- Uncertainty-aware training with MD data
- Efficient GPU utilization
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add pearl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pearl.models.pearl import Pearl
from pearl.training.unified_trainer import UnifiedPearlTrainer, UnifiedTrainingConfig
from pearl.data.pdb_loader import PDBDataset
from pearl.data.ppi_loader import PPIDataset
from pearl.data.preprocessing import ComplexPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_dataloaders(args):
    """Create dataloaders for protein-ligand and protein-protein tasks"""
    
    logger.info("Creating dataloaders...")
    
    # Protein-ligand dataset
    logger.info(f"Loading protein-ligand data from {args.ligand_data_dir}")
    ligand_dataset = PDBDataset(
        pdb_dir=args.ligand_data_dir,
        max_structures=args.max_ligand_structures
    )
    
    ligand_dataloader = DataLoader(
        ligand_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Protein-protein dataset
    logger.info(f"Loading protein-protein data from {args.ppi_data_dir}")
    ppi_dataset = PPIDataset(
        pdb_dir=args.ppi_data_dir,
        ppi_list_file=args.ppi_list_file
    )
    
    ppi_dataloader = DataLoader(
        ppi_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Protein-ligand dataset: {len(ligand_dataset)} structures")
    logger.info(f"Protein-protein dataset: {len(ppi_dataset)} complexes")
    
    # Validation dataloaders (optional)
    val_ligand_dataloader = None
    val_ppi_dataloader = None
    
    if args.val_ligand_data_dir:
        val_ligand_dataset = PDBDataset(
            pdb_dir=args.val_ligand_data_dir,
            max_structures=args.max_val_structures
        )
        val_ligand_dataloader = DataLoader(
            val_ligand_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        logger.info(f"Validation ligand dataset: {len(val_ligand_dataset)} structures")
    
    if args.val_ppi_data_dir:
        val_ppi_dataset = PPIDataset(
            pdb_dir=args.val_ppi_data_dir,
            ppi_list_file=args.val_ppi_list_file
        )
        val_ppi_dataloader = DataLoader(
            val_ppi_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        logger.info(f"Validation PPI dataset: {len(val_ppi_dataset)} complexes")
    
    return ligand_dataloader, ppi_dataloader, val_ligand_dataloader, val_ppi_dataloader


def main():
    parser = argparse.ArgumentParser(description="Train unified Pearl model")
    
    # Data directories
    parser.add_argument("--ligand-data-dir", required=True, help="Protein-ligand data directory")
    parser.add_argument("--ppi-data-dir", required=True, help="Protein-protein data directory")
    parser.add_argument("--ppi-list-file", help="PPI list file (pdb_id chain_a chain_b)")
    
    # Validation data
    parser.add_argument("--val-ligand-data-dir", help="Validation ligand data directory")
    parser.add_argument("--val-ppi-data-dir", help="Validation PPI data directory")
    parser.add_argument("--val-ppi-list-file", help="Validation PPI list file")
    
    # Dataset sizes
    parser.add_argument("--max-ligand-structures", type=int, default=None, 
                       help="Max ligand structures to load")
    parser.add_argument("--max-val-structures", type=int, default=1000,
                       help="Max validation structures")
    
    # Model parameters
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping")
    
    # Multi-task parameters
    parser.add_argument("--ligand-task-weight", type=float, default=1.0, 
                       help="Weight for ligand task")
    parser.add_argument("--ppi-task-weight", type=float, default=1.0,
                       help="Weight for PPI task")
    parser.add_argument("--task-sampling", default="balanced",
                       choices=["balanced", "proportional", "curriculum"],
                       help="Task sampling strategy")
    parser.add_argument("--curriculum-stages", nargs="+", default=["ligand", "ppi", "mixed"],
                       help="Curriculum stages")
    
    # Uncertainty-aware training
    parser.add_argument("--use-uncertainty-weighting", action="store_true",
                       help="Use uncertainty-aware weighting")
    parser.add_argument("--use-md-confidence", action="store_true",
                       help="Use MD-derived confidence scores")
    
    # Optimization
    parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw"],
                       help="Optimizer")
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "linear", "none"],
                       help="Learning rate scheduler")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Warmup steps")
    
    # Logging and checkpointing
    parser.add_argument("--log-frequency", type=int, default=100, help="Log frequency")
    parser.add_argument("--eval-frequency", type=int, default=1000, help="Eval frequency")
    parser.add_argument("--save-frequency", type=int, default=5000, help="Save frequency")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb-project", default="pearl-unified", help="W&B project name")
    
    # Device
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device")
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    
    # Checkpoint
    parser.add_argument("--resume-from", help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Print configuration
    logger.info("="*80)
    logger.info("UNIFIED PEARL TRAINING")
    logger.info("="*80)
    logger.info(f"Ligand data: {args.ligand_data_dir}")
    logger.info(f"PPI data: {args.ppi_data_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Task sampling: {args.task_sampling}")
    logger.info(f"Curriculum stages: {args.curriculum_stages}")
    logger.info(f"Uncertainty weighting: {args.use_uncertainty_weighting}")
    logger.info(f"MD confidence: {args.use_md_confidence}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Mixed precision: {args.mixed_precision}")
    logger.info("="*80)
    
    # Create dataloaders
    ligand_dataloader, ppi_dataloader, val_ligand_dataloader, val_ppi_dataloader = create_dataloaders(args)
    
    # Create model
    logger.info("Creating Pearl model...")
    model = Pearl(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create training configuration
    config = UnifiedTrainingConfig(
        ligand_task_weight=args.ligand_task_weight,
        ppi_task_weight=args.ppi_task_weight,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        task_sampling=args.task_sampling,
        curriculum_stages=args.curriculum_stages,
        use_uncertainty_weighting=args.use_uncertainty_weighting,
        use_md_confidence=args.use_md_confidence,
        optimizer=args.optimizer,
        scheduler=args.scheduler if args.scheduler != "none" else None,
        warmup_steps=args.warmup_steps,
        log_frequency=args.log_frequency,
        eval_frequency=args.eval_frequency,
        save_frequency=args.save_frequency,
        use_wandb=args.use_wandb,
        device=args.device,
        mixed_precision=args.mixed_precision
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = UnifiedPearlTrainer(
        model=model,
        config=config,
        ligand_dataloader=ligand_dataloader,
        ppi_dataloader=ppi_dataloader,
        val_ligand_dataloader=val_ligand_dataloader,
        val_ppi_dataloader=val_ppi_dataloader
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)


if __name__ == "__main__":
    main()

