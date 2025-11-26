"""
Unified Training for Pearl: Protein-Ligand + Protein-Protein

This module implements multi-task training for:
1. Protein-ligand cofolding (original Pearl)
2. Protein-protein interaction prediction (extended Pearl)

Features:
- Multi-task learning with task-specific losses
- Curriculum learning across both tasks
- Uncertainty-aware training with MD data
- Efficient GPU utilization for large-scale training
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import wandb

from pearl.models.pearl import Pearl
from pearl.training.losses import DiffusionLoss
from pearl.training.uncertainty_aware_losses import CombinedUncertaintyAwareLoss
from pearl.training.ppi_losses import CombinedPPILoss
from pearl.evaluation.metrics import compute_rmsd, compute_lddt
from pearl.evaluation.ppi_metrics import PPIMetrics

logger = logging.getLogger(__name__)


@dataclass
class UnifiedTrainingConfig:
    """Configuration for unified training"""
    
    # Task weights
    ligand_task_weight: float = 1.0
    ppi_task_weight: float = 1.0
    
    # Training parameters
    batch_size: int = 4
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # Multi-task strategy
    task_sampling: str = "balanced"  # "balanced", "proportional", "curriculum"
    curriculum_stages: List[str] = None  # ["ligand", "ppi", "mixed"]
    
    # Uncertainty-aware training
    use_uncertainty_weighting: bool = True
    use_md_confidence: bool = True
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_steps: int = 1000
    
    # Logging
    log_frequency: int = 100
    eval_frequency: int = 1000
    save_frequency: int = 5000
    use_wandb: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    
    def __post_init__(self):
        if self.curriculum_stages is None:
            self.curriculum_stages = ["ligand", "ppi", "mixed"]


class UnifiedPearlTrainer:
    """
    Unified trainer for protein-ligand and protein-protein tasks.
    
    Implements multi-task learning with:
    - Task-specific losses
    - Balanced sampling
    - Curriculum learning
    - Uncertainty-aware training
    """
    
    def __init__(
        self,
        model: Pearl,
        config: UnifiedTrainingConfig,
        ligand_dataloader: DataLoader,
        ppi_dataloader: DataLoader,
        val_ligand_dataloader: Optional[DataLoader] = None,
        val_ppi_dataloader: Optional[DataLoader] = None
    ):
        """
        Initialize unified trainer.
        
        Args:
            model: Pearl model
            config: Training configuration
            ligand_dataloader: DataLoader for protein-ligand data
            ppi_dataloader: DataLoader for protein-protein data
            val_ligand_dataloader: Validation DataLoader for ligand
            val_ppi_dataloader: Validation DataLoader for PPI
        """
        self.model = model.to(config.device)
        self.config = config
        
        self.ligand_dataloader = ligand_dataloader
        self.ppi_dataloader = ppi_dataloader
        self.val_ligand_dataloader = val_ligand_dataloader
        self.val_ppi_dataloader = val_ppi_dataloader
        
        # Setup losses
        self.ligand_loss_fn = CombinedUncertaintyAwareLoss() if config.use_uncertainty_weighting else DiffusionLoss()
        self.ppi_loss_fn = CombinedPPILoss()
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Metrics
        self.ppi_metrics = PPIMetrics()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.current_stage = 0
        
        # Initialize wandb
        if config.use_wandb:
            wandb.init(
                project="pearl-unified",
                config=vars(config)
            )
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer"""
        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler"""
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config.num_epochs
            )
        else:
            return None
    
    def train_epoch(self, task_type: str = "mixed") -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            task_type: "ligand", "ppi", or "mixed"
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        epoch_metrics = {
            "ligand_loss": 0.0,
            "ppi_loss": 0.0,
            "total_loss": 0.0,
            "n_ligand_batches": 0,
            "n_ppi_batches": 0
        }
        
        # Create iterators
        ligand_iter = iter(self.ligand_dataloader) if task_type in ["ligand", "mixed"] else None
        ppi_iter = iter(self.ppi_dataloader) if task_type in ["ppi", "mixed"] else None
        
        # Determine number of steps
        if task_type == "ligand":
            n_steps = len(self.ligand_dataloader)
        elif task_type == "ppi":
            n_steps = len(self.ppi_dataloader)
        else:  # mixed
            n_steps = max(len(self.ligand_dataloader), len(self.ppi_dataloader))
        
        for step in range(n_steps):
            # Sample task
            if task_type == "mixed":
                if self.config.task_sampling == "balanced":
                    current_task = "ligand" if step % 2 == 0 else "ppi"
                elif self.config.task_sampling == "proportional":
                    # Sample proportional to dataset size
                    ligand_prob = len(self.ligand_dataloader) / (len(self.ligand_dataloader) + len(self.ppi_dataloader))
                    current_task = "ligand" if torch.rand(1).item() < ligand_prob else "ppi"
                else:
                    current_task = "ligand" if step % 2 == 0 else "ppi"
            else:
                current_task = task_type
            
            # Get batch
            try:
                if current_task == "ligand" and ligand_iter is not None:
                    batch = next(ligand_iter)
                    loss = self._train_ligand_step(batch)
                    epoch_metrics["ligand_loss"] += loss
                    epoch_metrics["n_ligand_batches"] += 1
                elif current_task == "ppi" and ppi_iter is not None:
                    batch = next(ppi_iter)
                    loss = self._train_ppi_step(batch)
                    epoch_metrics["ppi_loss"] += loss
                    epoch_metrics["n_ppi_batches"] += 1
                else:
                    continue
                
                epoch_metrics["total_loss"] += loss
                
                # Logging
                if self.global_step % self.config.log_frequency == 0:
                    self._log_metrics({
                        f"{current_task}_loss": loss,
                        "learning_rate": self.optimizer.param_groups[0]['lr'],
                        "epoch": self.current_epoch,
                        "step": self.global_step
                    })
                
                # Evaluation
                if self.global_step % self.config.eval_frequency == 0:
                    self.evaluate()
                
                # Save checkpoint
                if self.global_step % self.config.save_frequency == 0:
                    self.save_checkpoint()
                
                self.global_step += 1
                
            except StopIteration:
                # Restart iterator
                if current_task == "ligand":
                    ligand_iter = iter(self.ligand_dataloader)
                else:
                    ppi_iter = iter(self.ppi_dataloader)
        
        # Average metrics
        if epoch_metrics["n_ligand_batches"] > 0:
            epoch_metrics["ligand_loss"] /= epoch_metrics["n_ligand_batches"]
        if epoch_metrics["n_ppi_batches"] > 0:
            epoch_metrics["ppi_loss"] /= epoch_metrics["n_ppi_batches"]
        epoch_metrics["total_loss"] /= (epoch_metrics["n_ligand_batches"] + epoch_metrics["n_ppi_batches"])
        
        return epoch_metrics
    
    def _train_ligand_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Train step for protein-ligand task"""
        # Move batch to device
        batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
            # Model forward pass (simplified - actual implementation depends on Pearl's interface)
            pred_coords = self.model(
                protein_features=batch["protein_features"],
                ligand_features=batch["ligand_features"],
                timestep=batch["timestep"]
            )
            
            # Compute loss
            if self.config.use_uncertainty_weighting:
                loss_dict = self.ligand_loss_fn(
                    pred_coords,
                    batch["true_coords"],
                    batch.get("confidence", None)
                )
                loss = loss_dict["total"]
            else:
                loss = self.ligand_loss_fn(pred_coords, batch["true_coords"])
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()
        
        return loss.item()

    def _train_ppi_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Train step for protein-protein task"""
        # Move batch to device
        batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
            # Model forward pass for PPI
            pred_coords_a, pred_coords_b = self.model(
                protein_a_features=batch["protein_a_features"],
                protein_b_features=batch["protein_b_features"],
                timestep=batch["timestep"]
            )

            # Compute PPI loss
            loss_dict = self.ppi_loss_fn(
                pred_coords_a,
                pred_coords_b,
                batch["true_coords_a"],
                batch["true_coords_b"],
                batch["interface_mask_a"],
                batch["interface_mask_b"]
            )
            loss = loss_dict["total"] * self.config.ppi_task_weight

        # Backward pass
        self.optimizer.zero_grad()

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()

        return loss.item()

    def train(self):
        """
        Complete training loop with curriculum learning.
        """
        logger.info("Starting unified training...")
        logger.info(f"Curriculum stages: {self.config.curriculum_stages}")

        for stage_idx, stage in enumerate(self.config.curriculum_stages):
            self.current_stage = stage_idx
            logger.info(f"\n{'='*80}")
            logger.info(f"STAGE {stage_idx + 1}/{len(self.config.curriculum_stages)}: {stage.upper()}")
            logger.info(f"{'='*80}\n")

            # Determine epochs for this stage
            epochs_per_stage = self.config.num_epochs // len(self.config.curriculum_stages)

            for epoch in range(epochs_per_stage):
                self.current_epoch = stage_idx * epochs_per_stage + epoch

                logger.info(f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}")

                # Train epoch
                metrics = self.train_epoch(task_type=stage)

                # Log epoch metrics
                logger.info(f"  Ligand loss: {metrics['ligand_loss']:.4f}")
                logger.info(f"  PPI loss: {metrics['ppi_loss']:.4f}")
                logger.info(f"  Total loss: {metrics['total_loss']:.4f}")

                self._log_metrics({
                    "epoch_ligand_loss": metrics["ligand_loss"],
                    "epoch_ppi_loss": metrics["ppi_loss"],
                    "epoch_total_loss": metrics["total_loss"],
                    "epoch": self.current_epoch
                })

                # Step scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

        logger.info("\nTraining complete!")

        # Final evaluation
        self.evaluate()

        # Save final model
        self.save_checkpoint(final=True)

    def evaluate(self):
        """Evaluate on validation sets"""
        self.model.eval()

        logger.info("Running evaluation...")

        eval_metrics = {}

        # Evaluate ligand task
        if self.val_ligand_dataloader is not None:
            ligand_metrics = self._evaluate_ligand()
            eval_metrics.update({f"val_ligand_{k}": v for k, v in ligand_metrics.items()})

        # Evaluate PPI task
        if self.val_ppi_dataloader is not None:
            ppi_metrics = self._evaluate_ppi()
            eval_metrics.update({f"val_ppi_{k}": v for k, v in ppi_metrics.items()})

        # Log metrics
        self._log_metrics(eval_metrics)

        logger.info("Evaluation complete")
        for k, v in eval_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        self.model.train()

    def _evaluate_ligand(self) -> Dict[str, float]:
        """Evaluate protein-ligand task"""
        total_rmsd = 0.0
        total_lddt = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch in self.val_ligand_dataloader:
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                pred_coords = self.model(
                    protein_features=batch["protein_features"],
                    ligand_features=batch["ligand_features"],
                    timestep=torch.zeros(batch["protein_features"].shape[0], device=self.config.device)
                )

                # Compute metrics
                rmsd = compute_rmsd(pred_coords, batch["true_coords"])
                lddt = compute_lddt(pred_coords, batch["true_coords"])

                total_rmsd += rmsd * batch["protein_features"].shape[0]
                total_lddt += lddt * batch["protein_features"].shape[0]
                n_samples += batch["protein_features"].shape[0]

        return {
            "rmsd": total_rmsd / n_samples,
            "lddt": total_lddt / n_samples
        }

    def _evaluate_ppi(self) -> Dict[str, float]:
        """Evaluate protein-protein task"""
        total_dockq = 0.0
        total_irmsd = 0.0
        total_fnat = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch in self.val_ppi_dataloader:
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                pred_coords_a, pred_coords_b = self.model(
                    protein_a_features=batch["protein_a_features"],
                    protein_b_features=batch["protein_b_features"],
                    timestep=torch.zeros(batch["protein_a_features"].shape[0], device=self.config.device)
                )

                # Compute metrics (convert to numpy)
                pred_a_np = pred_coords_a.cpu().numpy()
                pred_b_np = pred_coords_b.cpu().numpy()
                true_a_np = batch["true_coords_a"].cpu().numpy()
                true_b_np = batch["true_coords_b"].cpu().numpy()

                for i in range(pred_a_np.shape[0]):
                    results = self.ppi_metrics.compute_dockq(
                        pred_a_np[i], pred_b_np[i],
                        true_a_np[i], true_b_np[i]
                    )

                    total_dockq += results["dockq"]
                    total_irmsd += results["irmsd"]
                    total_fnat += results["fnat"]
                    n_samples += 1

        return {
            "dockq": total_dockq / n_samples,
            "irmsd": total_irmsd / n_samples,
            "fnat": total_fnat / n_samples
        }

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to wandb"""
        if self.config.use_wandb:
            wandb.log(metrics, step=self.global_step)

    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        if final:
            checkpoint_path = os.path.join(checkpoint_dir, "final_model.pt")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{self.global_step}.pt")

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "config": self.config
        }, checkpoint_path)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from step {self.global_step}, epoch {self.current_epoch}")
