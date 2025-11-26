"""
Training Loop for Pearl

Implements the training procedure including:
- Five-stage curriculum training
- Mixed precision (bfloat16/fp32) training
- Gradient accumulation
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, List
import time

from ..models.pearl import Pearl
from .losses import PearlLoss


class PearlTrainer:
    """
    Trainer for Pearl model with curriculum learning.
    """
    
    def __init__(
        self,
        model: Pearl,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_mixed_precision = use_mixed_precision
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Loss function
        self.criterion = PearlLoss(
            diffusion_weight=1.0,
            distance_weight=0.1,
            bond_weight=0.05
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_mixed_precision else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
    def train_step(
        self,
        protein_features: torch.Tensor,
        ligand_features: torch.Tensor,
        true_coords: torch.Tensor,
        bond_indices: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            protein_features: Protein features [batch, n_protein, feat_dim]
            ligand_features: Ligand features [batch, n_ligand, feat_dim]
            true_coords: Ground truth coordinates [batch, n_atoms, 3]
            bond_indices: Optional bond connectivity
            mask: Optional atom mask
            
        Returns:
            Dictionary of loss values
        """
        batch_size = protein_features.shape[0]
        n_atoms = true_coords.shape[1]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.model.diffusion.num_diffusion_steps,
            (batch_size,), device=self.device
        )
        
        # Add noise to coordinates
        noise = torch.randn_like(true_coords)
        noisy_coords, _ = self.model.diffusion.add_noise(true_coords, timesteps, noise)
        
        # Forward pass with mixed precision
        if self.use_mixed_precision:
            with autocast(dtype=torch.bfloat16):
                predicted_noise = self.model(
                    protein_features, ligand_features,
                    noisy_coords, timesteps,
                    templates=None,
                    protein_mask=mask[:, :protein_features.shape[1]] if mask is not None else None,
                    ligand_mask=mask[:, protein_features.shape[1]:] if mask is not None else None
                )
                
                # Compute losses (in fp32 for stability)
                with autocast(enabled=False):
                    losses = self.criterion(
                        predicted_noise.float(),
                        noise.float(),
                        noisy_coords.float(),
                        bond_indices,
                        mask
                    )
        else:
            predicted_noise = self.model(
                protein_features, ligand_features,
                noisy_coords, timesteps,
                templates=None,
                protein_mask=mask[:, :protein_features.shape[1]] if mask is not None else None,
                ligand_mask=mask[:, protein_features.shape[1]:] if mask is not None else None
            )
            
            losses = self.criterion(
                predicted_noise,
                noise,
                noisy_coords,
                bond_indices,
                mask
            )
        
        # Backward pass
        loss = losses['total'] / self.gradient_accumulation_steps
        
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights (if accumulation steps reached)
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            if self.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
        
        self.global_step += 1
        
        # Return losses as floats
        return {k: v.item() for k, v in losses.items()}
    
    def train_epoch(
        self,
        dataloader,
        log_interval: int = 100
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            log_interval: Steps between logging
            
        Returns:
            Average losses for the epoch
        """
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'diffusion': 0.0,
            'distance': 0.0,
            'bond': 0.0
        }
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            protein_features = batch['protein_features'].to(self.device)
            ligand_features = batch['ligand_features'].to(self.device)
            true_coords = batch['coordinates'].to(self.device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(self.device)
            bond_indices = batch.get('bond_indices', None)
            
            # Training step
            losses = self.train_step(
                protein_features, ligand_features,
                true_coords, bond_indices, mask
            )
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
            num_batches += 1
            
            # Logging
            if (batch_idx + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = log_interval / elapsed
                
                print(f"Epoch {self.epoch} | Step {batch_idx + 1}/{len(dataloader)} | "
                      f"Loss: {losses['total']:.4f} | "
                      f"Diffusion: {losses['diffusion']:.4f} | "
                      f"Distance: {losses['distance']:.4f} | "
                      f"Bond: {losses['bond']:.4f} | "
                      f"Speed: {steps_per_sec:.2f} steps/s")
                
                start_time = time.time()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        self.epoch += 1
        
        return epoch_losses
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Checkpoint loaded from {path}")
        print(f"Resuming from epoch {self.epoch}, step {self.global_step}")


class CurriculumScheduler:
    """
    Implements Pearl's five-stage curriculum training.
    
    Progressively increases task complexity and data diversity.
    """
    
    def __init__(self):
        self.stages = [
            {
                'name': 'Stage 1: Small crops, PDB only',
                'max_atoms': 100,
                'use_templates': False,
                'use_synthetic': False,
                'epochs': 10
            },
            {
                'name': 'Stage 2: Medium crops, PDB + distillation',
                'max_atoms': 200,
                'use_templates': False,
                'use_synthetic': False,
                'epochs': 10
            },
            {
                'name': 'Stage 3: Large crops, add templates',
                'max_atoms': 500,
                'use_templates': True,
                'use_synthetic': False,
                'epochs': 15
            },
            {
                'name': 'Stage 4: Full size, add synthetic data',
                'max_atoms': 1000,
                'use_templates': True,
                'use_synthetic': True,
                'epochs': 20
            },
            {
                'name': 'Stage 5: Full training',
                'max_atoms': None,  # No limit
                'use_templates': True,
                'use_synthetic': True,
                'epochs': 30
            }
        ]
        
        self.current_stage = 0
    
    def get_current_stage(self) -> Dict:
        """Get current training stage configuration."""
        return self.stages[self.current_stage]
    
    def advance_stage(self):
        """Move to next training stage."""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            print(f"\n{'='*60}")
            print(f"Advancing to {self.stages[self.current_stage]['name']}")
            print(f"{'='*60}\n")
    
    def is_final_stage(self) -> bool:
        """Check if in final training stage."""
        return self.current_stage == len(self.stages) - 1

