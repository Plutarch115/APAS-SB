"""
Density-Aware Loss Functions for Pearl

Loss functions that compare predicted density with experimental density.
This enables training Pearl to place atoms to match experimental data
directly, rather than just matching fitted coordinates.

Key features:
- Real-space correlation loss
- Fourier shell correlation loss
- Local density agreement (resolution-weighted)
- Hybrid coordinate + density loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from pearl.models.density_generator import DifferentiableDensityGenerator, DensityCorrelation


class RealSpaceCorrelationLoss(nn.Module):
    """
    Real-space correlation coefficient loss.
    
    Measures how well predicted density matches experimental density
    in real space (voxel-by-voxel comparison).
    
    Loss = 1 - CC
    where CC is correlation coefficient (range: [-1, 1])
    """
    
    def __init__(self, use_mask: bool = True):
        super().__init__()
        self.use_mask = use_mask
        self.correlation = DensityCorrelation(correlation_type='real_space')
    
    def forward(
        self,
        pred_density: torch.Tensor,  # [batch, nx, ny, nz]
        exp_density: torch.Tensor,   # [batch, nx, ny, nz]
        mask: Optional[torch.Tensor] = None,  # [batch, nx, ny, nz]
    ) -> torch.Tensor:
        """
        Compute real-space correlation loss.
        
        Returns:
            Loss value (scalar)
        """
        # Compute correlation coefficient
        cc = self.correlation(pred_density, exp_density, mask)
        
        # Loss = 1 - CC (minimize)
        # CC = 1 → loss = 0 (perfect match)
        # CC = 0 → loss = 1 (no correlation)
        # CC = -1 → loss = 2 (anti-correlated)
        loss = 1.0 - cc.mean()
        
        return loss


class FourierShellCorrelationLoss(nn.Module):
    """
    Fourier shell correlation (FSC) loss.
    
    Measures correlation in Fourier space, which is more sensitive
    to high-frequency features (fine details).
    """
    
    def __init__(self):
        super().__init__()
        self.correlation = DensityCorrelation(correlation_type='fourier')
    
    def forward(
        self,
        pred_density: torch.Tensor,
        exp_density: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute FSC loss.
        
        Returns:
            Loss value (scalar)
        """
        fsc = self.correlation(pred_density, exp_density)
        loss = 1.0 - fsc.mean()
        
        return loss


class LocalDensityLoss(nn.Module):
    """
    Local density agreement loss with resolution weighting.
    
    Computes MSE between predicted and experimental density,
    weighted by local resolution. Better resolution regions
    get higher weight.
    """
    
    def __init__(self, resolution_weighting: bool = True):
        super().__init__()
        self.resolution_weighting = resolution_weighting
    
    def forward(
        self,
        pred_density: torch.Tensor,      # [batch, nx, ny, nz]
        exp_density: torch.Tensor,       # [batch, nx, ny, nz]
        local_resolution: Optional[torch.Tensor] = None,  # [batch, nx, ny, nz]
    ) -> torch.Tensor:
        """
        Compute local density loss.
        
        Returns:
            Loss value (scalar)
        """
        # MSE loss
        mse = (pred_density - exp_density) ** 2
        
        if self.resolution_weighting and local_resolution is not None:
            # Convert local resolution to weights
            # Better resolution (lower value) → higher weight
            # weight = 1 / (resolution + 1)
            weights = 1.0 / (local_resolution + 1.0)
            
            # Normalize weights
            weights = weights / weights.mean()
            
            # Weighted MSE
            weighted_mse = (mse * weights).mean()
            
            return weighted_mse
        else:
            return mse.mean()


class DensityAwareLoss(nn.Module):
    """
    Combined density-aware loss function.
    
    Combines:
    1. Coordinate loss (for comparison with original Pearl)
    2. Real-space correlation loss
    3. Fourier shell correlation loss (optional)
    4. Local density loss (resolution-weighted)
    
    This is the main loss function for density-aware training.
    """
    
    def __init__(
        self,
        coord_weight: float = 0.3,           # Weight for coordinate loss
        density_weight: float = 0.7,         # Weight for density losses
        use_fourier: bool = True,            # Use FSC loss
        use_local_weighting: bool = True,    # Use resolution weighting
        grid_size: int = 64,                 # Density grid size
        voxel_size: float = 1.0,             # Voxel size (Å)
    ):
        super().__init__()
        self.coord_weight = coord_weight
        self.density_weight = density_weight
        self.use_fourier = use_fourier
        self.use_local_weighting = use_local_weighting
        
        # Density generator
        self.density_generator = DifferentiableDensityGenerator(
            grid_size=grid_size,
            voxel_size=voxel_size
        )
        
        # Loss components
        self.rsc_loss = RealSpaceCorrelationLoss()
        self.fsc_loss = FourierShellCorrelationLoss() if use_fourier else None
        self.local_loss = LocalDensityLoss(resolution_weighting=use_local_weighting)
    
    def forward(
        self,
        pred_coords: torch.Tensor,           # [batch, n_atoms, 3]
        true_coords: torch.Tensor,           # [batch, n_atoms, 3]
        atom_types: torch.Tensor,            # [batch, n_atoms]
        exp_density: torch.Tensor,           # [batch, nx, ny, nz]
        local_resolution: Optional[torch.Tensor] = None,  # [batch, nx, ny, nz]
        bfactors: Optional[torch.Tensor] = None,  # [batch, n_atoms]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined density-aware loss.
        
        Args:
            pred_coords: Predicted atomic coordinates
            true_coords: True atomic coordinates (for comparison)
            atom_types: Atom type indices
            exp_density: Experimental density map
            local_resolution: Local resolution map (optional)
            bfactors: B-factors (optional)
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # 1. Coordinate loss (RMSD)
        coord_loss = torch.sqrt(((pred_coords - true_coords) ** 2).sum(dim=-1).mean())
        losses['coord_rmsd'] = coord_loss
        
        # 2. Generate predicted density from coordinates
        pred_density = self.density_generator(
            coords=pred_coords,
            atom_types=atom_types,
            bfactors=bfactors
        )
        
        # 3. Real-space correlation loss
        rsc_loss = self.rsc_loss(pred_density, exp_density)
        losses['rsc_loss'] = rsc_loss
        
        # 4. Fourier shell correlation loss (optional)
        if self.use_fourier and self.fsc_loss is not None:
            fsc_loss = self.fsc_loss(pred_density, exp_density)
            losses['fsc_loss'] = fsc_loss
        
        # 5. Local density loss (resolution-weighted)
        if self.use_local_weighting and local_resolution is not None:
            local_loss = self.local_loss(pred_density, exp_density, local_resolution)
            losses['local_loss'] = local_loss
        else:
            local_loss = self.local_loss(pred_density, exp_density)
            losses['local_loss'] = local_loss
        
        # 6. Combined loss
        density_loss_components = [rsc_loss, local_loss]
        if self.use_fourier and self.fsc_loss is not None:
            density_loss_components.append(fsc_loss)
        
        density_loss = sum(density_loss_components) / len(density_loss_components)
        
        total_loss = (
            self.coord_weight * coord_loss +
            self.density_weight * density_loss
        )
        
        losses['density_loss'] = density_loss
        losses['total_loss'] = total_loss
        
        return losses


class HybridDensityCoordinateLoss(nn.Module):
    """
    Hybrid loss that adaptively weights density vs coordinate loss
    based on resolution.
    
    High resolution (< 2Å): Use mostly coordinate loss
    Medium resolution (2-4Å): Use balanced loss
    Low resolution (> 4Å): Use mostly density loss
    """
    
    def __init__(
        self,
        high_res_threshold: float = 2.0,   # Å
        low_res_threshold: float = 4.0,    # Å
        grid_size: int = 64,
        voxel_size: float = 1.0,
    ):
        super().__init__()
        self.high_res_threshold = high_res_threshold
        self.low_res_threshold = low_res_threshold
        
        self.density_generator = DifferentiableDensityGenerator(
            grid_size=grid_size,
            voxel_size=voxel_size
        )
        
        self.rsc_loss = RealSpaceCorrelationLoss()
        self.local_loss = LocalDensityLoss()
    
    def forward(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        atom_types: torch.Tensor,
        exp_density: torch.Tensor,
        resolution: torch.Tensor,  # [batch] overall resolution
        local_resolution: Optional[torch.Tensor] = None,
        bfactors: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hybrid loss with adaptive weighting.
        
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Coordinate loss
        coord_loss = torch.sqrt(((pred_coords - true_coords) ** 2).sum(dim=-1).mean())
        losses['coord_rmsd'] = coord_loss
        
        # Generate predicted density
        pred_density = self.density_generator(
            coords=pred_coords,
            atom_types=atom_types,
            bfactors=bfactors
        )
        
        # Density losses
        rsc_loss = self.rsc_loss(pred_density, exp_density)
        local_loss = self.local_loss(pred_density, exp_density, local_resolution)
        density_loss = (rsc_loss + local_loss) / 2.0
        
        losses['rsc_loss'] = rsc_loss
        losses['local_loss'] = local_loss
        losses['density_loss'] = density_loss
        
        # Adaptive weighting based on resolution
        # High res (< 2Å): coord_weight = 0.8, density_weight = 0.2
        # Medium res (2-4Å): coord_weight = 0.5, density_weight = 0.5
        # Low res (> 4Å): coord_weight = 0.2, density_weight = 0.8
        
        coord_weight = torch.zeros_like(resolution)
        density_weight = torch.zeros_like(resolution)
        
        # High resolution
        high_res_mask = resolution < self.high_res_threshold
        coord_weight[high_res_mask] = 0.8
        density_weight[high_res_mask] = 0.2
        
        # Low resolution
        low_res_mask = resolution > self.low_res_threshold
        coord_weight[low_res_mask] = 0.2
        density_weight[low_res_mask] = 0.8
        
        # Medium resolution (linear interpolation)
        medium_res_mask = ~high_res_mask & ~low_res_mask
        if medium_res_mask.any():
            # Linear interpolation between thresholds
            alpha = (resolution[medium_res_mask] - self.high_res_threshold) / \
                    (self.low_res_threshold - self.high_res_threshold)
            coord_weight[medium_res_mask] = 0.8 - 0.6 * alpha
            density_weight[medium_res_mask] = 0.2 + 0.6 * alpha
        
        # Combined loss (per-sample weighting)
        total_loss = (
            coord_weight * coord_loss +
            density_weight * density_loss
        ).mean()
        
        losses['total_loss'] = total_loss
        losses['coord_weight'] = coord_weight.mean()
        losses['density_weight'] = density_weight.mean()
        
        return losses

