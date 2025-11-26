"""
Differentiable Density Generator for Pearl

Convert predicted atomic coordinates to density maps for comparison
with experimental density. This is fully differentiable, allowing
gradients to flow from density comparison back to coordinates.

Key features:
- Gaussian atom representation
- Element-specific scattering factors
- B-factor modeling
- GPU-accelerated
- Fully differentiable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


# Atomic scattering factors (electrons) for common elements
# These are approximate values for CryoEM (Coulomb potential)
SCATTERING_FACTORS = {
    'H': 1.0,
    'C': 6.0,
    'N': 7.0,
    'O': 8.0,
    'S': 16.0,
    'P': 15.0,
    'F': 9.0,
    'Cl': 17.0,
    'Br': 35.0,
    'I': 53.0,
    'Ca': 20.0,
    'Mg': 12.0,
    'Zn': 30.0,
    'Fe': 26.0,
    'Cu': 29.0,
    'Mn': 25.0,
}


class DifferentiableDensityGenerator(nn.Module):
    """
    Generate density maps from atomic coordinates.
    
    Each atom is represented as a 3D Gaussian:
    ρ(r) = Σ_i A_i * exp(-B_i * |r - r_i|² / 4)
    
    where:
    - r_i: atom position
    - A_i: scattering amplitude (element-dependent)
    - B_i: B-factor (thermal motion)
    """
    
    def __init__(
        self,
        grid_size: int = 64,           # Grid size (voxels per dimension)
        voxel_size: float = 1.0,       # Angstroms per voxel
        default_bfactor: float = 20.0, # Default B-factor
        sigma_cutoff: float = 3.0,     # Cutoff in units of sigma
    ):
        super().__init__()
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.default_bfactor = default_bfactor
        self.sigma_cutoff = sigma_cutoff
        
        # Learnable scattering factors (initialized from physical values)
        self.register_buffer(
            'scattering_factors',
            torch.tensor([SCATTERING_FACTORS.get(elem, 6.0) 
                         for elem in ['H', 'C', 'N', 'O', 'S', 'P']])
        )
    
    def forward(
        self,
        coords: torch.Tensor,           # [batch, n_atoms, 3] in Angstroms
        atom_types: torch.Tensor,       # [batch, n_atoms] element indices
        bfactors: Optional[torch.Tensor] = None,  # [batch, n_atoms]
        box_center: Optional[torch.Tensor] = None, # [batch, 3]
    ) -> torch.Tensor:
        """
        Generate density map from coordinates.
        
        Args:
            coords: Atomic coordinates [batch, n_atoms, 3]
            atom_types: Atom type indices [batch, n_atoms]
            bfactors: B-factors [batch, n_atoms] (optional)
            box_center: Center of density box [batch, 3] (optional)
            
        Returns:
            Density map [batch, grid_size, grid_size, grid_size]
        """
        batch_size, n_atoms, _ = coords.shape
        device = coords.device
        
        # Default B-factors if not provided
        if bfactors is None:
            bfactors = torch.full(
                (batch_size, n_atoms),
                self.default_bfactor,
                device=device
            )
        
        # Default box center (center of mass)
        if box_center is None:
            box_center = coords.mean(dim=1)  # [batch, 3]
        
        # Create grid
        grid = self._create_grid(batch_size, box_center, device)  # [batch, grid³, 3]
        
        # Initialize density map
        density = torch.zeros(
            batch_size, self.grid_size, self.grid_size, self.grid_size,
            device=device
        )
        
        # Add contribution from each atom
        for atom_idx in range(n_atoms):
            atom_pos = coords[:, atom_idx, :]  # [batch, 3]
            atom_type = atom_types[:, atom_idx]  # [batch]
            atom_bfactor = bfactors[:, atom_idx]  # [batch]
            
            # Get scattering amplitude
            amplitude = self.scattering_factors[atom_type]  # [batch]
            
            # Compute Gaussian contribution
            gaussian = self._compute_gaussian(
                grid, atom_pos, amplitude, atom_bfactor
            )  # [batch, grid³]
            
            # Reshape and add to density
            gaussian = gaussian.view(
                batch_size, self.grid_size, self.grid_size, self.grid_size
            )
            density = density + gaussian
        
        return density
    
    def _create_grid(
        self,
        batch_size: int,
        box_center: torch.Tensor,  # [batch, 3]
        device: torch.device
    ) -> torch.Tensor:
        """
        Create 3D grid of positions.
        
        Returns:
            Grid positions [batch, grid³, 3]
        """
        # Create 1D grid
        grid_1d = torch.arange(
            self.grid_size, dtype=torch.float32, device=device
        )
        
        # Center grid around 0
        grid_1d = (grid_1d - self.grid_size / 2) * self.voxel_size
        
        # Create 3D meshgrid
        grid_x, grid_y, grid_z = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
        
        # Stack into [grid³, 3]
        grid = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1)
        
        # Expand for batch and add box center
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, grid³, 3]
        grid = grid + box_center.unsqueeze(1)  # Add box center
        
        return grid
    
    def _compute_gaussian(
        self,
        grid: torch.Tensor,        # [batch, grid³, 3]
        atom_pos: torch.Tensor,    # [batch, 3]
        amplitude: torch.Tensor,   # [batch]
        bfactor: torch.Tensor,     # [batch]
    ) -> torch.Tensor:
        """
        Compute Gaussian density for one atom.
        
        Gaussian: A * exp(-B * r² / 4)
        
        Returns:
            Density values [batch, grid³]
        """
        # Compute squared distance from atom
        # grid: [batch, grid³, 3]
        # atom_pos: [batch, 3] -> [batch, 1, 3]
        diff = grid - atom_pos.unsqueeze(1)  # [batch, grid³, 3]
        r_squared = (diff ** 2).sum(dim=-1)  # [batch, grid³]
        
        # Compute Gaussian
        # exp(-B * r² / 4)
        exponent = -bfactor.unsqueeze(1) * r_squared / 4.0  # [batch, grid³]
        gaussian = amplitude.unsqueeze(1) * torch.exp(exponent)  # [batch, grid³]
        
        return gaussian
    
    def forward_fast(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        bfactors: Optional[torch.Tensor] = None,
        box_center: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fast version using sparse computation.
        
        Only compute Gaussian within cutoff distance (3 sigma).
        This is much faster for large proteins.
        """
        batch_size, n_atoms, _ = coords.shape
        device = coords.device
        
        if bfactors is None:
            bfactors = torch.full(
                (batch_size, n_atoms),
                self.default_bfactor,
                device=device
            )
        
        if box_center is None:
            box_center = coords.mean(dim=1)
        
        # Initialize density
        density = torch.zeros(
            batch_size, self.grid_size, self.grid_size, self.grid_size,
            device=device
        )
        
        # For each atom, only update voxels within cutoff
        for b in range(batch_size):
            for atom_idx in range(n_atoms):
                atom_pos = coords[b, atom_idx]
                atom_type = atom_types[b, atom_idx]
                atom_bfactor = bfactors[b, atom_idx]
                
                # Compute sigma from B-factor
                # B = 8π²σ², so σ = sqrt(B / 8π²)
                sigma = torch.sqrt(atom_bfactor / (8 * np.pi ** 2))
                cutoff = self.sigma_cutoff * sigma
                
                # Find voxels within cutoff
                # Convert atom position to voxel coordinates
                voxel_pos = (atom_pos - box_center[b]) / self.voxel_size + self.grid_size / 2
                
                # Get bounding box
                voxel_min = torch.floor(voxel_pos - cutoff / self.voxel_size).long()
                voxel_max = torch.ceil(voxel_pos + cutoff / self.voxel_size).long()
                
                # Clip to grid bounds
                voxel_min = torch.clamp(voxel_min, 0, self.grid_size - 1)
                voxel_max = torch.clamp(voxel_max, 0, self.grid_size - 1)
                
                # Update voxels in bounding box
                for i in range(voxel_min[0], voxel_max[0] + 1):
                    for j in range(voxel_min[1], voxel_max[1] + 1):
                        for k in range(voxel_min[2], voxel_max[2] + 1):
                            # Compute voxel position in Angstroms
                            voxel_pos_ang = (
                                torch.tensor([i, j, k], dtype=torch.float32, device=device) 
                                - self.grid_size / 2
                            ) * self.voxel_size + box_center[b]
                            
                            # Compute distance
                            r_squared = ((voxel_pos_ang - atom_pos) ** 2).sum()
                            
                            # Compute Gaussian
                            amplitude = self.scattering_factors[atom_type]
                            gaussian_value = amplitude * torch.exp(-atom_bfactor * r_squared / 4.0)
                            
                            # Add to density
                            density[b, i, j, k] += gaussian_value
        
        return density


class DensityCorrelation(nn.Module):
    """
    Compute correlation between predicted and experimental density.
    
    Supports:
    - Real-space correlation coefficient
    - Fourier shell correlation
    - Local correlation (resolution-weighted)
    """
    
    def __init__(self, correlation_type: str = 'real_space'):
        super().__init__()
        self.correlation_type = correlation_type
    
    def forward(
        self,
        pred_density: torch.Tensor,  # [batch, nx, ny, nz]
        exp_density: torch.Tensor,   # [batch, nx, ny, nz]
        mask: Optional[torch.Tensor] = None,  # [batch, nx, ny, nz]
    ) -> torch.Tensor:
        """
        Compute correlation coefficient.
        
        Returns:
            Correlation coefficient [batch] (range: [-1, 1])
        """
        if self.correlation_type == 'real_space':
            return self._real_space_correlation(pred_density, exp_density, mask)
        elif self.correlation_type == 'fourier':
            return self._fourier_correlation(pred_density, exp_density)
        else:
            raise ValueError(f"Unknown correlation type: {self.correlation_type}")
    
    def _real_space_correlation(
        self,
        pred: torch.Tensor,
        exp: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Real-space correlation coefficient.
        
        CC = Σ(pred - pred_mean)(exp - exp_mean) / 
             sqrt(Σ(pred - pred_mean)² * Σ(exp - exp_mean)²)
        """
        if mask is not None:
            pred = pred * mask
            exp = exp * mask
        
        # Flatten spatial dimensions
        pred_flat = pred.flatten(start_dim=1)  # [batch, n_voxels]
        exp_flat = exp.flatten(start_dim=1)
        
        # Compute means
        pred_mean = pred_flat.mean(dim=1, keepdim=True)
        exp_mean = exp_flat.mean(dim=1, keepdim=True)
        
        # Center
        pred_centered = pred_flat - pred_mean
        exp_centered = exp_flat - exp_mean
        
        # Compute correlation
        numerator = (pred_centered * exp_centered).sum(dim=1)
        denominator = torch.sqrt(
            (pred_centered ** 2).sum(dim=1) * (exp_centered ** 2).sum(dim=1)
        )
        
        cc = numerator / (denominator + 1e-8)
        
        return cc
    
    def _fourier_correlation(
        self,
        pred: torch.Tensor,
        exp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fourier shell correlation (FSC).
        
        Average correlation across all Fourier shells.
        """
        # FFT
        pred_fft = torch.fft.fftn(pred, dim=(-3, -2, -1))
        exp_fft = torch.fft.fftn(exp, dim=(-3, -2, -1))
        
        # Compute correlation in Fourier space
        numerator = (pred_fft * torch.conj(exp_fft)).real.sum(dim=(-3, -2, -1))
        denominator = torch.sqrt(
            (torch.abs(pred_fft) ** 2).sum(dim=(-3, -2, -1)) *
            (torch.abs(exp_fft) ** 2).sum(dim=(-3, -2, -1))
        )
        
        fsc = numerator / (denominator + 1e-8)
        
        return fsc

