"""
Loss Functions for Pearl Training

Implements the loss functions used in Pearl's training:
- Coordinate prediction loss (MSE on predicted noise)
- Auxiliary losses for physical validity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class DiffusionLoss(nn.Module):
    """
    Main diffusion loss for coordinate prediction.
    
    Predicts the noise added to coordinates at each diffusion timestep.
    """
    
    def __init__(self, loss_type: str = 'mse'):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(
        self,
        predicted_noise: torch.Tensor,
        true_noise: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute diffusion loss.
        
        Args:
            predicted_noise: Predicted noise [batch, n_atoms, 3]
            true_noise: True noise that was added [batch, n_atoms, 3]
            mask: Optional mask for valid atoms [batch, n_atoms]
            
        Returns:
            Loss value
        """
        if self.loss_type == 'mse':
            loss = F.mse_loss(predicted_noise, true_noise, reduction='none')
        elif self.loss_type == 'l1':
            loss = F.l1_loss(predicted_noise, true_noise, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Sum over coordinates
        loss = loss.sum(dim=-1)  # [batch, n_atoms]
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss


class DistanceConstraintLoss(nn.Module):
    """
    Auxiliary loss to encourage physically valid distances.
    
    Penalizes atom pairs that are too close (steric clashes) or
    unreasonably far apart.
    """
    
    def __init__(
        self,
        min_distance: float = 1.0,  # Minimum allowed distance (Å)
        max_distance: float = 50.0,  # Maximum reasonable distance (Å)
        clash_weight: float = 10.0,
        distance_weight: float = 1.0
    ):
        super().__init__()
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.clash_weight = clash_weight
        self.distance_weight = distance_weight
        
    def forward(
        self,
        coordinates: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute distance constraint loss.
        
        Args:
            coordinates: Predicted coordinates [batch, n_atoms, 3]
            mask: Optional mask for valid atoms [batch, n_atoms]
            
        Returns:
            Loss value
        """
        batch_size, n_atoms, _ = coordinates.shape
        
        # Compute pairwise distances
        diff = coordinates.unsqueeze(2) - coordinates.unsqueeze(1)  # [batch, n_atoms, n_atoms, 3]
        distances = torch.norm(diff, dim=-1)  # [batch, n_atoms, n_atoms]
        
        # Mask diagonal (self-distances)
        eye_mask = ~torch.eye(n_atoms, dtype=torch.bool, device=coordinates.device)
        eye_mask = eye_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply atom mask if provided
        if mask is not None:
            pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)  # [batch, n_atoms, n_atoms]
            eye_mask = eye_mask & pair_mask.bool()
        
        # Clash loss (penalize distances below minimum)
        clash_violations = F.relu(self.min_distance - distances)
        clash_loss = (clash_violations * eye_mask.float()).sum() / (eye_mask.sum() + 1e-8)
        
        # Distance loss (penalize unreasonably large distances)
        distance_violations = F.relu(distances - self.max_distance)
        distance_loss = (distance_violations * eye_mask.float()).sum() / (eye_mask.sum() + 1e-8)
        
        total_loss = (
            self.clash_weight * clash_loss +
            self.distance_weight * distance_loss
        )
        
        return total_loss


class BondLengthLoss(nn.Module):
    """
    Auxiliary loss to maintain reasonable bond lengths.
    
    Encourages predicted structures to have chemically valid bond lengths.
    """
    
    def __init__(
        self,
        target_bond_length: float = 1.5,  # Target bond length (Å)
        tolerance: float = 0.3  # Tolerance around target
    ):
        super().__init__()
        self.target_bond_length = target_bond_length
        self.tolerance = tolerance
        
    def forward(
        self,
        coordinates: torch.Tensor,
        bond_indices: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute bond length loss.
        
        Args:
            coordinates: Predicted coordinates [batch, n_atoms, 3]
            bond_indices: Bond connectivity [n_bonds, 2]
            mask: Optional mask for valid bonds [n_bonds]
            
        Returns:
            Loss value
        """
        if bond_indices.shape[0] == 0:
            return torch.tensor(0.0, device=coordinates.device)
        
        # Get coordinates of bonded atoms
        atom1_coords = coordinates[:, bond_indices[:, 0]]  # [batch, n_bonds, 3]
        atom2_coords = coordinates[:, bond_indices[:, 1]]
        
        # Compute bond lengths
        bond_vectors = atom2_coords - atom1_coords
        bond_lengths = torch.norm(bond_vectors, dim=-1)  # [batch, n_bonds]
        
        # Compute deviation from target
        deviation = torch.abs(bond_lengths - self.target_bond_length)
        
        # Only penalize if outside tolerance
        violation = F.relu(deviation - self.tolerance)
        
        # Apply mask if provided
        if mask is not None:
            violation = violation * mask.unsqueeze(0)
            loss = violation.sum() / (mask.sum() + 1e-8)
        else:
            loss = violation.mean()
        
        return loss


class PearlLoss(nn.Module):
    """
    Combined loss function for Pearl training.
    
    Combines diffusion loss with auxiliary losses for physical validity.
    """
    
    def __init__(
        self,
        diffusion_weight: float = 1.0,
        distance_weight: float = 0.1,
        bond_weight: float = 0.05
    ):
        super().__init__()
        
        self.diffusion_weight = diffusion_weight
        self.distance_weight = distance_weight
        self.bond_weight = bond_weight
        
        self.diffusion_loss = DiffusionLoss(loss_type='mse')
        self.distance_loss = DistanceConstraintLoss()
        self.bond_loss = BondLengthLoss()
        
    def forward(
        self,
        predicted_noise: torch.Tensor,
        true_noise: torch.Tensor,
        predicted_coords: torch.Tensor,
        bond_indices: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            predicted_noise: Predicted noise [batch, n_atoms, 3]
            true_noise: True noise [batch, n_atoms, 3]
            predicted_coords: Predicted coordinates [batch, n_atoms, 3]
            bond_indices: Optional bond connectivity [n_bonds, 2]
            mask: Optional mask for valid atoms [batch, n_atoms]
            
        Returns:
            Dictionary of loss components and total loss
        """
        losses = {}
        
        # Main diffusion loss
        diff_loss = self.diffusion_loss(predicted_noise, true_noise, mask)
        losses['diffusion'] = diff_loss
        
        # Distance constraint loss
        dist_loss = self.distance_loss(predicted_coords, mask)
        losses['distance'] = dist_loss
        
        # Bond length loss (if bond information provided)
        if bond_indices is not None:
            bond_loss = self.bond_loss(predicted_coords, bond_indices, mask)
            losses['bond'] = bond_loss
        else:
            bond_loss = torch.tensor(0.0, device=predicted_noise.device)
            losses['bond'] = bond_loss
        
        # Total loss
        total_loss = (
            self.diffusion_weight * diff_loss +
            self.distance_weight * dist_loss +
            self.bond_weight * bond_loss
        )
        losses['total'] = total_loss
        
        return losses

