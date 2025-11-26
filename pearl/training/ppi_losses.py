"""
Protein-Protein Interaction (PPI) Losses for Pearl

This module implements PPI-specific loss functions:
- Interface RMSD loss
- Contact prediction loss
- Interface lDDT loss
- DockQ-inspired losses

Integrates with Pearl's diffusion training framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class InterfaceRMSDLoss(nn.Module):
    """
    Interface RMSD loss for protein-protein complexes.
    
    Focuses on interface residues rather than entire structure.
    """
    
    def __init__(
        self,
        interface_weight: float = 2.0,
        non_interface_weight: float = 1.0
    ):
        """
        Initialize interface RMSD loss.
        
        Args:
            interface_weight: Weight for interface residues
            non_interface_weight: Weight for non-interface residues
        """
        super().__init__()
        self.interface_weight = interface_weight
        self.non_interface_weight = non_interface_weight
    
    def forward(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        interface_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute interface-weighted RMSD loss.
        
        Args:
            pred_coords: Predicted coordinates (B, N, 3)
            true_coords: True coordinates (B, N, 3)
            interface_mask: Binary mask for interface residues (B, N)
            
        Returns:
            Loss value
        """
        # Compute squared deviations
        squared_deviations = ((pred_coords - true_coords) ** 2).sum(dim=-1)  # (B, N)
        
        # Apply interface weighting
        weights = torch.where(
            interface_mask > 0.5,
            torch.tensor(self.interface_weight, device=pred_coords.device),
            torch.tensor(self.non_interface_weight, device=pred_coords.device)
        )
        
        # Weighted RMSD
        weighted_squared_deviations = squared_deviations * weights
        rmsd = torch.sqrt(weighted_squared_deviations.mean())
        
        return rmsd


class ContactPredictionLoss(nn.Module):
    """
    Contact prediction loss for protein-protein interfaces.
    
    Predicts which residue pairs are in contact at the interface.
    """
    
    def __init__(
        self,
        contact_threshold: float = 8.0,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        """
        Initialize contact prediction loss.
        
        Args:
            contact_threshold: Distance threshold (Å) for contacts
            use_focal_loss: Use focal loss for class imbalance
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
        """
        super().__init__()
        self.contact_threshold = contact_threshold
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def forward(
        self,
        pred_coords_a: torch.Tensor,
        pred_coords_b: torch.Tensor,
        true_coords_a: torch.Tensor,
        true_coords_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contact prediction loss.
        
        Args:
            pred_coords_a: Predicted coordinates for chain A (B, N_a, 3)
            pred_coords_b: Predicted coordinates for chain B (B, N_b, 3)
            true_coords_a: True coordinates for chain A (B, N_a, 3)
            true_coords_b: True coordinates for chain B (B, N_b, 3)
            
        Returns:
            Loss value
        """
        # Compute true contacts
        true_distances = torch.cdist(true_coords_a, true_coords_b)  # (B, N_a, N_b)
        true_contacts = (true_distances < self.contact_threshold).float()
        
        # Compute predicted distances
        pred_distances = torch.cdist(pred_coords_a, pred_coords_b)  # (B, N_a, N_b)
        pred_contacts = torch.sigmoid(-0.5 * (pred_distances - self.contact_threshold))
        
        # Compute loss
        if self.use_focal_loss:
            # Focal loss for class imbalance
            bce = F.binary_cross_entropy(pred_contacts, true_contacts, reduction='none')
            pt = torch.where(true_contacts == 1, pred_contacts, 1 - pred_contacts)
            focal_weight = (1 - pt) ** self.focal_gamma
            alpha_weight = torch.where(
                true_contacts == 1,
                torch.tensor(self.focal_alpha, device=pred_contacts.device),
                torch.tensor(1 - self.focal_alpha, device=pred_contacts.device)
            )
            loss = (alpha_weight * focal_weight * bce).mean()
        else:
            # Standard BCE
            loss = F.binary_cross_entropy(pred_contacts, true_contacts)
        
        return loss


class InterfaceLDDTLoss(nn.Module):
    """
    Interface lDDT (local Distance Difference Test) loss.
    
    Measures local distance preservation at the interface.
    """
    
    def __init__(
        self,
        distance_thresholds: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
        inclusion_radius: float = 15.0
    ):
        """
        Initialize interface lDDT loss.
        
        Args:
            distance_thresholds: Distance thresholds for lDDT
            inclusion_radius: Radius for including neighbors
        """
        super().__init__()
        self.distance_thresholds = distance_thresholds
        self.inclusion_radius = inclusion_radius
    
    def forward(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        interface_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute interface lDDT loss.
        
        Args:
            pred_coords: Predicted coordinates (B, N, 3)
            true_coords: True coordinates (B, N, 3)
            interface_mask: Binary mask for interface residues (B, N)
            
        Returns:
            Loss value (1 - lDDT, so lower is better)
        """
        batch_size, n_atoms, _ = pred_coords.shape
        
        # Compute true distances
        true_distances = torch.cdist(true_coords, true_coords)  # (B, N, N)
        
        # Compute predicted distances
        pred_distances = torch.cdist(pred_coords, pred_coords)  # (B, N, N)
        
        # Distance differences
        distance_diffs = torch.abs(pred_distances - true_distances)  # (B, N, N)
        
        # Inclusion mask (only consider neighbors within radius)
        inclusion_mask = (true_distances < self.inclusion_radius).float()
        
        # Interface mask (only consider interface residues)
        interface_mask_2d = interface_mask.unsqueeze(-1) * interface_mask.unsqueeze(-2)  # (B, N, N)
        
        # Combined mask
        mask = inclusion_mask * interface_mask_2d
        
        # Compute lDDT for each threshold
        lddt_scores = []
        for threshold in self.distance_thresholds:
            preserved = (distance_diffs < threshold).float()
            lddt = (preserved * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-8)
            lddt_scores.append(lddt)
        
        # Average over thresholds
        lddt = torch.stack(lddt_scores, dim=0).mean(dim=0)  # (B,)
        
        # Return 1 - lDDT as loss (so lower is better)
        loss = 1.0 - lddt.mean()
        
        return loss


class DockQLoss(nn.Module):
    """
    DockQ-inspired loss for protein-protein docking quality.
    
    DockQ combines interface RMSD, ligand RMSD, and fraction of native contacts.
    """
    
    def __init__(
        self,
        contact_threshold: float = 5.0,
        irmsd_weight: float = 1.0,
        fnat_weight: float = 1.0
    ):
        """
        Initialize DockQ loss.
        
        Args:
            contact_threshold: Distance threshold for native contacts
            irmsd_weight: Weight for interface RMSD component
            fnat_weight: Weight for fraction of native contacts component
        """
        super().__init__()
        self.contact_threshold = contact_threshold
        self.irmsd_weight = irmsd_weight
        self.fnat_weight = fnat_weight
    
    def forward(
        self,
        pred_coords_a: torch.Tensor,
        pred_coords_b: torch.Tensor,
        true_coords_a: torch.Tensor,
        true_coords_b: torch.Tensor,
        interface_mask_a: torch.Tensor,
        interface_mask_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DockQ-inspired loss.
        
        Args:
            pred_coords_a: Predicted coordinates for chain A (B, N_a, 3)
            pred_coords_b: Predicted coordinates for chain B (B, N_b, 3)
            true_coords_a: True coordinates for chain A (B, N_a, 3)
            true_coords_b: True coordinates for chain B (B, N_b, 3)
            interface_mask_a: Interface mask for chain A (B, N_a)
            interface_mask_b: Interface mask for chain B (B, N_b)
            
        Returns:
            Loss value
        """
        # 1. Interface RMSD
        pred_interface_a = pred_coords_a * interface_mask_a.unsqueeze(-1)
        true_interface_a = true_coords_a * interface_mask_a.unsqueeze(-1)
        pred_interface_b = pred_coords_b * interface_mask_b.unsqueeze(-1)
        true_interface_b = true_coords_b * interface_mask_b.unsqueeze(-1)
        
        irmsd_a = torch.sqrt(((pred_interface_a - true_interface_a) ** 2).sum(dim=-1).mean(dim=-1))
        irmsd_b = torch.sqrt(((pred_interface_b - true_interface_b) ** 2).sum(dim=-1).mean(dim=-1))
        irmsd = (irmsd_a + irmsd_b) / 2.0
        
        # 2. Fraction of native contacts
        true_distances = torch.cdist(true_coords_a, true_coords_b)
        pred_distances = torch.cdist(pred_coords_a, pred_coords_b)
        
        true_contacts = (true_distances < self.contact_threshold).float()
        pred_contacts = (pred_distances < self.contact_threshold).float()
        
        # Fnat = (true positives) / (total native contacts)
        true_positives = (true_contacts * pred_contacts).sum(dim=(1, 2))
        total_native = true_contacts.sum(dim=(1, 2))
        fnat = true_positives / (total_native + 1e-8)
        
        # DockQ-inspired loss (lower is better)
        # Normalize RMSD to [0, 1] range (assuming max RMSD ~ 10 Å)
        normalized_irmsd = torch.clamp(irmsd / 10.0, 0.0, 1.0)
        
        # Combine components
        loss = (
            self.irmsd_weight * normalized_irmsd +
            self.fnat_weight * (1.0 - fnat)
        ).mean()
        
        return loss


class CombinedPPILoss(nn.Module):
    """
    Combined loss for protein-protein interaction prediction.
    
    Combines multiple PPI-specific losses.
    """
    
    def __init__(
        self,
        interface_rmsd_weight: float = 1.0,
        contact_weight: float = 0.5,
        interface_lddt_weight: float = 0.5,
        dockq_weight: float = 1.0
    ):
        """
        Initialize combined PPI loss.
        
        Args:
            interface_rmsd_weight: Weight for interface RMSD loss
            contact_weight: Weight for contact prediction loss
            interface_lddt_weight: Weight for interface lDDT loss
            dockq_weight: Weight for DockQ loss
        """
        super().__init__()
        self.interface_rmsd_loss = InterfaceRMSDLoss()
        self.contact_loss = ContactPredictionLoss()
        self.interface_lddt_loss = InterfaceLDDTLoss()
        self.dockq_loss = DockQLoss()
        
        self.interface_rmsd_weight = interface_rmsd_weight
        self.contact_weight = contact_weight
        self.interface_lddt_weight = interface_lddt_weight
        self.dockq_weight = dockq_weight
    
    def forward(
        self,
        pred_coords_a: torch.Tensor,
        pred_coords_b: torch.Tensor,
        true_coords_a: torch.Tensor,
        true_coords_b: torch.Tensor,
        interface_mask_a: torch.Tensor,
        interface_mask_b: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined PPI loss.
        
        Returns:
            Dictionary with individual losses and total loss
        """
        # Combine coordinates for some losses
        pred_coords = torch.cat([pred_coords_a, pred_coords_b], dim=1)
        true_coords = torch.cat([true_coords_a, true_coords_b], dim=1)
        interface_mask = torch.cat([interface_mask_a, interface_mask_b], dim=1)
        
        # Compute individual losses
        irmsd_loss = self.interface_rmsd_loss(pred_coords, true_coords, interface_mask)
        contact_loss = self.contact_loss(pred_coords_a, pred_coords_b, true_coords_a, true_coords_b)
        ilddt_loss = self.interface_lddt_loss(pred_coords, true_coords, interface_mask)
        dockq_loss = self.dockq_loss(
            pred_coords_a, pred_coords_b,
            true_coords_a, true_coords_b,
            interface_mask_a, interface_mask_b
        )
        
        # Combine losses
        total_loss = (
            self.interface_rmsd_weight * irmsd_loss +
            self.contact_weight * contact_loss +
            self.interface_lddt_weight * ilddt_loss +
            self.dockq_weight * dockq_loss
        )
        
        return {
            "total": total_loss,
            "interface_rmsd": irmsd_loss,
            "contact": contact_loss,
            "interface_lddt": ilddt_loss,
            "dockq": dockq_loss
        }

