"""
Uncertainty-Aware Loss Functions for Pearl

Incorporates experimental uncertainty (B-factors, local resolution) into training.
High-confidence regions are weighted more heavily than uncertain regions.

This addresses the limitation that Pearl doesn't account for:
1. Variable quality across different structures (resolution)
2. Variable quality within structures (B-factors, local resolution)
3. CryoEM data with heterogeneous local resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import numpy as np


class UncertaintyWeightedDiffusionLoss(nn.Module):
    """
    Diffusion loss weighted by experimental uncertainty.
    
    Atoms with high confidence (low B-factor, good local resolution)
    contribute more to the loss than uncertain atoms.
    """
    
    def __init__(
        self,
        loss_type: str = 'mse',
        weighting_scheme: str = 'inverse_variance',
        min_weight: float = 0.1,  # Minimum weight for uncertain atoms
        resolution_scaling: bool = True,  # Scale by overall resolution
    ):
        """
        Args:
            loss_type: 'mse' or 'l1'
            weighting_scheme: How to weight by confidence
                - 'linear': weight = confidence
                - 'squared': weight = confidence^2
                - 'inverse_variance': weight = confidence^2 (treat as 1/sigma^2)
                - 'sigmoid': weight = sigmoid(confidence)
            min_weight: Minimum weight (prevents ignoring uncertain regions)
            resolution_scaling: Scale loss by overall structure resolution
        """
        super().__init__()
        self.loss_type = loss_type
        self.weighting_scheme = weighting_scheme
        self.min_weight = min_weight
        self.resolution_scaling = resolution_scaling
        
    def forward(
        self,
        predicted_noise: torch.Tensor,  # [batch, n_atoms, 3]
        true_noise: torch.Tensor,  # [batch, n_atoms, 3]
        confidence: Optional[torch.Tensor] = None,  # [batch, n_atoms]
        resolution: Optional[torch.Tensor] = None,  # [batch]
        mask: Optional[torch.Tensor] = None,  # [batch, n_atoms]
    ) -> torch.Tensor:
        """
        Compute uncertainty-weighted diffusion loss.
        
        Args:
            predicted_noise: Predicted noise
            true_noise: True noise
            confidence: Per-atom confidence scores [0, 1]
            resolution: Overall resolution per structure (Å)
            mask: Valid atom mask
            
        Returns:
            Weighted loss value
        """
        # Compute base loss
        if self.loss_type == 'mse':
            loss = F.mse_loss(predicted_noise, true_noise, reduction='none')
        elif self.loss_type == 'l1':
            loss = F.l1_loss(predicted_noise, true_noise, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Sum over coordinates
        loss = loss.sum(dim=-1)  # [batch, n_atoms]
        
        # Apply confidence weighting
        if confidence is not None:
            weights = self._compute_weights(confidence)
            loss = loss * weights
        
        # Apply resolution scaling
        if self.resolution_scaling and resolution is not None:
            resolution_weights = self._compute_resolution_weights(resolution)
            # Expand to [batch, 1] for broadcasting
            resolution_weights = resolution_weights.unsqueeze(-1)
            loss = loss * resolution_weights
        
        # Apply mask
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss
    
    def _compute_weights(self, confidence: torch.Tensor) -> torch.Tensor:
        """
        Convert confidence scores to loss weights.
        
        Args:
            confidence: Confidence scores [batch, n_atoms]
            
        Returns:
            Loss weights [batch, n_atoms]
        """
        if self.weighting_scheme == 'linear':
            weights = confidence
        elif self.weighting_scheme == 'squared':
            weights = confidence ** 2
        elif self.weighting_scheme == 'inverse_variance':
            # Confidence ~ 1/sigma, so weight ~ 1/sigma^2
            weights = confidence ** 2
        elif self.weighting_scheme == 'sigmoid':
            # Sigmoid transformation for smooth weighting
            weights = torch.sigmoid(5 * (confidence - 0.5))
        else:
            raise ValueError(f"Unknown weighting scheme: {self.weighting_scheme}")
        
        # Apply minimum weight
        weights = torch.clamp(weights, min=self.min_weight)
        
        # Normalize so mean weight is 1.0 (preserves loss scale)
        weights = weights / (weights.mean() + 1e-8)
        
        return weights
    
    def _compute_resolution_weights(self, resolution: torch.Tensor) -> torch.Tensor:
        """
        Compute per-structure weights based on overall resolution.
        
        Better resolution (lower value) -> higher weight
        
        Args:
            resolution: Resolution in Angstroms [batch]
            
        Returns:
            Resolution weights [batch]
        """
        # Typical resolution ranges:
        # X-ray: 1.0-4.0 Å (excellent to moderate)
        # CryoEM: 2.0-10.0 Å (excellent to poor)
        
        # Convert resolution to weight: lower resolution = higher weight
        # Use exponential decay: weight = exp(-resolution / scale)
        scale = 3.0  # Angstroms
        weights = torch.exp(-resolution / scale)
        
        # Normalize
        weights = weights / (weights.mean() + 1e-8)
        
        return weights


class AdaptiveUncertaintyLoss(nn.Module):
    """
    Learns to predict and use uncertainty during training.
    
    The model learns to predict its own uncertainty (aleatoric uncertainty)
    in addition to the coordinate prediction. This is combined with
    experimental uncertainty (epistemic uncertainty).
    """
    
    def __init__(
        self,
        loss_type: str = 'mse',
        learn_uncertainty: bool = True,
        uncertainty_weight: float = 0.1,
    ):
        """
        Args:
            loss_type: Base loss type
            learn_uncertainty: Whether to learn predicted uncertainty
            uncertainty_weight: Weight for uncertainty regularization
        """
        super().__init__()
        self.loss_type = loss_type
        self.learn_uncertainty = learn_uncertainty
        self.uncertainty_weight = uncertainty_weight
        
    def forward(
        self,
        predicted_noise: torch.Tensor,  # [batch, n_atoms, 3]
        true_noise: torch.Tensor,  # [batch, n_atoms, 3]
        predicted_uncertainty: Optional[torch.Tensor] = None,  # [batch, n_atoms]
        experimental_confidence: Optional[torch.Tensor] = None,  # [batch, n_atoms]
        mask: Optional[torch.Tensor] = None,  # [batch, n_atoms]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive uncertainty loss.
        
        Uses both predicted uncertainty (learned) and experimental confidence.
        
        Args:
            predicted_noise: Model's noise prediction
            true_noise: Ground truth noise
            predicted_uncertainty: Model's uncertainty prediction (log variance)
            experimental_confidence: Experimental confidence scores
            mask: Valid atom mask
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Compute base loss
        if self.loss_type == 'mse':
            base_loss = F.mse_loss(predicted_noise, true_noise, reduction='none')
        elif self.loss_type == 'l1':
            base_loss = F.l1_loss(predicted_noise, true_noise, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        base_loss = base_loss.sum(dim=-1)  # [batch, n_atoms]
        
        # Apply learned uncertainty weighting
        if self.learn_uncertainty and predicted_uncertainty is not None:
            # Uncertainty-weighted loss (Kendall & Gal, 2017)
            # L = (1 / (2 * sigma^2)) * ||y - y_pred||^2 + (1/2) * log(sigma^2)
            # where predicted_uncertainty = log(sigma^2)
            
            precision = torch.exp(-predicted_uncertainty)  # 1 / sigma^2
            weighted_loss = 0.5 * precision * base_loss + 0.5 * predicted_uncertainty
            
            losses['uncertainty_weighted'] = weighted_loss.mean()
            
            # Regularization: penalize overly confident predictions
            losses['uncertainty_reg'] = self.uncertainty_weight * predicted_uncertainty.mean()
        else:
            weighted_loss = base_loss
        
        # Apply experimental confidence weighting
        if experimental_confidence is not None:
            # Combine learned and experimental uncertainty
            exp_weights = experimental_confidence ** 2  # Inverse variance weighting
            exp_weights = exp_weights / (exp_weights.mean() + 1e-8)
            weighted_loss = weighted_loss * exp_weights
        
        # Apply mask
        if mask is not None:
            weighted_loss = weighted_loss * mask
            total_loss = weighted_loss.sum() / (mask.sum() + 1e-8)
        else:
            total_loss = weighted_loss.mean()
        
        losses['total'] = total_loss
        losses['base'] = base_loss.mean()
        
        return losses


class ResolutionStratifiedLoss(nn.Module):
    """
    Stratifies loss by resolution bins.
    
    Ensures the model learns from both high and low resolution structures
    by balancing contributions across resolution ranges.
    """
    
    def __init__(
        self,
        resolution_bins: list = [0.0, 2.0, 3.0, 4.0, 10.0],  # Angstroms
        bin_weights: Optional[list] = None,  # Optional per-bin weights
        loss_type: str = 'mse',
    ):
        """
        Args:
            resolution_bins: Bin edges for resolution stratification
            bin_weights: Optional weights for each bin (default: equal)
            loss_type: Base loss type
        """
        super().__init__()
        self.resolution_bins = torch.tensor(resolution_bins)
        self.loss_type = loss_type
        
        if bin_weights is None:
            # Equal weight for each bin
            self.bin_weights = torch.ones(len(resolution_bins) - 1)
        else:
            self.bin_weights = torch.tensor(bin_weights)
        
        # Normalize weights
        self.bin_weights = self.bin_weights / self.bin_weights.sum()
        
    def forward(
        self,
        predicted_noise: torch.Tensor,  # [batch, n_atoms, 3]
        true_noise: torch.Tensor,  # [batch, n_atoms, 3]
        resolution: torch.Tensor,  # [batch]
        mask: Optional[torch.Tensor] = None,  # [batch, n_atoms]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute resolution-stratified loss.
        
        Args:
            predicted_noise: Predicted noise
            true_noise: True noise
            resolution: Per-structure resolution (Å)
            mask: Valid atom mask
            
        Returns:
            Dictionary with total loss and per-bin losses
        """
        # Compute base loss per atom
        if self.loss_type == 'mse':
            loss = F.mse_loss(predicted_noise, true_noise, reduction='none')
        elif self.loss_type == 'l1':
            loss = F.l1_loss(predicted_noise, true_noise, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        loss = loss.sum(dim=-1)  # [batch, n_atoms]
        
        # Apply mask
        if mask is not None:
            loss = loss * mask
        
        # Compute per-structure loss
        if mask is not None:
            per_structure_loss = loss.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        else:
            per_structure_loss = loss.mean(dim=1)
        
        # Assign structures to resolution bins
        resolution_bins = self.resolution_bins.to(resolution.device)
        bin_weights = self.bin_weights.to(resolution.device)
        
        bin_losses = []
        bin_counts = []
        
        for i in range(len(resolution_bins) - 1):
            bin_min = resolution_bins[i]
            bin_max = resolution_bins[i + 1]
            
            # Find structures in this bin
            in_bin = (resolution >= bin_min) & (resolution < bin_max)
            
            if in_bin.sum() > 0:
                bin_loss = per_structure_loss[in_bin].mean()
                bin_losses.append(bin_loss * bin_weights[i])
                bin_counts.append(in_bin.sum().item())
            else:
                bin_losses.append(torch.tensor(0.0, device=resolution.device))
                bin_counts.append(0)
        
        # Total loss is weighted sum of bin losses
        total_loss = sum(bin_losses)
        
        # Return detailed losses
        losses = {'total': total_loss}
        for i, (loss_val, count) in enumerate(zip(bin_losses, bin_counts)):
            bin_name = f"bin_{i}_({resolution_bins[i]:.1f}-{resolution_bins[i+1]:.1f}A)"
            losses[bin_name] = loss_val
            losses[f"{bin_name}_count"] = count
        
        return losses


class CombinedUncertaintyAwareLoss(nn.Module):
    """
    Combines all uncertainty-aware loss components.
    
    Integrates:
    1. Per-atom confidence weighting (B-factors, local resolution)
    2. Per-structure resolution weighting
    3. Resolution stratification
    4. Optional learned uncertainty
    """
    
    def __init__(
        self,
        base_weight: float = 1.0,
        resolution_stratify: bool = True,
        learn_uncertainty: bool = False,
    ):
        super().__init__()
        
        self.base_weight = base_weight
        self.resolution_stratify = resolution_stratify
        self.learn_uncertainty = learn_uncertainty
        
        # Main uncertainty-weighted loss
        self.uncertainty_loss = UncertaintyWeightedDiffusionLoss(
            weighting_scheme='inverse_variance',
            resolution_scaling=True,
        )
        
        # Optional resolution stratification
        if resolution_stratify:
            self.stratified_loss = ResolutionStratifiedLoss()
        
        # Optional adaptive uncertainty
        if learn_uncertainty:
            self.adaptive_loss = AdaptiveUncertaintyLoss()
    
    def forward(
        self,
        predicted_noise: torch.Tensor,
        true_noise: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        resolution: Optional[torch.Tensor] = None,
        predicted_uncertainty: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined uncertainty-aware loss.
        """
        losses = {}
        
        # Main uncertainty-weighted loss
        main_loss = self.uncertainty_loss(
            predicted_noise, true_noise,
            confidence=confidence,
            resolution=resolution,
            mask=mask,
        )
        losses['main'] = main_loss
        
        total_loss = self.base_weight * main_loss
        
        # Add resolution stratification
        if self.resolution_stratify and resolution is not None:
            strat_losses = self.stratified_loss(
                predicted_noise, true_noise,
                resolution=resolution,
                mask=mask,
            )
            losses.update({f'stratified_{k}': v for k, v in strat_losses.items()})
            total_loss = total_loss + 0.1 * strat_losses['total']
        
        # Add adaptive uncertainty
        if self.learn_uncertainty and predicted_uncertainty is not None:
            adaptive_losses = self.adaptive_loss(
                predicted_noise, true_noise,
                predicted_uncertainty=predicted_uncertainty,
                experimental_confidence=confidence,
                mask=mask,
            )
            losses.update({f'adaptive_{k}': v for k, v in adaptive_losses.items()})
            total_loss = total_loss + 0.1 * adaptive_losses['total']
        
        losses['total'] = total_loss
        
        return losses

