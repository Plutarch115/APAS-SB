"""
Boltz-2 Loss Functions for Binding Affinity Prediction.

Implements the three main loss functions used in Boltz-2:
1. Huber Loss: Robust regression loss for continuous affinity values
2. Pairwise Ranking Loss: Relative ranking of binding affinities
3. Focal Loss: Binary classification with hard negative mining

Reference: Boltz-2 paper (2025.06.14.659707v1.full.pdf)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class HuberLoss(nn.Module):
    """
    Huber Loss for robust regression of binding affinity values.
    
    Combines L1 and L2 loss:
    - L2 loss for small errors (smooth gradient)
    - L1 loss for large errors (robust to outliers)
    
    Used in Boltz-2 for continuous affinity prediction (ChEMBL, BindingDB, PDBbind).
    """
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            delta: Threshold for switching between L1 and L2 loss
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.delta = delta
        self.reduction = reduction
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            predictions: Predicted affinity values (batch_size,)
            targets: True affinity values (batch_size,)
            weights: Optional sample weights (batch_size,)
        
        Returns:
            Loss value
        """
        # Compute absolute error
        error = predictions - targets
        abs_error = torch.abs(error)
        
        # Huber loss formula
        quadratic = torch.min(abs_error, torch.tensor(self.delta, device=error.device))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        
        # Apply sample weights if provided
        if weights is not None:
            loss = loss * weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise Ranking Loss for relative binding affinity prediction.
    
    Ensures that if compound A binds stronger than compound B,
    the model predicts higher affinity for A than B.
    
    Used in Boltz-2 to learn relative rankings from noisy absolute values.
    """
    
    def __init__(self, margin: float = 0.5, reduction: str = 'mean'):
        """
        Args:
            margin: Minimum difference between predictions for ranked pairs
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            predictions: Predicted affinity values (batch_size,)
            targets: True affinity values (batch_size,)
            weights: Optional sample weights (batch_size,)
        
        Returns:
            Loss value
        """
        batch_size = predictions.size(0)
        
        # Create all pairwise comparisons
        # pred_i - pred_j should have same sign as target_i - target_j
        pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)  # (batch, batch)
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)  # (batch, batch)
        
        # Only consider pairs where targets differ significantly
        valid_pairs = torch.abs(target_diff) > 0.1
        
        # Ranking loss: max(0, margin - sign(target_diff) * pred_diff)
        target_sign = torch.sign(target_diff)
        loss = torch.clamp(self.margin - target_sign * pred_diff, min=0.0)
        
        # Only compute loss for valid pairs
        loss = loss * valid_pairs.float()
        
        # Apply sample weights if provided
        if weights is not None:
            weight_matrix = weights.unsqueeze(1) * weights.unsqueeze(0)
            loss = loss * weight_matrix
        
        # Apply reduction
        num_pairs = valid_pairs.sum()
        if self.reduction == 'mean' and num_pairs > 0:
            return loss.sum() / num_pairs
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with hard negative mining.
    
    Focuses training on hard examples by down-weighting easy examples.
    Particularly useful for imbalanced datasets (many decoys, few binders).
    
    Used in Boltz-2 for PubChem HTS, CeMM, MIDAS datasets.
    
    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    where p_t is the probability of the true class.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Weighting factor for positive class (0.25 in Boltz-2)
            gamma: Focusing parameter (2.0 in Boltz-2)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            predictions: Predicted probabilities (batch_size,) or logits
            targets: True labels (batch_size,) - 0 or 1
            weights: Optional sample weights (batch_size,)
        
        Returns:
            Loss value
        """
        # Convert logits to probabilities if needed
        if predictions.min() < 0 or predictions.max() > 1:
            probs = torch.sigmoid(predictions)
        else:
            probs = predictions
        
        # Compute focal loss
        targets = targets.float()
        
        # p_t: probability of true class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # α_t: alpha for true class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal loss formula
        focal_weight = (1 - p_t) ** self.gamma
        ce_loss = -torch.log(p_t + 1e-8)  # Add epsilon for numerical stability
        loss = alpha_t * focal_weight * ce_loss
        
        # Apply sample weights if provided
        if weights is not None:
            loss = loss * weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedBoltz2Loss(nn.Module):
    """
    Combined loss function for Boltz-2 multi-task training.

    Automatically selects appropriate loss based on task type:
    - Continuous affinity: Huber Loss + Pairwise Ranking Loss
    - Binary classification: Focal Loss

    Weights losses by data source quality and task importance.
    """

    def __init__(
        self,
        huber_delta: float = 1.0,
        ranking_margin: float = 0.5,
        ranking_weight: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        """
        Args:
            huber_delta: Delta parameter for Huber loss
            ranking_margin: Margin for pairwise ranking loss
            ranking_weight: Weight for ranking loss relative to Huber loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        self.huber_loss = HuberLoss(delta=huber_delta)
        self.ranking_loss = PairwiseRankingLoss(margin=ranking_margin)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.ranking_weight = ranking_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        task_types: list,
        weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            predictions: Predicted values (batch_size,)
            targets: True values (batch_size,)
            task_types: List of task types ('binding_affinity' or 'binary_classification')
            weights: Optional sample weights (batch_size,)

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Separate continuous and binary tasks
        continuous_mask = torch.tensor(
            [t == 'binding_affinity' for t in task_types],
            device=predictions.device
        )
        binary_mask = torch.tensor(
            [t == 'binary_classification' for t in task_types],
            device=predictions.device
        )

        total_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
        loss_dict = {}

        # Continuous affinity tasks: Huber + Ranking
        if continuous_mask.any():
            cont_preds = predictions[continuous_mask]
            cont_targets = targets[continuous_mask]
            cont_weights = weights[continuous_mask] if weights is not None else None

            # Huber loss
            huber = self.huber_loss(cont_preds, cont_targets, cont_weights)
            total_loss = total_loss + huber
            loss_dict['huber'] = huber.item()

            # Ranking loss (if enough samples)
            if cont_preds.size(0) >= 2:
                ranking = self.ranking_loss(cont_preds, cont_targets, cont_weights)
                total_loss = total_loss + self.ranking_weight * ranking
                loss_dict['ranking'] = ranking.item()

        # Binary classification tasks: Focal loss
        if binary_mask.any():
            bin_preds = predictions[binary_mask]
            bin_targets = targets[binary_mask]
            bin_weights = weights[binary_mask] if weights is not None else None

            focal = self.focal_loss(bin_preds, bin_targets, bin_weights)
            total_loss = total_loss + focal
            loss_dict['focal'] = focal.item()

        loss_dict['total'] = total_loss.item() if total_loss.numel() == 1 else total_loss.mean().item()

        return total_loss, loss_dict


def get_loss_function(task_type: str) -> nn.Module:
    """
    Get appropriate loss function for a given task type.

    Args:
        task_type: 'binding_affinity', 'binary_classification', 'ddg', 'kcat', or 'fitness'

    Returns:
        Loss function module
    """
    if task_type == 'binding_affinity':
        return HuberLoss(delta=1.0)
    elif task_type == 'binary_classification':
        return FocalLoss(alpha=0.25, gamma=2.0)
    elif task_type in ['ddg', 'kcat', 'fitness']:
        return HuberLoss(delta=1.0)  # Use Huber for other regression tasks
    else:
        raise ValueError(f"Unknown task type: {task_type}")

