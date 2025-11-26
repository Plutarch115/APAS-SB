"""
Loss functions for ΔΔG prediction training.

Implements multiple loss components:
1. MSE loss for ΔΔG prediction
2. Negative log-likelihood for uncertainty estimation
3. Calibration loss for confidence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np


class DDGLoss(nn.Module):
    """
    Loss function for ΔΔG prediction with uncertainty estimation.
    
    Combines:
    1. MSE loss for ΔΔG prediction
    2. Negative log-likelihood for uncertainty
    3. Regularization for confidence calibration
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        nll_weight: float = 0.5,
        calibration_weight: float = 0.1,
        contribution_weight: float = 0.05
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.nll_weight = nll_weight
        self.calibration_weight = calibration_weight
        self.contribution_weight = contribution_weight
        
    def forward(
        self,
        pred_ddg: torch.Tensor,
        true_ddg: torch.Tensor,
        ddg_confidence: torch.Tensor,
        residue_contrib: Optional[torch.Tensor] = None,
        data_weight: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ΔΔG prediction loss.
        
        Args:
            pred_ddg: [batch] - Predicted ΔΔG
            true_ddg: [batch] - True ΔΔG
            ddg_confidence: [batch] - Predicted uncertainty
            residue_contrib: [batch, n_atoms] - Per-residue contributions (optional)
            data_weight: [batch] - Per-sample weights (e.g., experimental vs pseudo-labels)
            
        Returns:
            Dictionary with total loss and components
        """
        if data_weight is None:
            data_weight = torch.ones_like(pred_ddg)
        
        # 1. MSE loss
        mse = (pred_ddg - true_ddg) ** 2
        weighted_mse = (mse * data_weight).mean()
        
        # 2. Negative log-likelihood (Gaussian)
        # Assumes errors are Gaussian with std = ddg_confidence
        # NLL = 0.5 * log(2π * σ²) + 0.5 * (y - μ)² / σ²
        nll = 0.5 * torch.log(2 * np.pi * ddg_confidence ** 2 + 1e-6) + \
              0.5 * (pred_ddg - true_ddg) ** 2 / (ddg_confidence ** 2 + 1e-6)
        weighted_nll = (nll * data_weight).mean()
        
        # 3. Calibration loss (confidence should match actual error)
        actual_error = torch.abs(pred_ddg - true_ddg)
        calibration_loss = F.mse_loss(ddg_confidence, actual_error)
        
        # 4. Contribution regularization (optional)
        # Encourage sparse contributions
        contrib_loss = torch.tensor(0.0, device=pred_ddg.device)
        if residue_contrib is not None:
            contrib_loss = torch.abs(residue_contrib).mean()
        
        # Total loss
        total_loss = (
            self.mse_weight * weighted_mse +
            self.nll_weight * weighted_nll +
            self.calibration_weight * calibration_loss +
            self.contribution_weight * contrib_loss
        )
        
        return {
            'total': total_loss,
            'mse': weighted_mse,
            'nll': weighted_nll,
            'calibration': calibration_loss,
            'contribution': contrib_loss,
            'mae': torch.abs(pred_ddg - true_ddg).mean()
        }


class DDGMetrics:
    """
    Metrics for evaluating ΔΔG predictions.
    """
    
    @staticmethod
    def compute_metrics(
        pred_ddg: torch.Tensor,
        true_ddg: torch.Tensor,
        confidence: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics for ΔΔG predictions.
        
        Args:
            pred_ddg: [N] - Predicted ΔΔG values
            true_ddg: [N] - True ΔΔG values
            confidence: [N] - Predicted confidence intervals (optional)
            
        Returns:
            Dictionary with metrics
        """
        pred_ddg = pred_ddg.detach().cpu().numpy()
        true_ddg = true_ddg.detach().cpu().numpy()
        
        # Basic metrics
        mae = np.mean(np.abs(pred_ddg - true_ddg))
        rmse = np.sqrt(np.mean((pred_ddg - true_ddg) ** 2))
        
        # Correlation
        pearson_r = np.corrcoef(pred_ddg, true_ddg)[0, 1]
        
        # Spearman correlation
        from scipy.stats import spearmanr
        spearman_rho, _ = spearmanr(pred_ddg, true_ddg)
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'pearson_r': float(pearson_r),
            'spearman_rho': float(spearman_rho)
        }
        
        # Confidence calibration metrics
        if confidence is not None:
            confidence = confidence.detach().cpu().numpy()
            actual_error = np.abs(pred_ddg - true_ddg)
            
            # Calibration error (how well confidence matches actual error)
            calibration_error = np.mean(np.abs(confidence - actual_error))
            
            # Fraction within confidence interval
            within_ci = np.mean(actual_error <= confidence)
            
            metrics.update({
                'calibration_error': float(calibration_error),
                'within_ci': float(within_ci),
                'mean_confidence': float(np.mean(confidence))
            })
        
        return metrics
    
    @staticmethod
    def compute_per_mutation_type_metrics(
        pred_ddg: torch.Tensor,
        true_ddg: torch.Tensor,
        mutation_types: list
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics broken down by mutation type.
        
        Args:
            pred_ddg: [N] - Predicted ΔΔG values
            true_ddg: [N] - True ΔΔG values
            mutation_types: List of mutation type strings
            
        Returns:
            Dictionary mapping mutation type to metrics
        """
        pred_ddg = pred_ddg.detach().cpu().numpy()
        true_ddg = true_ddg.detach().cpu().numpy()
        
        # Group by mutation type
        type_metrics = {}
        unique_types = set(mutation_types)
        
        for mut_type in unique_types:
            mask = np.array([mt == mut_type for mt in mutation_types])
            if mask.sum() == 0:
                continue
            
            pred_subset = pred_ddg[mask]
            true_subset = true_ddg[mask]
            
            mae = np.mean(np.abs(pred_subset - true_subset))
            rmse = np.sqrt(np.mean((pred_subset - true_subset) ** 2))
            
            if len(pred_subset) > 1:
                pearson_r = np.corrcoef(pred_subset, true_subset)[0, 1]
            else:
                pearson_r = 0.0
            
            type_metrics[mut_type] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'pearson_r': float(pearson_r),
                'count': int(mask.sum())
            }
        
        return type_metrics


class WeightedDataSampler:
    """
    Sampler for weighted data sources (experimental, MD-derived, pseudo-labels).
    """
    
    def __init__(
        self,
        data_sources: list,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            data_sources: List of data source labels for each sample
            weights: Dictionary mapping source to weight
        """
        self.data_sources = data_sources
        
        if weights is None:
            weights = {
                'experimental': 10.0,
                'md_fep': 1.0,
                'pseudo_label': 0.1
            }
        self.weights = weights
        
        # Compute sampling probabilities
        self.sample_weights = torch.tensor([
            self.weights.get(source, 1.0) for source in data_sources
        ])
        self.sample_weights = self.sample_weights / self.sample_weights.sum()
    
    def get_batch_weights(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get weights for a batch of samples.
        
        Args:
            indices: Batch indices
            
        Returns:
            Weights for each sample in batch
        """
        return self.sample_weights[indices]


def categorize_mutation(mutation: str) -> str:
    """
    Categorize mutation by amino acid properties.
    
    Args:
        mutation: Mutation string (e.g., "L99A")
        
    Returns:
        Category string
    """
    # Amino acid categories
    hydrophobic = set('AILMFVPW')
    polar = set('STNQ')
    charged_pos = set('KR')
    charged_neg = set('DE')
    aromatic = set('FYW')
    small = set('AGST')
    
    if len(mutation) < 2:
        return 'unknown'
    
    wt_aa = mutation[0]
    mut_aa = mutation[-1]
    
    # Determine categories
    def get_category(aa):
        if aa in hydrophobic:
            return 'hydrophobic'
        elif aa in polar:
            return 'polar'
        elif aa in charged_pos:
            return 'positive'
        elif aa in charged_neg:
            return 'negative'
        elif aa in aromatic:
            return 'aromatic'
        elif aa in small:
            return 'small'
        else:
            return 'other'
    
    wt_cat = get_category(wt_aa)
    mut_cat = get_category(mut_aa)
    
    # Special cases
    if wt_aa == mut_aa:
        return 'wild_type'
    elif wt_cat == mut_cat:
        return f'{wt_cat}_to_{mut_cat}'
    else:
        return f'{wt_cat}_to_{mut_cat}'

