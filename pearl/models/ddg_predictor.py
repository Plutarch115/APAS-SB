"""
ΔΔG Prediction Module for Pearl

Extends Pearl to predict binding free energy changes (ΔΔG) upon mutations
and ligand modifications, similar to Boltz-2 capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .pearl import Pearl


@dataclass
class DDGPrediction:
    """Container for ΔΔG prediction results"""
    ddg: float  # Predicted ΔΔG (kcal/mol)
    ddg_lower: float  # Lower confidence bound
    ddg_upper: float  # Upper confidence bound
    confidence: float  # Uncertainty estimate (kcal/mol)
    residue_contributions: torch.Tensor  # Per-residue contributions
    wt_coords: torch.Tensor  # Wild-type coordinates
    mut_coords: torch.Tensor  # Mutant coordinates
    attention_weights: torch.Tensor  # Attention weights


class DDGPredictionHead(nn.Module):
    """
    Neural network head for predicting ΔΔG from structural features.
    
    Takes difference features between wild-type and mutant structures
    and predicts the change in binding free energy.
    """
    
    def __init__(self, pair_dim: int = 128, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        
        # Global ΔΔG prediction from pair features
        self.global_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3)  # [mean, lower_std, upper_std]
        )
        
        # Per-residue contribution prediction
        self.residue_head = nn.Sequential(
            nn.Linear(pair_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # Attention mechanism for important residues
        self.attention = nn.MultiheadAttention(
            embed_dim=pair_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(pair_dim)
        
    def forward(
        self, 
        wt_pair: torch.Tensor, 
        mut_pair: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            wt_pair: [batch, n_atoms, n_atoms, pair_dim] - Wild-type pair features
            mut_pair: [batch, n_atoms, n_atoms, pair_dim] - Mutant pair features
            mask: [batch, n_atoms] - Mask for valid atoms
            
        Returns:
            Dictionary with:
                - ddg: [batch] - Predicted ΔΔG
                - ddg_lower: [batch] - Lower confidence bound
                - ddg_upper: [batch] - Upper confidence bound
                - ddg_confidence: [batch] - Uncertainty estimate
                - residue_contrib: [batch, n_atoms] - Per-residue contributions
                - attention_weights: [batch, n_atoms, n_atoms] - Attention weights
        """
        batch_size, n_atoms, _, pair_dim = wt_pair.shape
        
        # Compute difference features
        diff_pair = mut_pair - wt_pair  # [batch, n_atoms, n_atoms, pair_dim]
        
        # Average over second dimension to get per-residue features
        diff_features = diff_pair.mean(dim=2)  # [batch, n_atoms, pair_dim]
        
        # Normalize
        diff_features = self.layer_norm(diff_features)
        
        # Apply attention to focus on important residues
        attended_diff, attention_weights = self.attention(
            diff_features, diff_features, diff_features,
            key_padding_mask=~mask if mask is not None else None
        )  # [batch, n_atoms, pair_dim]
        
        # Global pooling for ΔΔG prediction
        if mask is not None:
            # Masked average
            mask_expanded = mask.unsqueeze(-1).float()  # [batch, n_atoms, 1]
            global_diff = (attended_diff * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            global_diff = attended_diff.mean(dim=1)  # [batch, pair_dim]
        
        # Predict ΔΔG with uncertainty
        ddg_pred = self.global_head(global_diff)  # [batch, 3]
        ddg_mean = ddg_pred[:, 0]
        ddg_lower_std = ddg_pred[:, 1]
        ddg_upper_std = ddg_pred[:, 2]
        
        # Ensure positive standard deviations and compute bounds
        ddg_lower_std = F.softplus(ddg_lower_std)
        ddg_upper_std = F.softplus(ddg_upper_std)
        ddg_lower = ddg_mean - ddg_lower_std
        ddg_upper = ddg_mean + ddg_upper_std
        
        # Per-residue contributions
        residue_contrib = self.residue_head(attended_diff).squeeze(-1)  # [batch, n_atoms]
        
        # Mask contributions
        if mask is not None:
            residue_contrib = residue_contrib * mask.float()
        
        return {
            'ddg': ddg_mean,
            'ddg_lower': ddg_lower,
            'ddg_upper': ddg_upper,
            'ddg_confidence': (ddg_lower_std + ddg_upper_std) / 2,
            'residue_contrib': residue_contrib,
            'attention_weights': attention_weights
        }


class PearlWithDDG(nn.Module):
    """
    Extended Pearl model with ΔΔG prediction capability.
    
    This model can:
    1. Predict structures (like base Pearl)
    2. Predict ΔΔG upon mutations
    3. Provide uncertainty estimates
    4. Identify key residues driving ΔΔG
    """
    
    def __init__(
        self,
        base_pearl: Pearl,
        freeze_pearl: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Base Pearl model
        self.pearl = base_pearl
        self.pair_dim = base_pearl.pair_dim
        
        # ΔΔG prediction head
        self.ddg_head = DDGPredictionHead(
            pair_dim=self.pair_dim,
            hidden_dim=512,
            dropout=dropout
        )
        
        # Whether to freeze Pearl weights during ΔΔG training
        self.set_freeze_pearl(freeze_pearl)
        
    def set_freeze_pearl(self, freeze: bool):
        """Freeze or unfreeze base Pearl parameters"""
        for param in self.pearl.parameters():
            param.requires_grad = not freeze
            
    def forward_structure(
        self,
        protein_features: torch.Tensor,
        ligand_features: torch.Tensor,
        protein_mask=None,
        ligand_mask=None
    ) -> torch.Tensor:
        """
        Forward pass for getting pair representation from trunk.

        Returns:
            pair_repr: Pair representation from trunk (for ΔΔG prediction)
        """
        # Use the Pearl model's forward method to get pair representation
        pair_repr = self.pearl(
            protein_features=protein_features,
            ligand_features=ligand_features,
            protein_mask=protein_mask,
            ligand_mask=ligand_mask
        )

        return pair_repr
    
    def forward(
        self,
        wt_protein_features: torch.Tensor,
        wt_ligand_features: torch.Tensor,
        mut_protein_features: torch.Tensor,
        mut_ligand_features: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None,
        ligand_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for ΔΔG prediction.

        Args:
            wt_protein_features: Wild-type protein features
            wt_ligand_features: Wild-type ligand features
            mut_protein_features: Mutant protein features
            mut_ligand_features: Mutant ligand features
            protein_mask: Mask for valid protein atoms
            ligand_mask: Mask for valid ligand atoms

        Returns:
            Dictionary with ΔΔG predictions
        """
        # Get pair representations from trunk
        wt_pair = self.forward_structure(
            wt_protein_features, wt_ligand_features,
            protein_mask=protein_mask, ligand_mask=ligand_mask
        )

        mut_pair = self.forward_structure(
            mut_protein_features, mut_ligand_features,
            protein_mask=protein_mask, ligand_mask=ligand_mask
        )

        # Combine masks
        mask = torch.cat([protein_mask, ligand_mask], dim=1) if protein_mask is not None else None

        # Predict ΔΔG
        ddg_output = self.ddg_head(wt_pair, mut_pair, mask=mask)

        result = {
            'ddg': ddg_output['ddg'],
            'ddg_lower': ddg_output['ddg_lower'],
            'ddg_upper': ddg_output['ddg_upper'],
            'ddg_confidence': ddg_output['ddg_confidence'],
            'residue_contrib': ddg_output['residue_contrib'],
            'attention_weights': ddg_output['attention_weights'],
        }

        return result
    
    @torch.no_grad()
    def predict_ddg(
        self,
        wt_protein_features: torch.Tensor,
        wt_ligand_features: torch.Tensor,
        mut_protein_features: torch.Tensor,
        mut_ligand_features: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None,
        ligand_mask: Optional[torch.Tensor] = None,
        num_samples: int = 5
    ) -> DDGPrediction:
        """
        Predict ΔΔG for a single mutation with uncertainty quantification.

        Args:
            wt_protein_features: Wild-type protein features
            wt_ligand_features: Wild-type ligand features
            mut_protein_features: Mutant protein features
            mut_ligand_features: Mutant ligand features
            protein_mask: Mask for valid protein atoms
            ligand_mask: Mask for valid ligand atoms
            num_samples: Number of samples for ensemble prediction

        Returns:
            DDGPrediction object with results
        """
        self.eval()

        # Ensemble prediction for better uncertainty
        ddg_samples = []
        confidence_samples = []

        for _ in range(num_samples):
            # Forward pass
            output = self.forward(
                wt_protein_features, wt_ligand_features,
                mut_protein_features, mut_ligand_features,
                protein_mask=protein_mask, ligand_mask=ligand_mask
            )

            ddg_samples.append(output['ddg'])
            confidence_samples.append(output['ddg_confidence'])

        # Aggregate predictions
        ddg_samples = torch.stack(ddg_samples)
        ddg_mean = ddg_samples.mean(dim=0)
        ddg_std = ddg_samples.std(dim=0)

        # Use ensemble std as additional uncertainty
        confidence_mean = torch.stack(confidence_samples).mean(dim=0)
        total_confidence = torch.sqrt(confidence_mean**2 + ddg_std**2)

        return DDGPrediction(
            ddg=ddg_mean.item(),
            ddg_lower=(ddg_mean - total_confidence).item(),
            ddg_upper=(ddg_mean + total_confidence).item(),
            confidence=total_confidence.item(),
            residue_contributions=output['residue_contrib'][0],
            wt_coords=None,  # Not computed in this simplified version
            mut_coords=None,  # Not computed in this simplified version
            attention_weights=output['attention_weights'][0]
        )

