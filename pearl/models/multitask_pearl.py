"""
Multi-task PEARL model for predicting multiple biochemical properties.

Extends PEARL with task-specific prediction heads for:
- Binding affinity (protein-ligand)
- ΔΔG (protein-protein interactions)
- kcat (catalytic activity)
- Fitness scores (deep mutational scanning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import numpy as np


class BindingAffinityHead(nn.Module):
    """
    Prediction head for protein-ligand binding affinity.
    
    Predicts pKd, pKi, or pIC50 values from pair representations.
    """
    
    def __init__(
        self,
        pair_dim: int = 128,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        
        # Attention mechanism to focus on binding site
        self.attention = nn.MultiheadAttention(
            pair_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Global affinity prediction
        self.affinity_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # mean, lower, upper
        )
        
        # Per-residue contribution
        self.residue_contrib = nn.Linear(pair_dim, 1)
    
    def forward(
        self,
        pair_repr: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pair_repr: [batch, n_atoms, n_atoms, pair_dim]
            protein_mask: [batch, n_atoms]
        
        Returns:
            Dictionary with:
            - affinity: [batch] predicted binding affinity
            - confidence: [batch] prediction confidence
            - residue_contrib: [batch, n_atoms] per-residue contributions
        """
        batch_size, n_atoms, _, pair_dim = pair_repr.shape
        
        # Average over second dimension to get per-atom features
        atom_features = pair_repr.mean(dim=2)  # [batch, n_atoms, pair_dim]
        
        # Apply attention
        attn_out, attn_weights = self.attention(
            atom_features, atom_features, atom_features,
            key_padding_mask=~protein_mask if protein_mask is not None else None
        )
        
        # Global pooling
        if protein_mask is not None:
            mask_expanded = protein_mask.unsqueeze(-1).float()
            global_features = (attn_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            global_features = attn_out.mean(dim=1)
        
        # Predict affinity with confidence bounds
        affinity_pred = self.affinity_head(global_features)  # [batch, 3]
        affinity_mean = affinity_pred[:, 0]
        affinity_lower = affinity_pred[:, 1]
        affinity_upper = affinity_pred[:, 2]
        
        # Ensure confidence bounds are positive
        confidence = F.softplus(affinity_upper - affinity_lower) + 0.1
        
        # Per-residue contributions
        residue_contrib = self.residue_contrib(atom_features).squeeze(-1)  # [batch, n_atoms]
        
        return {
            'affinity': affinity_mean,
            'confidence': confidence,
            'lower_bound': affinity_mean - confidence,
            'upper_bound': affinity_mean + confidence,
            'residue_contrib': residue_contrib,
            'attention_weights': attn_weights
        }


class CatalyticActivityHead(nn.Module):
    """
    Prediction head for catalytic activity (kcat).
    
    Predicts log(kcat) from enzyme-substrate pair representations.
    """
    
    def __init__(
        self,
        pair_dim: int = 128,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        
        # Attention to focus on active site
        self.attention = nn.MultiheadAttention(
            pair_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # kcat prediction head
        self.kcat_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # log(kcat) mean, lower, upper
        )
        
        # Active site residue identification
        self.active_site_head = nn.Linear(pair_dim, 1)
    
    def forward(
        self,
        pair_repr: torch.Tensor,
        enzyme_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pair_repr: [batch, n_atoms, n_atoms, pair_dim]
            enzyme_mask: [batch, n_atoms]
        
        Returns:
            Dictionary with:
            - log_kcat: [batch] predicted log(kcat)
            - confidence: [batch] prediction confidence
            - active_site_scores: [batch, n_atoms] active site probabilities
        """
        batch_size, n_atoms, _, pair_dim = pair_repr.shape
        
        # Average over second dimension
        atom_features = pair_repr.mean(dim=2)  # [batch, n_atoms, pair_dim]
        
        # Apply attention
        attn_out, attn_weights = self.attention(
            atom_features, atom_features, atom_features,
            key_padding_mask=~enzyme_mask if enzyme_mask is not None else None
        )
        
        # Global pooling
        if enzyme_mask is not None:
            mask_expanded = enzyme_mask.unsqueeze(-1).float()
            global_features = (attn_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            global_features = attn_out.mean(dim=1)
        
        # Predict log(kcat) with confidence bounds
        kcat_pred = self.kcat_head(global_features)  # [batch, 3]
        log_kcat_mean = kcat_pred[:, 0]
        log_kcat_lower = kcat_pred[:, 1]
        log_kcat_upper = kcat_pred[:, 2]
        
        # Confidence
        confidence = F.softplus(log_kcat_upper - log_kcat_lower) + 0.1
        
        # Active site scores
        active_site_scores = torch.sigmoid(self.active_site_head(atom_features).squeeze(-1))
        
        return {
            'log_kcat': log_kcat_mean,
            'confidence': confidence,
            'lower_bound': log_kcat_mean - confidence,
            'upper_bound': log_kcat_mean + confidence,
            'active_site_scores': active_site_scores,
            'attention_weights': attn_weights
        }


class FitnessScoreHead(nn.Module):
    """
    Prediction head for fitness scores from deep mutational scanning.
    
    Predicts normalized fitness scores from wild-type vs mutant comparisons.
    """
    
    def __init__(
        self,
        pair_dim: int = 128,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            pair_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Fitness prediction head
        self.fitness_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # fitness mean, lower, upper
        )
        
        # Mutation impact scores
        self.mutation_impact = nn.Linear(pair_dim, 1)
    
    def forward(
        self,
        pair_repr: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pair_repr: [batch, n_atoms, n_atoms, pair_dim]
            protein_mask: [batch, n_atoms]
        
        Returns:
            Dictionary with:
            - fitness: [batch] predicted fitness score
            - confidence: [batch] prediction confidence
            - mutation_impact: [batch, n_atoms] per-residue mutation impact
        """
        batch_size, n_atoms, _, pair_dim = pair_repr.shape
        
        # Average over second dimension
        atom_features = pair_repr.mean(dim=2)  # [batch, n_atoms, pair_dim]
        
        # Apply attention
        attn_out, attn_weights = self.attention(
            atom_features, atom_features, atom_features,
            key_padding_mask=~protein_mask if protein_mask is not None else None
        )
        
        # Global pooling
        if protein_mask is not None:
            mask_expanded = protein_mask.unsqueeze(-1).float()
            global_features = (attn_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            global_features = attn_out.mean(dim=1)
        
        # Predict fitness with confidence bounds
        fitness_pred = self.fitness_head(global_features)  # [batch, 3]
        fitness_mean = fitness_pred[:, 0]
        fitness_lower = fitness_pred[:, 1]
        fitness_upper = fitness_pred[:, 2]
        
        # Confidence
        confidence = F.softplus(fitness_upper - fitness_lower) + 0.1
        
        # Mutation impact scores
        mutation_impact = self.mutation_impact(atom_features).squeeze(-1)
        
        return {
            'fitness': fitness_mean,
            'confidence': confidence,
            'lower_bound': fitness_mean - confidence,
            'upper_bound': fitness_mean + confidence,
            'mutation_impact': mutation_impact,
            'attention_weights': attn_weights
        }


class MultiTaskPEARL(nn.Module):
    """
    PEARL extended for multiple prediction tasks.
    
    Supports:
    - binding_affinity: Protein-ligand binding (pKd, pKi, pIC50)
    - ddg_ppi: Protein-protein interaction ΔΔG
    - kcat: Catalytic activity
    - fitness: Deep mutational scanning fitness scores
    """
    
    def __init__(
        self,
        base_pearl: nn.Module,
        pair_dim: int = 128,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_pearl: bool = True
    ):
        super().__init__()
        
        self.pearl = base_pearl
        self.pair_dim = pair_dim
        
        # Freeze base PEARL if requested
        if freeze_pearl:
            for param in self.pearl.parameters():
                param.requires_grad = False
        
        # Task-specific prediction heads
        self.binding_head = BindingAffinityHead(pair_dim, hidden_dim, num_heads, dropout)
        self.kcat_head = CatalyticActivityHead(pair_dim, hidden_dim, num_heads, dropout)
        self.fitness_head = FitnessScoreHead(pair_dim, hidden_dim, num_heads, dropout)

        # Import DDG head from existing implementation
        from pearl.models.ddg_predictor import DDGPredictionHead
        self.ddg_head = DDGPredictionHead(pair_dim, hidden_dim, dropout)
    
    def forward_structure(
        self,
        protein_features: torch.Tensor,
        ligand_features: Optional[torch.Tensor] = None,
        protein_mask: Optional[torch.Tensor] = None,
        ligand_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get pair representation from PEARL.
        
        Args:
            protein_features: [batch, n_protein, feature_dim]
            ligand_features: [batch, n_ligand, feature_dim] (optional)
            protein_mask: [batch, n_protein]
            ligand_mask: [batch, n_ligand]
        
        Returns:
            pair_repr: [batch, n_atoms, n_atoms, pair_dim]
        """
        pair_repr = self.pearl(
            protein_features=protein_features,
            ligand_features=ligand_features,
            protein_mask=protein_mask,
            ligand_mask=ligand_mask
        )
        return pair_repr
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        task: str
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a specific task.
        
        Args:
            batch: Dictionary containing task-specific inputs
            task: One of ['binding_affinity', 'ddg_ppi', 'kcat', 'fitness']
        
        Returns:
            Dictionary with task-specific predictions
        """
        if task == 'binding_affinity':
            # Get pair representation
            pair_repr = self.forward_structure(
                protein_features=batch['protein_features'],
                ligand_features=batch['ligand_features'],
                protein_mask=batch.get('protein_mask'),
                ligand_mask=batch.get('ligand_mask')
            )
            return self.binding_head(pair_repr, batch.get('protein_mask'))
        
        elif task == 'ddg_ppi':
            # Get pair representations for wild-type and mutant
            wt_pair_repr = self.forward_structure(
                protein_features=batch['wt_protein_features'],
                protein_mask=batch.get('protein_mask')
            )
            mut_pair_repr = self.forward_structure(
                protein_features=batch['mut_protein_features'],
                protein_mask=batch.get('protein_mask')
            )
            return self.ddg_head(wt_pair_repr, mut_pair_repr, batch.get('protein_mask'))
        
        elif task == 'kcat':
            # Get pair representation for enzyme-substrate
            pair_repr = self.forward_structure(
                protein_features=batch['enzyme_features'],
                ligand_features=batch['substrate_features'],
                protein_mask=batch.get('enzyme_mask'),
                ligand_mask=batch.get('substrate_mask')
            )
            return self.kcat_head(pair_repr, batch.get('enzyme_mask'))
        
        elif task == 'fitness':
            # Get pair representations for wild-type and mutant
            wt_pair_repr = self.forward_structure(
                protein_features=batch['wt_protein_features'],
                protein_mask=batch.get('protein_mask')
            )
            mut_pair_repr = self.forward_structure(
                protein_features=batch['mut_protein_features'],
                protein_mask=batch.get('protein_mask')
            )
            # Use difference for fitness prediction
            diff_pair_repr = mut_pair_repr - wt_pair_repr
            return self.fitness_head(diff_pair_repr, batch.get('protein_mask'))
        
        else:
            raise ValueError(f"Unknown task: {task}")

