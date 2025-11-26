"""
Mock Pearl model for testing ΔΔG prediction.

Simplified version that only implements the trunk module for pair representation.
"""

import torch
import torch.nn as nn
from typing import Optional


class MockPearl(nn.Module):
    """
    Simplified Pearl model for ΔΔG testing.
    
    Only implements the trunk module to generate pair representations.
    Does not include diffusion or templating.
    """
    
    def __init__(
        self,
        protein_feature_dim: int = 64,
        ligand_feature_dim: int = 64,
        pair_dim: int = 128,
        trunk_blocks: int = 4,
        trunk_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.protein_feature_dim = protein_feature_dim
        self.ligand_feature_dim = ligand_feature_dim
        self.pair_dim = pair_dim
        
        # Input embeddings
        self.protein_embedding = nn.Sequential(
            nn.Linear(protein_feature_dim, pair_dim),
            nn.LayerNorm(pair_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.ligand_embedding = nn.Sequential(
            nn.Linear(ligand_feature_dim, pair_dim),
            nn.LayerNorm(pair_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Trunk module (simplified)
        trunk_layers = []
        for _ in range(trunk_blocks):
            trunk_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=pair_dim,
                    nhead=trunk_heads,
                    dim_feedforward=pair_dim * 4,
                    dropout=dropout,
                    batch_first=True
                )
            )
        self.trunk = nn.Sequential(*trunk_layers)
        
    def _compute_cross_pair(
        self,
        protein_features: torch.Tensor,
        ligand_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute protein-ligand cross-pair features.
        
        Args:
            protein_features: [batch, n_protein, protein_feat_dim]
            ligand_features: [batch, n_ligand, ligand_feat_dim]
            
        Returns:
            cross_pair: [batch, n_protein, n_ligand, pair_dim]
        """
        batch_size = protein_features.shape[0]
        n_protein = protein_features.shape[1]
        n_ligand = ligand_features.shape[1]
        
        # Embed features
        protein_emb = self.protein_embedding(protein_features)  # [batch, n_protein, pair_dim]
        ligand_emb = self.ligand_embedding(ligand_features)     # [batch, n_ligand, pair_dim]
        
        # Outer product for cross-pair
        protein_expanded = protein_emb.unsqueeze(2)  # [batch, n_protein, 1, pair_dim]
        ligand_expanded = ligand_emb.unsqueeze(1)    # [batch, 1, n_ligand, pair_dim]
        
        # Simple addition (could be more sophisticated)
        cross_pair = protein_expanded + ligand_expanded  # [batch, n_protein, n_ligand, pair_dim]
        
        return cross_pair
    
    def forward(
        self,
        protein_features: torch.Tensor,
        ligand_features: Optional[torch.Tensor] = None,
        protein_mask: Optional[torch.Tensor] = None,
        ligand_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass to get pair representation.

        Args:
            protein_features: [batch, n_protein, protein_feat_dim]
            ligand_features: [batch, n_ligand, ligand_feat_dim] (optional)
            protein_mask: [batch, n_protein]
            ligand_mask: [batch, n_ligand]

        Returns:
            pair_repr: [batch, n_atoms, n_atoms, pair_dim]
        """
        batch_size = protein_features.shape[0]
        n_protein = protein_features.shape[1]

        # Embed protein features
        protein_emb = self.protein_embedding(protein_features)  # [batch, n_protein, pair_dim]

        # Handle optional ligand features
        if ligand_features is not None:
            n_ligand = ligand_features.shape[1]
            ligand_emb = self.ligand_embedding(ligand_features)  # [batch, n_ligand, pair_dim]
            # Combine into full sequence
            full_emb = torch.cat([protein_emb, ligand_emb], dim=1)  # [batch, n_atoms, pair_dim]
        else:
            # Protein-only (for protein-protein interactions)
            full_emb = protein_emb

        # Process through trunk
        # Note: trunk expects [batch, seq_len, features]
        trunk_out = full_emb
        for layer in self.trunk:
            trunk_out = layer(trunk_out)  # [batch, n_atoms, pair_dim]

        # Create pair representation via outer product
        # [batch, n_atoms, 1, pair_dim] + [batch, 1, n_atoms, pair_dim]
        pair_repr = trunk_out.unsqueeze(2) + trunk_out.unsqueeze(1)
        # Result: [batch, n_atoms, n_atoms, pair_dim]

        return pair_repr

