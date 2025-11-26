"""
Pearl: Placing Every Atom in the Right Location

Main model implementation combining all components:
- Trunk module with triangle multiplication
- SO(3)-equivariant diffusion module
- Multi-chain templating system
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple

from .trunk import TrunkModule
from .diffusion import DiffusionModule
from .templating import MultiChainTemplateStack, Template


class Pearl(nn.Module):
    """
    Pearl foundation model for protein-ligand cofolding.
    
    Achieves state-of-the-art performance through:
    1. Large-scale synthetic data training
    2. SO(3)-equivariant diffusion module
    3. Multi-chain templating system
    """
    
    def __init__(
        self,
        # Input dimensions
        protein_feature_dim: int = 64,
        ligand_feature_dim: int = 64,
        
        # Trunk parameters
        pair_dim: int = 128,
        trunk_blocks: int = 4,
        trunk_heads: int = 8,
        
        # Diffusion parameters
        scalar_dim: int = 256,
        vector_dim: int = 64,
        diffusion_blocks: int = 8,
        diffusion_heads: int = 8,
        num_diffusion_steps: int = 200,
        
        # Templating parameters
        max_templates: int = 4,
        
        # Training parameters
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.protein_feature_dim = protein_feature_dim
        self.ligand_feature_dim = ligand_feature_dim
        self.pair_dim = pair_dim
        
        # Input embeddings
        self.protein_embedding = ProteinEmbedding(protein_feature_dim, pair_dim)
        self.ligand_embedding = LigandEmbedding(ligand_feature_dim, pair_dim)
        
        # Multi-chain templating
        self.template_stack = MultiChainTemplateStack(pair_dim, max_templates)
        
        # Trunk module
        self.trunk = TrunkModule(
            pair_dim=pair_dim,
            num_blocks=trunk_blocks,
            num_heads=trunk_heads,
            dropout=dropout
        )
        
        # Diffusion module
        self.diffusion = DiffusionModule(
            scalar_dim=scalar_dim,
            vector_dim=vector_dim,
            pair_dim=pair_dim,
            num_blocks=diffusion_blocks,
            num_heads=diffusion_heads,
            num_diffusion_steps=num_diffusion_steps,
            dropout=dropout
        )
        
    def forward(
        self,
        protein_features: torch.Tensor,  # [batch, n_protein, protein_feat_dim]
        ligand_features: torch.Tensor,   # [batch, n_ligand, ligand_feat_dim]
        positions: torch.Tensor,          # [batch, n_atoms, 3]
        timestep: torch.Tensor,           # [batch]
        templates: Optional[List[Template]] = None,
        protein_mask: Optional[torch.Tensor] = None,
        ligand_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            protein_features: Protein sequence/structure features
            ligand_features: Ligand topology features
            positions: Current (noisy) 3D coordinates
            timestep: Diffusion timestep
            templates: Optional structural templates
            protein_mask: Mask for valid protein atoms
            ligand_mask: Mask for valid ligand atoms
            
        Returns:
            Predicted coordinate updates
        """
        batch_size = protein_features.shape[0]
        n_protein = protein_features.shape[1]
        n_ligand = ligand_features.shape[1]
        n_atoms = n_protein + n_ligand
        
        # Embed protein and ligand features into pair representation
        protein_pair = self.protein_embedding(protein_features)
        ligand_pair = self.ligand_embedding(ligand_features)
        
        # Combine into full pair representation
        pair_repr = torch.zeros(
            batch_size, n_atoms, n_atoms, self.pair_dim,
            device=protein_features.device
        )
        
        # Protein-protein block
        pair_repr[:, :n_protein, :n_protein] = protein_pair
        
        # Ligand-ligand block
        pair_repr[:, n_protein:, n_protein:] = ligand_pair
        
        # Protein-ligand cross terms (initialize with learned embeddings)
        cross_pair = self._compute_cross_pair(protein_features, ligand_features)
        pair_repr[:, :n_protein, n_protein:] = cross_pair
        pair_repr[:, n_protein:, :n_protein] = cross_pair.transpose(1, 2)
        
        # Add template information if provided
        if templates is not None:
            for i, template_list in enumerate(templates):
                if template_list:
                    template_repr = self.template_stack(
                        template_list,
                        protein_mask[i] if protein_mask is not None else None,
                        ligand_mask[i] if ligand_mask is not None else None
                    )
                    pair_repr[i] = pair_repr[i] + template_repr
        
        # Process through trunk
        pair_repr = self.trunk(pair_repr)
        
        # Predict coordinate updates via diffusion
        coord_updates = self.diffusion(
            positions, pair_repr, timestep,
            mask=torch.cat([protein_mask, ligand_mask], dim=1) if protein_mask is not None else None
        )
        
        return coord_updates
    
    def _compute_cross_pair(
        self,
        protein_features: torch.Tensor,
        ligand_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute protein-ligand cross-pair representation.
        
        Args:
            protein_features: [batch, n_protein, protein_feat_dim]
            ligand_features: [batch, n_ligand, ligand_feat_dim]
            
        Returns:
            Cross-pair representation [batch, n_protein, n_ligand, pair_dim]
        """
        batch_size = protein_features.shape[0]
        n_protein = protein_features.shape[1]
        n_ligand = ligand_features.shape[1]
        
        # Simple outer product for cross terms
        protein_proj = self.protein_embedding.to_pair_dim(protein_features)
        ligand_proj = self.ligand_embedding.to_pair_dim(ligand_features)
        
        cross_pair = torch.einsum('bpi,blj->bplij', protein_proj, ligand_proj)
        cross_pair = cross_pair.reshape(batch_size, n_protein, n_ligand, -1)
        
        # Project to pair_dim if needed
        if cross_pair.shape[-1] != self.pair_dim:
            cross_pair = nn.Linear(cross_pair.shape[-1], self.pair_dim, device=cross_pair.device)(cross_pair)
        
        return cross_pair
    
    @torch.no_grad()
    def predict_structure(
        self,
        protein_features: torch.Tensor,
        ligand_features: torch.Tensor,
        templates: Optional[List[Template]] = None,
        protein_mask: Optional[torch.Tensor] = None,
        ligand_mask: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Predict 3D structure(s) via sampling.
        
        Args:
            protein_features: Protein features
            ligand_features: Ligand features
            templates: Optional structural templates
            protein_mask: Mask for valid protein atoms
            ligand_mask: Mask for valid ligand atoms
            num_samples: Number of structures to sample
            num_steps: Number of diffusion steps
            
        Returns:
            Sampled 3D coordinates [num_samples, n_atoms, 3]
        """
        batch_size = protein_features.shape[0]
        n_protein = protein_features.shape[1]
        n_ligand = ligand_features.shape[1]
        n_atoms = n_protein + n_ligand
        
        # Prepare pair representation (same as forward)
        protein_pair = self.protein_embedding(protein_features)
        ligand_pair = self.ligand_embedding(ligand_features)
        
        pair_repr = torch.zeros(
            batch_size, n_atoms, n_atoms, self.pair_dim,
            device=protein_features.device
        )
        pair_repr[:, :n_protein, :n_protein] = protein_pair
        pair_repr[:, n_protein:, n_protein:] = ligand_pair
        
        cross_pair = self._compute_cross_pair(protein_features, ligand_features)
        pair_repr[:, :n_protein, n_protein:] = cross_pair
        pair_repr[:, n_protein:, :n_protein] = cross_pair.transpose(1, 2)
        
        if templates is not None:
            for i, template_list in enumerate(templates):
                if template_list:
                    template_repr = self.template_stack(
                        template_list,
                        protein_mask[i] if protein_mask is not None else None,
                        ligand_mask[i] if ligand_mask is not None else None
                    )
                    pair_repr[i] = pair_repr[i] + template_repr
        
        # Process through trunk
        pair_repr = self.trunk(pair_repr)
        
        # Sample structures
        samples = []
        for _ in range(num_samples):
            positions = self.diffusion.sample(
                pair_repr, n_atoms,
                mask=torch.cat([protein_mask, ligand_mask], dim=1) if protein_mask is not None else None,
                num_steps=num_steps
            )
            samples.append(positions)
        
        return torch.stack(samples, dim=0)


class ProteinEmbedding(nn.Module):
    """Embeds protein features into pair representation."""
    
    def __init__(self, feature_dim: int, pair_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.pair_dim = pair_dim
        
        self.node_embedding = nn.Linear(feature_dim, pair_dim)
        self.pair_projection = nn.Linear(pair_dim * 2, pair_dim)
        
    def to_pair_dim(self, features: torch.Tensor) -> torch.Tensor:
        """Project features to pair dimension."""
        return self.node_embedding(features)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, n_protein, feature_dim]
        Returns:
            Pair representation [batch, n_protein, n_protein, pair_dim]
        """
        node_emb = self.node_embedding(features)
        
        # Outer sum for pair representation
        pair_repr = node_emb.unsqueeze(2) + node_emb.unsqueeze(1)
        
        return pair_repr


class LigandEmbedding(nn.Module):
    """Embeds ligand features into pair representation."""
    
    def __init__(self, feature_dim: int, pair_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.pair_dim = pair_dim
        
        self.node_embedding = nn.Linear(feature_dim, pair_dim)
        self.pair_projection = nn.Linear(pair_dim * 2, pair_dim)
        
    def to_pair_dim(self, features: torch.Tensor) -> torch.Tensor:
        """Project features to pair dimension."""
        return self.node_embedding(features)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, n_ligand, feature_dim]
        Returns:
            Pair representation [batch, n_ligand, n_ligand, pair_dim]
        """
        node_emb = self.node_embedding(features)
        
        # Outer sum for pair representation
        pair_repr = node_emb.unsqueeze(2) + node_emb.unsqueeze(1)
        
        return pair_repr

