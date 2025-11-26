"""
Multi-Chain Templating System for Pearl

Implements Pearl's novel multi-chain templating that supports both protein
and non-polymeric components (ligands, cofactors).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Template:
    """
    Represents a structural template for cofolding.
    
    Attributes:
        protein_coords: Protein atom coordinates [n_protein_atoms, 3]
        protein_features: Protein atom features [n_protein_atoms, feat_dim]
        ligand_coords: Optional ligand coordinates [n_ligand_atoms, 3]
        ligand_features: Optional ligand features [n_ligand_atoms, feat_dim]
        cofactor_coords: Optional cofactor coordinates [n_cofactor_atoms, 3]
        cofactor_features: Optional cofactor features [n_cofactor_atoms, feat_dim]
        confidence: Template confidence score
    """
    protein_coords: torch.Tensor
    protein_features: torch.Tensor
    ligand_coords: Optional[torch.Tensor] = None
    ligand_features: Optional[torch.Tensor] = None
    cofactor_coords: Optional[torch.Tensor] = None
    cofactor_features: Optional[torch.Tensor] = None
    confidence: float = 1.0


class TemplateEmbedding(nn.Module):
    """
    Embeds template structures into pair representations.
    
    Supports multi-chain templates including non-polymeric components.
    """
    
    def __init__(
        self,
        pair_dim: int = 128,
        num_bins: int = 64,
        min_dist: float = 2.0,
        max_dist: float = 22.0
    ):
        super().__init__()
        
        self.pair_dim = pair_dim
        self.num_bins = num_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        
        # Distance bins for template distances
        self.register_buffer(
            'distance_bins',
            torch.linspace(min_dist, max_dist, num_bins)
        )
        
        # Embeddings for different template components
        self.protein_template_embedding = nn.Linear(num_bins, pair_dim)
        self.ligand_template_embedding = nn.Linear(num_bins, pair_dim)
        self.cofactor_template_embedding = nn.Linear(num_bins, pair_dim)
        
        # Cross-component embeddings (protein-ligand, protein-cofactor, etc.)
        self.cross_component_embedding = nn.Linear(num_bins, pair_dim)
        
        # Confidence weighting
        self.confidence_projection = nn.Linear(1, pair_dim)
        
    def _compute_distance_features(
        self,
        coords1: torch.Tensor,
        coords2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute binned distance features between two sets of coordinates.
        
        Args:
            coords1: [n_atoms1, 3]
            coords2: [n_atoms2, 3]
            
        Returns:
            Distance features [n_atoms1, n_atoms2, num_bins]
        """
        # Compute pairwise distances
        diff = coords1.unsqueeze(1) - coords2.unsqueeze(0)  # [n_atoms1, n_atoms2, 3]
        distances = torch.norm(diff, dim=-1)  # [n_atoms1, n_atoms2]
        
        # Bin distances using RBF-like encoding
        distances = distances.unsqueeze(-1)  # [n_atoms1, n_atoms2, 1]
        distance_bins = self.distance_bins.view(1, 1, -1)  # [1, 1, num_bins]
        
        # Gaussian RBF
        sigma = (self.max_dist - self.min_dist) / self.num_bins
        distance_features = torch.exp(-((distances - distance_bins) ** 2) / (2 * sigma ** 2))
        
        return distance_features
    
    def forward(
        self,
        template: Template,
        target_protein_mask: torch.Tensor,
        target_ligand_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Embed template into pair representation.
        
        Args:
            template: Template structure
            target_protein_mask: Mask for target protein atoms [n_protein_atoms]
            target_ligand_mask: Optional mask for target ligand atoms [n_ligand_atoms]
            
        Returns:
            Template pair representation [n_total_atoms, n_total_atoms, pair_dim]
        """
        device = template.protein_coords.device
        n_protein = template.protein_coords.shape[0]
        
        # Initialize pair representation
        total_atoms = n_protein
        if template.ligand_coords is not None:
            total_atoms += template.ligand_coords.shape[0]
        if template.cofactor_coords is not None:
            total_atoms += template.cofactor_coords.shape[0]
        
        pair_repr = torch.zeros(total_atoms, total_atoms, self.pair_dim, device=device)
        
        # Protein-protein distances
        protein_dist_features = self._compute_distance_features(
            template.protein_coords, template.protein_coords
        )
        protein_pair = self.protein_template_embedding(protein_dist_features)
        pair_repr[:n_protein, :n_protein] = protein_pair
        
        offset = n_protein
        
        # Protein-ligand distances
        if template.ligand_coords is not None:
            n_ligand = template.ligand_coords.shape[0]
            
            # Ligand-ligand
            ligand_dist_features = self._compute_distance_features(
                template.ligand_coords, template.ligand_coords
            )
            ligand_pair = self.ligand_template_embedding(ligand_dist_features)
            pair_repr[offset:offset+n_ligand, offset:offset+n_ligand] = ligand_pair
            
            # Protein-ligand cross terms
            cross_dist_features = self._compute_distance_features(
                template.protein_coords, template.ligand_coords
            )
            cross_pair = self.cross_component_embedding(cross_dist_features)
            pair_repr[:n_protein, offset:offset+n_ligand] = cross_pair
            pair_repr[offset:offset+n_ligand, :n_protein] = cross_pair.transpose(0, 1)
            
            offset += n_ligand
        
        # Protein-cofactor distances
        if template.cofactor_coords is not None:
            n_cofactor = template.cofactor_coords.shape[0]
            
            # Cofactor-cofactor
            cofactor_dist_features = self._compute_distance_features(
                template.cofactor_coords, template.cofactor_coords
            )
            cofactor_pair = self.cofactor_template_embedding(cofactor_dist_features)
            pair_repr[offset:offset+n_cofactor, offset:offset+n_cofactor] = cofactor_pair
            
            # Protein-cofactor cross terms
            cross_dist_features = self._compute_distance_features(
                template.protein_coords, template.cofactor_coords
            )
            cross_pair = self.cross_component_embedding(cross_dist_features)
            pair_repr[:n_protein, offset:offset+n_cofactor] = cross_pair
            pair_repr[offset:offset+n_cofactor, :n_protein] = cross_pair.transpose(0, 1)
        
        # Weight by confidence
        confidence_weight = self.confidence_projection(
            torch.tensor([[template.confidence]], device=device)
        )
        pair_repr = pair_repr * torch.sigmoid(confidence_weight)
        
        return pair_repr


class TemplateRetrieval(nn.Module):
    """
    Retrieves relevant templates from a database.
    
    Supports both sequence-based and structure-based retrieval.
    """
    
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Sequence encoder for retrieval
        self.sequence_encoder = nn.LSTM(
            input_size=20,  # 20 amino acids
            hidden_size=embedding_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Projection to embedding space
        self.projection = nn.Linear(embedding_dim * 2, embedding_dim)
        
    def encode_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Encode protein sequence to embedding.
        
        Args:
            sequence: One-hot encoded sequence [batch, seq_len, 20]
            
        Returns:
            Sequence embedding [batch, embedding_dim]
        """
        output, (hidden, _) = self.sequence_encoder(sequence)
        
        # Use final hidden states from both directions
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        embedding = self.projection(hidden)
        
        return embedding
    
    def retrieve_templates(
        self,
        query_sequence: torch.Tensor,
        template_database: List[Template],
        top_k: int = 5
    ) -> List[Template]:
        """
        Retrieve top-k most similar templates.
        
        Args:
            query_sequence: Query protein sequence
            template_database: List of available templates
            top_k: Number of templates to retrieve
            
        Returns:
            List of top-k templates
        """
        # Encode query
        query_embedding = self.encode_sequence(query_sequence)
        
        # Compute similarities (simplified - in practice would use more sophisticated matching)
        # This is a placeholder for actual template retrieval logic
        
        # For now, return top_k templates by confidence
        sorted_templates = sorted(
            template_database,
            key=lambda t: t.confidence,
            reverse=True
        )
        
        return sorted_templates[:top_k]


class MultiChainTemplateStack(nn.Module):
    """
    Processes multiple templates and combines them into a single representation.
    """
    
    def __init__(self, pair_dim: int = 128, max_templates: int = 4):
        super().__init__()
        
        self.pair_dim = pair_dim
        self.max_templates = max_templates
        
        # Template embedding
        self.template_embedding = TemplateEmbedding(pair_dim)
        
        # Attention over templates
        self.template_attention = nn.MultiheadAttention(
            embed_dim=pair_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(pair_dim, pair_dim)
        
    def forward(
        self,
        templates: List[Template],
        target_protein_mask: torch.Tensor,
        target_ligand_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process multiple templates into combined pair representation.
        
        Args:
            templates: List of template structures
            target_protein_mask: Mask for target protein
            target_ligand_mask: Optional mask for target ligand
            
        Returns:
            Combined template pair representation
        """
        if not templates:
            # Return zero representation if no templates
            n_atoms = target_protein_mask.shape[0]
            if target_ligand_mask is not None:
                n_atoms += target_ligand_mask.shape[0]
            return torch.zeros(n_atoms, n_atoms, self.pair_dim, device=target_protein_mask.device)
        
        # Embed each template
        template_reprs = []
        for template in templates[:self.max_templates]:
            template_repr = self.template_embedding(
                template, target_protein_mask, target_ligand_mask
            )
            template_reprs.append(template_repr)
        
        # Stack templates
        template_stack = torch.stack(template_reprs, dim=0)  # [n_templates, n_atoms, n_atoms, pair_dim]
        
        # Average over templates (could use attention instead)
        combined_repr = template_stack.mean(dim=0)
        
        return combined_repr

