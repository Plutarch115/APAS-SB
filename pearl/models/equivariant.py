"""
SO(3)-Equivariant Transformer Blocks for Pearl

Implements the equivariant transformer (EqT) blocks with:
- Equivariant self-attention
- Equivariant transition (feed-forward) layers
- Gated nonlinearities for vector components

Based on Figure 2 of the Pearl technical report.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class EquivariantSelfAttention(nn.Module):
    """
    SO(3)-equivariant self-attention module.
    
    Processes scalar and vector features while maintaining equivariance
    to 3D rotations.
    """
    
    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.num_heads = num_heads
        self.head_dim = scalar_dim // num_heads
        
        assert scalar_dim % num_heads == 0, "scalar_dim must be divisible by num_heads"
        
        # Scalar query, key, value projections
        self.scalar_q = nn.Linear(scalar_dim, scalar_dim)
        self.scalar_k = nn.Linear(scalar_dim, scalar_dim)
        self.scalar_v = nn.Linear(scalar_dim, scalar_dim)
        
        # Vector query, key projections (no value for vectors in attention)
        self.vector_q = nn.Linear(vector_dim, vector_dim)
        self.vector_k = nn.Linear(vector_dim, vector_dim)
        
        # Output projections
        self.scalar_out = nn.Linear(scalar_dim, scalar_dim)
        self.vector_out = nn.Linear(vector_dim, vector_dim)
        
        # Pair representation for attention bias
        self.pair_bias = nn.Linear(scalar_dim, num_heads)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self,
        scalar_features: torch.Tensor,  # [batch, n_atoms, scalar_dim]
        vector_features: torch.Tensor,  # [batch, n_atoms, vector_dim, 3]
        pair_repr: Optional[torch.Tensor] = None  # [batch, n_atoms, n_atoms, pair_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            scalar_features: Scalar node features
            vector_features: Vector node features (3D vectors)
            pair_repr: Optional pairwise representation for attention bias
            
        Returns:
            Updated scalar and vector features
        """
        batch_size, n_atoms, _ = scalar_features.shape
        
        # Scalar attention
        q_scalar = self.scalar_q(scalar_features).view(batch_size, n_atoms, self.num_heads, self.head_dim)
        k_scalar = self.scalar_k(scalar_features).view(batch_size, n_atoms, self.num_heads, self.head_dim)
        v_scalar = self.scalar_v(scalar_features).view(batch_size, n_atoms, self.num_heads, self.head_dim)
        
        # Compute attention weights from scalar features
        # [batch, num_heads, n_atoms, n_atoms]
        attn_weights = torch.einsum('bqhd,bkhd->bhqk', q_scalar, k_scalar) * self.scale
        
        # Add pair representation bias if provided
        if pair_repr is not None:
            pair_bias = self.pair_bias(pair_repr)  # [batch, n_atoms, n_atoms, num_heads]
            pair_bias = pair_bias.permute(0, 3, 1, 2)  # [batch, num_heads, n_atoms, n_atoms]
            attn_weights = attn_weights + pair_bias
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to scalar values
        scalar_out = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v_scalar)
        scalar_out = scalar_out.reshape(batch_size, n_atoms, self.scalar_dim)
        scalar_out = self.scalar_out(scalar_out)
        
        # Vector attention: apply same attention weights to vector features
        # This maintains equivariance
        vector_out = torch.einsum('bhqk,bkvc->bqhvc', attn_weights, 
                                  vector_features.view(batch_size, n_atoms, self.num_heads, -1, 3))
        vector_out = vector_out.reshape(batch_size, n_atoms, self.vector_dim, 3)
        vector_out = self.vector_out(vector_out.transpose(-2, -1)).transpose(-2, -1)
        
        return scalar_out, vector_out


class EquivariantTransition(nn.Module):
    """
    Equivariant feed-forward transition layer with gated nonlinearity.
    
    Uses gated activation for vector components to maintain equivariance.
    """
    
    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.hidden_dim = hidden_dim or scalar_dim * 4
        
        # Scalar pathway
        self.scalar_fc1 = nn.Linear(scalar_dim, self.hidden_dim)
        self.scalar_fc2 = nn.Linear(self.hidden_dim, scalar_dim)
        
        # Vector pathway with gating
        self.vector_fc1 = nn.Linear(vector_dim, self.hidden_dim)
        self.vector_gate = nn.Linear(vector_dim, self.hidden_dim)  # Gate for vectors
        self.vector_fc2 = nn.Linear(self.hidden_dim, vector_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        scalar_features: torch.Tensor,  # [batch, n_atoms, scalar_dim]
        vector_features: torch.Tensor   # [batch, n_atoms, vector_dim, 3]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply equivariant feed-forward transformation.
        
        Args:
            scalar_features: Scalar node features
            vector_features: Vector node features
            
        Returns:
            Updated scalar and vector features
        """
        # Scalar pathway with standard feed-forward
        scalar_out = self.scalar_fc1(scalar_features)
        scalar_out = F.gelu(scalar_out)
        scalar_out = self.dropout(scalar_out)
        scalar_out = self.scalar_fc2(scalar_out)
        
        # Vector pathway with gated nonlinearity
        # Compute vector norms for gating (invariant to rotation)
        vector_norms = torch.norm(vector_features, dim=-1, keepdim=True)  # [batch, n_atoms, vector_dim, 1]
        
        # Transform vectors
        v_transformed = self.vector_fc1(vector_features.transpose(-2, -1)).transpose(-2, -1)
        
        # Compute gates from vector norms (scalar values)
        gates = self.vector_gate(vector_norms.squeeze(-1))  # [batch, n_atoms, hidden_dim]
        gates = torch.sigmoid(gates).unsqueeze(-1)  # [batch, n_atoms, hidden_dim, 1]
        
        # Apply gated nonlinearity
        v_gated = v_transformed * gates
        v_gated = self.dropout(v_gated)
        
        # Final projection
        vector_out = self.vector_fc2(v_gated.transpose(-2, -1)).transpose(-2, -1)
        
        return scalar_out, vector_out


class EquivariantTransformerBlock(nn.Module):
    """
    Complete equivariant transformer block combining attention and transition.
    
    This is the core building block of Pearl's diffusion module.
    """
    
    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        num_heads: int = 8,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = EquivariantSelfAttention(
            scalar_dim=scalar_dim,
            vector_dim=vector_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.transition = EquivariantTransition(
            scalar_dim=scalar_dim,
            vector_dim=vector_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Layer normalization for scalars
        self.norm1_scalar = nn.LayerNorm(scalar_dim)
        self.norm2_scalar = nn.LayerNorm(scalar_dim)
        
        # For vectors, we normalize by their norms
        self.norm1_vector = VectorLayerNorm(vector_dim)
        self.norm2_vector = VectorLayerNorm(vector_dim)
        
    def forward(
        self,
        scalar_features: torch.Tensor,
        vector_features: torch.Tensor,
        pair_repr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the equivariant transformer block.
        
        Args:
            scalar_features: [batch, n_atoms, scalar_dim]
            vector_features: [batch, n_atoms, vector_dim, 3]
            pair_repr: Optional [batch, n_atoms, n_atoms, pair_dim]
            
        Returns:
            Updated scalar and vector features
        """
        # Attention with residual
        scalar_attn, vector_attn = self.attention(
            scalar_features, vector_features, pair_repr
        )
        scalar_features = self.norm1_scalar(scalar_features + scalar_attn)
        vector_features = self.norm1_vector(vector_features + vector_attn)
        
        # Transition with residual
        scalar_trans, vector_trans = self.transition(scalar_features, vector_features)
        scalar_features = self.norm2_scalar(scalar_features + scalar_trans)
        vector_features = self.norm2_vector(vector_features + vector_trans)
        
        return scalar_features, vector_features


class VectorLayerNorm(nn.Module):
    """Layer normalization for vector features that maintains equivariance."""
    
    def __init__(self, vector_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(vector_dim))
        
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vectors: [batch, n_atoms, vector_dim, 3]
        Returns:
            Normalized vectors
        """
        # Compute norms
        norms = torch.norm(vectors, dim=-1, keepdim=True)  # [batch, n_atoms, vector_dim, 1]
        
        # Normalize
        mean_norm = norms.mean(dim=-2, keepdim=True)
        std_norm = norms.std(dim=-2, keepdim=True)
        
        normalized_norms = (norms - mean_norm) / (std_norm + self.eps)
        
        # Scale and maintain direction
        scale = self.scale.view(1, 1, -1, 1)
        normalized_vectors = vectors / (norms + self.eps) * normalized_norms * scale
        
        return normalized_vectors

