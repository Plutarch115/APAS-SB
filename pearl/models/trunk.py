"""
Trunk Module with Triangle Multiplication for Pearl

Implements the lightweight trunk module that learns rich, position-independent
pairwise representations to condition the diffusion module.

Based on the Pearl technical report and inspired by AlphaFold's Pairformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class TriangleMultiplication(nn.Module):
    """
    Triangle multiplication module for updating pairwise representations.
    
    Implements both "outgoing" and "incoming" triangle updates that capture
    transitive relationships in the pairwise representation.
    """
    
    def __init__(
        self,
        pair_dim: int,
        hidden_dim: Optional[int] = None,
        mode: str = "outgoing"
    ):
        super().__init__()
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim or pair_dim
        self.mode = mode
        
        assert mode in ["outgoing", "incoming"], "mode must be 'outgoing' or 'incoming'"
        
        # Projections for left and right edges
        self.left_projection = nn.Linear(pair_dim, self.hidden_dim)
        self.right_projection = nn.Linear(pair_dim, self.hidden_dim)
        
        # Gate for controlling update magnitude
        self.gate = nn.Linear(pair_dim, self.hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_dim, pair_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(pair_dim)
        
    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """
        Apply triangle multiplication update.
        
        Args:
            pair_repr: [batch, n_res, n_res, pair_dim]
            
        Returns:
            Updated pair representation
        """
        batch_size, n_res, _, _ = pair_repr.shape
        
        # Normalize input
        pair_repr_norm = self.layer_norm(pair_repr)
        
        # Project to left and right edges
        left = self.left_projection(pair_repr_norm)  # [batch, n_res, n_res, hidden_dim]
        right = self.right_projection(pair_repr_norm)
        
        # Compute gate
        gate = torch.sigmoid(self.gate(pair_repr_norm))
        
        # Triangle multiplication
        if self.mode == "outgoing":
            # Update z_ij based on z_ik and z_kj
            # Equation: z_ij += sum_k (left_ik * right_kj)
            update = torch.einsum('bikc,bkjc->bijc', left, right)
        else:  # incoming
            # Update z_ij based on z_ki and z_jk
            # Equation: z_ij += sum_k (left_ki * right_jk)
            update = torch.einsum('bkic,bjkc->bijc', left, right)
        
        # Apply gate and output projection
        update = gate * update
        update = self.output_projection(update)
        
        return update


class TriangleAttention(nn.Module):
    """
    Triangle attention module for pairwise representations.
    
    Applies attention along rows or columns of the pair representation.
    """
    
    def __init__(
        self,
        pair_dim: int,
        num_heads: int = 8,
        orientation: str = "per_row",
        dropout: float = 0.1
    ):
        super().__init__()
        self.pair_dim = pair_dim
        self.num_heads = num_heads
        self.head_dim = pair_dim // num_heads
        self.orientation = orientation
        
        assert pair_dim % num_heads == 0, "pair_dim must be divisible by num_heads"
        assert orientation in ["per_row", "per_column"], "Invalid orientation"
        
        # Q, K, V projections
        self.q_proj = nn.Linear(pair_dim, pair_dim)
        self.k_proj = nn.Linear(pair_dim, pair_dim)
        self.v_proj = nn.Linear(pair_dim, pair_dim)
        
        # Bias from pair representation
        self.bias_proj = nn.Linear(pair_dim, num_heads)
        
        # Output projection
        self.out_proj = nn.Linear(pair_dim, pair_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(pair_dim)
        
    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """
        Apply triangle attention.
        
        Args:
            pair_repr: [batch, n_res, n_res, pair_dim]
            
        Returns:
            Updated pair representation
        """
        batch_size, n_res, _, _ = pair_repr.shape
        
        # Normalize
        pair_norm = self.layer_norm(pair_repr)
        
        # Project to Q, K, V
        q = self.q_proj(pair_norm).view(batch_size, n_res, n_res, self.num_heads, self.head_dim)
        k = self.k_proj(pair_norm).view(batch_size, n_res, n_res, self.num_heads, self.head_dim)
        v = self.v_proj(pair_norm).view(batch_size, n_res, n_res, self.num_heads, self.head_dim)
        
        # Compute bias
        bias = self.bias_proj(pair_norm)  # [batch, n_res, n_res, num_heads]
        
        if self.orientation == "per_row":
            # Attention along rows (axis 1)
            q = q.permute(0, 1, 3, 2, 4)  # [batch, n_res, num_heads, n_res, head_dim]
            k = k.permute(0, 1, 3, 2, 4)
            v = v.permute(0, 1, 3, 2, 4)
            bias = bias.permute(0, 1, 3, 2)  # [batch, n_res, num_heads, n_res]
            
            # Compute attention
            attn = torch.einsum('bihqd,bihkd->bihqk', q, k) * self.scale
            attn = attn + bias.unsqueeze(-2)
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            
            # Apply attention
            out = torch.einsum('bihqk,bihkd->bihqd', attn, v)
            out = out.permute(0, 1, 3, 2, 4).reshape(batch_size, n_res, n_res, self.pair_dim)
        else:  # per_column
            # Attention along columns (axis 2)
            q = q.permute(0, 2, 3, 1, 4)  # [batch, n_res, num_heads, n_res, head_dim]
            k = k.permute(0, 2, 3, 1, 4)
            v = v.permute(0, 2, 3, 1, 4)
            bias = bias.permute(0, 2, 3, 1)  # [batch, n_res, num_heads, n_res]
            
            # Compute attention
            attn = torch.einsum('bjhqd,bjhkd->bjhqk', q, k) * self.scale
            attn = attn + bias.unsqueeze(-2)
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            
            # Apply attention
            out = torch.einsum('bjhqk,bjhkd->bjhqd', attn, v)
            out = out.permute(0, 3, 1, 2, 4).reshape(batch_size, n_res, n_res, self.pair_dim)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


class PairformerBlock(nn.Module):
    """
    Complete Pairformer block combining triangle operations.
    
    This is the core building block of the trunk module.
    """
    
    def __init__(
        self,
        pair_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Triangle multiplication (outgoing and incoming)
        self.triangle_mult_out = TriangleMultiplication(pair_dim, mode="outgoing")
        self.triangle_mult_in = TriangleMultiplication(pair_dim, mode="incoming")
        
        # Triangle attention (per row and per column)
        self.triangle_attn_row = TriangleAttention(pair_dim, num_heads, "per_row", dropout)
        self.triangle_attn_col = TriangleAttention(pair_dim, num_heads, "per_column", dropout)
        
        # Transition (feed-forward)
        self.transition = PairTransition(pair_dim, dropout=dropout)
        
    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Pairformer block.
        
        Args:
            pair_repr: [batch, n_res, n_res, pair_dim]
            
        Returns:
            Updated pair representation
        """
        # Triangle multiplication updates
        pair_repr = pair_repr + self.triangle_mult_out(pair_repr)
        pair_repr = pair_repr + self.triangle_mult_in(pair_repr)
        
        # Triangle attention updates
        pair_repr = pair_repr + self.triangle_attn_row(pair_repr)
        pair_repr = pair_repr + self.triangle_attn_col(pair_repr)
        
        # Transition
        pair_repr = pair_repr + self.transition(pair_repr)
        
        return pair_repr


class PairTransition(nn.Module):
    """Feed-forward transition for pair representations."""
    
    def __init__(self, pair_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim or pair_dim * 4
        
        self.layer_norm = nn.LayerNorm(pair_dim)
        self.fc1 = nn.Linear(pair_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, pair_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transition."""
        x = self.layer_norm(pair_repr)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TrunkModule(nn.Module):
    """
    Complete trunk module for Pearl.
    
    Learns rich, position-independent pairwise representations using
    lightweight triangle multiplication.
    """
    
    def __init__(
        self,
        pair_dim: int = 128,
        num_blocks: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.pair_dim = pair_dim
        self.num_blocks = num_blocks
        
        # Stack of Pairformer blocks
        self.blocks = nn.ModuleList([
            PairformerBlock(pair_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        
    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """
        Process pair representation through trunk.
        
        Args:
            pair_repr: [batch, n_res, n_res, pair_dim]
            
        Returns:
            Processed pair representation
        """
        for block in self.blocks:
            pair_repr = block(pair_repr)
        
        return pair_repr

