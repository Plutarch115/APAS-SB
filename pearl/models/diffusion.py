"""
Diffusion Module for Pearl

Implements the SO(3)-equivariant diffusion module for 3D coordinate prediction.
Uses denoising diffusion probabilistic models (DDPM) with equivariant transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .equivariant import EquivariantTransformerBlock


class DiffusionModule(nn.Module):
    """
    SO(3)-equivariant diffusion module for predicting 3D coordinates.
    
    Uses a stack of equivariant transformer blocks conditioned on
    pair representations from the trunk module.
    """
    
    def __init__(
        self,
        scalar_dim: int = 256,
        vector_dim: int = 64,
        pair_dim: int = 128,
        num_blocks: int = 8,
        num_heads: int = 8,
        num_diffusion_steps: int = 200,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.pair_dim = pair_dim
        self.num_blocks = num_blocks
        self.num_diffusion_steps = num_diffusion_steps
        
        # Time embedding for diffusion timestep
        self.time_embedding = TimeEmbedding(scalar_dim)
        
        # Initial projection from pair representation to node features
        self.pair_to_scalar = nn.Linear(pair_dim, scalar_dim)
        
        # Initial vector features (positions)
        self.pos_to_vector = nn.Linear(3, vector_dim * 3)
        
        # Stack of equivariant transformer blocks
        self.blocks = nn.ModuleList([
            EquivariantTransformerBlock(
                scalar_dim=scalar_dim,
                vector_dim=vector_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])
        
        # Output projection to predict coordinate updates
        self.vector_to_pos = nn.Linear(vector_dim, 1)
        
        # Diffusion schedule (cosine schedule)
        self.register_buffer('betas', self._cosine_beta_schedule(num_diffusion_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                           torch.sqrt(1.0 - self.alphas_cumprod))
        
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule for diffusion as proposed in Improved DDPM.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward(
        self,
        positions: torch.Tensor,  # [batch, n_atoms, 3]
        pair_repr: torch.Tensor,  # [batch, n_atoms, n_atoms, pair_dim]
        timestep: torch.Tensor,   # [batch]
        mask: Optional[torch.Tensor] = None  # [batch, n_atoms]
    ) -> torch.Tensor:
        """
        Predict noise/coordinate updates for given positions at timestep.
        
        Args:
            positions: Current noisy 3D coordinates
            pair_repr: Pairwise representation from trunk
            timestep: Diffusion timestep
            mask: Optional mask for valid atoms
            
        Returns:
            Predicted coordinate updates
        """
        batch_size, n_atoms, _ = positions.shape
        
        # Time embedding
        time_emb = self.time_embedding(timestep)  # [batch, scalar_dim]
        time_emb = time_emb.unsqueeze(1).expand(-1, n_atoms, -1)  # [batch, n_atoms, scalar_dim]
        
        # Initialize scalar features from pair representation
        # Average over one dimension to get node features
        scalar_features = self.pair_to_scalar(pair_repr.mean(dim=2))  # [batch, n_atoms, scalar_dim]
        scalar_features = scalar_features + time_emb
        
        # Initialize vector features from positions
        vector_features = self.pos_to_vector(positions)  # [batch, n_atoms, vector_dim * 3]
        vector_features = vector_features.view(batch_size, n_atoms, self.vector_dim, 3)
        
        # Apply equivariant transformer blocks
        for block in self.blocks:
            scalar_features, vector_features = block(
                scalar_features, vector_features, pair_repr
            )
        
        # Predict coordinate updates from vector features
        # Sum over vector dimension to get final 3D displacement
        coord_updates = torch.sum(
            vector_features * self.vector_to_pos(
                vector_features.transpose(-2, -1)
            ).transpose(-2, -1),
            dim=2
        )  # [batch, n_atoms, 3]
        
        # Apply mask if provided
        if mask is not None:
            coord_updates = coord_updates * mask.unsqueeze(-1)
        
        return coord_updates
    
    def add_noise(
        self,
        positions: torch.Tensor,
        timestep: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to positions according to diffusion schedule.
        
        Args:
            positions: Clean 3D coordinates [batch, n_atoms, 3]
            timestep: Timestep [batch]
            noise: Optional pre-generated noise
            
        Returns:
            Noisy positions and the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(positions)
        
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timestep]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timestep]
        
        # Reshape for broadcasting
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.view(-1, 1, 1)
        
        noisy_positions = (
            sqrt_alpha_cumprod * positions +
            sqrt_one_minus_alpha_cumprod * noise
        )
        
        return noisy_positions, noise
    
    @torch.no_grad()
    def sample(
        self,
        pair_repr: torch.Tensor,
        n_atoms: int,
        mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Sample 3D coordinates using reverse diffusion process.
        
        Args:
            pair_repr: Pairwise representation [batch, n_atoms, n_atoms, pair_dim]
            n_atoms: Number of atoms
            mask: Optional mask for valid atoms
            num_steps: Number of diffusion steps (default: use all)
            
        Returns:
            Sampled 3D coordinates
        """
        batch_size = pair_repr.shape[0]
        device = pair_repr.device
        
        if num_steps is None:
            num_steps = self.num_diffusion_steps
        
        # Start from random noise
        positions = torch.randn(batch_size, n_atoms, 3, device=device)
        
        # Reverse diffusion process
        for t in reversed(range(num_steps)):
            timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.forward(positions, pair_repr, timestep, mask)
            
            # Compute denoising update
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            # Compute mean of reverse process
            positions = (1.0 / torch.sqrt(alpha)) * (
                positions - (beta / torch.sqrt(1.0 - alpha_cumprod)) * predicted_noise
            )
            
            # Add noise (except for last step)
            if t > 0:
                noise = torch.randn_like(positions)
                sigma = torch.sqrt(beta)
                positions = positions + sigma * noise
            
            # Apply mask
            if mask is not None:
                positions = positions * mask.unsqueeze(-1)
        
        return positions


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestep: [batch]
        Returns:
            Time embedding [batch, dim]
        """
        # Sinusoidal embedding
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timestep.device) * -emb)
        emb = timestep.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # MLP
        emb = self.mlp(emb)
        
        return emb

