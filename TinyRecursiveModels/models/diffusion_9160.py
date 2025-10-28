"""
Diffusion model for full L_level recursion (seq_len=9160).

This model learns the full trajectory of one H-cycle:
    Input: (x, y_init, z_init) - each [B, 916, 512]
    Output: (z1, z2, z3, z4, z5, z6, y_next) - [B, 7*916, 512] = [B, 6412, 512]

Architecture: Transformer with Flash Attention for full sequence modeling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

from models.layers import rms_norm


def _timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embeddings."""
    half_dim = dim // 2
    exponent = -math.log(10000.0) / max(half_dim - 1, 1)
    freqs = torch.exp(torch.arange(half_dim, device=timesteps.device) * exponent)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


@dataclass
class DiffusionSchedule:
    """DDPM noise schedule."""
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas: torch.Tensor

    @classmethod
    def create(cls, timesteps: int, device: torch.device) -> "DiffusionSchedule":
        # Linear schedule
        betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        return cls(
            betas=betas,
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
            sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
            sqrt_recip_alphas=torch.sqrt(1.0 / alphas),
        )


class FullLevelDiffusion(nn.Module):
    """
    Diffusion model for full L_level trajectory (one H-cycle).
    
    Predicts the noise added to a full trajectory given initial conditions.
    
    Input conditioning:
        - x: [B, 916, 512] - input embeddings
        - y_init: [B, 916, 512] - initial y (z_H)
        - z_init: [B, 916, 512] - initial z (z_L)
    
    Target:
        - trajectory: [B, 6412, 512] - concatenation of (z1, z2, z3, z4, z5, z6, y_next)
                      where each is [B, 916, 512]
    """
    
    def __init__(
        self,
        token_len: int = 916,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        num_trajectory_steps: int = 7,  # 6 z updates + 1 y update
    ):
        super().__init__()
        self.token_len = token_len
        self.hidden_dim = hidden_dim
        self.num_trajectory_steps = num_trajectory_steps
        self.full_seq_len = token_len * (3 + num_trajectory_steps)  # (x,y,z) + trajectory
        
        # Positional embedding for trajectory steps
        self.trajectory_pos_emb = nn.Parameter(
            torch.randn(num_trajectory_steps, 1, hidden_dim) * 0.02
        )
        
        # Conditioning projection (x, y_init, z_init)
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        noisy_target: torch.Tensor,
        timestep: torch.Tensor,
        x: torch.Tensor,
        y_init: torch.Tensor,
        z_init: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_target: [B, 6412, hidden_dim] - noisy trajectory
            timestep: [B] - diffusion timestep
            x: [B, 916, hidden_dim] - input embeddings
            y_init: [B, 916, hidden_dim] - initial y
            z_init: [B, 916, hidden_dim] - initial z
        
        Returns:
            predicted_noise: [B, 6412, hidden_dim]
        """
        B = noisy_target.shape[0]
        
        # Reshape noisy target to [B, 7, 916, hidden_dim]
        noisy_reshaped = noisy_target.view(B, self.num_trajectory_steps, self.token_len, self.hidden_dim)
        
        # Add positional embeddings for trajectory steps
        noisy_with_pos = noisy_reshaped + self.trajectory_pos_emb.unsqueeze(0)  # [B, 7, 916, hidden_dim]
        
        # Project conditioning
        cond_x = self.cond_proj(x)
        cond_y = self.cond_proj(y_init)
        cond_z = self.cond_proj(z_init)
        
        # Concatenate: [x, y_init, z_init, trajectory]
        # Shape: [B, 10, 916, hidden_dim]
        full_input = torch.cat([
            cond_x.unsqueeze(1),
            cond_y.unsqueeze(1),
            cond_z.unsqueeze(1),
            noisy_with_pos
        ], dim=1)
        
        # Flatten to [B, 9160, hidden_dim]
        full_input_flat = full_input.view(B, -1, self.hidden_dim)
        
        # Add time embedding
        time_emb = self.time_mlp(_timestep_embedding(timestep, self.hidden_dim))  # [B, hidden_dim]
        full_input_flat = full_input_flat + time_emb.unsqueeze(1)
        
        # Normalize
        full_input_flat = rms_norm(full_input_flat, variance_epsilon=1e-5)
        
        # Transform
        hidden = self.transformer(full_input_flat)
        
        # Extract trajectory portion (skip first 3*916 tokens)
        trajectory_hidden = hidden[:, 3*self.token_len:, :]  # [B, 6412, hidden_dim]
        
        # Project to output
        output = self.output_proj(trajectory_hidden)
        
        return output


class DiffusionTrainer9160:
    """Trainer for full L_level diffusion model."""
    
    def __init__(
        self,
        model: FullLevelDiffusion,
        timesteps: int = 30,
        lr: float = 3e-4,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model.to(device)
        self.schedule = DiffusionSchedule.create(timesteps, device)
        self.timesteps = timesteps
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.global_step = 0
    
    def train_step(
        self,
        target: torch.Tensor,
        x: torch.Tensor,
        y_init: torch.Tensor,
        z_init: torch.Tensor,
    ) -> float:
        """
        Single training step.
        
        Args:
            target: [B, 6412, hidden_dim] - trajectory (z1...z6, y_next)
            x: [B, 916, hidden_dim] - input embeddings
            y_init: [B, 916, hidden_dim] - initial y
            z_init: [B, 916, hidden_dim] - initial z
        
        Returns:
            loss value
        """
        self.model.train()
        batch_size = target.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)
        
        # Sample noise
        noise = torch.randn_like(target)
        
        # Forward diffusion: add noise to target
        sqrt_alpha = self.schedule.sqrt_alphas_cumprod[timesteps].view(batch_size, 1, 1)
        sqrt_one_minus = self.schedule.sqrt_one_minus_alphas_cumprod[timesteps].view(batch_size, 1, 1)
        noisy_target = sqrt_alpha * target + sqrt_one_minus * noise
        
        # Predict noise
        pred_noise = self.model(noisy_target, timesteps, x, y_init, z_init)
        
        # Loss
        loss = F.mse_loss(pred_noise, noise)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        
        self.global_step += 1
        return float(loss.item())
    
    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        y_init: torch.Tensor,
        z_init: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate trajectory via reverse diffusion.
        
        Args:
            x: [B, 916, hidden_dim] - input embeddings
            y_init: [B, 916, hidden_dim] - initial y
            z_init: [B, 916, hidden_dim] - initial z
        
        Returns:
            trajectory: [B, 6412, hidden_dim]
        """
        self.model.eval()
        
        batch_size = x.shape[0]
        seq_len = self.model.num_trajectory_steps * self.model.token_len
        
        # Start from pure noise
        sample = torch.randn(batch_size, seq_len, self.model.hidden_dim, device=self.device)
        
        # Reverse diffusion
        for step in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            
            # Predict noise
            pred_noise = self.model(sample, t, x, y_init, z_init)
            
            # Denoise
            beta = self.schedule.betas[step]
            sqrt_recip_alpha = self.schedule.sqrt_recip_alphas[step]
            sqrt_one_minus_alpha_cumprod = self.schedule.sqrt_one_minus_alphas_cumprod[step]
            
            sample = sqrt_recip_alpha * (sample - beta / sqrt_one_minus_alpha_cumprod * pred_noise)
            
            # Add noise (except last step)
            if step > 0:
                sample = sample + torch.sqrt(beta) * torch.randn_like(sample)
        
        return sample
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'schedule': {
                'betas': self.schedule.betas,
                'alphas': self.schedule.alphas,
                'alphas_cumprod': self.schedule.alphas_cumprod,
            },
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']

