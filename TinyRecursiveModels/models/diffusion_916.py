"""
Diffusion model for single TRM layer (seq_len=916).

This model learns to predict a single layer update in the TRM:
    - z = net(x, y, z)  -> predict next z given (x+y+z)
    - y = net(y, z)      -> predict next y given (y+z)

Architecture: Uses the same TRM block architecture (Attention + MLP) for consistency.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional

from models.layers import SwiGLU, rms_norm, Attention, RotaryEmbedding, CosSin


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


class TRMBlock(nn.Module):
    """
    TRM-style block with self-attention and MLP.
    Same architecture as TinyRecursiveReasoningModel_ACTV1Block.
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        expansion: float = 4.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm_eps = norm_eps
        
        # Self attention
        self.self_attn = Attention(
            hidden_size=hidden_dim,
            head_dim=hidden_dim // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False
        )
        
        # MLP
        self.mlp = SwiGLU(hidden_dim, expansion)
    
    def forward(self, hidden_states: torch.Tensor, cos_sin: Optional[CosSin] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, D]
            cos_sin: Optional RoPE embeddings
        
        Returns:
            [B, L, D]
        """
        # Self Attention with residual
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps
        )
        
        # MLP with residual
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps
        )
        
        return hidden_states


class SingleLayerDiffusion(nn.Module):
    """
    Diffusion model for a single TRM layer update.
    Uses the same TRM block architecture for consistency.
    
    Predicts the noise added to a target latent state given:
        - noisy_target: [B, 916, 512] - noisy version of target
        - timestep: [B] - diffusion timestep
        - *inputs: variable number of condition tensors, each [B, 916, 512]
                  For z-update: (z_prev, y, x)
                  For y-update: (y_prev, z)
    """
    
    def __init__(
        self,
        seq_len: int = 916,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        expansion: float = 4.0,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # RoPE embeddings (same as TRM)
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_dim // num_heads,
            max_position_embeddings=seq_len,
            base=rope_theta
        )
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        # Main processing layers: TRM-style blocks with attention
        self.layers = nn.ModuleList([
            TRMBlock(hidden_dim, num_heads, expansion) for _ in range(num_layers)
        ])
    
    def forward(
        self,
        noisy_target: torch.Tensor,
        timestep: torch.Tensor,
        *condition_inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            noisy_target: [B, seq_len, hidden_dim] - noisy target state
            timestep: [B] - diffusion timestep indices
            *condition_inputs: variable number of [B, seq_len, hidden_dim] tensors
        
        Returns:
            predicted_noise: [B, seq_len, hidden_dim]
        """
        # Sum all conditioning inputs (additive as in TRM)
        condition = sum(condition_inputs)  # [B, seq_len, hidden_dim]
        
        # Time embedding
        time_emb = self.time_mlp(_timestep_embedding(timestep, self.hidden_dim))  # [B, hidden_dim]
        time_emb = time_emb.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Combine noisy target + condition + time
        hidden = noisy_target + condition + time_emb
        hidden = rms_norm(hidden, variance_epsilon=1e-5)
        
        # Get RoPE embeddings
        cos_sin = self.rotary_emb()
        
        # Process through TRM-style blocks (with attention!)
        for layer in self.layers:
            hidden = layer(hidden, cos_sin)
        
        # Output (predicts noise)
        return hidden


class DiffusionTrainer916:
    """Trainer for single-layer diffusion model."""
    
    def __init__(
        self,
        model: SingleLayerDiffusion,
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
        *condition_inputs: torch.Tensor
    ) -> float:
        """
        Single training step.
        
        Args:
            target: [B, seq_len, hidden_dim] - clean target state
            *condition_inputs: conditioning tensors
        
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
        pred_noise = self.model(noisy_target, timesteps, *condition_inputs)
        
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
        *condition_inputs: torch.Tensor,
        num_samples: int = None,
    ) -> torch.Tensor:
        """
        Generate samples via reverse diffusion.
        
        Args:
            *condition_inputs: conditioning tensors [B, seq_len, hidden_dim]
            num_samples: if provided, override batch size
        
        Returns:
            samples: [B, seq_len, hidden_dim]
        """
        self.model.eval()
        
        # Determine batch size
        if num_samples is not None:
            batch_size = num_samples
            # Expand conditioning if needed
            condition_inputs = tuple(
                cond.expand(batch_size, -1, -1) if cond.shape[0] == 1 else cond
                for cond in condition_inputs
            )
        else:
            batch_size = condition_inputs[0].shape[0]
        
        seq_len = condition_inputs[0].shape[1]
        hidden_dim = condition_inputs[0].shape[2]
        
        # Start from pure noise
        x = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
        
        # Reverse diffusion
        for step in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            
            # Predict noise
            pred_noise = self.model(x, t, *condition_inputs)
            
            # Denoise
            beta = self.schedule.betas[step]
            sqrt_recip_alpha = self.schedule.sqrt_recip_alphas[step]
            sqrt_one_minus_alpha_cumprod = self.schedule.sqrt_one_minus_alphas_cumprod[step]
            
            x = sqrt_recip_alpha * (x - beta / sqrt_one_minus_alpha_cumprod * pred_noise)
            
            # Add noise (except last step)
            if step > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)
        
        return x
    
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

