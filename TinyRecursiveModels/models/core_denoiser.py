"""One-step core denoisers for TRM block replacement."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class OneStepCoreDenoiser(nn.Module):
    """
    Simple transformer encoder that takes concatenated (noisy_input, noise) and predicts the clean update.

    Input shape: [B, 2 * seq_len, hidden_dim]
    Output shape: [B, seq_len, hidden_dim]
    """

    def __init__(
        self,
        seq_len: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        expansion: float = 4.0,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * expansion),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.seq_len = seq_len

    def forward(self, concat_input: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(concat_input)
        hidden = hidden[:, : self.seq_len, :]
        return self.output_proj(hidden)


class CoreDenoiserSet(nn.Module):
    """Container for one-step denoisers across all core indices."""

    def __init__(
        self,
        num_cores: int,
        seq_len: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        expansion: float = 4.0,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.cores = nn.ModuleList(
            [
                OneStepCoreDenoiser(
                    seq_len=seq_len,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    expansion=expansion,
                )
                for _ in range(num_cores)
            ]
        )

    def forward_core(self, core_idx: int, x_noisy: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([x_noisy, noise], dim=1)
        return self.cores[core_idx](concat)

    def __len__(self) -> int:  # pragma: no cover - helper
        return len(self.cores)
