"""ContextFiLM: Feature-wise Linear Modulation

Conditioned on country and format.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ContextFiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on country and format.

    Takes country index and format index as conditioning signals,
    produces γ and β to modulate feature sequences::

        x = x * (1 + γ.unsqueeze(1)) + β.unsqueeze(1)

    The last layer of the MLP is initialised to zeros so that FiLM
    starts as an identity transform (γ≈0, β≈0).  This guarantees
    that adding FiLM does not break training from existing
    checkpoints.
    """

    def __init__(
        self,
        num_countries: int,
        country_emb_dim: int = 128,
        format_emb_dim: int = 64,
        feature_dim: int = 512,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.country_emb = nn.Embedding(num_countries, country_emb_dim)
        self.format_emb = nn.Embedding(2, format_emb_dim)

        concat_dim = country_emb_dim + format_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim * 2),
        )

        # Zero-init last layer → FiLM starts as identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.feature_dim = feature_dim

    def forward(
        self,
        x: Tensor,
        country_idx: Tensor,
        format_idx: Tensor,
    ) -> Tensor:
        """Modulate features conditioned on country and format.

        Args:
            x: ``(B, T, C)`` feature sequence.
            country_idx: ``(B,)`` integer country indices.
            format_idx: ``(B,)`` integer format indices
                (0=standard, 1=square).

        Returns:
            ``(B, T, C)`` modulated features.
        """
        ctx = torch.cat(
            [self.country_emb(country_idx), self.format_emb(format_idx)],
            dim=-1,
        )
        params = self.mlp(ctx)  # (B, 2*C)
        gamma, beta = params.chunk(2, dim=-1)  # each (B, C)

        return x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
