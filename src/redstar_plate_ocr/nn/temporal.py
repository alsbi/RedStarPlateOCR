"""TemporalBridge: Conv1d bridge that adds local temporal context."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class TemporalBridge(nn.Module):
    """Conv1d bridge that adds local temporal context before BiLSTM.

    Applies a lightweight Conv1d(k=3, p=1) + BatchNorm + SiLU with a
    **residual connection** so that sinusoidal positional encoding is
    preserved while the convolution adds neighbourhood context.

    Without the residual, Conv1d with k=3 would blur the PE signal.
    With residual + default Kaiming init, the bridge starts as a weak
    perturbation and gradually learns useful local patterns.
    """

    def __init__(self, channels: int = 512) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """Apply temporal bridge with residual connection.

        Args:
            x: (B, T, C) feature sequence

        Returns:
            (B, T, C) with local context added via residual
        """
        residual = x
        out = self.act(self.bn(self.conv(x.permute(0, 2, 1))))
        return residual + out.permute(0, 2, 1)
