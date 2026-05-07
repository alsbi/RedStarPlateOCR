"""Multi-scale feature fusion via lateral connection."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class MultiScaleFusion(nn.Module):
    """Fuse Stage1 and Stage2/3 features via lateral connection.

    Uses strided convolution (not avg_pool) to preserve
    high-frequency details from Stage1.
    """

    def __init__(
        self,
        stage1_channels: int,
        stage2_channels: int,
    ) -> None:
        super().__init__()
        self.lateral = nn.Conv2d(
            stage1_channels,
            stage2_channels,
            kernel_size=2,
            stride=2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(stage2_channels)

    def forward(
        self,
        stage1_out: Tensor,
        stage2_out: Tensor,
    ) -> Tensor:
        lateral = self.bn(self.lateral(stage1_out))
        return stage2_out + lateral
