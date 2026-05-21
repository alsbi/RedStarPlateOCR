"""PlateBackbone: MobileNetV3-style backbone with DWSep+SE blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


def _get_activation(name: str) -> nn.Module:
    """Return activation module by name."""
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name in ("hardswish", "hard_swish"):
        return nn.Hardswish(inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unknown activation: {name}")


def _get_gate_activation(name: str) -> nn.Module:
    """Return gate activation module by name."""
    if name == "sigmoid":
        return nn.Sigmoid()
    if name in ("hardsigmoid", "hard_sigmoid"):
        return nn.Hardsigmoid(inplace=True)
    raise ValueError(f"Unknown gate activation: {name}")


@dataclass
class BackboneOutput:
    """Output of PlateBackbone with intermediate features."""

    stage1: Tensor  # (B, C1, H/2, W/2)
    final: Tensor  # (B, C2, H/4, W/4)


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization."""

    def __init__(self, rate: float = 0.0):
        super().__init__()
        self.rate = rate

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.rate == 0.0:
            return x
        keep_prob = 1.0 - self.rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * (rand < keep_prob) / keep_prob


class SEAttention(nn.Module):
    """Squeeze-and-Excitation attention module."""

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        gate_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        mid = channels // reduction
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, mid)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid, channels)
        self.gate = _get_gate_activation(gate_activation)

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.shape
        scale = self.pool(x).view(b, c)
        scale = self.act(self.fc1(scale))
        scale = self.gate(self.fc2(scale))
        return scale.view(b, c, 1, 1)


class DWSepBlock(nn.Module):
    """Depthwise-Separable block with optional attention."""

    def __init__(
        self,
        channels: int,
        se_reduction: int = 4,
        drop_path_rate: float = 0.0,
        attention: str = "se",
        kernel_size: int = 3,
        activation: str = "silu",
        gate_activation: str = "sigmoid",
    ):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            _get_activation(activation),
        )
        self.se: SEAttention | None = None
        if attention == "se":
            self.se = SEAttention(
                channels,
                se_reduction,
                gate_activation=gate_activation,
            )
        self.pw = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.dw(x)
        if self.se is not None:
            out = out * self.se(out)
        out = self.pw(out)
        out = self.drop_path(out)
        return out + residual


class PlateBackbone(nn.Module):
    """MobileNetV3-style backbone for plate recognition."""

    def __init__(
        self,
        stem_channels: int = 128,
        stage1_channels: int = 128,
        stage1_blocks: int = 2,
        stage2_channels: int = 256,
        stage2_blocks: int = 3,
        stage3_channels: int | None = None,
        stage3_blocks: int = 0,
        se_reduction: int = 4,
        drop_path_rate: float = 0.05,
        attention: str = "se",
        activation: str = "silu",
        gate_activation: str = "sigmoid",
        stage2_kernel_size: int = 3,
        stage3_kernel_size: int = 3,
    ):
        super().__init__()
        self._final_channels = stage3_channels or stage2_channels
        self.stem = self._build_stem(stem_channels, activation)
        self._stem_to_stage1 = self._build_stem_to_stage1(
            stem_channels, stage1_channels, activation
        )
        self.stage1 = self._build_stage(
            stage1_channels,
            se_reduction,
            0.0,
            attention,
            stage1_blocks,
            activation=activation,
            gate_activation=gate_activation,
        )
        self.down = self._build_down(
            stage1_channels, stage2_channels, activation
        )
        dp_rates = self._compute_dp_rates(
            stage2_blocks, stage3_blocks, drop_path_rate
        )
        self.stage2 = self._build_stage(
            stage2_channels,
            se_reduction,
            dp_rates,
            attention,
            stage2_blocks,
            activation=activation,
            gate_activation=gate_activation,
            kernel_size=stage2_kernel_size,
        )
        s3_ch = stage3_channels or stage2_channels
        if stage3_channels is not None and stage3_channels != stage2_channels:
            self.expand_stage3 = self._build_expand(
                stage2_channels, s3_ch, activation
            )
        else:
            self.expand_stage3 = None
        self.stage3 = self._build_stage(
            s3_ch,
            se_reduction,
            dp_rates,
            attention,
            stage3_blocks,
            offset=stage2_blocks,
            activation=activation,
            gate_activation=gate_activation,
            kernel_size=stage3_kernel_size,
        )

    @property
    def final_channels(self) -> int:
        """Number of channels in the final backbone output."""
        return self._final_channels

    @staticmethod
    def _build_stem(
        channels: int,
        activation: str = "silu",
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            _get_activation(activation),
        )

    @staticmethod
    def _build_stem_to_stage1(
        stem_channels: int,
        stage1_channels: int,
        activation: str = "silu",
    ) -> nn.Module:
        if stem_channels == stage1_channels:
            return nn.Identity()
        return nn.Sequential(
            nn.Conv2d(stem_channels, stage1_channels, 1, bias=False),
            nn.BatchNorm2d(stage1_channels),
            _get_activation(activation),
        )

    @staticmethod
    def _build_stage(
        channels: int,
        se_reduction: int,
        drop_path: float | list[float],
        attention: str,
        num_blocks: int,
        offset: int = 0,
        activation: str = "silu",
        gate_activation: str = "sigmoid",
        kernel_size: int = 3,
    ) -> nn.Sequential:
        rates: list[float] = (
            drop_path
            if isinstance(drop_path, list)
            else [drop_path] * num_blocks
        )
        return nn.Sequential(
            *[
                DWSepBlock(
                    channels,
                    se_reduction,
                    drop_path_rate=rates[offset + i],
                    attention=attention,
                    kernel_size=kernel_size,
                    activation=activation,
                    gate_activation=gate_activation,
                )
                for i in range(num_blocks)
            ]
        )

    @staticmethod
    def _build_down(
        in_channels: int,
        out_channels: int,
        activation: str = "silu",
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                stride=2,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            _get_activation(activation),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            _get_activation(activation),
        )

    @staticmethod
    def _build_expand(
        in_channels: int,
        out_channels: int,
        activation: str = "silu",
    ) -> nn.Sequential:
        """1×1 conv expand from stage2 to stage3 channels."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            _get_activation(activation),
        )

    @staticmethod
    def _compute_dp_rates(
        stage2_blocks: int, stage3_blocks: int, drop_path_rate: float
    ) -> list[float]:
        total = stage2_blocks + stage3_blocks
        if total > 1:
            return [drop_path_rate * i / (total - 1) for i in range(total)]
        if total == 1:
            return [drop_path_rate]
        return []

    def forward(self, x: Tensor) -> BackboneOutput:
        x = self.stem(x)
        stage1_out = self.stage1(self._stem_to_stage1(x))
        x = self.down(stage1_out)
        x = self.stage2(x)
        if self.expand_stage3 is not None:
            x = self.expand_stage3(x)
        x = self.stage3(x)
        return BackboneOutput(stage1=stage1_out, final=x)
