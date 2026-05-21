"""Tests for MultiScaleFusion via lateral connection."""

from __future__ import annotations

import torch

from redstar_plate_ocr.nn.fusion import MultiScaleFusion


class TestMultiScaleFusion:
    """MultiScaleFusion: shape, channels, strided conv."""

    def test_output_shape(self) -> None:
        """Fused output has stage2 spatial dims and channels."""
        fusion = MultiScaleFusion(
            stage1_channels=128,
            stage2_channels=256,
        )
        stage1 = torch.randn(2, 128, 40, 128)  # H/2, W/2
        stage2 = torch.randn(2, 256, 20, 64)  # H/4, W/4
        out = fusion(stage1, stage2)
        assert out.shape == (2, 256, 20, 64)

    def test_lateral_uses_strided_conv(self) -> None:
        """Lateral connection uses Conv2d with stride=2."""
        fusion = MultiScaleFusion(
            stage1_channels=64,
            stage2_channels=128,
        )
        assert fusion.lateral.stride == (2, 2)
        assert fusion.lateral.kernel_size == (2, 2)

    def test_gradient_flows(self) -> None:
        """Gradient flows through fusion to both inputs."""
        fusion = MultiScaleFusion(
            stage1_channels=64,
            stage2_channels=128,
        )
        s1 = torch.randn(1, 64, 40, 128, requires_grad=True)
        s2 = torch.randn(1, 128, 20, 64, requires_grad=True)
        out = fusion(s1, s2)
        out.sum().backward()
        assert s1.grad is not None
        assert s1.grad.abs().sum() > 0
        assert s2.grad is not None
        assert s2.grad.abs().sum() > 0

    def test_different_channel_configs(self) -> None:
        """Works with different channel configurations."""
        fusion = MultiScaleFusion(
            stage1_channels=192,
            stage2_channels=384,
        )
        s1 = torch.randn(1, 192, 40, 128)
        s2 = torch.randn(1, 384, 20, 64)
        out = fusion(s1, s2)
        assert out.shape == (1, 384, 20, 64)

    def test_has_batchnorm(self) -> None:
        """Fusion has BatchNorm after lateral conv."""
        fusion = MultiScaleFusion(
            stage1_channels=64,
            stage2_channels=128,
        )
        assert isinstance(fusion.bn, torch.nn.BatchNorm2d)
        assert fusion.bn.num_features == 128
