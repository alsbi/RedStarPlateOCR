"""Tests for AttentionPool and AdaptiveCompression with attention."""

from __future__ import annotations

import torch

from redstar_plate_ocr.nn.compression import (
    AdaptiveCompression,
    AttentionPool,
)


class TestAttentionPool:
    """AttentionPool: shape, masking, gradients."""

    def test_output_shape_standard(self) -> None:
        """(B,C,H,W) + mask -> (B,C,W)."""
        pool = AttentionPool(in_channels=256)
        features = torch.randn(2, 256, 20, 64)
        mask = torch.ones(2, 1, 20, 64)
        out = pool(features, mask)
        assert out.shape == (2, 256, 64)

    def test_output_shape_square_halves(self) -> None:
        """Top/bot halves: (B,C,H/2,W) -> (B,C,W)."""
        pool = AttentionPool(in_channels=256)
        features = torch.randn(2, 256, 10, 64)
        mask = torch.ones(2, 1, 10, 64)
        out = pool(features, mask)
        assert out.shape == (2, 256, 64)

    def test_masking_zeros_out_padded_rows(self) -> None:
        """Padded rows (mask=0) contribute near-zero."""
        pool = AttentionPool(in_channels=64)
        features = torch.randn(1, 64, 10, 12)
        mask = torch.ones(1, 1, 10, 12)
        mask[0, 0, 5:, :] = 0.0  # bottom half padded
        out = pool(features, mask)
        # Output should be finite
        assert torch.isfinite(out).all()

    def test_gradient_flows(self) -> None:
        """Gradient flows through AttentionPool."""
        pool = AttentionPool(in_channels=64)
        features = torch.randn(
            1,
            64,
            10,
            12,
            requires_grad=True,
        )
        mask = torch.ones(1, 1, 10, 12)
        out = pool(features, mask)
        out.sum().backward()
        assert features.grad is not None
        assert features.grad.abs().sum() > 0

    def test_reduction_parameter(self) -> None:
        """reduction=2 produces smaller mid dimension."""
        pool = AttentionPool(in_channels=64, reduction=2)
        mid = 64 // 2  # 32
        assert pool.attn_proj[0].out_channels == mid

    def test_all_mask_zero_produces_finite(self) -> None:
        """When mask is all zeros, output is still finite."""
        pool = AttentionPool(in_channels=64)
        features = torch.randn(1, 64, 5, 8)
        mask = torch.zeros(1, 1, 5, 8)
        out = pool(features, mask)
        assert torch.isfinite(out).all()


class TestAdaptiveCompressionAttention:
    """AdaptiveCompression uses AttentionPool."""

    def test_has_attn_pool(self) -> None:
        """AdaptiveCompression has attn_pool attribute."""
        comp = AdaptiveCompression(in_channels=256)
        assert hasattr(comp, "attn_pool")
        assert isinstance(comp.attn_pool, AttentionPool)

    def test_forward_standard_shape(self) -> None:
        """forward_standard with attention produces (B,W,C)."""
        comp = AdaptiveCompression(
            canvas_height=80,
            canvas_width=256,
            in_channels=256,
        )
        features = torch.randn(2, 256, 20, 64)
        orig_h = torch.tensor([80, 80])
        orig_w = torch.tensor([256, 256])
        out = comp.forward_standard(features, orig_h, orig_w)
        assert out.shape == (2, 64, 256)

    def test_forward_square_shape(self) -> None:
        """forward_square with attention produces (B,2*W,C)."""
        comp = AdaptiveCompression(
            canvas_height=80,
            canvas_width=256,
            in_channels=256,
        )
        features = torch.randn(2, 256, 20, 64)
        orig_h = torch.tensor([80, 80])
        orig_w = torch.tensor([256, 256])
        out = comp.forward_square(features, orig_h, orig_w)
        assert out.shape == (2, 128, 256)

    def test_no_masked_mean_method(self) -> None:
        """_masked_mean static method removed."""
        comp = AdaptiveCompression(in_channels=256)
        assert not hasattr(comp, "_masked_mean")

    def test_gradient_flows_through_compression(self) -> None:
        """Gradient flows through attention-based compression."""
        comp = AdaptiveCompression(
            canvas_height=80,
            canvas_width=256,
            in_channels=256,
        )
        features = torch.randn(
            1,
            256,
            20,
            64,
            requires_grad=True,
        )
        orig_h = torch.tensor([80])
        orig_w = torch.tensor([256])
        out = comp.forward_standard(features, orig_h, orig_w)
        out.sum().backward()
        assert features.grad is not None
        assert features.grad.abs().sum() > 0
