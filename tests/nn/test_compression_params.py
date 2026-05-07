"""Tests for T4.1: parameterize compute_content_mask."""

from __future__ import annotations

import pytest
import torch

from redstar_plate_ocr.nn.compression import (
    AdaptiveCompression,
    AttentionPool,
    compute_content_mask,
)


class TestComputeContentMaskRequiredParams:
    """compute_content_mask requires feat_h, feat_w, stride."""

    def test_no_defaults_raises(self) -> None:
        """Calling without feat_h/feat_w/stride raises TypeError."""
        orig_h = torch.tensor([80, 80])
        orig_w = torch.tensor([192, 192])
        with pytest.raises(TypeError):
            compute_content_mask(orig_h, orig_w)  # type: ignore[call-arg]

    def test_explicit_params_works(self) -> None:
        """Explicit feat_h=20, feat_w=48, stride=4 works."""
        orig_h = torch.tensor([80, 80])
        orig_w = torch.tensor([192, 192])
        mask = compute_content_mask(
            orig_h,
            orig_w,
            feat_h=20,
            feat_w=48,
            stride=4,
        )
        assert mask.shape == (2, 1, 20, 48)
        assert (mask == 1.0).all()

    def test_custom_params(self) -> None:
        """Custom feat_h/feat_w/stride produce correct shape."""
        orig_h = torch.tensor([40, 40])
        orig_w = torch.tensor([96, 96])
        mask = compute_content_mask(
            orig_h,
            orig_w,
            feat_h=10,
            feat_w=24,
            stride=4,
        )
        assert mask.shape == (2, 1, 10, 24)
        assert (mask == 1.0).all()


class TestAdaptiveCompressionParams:
    """AdaptiveCompression stores feat_h/feat_w/stride."""

    def test_default_params(self) -> None:
        """Default canvas 80x192, stride=4 -> feat_h=20, feat_w=48."""
        comp = AdaptiveCompression(in_channels=256)
        assert comp.feat_h == 20
        assert comp.feat_w == 48
        assert comp.stride == 4

    def test_custom_canvas(self) -> None:
        """Custom canvas_height/width computes correct feat dims."""
        comp = AdaptiveCompression(
            canvas_height=40,
            canvas_width=96,
            stride=4,
            in_channels=256,
        )
        assert comp.feat_h == 10
        assert comp.feat_w == 24

    def test_custom_stride(self) -> None:
        """Custom stride computes correct feat dims."""
        comp = AdaptiveCompression(
            canvas_height=80,
            canvas_width=192,
            stride=8,
            in_channels=256,
        )
        assert comp.feat_h == 10
        assert comp.feat_w == 24

    def test_forward_standard_uses_orig_dims(self) -> None:
        """forward_standard takes orig_h/orig_w, computes mask."""
        comp = AdaptiveCompression(in_channels=256)
        features = torch.randn(2, 256, 20, 48)
        orig_h = torch.tensor([80, 80])
        orig_w = torch.tensor([192, 192])
        out = comp.forward_standard(features, orig_h, orig_w)
        assert out.shape == (2, 48, 256)

    def test_forward_square_uses_orig_dims(self) -> None:
        """forward_square takes orig_h/orig_w, computes mask."""
        comp = AdaptiveCompression(in_channels=256)
        features = torch.randn(2, 256, 20, 48)
        orig_h = torch.tensor([80, 80])
        orig_w = torch.tensor([192, 192])
        out = comp.forward_square(features, orig_h, orig_w)
        assert out.shape == (2, 96, 256)

    def test_compute_content_mask_method(self) -> None:
        """compute_content_mask method uses stored params."""
        comp = AdaptiveCompression(
            canvas_height=40,
            canvas_width=96,
            stride=4,
            in_channels=256,
        )
        orig_h = torch.tensor([40, 40])
        orig_w = torch.tensor([96, 96])
        mask = comp.compute_content_mask(orig_h, orig_w)
        assert mask.shape == (2, 1, 10, 24)

    def test_has_attention_pool(self) -> None:
        """AdaptiveCompression has AttentionPool."""
        comp = AdaptiveCompression(in_channels=256)
        assert isinstance(comp.attn_pool, AttentionPool)

    def test_in_channels_propagated(self) -> None:
        """in_channels is used by AttentionPool."""
        comp = AdaptiveCompression(in_channels=128)
        assert comp.attn_pool.attn_proj[0].in_channels == 128
