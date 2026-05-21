"""Tests for BackboneOutput dataclass and DWSepBlock refactoring."""

from __future__ import annotations

from dataclasses import fields

import torch

from redstar_plate_ocr.nn.backbone import (
    BackboneOutput,
    DWSepBlock,
    PlateBackbone,
    SEAttention,
)


class TestBackboneOutput:
    """BackboneOutput dataclass: type, shape."""

    def test_has_stage1_and_final_fields(self) -> None:
        """BackboneOutput has stage1 and final fields."""
        field_names = {f.name for f in fields(BackboneOutput)}
        assert "stage1" in field_names
        assert "final" in field_names

    def test_backbone_returns_backbone_output(self) -> None:
        """PlateBackbone.forward returns BackboneOutput."""
        model = PlateBackbone(
            stem_channels=32,
            stage1_channels=32,
            stage1_blocks=1,
            stage2_channels=64,
            stage2_blocks=1,
            stage3_blocks=0,
        )
        x = torch.randn(1, 3, 80, 256)
        out = model(x)
        assert isinstance(out, BackboneOutput)

    def test_stage1_shape(self) -> None:
        """stage1 has (B, C1, H/2, W/2) shape."""
        model = PlateBackbone(
            stem_channels=32,
            stage1_channels=32,
            stage1_blocks=1,
            stage2_channels=64,
            stage2_blocks=1,
            stage3_blocks=0,
        )
        x = torch.randn(1, 3, 80, 256)
        out = model(x)
        assert out.stage1.shape == (1, 32, 40, 128)

    def test_final_shape(self) -> None:
        """final has (B, C2, H/4, W/4) shape."""
        model = PlateBackbone(
            stem_channels=32,
            stage1_channels=32,
            stage1_blocks=1,
            stage2_channels=64,
            stage2_blocks=1,
            stage3_blocks=0,
        )
        x = torch.randn(1, 3, 80, 256)
        out = model(x)
        assert out.final.shape == (1, 64, 20, 64)

    def test_default_backbone_output(self) -> None:
        """Default PlateBackbone returns BackboneOutput."""
        model = PlateBackbone()
        x = torch.randn(1, 3, 80, 256)
        out = model(x)
        assert isinstance(out, BackboneOutput)
        assert out.stage1.shape[1] == 128
        assert out.final.shape[1] == 256


class TestDWSepBlock:
    """DWSepBlock with attention parameter."""

    def test_default_attention_is_se(self) -> None:
        """Default attention='se' creates SE module."""
        block = DWSepBlock(channels=64)
        assert hasattr(block, "se")
        assert isinstance(block.se, SEAttention)

    def test_attention_none_no_se(self) -> None:
        """attention='none' does not create SE module."""
        block = DWSepBlock(channels=64, attention="none")
        assert not hasattr(block, "se") or block.se is None

    def test_forward_with_se(self) -> None:
        """Forward pass works with SE attention."""
        block = DWSepBlock(channels=64, attention="se")
        x = torch.randn(1, 64, 10, 24)
        out = block(x)
        assert out.shape == (1, 64, 10, 24)

    def test_forward_without_se(self) -> None:
        """Forward pass works without attention."""
        block = DWSepBlock(channels=64, attention="none")
        x = torch.randn(1, 64, 10, 24)
        out = block(x)
        assert out.shape == (1, 64, 10, 24)

    def test_gradient_flows(self) -> None:
        """Gradient flows through DWSepBlock."""
        block = DWSepBlock(channels=64)
        x = torch.randn(1, 64, 10, 24, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestSEAttention:
    """SEAttention standalone module."""

    def test_output_shape(self) -> None:
        """SE attention produces (B, C, 1, 1) scale."""
        se = SEAttention(channels=64, reduction=4)
        x = torch.randn(2, 64, 10, 24)
        out = se(x)
        assert out.shape == (2, 64, 1, 1)

    def test_values_between_zero_and_one(self) -> None:
        """SE output values are in [0, 1] regardless of gate type."""
        se = SEAttention(channels=64, reduction=4)
        x = torch.randn(2, 64, 10, 24)
        out = se(x)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()
