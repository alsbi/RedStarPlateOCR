"""Tests for Stage3 and linear DropPath in PlateBackbone."""

from __future__ import annotations

import torch

from redstar_plate_ocr.nn.backbone import (
    InvertedResidualBlock,
    PlateBackbone,
)


def _get_drop_path_rate(mod: torch.nn.Module) -> float:
    """Extract drop_path rate from any block type."""
    return mod.drop_path.rate


def test_backbone_stage3_output_shape():
    """With stage3_blocks=2, output shape unchanged (same ch, same res)."""
    model = PlateBackbone(
        stem_channels=32,
        stage1_channels=32,
        stage1_blocks=1,
        stage2_channels=64,
        stage2_blocks=2,
        stage3_blocks=2,
        se_reduction=4,
        drop_path_rate=0.05,
    )
    x = torch.randn(2, 3, 80, 256)
    out = model(x)
    # Stage3 keeps H/4, W/4 and stage2_channels
    assert out.final.shape == (2, 64, 20, 64)


def test_backbone_no_stage3_backward_compat():
    """With stage3_blocks=0, behaviour identical to old backbone."""
    model = PlateBackbone(
        stem_channels=32,
        stage1_channels=32,
        stage1_blocks=1,
        stage2_channels=64,
        stage2_blocks=2,
        stage3_blocks=0,
        se_reduction=4,
        drop_path_rate=0.05,
    )
    x = torch.randn(2, 3, 80, 256)
    out = model(x)
    assert out.final.shape == (2, 64, 20, 64)
    # Verify stage3 is empty/absent
    assert len(list(model.stage3.children())) == 0


def test_backbone_linear_drop_path():
    """DropPath rates grow linearly across Stage2+Stage3 blocks."""
    stage2_blocks = 4
    stage3_blocks = 2
    drop_path_rate = 0.05
    model = PlateBackbone(
        stem_channels=32,
        stage1_channels=32,
        stage1_blocks=1,
        stage2_channels=64,
        stage2_blocks=stage2_blocks,
        stage3_blocks=stage3_blocks,
        se_reduction=4,
        drop_path_rate=drop_path_rate,
    )

    total = stage2_blocks + stage3_blocks
    expected_rates = [drop_path_rate * i / (total - 1) for i in range(total)]

    # Stage1: all zero
    for mod in model.stage1:
        assert _get_drop_path_rate(mod) == 0.0

    # Stage2: linear growth
    for i, mod in enumerate(model.stage2):
        rate = _get_drop_path_rate(mod)
        assert abs(rate - expected_rates[i]) < 1e-7, (
            f"Stage2 block {i}: expected {expected_rates[i]}, got {rate}"
        )

    # Stage3: linear growth continued
    for i, mod in enumerate(model.stage3):
        rate = _get_drop_path_rate(mod)
        idx = stage2_blocks + i
        assert abs(rate - expected_rates[idx]) < 1e-7, (
            f"Stage3 block {i}: expected {expected_rates[idx]}, got {rate}"
        )


def test_backbone_linear_drop_path_inverted_residual():
    """DropPath rates grow linearly with InvertedResidualBlock in stage3."""
    stage2_blocks = 2
    stage3_blocks = 2
    drop_path_rate = 0.05
    model = PlateBackbone(
        stem_channels=32,
        stage1_channels=32,
        stage1_blocks=1,
        stage2_channels=64,
        stage2_blocks=stage2_blocks,
        stage3_blocks=stage3_blocks,
        stage3_expand_ratio=2,
        se_reduction=4,
        drop_path_rate=drop_path_rate,
    )

    total = stage2_blocks + stage3_blocks
    expected_rates = [drop_path_rate * i / (total - 1) for i in range(total)]

    # Stage3 uses InvertedResidualBlock
    for i, mod in enumerate(model.stage3):
        assert isinstance(mod, InvertedResidualBlock), (
            f"Expected InvertedResidualBlock, got {type(mod).__name__}"
        )
        rate = _get_drop_path_rate(mod)
        idx = stage2_blocks + i
        assert abs(rate - expected_rates[idx]) < 1e-7


def test_backbone_default_no_stage3():
    """Default PlateBackbone() has no stage3 (backward compat)."""
    model = PlateBackbone()
    assert (
        not hasattr(model, "stage3") or len(list(model.stage3.children())) == 0
    )


def test_backbone_stage3_same_channels_no_downsample():
    """Stage3 uses stage2_channels and does NOT downsample."""
    model = PlateBackbone(
        stem_channels=32,
        stage1_channels=32,
        stage1_blocks=1,
        stage2_channels=64,
        stage2_blocks=1,
        stage3_blocks=3,
        se_reduction=4,
    )
    x = torch.randn(1, 3, 80, 256)
    out = model(x)
    # Same channels as stage2, same spatial resolution
    assert out.final.shape[1] == 64
    assert out.final.shape[2] == 20
    assert out.final.shape[3] == 64


def test_backbone_stage3_inverted_residual_no_downsample():
    """InvertedResidualBlock in stage3 preserves spatial dims."""
    model = PlateBackbone(
        stem_channels=32,
        stage1_channels=32,
        stage1_blocks=1,
        stage2_channels=64,
        stage2_blocks=1,
        stage3_blocks=2,
        stage3_expand_ratio=2,
        se_reduction=4,
    )
    x = torch.randn(1, 3, 80, 256)
    out = model(x)
    assert out.final.shape[1] == 64
    assert out.final.shape[2] == 20
    assert out.final.shape[3] == 64
