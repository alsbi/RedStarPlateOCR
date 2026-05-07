"""Tests for Stage3 and linear DropPath in PlateBackbone."""

from __future__ import annotations

from typing import cast

import torch

from redstar_plate_ocr.nn.backbone import DWSepBlock, PlateBackbone


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
    x = torch.randn(2, 3, 80, 192)
    out = model(x)
    # Stage3 keeps H/4, W/4 and stage2_channels
    assert out.final.shape == (2, 64, 20, 48)


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
    x = torch.randn(2, 3, 80, 192)
    out = model(x)
    assert out.final.shape == (2, 64, 20, 48)
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
        block = cast(DWSepBlock, mod)
        assert block.drop_path.rate == 0.0

    # Stage2: linear growth
    for i, mod in enumerate(model.stage2):
        block = cast(DWSepBlock, mod)
        assert abs(block.drop_path.rate - expected_rates[i]) < 1e-7, (
            f"Stage2 block {i}: expected {expected_rates[i]}, "
            f"got {block.drop_path.rate}"
        )

    # Stage3: linear growth continued
    for i, mod in enumerate(model.stage3):
        block = cast(DWSepBlock, mod)
        idx = stage2_blocks + i
        assert abs(block.drop_path.rate - expected_rates[idx]) < 1e-7, (
            f"Stage3 block {i}: expected {expected_rates[idx]}, "
            f"got {block.drop_path.rate}"
        )


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
    x = torch.randn(1, 3, 80, 192)
    out = model(x)
    # Same channels as stage2, same spatial resolution
    assert out.final.shape[1] == 64
    assert out.final.shape[2] == 20
    assert out.final.shape[3] == 48
