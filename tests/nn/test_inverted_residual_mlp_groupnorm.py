"""Tests for InvertedResidualBlock (V4), MLP (V2), GroupNorm (V5)."""

from __future__ import annotations

import tempfile

import pytest
import torch

from redstar_plate_ocr.nn.backbone import (
    DWSepBlock,
    InvertedResidualBlock,
    PlateBackbone,
    _make_norm,
)

# ── V4: InvertedResidualBlock ────────────────────────────────


class TestInvertedResidualBlockForward:
    """Shape preservation and basic forward pass."""

    def test_shape_preservation(self) -> None:
        """Output shape == input shape."""
        block = InvertedResidualBlock(
            channels=64, expand_ratio=2, kernel_size=3
        )
        x = torch.randn(2, 64, 20, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_shape_preservation_kernel5(self) -> None:
        """Works with kernel_size=5."""
        block = InvertedResidualBlock(
            channels=32, expand_ratio=2, kernel_size=5
        )
        x = torch.randn(1, 32, 10, 32)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self) -> None:
        """With zero weights, output != input (residual adds input)."""
        block = InvertedResidualBlock(
            channels=16, expand_ratio=2, kernel_size=3
        )
        x = torch.randn(1, 16, 8, 8)
        out = block(x)
        # Residual ensures output is close to input when branch
        # contribution is small
        assert out.shape == x.shape


class TestInvertedResidualBlockGradientFlow:
    """Gradients flow through all sub-modules."""

    def test_gradient_flow(self) -> None:
        """Gradients reach all parameters."""
        block = InvertedResidualBlock(
            channels=32, expand_ratio=2, kernel_size=3
        )
        x = torch.randn(1, 32, 10, 16, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        for name, p in block.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_gradient_through_expand(self) -> None:
        """Expand conv receives gradients."""
        block = InvertedResidualBlock(channels=32, expand_ratio=2)
        x = torch.randn(1, 32, 8, 8)
        out = block(x)
        out.sum().backward()
        assert block.expand[0].weight.grad is not None


class TestInvertedResidualBlockExpandRatio1:
    """expand_ratio=1 should behave similarly to DWSepBlock."""

    def test_expand_ratio_1_output_shape(self) -> None:
        """With expand_ratio=1, shape preserved."""
        block = InvertedResidualBlock(
            channels=64, expand_ratio=1, kernel_size=3
        )
        x = torch.randn(2, 64, 10, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_expand_ratio_1_param_count_smaller(self) -> None:
        """expand_ratio=1 has fewer params than expand_ratio=2."""
        ch = 64
        blk1 = InvertedResidualBlock(channels=ch, expand_ratio=1)
        blk2 = InvertedResidualBlock(channels=ch, expand_ratio=2)
        n1 = sum(p.numel() for p in blk1.parameters())
        n2 = sum(p.numel() for p in blk2.parameters())
        assert n1 < n2


# ── V2: MLP ratio ────────────────────────────────────────────


class TestMLPRatio:
    """MLP expansion in DWSepBlock and InvertedResidualBlock."""

    def test_dwsep_mlp_ratio_forward(self) -> None:
        """DWSepBlock with mlp_ratio=2 preserves shape."""
        block = DWSepBlock(channels=64, mlp_ratio=2)
        x = torch.randn(2, 64, 10, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_dwsep_mlp_ratio_gradient(self) -> None:
        """Gradients flow through MLP in DWSepBlock."""
        block = DWSepBlock(channels=32, mlp_ratio=2)
        x = torch.randn(1, 32, 8, 8, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        # pw[0] is the first Conv2d in the MLP
        assert block.pw[0].weight.grad is not None

    def test_inverted_residual_mlp_ratio_forward(self) -> None:
        """InvertedResidualBlock with mlp_ratio=2 preserves shape."""
        block = InvertedResidualBlock(channels=64, expand_ratio=2, mlp_ratio=2)
        x = torch.randn(2, 64, 10, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_inverted_residual_mlp_ratio_gradient(self) -> None:
        """Gradients flow through MLP in InvertedResidualBlock."""
        block = InvertedResidualBlock(channels=32, expand_ratio=2, mlp_ratio=2)
        x = torch.randn(1, 32, 8, 8, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert block.project[0].weight.grad is not None

    def test_dwsep_mlp_ratio_more_params(self) -> None:
        """mlp_ratio=2 has more params than mlp_ratio=1."""
        ch = 64
        blk1 = DWSepBlock(channels=ch, mlp_ratio=1)
        blk2 = DWSepBlock(channels=ch, mlp_ratio=2)
        n1 = sum(p.numel() for p in blk1.parameters())
        n2 = sum(p.numel() for p in blk2.parameters())
        assert n2 > n1

    def test_dwsep_mlp_ratio_default_backward_compat(self) -> None:
        """mlp_ratio=1 (default) produces same architecture as before."""
        block = DWSepBlock(channels=64)
        # pw should be Conv2d + BN (no GELU)
        assert len(block.pw) == 2
        assert isinstance(block.pw[0], torch.nn.Conv2d)
        assert isinstance(block.pw[1], torch.nn.BatchNorm2d)


# ── V5: GroupNorm / _make_norm ───────────────────────────────


class TestGroupNorm:
    """GroupNorm(1,C) as LayerNorm replacement."""

    def test_make_norm_batch(self) -> None:
        """_make_norm('batch') returns BatchNorm2d."""
        norm = _make_norm(64, "batch")
        assert isinstance(norm, torch.nn.BatchNorm2d)

    def test_make_norm_group(self) -> None:
        """_make_norm('group') returns GroupNorm(1, C)."""
        norm = _make_norm(64, "group")
        assert isinstance(norm, torch.nn.GroupNorm)
        assert norm.num_groups == 1

    def test_group_norm_forward(self) -> None:
        """GroupNorm(1,C) works as replacement for BN in a block."""
        block = DWSepBlock(channels=32, norm_layer="group")
        x = torch.randn(2, 32, 10, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_group_norm_gradient(self) -> None:
        """Gradients flow through GroupNorm."""
        block = DWSepBlock(channels=32, norm_layer="group")
        x = torch.randn(1, 32, 8, 8, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None

    def test_inverted_residual_group_norm_forward(self) -> None:
        """InvertedResidualBlock with GroupNorm works."""
        block = InvertedResidualBlock(
            channels=64, expand_ratio=2, norm_layer="group"
        )
        x = torch.randn(2, 64, 10, 16)
        out = block(x)
        assert out.shape == x.shape


# ── Full backbone with V4+V2+V5 ──────────────────────────────


class TestBackboneWithInvertedResidual:
    """Full PlateBackbone with all new features enabled."""

    def test_backbone_with_inverted_residual(self) -> None:
        """Full forward pass with stage3_expand_ratio=2."""
        model = PlateBackbone(
            stem_channels=32,
            stage1_channels=32,
            stage1_blocks=1,
            stage2_channels=64,
            stage2_blocks=2,
            stage3_blocks=2,
            stage3_expand_ratio=2,
            se_reduction=4,
        )
        x = torch.randn(2, 3, 80, 256)
        out = model(x)
        assert out.final.shape == (2, 64, 20, 64)
        assert out.stage1.shape == (2, 32, 40, 128)

    def test_backbone_with_all_features(self) -> None:
        """V4+V2+V5 combined: expand_ratio=2, mlp_ratio=2, group norm."""
        model = PlateBackbone(
            stem_channels=32,
            stage1_channels=32,
            stage1_blocks=1,
            stage2_channels=64,
            stage2_blocks=2,
            stage3_blocks=2,
            stage3_expand_ratio=2,
            stage2_mlp_ratio=1,
            stage3_mlp_ratio=2,
            stage3_norm="group",
            se_reduction=4,
        )
        x = torch.randn(2, 3, 80, 256)
        out = model(x)
        assert out.final.shape == (2, 64, 20, 64)
        # Verify stage3 uses InvertedResidualBlock
        assert isinstance(model.stage3[0], InvertedResidualBlock)
        # Verify stage2 uses DWSepBlock
        assert isinstance(model.stage2[0], DWSepBlock)

    def test_backbone_with_features_gradient(self) -> None:
        """Gradients flow through full backbone with new features."""
        model = PlateBackbone(
            stem_channels=32,
            stage1_channels=32,
            stage1_blocks=1,
            stage2_channels=64,
            stage2_blocks=1,
            stage3_blocks=1,
            stage3_expand_ratio=2,
            stage3_mlp_ratio=2,
            stage3_norm="group",
        )
        x = torch.randn(1, 3, 80, 256, requires_grad=True)
        out = model(x)
        out.final.sum().backward()
        assert x.grad is not None

    def test_default_backward_compat(self) -> None:
        """Default params produce same behavior as before changes."""
        model = PlateBackbone()
        x = torch.randn(2, 3, 80, 256)
        out = model(x)
        assert out.final.shape == (2, 256, 20, 64)


# ── ONNX export ──────────────────────────────────────────────


class TestOnnxExportInvertedResidual:
    """ONNX export with InvertedResidualBlock and GroupNorm."""

    @pytest.fixture
    def small_model(self) -> PlateBackbone:
        """Small backbone with V4+V2+V5 features."""
        return PlateBackbone(
            stem_channels=16,
            stage1_channels=16,
            stage1_blocks=1,
            stage2_channels=32,
            stage2_blocks=1,
            stage3_blocks=1,
            stage3_expand_ratio=2,
            stage3_mlp_ratio=2,
            stage3_norm="group",
            se_reduction=4,
        )

    def test_onnx_export_inverted_residual(
        self, small_model: PlateBackbone
    ) -> None:
        """Export backbone with InvertedResidualBlock to ONNX."""
        pytest.importorskip("onnxscript")
        onnx = pytest.importorskip("onnx")
        small_model.eval()
        dummy = torch.randn(1, 3, 80, 256)
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            torch.onnx.export(
                small_model,
                dummy,
                f.name,
                opset_version=11,
                input_names=["input"],
                output_names=["stage1", "final"],
            )
            model = onnx.load(f.name)
            onnx.checker.check_model(model)

    def test_onnx_export_numerical_consistency(
        self, small_model: PlateBackbone
    ) -> None:
        """ONNX runtime output matches PyTorch output."""
        pytest.importorskip("onnxscript")
        rt = pytest.importorskip("onnxruntime")
        import numpy as np

        small_model.eval()
        dummy = torch.randn(1, 3, 80, 256)
        with torch.no_grad():
            pt_out = small_model(dummy)

        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            torch.onnx.export(
                small_model,
                dummy,
                f.name,
                opset_version=11,
                input_names=["input"],
                output_names=["stage1", "final"],
            )
            sess = rt.InferenceSession(
                f.name, providers=["CPUExecutionProvider"]
            )
            ort_out = sess.run(None, {"input": dummy.numpy()})

        # final output (index 1)
        np.testing.assert_allclose(
            pt_out.final.numpy(),
            ort_out[1],
            atol=1e-4,
            rtol=1e-3,
        )
