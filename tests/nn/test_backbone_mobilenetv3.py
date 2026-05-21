"""Tests for MobileNetV3 P1+P4+P2 features in backbone."""

from __future__ import annotations

import pytest
import torch

from redstar_plate_ocr.nn.backbone import (
    DWSepBlock,
    PlateBackbone,
    SEAttention,
    _get_activation,
    _get_gate_activation,
)

# ── Helper tests ──────────────────────────────────────────────


class TestGetActivation:
    """_get_activation returns correct module by name."""

    @pytest.mark.parametrize(
        "name,expected_type",
        [
            ("silu", torch.nn.SiLU),
            ("hardswish", torch.nn.Hardswish),
            ("hard_swish", torch.nn.Hardswish),
            ("relu", torch.nn.ReLU),
        ],
    )
    def test_valid_names(self, name: str, expected_type: type) -> None:
        act = _get_activation(name)
        assert isinstance(act, expected_type)

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown activation"):
            _get_activation("tanh")


class TestGetGateActivation:
    """_get_gate_activation returns correct module by name."""

    @pytest.mark.parametrize(
        "name,expected_type",
        [
            ("sigmoid", torch.nn.Sigmoid),
            ("hardsigmoid", torch.nn.Hardsigmoid),
            ("hard_sigmoid", torch.nn.Hardsigmoid),
        ],
    )
    def test_valid_names(self, name: str, expected_type: type) -> None:
        act = _get_gate_activation(name)
        assert isinstance(act, expected_type)

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown gate activation"):
            _get_gate_activation("relu")


# ── SEAttention with gate_activation ──────────────────────────


class TestSEGateActivation:
    """SEAttention supports different gate activations (P4)."""

    def test_hardsigmoid_output_shape(self) -> None:
        """Hardsigmoid SE produces same shape as sigmoid."""
        se = SEAttention(
            channels=64, reduction=4, gate_activation="hardsigmoid"
        )
        x = torch.randn(2, 64, 10, 24)
        out = se(x)
        assert out.shape == (2, 64, 1, 1)

    def test_hardsigmoid_values_in_range(self) -> None:
        """Hardsigmoid output is clamped to [0, 1]."""
        se = SEAttention(
            channels=64, reduction=4, gate_activation="hardsigmoid"
        )
        x = torch.randn(2, 64, 10, 24)
        out = se(x)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_sigmoid_default_backward_compat(self) -> None:
        """Default gate_activation='sigmoid' preserves old behavior."""
        se = SEAttention(channels=64, reduction=4)
        assert isinstance(se.gate, torch.nn.Sigmoid)

    def test_hardsigmoid_view_compatibility(self) -> None:
        """Hardsigmoid output can be reshaped via .view(b,c,1,1)."""
        se = SEAttention(
            channels=32, reduction=4, gate_activation="hardsigmoid"
        )
        x = torch.randn(1, 32, 8, 16)
        out = se(x)
        # Verify shape matches what DWSepBlock expects
        assert out.shape == (1, 32, 1, 1)
        # Verify broadcasting works
        scaled = x * out
        assert scaled.shape == x.shape


# ── DWSepBlock with activation + kernel_size ──────────────────


class TestDWSepBlockActivation:
    """DWSepBlock supports configurable activation (P1)."""

    def test_hardswish_forward(self) -> None:
        """DWSepBlock with hardswish activation runs forward."""
        block = DWSepBlock(channels=64, activation="hardswish", attention="se")
        x = torch.randn(1, 64, 10, 24)
        out = block(x)
        assert out.shape == (1, 64, 10, 24)

    def test_hardswish_gradient_flow(self) -> None:
        """Gradient flows through Hardswish activation."""
        block = DWSepBlock(channels=64, activation="hardswish", attention="se")
        x = torch.randn(1, 64, 10, 24, requires_grad=True)
        y = block(x)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_relu_forward(self) -> None:
        """DWSepBlock with relu activation runs forward."""
        block = DWSepBlock(channels=64, activation="relu", attention="se")
        x = torch.randn(1, 64, 10, 24)
        out = block(x)
        assert out.shape == (1, 64, 10, 24)


class TestDWSepBlockKernelSize:
    """DWSepBlock supports variable kernel_size (P2)."""

    def test_kernel5_forward(self) -> None:
        """kernel_size=5 produces same spatial dimensions."""
        block = DWSepBlock(channels=64, kernel_size=5, attention="se")
        x = torch.randn(1, 64, 10, 24)
        out = block(x)
        assert out.shape == (1, 64, 10, 24)

    def test_kernel7_forward(self) -> None:
        """kernel_size=7 produces same spatial dimensions."""
        block = DWSepBlock(channels=64, kernel_size=7, attention="se")
        x = torch.randn(1, 64, 10, 24)
        out = block(x)
        assert out.shape == (1, 64, 10, 24)

    def test_kernel5_gradient_flow(self) -> None:
        """Gradient flows through kernel_size=5 DW conv."""
        block = DWSepBlock(channels=64, kernel_size=5, attention="se")
        x = torch.randn(1, 64, 10, 24, requires_grad=True)
        y = block(x)
        y.sum().backward()
        assert x.grad is not None

    def test_kernel3_default_backward_compat(self) -> None:
        """Default kernel_size=3 preserves old behavior."""
        block = DWSepBlock(channels=64)
        dw_conv = block.dw[0]
        assert dw_conv.kernel_size == (3, 3)
        assert dw_conv.padding == (1, 1)


# ── PlateBackbone integration ─────────────────────────────────


class TestBackboneHardSwish:
    """PlateBackbone with Hardswish + Hardsigmoid (P1+P4)."""

    def test_hardswish_full_forward(self) -> None:
        """Full backbone forward with hardswish/hardsigmoid."""
        model = PlateBackbone(
            stem_channels=32,
            stage1_channels=32,
            stage1_blocks=1,
            stage2_channels=64,
            stage2_blocks=1,
            stage3_blocks=0,
            activation="hardswish",
            gate_activation="hardsigmoid",
        )
        x = torch.randn(1, 3, 80, 256)
        out = model(x)
        assert out.stage1.shape == (1, 32, 40, 128)
        assert out.final.shape == (1, 64, 20, 64)

    def test_hardswish_gradient_flow(self) -> None:
        """Gradient flows through full backbone with hardswish."""
        model = PlateBackbone(
            stem_channels=32,
            stage1_channels=32,
            stage1_blocks=1,
            stage2_channels=64,
            stage2_blocks=1,
            stage3_blocks=0,
            activation="hardswish",
            gate_activation="hardsigmoid",
        )
        x = torch.randn(1, 3, 80, 256, requires_grad=True)
        out = model(x)
        out.final.sum().backward()
        assert x.grad is not None


class TestBackboneKernelSize:
    """PlateBackbone with variable kernel_size per stage (P2)."""

    def test_stage2_kernel5(self) -> None:
        """stage2_kernel_size=5 works and keeps spatial dims."""
        model = PlateBackbone(
            stem_channels=32,
            stage1_channels=32,
            stage1_blocks=1,
            stage2_channels=64,
            stage2_blocks=1,
            stage3_blocks=0,
            stage2_kernel_size=5,
        )
        x = torch.randn(1, 3, 80, 256)
        out = model(x)
        assert out.final.shape == (1, 64, 20, 64)

    def test_stage3_kernel5(self) -> None:
        """stage3_kernel_size=5 works with stage3 blocks."""
        model = PlateBackbone(
            stem_channels=32,
            stage1_channels=32,
            stage1_blocks=1,
            stage2_channels=64,
            stage2_blocks=1,
            stage3_blocks=2,
            stage3_kernel_size=5,
        )
        x = torch.randn(1, 3, 80, 256)
        out = model(x)
        assert out.final.shape == (1, 64, 20, 64)

    def test_both_stages_kernel5(self) -> None:
        """Both stage2+stage3 with kernel_size=5."""
        model = PlateBackbone(
            stem_channels=32,
            stage1_channels=32,
            stage1_blocks=1,
            stage2_channels=64,
            stage2_blocks=1,
            stage3_blocks=1,
            stage2_kernel_size=5,
            stage3_kernel_size=5,
        )
        x = torch.randn(1, 3, 80, 256)
        out = model(x)
        assert out.stage1.shape == (1, 32, 40, 128)
        assert out.final.shape == (1, 64, 20, 64)

    def test_default_kernel3_backward_compat(self) -> None:
        """Default kernel_size=3 preserves old behavior."""
        model = PlateBackbone()
        # Stage1 always uses default kernel_size=3
        block0 = model.stage1[0]
        assert block0.dw[0].kernel_size == (3, 3)


class TestBackboneFullMobilenetV3:
    """PlateBackbone with all P1+P2+P4 features combined."""

    def test_hardswish_hardsigmoid_kernel5(self) -> None:
        """Full MobileNetV3-style config: hardswish+hardsigmoid+k5."""
        model = PlateBackbone(
            stem_channels=32,
            stage1_channels=32,
            stage1_blocks=1,
            stage2_channels=64,
            stage2_blocks=2,
            stage3_blocks=1,
            activation="hardswish",
            gate_activation="hardsigmoid",
            stage2_kernel_size=5,
            stage3_kernel_size=5,
        )
        x = torch.randn(2, 3, 80, 256)
        out = model(x)
        assert out.stage1.shape == (2, 32, 40, 128)
        assert out.final.shape == (2, 64, 20, 64)

    def test_param_count_with_new_features(self) -> None:
        """Param count stays reasonable with new features."""
        model = PlateBackbone(
            stem_channels=32,
            stage1_channels=32,
            stage1_blocks=1,
            stage2_channels=64,
            stage2_blocks=2,
            stage3_blocks=1,
            activation="hardswish",
            gate_activation="hardsigmoid",
            stage2_kernel_size=5,
            stage3_kernel_size=5,
        )
        n = sum(p.numel() for p in model.parameters())
        # kernel_size=5 adds some params vs k=3 but should be <100k
        assert n > 0
        assert n < 200_000, f"Too many params: {n}"
