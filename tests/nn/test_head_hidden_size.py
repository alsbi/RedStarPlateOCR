"""Tests for UnifiedCTCHead hidden_size hook (R4)."""

from __future__ import annotations

import torch
import torch.nn as nn

from redstar_plate_ocr.nn.heads import UnifiedCTCHead


def test_unified_ctc_head_hidden_size_creates_sequential():
    """With hidden_size=128, proj is Sequential with 4 layers."""
    head = UnifiedCTCHead(
        input_size=512,
        hidden_size=128,
        union_alphabet_size=37,
    )
    assert isinstance(head.proj, nn.Sequential)
    assert len(head.proj) == 4  # Linear + LayerNorm + ReLU + Dropout
    assert isinstance(head.proj[0], nn.Linear)
    assert head.proj[0].in_features == 512
    assert head.proj[0].out_features == 128
    assert isinstance(head.proj[1], nn.LayerNorm)
    assert isinstance(head.proj[2], nn.ReLU)
    assert isinstance(head.proj[3], nn.Dropout)
    assert isinstance(head.fc, nn.Linear)
    assert head.fc.out_features == 37


def test_unified_ctc_head_no_hidden_size_creates_sequential():
    """With hidden_size=None (default), proj is Sequential (Linear+ReLU)."""
    head = UnifiedCTCHead(
        input_size=512,
        union_alphabet_size=37,
    )
    assert isinstance(head.proj, nn.Sequential)
    assert len(head.proj) == 2  # Linear + ReLU
    assert isinstance(head.proj[0], nn.Linear)
    assert head.proj[0].in_features == 512
    assert head.proj[0].out_features == 512


def test_unified_ctc_head_hidden_size_forward():
    """Forward pass works with hidden_size."""
    head = UnifiedCTCHead(
        input_size=512,
        hidden_size=128,
        union_alphabet_size=37,
    )
    x = torch.randn(2, 10, 512)
    mask = torch.zeros(2, 37)
    out = head(x, mask)
    assert out.shape == (2, 10, 37)
    assert torch.isfinite(out).all()


def test_unified_ctc_head_no_hidden_size_forward():
    """Forward pass works without hidden_size."""
    head = UnifiedCTCHead(
        input_size=512,
        union_alphabet_size=37,
    )
    x = torch.randn(2, 10, 512)
    mask = torch.zeros(2, 37)
    out = head(x, mask)
    assert out.shape == (2, 10, 37)
    assert torch.isfinite(out).all()
