"""Tests for hidden_size in FormatHead/CountryHead + model wiring."""

from __future__ import annotations

import torch
import torch.nn as nn

from redstar_plate_ocr.nn.compression import compute_content_mask
from redstar_plate_ocr.nn.heads import (
    CountryHead,
    FormatHead,
)

# --- FormatHead ---


def _make_mask(orig_h: torch.Tensor, orig_w: torch.Tensor) -> torch.Tensor:
    """Shortcut to compute content_mask with default feat params."""
    return compute_content_mask(orig_h, orig_w, feat_h=20, feat_w=64, stride=4)


def test_format_head_no_hidden_is_linear():
    """Default FormatHead uses single Linear layer (shape_enc + fc)."""
    head = FormatHead(in_channels=256)
    assert isinstance(head.fc, nn.Linear)
    assert head.fc.out_features == 2
    # Input: 128 (shape_enc) + 2 (h/w ratios) = 130
    assert head.fc.in_features == 130


def test_format_head_with_hidden_is_sequential():
    """FormatHead with hidden_size uses Sequential(Linear, ReLU, Linear)."""
    head = FormatHead(in_channels=256, hidden_size=128)
    assert isinstance(head.fc, nn.Sequential)
    assert len(head.fc) == 3
    assert isinstance(head.fc[0], nn.Linear)
    assert head.fc[0].in_features == 130  # 128 (shape_enc) + 2 (h/w)
    assert head.fc[0].out_features == 128
    assert isinstance(head.fc[1], nn.ReLU)
    assert isinstance(head.fc[2], nn.Linear)
    assert head.fc[2].out_features == 2


def test_format_head_hidden_forward_shape():
    """FormatHead with hidden_size produces (B, 2) output."""
    head = FormatHead(in_channels=256, hidden_size=64)
    x = torch.randn(3, 256, 20, 64)
    orig_h = torch.tensor([80, 60, 80], dtype=torch.int64)
    orig_w = torch.tensor([256, 256, 256], dtype=torch.int64)
    cmask = _make_mask(orig_h, orig_w)
    out = head(x, content_mask=cmask, orig_h=orig_h, orig_w=orig_w)
    assert out.shape == (3, 2)
    assert torch.isfinite(out).all()


def test_format_head_no_hidden_forward_shape():
    """FormatHead without hidden_size produces (B, 2) output."""
    head = FormatHead(in_channels=256)
    x = torch.randn(3, 256, 20, 64)
    orig_h = torch.tensor([80, 60, 80], dtype=torch.int64)
    orig_w = torch.tensor([256, 256, 256], dtype=torch.int64)
    cmask = _make_mask(orig_h, orig_w)
    out = head(x, content_mask=cmask, orig_h=orig_h, orig_w=orig_w)
    assert out.shape == (3, 2)


def test_format_head_ignores_visual_features():
    """FormatHead output depends only on plate dims/shape, not pixels."""
    head = FormatHead(in_channels=256)
    head.eval()
    orig_h = torch.tensor([80, 60], dtype=torch.int64)
    orig_w = torch.tensor([256, 256], dtype=torch.int64)
    cmask = _make_mask(orig_h, orig_w)

    x1 = torch.randn(2, 256, 20, 64)
    x2 = torch.randn(2, 256, 20, 64)  # different features

    with torch.no_grad():
        out1 = head(x1, content_mask=cmask, orig_h=orig_h, orig_w=orig_w)
        out2 = head(x2, content_mask=cmask, orig_h=orig_h, orig_w=orig_w)

    # Same shape/dims → same logits regardless of features
    assert torch.allclose(out1, out2, atol=1e-6), (
        "FormatHead should depend only on shape/dims"
    )


def test_format_head_different_dims_different_output():
    """Different plate dimensions produce different format predictions."""
    head = FormatHead(in_channels=256)
    head.eval()
    x = torch.randn(1, 256, 20, 64)

    orig_h_std = torch.tensor([80], dtype=torch.int64)
    orig_w_std = torch.tensor([256], dtype=torch.int64)
    orig_h_sq = torch.tensor([80], dtype=torch.int64)
    orig_w_sq = torch.tensor([80], dtype=torch.int64)

    mask_std = _make_mask(orig_h_std, orig_w_std)
    mask_sq = _make_mask(orig_h_sq, orig_w_sq)

    with torch.no_grad():
        out_std = head(
            x,
            content_mask=mask_std,
            orig_h=orig_h_std,
            orig_w=orig_w_std,
        )
        out_sq = head(
            x,
            content_mask=mask_sq,
            orig_h=orig_h_sq,
            orig_w=orig_w_sq,
        )

    assert not torch.allclose(out_std, out_sq, atol=1e-6), (
        "Different plate dimensions should produce different format logits"
    )


def test_format_head_learns_standard_vs_square():
    """FormatHead can learn to distinguish standard vs square."""
    import torch.nn.functional as F

    head = FormatHead(in_channels=256, hidden_size=32)
    head.train()
    optimizer = torch.optim.Adam(head.parameters(), lr=0.02)

    # 8 standard (h=80, w=256), 8 square (h=80, w=80)
    orig_h = torch.tensor([80] * 8 + [80] * 8, dtype=torch.int64)
    orig_w = torch.tensor([256] * 8 + [80] * 8, dtype=torch.int64)
    labels = torch.tensor([0] * 8 + [1] * 8, dtype=torch.long)
    cmask = _make_mask(orig_h, orig_w)
    features = torch.randn(16, 256, 20, 64)

    for _ in range(50):
        optimizer.zero_grad()
        logits = head(
            features,
            content_mask=cmask,
            orig_h=orig_h,
            orig_w=orig_w,
        )
        F.cross_entropy(logits, labels).backward()
        optimizer.step()

    head.eval()
    with torch.no_grad():
        logits = head(
            features,
            content_mask=cmask,
            orig_h=orig_h,
            orig_w=orig_w,
        )
        preds = logits.argmax(1)
        assert (preds == labels).all(), (
            f"Should learn standard=0, square=1; got {preds.tolist()}"
        )


# --- CountryHead ---


def test_country_head_no_hidden_is_linear():
    """Default CountryHead uses single Linear layer."""
    head = CountryHead(in_channels=256, num_countries=7)
    assert isinstance(head.fc, nn.Linear)
    assert head.fc.out_features == 7


def test_country_head_with_hidden_is_sequential():
    """CountryHead with hidden_size uses Sequential."""
    head = CountryHead(in_channels=256, num_countries=7, hidden_size=64)
    assert isinstance(head.fc, nn.Sequential)
    assert len(head.fc) == 3
    assert isinstance(head.fc[0], nn.Linear)
    assert head.fc[0].out_features == 64
    assert isinstance(head.fc[2], nn.Linear)
    assert head.fc[2].out_features == 7


def test_country_head_hidden_forward_shape():
    """CountryHead with hidden_size produces (B, num_countries)."""
    head = CountryHead(in_channels=256, num_countries=5, hidden_size=32)
    x = torch.randn(2, 256, 20, 64)
    out = head(x)
    assert out.shape == (2, 5)
    assert torch.isfinite(out).all()


def test_country_head_no_hidden_forward_shape():
    """CountryHead without hidden_size produces (B, num_countries)."""
    head = CountryHead(in_channels=256, num_countries=5)
    x = torch.randn(2, 256, 20, 64)
    out = head(x)
    assert out.shape == (2, 5)
