"""Tests for compute_input_lengths from AdaptiveCompression."""

from __future__ import annotations

import torch

from redstar_plate_ocr.nn.compression import AdaptiveCompression


def test_full_mask_standard_returns_seq_len():
    """Полное заполнение content_mask → input_lengths == feat_w."""
    comp = AdaptiveCompression(canvas_height=80, canvas_width=192, stride=4)
    # feat_h=20, feat_w=48
    mask = torch.ones(2, 1, 20, 48)
    result = comp.compute_input_lengths(mask, ["standard", "standard"])
    assert result.shape == (2,)
    assert result[0].item() == 48 + 4  # +4 safety margin
    assert result[1].item() == 48 + 4


def test_half_mask_standard_returns_half():
    """Половинное заполнение → input_lengths == feat_w/2."""
    comp = AdaptiveCompression(canvas_height=80, canvas_width=192, stride=4)
    mask = torch.zeros(1, 1, 20, 48)
    mask[0, 0, :, :24] = 1.0
    result = comp.compute_input_lengths(mask, ["standard"])
    assert result[0].item() == 24 + 4  # +4 safety margin


def test_square_type_top_w_plus_bot_w():
    """Square тип → input_lengths = feat_w + bot_present."""
    comp = AdaptiveCompression(canvas_height=80, canvas_width=192, stride=4)
    # feat_h=20, feat_w=48, mid=10
    mask = torch.zeros(1, 1, 20, 48)
    # top half: 8 cols filled
    mask[0, 0, :10, :8] = 1.0
    # bot half: 6 cols filled
    mask[0, 0, 10:, :6] = 1.0
    result = comp.compute_input_lengths(mask, ["square"])
    # feat_w(48) + bot_present(6) + safety_margin(4) = 58
    assert result[0].item() == 58


def test_clamp_min_2():
    """Узкий номер → clamp(min=2) в train_epoch."""
    comp = AdaptiveCompression(canvas_height=80, canvas_width=192, stride=4)
    mask = torch.zeros(1, 1, 20, 48)
    # Только 1 колонка заполнена
    mask[0, 0, :, :1] = 1.0
    result = comp.compute_input_lengths(mask, ["standard"])
    # Метод возвращает 1+4=5 (1 col + margin), clamp(min=2) — в train_epoch
    assert result[0].item() == 5
    # clamp(min=2) → 5 already ≥ 2
    clamped = result.clamp(min=2, max=48)
    assert clamped[0].item() == 5


def test_clamp_max_seq_len():
    """input_lengths не может превышать seq_len."""
    comp = AdaptiveCompression(canvas_height=80, canvas_width=192, stride=4)
    mask = torch.ones(1, 1, 20, 48)
    result = comp.compute_input_lengths(mask, ["standard"])
    assert result[0].item() == 48 + 4  # +4 safety margin
    # clamp(max=30) — имитация seq_len < feat_w
    clamped = result.clamp(min=2, max=30)
    assert clamped[0].item() == 30


def test_mixed_batch():
    """Смешанный батч: standard + square."""
    comp = AdaptiveCompression(canvas_height=80, canvas_width=192, stride=4)
    mask = torch.zeros(2, 1, 20, 48)
    # standard: 30 cols
    mask[0, 0, :, :30] = 1.0
    # square: top=20, bot=15
    mask[1, 0, :10, :20] = 1.0
    mask[1, 0, 10:, :15] = 1.0
    result = comp.compute_input_lengths(
        mask,
        ["standard", "square"],
    )
    assert result[0].item() == 30 + 4  # +4 safety margin
    assert result[1].item() == 63 + 4  # feat_w(48) + bot_present(15) + margin


def test_square_partial_content_mask():
    """Square plate with partial content_mask in bottom half."""
    comp = AdaptiveCompression(canvas_height=80, canvas_width=192, stride=4)
    mask = torch.zeros(1, 1, 20, 48)
    # Top half: all 48 cols filled
    mask[0, 0, :10, :] = 1.0
    # Bottom half: only 10 cols filled
    mask[0, 0, 10:, :10] = 1.0
    result = comp.compute_input_lengths(mask, ["square"])
    # feat_w(48) + bot_present(10) + safety_margin(4) = 62
    assert result[0].item() == 62
