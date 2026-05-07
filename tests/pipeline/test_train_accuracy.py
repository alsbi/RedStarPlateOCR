"""Tests for per-batch accuracy in train_epoch and epoch summary."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from redstar_plate_ocr.pipeline.train_epoch import _compute_batch_accuracy
from redstar_plate_ocr.pipeline.utils import format_epoch_stats


def _make_plate_config_mock():
    """Create a mock PlateConfig with union_alphabet."""
    cfg = MagicMock()
    cfg.union_alphabet = "AB0123456789"
    return cfg


def _make_model_output_mock(
    bsz: int = 2,
    n_formats: int = 3,
    n_countries: int = 5,
    seq_len: int = 10,
    alphabet_size: int = 14,
):
    """Create a mock ModelOutput."""
    output = MagicMock()
    output.format_logits = torch.zeros(bsz, n_formats)
    output.country_logits = torch.zeros(bsz, n_countries)
    output.ctc_output = torch.zeros(bsz, seq_len, alphabet_size)
    return output


# --- _compute_batch_accuracy (single-pass) ---


def test_compute_batch_accuracy_all_correct():
    """All predictions match GT -> plate_acc=1.0, char_acc=1.0."""
    output = _make_model_output_mock(bsz=2)
    gt_format = torch.zeros(2, dtype=torch.long)
    gt_country = torch.zeros(2, dtype=torch.long)
    gt_texts = ["A01", "B12"]
    from redstar_plate_ocr.pipeline import train_epoch

    original = train_epoch.greedy_decode
    call_count = 0

    def fake_greedy(logits, alphabet, input_length=None):
        nonlocal call_count
        result = ["A01", "B12"][call_count]
        call_count += 1
        return result

    train_epoch.greedy_decode = fake_greedy
    try:
        fmt_acc, ctry_acc, plate_acc, char_acc = _compute_batch_accuracy(
            output,
            gt_format,
            gt_country,
            gt_texts,
            _make_plate_config_mock(),
        )
        assert plate_acc == 1.0
        assert char_acc == 1.0
    finally:
        train_epoch.greedy_decode = original


def test_compute_batch_accuracy_half_correct():
    """Half predictions match GT -> plate_acc=0.5."""
    output = _make_model_output_mock(bsz=2)
    gt_format = torch.zeros(2, dtype=torch.long)
    gt_country = torch.zeros(2, dtype=torch.long)
    gt_texts = ["A01", "B12"]
    from redstar_plate_ocr.pipeline import train_epoch

    original = train_epoch.greedy_decode
    call_count = 0

    def fake_greedy(logits, alphabet, input_length=None):
        nonlocal call_count
        result = ["A01", "XXX"][call_count]
        call_count += 1
        return result

    train_epoch.greedy_decode = fake_greedy
    try:
        fmt_acc, ctry_acc, plate_acc, char_acc = _compute_batch_accuracy(
            output,
            gt_format,
            gt_country,
            gt_texts,
            _make_plate_config_mock(),
        )
        assert plate_acc == 0.5
    finally:
        train_epoch.greedy_decode = original


def test_compute_batch_accuracy_empty_batch():
    """Empty batch -> plate_acc=0.0, char_acc=0.0."""
    output = _make_model_output_mock(bsz=0)
    gt_format = torch.zeros(0, dtype=torch.long)
    gt_country = torch.zeros(0, dtype=torch.long)
    gt_texts: list[str] = []
    fmt_acc, ctry_acc, plate_acc, char_acc = _compute_batch_accuracy(
        output,
        gt_format,
        gt_country,
        gt_texts,
        _make_plate_config_mock(),
    )
    assert plate_acc == 0.0
    assert char_acc == 0.0


# --- format_epoch_stats with train accuracy ---


def test_format_epoch_stats_without_train_acc():
    """No train accuracy -> no tr_acc/tr_fmt/tr_ctry in output."""
    val = {"val_plate_accuracy": 0.8, "val_cer": 0.1}
    result = format_epoch_stats(val, {}, 1.5)
    assert "tr_plate" not in result
    assert "tr_fmt" not in result
    assert "tr_ctry" not in result


def test_format_epoch_stats_train_acc_not_in_progressbar():
    """Train accuracy NOT shown in progress bar (logged to file)."""
    val = {
        "val_plate_accuracy": 0.8,
        "val_cer": 0.1,
        "train_plate_acc": 0.72,
        "train_fmt_acc": 0.95,
        "train_ctry_acc": 0.88,
    }
    result = format_epoch_stats(val, {}, 1.5)
    assert "tr_plate" not in result
    assert "tr_fmt" not in result
    assert "tr_ctry" not in result
    assert "plate=80.000%" in result
    assert "cer=0.1000" in result


def test_format_epoch_stats_compact_format():
    """Compact format: .3% for plate/char, includes std/sq."""
    val = {
        "val_plate_accuracy": 0.856,
        "val_cer": 0.12,
        "val_char_accuracy": 0.934,
        "val_country_accuracy": 0.7,
        "val_format_accuracy": 0.6,
        "val_standard_accuracy": 0.5,
        "val_square_accuracy": 0.4,
    }
    result = format_epoch_stats(val, {}, 1.5)
    assert "plate=85.600%" in result
    assert "char=93.400%" in result
    assert "region=70.000%" in result
    assert "fmt=60.000%" in result
    assert "std=50.000%" in result
    assert "sq=40.000%" in result


def test_format_epoch_stats_arrows_not_on_first_epoch():
    """Arrows only on improvement, not on first epoch."""
    val = {
        "val_plate_accuracy": 0.9,
        "val_cer": 0.05,
        "val_char_accuracy": 0.95,
    }
    # First epoch (best_metrics empty) -> no arrows
    result = format_epoch_stats(val, {}, 1.5)
    assert "↑" not in result
    assert "↓" not in result

    # Improvement -> plate ↑, cer ↓
    best = {
        "val_plate_accuracy": 0.8,
        "val_cer": 0.1,
    }
    result = format_epoch_stats(val, best, 1.5)
    assert "↑" in result
    assert "↓" in result

    # No improvement -> no arrows
    best = {
        "val_plate_accuracy": 0.95,
        "val_cer": 0.03,
    }
    result = format_epoch_stats(val, best, 1.5)
    assert "↑" not in result
    assert "↓" not in result
