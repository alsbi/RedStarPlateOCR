"""Tests for Trainer and Evaluator (Milestone 5)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from redstar_plate_ocr.nn.metrics import CharacterAccuracy
from redstar_plate_ocr.nn.model import PlateOCRModel
from redstar_plate_ocr.pipeline.evaluator import Evaluator
from redstar_plate_ocr.pipeline.trainer import (
    Trainer,
    get_device_and_amp,
)
from redstar_plate_ocr.pipeline.utils import greedy_decode
from redstar_plate_ocr.plate.config import PlateConfig


def _make_model(plate_config: PlateConfig) -> PlateOCRModel:
    """Create a default model for tests."""
    cfg = plate_config
    return PlateOCRModel(plate_config=cfg)


# --- greedy_decode tests ---


def test_greedy_decode_basic():
    """greedy_decode decodes simple CTC output."""
    alphabet = "AB"
    logits = torch.tensor(
        [
            [0.1, 0.0, 10.0],  # blank
            [10.0, 0.0, 0.1],  # A
            [10.0, 0.0, 0.1],  # A (repeat → collapse)
            [0.0, 10.0, 0.1],  # B
        ],
    )
    result = greedy_decode(logits, alphabet)
    assert result == "AB"


def test_greedy_decode_blank_between():
    """greedy_decode allows repeat after blank."""
    alphabet = "AB"
    logits = torch.tensor(
        [
            [10.0, 0.0, 0.1],  # A
            [0.1, 0.0, 10.0],  # blank
            [10.0, 0.0, 0.1],  # A
        ],
    )
    result = greedy_decode(logits, alphabet)
    assert result == "AA"


def test_greedy_decode_all_blank():
    """greedy_decode returns empty for all-blank."""
    alphabet = "AB"
    logits = torch.tensor(
        [
            [0.1, 0.1, 10.0],
            [0.1, 0.1, 10.0],
        ],
    )
    result = greedy_decode(logits, alphabet)
    assert result == ""


# --- Evaluator metrics test ---


def test_evaluator_metrics(plate_config: PlateConfig):
    """evaluate returns all 5 metric keys."""
    cfg = plate_config
    model = _make_model(plate_config)
    device = torch.device("cpu")
    evaluator = Evaluator(cfg, device)

    batch = _make_mock_batch()
    mock_loader = [batch]  # type: ignore[list-item]

    result = evaluator.evaluate(model, mock_loader)  # type: ignore[arg-type]
    assert "val_plate_accuracy" in result
    assert "val_cer" in result
    assert "val_country_accuracy" in result
    assert "val_format_accuracy" in result
    assert "val_square_accuracy" in result
    assert "val_char_accuracy" in result


def _make_mock_batch() -> dict:
    """Create a mock batch dict for evaluator test."""
    bsz = 2
    images = torch.randn(bsz, 3, 80, 256)
    return {
        "image": images,
        "plate_text": ["A000AA00", "A111BB11"],
        "region": ["RU", "RU"],
        "plate_type": ["standard", "standard"],
        "orig_h": torch.tensor([80, 80]),
        "orig_w": torch.tensor([256, 256]),
    }


# --- device_and_amp tests ---


def test_device_and_amp_cpu():
    """CPU → amp=False when no CUDA/MPS."""
    with (
        patch("redstar_plate_ocr.pipeline.trainer.torch.cuda") as mock_cuda,
        patch(
            "redstar_plate_ocr.pipeline.trainer.torch.backends"
        ) as mock_backends,
    ):
        mock_cuda.is_available.return_value = False
        mock_backends.mps.is_available.return_value = False
        device, amp = get_device_and_amp(True)
        assert device == torch.device("cpu")
        assert amp is False


def test_device_and_amp_cuda():
    """CUDA → amp=True when use_amp=True."""
    with patch("redstar_plate_ocr.pipeline.trainer.torch.cuda") as mock_cuda:
        mock_cuda.is_available.return_value = True
        device, amp = get_device_and_amp(True)
        assert device == torch.device("cuda")
        assert amp is True


def test_device_and_amp_mps():
    """MPS → amp=False."""
    with (
        patch("redstar_plate_ocr.pipeline.trainer.torch.cuda") as mock_cuda,
        patch(
            "redstar_plate_ocr.pipeline.trainer.torch.backends"
        ) as mock_backends,
    ):
        mock_cuda.is_available.return_value = False
        mock_backends.mps.is_available.return_value = True
        device, amp = get_device_and_amp(True)
        assert device == torch.device("mps")
        assert amp is False


# --- Trainer init test ---


def test_trainer_init(plate_config: PlateConfig):
    """Trainer creates without errors."""
    cfg = plate_config
    model = _make_model(plate_config)

    train_ds = MagicMock()
    train_ds.samples = [
        {
            "image_path": "a.jpg",
            "plate_text": "A000AA00",
            "region": "RU",
            "plate_type": "standard",
        },
    ]
    val_ds = MagicMock()
    val_ds.samples = [
        {
            "image_path": "b.jpg",
            "plate_text": "A111BB11",
            "region": "RU",
            "plate_type": "standard",
        },
    ]

    training_cfg = {
        "training": {
            "epochs": 1,
            "lr": 1e-3,
            "batch_size": 1,
            "use_amp": False,
        },
    }

    with patch(
        "redstar_plate_ocr.pipeline.trainer.get_device_and_amp",
        return_value=(torch.device("cpu"), False),
    ):
        trainer = Trainer(
            model=model,
            plate_config=cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            cfg=training_cfg,
        )
    assert trainer.use_amp is False


# --- epoch stats formatting tests (via utils) ---


def test_format_epoch_stats_shows_compact_metrics():
    """format_epoch_stats shows compact metrics, no arrows on first."""
    from redstar_plate_ocr.pipeline.utils import (
        format_epoch_stats,
    )

    val_metrics = {
        "val_cer": 0.123,
        "val_plate_accuracy": 0.85,
        "val_country_accuracy": 0.92,
        "val_format_accuracy": 0.78,
        "val_square_accuracy": 0.60,
        "val_char_accuracy": 0.88,
    }
    best_metrics: dict[str, float] = {}
    result = format_epoch_stats(
        val_metrics,
        best_metrics,
        1.5,
    )
    assert "loss=1.5000" in result
    assert "plate=85.000%" in result
    assert "cer=0.1230" in result
    assert "char=88.000%" in result
    # First epoch → no arrows
    assert "↑" not in result
    assert "↓" not in result
    # region and fmt are shown; std and sq are shown
    assert "region=92.000%" in result
    assert "fmt=78.000%" in result
    assert "std=0.000%" in result
    assert "sq=60.000%" in result


def test_format_epoch_stats_improvement_shows_arrows():
    """format_epoch_stats shows ↑ for plate, ↓ for cer on improvement."""
    from redstar_plate_ocr.pipeline.utils import (
        format_epoch_stats,
    )

    val_metrics = {
        "val_cer": 0.05,
        "val_plate_accuracy": 0.9,
        "val_char_accuracy": 0.95,
    }
    best_metrics: dict[str, float] = {
        "val_plate_accuracy": 0.8,
        "val_cer": 0.1,
    }
    result = format_epoch_stats(
        val_metrics,
        best_metrics,
        2.0,
    )
    assert "plate=90.000%↑" in result
    assert "cer=0.0500↓" in result


# --- CharacterAccuracy edge-case tests ---


def test_char_accuracy_pred_longer_than_target():
    """CharacterAccuracy never returns negative when pred > tgt."""
    metric = CharacterAccuracy()
    # pred="ABCDE" (5 chars), tgt="AB" (2 chars)
    # dist = 3, correct = max(2 - 3, 0) = 0
    metric.update(["ABCDE"], ["AB"])
    assert metric.compute() == 0.0


def test_char_accuracy_partial_match():
    """CharacterAccuracy returns correct ratio for partial match."""
    metric = CharacterAccuracy()
    # pred="ABC", tgt="AXC" → dist=1, correct = max(3-1,0) = 2
    metric.update(["ABC"], ["AXC"])
    assert metric.compute() == pytest.approx(2 / 3)


def test_char_accuracy_exact_match():
    """CharacterAccuracy returns 1.0 for exact match."""
    metric = CharacterAccuracy()
    metric.update(["ABC"], ["ABC"])
    assert metric.compute() == 1.0
