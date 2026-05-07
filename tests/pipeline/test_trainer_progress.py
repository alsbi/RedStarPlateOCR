"""Tests for trainer progress bar and logging improvements."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from redstar_plate_ocr.pipeline.trainer import Trainer
from redstar_plate_ocr.plate.config import PlateConfig


def _make_trainer(plate_config: PlateConfig, **overrides) -> Trainer:
    """Create a minimal Trainer instance for testing."""
    pc = plate_config
    from redstar_plate_ocr.nn.model import PlateOCRModel

    model = PlateOCRModel(plate_config=pc)
    train_ds = MagicMock()
    val_ds = MagicMock()
    cfg = {
        "training": {
            "epochs": 10,
            "lr": 1e-3,
            "batch_size": 32,
            "warmup_epochs": 2,
            "no_aug_epochs": 3,
            "use_amp": False,
            "update_every_n_batches": 5,
        },
        "preprocessing": {},
        "augmentation": {},
    }
    with patch(
        "redstar_plate_ocr.pipeline.trainer.get_device_and_amp",
        return_value=(torch.device("cpu"), False),
    ):
        return Trainer(
            model=model,
            plate_config=pc,
            train_dataset=train_ds,
            val_dataset=val_ds,
            cfg=cfg,
            **overrides,
        )


# --- update_every_n_batches ---


def test_update_every_n_batches_default(plate_config: PlateConfig):
    """Default update_every_n_batches is 3."""
    pc = plate_config
    from redstar_plate_ocr.nn.model import PlateOCRModel

    model = PlateOCRModel(plate_config=pc)
    cfg = {"training": {"use_amp": False}}
    with patch(
        "redstar_plate_ocr.pipeline.trainer.get_device_and_amp",
        return_value=(torch.device("cpu"), False),
    ):
        trainer = Trainer(
            model=model,
            plate_config=pc,
            train_dataset=MagicMock(),
            val_dataset=MagicMock(),
            cfg=cfg,
        )
    assert trainer.config.update_every_n_batches == 3


def test_update_every_n_batches_custom(plate_config: PlateConfig):
    """Custom update_every_n_batches from config."""
    trainer = _make_trainer(plate_config)
    assert trainer.config.update_every_n_batches == 5


# --- _format_epoch_stats (via utils.format_epoch_stats) ---


def test_format_epoch_stats_basic():
    """format_epoch_stats shows loss, plate, cer in compact format."""
    from redstar_plate_ocr.pipeline.utils import (
        format_epoch_stats,
    )

    val_metrics = {
        "val_plate_accuracy": 0.85,
        "val_cer": 0.12,
    }
    best_metrics = {
        "val_plate_accuracy": 0.90,
    }
    result = format_epoch_stats(
        val_metrics,
        best_metrics,
        0.5,
    )
    assert "loss=0.5000" in result
    assert "plate=85.000%" in result
    assert "cer=0.1200" in result


def test_format_epoch_stats_empty_best_no_arrows():
    """format_epoch_stats with empty best: no arrows (first epoch)."""
    from redstar_plate_ocr.pipeline.utils import (
        format_epoch_stats,
    )

    val_metrics = {
        "val_plate_accuracy": 0.85,
        "val_cer": 0.12,
    }
    result = format_epoch_stats(
        val_metrics,
        {},
        0.5,
    )
    assert "plate=85.000%" in result
    assert "cer=0.1200" in result
    assert "↑" not in result
    assert "↓" not in result


def test_format_epoch_stats_shows_best_indicators():
    """format_epoch_stats shows ↑ for plate, ↓ for cer on improvement."""
    from redstar_plate_ocr.pipeline.utils import (
        format_epoch_stats,
    )

    val_metrics = {
        "val_cer": 0.050,
        "val_plate_accuracy": 0.95,
        "val_country_accuracy": 0.90,
        "val_format_accuracy": 0.88,
        "val_square_accuracy": 0.70,
    }
    best_metrics = {
        "val_cer": 0.100,
        "val_plate_accuracy": 0.90,
    }
    result = format_epoch_stats(
        val_metrics,
        best_metrics,
        0.5,
    )
    assert "plate=95.000%↑" in result
    assert "cer=0.0500↓" in result


# --- _log_training_config ---


def test_log_training_config_no_error(plate_config: PlateConfig):
    """_log_training_config runs without error."""
    trainer = _make_trainer(plate_config)
    trainer._log_training_config()
