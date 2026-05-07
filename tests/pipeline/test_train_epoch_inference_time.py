"""Tests for avg_batch_ms in train_epoch."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from redstar_plate_ocr.pipeline.train_epoch import (
    _format_batch_stats,
    run_train_epoch,
)


def test_format_batch_stats_contains_batch_ms():
    """_format_batch_stats includes XXms."""
    running = {"loss": 0.5, "ctc": 0.3}
    result = _format_batch_stats(
        running,
        fmt_acc=0.8,
        ctry_acc=0.9,
        plate_acc=0.7,
        char_acc=0.6,
        avg_batch_ms=12.3,
    )
    assert "12ms" in result


def test_format_batch_stats_default_batch_ms():
    """_format_batch_stats with default avg_batch_ms=0."""
    running = {"loss": 0.5, "ctc": 0.3}
    result = _format_batch_stats(
        running,
        fmt_acc=0.8,
        ctry_acc=0.9,
        plate_acc=0.7,
        char_acc=0.6,
    )
    assert "0ms" in result


def test_run_train_epoch_returns_avg_batch_ms():
    """run_train_epoch returns avg_batch_ms in result dict."""
    from redstar_plate_ocr.pipeline import train_epoch

    trainer = MagicMock()
    trainer.device = torch.device("cpu")
    trainer.use_amp = False
    trainer.config.gradient_accumulation_steps = 1
    trainer.config.gradient_clip = 1.0
    trainer.config.update_every_n_batches = 1
    trainer._interrupt_requested = False

    output = MagicMock()
    output.ctc_output = torch.zeros(2, 5, 10)
    output.format_logits = torch.zeros(2, 3)
    output.country_logits = torch.zeros(2, 3)
    output.content_mask = torch.ones(2, 5, dtype=torch.bool)
    output.plate_types = ["standard", "standard"]
    trainer.model = MagicMock(return_value=output)
    trainer.model.compression = MagicMock()
    trainer.model.compression.compute_input_lengths.return_value = (
        torch.tensor([5, 5])
    )
    trainer.model.plate_config = MagicMock()
    trainer.model.plate_config.union_alphabet = "AB0123456789"

    loss_dict = {
        "total": torch.tensor(1.0, requires_grad=True),
        "ctc": torch.tensor(0.5),
        "country": torch.tensor(0.2),
        "format": torch.tensor(0.1),
    }
    trainer.combined_loss = MagicMock(return_value=loss_dict)
    trainer.scaler = MagicMock()
    trainer.optimizer = MagicMock()
    trainer.optimizer.param_groups = [{"lr": 0.001, "params": []}]
    trainer.country_optimizer = MagicMock()
    trainer.country_optimizer.param_groups = [{"lr": 0.001, "params": []}]
    trainer.model.encode_countries = MagicMock(
        return_value=torch.zeros(2, dtype=torch.long),
    )

    batch = {
        "image": torch.rand(2, 3, 32, 128),
        "orig_h": [32, 32],
        "orig_w": [128, 128],
        "region": ["RU", "RU"],
        "plate_type": ["standard", "standard"],
        "plate_text": ["A01", "B02"],
    }
    loader = [batch]

    progress = MagicMock()
    task_id = MagicMock()

    with patch.object(
        train_epoch,
        "_compute_batch_accuracy",
        return_value=(0.5, 0.5, 0.5, 0.5),
    ):
        result = run_train_epoch(
            trainer,
            loader,
            0.0,
            progress,
            task_id,
        )

    assert "avg_batch_ms" in result
    assert result["avg_batch_ms"] >= 0.0
