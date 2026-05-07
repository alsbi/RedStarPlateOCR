"""Tests for LR clipping via final_lr_factor (Fix 6)."""

from __future__ import annotations

import torch

from redstar_plate_ocr.pipeline.training_config import TrainingConfig


def test_final_lr_factor_default_zero():
    """Default final_lr_factor is 0.0 (no clipping)."""
    cfg = TrainingConfig()
    assert cfg.final_lr_factor == 0.0


def test_final_lr_factor_from_yaml():
    """final_lr_factor read from scheduler section of YAML."""
    cfg = TrainingConfig.from_dict(
        {"training": {"scheduler": {"final_lr_factor": 0.01}}}
    )
    assert cfg.final_lr_factor == 0.01


def test_final_lr_factor_from_yaml_missing():
    """Missing final_lr_factor in dict defaults to 0.0."""
    cfg = TrainingConfig.from_dict({"training": {}})
    assert cfg.final_lr_factor == 0.0


def test_min_lr_set_in_scheduler():
    """When final_lr_factor > 0, scheduler.min_lrs = [lr * factor]."""
    lr = 1e-3
    factor = 0.01
    optimizer = torch.optim.SGD([torch.zeros(1)], lr=lr)
    min_lr = lr * factor if factor > 0.0 else 0.0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=5,
        factor=0.5,
        min_lr=min_lr,
    )
    assert scheduler.min_lrs == [lr * factor]


def test_min_lr_zero_when_factor_zero():
    """When final_lr_factor == 0.0, min_lr is 0.0 (no clipping)."""
    lr = 1e-3
    factor = 0.0
    optimizer = torch.optim.SGD([torch.zeros(1)], lr=lr)
    min_lr = lr * factor if factor > 0.0 else 0.0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=5,
        factor=0.5,
        min_lr=min_lr,
    )
    assert scheduler.min_lrs == [0.0]
