"""Tests for synergy_weight config propagation."""

from redstar_plate_ocr.pipeline.training_config import TrainingConfig


def test_synergy_weight_default_is_zero():
    """Default synergy_weight is 0.0 for backward compat."""
    cfg = TrainingConfig()
    assert cfg.synergy_weight == 0.0


def test_synergy_weight_from_dict_missing_key():
    """Missing synergy_weight in dict defaults to 0.0."""
    cfg = TrainingConfig.from_dict({"training": {}})
    assert cfg.synergy_weight == 0.0


def test_synergy_weight_from_dict_explicit():
    """Explicit synergy_weight in dict is propagated."""
    cfg = TrainingConfig.from_dict({"training": {"synergy_weight": 0.05}})
    assert cfg.synergy_weight == 0.05


def test_synergy_weight_from_yaml_config():
    """synergy_weight read from model.yaml training section."""
    from pathlib import Path

    import yaml

    yaml_path = Path("configs/model.yaml")
    cfg_raw = yaml.safe_load(yaml_path.read_text())
    cfg = TrainingConfig.from_dict(cfg_raw)
    assert cfg.synergy_weight == 0.6
