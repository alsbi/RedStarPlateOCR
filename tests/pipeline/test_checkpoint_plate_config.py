"""Tests for plate_config in checkpoint (Fix 8)."""

from __future__ import annotations

import pytest

from redstar_plate_ocr.pipeline.checkpoint import (
    build_checkpoint,
    validate_checkpoint_compat,
)
from redstar_plate_ocr.plate.config import (
    PlateConfig,
    RegionConfig,
    ValidChars,
)


def _make_config(
    countries: list[str] | None = None,
    forbidden: list[str] | None = None,
) -> PlateConfig:
    """Create a minimal PlateConfig for testing."""
    if countries is None:
        countries = ["RU"]
    regions: dict[str, RegionConfig] = {}
    for code in countries:
        regions[code] = RegionConfig(
            pattern=["A000AA"],
            valid_chars=ValidChars(letters="AB", digits="0123456789"),
            forbidden_combos=forbidden if forbidden else [],
        )
    return PlateConfig(regions=regions)


def test_yaml_roundtrip() -> None:
    """PlateConfig round-trip via YAML string preserves data."""
    cfg = _make_config()
    yaml_str = cfg.to_yaml_string()
    restored = PlateConfig.from_yaml_string(yaml_str)
    assert restored.regions.keys() == cfg.regions.keys()
    for key in cfg.regions:
        assert restored.regions[key].pattern == cfg.regions[key].pattern
        assert (
            restored.regions[key].valid_chars.letters
            == cfg.regions[key].valid_chars.letters
        )
        assert (
            restored.regions[key].valid_chars.digits
            == cfg.regions[key].valid_chars.digits
        )


def test_yaml_roundtrip_with_forbidden_empty() -> None:
    """Round-trip with forbidden_combos: [] stays [], not None."""
    cfg = _make_config(forbidden=[])
    yaml_str = cfg.to_yaml_string()
    restored = PlateConfig.from_yaml_string(yaml_str)
    for key in cfg.regions:
        assert restored.regions[key].forbidden_combos == []


def test_yaml_roundtrip_with_forbidden_nonempty() -> None:
    """Round-trip with non-empty forbidden_combos."""
    cfg = _make_config(forbidden=["AA"])
    yaml_str = cfg.to_yaml_string()
    restored = PlateConfig.from_yaml_string(yaml_str)
    for key in cfg.regions:
        assert restored.regions[key].forbidden_combos == ["AA"]


def test_checkpoint_contains_config() -> None:
    """build_checkpoint includes plate_config_yaml when given."""
    cfg = _make_config()
    yaml_str = cfg.to_yaml_string()
    ckpt = build_checkpoint(
        epoch=1,
        model_state={},
        optimizer_state={},
        scheduler_state={},
        scaler_state={},
        best_metric=0.5,
        plate_config_yaml=yaml_str,
    )
    assert "plate_config_yaml" in ckpt
    assert ckpt["plate_config_yaml"] == yaml_str


def test_checkpoint_no_config_by_default() -> None:
    """build_checkpoint without plate_config_yaml has no key."""
    ckpt = build_checkpoint(
        epoch=1,
        model_state={},
        optimizer_state={},
        scheduler_state={},
        scaler_state={},
        best_metric=0.5,
    )
    assert "plate_config_yaml" not in ckpt


def test_validate_compat_match() -> None:
    """Compatible checkpoint returns no warnings."""
    cfg = _make_config()
    yaml_str = cfg.to_yaml_string()
    ckpt = build_checkpoint(
        epoch=1,
        model_state={},
        optimizer_state={},
        scheduler_state={},
        scaler_state={},
        best_metric=0.5,
        plate_config_yaml=yaml_str,
    )
    warns = validate_checkpoint_compat(ckpt, cfg)
    assert warns == []


def test_validate_compat_mismatch_countries() -> None:
    """Mismatched country_list raises ValueError."""
    cfg_saved = _make_config(countries=["RU"])
    cfg_current = _make_config(countries=["BY"])
    ckpt = build_checkpoint(
        epoch=1,
        model_state={},
        optimizer_state={},
        scheduler_state={},
        scaler_state={},
        best_metric=0.5,
        plate_config_yaml=cfg_saved.to_yaml_string(),
    )
    with pytest.raises(ValueError, match="countries"):
        validate_checkpoint_compat(ckpt, cfg_current)


def test_validate_compat_mismatch_alphabet() -> None:
    """Mismatched union_alphabet_size raises ValueError."""
    cfg_saved = PlateConfig(
        regions={
            "RU": RegionConfig(
                pattern=["A000AA"],
                valid_chars=ValidChars(letters="AB", digits="0123456789"),
            ),
        }
    )
    cfg_current = PlateConfig(
        regions={
            "RU": RegionConfig(
                pattern=["A000AA"],
                valid_chars=ValidChars(letters="ABC", digits="0123456789"),
            ),
        }
    )
    ckpt = build_checkpoint(
        epoch=1,
        model_state={},
        optimizer_state={},
        scheduler_state={},
        scaler_state={},
        best_metric=0.5,
        plate_config_yaml=cfg_saved.to_yaml_string(),
    )
    with pytest.raises(ValueError, match="alphabet_size"):
        validate_checkpoint_compat(ckpt, cfg_current)


def test_validate_compat_no_config() -> None:
    """Old checkpoint without plate_config_yaml returns warnings."""
    cfg = _make_config()
    ckpt = build_checkpoint(
        epoch=1,
        model_state={},
        optimizer_state={},
        scheduler_state={},
        scaler_state={},
        best_metric=0.5,
    )
    warns = validate_checkpoint_compat(ckpt, cfg)
    assert len(warns) == 1
    assert "no plate_config_yaml" in warns[0]
