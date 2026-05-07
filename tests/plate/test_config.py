"""Tests for plate configuration models."""

import pytest

from redstar_plate_ocr.plate.config import (
    PlateConfig,
    RegionConfig,
    ValidChars,
)


@pytest.fixture
def plate_config() -> PlateConfig:
    """Load PlateConfig from configs/plate.yaml."""
    return PlateConfig.from_yaml("configs/plate.yaml")


def test_load_plate_config(plate_config: PlateConfig):
    """Config loads from YAML without errors."""
    assert plate_config.regions
    assert "RU" in plate_config.regions


# --- RegionConfig unit tests ---


def test_coerce_pattern_str_to_list():
    """String pattern auto-wraps to list."""
    region = RegionConfig(
        pattern="X000XX00o",
        valid_chars=ValidChars(letters="ABC", digits="012"),
    )
    assert region.pattern == ["X000XX00o"]


def test_pattern_list_accepted():
    """List pattern accepted as-is."""
    region = RegionConfig(
        pattern=["X0000XX", "000000XXX"],
        valid_chars=ValidChars(letters="ABC", digits="012"),
    )
    assert region.pattern == ["X0000XX", "000000XXX"]


def test_get_patterns_returns_copy():
    """get_patterns() returns a copy, not mutable reference."""
    region = RegionConfig(
        pattern=["X0000XX", "000000XXX"],
        valid_chars=ValidChars(letters="ABC", digits="012"),
    )
    patterns = region.get_patterns()
    patterns.append("HACKED")
    assert region.pattern == ["X0000XX", "000000XXX"]


def test_get_patterns_returns_all():
    """get_patterns() returns all patterns."""
    region = RegionConfig(
        pattern=["X0000XX", "000000XXX"],
        valid_chars=ValidChars(letters="ABC", digits="012"),
    )
    assert region.get_patterns() == ["X0000XX", "000000XXX"]


def test_pattern_empty_list_raises():
    """Empty pattern list raises ValueError."""
    with pytest.raises(ValueError, match="pattern must not be empty"):
        RegionConfig(
            pattern=[],
            valid_chars=ValidChars(letters="ABC", digits="012"),
        )


def test_pattern_empty_string_raises():
    """Pattern with empty string raises ValueError."""
    with pytest.raises(
        ValueError, match="pattern must not contain empty strings"
    ):
        RegionConfig(
            pattern=[""],
            valid_chars=ValidChars(letters="ABC", digits="012"),
        )


def test_no_duplicate_chars(plate_config: PlateConfig):
    """No duplicate characters in any region's alphabet."""
    for code, region in plate_config.regions.items():
        alphabet = region.raw_alphabet()
        assert len(alphabet) == len(set(alphabet)), f"Duplicates in {code}"


def test_kz_has_two_patterns(plate_config: PlateConfig):
    """KZ has both standard and square patterns."""
    assert plate_config.regions["KZ"].pattern == [
        "000XXX00",
        "00000XXX",
    ]


def test_uz_has_two_patterns(plate_config: PlateConfig):
    """UZ has both standard and square patterns."""
    assert plate_config.regions["UZ"].pattern == [
        "00X000XX",
        "X000XX00",
    ]


def test_kg_has_two_patterns(plate_config: PlateConfig):
    """KG has both standard and square patterns."""
    assert plate_config.regions["KG"].pattern == [
        "X0000XX",
        "00000XXX",
    ]


def test_ua_has_single_pattern(plate_config: PlateConfig):
    """UA has single 8-char pattern."""
    assert plate_config.regions["UA"].pattern == [
        "XX0000XX",
    ]


# --- Field: enabled ---


def test_enabled_filters_country_list():
    """Countries with enabled=false are excluded from country_list."""
    cfg = PlateConfig(
        regions={
            "RU": RegionConfig(
                pattern="X000XX00o",
                valid_chars=ValidChars(letters="ABC", digits="012"),
            ),
            "AM": RegionConfig(
                pattern="00XX000",
                valid_chars=ValidChars(letters="DEF", digits="345"),
                enabled=False,
            ),
        }
    )
    assert cfg.country_list == ["RU"]
    assert cfg.num_countries == 1


def test_enabled_filters_union_alphabet():
    """Alphabet of disabled country is not in union_alphabet."""
    cfg = PlateConfig(
        regions={
            "RU": RegionConfig(
                pattern="X000XX00o",
                valid_chars=ValidChars(letters="ABC", digits="012"),
            ),
            "AM": RegionConfig(
                pattern="00XX000",
                valid_chars=ValidChars(letters="DEF", digits="345"),
                enabled=False,
            ),
        }
    )
    assert "D" not in cfg.union_alphabet
    assert "A" in cfg.union_alphabet


def test_enabled_default_true():
    """enabled defaults to True."""
    region = RegionConfig(
        pattern="X000XX00o",
        valid_chars=ValidChars(letters="ABC", digits="012"),
    )
    assert region.enabled is True
