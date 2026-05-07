"""Tests for PostProcessor."""

from __future__ import annotations

from redstar_plate_ocr.plate.config import (
    PlateConfig,
    RegionConfig,
    ValidChars,
)
from redstar_plate_ocr.plate.postprocessor import PostProcessor
from redstar_plate_ocr.plate.results import RawResult


def _make_config(
    forbidden: list[str] | None = None,
) -> PlateConfig:
    """Create a minimal PlateConfig for testing (RU)."""
    return PlateConfig(
        regions={
            "RU": RegionConfig(
                pattern=["X000XX00"],
                valid_chars=ValidChars(
                    letters="ABEKMHOPCTYX",
                    digits="0123456789",
                ),
                forbidden_combos=forbidden or [],
            ),
        },
    )


def _make_kg_config() -> PlateConfig:
    """Create PlateConfig for KG with multiple patterns."""
    return PlateConfig(
        regions={
            "KG": RegionConfig(
                pattern=["X0000XX", "000000XXX"],
                valid_chars=ValidChars(
                    letters="ABDEHIKLMNOPRSTUVWXYZ",
                    digits="0123456789",
                ),
            ),
        },
    )


def test_process_no_forbidden_no_pattern_change() -> None:
    """PostProcessor returns result unchanged when no rules apply."""
    config = _make_config()
    pp = PostProcessor(config)
    raw = RawResult(
        text="A123BC77",
        text_confidence=0.9,
        country="RU",
        country_confidence=0.95,
        plate_type="standard",
    )
    result = pp.process(raw)
    assert result.text == "A123BC77"
    assert not result.needs_review
    assert result.text_confidence == 0.9
    assert result.country == "RU"


def test_process_forbidden_marks_review() -> None:
    """PostProcessor sets needs_review when text has forbidden combo."""
    config = _make_config(forbidden=["BC7"])
    pp = PostProcessor(config)
    raw = RawResult(
        text="A123BC77",
        text_confidence=0.9,
        country="RU",
        country_confidence=0.95,
        plate_type="standard",
    )
    result = pp.process(raw)
    assert result.needs_review is True


def test_process_forbidden_with_hypotheses_selects_valid() -> None:
    """PostProcessor selects best valid hypothesis via forbidden."""
    config = _make_config(forbidden=["77"])
    pp = PostProcessor(config)
    raw = RawResult(
        text="A123BC77",
        text_confidence=0.9,
        country="RU",
        country_confidence=0.95,
        plate_type="standard",
    )
    hypotheses = [
        ("A123BC77", 0.9),
        ("A123BC78", 0.8),
    ]
    result = pp.process(raw, hypotheses=hypotheses)
    assert result.text == "A123BC78"
    assert not result.needs_review


def test_process_forbidden_all_hypotheses_invalid() -> None:
    """All hypotheses contain forbidden -> needs_review=True."""
    config = _make_config(forbidden=["BC"])
    pp = PostProcessor(config)
    raw = RawResult(
        text="A123BC77",
        text_confidence=0.9,
        country="RU",
        country_confidence=0.95,
        plate_type="standard",
    )
    hypotheses = [
        ("A123BC77", 0.9),
        ("A123BC78", 0.8),
    ]
    result = pp.process(raw, hypotheses=hypotheses)
    assert result.needs_review is True


def test_process_unknown_country_no_region() -> None:
    """Unknown country -> no forbidden/pattern, needs_review from raw."""
    config = _make_config()
    pp = PostProcessor(config)
    raw = RawResult(
        text="ABC",
        text_confidence=0.5,
        country="XX",
        country_confidence=0.3,
        plate_type="standard",
        needs_review=True,
    )
    result = pp.process(raw)
    assert result.text == "ABC"
    assert result.needs_review is True


def test_process_pattern_validation_corrects_text() -> None:
    """Pattern validator corrects invalid characters."""
    config = _make_config()
    pp = PostProcessor(config)
    raw = RawResult(
        text="1123BC77",
        text_confidence=0.9,
        country="RU",
        country_confidence=0.95,
        plate_type="standard",
    )
    result = pp.process(raw)
    assert result.text != "1123BC77"


def test_process_preserves_confidences() -> None:
    """PostProcessor preserves confidence values from raw."""
    config = _make_config()
    pp = PostProcessor(config)
    raw = RawResult(
        text="A123BC77",
        text_confidence=0.85,
        country="RU",
        country_confidence=0.72,
        plate_type="standard",
    )
    result = pp.process(raw)
    assert result.text_confidence == 0.85
    assert result.country_confidence == 0.72
    assert result.country == "RU"
    assert result.plate_type == "standard"


def test_process_kg_standard_7char() -> None:
    """KG 7-char text matches X0000XX, no correction."""
    config = _make_kg_config()
    pp = PostProcessor(config)
    raw = RawResult(
        text="E2695BP",
        text_confidence=0.9,
        country="KG",
        country_confidence=0.95,
        plate_type="standard",
    )
    result = pp.process(raw)
    assert result.text == "E2695BP"
    assert not result.needs_review


def test_process_kg_standard_9char() -> None:
    """KG 9-char text matches 000000XXX, no correction."""
    config = _make_kg_config()
    pp = PostProcessor(config)
    raw = RawResult(
        text="069759DLI",
        text_confidence=0.9,
        country="KG",
        country_confidence=0.95,
        plate_type="standard",
    )
    result = pp.process(raw)
    assert result.text == "069759DLI"
    assert not result.needs_review
