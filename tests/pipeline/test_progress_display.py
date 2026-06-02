"""Tests for ProgressDisplay and format_epoch_stats."""

from __future__ import annotations

from rich.console import Console

from redstar_plate_ocr.pipeline.progress_display import ProgressDisplay
from redstar_plate_ocr.pipeline.utils import (
    _country_flag,
    _format_per_country,
    format_epoch_stats,
)

# ── _country_flag ───────────────────────────────────


def test_country_flag_by() -> None:
    """BY → 🇧🇾"""
    assert _country_flag("BY").strip() == "🇧🇾"


def test_country_flag_ru() -> None:
    """RU → 🇷🇺"""
    assert _country_flag("RU").strip() == "🇷🇺"


def test_country_flag_unknown() -> None:
    """Non-2-letter code returned as-is."""
    assert _country_flag("XYZ").strip() == "XYZ"


# ── _format_per_country ────────────────────────────


def test_format_per_country_basic() -> None:
    """Format 7 countries from val_region_* keys."""
    metrics: dict[str, float] = {
        "val_region_BY": 0.982,
        "val_region_GE": 0.727,
        "val_region_KZ": 0.85,
        "val_region_KG": 0.60,
        "val_region_RU": 0.91,
        "val_region_UA": 0.78,
        "val_region_UZ": 0.55,
    }
    result = _format_per_country(metrics)
    # Flags present with .1% format
    assert "🇧🇾" in result
    assert "98.2%" in result
    assert "🇬🇪" in result
    assert "72.7%" in result
    # All countries accounted for (7 flag emojis)
    for code in ["BY", "GE", "KZ", "KG", "RU", "UA", "UZ"]:
        assert _country_flag(code).strip() in result


def test_format_per_country_empty() -> None:
    """No val_region_* keys → empty string."""
    metrics: dict[str, float] = {
        "val_plate_accuracy": 0.9,
        "val_cer": 0.05,
    }
    assert _format_per_country(metrics) == ""


def test_format_per_country_partial() -> None:
    """Only some countries present — flags + compact percent."""
    metrics: dict[str, float] = {
        "val_region_BY": 0.95,
        "val_region_RU": 0.88,
    }
    result = _format_per_country(metrics)
    assert "🇧🇾" in result and "95.0%" in result
    assert "🇷🇺" in result and "88.0%" in result


# ── format_epoch_stats ─────────────────────────────


def test_format_epoch_stats_with_per_country() -> None:
    """Full string with per-country section."""
    val_metrics: dict[str, float] = {
        "val_plate_accuracy": 0.92,
        "val_cer": 0.03,
        "val_char_accuracy": 0.97,
        "val_country_accuracy": 0.95,
        "val_format_accuracy": 0.98,
        "val_standard_accuracy": 0.90,
        "val_square_accuracy": 0.80,
        "val_region_BY": 0.98,
        "val_region_GE": 0.72,
    }
    best_metrics: dict[str, float] = {
        "val_plate_accuracy": 0.90,
        "val_cer": 0.04,
    }
    result = format_epoch_stats(
        val_metrics,
        best_metrics,
        train_loss=0.5,
        epoch_duration=120.0,
    )
    # Three sections separated by │
    sections = result.split(" │ ")
    assert len(sections) == 3
    # Section 1: core metrics (.1% format)
    assert "plate=92.0%↑" in sections[0]
    assert "cer=0.0300↓" in sections[0]
    # Section 2: per-country with flags
    assert "🇧🇾" in sections[1]
    assert "98.0%" in sections[1]
    assert "🇬🇪" in sections[1]
    assert "72.0%" in sections[1]
    # Section 3: system
    assert "loss=0.5000" in sections[2]
    assert "2m00s" in sections[2]


def test_format_epoch_stats_without_per_country() -> None:
    """String without per-country → two sections."""
    val_metrics: dict[str, float] = {
        "val_plate_accuracy": 0.92,
        "val_cer": 0.03,
        "val_char_accuracy": 0.97,
        "val_country_accuracy": 0.95,
        "val_format_accuracy": 0.98,
        "val_standard_accuracy": 0.85,
        "val_square_accuracy": 0.75,
    }
    best_metrics: dict[str, float] = {
        "val_plate_accuracy": 0.90,
        "val_cer": 0.04,
    }
    result = format_epoch_stats(
        val_metrics,
        best_metrics,
        train_loss=0.5,
    )
    sections = result.split(" │ ")
    assert len(sections) == 2
    assert "plate=92.0%↑" in sections[0]
    assert "loss=0.5000" in sections[1]


def test_format_epoch_stats_first_epoch() -> None:
    """First epoch: no ↑/↓ tags (is_first=True)."""
    val_metrics: dict[str, float] = {
        "val_plate_accuracy": 0.80,
        "val_cer": 0.10,
        "val_char_accuracy": 0.90,
        "val_country_accuracy": 0.85,
        "val_format_accuracy": 0.88,
        "val_standard_accuracy": 0.70,
        "val_square_accuracy": 0.60,
    }
    result = format_epoch_stats(val_metrics, {}, train_loss=1.0)
    assert "↑" not in result
    assert "↓" not in result
    assert "plate=80.0%" in result


def test_format_epoch_stats_cached() -> None:
    """Cached tag appears when is_cached=True."""
    val_metrics: dict[str, float] = {
        "val_plate_accuracy": 0.80,
        "val_cer": 0.10,
        "val_char_accuracy": 0.90,
        "val_country_accuracy": 0.85,
        "val_format_accuracy": 0.88,
        "val_standard_accuracy": 0.70,
        "val_square_accuracy": 0.60,
    }
    result = format_epoch_stats(
        val_metrics,
        {},
        train_loss=1.0,
        is_cached=True,
    )
    assert "(cached)" in result


# ── ProgressDisplay lifecycle ───────────────────────


def test_progress_display_lifecycle() -> None:
    """Enter/exit, add/update/remove batch task."""
    display = ProgressDisplay(total_epochs=5)
    with display:
        task_id = display.add_batch_task(
            description="Batches",
            total=10,
            stats="",
        )
        assert isinstance(task_id, int)
        display.update_batch(
            task_id,
            advance=3,
            stats="loss=0.5",
        )
        display.remove_batch_task(task_id)


def test_progress_display_epoch_summary() -> None:
    """update_epoch_summary() updates the epoch text."""
    display = ProgressDisplay(total_epochs=5)
    with display:
        display.update_epoch_summary("Epoch 1/5 plate=90%")
        assert display._epoch_text.plain == "Epoch 1/5 plate=90%"
        display.update_epoch_summary("Epoch 2/5 plate=92%")
        assert display._epoch_text.plain == "Epoch 2/5 plate=92%"


# ── _OptionalText regression ───────────────────────


def test_optional_text_empty() -> None:
    """Empty plain text yields nothing from __rich_console__."""
    from redstar_plate_ocr.pipeline.progress_display import _OptionalText

    obj = _OptionalText("")
    console = Console(width=80)
    options = console.options
    results = list(obj.__rich_console__(console, options))
    assert results == []


def test_optional_text_no_wrap() -> None:
    """Non-empty text yields a Rich Text with no_wrap=True."""
    from rich.text import Text

    from redstar_plate_ocr.pipeline.progress_display import _OptionalText

    obj = _OptionalText("hello")
    console = Console(width=80)
    results = list(obj.__rich_console__(console, console.options))
    assert len(results) == 1
    assert isinstance(results[0], Text)
    assert results[0].no_wrap is True
    assert results[0].plain == "hello"


def test_optional_text_truncates_wide_emoji() -> None:
    """Long lines with wide emoji are truncated to fit console width.

    Regression for terminal line-wrapping with country flags (🇧🇾 etc.).
    When _OptionalText renders a too-long line and the console width is
    constrained the yielded Rich Text must be truncated and no_wrap=True.
    """
    from rich.text import Text

    from redstar_plate_ocr.pipeline.progress_display import _OptionalText

    text = "🎯 plate=88.2% │ 🇧🇾 89.9% 🇬🇪 46.3% 🇰🇬 97.7% │ 📉 loss=1.4042"
    obj = _OptionalText(text)
    console = Console(width=40)
    results = list(obj.__rich_console__(console, console.options))
    assert len(results) == 1
    assert isinstance(results[0], Text)
    assert results[0].no_wrap is True
    assert results[0].cell_len <= 40


def test_optional_text_no_trunc_when_wide_enough() -> None:
    """When max_width exceeds text length nothing is truncated."""
    from rich.text import Text

    from redstar_plate_ocr.pipeline.progress_display import _OptionalText

    text = "short line"
    obj = _OptionalText(text)
    console = Console(width=200)
    results = list(obj.__rich_console__(console, console.options))
    assert isinstance(results[0], Text)
    assert results[0].plain == "short line"
