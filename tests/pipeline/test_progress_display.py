"""Tests for ProgressDisplay and format_epoch_stats."""

from __future__ import annotations

from redstar_plate_ocr.pipeline.progress_display import ProgressDisplay
from redstar_plate_ocr.pipeline.utils import (
    _format_per_country,
    format_epoch_stats,
)

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
    assert "BY=98.200%" in result
    assert "GE=72.700%" in result
    assert "KZ=85.000%" in result
    assert "KG=60.000%" in result
    assert "RU=91.000%" in result
    assert "UA=78.000%" in result
    assert "UZ=55.000%" in result
    # Sorted order
    parts = result.split(" ")
    assert len(parts) == 7


def test_format_per_country_empty() -> None:
    """No val_region_* keys → empty string."""
    metrics: dict[str, float] = {
        "val_plate_accuracy": 0.9,
        "val_cer": 0.05,
    }
    assert _format_per_country(metrics) == ""


def test_format_per_country_partial() -> None:
    """Only some countries present."""
    metrics: dict[str, float] = {
        "val_region_BY": 0.95,
        "val_region_RU": 0.88,
    }
    result = _format_per_country(metrics)
    assert result == "BY=95.000% RU=88.000%"


# ── format_epoch_stats ─────────────────────────────


def test_format_epoch_stats_with_per_country() -> None:
    """Full string with per-country section."""
    val_metrics: dict[str, float] = {
        "val_plate_accuracy": 0.92,
        "val_cer": 0.03,
        "val_char_accuracy": 0.97,
        "val_country_accuracy": 0.95,
        "val_format_accuracy": 0.98,
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
    # Section 1: core metrics
    assert "plate=92.000%↑" in sections[0]
    assert "cer=0.0300↓" in sections[0]
    # Section 2: per-country
    assert "BY=98.000%" in sections[1]
    assert "GE=72.000%" in sections[1]
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
    assert "plate=92.000%↑" in sections[0]
    assert "loss=0.5000" in sections[1]


def test_format_epoch_stats_first_epoch() -> None:
    """First epoch: no ↑/↓ tags (is_first=True)."""
    val_metrics: dict[str, float] = {
        "val_plate_accuracy": 0.80,
        "val_cer": 0.10,
        "val_char_accuracy": 0.90,
        "val_country_accuracy": 0.85,
        "val_format_accuracy": 0.88,
    }
    result = format_epoch_stats(val_metrics, {}, train_loss=1.0)
    assert "↑" not in result
    assert "↓" not in result
    assert "plate=80.000%" in result


def test_format_epoch_stats_cached() -> None:
    """Cached tag appears when is_cached=True."""
    val_metrics: dict[str, float] = {
        "val_plate_accuracy": 0.80,
        "val_cer": 0.10,
        "val_char_accuracy": 0.90,
        "val_country_accuracy": 0.85,
        "val_format_accuracy": 0.88,
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
