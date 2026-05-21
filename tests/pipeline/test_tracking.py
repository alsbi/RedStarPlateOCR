"""Tests for TrackIO MetricsTracker with graceful degradation."""

from __future__ import annotations

import logging

from redstar_plate_ocr.pipeline.tracking import MetricsTracker


class TestMetricsTrackerDisabled:
    """Tests when tracker is disabled by config."""

    def test_disabled_by_config(self):
        tracker = MetricsTracker(enabled=False)
        assert not tracker.enabled

    def test_log_when_disabled(self, caplog):
        tracker = MetricsTracker(enabled=False)
        with caplog.at_level(logging.INFO):
            tracker.log({"loss": 0.5}, step=1)
        assert "loss=0.5000" in caplog.text

    def test_log_images_when_disabled(self):
        tracker = MetricsTracker(enabled=False)
        # Should not raise
        tracker.log_images({"img": None}, step=1)

    def test_finish_when_disabled(self):
        tracker = MetricsTracker(enabled=False)
        # Should not raise
        tracker.finish()


class TestMetricsTrackerFallback:
    """Tests when TrackIO is not installed (graceful fallback)."""

    def test_enabled_but_not_installed(self):
        if MetricsTracker.is_available():
            return
        tracker = MetricsTracker(project="test", name="run", enabled=True)
        # Should be disabled because trackio not installed
        assert not tracker.enabled

    def test_log_falls_back_to_logging(self, caplog):
        if MetricsTracker.is_available():
            return
        tracker = MetricsTracker(enabled=True)
        with caplog.at_level(logging.INFO):
            tracker.log({"cer": 0.1}, step=5)
        assert "cer=0.1000" in caplog.text


class TestMetricsTrackerIsAvailable:
    """Test static method."""

    def test_is_available_returns_bool(self):
        result = MetricsTracker.is_available()
        assert isinstance(result, bool)
