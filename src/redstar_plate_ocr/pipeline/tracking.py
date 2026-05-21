"""TrackIO integration with graceful degradation."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_trackio_available = False
try:
    import trackio as _trackio_module  # noqa: F401

    _trackio_available = True
except ImportError:
    pass


class MetricsTracker:
    """Lightweight wrapper around TrackIO with graceful fallback."""

    def __init__(
        self,
        project: str | None = None,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled and _trackio_available
        self._project = project
        self._name = name

        if not enabled:
            logger.info("TrackIO disabled by configuration")
            return

        if not _trackio_available:
            logger.warning(
                "TrackIO not installed — "
                "metrics will be logged to console only"
            )
            return

        if project and name:
            import trackio

            trackio.init(project=project, name=name, config=config or {})
            logger.info(
                "TrackIO initialized: project=%s, name=%s",
                project,
                name,
            )

    def log(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log metrics dict."""
        if not self.enabled:
            parts = [
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sorted(metrics.items())
            ]
            logger.info(f"[metrics step={step}] {', '.join(parts)}")
            return

        import trackio

        trackio.log(metrics, step=step)

    def log_images(
        self,
        images: dict[str, Any],
        step: int | None = None,
    ) -> None:
        """Log images for visual debugging."""
        if not self.enabled:
            return

        import trackio

        trackio.log(images, step=step)

    def finish(self) -> None:
        """Finish tracking session."""
        if not self.enabled:
            return
        try:
            import trackio

            trackio.finish()
        except Exception as e:
            logger.warning("TrackIO finish error: %s", e)

    @staticmethod
    def is_available() -> bool:
        """Return ``True`` if TrackIO package is installed."""
        return _trackio_available
