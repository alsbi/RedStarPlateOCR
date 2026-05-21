"""TrackIO integration with graceful degradation."""

from __future__ import annotations

import logging
import webbrowser
from typing import Any

logger = logging.getLogger(__name__)

_trackio_available = False
try:
    import trackio as _trackio_module  # noqa: F401

    _trackio_available = True
except ImportError:
    pass


class DashboardLauncher:
    """Launches TrackIO dashboard as a background process.

    Uses ``trackio.show()`` with ``block_thread=False`` so the
    dashboard runs in a separate thread while training continues.
    """

    def __init__(
        self,
        project: str,
        port: int | None = None,
        open_browser: bool = True,
    ) -> None:
        self._project = project
        self._port = port
        self._open_browser = open_browser
        self._app: object | None = None
        self._url: str | None = None

    def start(self) -> None:
        """Start the dashboard server in background."""
        if not _trackio_available:
            logger.warning("TrackIO not installed — cannot launch dashboard")
            return

        try:
            import trackio

            kwargs: dict[str, Any] = {
                "project": self._project,
                "open_browser": False,
                "block_thread": False,
            }
            if self._port is not None:
                kwargs["server_port"] = self._port

            app, url, _share_url, _full_url = trackio.show(**kwargs)
            self._app = app
            self._url = url
            logger.info("TrackIO dashboard started at %s", url)

            if self._open_browser and url:
                import threading

                def _open() -> None:
                    import time

                    time.sleep(3)
                    webbrowser.open(url)

                threading.Thread(target=_open, daemon=True).start()
        except Exception as e:
            logger.warning("Failed to launch TrackIO dashboard: %s", e)

    def stop(self) -> None:
        """Stop the dashboard server."""
        if self._app is not None:
            try:
                # trackio show returns an app with .close()
                close = getattr(self._app, "close", None)
                if callable(close):
                    close()
                    logger.info("TrackIO dashboard stopped")
            except Exception as e:
                logger.warning("Error stopping TrackIO dashboard: %s", e)
            finally:
                self._app = None

    @property
    def url(self) -> str | None:
        """Return the dashboard URL if running."""
        return self._url


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
