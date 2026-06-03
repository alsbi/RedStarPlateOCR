"""TrackIO integration with graceful degradation."""

from __future__ import annotations

import logging
import subprocess
import threading
import webbrowser
from queue import Empty, Full, Queue
from typing import Any

logger = logging.getLogger(__name__)

_trackio_available = False
try:
    import trackio as _trackio_module  # noqa: F401

    _trackio_available = True
except ImportError:
    pass


class DashboardLauncher:
    """Launches TrackIO dashboard as a subprocess.

    Uses subprocess instead of in-process thread to avoid
    SQLite/GIL contention with the training loop.
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
        self._process: subprocess.Popen | None = None
        self._url: str | None = None

    def start(self) -> None:
        """Start the dashboard server as a subprocess."""
        if not _trackio_available:
            logger.warning("TrackIO not installed — cannot launch dashboard")
            return

        import sys

        cmd = [
            sys.executable,
            "-m",
            "trackio",
            "show",
            "--project",
            self._project,
        ]
        if self._port is not None:
            cmd.extend(["--server-port", str(self._port)])

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            port = self._port or 7860
            self._url = f"http://localhost:{port}"
            logger.info("TrackIO dashboard started at %s", self._url)

            if self._open_browser and self._url:

                def _open() -> None:
                    import time

                    time.sleep(4)
                    webbrowser.open(self._url)  # type: ignore[arg-type]

                threading.Thread(target=_open, daemon=True).start()
        except Exception as e:
            logger.warning("Failed to launch TrackIO dashboard: %s", e)

    def stop(self) -> None:
        """Stop the dashboard server."""
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None
            logger.info("TrackIO dashboard stopped")

    @property
    def url(self) -> str | None:
        """Return the dashboard URL if running."""
        return self._url


class MetricsTracker:
    """Lightweight wrapper around TrackIO with graceful fallback.

    Key design decisions for non-blocking operation:

    - Uses ``embed=False`` and ``auto_log_gpu=False`` in
      ``trackio.init()`` to prevent background threads that compete
      for SQLite locks.
    - Uses an async queue + background writer thread so that
      ``trackio.log()`` never blocks the training loop.
    """

    _QUEUE_MAX_SIZE = 1024

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
        self._queue: Queue[tuple[dict[str, float], int | None]] | None = None
        self._writer_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

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

            # CRITICAL: embed=False prevents auto-dashboard launch;
            # auto_log_gpu=False prevents background GPU logging thread.
            # Both avoid SQLite contention that causes hangs on
            # Apple Silicon.
            trackio.init(
                project=project,
                name=name,
                config=config or {},
                embed=False,
                auto_log_gpu=False,
            )
            logger.info(
                "TrackIO initialized: project=%s, name=%s",
                project,
                name,
            )

            # Start async writer thread
            self._queue = Queue(maxsize=self._QUEUE_MAX_SIZE)
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name="trackio-writer",
            )
            self._writer_thread.start()

    def _writer_loop(self) -> None:
        """Background thread that drains the metrics queue."""
        assert self._queue is not None
        import trackio

        while not self._stop_event.is_set():
            try:
                metrics, step = self._queue.get(timeout=1.0)
            except Empty:
                continue
            try:
                trackio.log(metrics, step=step)
            except Exception as e:
                logger.warning("TrackIO log error: %s", e)

        # Drain remaining items after stop signal
        while not self._queue.empty():
            try:
                metrics, step = self._queue.get_nowait()
                trackio.log(metrics, step=step)
            except Exception:
                break

    def log(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log metrics dict (non-blocking via queue)."""
        if not self.enabled:
            parts = [
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sorted(metrics.items())
            ]
            logger.info(f"[metrics step={step}] {', '.join(parts)}")
            return

        if self._queue is not None:
            try:
                self._queue.put_nowait((metrics, step))
            except Full:
                logger.warning(
                    "TrackIO queue full — dropping metrics step=%s",
                    step,
                )

    def log_images(
        self,
        images: dict[str, Any],
        step: int | None = None,
    ) -> None:
        """Log images for visual debugging (async via queue)."""
        if not self.enabled:
            return

        if self._queue is not None:
            try:
                self._queue.put_nowait((images, step))
            except Full:
                logger.warning(
                    "TrackIO queue full — dropping images step=%s",
                    step,
                )

    def finish(self) -> None:
        """Finish tracking session."""
        if not self.enabled:
            return

        # Signal writer thread to stop and wait for drain
        if self._stop_event is not None:
            self._stop_event.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.0)

        try:
            import trackio

            trackio.finish()
        except Exception as e:
            logger.warning("TrackIO finish error: %s", e)

    @staticmethod
    def is_available() -> bool:
        """Return ``True`` if TrackIO package is installed."""
        return _trackio_available
