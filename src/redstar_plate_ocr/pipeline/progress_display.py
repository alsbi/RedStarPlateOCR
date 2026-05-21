"""Training progress display: epoch summary + batch bar + validation marker."""

from __future__ import annotations

from types import TracebackType

from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TaskID, TextColumn
from rich.text import Text


class ProgressDisplay:
    """Three-line training progress display.

    Line 1: Epoch summary (val metrics + loss + ETA)
    Line 2: Batch progress bar (during training)
    Line 3: Validation marker (temporary, during validation)
    """

    def __init__(
        self,
        total_epochs: int,
        refresh_per_second: float = 5,
    ) -> None:
        self._epoch_text = Text("")
        self._batch_progress = Progress(
            TextColumn("  {task.description}"),
            BarColumn(bar_width=30),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("│ {task.fields[stats]}"),
            refresh_per_second=refresh_per_second,
        )
        self._warmup_text = Text("")
        self._validation_text = Text("")
        self._group = Group(
            self._epoch_text,
            self._batch_progress,
            self._warmup_text,
            self._validation_text,
        )
        self._live = Live(
            self._group,
            refresh_per_second=refresh_per_second,
        )

    def __enter__(self) -> ProgressDisplay:
        self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._live.__exit__(exc_type, exc_val, exc_tb)

    # ── Epoch line ──────────────────────────────────

    def update_epoch_summary(self, text: str) -> None:
        """Update line 1 (epoch summary text)."""
        self._epoch_text.plain = text

    # ── Batch line ───────────────────────────────────

    def add_batch_task(
        self,
        description: str = "Batches",
        total: int = 0,
        stats: str = "",
    ) -> TaskID:
        """Add batch progress task (line 2)."""
        return self._batch_progress.add_task(
            description,
            total=total,
            stats=stats,
        )

    def update_batch(
        self,
        task_id: TaskID,
        *,
        advance: float = 0,
        stats: str = "",
    ) -> None:
        """Update batch task."""
        self._batch_progress.update(
            task_id,
            advance=advance,
            stats=stats,
        )

    def remove_batch_task(self, task_id: TaskID) -> None:
        """Remove batch task."""
        self._batch_progress.remove_task(task_id)

    # ── Warmup status line ──────────────────────────

    def update_warmup_status(self, text: str) -> None:
        """Update warmup status line (between batch bar and
        validation marker).  Only shown when ``enable_warmup``
        is True — call with empty string to clear.
        """
        self._warmup_text.plain = text

    def hide_warmup_status(self) -> None:
        """Remove warmup status line."""
        self._warmup_text.plain = ""

    # ── Validation marker ────────────────────────────

    def show_validation(self, text: str = "⏳ Validating...") -> None:
        """Show validation marker (line 3) without changing epoch summary."""
        self._validation_text.plain = text

    def hide_validation(self) -> None:
        """Remove validation marker."""
        self._validation_text.plain = ""

    # ── Console access ──────────────────────────────

    @property
    def console(self) -> Console:
        """Access Rich console for printing panels etc."""
        return self._batch_progress.console
