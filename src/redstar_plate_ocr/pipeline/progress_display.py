"""Training progress display with visual warmup transition bars."""

from __future__ import annotations

from types import TracebackType

from rich.console import Console, Group, RenderResult
from rich.live import Live
from rich.progress import BarColumn, Progress, TaskID, TextColumn
from rich.text import Text


def _mini_bar(
    value: float,
    width: int = 10,
    fill: str = "█",
    empty: str = "░",
) -> str:
    """Render a compact text progress bar for *value* in [0, 1]."""
    filled = round(value * width)
    return fill * filled + empty * (width - filled)


class _OptionalText:
    """Rich renderable that yields nothing when plain is empty.

    Unlike ``Text("")`` which still takes a line in a :class:`Group`,
    this collapses completely when the text is blank.

    Automatically truncates each line to the available console width
    so that terminal line-wrapping does not break :class:`Live`
    rendering when wide emoji (country flags) are present.
    """

    def __init__(self, text: str = "") -> None:
        self.plain = text

    def __rich_console__(
        self,
        console: Console,
        options,
    ) -> RenderResult:
        if not self.plain:
            return
        max_width = options.max_width
        if max_width and max_width > 0:
            lines = self.plain.split("\n")
            truncated_lines = []
            for line in lines:
                text = Text(line, no_wrap=True)
                if text.cell_len > max_width:
                    text.truncate(max_width, overflow="ellipsis")
                truncated_lines.append(text.plain)
            yield Text("\n".join(truncated_lines), no_wrap=True)
        else:
            yield Text(self.plain, no_wrap=True)


class ProgressDisplay:
    """Rich training progress display.

    Layout (top to bottom):
      1. Epoch summary: val metrics + loss + ETA
      2. Warmup panel (conditional): severe→std transition bars + counters
      3. Batch progress bar + per-batch stats with grad norm / accum info
      4. Validation marker (temporary, during validation)

    Empty lines (warmup / validation) collapse automatically so there
    are no visual gaps when they are not active.
    """

    def __init__(
        self,
        total_epochs: int,
        refresh_per_second: float = 5,
    ) -> None:
        self._total_epochs = total_epochs

        # ── Line renderables (collapsible when empty) ────
        self._epoch_text = _OptionalText()
        self._warmup_text = _OptionalText()
        self._validation_text = _OptionalText()

        # ── Batch progress bar ──────────────────────────
        self._batch_progress = Progress(
            TextColumn("  {task.description}"),
            BarColumn(bar_width=30),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("│ {task.fields[stats]}"),
            refresh_per_second=refresh_per_second,
        )

        # ── Live group ──────────────────────────────────
        self._group = Group(
            self._epoch_text,
            self._warmup_text,
            self._batch_progress,
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
        # Ensure Rich Live always stops — even on KeyboardInterrupt
        try:
            self._live.stop()
        except Exception:
            pass

    # ── Epoch summary line ──────────────────────────────

    def update_epoch_summary(self, text: str) -> None:
        """Update epoch metrics summary line."""
        self._epoch_text.plain = text

    def show_stopping(self, force: bool = False) -> None:
        """Show stopping message in progress display."""
        label = "Force stopping..." if force else "Stopping..."
        self._validation_text.plain = f"⏹ {label}"

    # ── Batch line ───────────────────────────────────────

    def add_batch_task(
        self,
        description: str = "Batches",
        total: int = 0,
        stats: str = "",
    ) -> TaskID:
        """Add batch progress task."""
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

    # ── Warmup status panel ─────────────────────────────

    def update_warmup_status(self, text: str) -> None:
        """Update warmup status from pre-formatted string (backward-compat)."""
        self._warmup_text.plain = text

    def update_warmup_detail(
        self,
        *,
        severe_severity: float = 0.0,
        std_severity: float = 0.0,
        preprocessing_enabled: bool = False,
        best_word_acc: float = 0.0,
        epochs_without_improvement: int = 0,
        patience_severe: int = 1,
        phase: str = "",
        lr_warmup_epoch: int | None = None,
        lr_warmup_total: int | None = None,
    ) -> None:
        """Render detailed warmup panel with visual transition bars.

        Shows a two-row display:
          Row 1: Severe ████████░░ 85% → Std ░░░░░░░░░░  0% Preproc=off
          Row 2: no_improve=3/10 best_acc=12.0% LR_ramp=3/5

        When *lr_warmup_epoch* / *lr_warmup_total* are given but
        severe/std bars are zero (plain LR warmup without adaptive aug),
        a compact single-line display is shown instead.
        """
        # ── Plain LR warmup (no severe scheduler) ───────
        if (
            lr_warmup_epoch is not None
            and lr_warmup_total is not None
            and severe_severity == 0.0
            and std_severity == 0.0
        ):
            self._warmup_text.plain = (
                f"  📈  LR warmup {lr_warmup_epoch}/{lr_warmup_total}"
                f" │ Phase={phase or 'Warmup'}"
            )
            return
        # ── Severe → Std transition bar ────────────────
        sev_pct = int(severe_severity * 100)
        std_pct = int(std_severity * 100)
        sev_bar = _mini_bar(severe_severity, width=10)
        std_bar = _mini_bar(std_severity, width=10)
        preproc_str = "on" if preprocessing_enabled else "off"

        # Choose emoji based on dominant aug mode
        if severe_severity > 0:
            icon = "🔥  "
        elif std_severity > 0:
            icon = "⚡  "
        else:
            icon = "✅  "

        row1 = (
            f"  {icon}Severe {sev_bar} {sev_pct:3d}%"
            f" → Std {std_bar} {std_pct:3d}%"
            f" │ Preproc={preproc_str}"
        )
        if phase:
            row1 += f" │ Phase={phase}"

        # ── Counters row ────────────────────────────────
        parts: list[str] = []
        if severe_severity > 0:
            parts.append(
                "⏳  no_improve="
                f"{epochs_without_improvement}/{patience_severe}"
            )
        parts.append(f"🏆  best={best_word_acc:.1%}")

        if lr_warmup_epoch is not None and lr_warmup_total is not None:
            parts.append(f"📈  LR↑  {lr_warmup_epoch}/{lr_warmup_total}")

        row2 = "     " + " │ ".join(parts)

        self._warmup_text.plain = f"{row1}\n{row2}"

    def hide_warmup_status(self) -> None:
        """Remove warmup status lines."""
        self._warmup_text.plain = ""

    # ── Validation marker ────────────────────────────────

    def show_validation(self, text: str = "⏳  Validating...") -> None:
        """Show validation marker without changing epoch summary."""
        self._validation_text.plain = text

    def hide_validation(self) -> None:
        """Remove validation marker."""
        self._validation_text.plain = ""

    # ── Console access ──────────────────────────────────

    @property
    def console(self) -> Console:
        """Access Rich console for printing panels etc."""
        return self._batch_progress.console
