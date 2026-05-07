"""Process single training epoch logic."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch.utils.data import DataLoader

from redstar_plate_ocr.pipeline.progress_display import ProgressDisplay
from redstar_plate_ocr.pipeline.training_config import TrainingConfig
from redstar_plate_ocr.pipeline.utils import (
    format_epoch_stats,
    format_train_epoch_stats,
    log_epoch_summary,
)

if TYPE_CHECKING:
    from redstar_plate_ocr.pipeline.trainer import Trainer

_ValidationResult = tuple[
    dict[str, float],
    dict[str, float],
    float,
    dict[str, float],
    int,
    bool,
]

logger = logging.getLogger(__name__)


def compute_sampling_prob(
    epoch: int,
    max_prob: float,
    ramp_epochs: int,
) -> float:
    """Compute scheduled sampling probability with linear ramp.

    prob = 0.0 at epoch 0, linearly increases to max_prob
    over ramp_epochs, then stays at max_prob.
    """
    if ramp_epochs <= 0:
        return max_prob
    return min(max_prob, max_prob * epoch / ramp_epochs)


class _EpochResult:
    """Result of processing one training epoch."""

    __slots__ = (
        "best_metric",
        "best_metrics",
        "last_metrics",
        "patience_counter",
        "should_stop",
        "was_interrupted",
        "epoch_stats",
    )

    def __init__(
        self,
        best_metric: float,
        best_metrics: dict[str, float],
        last_metrics: dict[str, float],
        patience_counter: int,
        should_stop: bool,
        was_interrupted: bool,
        epoch_stats: str = "",
    ) -> None:
        self.best_metric = best_metric
        self.best_metrics = best_metrics
        self.last_metrics = last_metrics
        self.patience_counter = patience_counter
        self.should_stop = should_stop
        self.was_interrupted = was_interrupted
        self.epoch_stats = epoch_stats


def _inject_train_acc(
    val_metrics: dict[str, float],
    train_result: dict[str, float],
) -> None:
    """Inject train accuracy keys into val_metrics."""
    for key in ("fmt_acc", "ctry_acc", "plate_acc", "char_acc"):
        if key in train_result:
            val_metrics[f"train_{key}"] = train_result[key]


def _run_e2e_eval(
    trainer: Trainer,
    val_loader: DataLoader,
    val_metrics: dict[str, float],
) -> None:
    """Run E2E evaluation and prefix keys."""
    e2e_metrics = trainer.evaluator.evaluate(
        trainer.model,
        val_loader,
        interrupt_check=lambda: trainer._interrupt_requested,
        e2e=True,
    )
    for k, v in e2e_metrics.items():
        val_metrics[f"val_e2e_{k.removeprefix('val_')}"] = v


def _check_improvement(
    sched_val: float,
    best_metric: float,
    es_mode: str,
) -> bool:
    """Check if validation metric improved."""
    if es_mode == "max":
        return sched_val > best_metric
    return sched_val < best_metric


def _make_interrupt_result(
    best_metric: float,
    best_metrics: dict[str, float],
    patience_counter: int,
    epoch: int,
) -> _EpochResult:
    """Create result for interrupted epoch."""
    logger.info("Training interrupted at epoch %d", epoch + 1)
    return _EpochResult(
        best_metric=best_metric,
        best_metrics=best_metrics,
        last_metrics={},
        patience_counter=patience_counter,
        should_stop=False,
        was_interrupted=True,
    )


def _make_stop_result(
    best_metric: float,
    best_metrics: dict[str, float],
    last_metrics: dict[str, float],
    patience_counter: int,
    epoch: int,
) -> _EpochResult:
    """Create result for early-stopped epoch."""
    logger.warning("Early stopping at epoch %d", epoch + 1)
    return _EpochResult(
        best_metric=best_metric,
        best_metrics=best_metrics,
        last_metrics=last_metrics,
        patience_counter=patience_counter,
        should_stop=True,
        was_interrupted=False,
    )


@dataclass(frozen=True, slots=True)
class _ValidationOutcome:
    """Outcome of a validation run."""

    val_metrics: dict[str, float]
    last_metrics: dict[str, float]
    best_metric: float
    best_metrics: dict[str, float]
    patience_counter: int
    should_stop: bool

    def __iter__(self):
        """Unpack like a tuple."""
        yield from (
            self.val_metrics, self.last_metrics, self.best_metric,
            self.best_metrics, self.patience_counter, self.should_stop,
        )


def _run_validation(
    trainer: Trainer,
    val_loader: DataLoader,
    train_result: dict[str, float],
    config: TrainingConfig,
    epoch: int,
    best_metric: float,
    best_metrics: dict[str, float],
    patience_counter: int,
    progress_display: ProgressDisplay,
) -> _ValidationOutcome:
    """Run validation, update best metrics, and check early stopping."""
    progress_display.show_validation("⏳ Validating...")
    val_metrics = trainer.evaluator.evaluate(
        trainer.model,
        val_loader,
        interrupt_check=lambda: trainer._interrupt_requested,
    )
    _inject_train_acc(val_metrics, train_result)
    last_metrics = dict(val_metrics)

    sched_val = val_metrics.get(config.es_metric, 0.0)
    trainer.scheduler.step(sched_val)

    if config.e2e_eval:
        _run_e2e_eval(trainer, val_loader, val_metrics)
        last_metrics = dict(val_metrics)

    progress_display.show_validation(
        "⏳ Validating... 💾 Saving...",
    )
    trainer._save_checkpoint("last.pt", epoch, sched_val)

    improved = _check_improvement(
        sched_val, best_metric, config.es_mode
    )
    if improved:
        best_metric = sched_val
        best_metrics = dict(val_metrics)
        patience_counter = 0
        trainer._save_checkpoint(
            "best.pt",
            epoch,
            best_metric,
        )
    else:
        patience_counter += 1
        if patience_counter >= config.es_patience:
            return _ValidationOutcome(
                val_metrics, last_metrics, best_metric,
                best_metrics, patience_counter, True,
            )

    return _ValidationOutcome(
        val_metrics, last_metrics, best_metric,
        best_metrics, patience_counter, False,
    )


def process_epoch(
    trainer: Trainer,
    epoch: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    progress_display: ProgressDisplay,
    best_metric: float,
    best_metrics: dict[str, float],
    patience_counter: int,
    epoch_start: float = 0.0,
) -> _EpochResult:
    """Process one training epoch."""
    config: TrainingConfig = trainer.config
    phase = _get_phase(config, epoch)
    cur_lr = trainer.optimizer.param_groups[0]["lr"]
    desc = f"Epoch {epoch + 1}/{config.epochs} [{phase}] LR={cur_lr:.4f}"

    sampling_prob = compute_sampling_prob(
        epoch,
        config.scheduled_sampling_max_prob,
        config.scheduled_sampling_ramp_epochs,
    )

    batch_task = progress_display.add_batch_task(
        desc,
        total=len(train_loader),
        stats="",
    )
    progress_display.update_epoch_summary(
        format_train_epoch_stats(0.0, best_metrics)
    )
    train_result = trainer._train_epoch(
        train_loader,
        sampling_prob,
        progress_display,
        batch_task,
        epoch_start=epoch_start,
        current_epoch=epoch,
    )

    train_loss = train_result.get("loss", 0.0)
    progress_display.update_epoch_summary(
        format_train_epoch_stats(train_loss, best_metrics)
    )
    progress_display.remove_batch_task(batch_task)

    if trainer._interrupt_requested:
        return _make_interrupt_result(
            best_metric, best_metrics, patience_counter, epoch
        )

    should_validate = (epoch + 1) % config.val_every_n_epochs == 0

    val_metrics: dict[str, float] = dict(best_metrics)
    is_cached = True
    last_metrics: dict[str, float] = dict(best_metrics)

    if should_validate:
        result = _run_validation(
            trainer, val_loader, train_result, config,
            epoch, best_metric, best_metrics, patience_counter,
            progress_display,
        )
        if result.should_stop:
            return _make_stop_result(
                best_metric, best_metrics,
                result.last_metrics, result.patience_counter, epoch,
            )
        val_metrics = result.val_metrics
        last_metrics = result.last_metrics
        best_metric = result.best_metric
        best_metrics = result.best_metrics
        patience_counter = result.patience_counter
        is_cached = False
    else:
        # Skip validation — keep last known metrics
        trainer._save_checkpoint("last.pt", epoch, best_metric)

    epoch_duration = time.monotonic() - epoch_start if epoch_start else 0.0

    log_epoch_summary(
        epoch,
        config.epochs,
        phase,
        cur_lr,
        train_loss,
        val_metrics,
        best_metrics,
        logger,
        epoch_duration=epoch_duration,
    )
    epoch_stats = format_epoch_stats(
        val_metrics,
        best_metrics,
        train_loss,
        is_cached=is_cached,
        epoch_duration=epoch_duration,
    )
    progress_display.hide_validation()
    progress_display.update_epoch_summary(epoch_stats)

    return _EpochResult(
        best_metric=best_metric,
        best_metrics=best_metrics,
        last_metrics=last_metrics,
        patience_counter=patience_counter,
        should_stop=False,
        was_interrupted=False,
        epoch_stats=epoch_stats,
    )


def _get_phase(
    config: TrainingConfig,
    epoch: int,
) -> str:
    """Get training phase name for given epoch."""
    if epoch < config.warmup_epochs:
        return "Warmup"
    single_aug_end = config.warmup_epochs + config.single_aug_epochs
    if epoch < single_aug_end:
        return "SingleAug"
    if epoch >= config.epochs - config.no_aug_epochs:
        return "NoAug"
    return "Main"
