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
    from redstar_plate_ocr.data.severe_aug import SevereAugScheduler
    from redstar_plate_ocr.pipeline.tracking import MetricsTracker
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
        interrupt_check=lambda: (
            trainer._interrupt_requested or trainer._force_stop
        ),
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
            self.val_metrics,
            self.last_metrics,
            self.best_metric,
            self.best_metrics,
            self.patience_counter,
            self.should_stop,
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
    severe_scheduler: SevereAugScheduler | None = None,
) -> _ValidationOutcome:
    """Run validation, update best metrics, and check early stopping."""
    progress_display.show_validation("⏳ Validating...")
    val_metrics = trainer.evaluator.evaluate(
        trainer.model,
        val_loader,
        interrupt_check=lambda: (
            trainer._interrupt_requested or trainer._force_stop
        ),
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

    improved = _check_improvement(sched_val, best_metric, config.es_mode)
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
        # Early stopping with warmup condition:
        # disabled while severe augmentation is active.
        # After severe is off, use scheduler's own patience
        # to avoid shared-state conflict with update_schedule.
        if severe_scheduler is not None:
            if severe_scheduler.severe_severity > 0:
                # Severe augmentation still active — skip early stopping
                # to avoid premature termination while model adapts.
                pass
            elif patience_counter >= severe_scheduler.early_stop_patience:
                return _ValidationOutcome(
                    val_metrics,
                    last_metrics,
                    best_metric,
                    best_metrics,
                    patience_counter,
                    True,
                )
        elif patience_counter >= config.es_patience:
            return _ValidationOutcome(
                val_metrics,
                last_metrics,
                best_metric,
                best_metrics,
                patience_counter,
                True,
            )

    # Update severe scheduler after validation
    if severe_scheduler is not None:
        word_acc = val_metrics.get("val_plate_accuracy", 0.0)
        severe_scheduler.update_schedule(word_acc, epoch)
        logger.info(
            "Warmup schedule: severe=%.3f, "
            "std=%.3f, "
            "preprocessing=%s, "
            "best_acc=%.4f",
            severe_scheduler.severe_severity,
            severe_scheduler.std_severity,
            "on" if severe_scheduler.preprocessing_enabled else "off",
            severe_scheduler.best_word_acc,
        )
        _update_warmup_display(
            progress_display,
            severe_scheduler,
            phase=_get_phase(config, epoch, severe_scheduler),
            lr_warmup_epoch=min(epoch + 1, config.warmup_epochs)
            if epoch < config.warmup_epochs
            else None,
            lr_warmup_total=config.warmup_epochs
            if epoch < config.warmup_epochs
            else None,
        )

    return _ValidationOutcome(
        val_metrics,
        last_metrics,
        best_metric,
        best_metrics,
        patience_counter,
        False,
    )


def _update_warmup_display(
    progress_display: ProgressDisplay,
    severe_scheduler: SevereAugScheduler,
    *,
    phase: str = "",
    lr_warmup_epoch: int | None = None,
    lr_warmup_total: int | None = None,
) -> None:
    """Update warmup status line with visual transition bars."""
    progress_display.update_warmup_detail(
        severe_severity=severe_scheduler.severe_severity,
        std_severity=severe_scheduler.std_severity,
        preprocessing_enabled=severe_scheduler.preprocessing_enabled,
        best_word_acc=severe_scheduler.best_word_acc,
        epochs_without_improvement=severe_scheduler.epochs_without_improvement,
        patience_severe=severe_scheduler.patience_severe,
        phase=phase,
        lr_warmup_epoch=lr_warmup_epoch,
        lr_warmup_total=lr_warmup_total,
    )


def _build_train_metrics(
    train_result: dict[str, float],
) -> dict[str, float]:
    """Build train-related metrics for logging."""
    cur_lr = train_result.get("lr", 0.0)
    return {
        "train/loss": train_result.get("loss", 0.0),
        "train/ctc_loss": train_result.get("ctc", 0.0),
        "train/format_loss": train_result.get("format", 0.0),
        "train/country_loss": train_result.get("country", 0.0),
        "train/order_loss": train_result.get("order", 0.0),
        "train/char_aux_loss": train_result.get("char_aux", 0.0),
        "train/synergy_loss": train_result.get("synergy", 0.0),
        "train/length_loss": train_result.get("length", 0.0),
        "train/format_acc": train_result.get("fmt_acc", 0.0),
        "train/country_acc": train_result.get("ctry_acc", 0.0),
        "train/plate_acc": train_result.get("plate_acc", 0.0),
        "train/char_acc": train_result.get("char_acc", 0.0),
        "train/lr": cur_lr,
    }


def _build_val_core_metrics(
    val_metrics: dict[str, float],
) -> dict[str, float]:
    """Build core validation metrics for logging."""
    return {
        "val/cer": val_metrics.get("val_cer", 0.0),
        "val/plate_accuracy": val_metrics.get(
            "val_plate_accuracy",
            0.0,
        ),
        "val/char_accuracy": val_metrics.get(
            "val_char_accuracy",
            0.0,
        ),
        "val/ned": val_metrics.get("val_ned", 0.0),
        "val/format_acc": val_metrics.get(
            "val_format_accuracy",
            0.0,
        ),
        "val/country_acc": val_metrics.get(
            "val_country_accuracy",
            0.0,
        ),
        "val/square_accuracy": val_metrics.get(
            "val_square_accuracy",
            0.0,
        ),
        "val/standard_accuracy": val_metrics.get(
            "val_standard_accuracy",
            0.0,
        ),
    }


def _collect_prefixed_metrics(
    val_metrics: dict[str, float],
    src_prefix: str,
    exclude_prefix: str | None,
    dst_prefix: str,
) -> dict[str, float]:
    """Collect metrics whose keys start with *src_prefix*.

    If *exclude_prefix* is given, keys starting with it are skipped.
    The collected keys have *src_prefix* replaced by *dst_prefix*.
    """
    result: dict[str, float] = {}
    for key, value in val_metrics.items():
        if not key.startswith(src_prefix):
            continue
        if exclude_prefix and key.startswith(exclude_prefix):
            continue
        suffix = key.removeprefix(src_prefix)
        result[f"{dst_prefix}{suffix}"] = value
    return result


def _log_metrics(
    tracker: MetricsTracker,
    epoch: int,
    train_result: dict[str, float],
    val_metrics: dict[str, float],
    severe_scheduler: SevereAugScheduler | None = None,
) -> None:
    """Log all metrics via TrackIO / console."""
    metrics_to_log: dict[str, float] = {}
    metrics_to_log.update(_build_train_metrics(train_result))
    metrics_to_log.update(_build_val_core_metrics(val_metrics))

    # Warmup status
    if severe_scheduler is not None:
        metrics_to_log["warmup/severe_severity"] = (
            severe_scheduler.severe_severity
        )
        metrics_to_log["warmup/std_severity"] = severe_scheduler.std_severity

    # Per-country metrics (CER + plate accuracy)
    metrics_to_log.update(
        _collect_prefixed_metrics(
            val_metrics,
            "val_cer_",
            "val_cer_fmt_",
            "val/cer_",
        )
    )
    metrics_to_log.update(
        _collect_prefixed_metrics(
            val_metrics,
            "val_plateacc_",
            "val_plateacc_fmt_",
            "val/plateacc_",
        )
    )

    # Per-format metrics (CER + plate accuracy)
    metrics_to_log.update(
        _collect_prefixed_metrics(
            val_metrics,
            "val_cer_fmt_",
            None,
            "val/cer_fmt_",
        )
    )
    metrics_to_log.update(
        _collect_prefixed_metrics(
            val_metrics,
            "val_plateacc_fmt_",
            None,
            "val/plateacc_fmt_",
        )
    )

    # Per-region plate accuracy
    metrics_to_log.update(
        _collect_prefixed_metrics(
            val_metrics,
            "val_region_",
            None,
            "val/region_",
        )
    )

    # BSR / ATR — always log when available
    bsr = val_metrics.get("val_bigram_swap_rate")
    atr = val_metrics.get("val_adjacent_transposition_rate")
    if bsr is not None:
        metrics_to_log["val/bigram_swap_rate"] = bsr
    if atr is not None:
        metrics_to_log["val/adjacent_transposition_rate"] = atr

    tracker.log(metrics_to_log, step=epoch)


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
    severe_scheduler: SevereAugScheduler | None = None,
    tracker: MetricsTracker | None = None,
) -> _EpochResult:
    """Process one training epoch."""
    config: TrainingConfig = trainer.config
    phase = _get_phase(config, epoch, severe_scheduler)
    cur_lr = trainer.optimizer.param_groups[0]["lr"]

    # Phase icon for batch desc
    _PHASE_ICONS = {
        "Warmup": "🔥  ",
        "SevereWarmup": "🔥  ",
        "SingleAug": "⚡  ",
        "Main": "🏁  ",
        "NoAug": "🧹  ",
    }
    phase_icon = _PHASE_ICONS.get(phase, "")

    # Build header with extra context
    header_parts = [f"Ep{epoch + 1}/{config.epochs}"]
    if phase_icon:
        header_parts.append(phase_icon)
    if epoch < config.warmup_epochs:
        header_parts.append(f"LR↑  {epoch + 1}/{config.warmup_epochs}")
    if config.gradient_accumulation_steps > 1:
        header_parts.append(f"×{config.gradient_accumulation_steps}")
    desc = " ".join(header_parts) + f" lr={cur_lr:.6g}"

    sampling_prob = compute_sampling_prob(
        epoch,
        config.scheduled_sampling_max_prob,
        config.scheduled_sampling_ramp_epochs,
    )

    # Append early-stopping countdown to desc
    if config.es_patience > 0:
        desc += f" 🛑  {patience_counter}/{config.es_patience}"

    batch_task = progress_display.add_batch_task(
        desc,
        total=len(train_loader),
        stats="",
    )
    # Warmup status line
    if severe_scheduler is not None:
        _update_warmup_display(
            progress_display,
            severe_scheduler,
            phase=phase,
            lr_warmup_epoch=min(epoch + 1, config.warmup_epochs)
            if epoch < config.warmup_epochs
            else None,
            lr_warmup_total=config.warmup_epochs
            if epoch < config.warmup_epochs
            else None,
        )
    elif epoch < config.warmup_epochs and config.warmup_epochs > 0:
        # Plain LR warmup (no adaptive aug)
        progress_display.update_warmup_detail(
            phase=phase,
            lr_warmup_epoch=epoch + 1,
            lr_warmup_total=config.warmup_epochs,
        )
    else:
        progress_display.hide_warmup_status()
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
        tracker=tracker,
        log_interval=config.log_interval,
        log_grad_interval=config.log_grad_interval,
    )

    train_loss = train_result.get("loss", 0.0)
    progress_display.update_epoch_summary(
        format_train_epoch_stats(train_loss, best_metrics)
    )
    progress_display.remove_batch_task(batch_task)

    if trainer._interrupt_requested or trainer._force_stop:
        return _make_interrupt_result(
            best_metric, best_metrics, patience_counter, epoch
        )

    should_validate = (epoch + 1) % config.val_every_n_epochs == 0

    val_metrics: dict[str, float] = dict(best_metrics)
    last_metrics: dict[str, float] = dict(best_metrics)

    if should_validate:
        result = _run_validation(
            trainer,
            val_loader,
            train_result,
            config,
            epoch,
            best_metric,
            best_metrics,
            patience_counter,
            progress_display,
            severe_scheduler=severe_scheduler,
        )
        if result.should_stop:
            return _make_stop_result(
                best_metric,
                best_metrics,
                result.last_metrics,
                result.patience_counter,
                epoch,
            )
        val_metrics = result.val_metrics
        last_metrics = result.last_metrics
        best_metric = result.best_metric
        best_metrics = result.best_metrics
        patience_counter = result.patience_counter

        # Log metrics to TrackIO / console after validation
        if tracker is not None:
            train_result["lr"] = cur_lr
            _log_metrics(
                tracker,
                epoch,
                train_result,
                val_metrics,
                severe_scheduler=severe_scheduler,
            )
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
    # Log warmup scheduler state after every epoch
    if severe_scheduler is not None:
        logger.info(
            "Warmup: phase=%s severe=%.3f std=%.3f preproc=%s best_acc=%.4f",
            phase,
            severe_scheduler.severe_severity,
            severe_scheduler.std_severity,
            "on" if severe_scheduler.preprocessing_enabled else "off",
            severe_scheduler.best_word_acc,
        )
    epoch_stats = format_epoch_stats(
        val_metrics,
        best_metrics,
        train_loss,
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
    severe_scheduler: SevereAugScheduler | None = None,
) -> str:
    """Get training phase name for given epoch."""
    if severe_scheduler is not None and severe_scheduler.severe_severity > 0:
        return "SevereWarmup"
    if epoch < config.warmup_epochs:
        return "Warmup"
    single_aug_end = config.warmup_epochs + config.single_aug_epochs
    if epoch < single_aug_end:
        return "SingleAug"
    if epoch >= config.epochs - config.no_aug_epochs:
        return "NoAug"
    return "Main"
