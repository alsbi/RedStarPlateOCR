"""Single training epoch logic."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import torch
from rich.progress import TaskID

from redstar_plate_ocr.nn.metrics import levenshtein_distance
from redstar_plate_ocr.pipeline.progress_display import ProgressDisplay
from redstar_plate_ocr.pipeline.utils import (
    format_train_epoch_stats,
    greedy_decode,
    to_long_tensor,
)
from redstar_plate_ocr.plate.config import PlateConfig

if TYPE_CHECKING:
    from redstar_plate_ocr.pipeline.tracking import MetricsTracker
    from redstar_plate_ocr.pipeline.trainer import Trainer

logger = logging.getLogger(__name__)

# Loss component keys tracked from CombinedLoss.forward()
_LOSS_COMPONENTS = (
    "ctc",
    "country",
    "format",
    "order",
    "char_aux",
    "synergy",
    "length",
)


def _compute_char_aux_ramp_weight(
    base_weight: float,
    peak_weight: float | None,
    ramp_epochs: int,
    current_epoch: int,
    total_epochs: int,
) -> float:
    """Compute char_aux_weight with optional ramp schedule.

    If *peak_weight* is None or *ramp_epochs* is 0, returns
    *base_weight* unchanged (no ramp).

    Ramp schedule:
        epoch 0..ramp_epochs-1   → linear rise base→peak
        epoch ramp_epochs..end    → linear decay peak→base

    This gives the auxiliary head a strong gradient signal early
    in training (when backbone features are still random) and then
    tapers off so the main CTC path dominates.
    """
    if peak_weight is None or ramp_epochs <= 0:
        return base_weight
    if current_epoch < ramp_epochs:
        factor = current_epoch / ramp_epochs
        return base_weight + (peak_weight - base_weight) * factor
    decay_epochs = total_epochs - ramp_epochs
    if decay_epochs > 0:
        factor = min(1.0, (current_epoch - ramp_epochs) / decay_epochs)
        return peak_weight - (peak_weight - base_weight) * factor
    return peak_weight


_BatchTensors = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
    list[str],
    list[str],
    torch.Tensor,
    torch.Tensor,
]


def _setup_batch(
    trainer: Trainer,
    batch: dict,
) -> _BatchTensors:
    """Prepare batch tensors and labels."""
    images = batch["image"].to(trainer.device, non_blocking=True)
    orig_h = to_long_tensor(batch["orig_h"], trainer.device)
    orig_w = to_long_tensor(batch["orig_w"], trainer.device)
    gt_regions = batch["region"]
    gt_plate_types = batch["plate_type"]
    gt_texts = batch["plate_text"]
    format_labels = [1 if pt == "square" else 0 for pt in gt_plate_types]
    gt_format = torch.tensor(
        format_labels, dtype=torch.long, device=trainer.device
    )
    gt_country = trainer.model.encode_countries(gt_regions).to(trainer.device)
    return (
        images,
        orig_h,
        orig_w,
        gt_regions,
        gt_plate_types,
        gt_texts,
        gt_format,
        gt_country,
    )


def _compute_loss_with_lengths(
    trainer: Trainer,
    output,
    gt_format: torch.Tensor,
    gt_country: torch.Tensor,
    gt_texts: list[str],
    input_lengths: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Forward pass and loss computation (input_lengths pre-computed)."""
    loss_dict = trainer.combined_loss(
        output,
        gt_format,
        gt_country,
        gt_texts,
        input_lengths,
    )
    return loss_dict, loss_dict["total"]


def _backward_step(
    trainer: Trainer,
    loss: torch.Tensor,
) -> None:
    """Backward pass with optional AMP scaling and NaN guard."""
    if not torch.isfinite(loss):
        logger.warning(
            "Non-finite loss %.4f, skipping backward",
            loss.item(),
        )
        trainer.optimizer.zero_grad()
        trainer.country_optimizer.zero_grad()
        return
    if trainer.use_amp:
        trainer.scaler.scale(loss).backward()
    else:
        loss.backward()


def _optimizer_step(
    trainer: Trainer,
    step: int,
    force: bool = False,
) -> tuple[int, bool, float]:
    """Gradient accumulation and optimizer step.

    Returns updated step counter, whether step was performed,
    and combined gradient norm (0.0 if no step).
    """
    config = trainer.config
    accum_steps = config.gradient_accumulation_steps
    # When forced: only flush if there are un-stepped accumulated gradients.
    # If the last natural step already consumed all grads (step % accum == 0),
    # a phantom step on zero grads would corrupt Adam momentum and inflate
    # the global_step / LR schedule.
    if force and step % accum_steps == 0:
        return step, False, 0.0
    if not force and (step + 1) % accum_steps != 0:
        return step + 1, False, 0.0

    grad_clip = config.gradient_clip
    if trainer.use_amp:
        trainer.scaler.unscale_(trainer.optimizer)
        trainer.scaler.unscale_(trainer.country_optimizer)

    main_gn = torch.nn.utils.clip_grad_norm_(
        trainer.optimizer.param_groups[0]["params"],
        max_norm=grad_clip,
    )
    ctry_gn = torch.nn.utils.clip_grad_norm_(
        trainer.country_optimizer.param_groups[0]["params"],
        max_norm=grad_clip,
    )
    # Combined norm across both param groups
    total_gn = (main_gn.item() ** 2 + ctry_gn.item() ** 2) ** 0.5

    if trainer.use_amp:
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.step(trainer.country_optimizer)
        trainer.scaler.update()
    else:
        trainer.optimizer.step()
        trainer.country_optimizer.step()

    trainer.optimizer.zero_grad()
    trainer.country_optimizer.zero_grad()
    return step + 1, True, total_gn


def _compute_batch_accuracy(
    output,
    gt_format: torch.Tensor,
    gt_country: torch.Tensor,
    gt_texts: list[str],
    plate_config: PlateConfig,
    input_lengths: torch.Tensor | None = None,
) -> tuple[float, float, float, float]:
    """Compute per-batch format, country, plate and char accuracy.

    Single-pass greedy decode: computes both plate accuracy
    (exact match) and character accuracy (levenshtein) together.
    """
    pred_fmt = output.format_logits.argmax(dim=1)
    fmt_acc = (pred_fmt == gt_format).float().mean().item()
    pred_ctry = output.country_logits.argmax(dim=1)
    ctry_acc = (pred_ctry == gt_country).float().mean().item()
    # Single-pass: plate_acc + char_acc together
    plate_correct = 0
    char_correct = 0
    char_total = 0
    bsz = output.ctc_output.shape[0]
    union_alphabet = plate_config.union_alphabet
    for i in range(bsz):
        inp_len = int(input_lengths[i]) if input_lengths is not None else None
        pred = greedy_decode(
            output.ctc_output[i],
            union_alphabet,
            input_length=inp_len,
        )
        tgt = gt_texts[i]
        if pred == tgt:
            plate_correct += 1
        dist = levenshtein_distance(pred, tgt)
        char_correct += max(len(tgt) - dist, 0)
        char_total += max(len(tgt), 1)
    plate_acc = plate_correct / bsz if bsz > 0 else 0.0
    char_acc = char_correct / char_total if char_total else 0.0
    return fmt_acc, ctry_acc, plate_acc, char_acc


def _format_batch_stats(
    running: dict[str, float],
    fmt_acc: float,
    ctry_acc: float,
    plate_acc: float,
    char_acc: float,
    avg_batch_ms: float = 0.0,
    grad_norm: float = 0.0,
    accum_step: int = 0,
    accum_total: int = 1,
) -> str:
    """Format progress-bar stats string."""
    left = (
        f"loss={running['loss']:.4f} "
        f"plate={plate_acc:.3%} "
        f"char={char_acc:.3%} "
        f"region={ctry_acc:.3%} "
        f"fmt={fmt_acc:.3%}"
    )
    # Right section: grad norm + accum + timing
    right_parts: list[str] = []
    if grad_norm > 0.0:
        right_parts.append(f"📏  {grad_norm:.2f}")
    if accum_total > 1:
        right_parts.append(f"📦  {accum_step}/{accum_total}")
    right_parts.append(f"⏱  {avg_batch_ms:.0f}ms")
    right = " ".join(right_parts)
    return f"{left} │ {right}"


def _should_update_progress(
    batch_idx: int,
    loader_len: int,
    batches_since_update: int,
    update_every: int,
) -> bool:
    """Check if progress bar should be updated."""
    if batches_since_update >= update_every:
        return True
    return (batch_idx + 1) == loader_len


def _update_progress(
    trainer: Trainer,
    output,
    gt_format: torch.Tensor,
    gt_country: torch.Tensor,
    gt_texts: list[str],
    running: dict[str, float],
    avg_batch_ms: float,
    progress_display: ProgressDisplay,
    task_id: TaskID,
    batches_since_update: int,
    running_fmt_acc: list[float],
    running_ctry_acc: list[float],
    running_plate_acc: list[float],
    running_char_acc: list[float],
    input_lengths: torch.Tensor | None = None,
    grad_norm: float = 0.0,
    accum_step: int = 0,
    accum_total: int = 1,
) -> None:
    """Compute batch accuracy and update progress display."""
    with torch.no_grad():
        result = _compute_batch_accuracy(
            output,
            gt_format,
            gt_country,
            gt_texts,
            trainer.model.plate_config,
            input_lengths=input_lengths,
        )
        fmt_acc, ctry_acc, plate_acc, char_acc = result
    running_fmt_acc.append(fmt_acc)
    running_ctry_acc.append(ctry_acc)
    running_plate_acc.append(plate_acc)
    running_char_acc.append(char_acc)
    stats = _format_batch_stats(
        running,
        fmt_acc,
        ctry_acc,
        plate_acc,
        char_acc,
        avg_batch_ms=avg_batch_ms,
        grad_norm=grad_norm,
        accum_step=accum_step,
        accum_total=accum_total,
    )
    best = getattr(trainer, "_best_metrics", None)
    epoch_summary = format_train_epoch_stats(
        running["loss"],
        best if isinstance(best, dict) else {},
    )
    progress_display.update_epoch_summary(epoch_summary)
    progress_display.update_batch(
        task_id,
        advance=batches_since_update,
        stats=stats,
    )


def _update_running_loss(
    running: dict[str, float],
    loss_dict: dict[str, torch.Tensor],
) -> None:
    """Update running loss dictionary from loss dict.

    Stores latest value for display and accumulates sums
    for epoch-level averaging via ``_compute_final_loss_avgs``.
    """
    running["_loss_count"] = running.get("_loss_count", 0) + 1
    # Latest total loss for progress bar display
    loss_val = loss_dict["total"].item()
    running["loss"] = loss_val
    running["_sum_loss"] = running.get("_sum_loss", 0.0) + loss_val
    # Per-component: latest value + running sum for averaging
    for key in _LOSS_COMPONENTS:
        if key in loss_dict:
            val = loss_dict[key].item()
            running[key] = val
            running[f"_sum_{key}"] = running.get(f"_sum_{key}", 0.0) + val


def _compute_final_loss_avgs(running: dict[str, float]) -> None:
    """Replace per-component losses with epoch averages."""
    count = running.get("_loss_count", 0)
    if count <= 0:
        return
    for key in _LOSS_COMPONENTS:
        sum_key = f"_sum_{key}"
        if sum_key in running:
            running[key] = running[sum_key] / count
    # Also average the total loss
    if "_sum_loss" in running:
        running["loss"] = running["_sum_loss"] / count
    # Cleanup internal keys
    for k in list(running):
        if k.startswith("_sum_") or k == "_loss_count":
            del running[k]


def _compute_final_accuracies(
    running: dict[str, float],
    running_fmt_acc: list[float],
    running_ctry_acc: list[float],
    running_plate_acc: list[float],
    running_char_acc: list[float],
) -> None:
    """Compute and store average accuracies in running dict."""
    if not running_fmt_acc:
        return
    n = len(running_fmt_acc)
    running["fmt_acc"] = sum(running_fmt_acc) / n
    running["ctry_acc"] = sum(running_ctry_acc) / n
    running["plate_acc"] = sum(running_plate_acc) / n
    running["char_acc"] = sum(running_char_acc) / n


def _build_step_metrics(
    running: dict[str, float],
    cur_lr: float,
) -> dict[str, float]:
    """Build step-level metrics dict for tracker logging."""
    metrics: dict[str, float] = {
        "train/step_loss": running["loss"],
        "train/step_lr": cur_lr,
    }
    for key in _LOSS_COMPONENTS:
        if key in running:
            metrics[f"train/step_{key}_loss"] = running[key]
    return metrics


def _log_tracker_step(
    tracker: MetricsTracker,
    global_step: int,
    log_interval: int,
    log_grad_interval: int,
    running: dict[str, float],
    cur_lr: float,
    grad_norm: float,
) -> None:
    """Log step-level metrics and grad norm to tracker."""
    if global_step % log_interval == 0:
        step_metrics = _build_step_metrics(running, cur_lr)
        tracker.log(step_metrics, step=global_step)
    if (
        log_grad_interval > 0
        and global_step % log_grad_interval == 0
        and grad_norm > 0.0
    ):
        tracker.log(
            {"train/grad_norm": grad_norm},
            step=global_step,
        )


def _compute_avg_batch_ms(
    epoch_start: float,
    batch_idx: int,
) -> float:
    """Compute average batch time in ms; 0.0 if not started."""
    if epoch_start <= 0.0:
        return 0.0
    elapsed = (time.monotonic() - epoch_start) * 1000
    return elapsed / (batch_idx + 1)


def _forward_and_backward(
    trainer: Trainer,
    images: torch.Tensor,
    orig_h: torch.Tensor,
    orig_w: torch.Tensor,
    gt_regions: list[str],
    gt_plate_types: list[str],
    gt_texts: list[str],
    gt_format: torch.Tensor,
    gt_country: torch.Tensor,
    sampling_prob: float,
    current_epoch: int,
) -> tuple[object, dict[str, torch.Tensor], torch.Tensor]:
    """Forward + loss + backward; return output, loss_dict, input_lengths."""
    with torch.amp.autocast(trainer.device.type, enabled=trainer.use_amp):
        output = trainer.model(
            images,
            orig_h,
            orig_w,
            gt_countries=gt_regions,
            gt_plate_types=gt_plate_types,
            scheduled_sampling_prob=sampling_prob,
            epoch=current_epoch,
        )

        seq_len = output.ctc_output.shape[1]
        per_sample_types = list(gt_plate_types)
        input_lengths = trainer.model.compression.compute_input_lengths(
            output.content_mask, per_sample_types
        ).to(trainer.device)
        input_lengths = input_lengths.clamp(min=2, max=seq_len)

        loss_dict, total_loss = _compute_loss_with_lengths(
            trainer,
            output,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )
        loss = total_loss / trainer.config.gradient_accumulation_steps
        _backward_step(trainer, loss)

    return output, loss_dict, input_lengths


def run_train_epoch(
    trainer: Trainer,
    loader,
    sampling_prob: float,
    progress_display: ProgressDisplay,
    task_id: TaskID,
    epoch_start: float = 0.0,
    current_epoch: int = 0,
    tracker: MetricsTracker | None = None,
    log_interval: int = 20,
    log_grad_interval: int = 100,
) -> dict[str, float]:
    """Run one training epoch with Rich progress bar."""
    trainer.model.train()
    trainer.optimizer.zero_grad()
    trainer.country_optimizer.zero_grad()
    step = 0
    global_step = 0
    running: dict[str, float] = {}
    batches_since_update = 0
    running_fmt_acc: list[float] = []
    running_ctry_acc: list[float] = []
    running_plate_acc: list[float] = []
    running_char_acc: list[float] = []
    avg_batch_ms: float = 0.0
    last_grad_norm: float = 0.0
    accum_steps = trainer.config.gradient_accumulation_steps

    # Apply char_aux ramp schedule if configured
    cfg = trainer.config
    ramp_weight = _compute_char_aux_ramp_weight(
        base_weight=cfg.char_aux_weight,
        peak_weight=cfg.char_aux_peak_weight,
        ramp_epochs=cfg.char_aux_ramp_epochs,
        current_epoch=current_epoch,
        total_epochs=cfg.epochs,
    )
    trainer.combined_loss.char_aux_weight = ramp_weight

    for batch_idx, batch in enumerate(loader):
        (
            images,
            orig_h,
            orig_w,
            gt_regions,
            gt_plate_types,
            gt_texts,
            gt_format,
            gt_country,
        ) = _setup_batch(trainer, batch)

        output, loss_dict, input_lengths = _forward_and_backward(
            trainer,
            images,
            orig_h,
            orig_w,
            gt_regions,
            gt_plate_types,
            gt_texts,
            gt_format,
            gt_country,
            sampling_prob,
            current_epoch,
        )

        avg_batch_ms = _compute_avg_batch_ms(epoch_start, batch_idx)

        step, did_step, grad_norm = _optimizer_step(trainer, step)
        if did_step:
            last_grad_norm = grad_norm

        cur_lr = trainer.optimizer.param_groups[0]["lr"]
        _update_running_loss(running, loss_dict)

        logger.debug(
            "Batch %d/%d loss=%.4f ctc=%.4f lr=%.4f",
            batch_idx + 1,
            len(loader),
            running["loss"],
            running.get("ctc", 0.0),
            cur_lr,
        )

        # Real-time step-level logging to tracker (P1)
        if did_step and tracker is not None:
            global_step += 1
            _log_tracker_step(
                tracker,
                global_step,
                log_interval,
                log_grad_interval,
                running,
                cur_lr,
                grad_norm,
            )

        batches_since_update += 1
        if _should_update_progress(
            batch_idx,
            len(loader),
            batches_since_update,
            trainer.config.update_every_n_batches,
        ):
            _update_progress(
                trainer,
                output,
                gt_format,
                gt_country,
                gt_texts,
                running,
                avg_batch_ms,
                progress_display,
                task_id,
                batches_since_update,
                running_fmt_acc,
                running_ctry_acc,
                running_plate_acc,
                running_char_acc,
                input_lengths=input_lengths,
                grad_norm=last_grad_norm,
                accum_step=step % accum_steps if not did_step else accum_steps,
                accum_total=accum_steps,
            )
            batches_since_update = 0

        if trainer._interrupt_requested or trainer._force_stop:
            break

    running["avg_batch_ms"] = avg_batch_ms

    # If force-stopped, skip gradient flush and final accuracy computation
    # to exit as fast as possible — the interrupted checkpoint will capture
    # model state as-is.
    if trainer._force_stop:
        _compute_final_loss_avgs(running)
        return running

    # Flush any remaining accumulated gradients at end of epoch.
    # Only flush if there are un-stepped batches (avoids phantom optimizer
    # steps when len(loader) % accum_steps == 0).
    batches_since_step = step % trainer.config.gradient_accumulation_steps
    if batches_since_step > 0:
        step, did_step, grad_norm = _optimizer_step(trainer, step, force=True)
        if did_step and tracker is not None:
            global_step += 1
            _log_tracker_step(
                tracker,
                global_step,
                log_interval,
                log_grad_interval,
                running,
                trainer.optimizer.param_groups[0]["lr"],
                grad_norm,
            )

    _compute_final_accuracies(
        running,
        running_fmt_acc,
        running_ctry_acc,
        running_plate_acc,
        running_char_acc,
    )
    _compute_final_loss_avgs(running)
    return running
