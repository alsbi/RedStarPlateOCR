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
    from redstar_plate_ocr.pipeline.trainer import Trainer

logger = logging.getLogger(__name__)


_BatchTensors = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, list[str],
    list[str], list[str], torch.Tensor, torch.Tensor,
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
    format_labels = [
        1 if pt == "square" else 0 for pt in gt_plate_types
    ]
    gt_format = torch.tensor(
        format_labels, dtype=torch.long, device=trainer.device
    )
    gt_country = trainer.model.encode_countries(gt_regions).to(
        trainer.device
    )
    return (
        images, orig_h, orig_w, gt_regions, gt_plate_types,
        gt_texts, gt_format, gt_country,
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
    """Backward pass with optional AMP scaling."""
    if trainer.use_amp:
        trainer.scaler.scale(loss).backward()
    else:
        loss.backward()


def _optimizer_step(
    trainer: Trainer,
    step: int,
) -> tuple[int, bool]:
    """Gradient accumulation and optimizer step.

    Returns updated step counter and whether step was performed.
    """
    config = trainer.config
    if step % config.gradient_accumulation_steps != 0:
        return step + 1, False

    grad_clip = config.gradient_clip
    if trainer.use_amp:
        trainer.scaler.unscale_(trainer.optimizer)
        trainer.scaler.unscale_(trainer.country_optimizer)

    torch.nn.utils.clip_grad_norm_(
        trainer.optimizer.param_groups[0]["params"],
        max_norm=grad_clip,
    )
    torch.nn.utils.clip_grad_norm_(
        trainer.country_optimizer.param_groups[0]["params"],
        max_norm=grad_clip,
    )

    if trainer.use_amp:
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.step(trainer.country_optimizer)
        trainer.scaler.update()
    else:
        trainer.optimizer.step()
        trainer.country_optimizer.step()

    trainer.optimizer.zero_grad()
    trainer.country_optimizer.zero_grad()
    return step + 1, True


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
            output.ctc_output[i], union_alphabet,
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
) -> str:
    """Format progress-bar stats string."""
    left = (
        f"loss={running['loss']:.4f} "
        f"plate={plate_acc:.3%} "
        f"char={char_acc:.3%} "
        f"region={ctry_acc:.3%} "
        f"fmt={fmt_acc:.3%}"
    )
    right = f"{avg_batch_ms:.0f}ms"
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
) -> None:
    """Compute batch accuracy and update progress display."""
    with torch.no_grad():
        result = _compute_batch_accuracy(
            output, gt_format, gt_country, gt_texts,
            trainer.model.plate_config,
            input_lengths=input_lengths,
        )
        fmt_acc, ctry_acc, plate_acc, char_acc = result
    running_fmt_acc.append(fmt_acc)
    running_ctry_acc.append(ctry_acc)
    running_plate_acc.append(plate_acc)
    running_char_acc.append(char_acc)
    stats = _format_batch_stats(
        running, fmt_acc, ctry_acc, plate_acc, char_acc,
        avg_batch_ms=avg_batch_ms,
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
    """Update running loss dictionary from loss dict."""
    running["loss"] = loss_dict["total"].item()
    running["ctc"] = loss_dict["ctc"].item()
    running["country"] = loss_dict["country"].item()
    running["format"] = loss_dict["format"].item()
    if "order" in loss_dict:
        running["order"] = loss_dict["order"].item()


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


def run_train_epoch(
    trainer: Trainer,
    loader,
    sampling_prob: float,
    progress_display: ProgressDisplay,
    task_id: TaskID,
    epoch_start: float = 0.0,
    current_epoch: int = 0,
) -> dict[str, float]:
    """Run one training epoch with Rich progress bar."""
    trainer.model.train()
    trainer.optimizer.zero_grad()
    trainer.country_optimizer.zero_grad()
    step = 0
    running: dict[str, float] = {}
    batches_since_update = 0
    running_fmt_acc: list[float] = []
    running_ctry_acc: list[float] = []
    running_plate_acc: list[float] = []
    running_char_acc: list[float] = []
    avg_batch_ms: float = 0.0

    for batch_idx, batch in enumerate(loader):
        (
            images, orig_h, orig_w, gt_regions, gt_plate_types,
            gt_texts, gt_format, gt_country,
        ) = _setup_batch(trainer, batch)

        with torch.amp.autocast(
            trainer.device.type, enabled=trainer.use_amp
        ):
            output = trainer.model(
                images, orig_h, orig_w,
                gt_countries=gt_regions,
                gt_plate_types=gt_plate_types,
                scheduled_sampling_prob=sampling_prob,
                epoch=current_epoch,
            )

            # Compute input_lengths once for both loss and metrics
            seq_len = output.ctc_output.shape[1]
            per_sample_types = list(gt_plate_types)
            input_lengths = trainer.model.compression.compute_input_lengths(
                output.content_mask, per_sample_types
            ).to(trainer.device)
            input_lengths = input_lengths.clamp(min=2, max=seq_len)

            loss_dict, total_loss = _compute_loss_with_lengths(
                trainer, output, gt_format, gt_country, gt_texts,
                input_lengths,
            )
            loss = total_loss / trainer.config.gradient_accumulation_steps

            if epoch_start > 0.0:
                elapsed = (time.monotonic() - epoch_start) * 1000
                avg_batch_ms = elapsed / (batch_idx + 1)

            _backward_step(trainer, loss)

        step, _ = _optimizer_step(trainer, step)

        cur_lr = trainer.optimizer.param_groups[0]["lr"]
        _update_running_loss(running, loss_dict)

        logger.debug(
            "Batch %d/%d loss=%.4f ctc=%.4f lr=%.4f",
            batch_idx + 1, len(loader), running["loss"], running["ctc"],
            cur_lr,
        )

        batches_since_update += 1
        if _should_update_progress(
            batch_idx,
            len(loader),
            batches_since_update,
            trainer.config.update_every_n_batches,
        ):
            _update_progress(
                trainer, output, gt_format, gt_country, gt_texts,
                running, avg_batch_ms, progress_display, task_id,
                batches_since_update,
                running_fmt_acc, running_ctry_acc,
                running_plate_acc, running_char_acc,
                input_lengths=input_lengths,
            )
            batches_since_update = 0

        if trainer._interrupt_requested:
            break

    running["avg_batch_ms"] = avg_batch_ms
    _compute_final_accuracies(
        running,
        running_fmt_acc,
        running_ctry_acc,
        running_plate_acc,
        running_char_acc,
    )
    return running
