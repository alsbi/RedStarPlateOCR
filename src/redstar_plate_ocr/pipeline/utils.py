"""Shared utility functions for pipeline modules."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def to_long_tensor(
    data: list | Tensor,
    device: torch.device | None = None,
) -> Tensor:
    """Convert list or Tensor to long Tensor without warning."""
    if isinstance(data, Tensor):
        t = data.detach().long()
        return t.to(device, non_blocking=True) if device else t
    return torch.tensor(data, dtype=torch.long, device=device)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def resolve_country_from_probs(
    country_probs: np.ndarray | Tensor,
    country_list: list[str],
) -> str:
    """Resolve country from probabilities.

    argmax always returns valid index (no sentinel class).
    """
    c_idx = int(country_probs.argmax())
    return country_list[c_idx]


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    mins, secs = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins}m{secs:02d}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h{mins:02d}m"


def find_csv(data_dir: str, split: str) -> str:
    """Find CSV file: try data_dir/split.csv then data_dir/split/split.csv."""
    p1 = Path(data_dir) / f"{split}.csv"
    if p1.exists():
        return str(p1)
    p2 = Path(data_dir) / split / f"{split}.csv"
    if p2.exists():
        return str(p2)
    return str(p1)


def greedy_decode(
    logits: Tensor,
    alphabet: str,
    input_length: int | None = None,
) -> str:
    """CTC greedy decode: argmax → remove blank → collapse.

    Args:
        logits: (T, alphabet_size) log-probabilities.
        alphabet: character set string.
        input_length: if given, only decode first N timesteps.

    Returns:
        Decoded text string.
    """
    blank_idx = len(alphabet)
    if input_length is not None and input_length < logits.shape[0]:
        logits = logits[:input_length]
    indices = logits.argmax(dim=1)
    if not indices.is_cpu:
        pad = torch.full(
            (1,),
            -1,
            device=indices.device,
            dtype=indices.dtype,
        )
        shifted = torch.cat([pad, indices[:-1]])
        keep = (indices != blank_idx) & (indices != shifted)
        valid_idx = indices[keep]
        valid_idx = valid_idx[valid_idx < len(alphabet)]
        return "".join(alphabet[i] for i in valid_idx.tolist())
    idx_np = indices.cpu().numpy()
    non_blank = idx_np != blank_idx
    shifted = np.empty_like(idx_np)
    shifted[0] = -1
    shifted[1:] = idx_np[:-1]
    keep = non_blank & (idx_np != shifted)
    chars_idx = idx_np[keep]
    valid = chars_idx < len(alphabet)
    return "".join(alphabet[i] for i in chars_idx[valid])


def _build_tags(
    plate: float,
    best_plate: float,
    cer: float,
    best_cer: float,
    is_first: bool,
) -> tuple[str, str]:
    """Return (plate_tag, cer_tag) based on improvement."""
    if is_first:
        return "", ""
    return (
        "↑" if plate > best_plate else "",
        "↓" if cer < best_cer else "",
    )


def _format_section_1(
    plate: float,
    cer: float,
    char: float,
    country: float,
    fmt: float,
    std: float,
    sq: float,
    plate_tag: str,
    cer_tag: str,
    cache_tag: str,
) -> str:
    """Format core metrics section."""
    return (
        f"🎯  plate={plate:.1%}{plate_tag}{cache_tag} "
        f"cer={cer:.4f}{cer_tag} "
        f"char={char:.1%} "
        f"region={country:.1%} "
        f"fmt={fmt:.1%} "
        f"std={std:.1%} "
        f"sq={sq:.1%}"
    )


def _format_sys(
    train_loss: float,
    epoch_duration: float,
) -> str:
    """Format system section (loss + duration)."""
    parts = [f"📉  loss={train_loss:.4f}"]
    if epoch_duration > 0:
        parts.append(f"⏱  {format_duration(epoch_duration)}")
    return " ".join(parts)


def format_epoch_stats(
    val_metrics: dict[str, float],
    best_metrics: dict[str, float],
    train_loss: float,
    *,
    is_cached: bool = False,
    epoch_duration: float = 0.0,
) -> str:
    """Format epoch stats for line 1 of progress bar.

    Three sections separated by │:
    1. Core metrics: plate, cer, char, region, fmt
    2. Per-country: BY=X% GE=X% ...
    3. System: loss, duration, ETA
    """
    is_first = not best_metrics

    plate = val_metrics.get("val_plate_accuracy", 0.0)
    best_plate = best_metrics.get("val_plate_accuracy", 0.0)
    cer = val_metrics.get("val_cer", 0.0)
    best_cer = best_metrics.get("val_cer", 0.0)
    char = val_metrics.get("val_char_accuracy", 0.0)
    country = val_metrics.get("val_country_accuracy", 0.0)
    fmt = val_metrics.get("val_format_accuracy", 0.0)
    std = val_metrics.get("val_standard_accuracy", 0.0)
    sq = val_metrics.get("val_square_accuracy", 0.0)

    plate_tag, cer_tag = _build_tags(
        plate,
        best_plate,
        cer,
        best_cer,
        is_first,
    )
    cache_tag = " (cached)" if is_cached else ""

    core = _format_section_1(
        plate,
        cer,
        char,
        country,
        fmt,
        std,
        sq,
        plate_tag,
        cer_tag,
        cache_tag,
    )
    per_country = _format_per_country(val_metrics)
    sys_str = _format_sys(train_loss, epoch_duration)

    if per_country:
        return f"{core} │ {per_country} │ {sys_str}"
    return f"{core} │ {sys_str}"


def format_train_epoch_stats(
    train_loss: float,
    best_metrics: dict[str, float],
    *,
    epoch_duration: float = 0.0,
) -> str:
    """Format epoch stats during training (val=N/A before first validation).

    Uses best_metrics if available (cached), otherwise shows placeholders.
    """
    if best_metrics:
        return format_epoch_stats(
            best_metrics,
            best_metrics,
            train_loss,
            is_cached=True,
            epoch_duration=epoch_duration,
        )
    core = "plate=— cer=— char=— region=— fmt=—"
    sys_parts: list[str] = [f"loss={train_loss:.4f}"]
    if epoch_duration > 0:
        sys_parts.append(format_duration(epoch_duration))
    sys_str = " ".join(sys_parts)
    return f"{core} │ {sys_str}"


def _country_flag(code: str) -> str:
    """Convert 2-letter country code to flag emoji (e.g. 'BY' → '🇧🇾')."""
    if len(code) != 2:
        return code
    base = ord("\U0001f1e6") - ord("A")  # Regional indicator A = 🇦
    ch0 = chr(base + ord(code[0].upper()))
    ch1 = chr(base + ord(code[1].upper()))
    return f"{ch0}{ch1} "


def _format_per_country(val_metrics: dict[str, float]) -> str:
    """Format per-country plate accuracy: 🇧🇾 98.2% 🇬🇪 72.7% ..."""
    parts: list[str] = []
    for key in sorted(val_metrics):
        if key.startswith("val_region_"):
            country = key[len("val_region_") :]
            acc = val_metrics[key]
            flag = _country_flag(country)
            parts.append(f"{flag}{acc:.1%}")
    return " ".join(parts)


def _build_main_log_line(
    epoch: int,
    total_epochs: int,
    phase: str,
    lr: float,
    train_loss: float,
    val_metrics: dict[str, float],
    time_str: str,
) -> str:
    """Build main epoch log line."""
    plate = val_metrics.get("val_plate_accuracy", 0.0)
    region = val_metrics.get("val_country_accuracy", 0.0)
    fmt = val_metrics.get("val_format_accuracy", 0.0)
    cer = val_metrics.get("val_cer", 0.0)
    char = val_metrics.get("val_char_accuracy", 0.0)
    std = val_metrics.get("val_standard_accuracy", 0.0)
    sq = val_metrics.get("val_square_accuracy", 0.0)
    parts = [
        f"Epoch {epoch + 1}/{total_epochs} [{phase}]",
        f"plate={plate:.4f}",
        f"std_plate={std:.4f}",
        f"sq_plate={sq:.4f}",
        f"region={region:.4f}",
        f"format={fmt:.4f}",
        f"cer={cer:.4f}",
        f"char={char:.4f}",
        f"loss={train_loss:.4f}",
        f"lr={lr:.4f}{time_str}",
    ]
    return " ".join(parts)


def _log_train_acc(
    log: logging.Logger,
    val_metrics: dict[str, float],
) -> None:
    """Log train accuracy if present."""
    if "train_plate_acc" not in val_metrics:
        return
    log.info(
        "Train accuracy: plate=%.4f fmt=%.4f country=%.4f char=%.4f",
        val_metrics.get("train_plate_acc", 0.0),
        val_metrics.get("train_fmt_acc", 0.0),
        val_metrics.get("train_ctry_acc", 0.0),
        val_metrics.get("train_char_acc", 0.0),
    )


def _log_region_stats(
    log: logging.Logger,
    val_metrics: dict[str, float],
) -> None:
    """Log per-region stats at DEBUG level."""
    for key, v in sorted(val_metrics.items()):
        if key.startswith("val_region_"):
            region_name = key[len("val_region_") :]
            log.debug("Region %s: plate=%.4f", region_name, v)


def _log_best_metrics(
    log: logging.Logger,
    best_metrics: dict[str, float],
) -> None:
    """Log best metrics at INFO level."""
    if not best_metrics:
        return
    best_plate = best_metrics.get("val_plate_accuracy", 0.0)
    log.info(
        "Best plate=%.4f cer=%.4f",
        best_plate,
        best_metrics.get("val_cer", 0.0),
    )


def log_epoch_summary(
    epoch: int,
    total_epochs: int,
    phase: str,
    lr: float,
    train_loss: float,
    val_metrics: dict[str, float],
    best_metrics: dict[str, float],
    logger_obj: object,
    epoch_duration: float = 0.0,
) -> None:
    """Log epoch summary based on verbose level."""
    log = (
        logger_obj
        if isinstance(logger_obj, logging.Logger)
        else logging.getLogger(__name__)
    )
    time_str = (
        f" time={format_duration(epoch_duration)}"
        if epoch_duration > 0
        else ""
    )
    log_line = _build_main_log_line(
        epoch,
        total_epochs,
        phase,
        lr,
        train_loss,
        val_metrics,
        time_str,
    )
    log.info(log_line)
    _log_train_acc(log, val_metrics)
    _log_region_stats(log, val_metrics)
    _log_best_metrics(log, best_metrics)
