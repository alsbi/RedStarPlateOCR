"""Checkpoint save/load utilities."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import torch

from redstar_plate_ocr.plate.config import PlateConfig

logger = logging.getLogger(__name__)


def save_checkpoint(
    ckpt: dict[str, Any],
    path: Path,
    save_thread: threading.Thread | None = None,
) -> threading.Thread:
    """Save checkpoint asynchronously.

    Args:
        ckpt: Checkpoint dict.
        path: Output path.
        save_thread: Previous save thread to wait for.

    Returns:
        New save thread.
    """
    if save_thread is not None:
        save_thread.join()

    path.parent.mkdir(parents=True, exist_ok=True)

    def _write() -> None:
        torch.save(ckpt, path)
        logger.info("Checkpoint saved: %s", path)

    thread = threading.Thread(target=_write, daemon=True)
    thread.start()
    return thread


def build_checkpoint(
    epoch: int,
    model_state: dict,
    optimizer_state: dict,
    scheduler_state: dict,
    scaler_state: dict,
    best_metric: float,
    plate_config_yaml: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build checkpoint dict."""
    ckpt: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "scheduler_state_dict": scheduler_state,
        "scaler_state_dict": scaler_state,
        "best_metric": best_metric,
    }
    if plate_config_yaml is not None:
        ckpt["plate_config_yaml"] = plate_config_yaml
    ckpt.update(extra)
    return ckpt


def validate_checkpoint_compat(
    ckpt: dict[str, Any],
    plate_config: PlateConfig,
) -> list[str]:
    """Validate checkpoint compatibility. Returns list of warnings."""
    warnings_list: list[str] = []
    saved = ckpt.get("plate_config_yaml")
    if saved is None:
        warnings_list.append(
            "Checkpoint has no plate_config_yaml — "
            "skipping compatibility validation"
        )
        return warnings_list
    try:
        saved_config = PlateConfig.from_yaml_string(saved)
    except Exception as e:
        warnings_list.append(f"Cannot parse saved plate_config: {e}")
        return warnings_list

    if set(saved_config.country_list) != set(plate_config.country_list):
        raise ValueError(
            f"Checkpoint countries {sorted(saved_config.country_list)} "
            f"!= current {sorted(plate_config.country_list)}"
        )
    if saved_config.union_alphabet_size != plate_config.union_alphabet_size:
        raise ValueError(
            f"Checkpoint alphabet_size "
            f"{saved_config.union_alphabet_size} "
            f"!= current {plate_config.union_alphabet_size}"
        )
    return warnings_list


def _migrate_country_head(
    state_dict: dict[str, Any],
    num_c: int,
) -> list[str]:
    """Trim country_head params from N+1 to N classes."""
    warnings_list: list[str] = []
    country_keys = [k for k in state_dict if k.startswith("country_head.fc.")]
    for key in country_keys:
        tensor = state_dict[key]
        if tensor.shape[0] == num_c + 1:
            state_dict[key] = tensor[:num_c]
            warnings_list.append(
                f"Migrated {key}: {num_c + 1} → {num_c} (removed sentinel)"
            )
    return warnings_list


def migrate_checkpoint(
    state_dict: dict[str, Any],
    plate_config: PlateConfig,
) -> list[str]:
    """Migrate checkpoint state_dict for compatibility.

    Handles:
    - CountryHead sentinel removal (8→N classes)
    - Country index remapping from old→new indices

    Returns list of migration warnings.
    """
    warnings_list: list[str] = []
    num_c = plate_config.num_countries

    warnings_list.extend(_migrate_country_head(state_dict, num_c))

    if warnings_list:
        logger.info("Checkpoint migrated: %s", warnings_list)
    return warnings_list


def _migrate_optimizer_entry(
    state_entry: dict[str, Any],
    num_c: int,
    name: str,
) -> list[str]:
    """Trim optimizer state for one country_head param."""
    warnings_list: list[str] = []
    for skey in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        t = state_entry.get(skey)
        if t is not None and t.shape[0] == num_c + 1:
            state_entry[skey] = t[:num_c]
            warnings_list.append(
                f"Migrated optimizer {name}.{skey}: {num_c + 1} → {num_c}"
            )
    return warnings_list


def migrate_optimizer_state(
    optimizer_state_dict: dict[str, Any],
    model: torch.nn.Module,
    plate_config: PlateConfig,
) -> list[str]:
    """Migrate optimizer state for resized country_head parameters.

    Must be called AFTER model.load_state_dict() so that
    model.named_parameters() reflects the (already trimmed) shapes.

    Returns list of migration warnings.
    """
    warnings_list: list[str] = []
    num_c = plate_config.num_countries
    opt_state = optimizer_state_dict.get("state")
    if opt_state is None:
        return warnings_list

    params = list(model.named_parameters())
    _migrate_country_head_entries(opt_state, params, num_c, warnings_list)

    if warnings_list:
        logger.info("Optimizer state migrated: %s", warnings_list)
    return warnings_list


def _migrate_country_head_entries(
    opt_state: dict[int, Any],
    params: list[tuple[str, torch.nn.Parameter]],
    num_c: int,
    warnings_list: list[str],
) -> None:
    for idx, (name, param) in enumerate(params):
        if not name.startswith("country_head.fc."):
            continue
        if param.shape[0] != num_c:
            continue
        state_entry = opt_state.get(idx)
        if state_entry is None:
            continue
        warnings_list.extend(
            _migrate_optimizer_entry(state_entry, num_c, name)
        )
