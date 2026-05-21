"""Dataset validation logic."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from redstar_plate_ocr.pipeline.utils import find_csv

if TYPE_CHECKING:
    from redstar_plate_ocr.plate.config import PlateConfig

logger = logging.getLogger(__name__)


def _validate_image(
    idx: int,
    row: dict,
    data_dir: str,
) -> list[str]:
    """Validate image path for a row."""
    img_path_raw = row.get("image_path", "").strip()
    if not img_path_raw:
        return [f"Row {idx}: missing image_path"]
    img_path = Path(data_dir) / img_path_raw
    if not img_path.exists():
        return [f"Row {idx}: image not found: {img_path}"]
    return []


def _validate_region_and_type(
    idx: int,
    row: dict,
    plate_config: PlateConfig,
) -> list[str]:
    """Validate region and plate_type; return errors or []."""
    region = row.get("region", "")
    if region not in plate_config.regions:
        return [f"Row {idx}: unknown region: {region}"]

    plate_type = row.get("plate_type", "")
    from redstar_plate_ocr.plate.config import PLATE_TYPES  # cycle-avoid

    if plate_type not in PLATE_TYPES:
        return [
            f"Row {idx}: invalid plate_type "
            f"'{plate_type}' for region '{region}'",
        ]
    return []


def _validate_text_chars(
    idx: int,
    row: dict,
    plate_config: PlateConfig,
) -> list[str]:
    """Validate text characters against region alphabet."""
    region = row.get("region", "")
    region_cfg = plate_config.regions.get(region)
    if region_cfg is None:
        return []
    plate_text = row.get("plate_text", "")
    alphabet = region_cfg.raw_alphabet()
    bad = [c for c in plate_text if c not in alphabet]
    if bad:
        return [
            f"Row {idx}: invalid chars {bad} in '{plate_text}' for {region}",
        ]
    return []


def validate_row(
    idx: int,
    row: dict,
    data_dir: str,
    plate_config: PlateConfig,
) -> list[str]:
    """Validate a single CSV row, return error strings."""
    errors: list[str] = []

    errors.extend(_validate_image(idx, row, data_dir))

    region_errors = _validate_region_and_type(idx, row, plate_config)
    errors.extend(region_errors)

    if not region_errors:
        errors.extend(_validate_text_chars(idx, row, plate_config))

    return errors


def _load_csv_rows(csv_path: Path) -> tuple[list[dict], str]:
    """Load CSV rows; return (rows, error) where error is empty on success."""
    try:
        with open(csv_path, newline="") as f:
            return list(csv.DictReader(f)), ""
    except Exception as e:
        return [], f"Cannot read CSV: {e}"


def _count_sample(
    counts: dict[str, dict[str, int]],
    row: dict,
) -> None:
    """Increment count for region/plate_type."""
    region = row.get("region", "")
    plate_type = row.get("plate_type", "")
    counts.setdefault(region, {})
    counts[region].setdefault(plate_type, 0)
    counts[region][plate_type] += 1


def _log_counts(counts: dict[str, dict[str, int]]) -> None:
    """Log sample counts per region/plate_type."""
    for region, types in sorted(counts.items()):
        for pt, cnt in sorted(types.items()):
            logger.info("  %s/%s: %d samples", region, pt, cnt)


def validate_dataset(
    plate_config_path: str,
    data_dir: str,
    split: str,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Validate dataset, return (errors, counts).

    counts: dict[region][plate_type] -> count
    """
    from redstar_plate_ocr.plate.config import PlateConfig  # cycle-avoid

    pc = PlateConfig.from_yaml(plate_config_path)
    csv_path = Path(find_csv(data_dir, split))
    errors: list[str] = []

    if not csv_path.exists():
        errors.append(f"CSV not found: {csv_path}")
        return errors, {}

    rows, err = _load_csv_rows(csv_path)
    if err:
        errors.append(err)
        return errors, {}

    counts: dict[str, dict[str, int]] = {}
    for i, row in enumerate(rows):
        errors.extend(validate_row(i, row, data_dir, pc))
        _count_sample(counts, row)

    _log_counts(counts)
    return errors, counts
