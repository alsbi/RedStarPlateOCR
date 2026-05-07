"""Configuration helpers for CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redstar_plate_ocr.plate.config import PlateConfig


def _load_plate_config(path: str) -> PlateConfig:
    """Load PlateConfig from YAML."""
    from redstar_plate_ocr.plate.config import PlateConfig

    return PlateConfig.from_yaml(path)


_BACKBONE_KEYS = frozenset(
    {
        "stem_channels",
        "stage1_channels",
        "stage1_blocks",
        "stage2_channels",
        "stage2_blocks",
        "se_reduction",
        "drop_path_rate",
    }
)
_LSTM_KEYS = frozenset(
    {
        "input_size",
        "hidden_size",
        "num_layers",
        "dropout",
    }
)


def _head_hidden_kw(raw_ctc: dict) -> dict:
    head_hidden = raw_ctc.get("head_hidden_size")
    # 0 or falsy → treat as "no hidden layer" (None)
    return {"head_hidden": head_hidden} if head_hidden else {}


def _preproc_dim(raw_preproc: dict, key: str, default: int) -> int:
    return raw_preproc.get(key, default)


def _filter_keys(src: dict, allowed: frozenset) -> dict:
    return {k: v for k, v in src.items() if k in allowed}


def _model_kwargs_from_cfg(cfg: dict | None) -> dict:
    """Extract PlateOCRModel kwargs from model config dict."""
    if cfg is None:
        return {}
    raw_backbone = cfg.get("backbone", {})
    raw_lstm = cfg.get("lstm", {})
    raw_ctc = cfg.get("ctc", {})
    raw_preproc = cfg.get("preprocessing", {})
    return {
        "backbone_cfg": _filter_keys(
            raw_backbone, _BACKBONE_KEYS
        ),
        "classification_cfg": cfg.get("classification", {}),
        "lstm_cfg": _filter_keys(raw_lstm, _LSTM_KEYS),
        "canvas_height": _preproc_dim(
            raw_preproc, "canvas_height", 80
        ),
        "canvas_width": _preproc_dim(
            raw_preproc, "canvas_width", 192
        ),
        **_head_hidden_kw(raw_ctc),
        "char_aux": cfg.get("char_aux", {}).get("enabled", False),
    }
