"""Neural network type definitions."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class ModelOutput:
    """Output of PlateOCRModel."""

    format_logits: Tensor
    country_logits: Tensor
    ctc_output: Tensor
    content_mask: Tensor
    plate_types: list[str]
    char_aux_logits: Tensor | None = None
