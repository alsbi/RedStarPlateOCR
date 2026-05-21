"""Value objects for recognition results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class RawResult:
    """Raw recognition result before post-processing."""

    text: str
    text_confidence: float
    country: str
    country_confidence: float
    plate_type: str
    needs_review: bool = False
    # CTC alignment: timestep index for each character in *text*.
    # None when not available (e.g. beam search decode).
    ctc_alignment: list[int] | None = None
    # CTC log-probabilities (T, V) tensor — needed for
    # logit-based adjacent-swap correction.
    ctc_logits: "Tensor | None" = field(default=None, repr=False)


@dataclass(frozen=True)
class RecognitionResult:
    """Result of plate recognition."""

    text: str
    text_confidence: float
    country: str
    country_confidence: float
    plate_type: str  # "standard" | "square"
    needs_review: bool = False
