"""Value objects for recognition results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RawResult:
    """Raw recognition result before post-processing."""

    text: str
    text_confidence: float
    country: str
    country_confidence: float
    plate_type: str
    needs_review: bool = False


@dataclass(frozen=True)
class RecognitionResult:
    """Result of plate recognition."""

    text: str
    text_confidence: float
    country: str
    country_confidence: float
    plate_type: str  # "standard" | "square"
    needs_review: bool = False
