"""Plate domain layer."""

from redstar_plate_ocr.plate.config import PLATE_TYPES, PlateConfig
from redstar_plate_ocr.plate.forbidden import ForbiddenFilter
from redstar_plate_ocr.plate.pattern import (
    PatternValidator,
    ValidationResult,
)
from redstar_plate_ocr.plate.postprocessor import PostProcessor
from redstar_plate_ocr.plate.results import RawResult, RecognitionResult

__all__ = [
    "PLATE_TYPES",
    "PlateConfig",
    "ForbiddenFilter",
    "PatternValidator",
    "PostProcessor",
    "RawResult",
    "RecognitionResult",
    "ValidationResult",
]
