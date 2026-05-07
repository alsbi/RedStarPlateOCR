"""Backward-compatible re-export from pipeline.preprocess."""

from redstar_plate_ocr.pipeline.preprocess import (  # noqa: F401
    PreprocessPipeline,
    auto_unpad,
)

__all__ = ["PreprocessPipeline", "auto_unpad"]
