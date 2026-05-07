"""Backward-compatible re-export from pipeline.preprocess."""

from redstar_plate_ocr.pipeline.preprocess import PreprocessPipeline, auto_unpad  # noqa: F401

__all__ = ["PreprocessPipeline", "auto_unpad"]
