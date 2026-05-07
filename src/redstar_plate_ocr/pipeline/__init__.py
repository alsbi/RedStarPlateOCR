"""Pipeline layer: training, evaluation, inference."""

from redstar_plate_ocr.pipeline.recognizer import (
    ONNXRecognizer,
    PyTorchRecognizer,
    Recognizer,
)

__all__ = [
    "ONNXRecognizer",
    "PyTorchRecognizer",
    "Recognizer",
    "Trainer",
    "TrainingConfig",
]


def __getattr__(name):
    if name == "Trainer":
        from redstar_plate_ocr.pipeline.trainer import Trainer

        return Trainer
    if name == "TrainingConfig":
        from redstar_plate_ocr.pipeline.training_config import (
            TrainingConfig,
        )

        return TrainingConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
