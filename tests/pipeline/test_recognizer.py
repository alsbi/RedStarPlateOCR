"""Tests for PyTorchRecognizer (T6.4)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch

from redstar_plate_ocr.pipeline.recognizer import (
    PyTorchRecognizer,
    Recognizer,
)
from redstar_plate_ocr.plate.config import PlateConfig
from redstar_plate_ocr.plate.results import RecognitionResult


def _make_mock_model(
    plate_config: PlateConfig,
) -> MagicMock:
    """Create a mock model that returns plausible output."""
    from redstar_plate_ocr.nn.model import ModelOutput

    mock = MagicMock()
    # Simulate format=standard (0), country=RU (0)
    format_logits = torch.tensor([[10.0, 0.0]])
    country_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # CTC output: (1, 64, union_alphabet_size)
    union_size = plate_config.union_alphabet_size
    ctc_output = torch.zeros(1, 64, union_size)
    # Make blank dominant everywhere
    ctc_output[:, :, -1] = 5.0
    content_mask = torch.ones(1, 1, 20, 64)

    mock.return_value = ModelOutput(
        format_logits=format_logits,
        country_logits=country_logits,
        ctc_output=ctc_output,
        content_mask=content_mask,
        plate_types=["standard"],
    )
    mock.plate_config = plate_config
    return mock


class TestRecognizerProtocol:
    """Recognizer is a Protocol."""

    def test_pytorch_recognizer_implements_protocol(
        self,
        plate_config: PlateConfig,
    ) -> None:
        """PyTorchRecognizer satisfies Recognizer protocol."""
        pc = plate_config
        model = _make_mock_model(pc)
        recognizer = PyTorchRecognizer(model=model, plate_config=pc)
        assert isinstance(recognizer, Recognizer)


class TestPyTorchRecognizer:
    """PyTorchRecognizer integration tests."""

    def test_recognize_returns_result(
        self,
        plate_config: PlateConfig,
    ) -> None:
        """recognize() returns RecognitionResult."""
        pc = plate_config
        model = _make_mock_model(pc)
        recognizer = PyTorchRecognizer(model=model, plate_config=pc)
        image = np.zeros((80, 256, 3), dtype=np.uint8)
        result = recognizer.recognize(image)
        assert isinstance(result, RecognitionResult)
        assert isinstance(result.text, str)
        assert isinstance(result.country, str)
        assert isinstance(result.plate_type, str)
        assert 0.0 <= result.text_confidence <= 1.0
        assert 0.0 <= result.country_confidence <= 1.0

    def test_recognize_country_confidence_threshold(
        self,
        plate_config: PlateConfig,
    ) -> None:
        """Low country confidence → best real country + needs_review."""
        pc = plate_config
        model = _make_mock_model(pc)
        # Make country logits ambiguous
        from redstar_plate_ocr.nn.model import ModelOutput

        union_size = pc.union_alphabet_size
        model.return_value = ModelOutput(
            format_logits=torch.tensor([[10.0, 0.0]]),
            country_logits=torch.zeros(1, 8),
            ctc_output=torch.zeros(1, 64, union_size),
            content_mask=torch.ones(1, 1, 20, 64),
            plate_types=["standard"],
        )
        recognizer = PyTorchRecognizer(model=model, plate_config=pc)
        image = np.zeros((80, 256, 3), dtype=np.uint8)
        result = recognizer.recognize(image)
        # Low confidence → best real country + needs_review
        assert result.country in pc.country_list
        assert result.needs_review is True

    def test_recognize_format_prediction(
        self,
        plate_config: PlateConfig,
    ) -> None:
        """Format prediction from format_logits."""
        pc = plate_config
        model = _make_mock_model(pc)
        from redstar_plate_ocr.nn.model import ModelOutput

        union_size = pc.union_alphabet_size
        model.return_value = ModelOutput(
            format_logits=torch.tensor([[0.0, 10.0]]),  # square
            country_logits=torch.tensor(
                [[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            ),
            ctc_output=torch.zeros(1, 128, union_size),
            content_mask=torch.ones(1, 1, 20, 64),
            plate_types=["square"],
        )
        recognizer = PyTorchRecognizer(model=model, plate_config=pc)
        image = np.zeros((80, 256, 3), dtype=np.uint8)
        result = recognizer.recognize(image)
        assert result.plate_type == "square"

    def test_recognize_square_kept_for_any_country(
        self,
        plate_config: PlateConfig,
    ) -> None:
        """Square prediction is kept for any country
        (square is always valid)."""
        pc = plate_config
        model = _make_mock_model(pc)
        from redstar_plate_ocr.nn.model import ModelOutput

        uz_idx = pc.country_list.index("UZ")
        country_logits = torch.zeros(1, 8)
        country_logits[0, uz_idx] = 10.0
        union_size = pc.union_alphabet_size
        model.return_value = ModelOutput(
            format_logits=torch.tensor([[0.0, 10.0]]),  # square
            country_logits=country_logits,
            ctc_output=torch.zeros(1, 64, union_size),
            content_mask=torch.ones(1, 1, 20, 64),
            plate_types=["standard"],
        )
        recognizer = PyTorchRecognizer(model=model, plate_config=pc)
        image = np.zeros((80, 256, 3), dtype=np.uint8)
        result = recognizer.recognize(image)
        assert result.country == "UZ"
        assert result.plate_type == "square"

    def test_recognize_square_preserved_for_valid_country(
        self,
        plate_config: PlateConfig,
    ) -> None:
        """Square prediction kept for country with square type."""
        pc = plate_config
        model = _make_mock_model(pc)
        from redstar_plate_ocr.nn.model import ModelOutput

        union_size = pc.union_alphabet_size
        model.return_value = ModelOutput(
            format_logits=torch.tensor([[0.0, 10.0]]),  # square
            country_logits=torch.tensor(
                [[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            ),
            ctc_output=torch.zeros(1, 128, union_size),
            content_mask=torch.ones(1, 1, 20, 64),
            plate_types=["square"],
        )
        recognizer = PyTorchRecognizer(model=model, plate_config=pc)
        image = np.zeros((80, 256, 3), dtype=np.uint8)
        result = recognizer.recognize(image)
        assert result.plate_type == "square"
