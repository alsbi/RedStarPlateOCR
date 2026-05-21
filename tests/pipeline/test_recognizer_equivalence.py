"""T0.4: Verify PyTorchRecognizer and ONNXRecognizer produce
consistent results."""

import pytest


@pytest.fixture
def sample_image():
    """Create a simple test image (80x256 white rectangle)."""
    import numpy as np

    return np.ones((80, 256, 3), dtype=np.uint8) * 255


def test_pytorch_recognizer_returns_result(
    plate_config,
    sample_image,
):
    """PyTorchRecognizer.recognize() returns valid
    RecognitionResult."""
    from redstar_plate_ocr.nn.model import PlateOCRModel
    from redstar_plate_ocr.pipeline.recognizer import PyTorchRecognizer

    model = PlateOCRModel(plate_config=plate_config)
    model.eval()
    recognizer = PyTorchRecognizer(
        model=model,
        plate_config=plate_config,
    )
    result = recognizer.recognize(sample_image)

    # Must return a RecognitionResult
    assert hasattr(result, "text")
    assert hasattr(result, "country")
    assert hasattr(result, "plate_type")
    assert hasattr(result, "text_confidence")
    assert hasattr(result, "country_confidence")
    assert result.plate_type in ("standard", "square")
    assert 0.0 <= result.text_confidence <= 1.0
    assert 0.0 <= result.country_confidence <= 1.0


def test_onnx_export_and_inference(
    plate_config,
    sample_image,
    tmp_path,
):
    """ONNX export + inference produces valid result."""
    try:
        import onnxruntime  # type: ignore[import-untyped,reportMissingImports]  # noqa: F401
    except ImportError:
        pytest.skip("onnxruntime not installed")

    from redstar_plate_ocr.nn.model import PlateOCRModel
    from redstar_plate_ocr.pipeline.exporter import Exporter
    from redstar_plate_ocr.pipeline.recognizer import (
        ONNXRecognizer,
        PyTorchRecognizer,
    )

    model = PlateOCRModel(plate_config=plate_config)
    model.eval()

    # Export to ONNX
    onnx_path = str(tmp_path / "test_model.onnx")
    exporter = Exporter()
    exporter.export_onnx(model, onnx_path)

    # PyTorch inference
    pt_recognizer = PyTorchRecognizer(
        model=model,
        plate_config=plate_config,
    )
    pt_result = pt_recognizer.recognize(sample_image)

    # ONNX inference
    onnx_recognizer = ONNXRecognizer(
        model_path=onnx_path,
        plate_config=plate_config,
    )
    onnx_result = onnx_recognizer.recognize(sample_image)

    # Both must return valid results
    assert pt_result.plate_type == onnx_result.plate_type
    assert pt_result.country == onnx_result.country
    # Text may differ slightly due to numerical differences
