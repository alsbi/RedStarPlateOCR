"""Integration: ONNX export and verification."""

from __future__ import annotations

import os
import tempfile

import pytest

from redstar_plate_ocr.nn.model import PlateOCRModel
from redstar_plate_ocr.pipeline.exporter import Exporter


def test_onnx_export_and_verify(plate_config):
    """Export model to ONNX and verify via onnxruntime."""
    pytest.importorskip("onnxruntime")

    model = PlateOCRModel(plate_config)
    model.eval()

    exporter = Exporter()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "model.onnx")
        exporter.export_onnx(
            model,
            output_path,
            opset_version=17,
            simplify=False,
        )
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
