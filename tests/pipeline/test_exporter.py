"""Tests for ONNX Exporter and ONNXRecognizer (T7.2, T7.3)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from redstar_plate_ocr.nn.model import PlateOCRModel
from redstar_plate_ocr.plate.config import PlateConfig
from redstar_plate_ocr.plate.results import RecognitionResult


def _make_model(plate_config: PlateConfig) -> PlateOCRModel:
    """Create a small model for testing."""
    return PlateOCRModel(
        plate_config=plate_config,
        backbone_cfg={
            "stem_channels": 16,
            "stage1_channels": 16,
            "stage1_blocks": 1,
            "stage2_channels": 32,
            "stage2_blocks": 1,
            "se_reduction": 4,
        },
        lstm_cfg={"input_size": 32, "hidden_size": 32, "num_layers": 1},
    )


class TestExporter:
    """ONNX Exporter tests."""

    def test_export_onnx_creates_file(self, plate_config: PlateConfig) -> None:
        """export_onnx creates an ONNX file."""
        onnx = pytest.importorskip("onnx")
        pc = plate_config
        model = _make_model(pc)
        model.eval()
        from redstar_plate_ocr.pipeline.exporter import Exporter

        exporter = Exporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "model.onnx")
            exporter.export_onnx(
                model,
                out_path,
                opset_version=17,
                simplify=False,
                dynamic_batch=False,
                embed_config=False,
            )
            assert Path(out_path).exists()
            onnx_model = onnx.load(out_path)
            assert len(onnx_model.graph.input) >= 1

    def test_embed_config_writes_metadata(
        self,
        plate_config: PlateConfig,
    ) -> None:
        """embed_config=True writes plate_config_yaml into metadata."""
        onnx = pytest.importorskip("onnx")
        pc = plate_config
        model = _make_model(pc)
        model.eval()
        from redstar_plate_ocr.pipeline.exporter import Exporter

        exporter = Exporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "model.onnx")
            exporter.export_onnx(
                model,
                out_path,
                opset_version=17,
                simplify=False,
                dynamic_batch=False,
                embed_config=True,
            )
            onnx_model = onnx.load(out_path)
            meta = {p.key: p.value for p in onnx_model.metadata_props}
            assert "plate_config_yaml" in meta
            restored = PlateConfig.from_yaml_string(meta["plate_config_yaml"])
            assert restored.country_list == pc.country_list

    def test_embed_config_false_no_metadata(
        self,
        plate_config: PlateConfig,
    ) -> None:
        """embed_config=False does not write metadata."""
        onnx = pytest.importorskip("onnx")
        pc = plate_config
        model = _make_model(pc)
        model.eval()
        from redstar_plate_ocr.pipeline.exporter import Exporter

        exporter = Exporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "model.onnx")
            exporter.export_onnx(
                model,
                out_path,
                opset_version=17,
                simplify=False,
                dynamic_batch=False,
                embed_config=False,
            )
            onnx_model = onnx.load(out_path)
            meta = {p.key: p.value for p in onnx_model.metadata_props}
            assert "plate_config_yaml" not in meta


class TestReadPlateConfigFromOnnx:
    """Tests for read_plate_config_from_onnx helper."""

    def test_reads_embedded_config(
        self,
        plate_config: PlateConfig,
    ) -> None:
        """read_plate_config_from_onnx returns PlateConfig when embedded."""
        pytest.importorskip("onnx")
        pc = plate_config
        model = _make_model(pc)
        model.eval()
        from redstar_plate_ocr.pipeline.exporter import (
            Exporter,
            read_plate_config_from_onnx,
        )

        exporter = Exporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "model.onnx")
            exporter.export_onnx(
                model,
                out_path,
                opset_version=17,
                simplify=False,
                dynamic_batch=False,
                embed_config=True,
            )
            result = read_plate_config_from_onnx(out_path)
            assert result is not None
            assert result.country_list == pc.country_list

    def test_returns_none_when_no_config(
        self,
        plate_config: PlateConfig,
    ) -> None:
        """read_plate_config_from_onnx returns None when not embedded."""
        pytest.importorskip("onnx")
        pc = plate_config
        model = _make_model(pc)
        model.eval()
        from redstar_plate_ocr.pipeline.exporter import (
            Exporter,
            read_plate_config_from_onnx,
        )

        exporter = Exporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "model.onnx")
            exporter.export_onnx(
                model,
                out_path,
                opset_version=17,
                simplify=False,
                dynamic_batch=False,
                embed_config=False,
            )
            result = read_plate_config_from_onnx(out_path)
            assert result is None


class TestONNXRecognizer:
    """ONNXRecognizer tests."""

    def test_onnx_recognizer_returns_result(
        self, plate_config: PlateConfig
    ) -> None:
        """ONNXRecognizer.recognize returns RecognitionResult."""
        pytest.importorskip("onnxruntime")
        pc = plate_config
        model = _make_model(pc)
        model.eval()
        from redstar_plate_ocr.pipeline.exporter import Exporter

        exporter = Exporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = str(Path(tmpdir) / "model.onnx")
            exporter.export_onnx(
                model,
                onnx_path,
                opset_version=17,
                simplify=False,
                dynamic_batch=False,
            )
            from redstar_plate_ocr.pipeline.recognizer import (
                ONNXRecognizer,
            )

            rec = ONNXRecognizer(onnx_path, pc)
            image = np.zeros((80, 192, 3), dtype=np.uint8)
            result = rec.recognize(image)
            assert isinstance(result, RecognitionResult)
            assert isinstance(result.text, str)
            assert isinstance(result.country, str)
            assert isinstance(result.plate_type, str)

    def test_onnx_recognizer_reads_config_from_metadata(
        self,
        plate_config: PlateConfig,
    ) -> None:
        """ONNXRecognizer works without explicit plate_config."""
        pytest.importorskip("onnxruntime")
        pc = plate_config
        model = _make_model(pc)
        model.eval()
        from redstar_plate_ocr.pipeline.exporter import Exporter

        exporter = Exporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = str(Path(tmpdir) / "model.onnx")
            exporter.export_onnx(
                model,
                onnx_path,
                opset_version=17,
                simplify=False,
                dynamic_batch=False,
                embed_config=True,
            )
            from redstar_plate_ocr.pipeline.recognizer import (
                ONNXRecognizer,
            )

            # No plate_config passed — should be read from metadata
            rec = ONNXRecognizer(onnx_path)
            assert rec.plate_config.country_list == pc.country_list

            image = np.zeros((80, 192, 3), dtype=np.uint8)
            result = rec.recognize(image)
            assert isinstance(result, RecognitionResult)

    def test_onnx_recognizer_raises_without_config(
        self,
        plate_config: PlateConfig,
    ) -> None:
        """ONNXRecognizer raises ValueError when config is missing."""
        pytest.importorskip("onnxruntime")
        pc = plate_config
        model = _make_model(pc)
        model.eval()
        from redstar_plate_ocr.pipeline.exporter import Exporter

        exporter = Exporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = str(Path(tmpdir) / "model.onnx")
            exporter.export_onnx(
                model,
                onnx_path,
                opset_version=17,
                simplify=False,
                dynamic_batch=False,
                embed_config=False,
            )
            from redstar_plate_ocr.pipeline.recognizer import (
                ONNXRecognizer,
            )

            with pytest.raises(ValueError, match="plate_config not provided"):
                ONNXRecognizer(onnx_path)

