"""ONNX export for PlateOCRModel."""

from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from redstar_plate_ocr.nn.model import PlateOCRModel
from redstar_plate_ocr.plate.config import PlateConfig

logger = logging.getLogger(__name__)

_METADATA_KEY = "plate_config_yaml"
_PREPROCESS_KEY = "preprocessing_json"


class ONNXWrapper(torch.nn.Module):
    """Wrapper computing both paths for ONNX export."""

    def __init__(
        self,
        model: PlateOCRModel,
    ) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        images: Tensor,
        orig_h: Tensor,
        orig_w: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute all paths with UnifiedCTCHead.forward_raw."""
        backbone_out = self.model.backbone(images)
        features = self.model.fusion(
            backbone_out.stage1,
            backbone_out.final,
        )
        content_mask = self.model.compression.compute_content_mask(
            orig_h, orig_w,
        )
        format_logits = self.model.format_head(
            features, content_mask=content_mask,
            orig_h=orig_h, orig_w=orig_w,
        )
        country_logits = self.model.country_head(features, content_mask)

        # Standard path
        comp_std = self.model.compression.forward_standard(
            features,
            orig_h,
            orig_w,
        )
        lstm_std = self.model.bilstm(comp_std)
        standard_ctc = self.model.ctc_head.forward_raw(lstm_std)

        # Square path
        comp_sq = self.model.compression.forward_square(
            features,
            orig_h,
            orig_w,
        )
        lstm_sq = self.model.bilstm(comp_sq)
        square_ctc = self.model.ctc_head.forward_raw(lstm_sq)

        return format_logits, country_logits, standard_ctc, square_ctc


class Exporter:
    """Export PlateOCRModel to ONNX format."""

    def export_onnx(
        self,
        model: PlateOCRModel,
        output_path: str,
        plate_config: PlateConfig | None = None,
        opset_version: int = 18,
        simplify: bool = True,
        dynamic_batch: bool = True,
        embed_config: bool = True,
        preprocessing: dict | None = None,
    ) -> None:
        """Export model to ONNX format.

        Args:
            model: PlateOCRModel to export.
            output_path: Destination .onnx file path.
            plate_config: Config to embed.  Defaults to
                ``model.plate_config`` when *embed_config* is True.
            opset_version: ONNX opset version.
            simplify: Run onnxsim simplification.
            dynamic_batch: Allow variable batch dimension.
            embed_config: Embed *plate_config* YAML into ONNX
                metadata so :class:`ONNXRecognizer` can load it
                automatically.
            preprocessing: Preprocessing config dict to embed into
                ONNX metadata.  Must contain canvas_height,
                canvas_width, pad_color, normalization.mean/std.
        """
        # ONNX tracing must happen on CPU — MPS/CUDA ops are not
        # universally supported by the ONNX tracer.
        export_device = torch.device("cpu")
        model = model.to(export_device)
        model.eval()
        wrapper = ONNXWrapper(model)
        wrapper.eval()

        if plate_config is None:
            plate_config = model.plate_config

        dummy_images = torch.randn(1, 3, 80, 192, device=export_device)
        dummy_orig_h = torch.tensor([80], dtype=torch.int64, device=export_device)
        dummy_orig_w = torch.tensor([192], dtype=torch.int64, device=export_device)

        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {"image": {0: "batch"}}

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Suppress torch.onnx.export graph dump to stdout
        with contextlib.redirect_stdout(io.StringIO()):
            torch.onnx.export(
                wrapper,
                (dummy_images, dummy_orig_h, dummy_orig_w),
                output_path,
                opset_version=opset_version,
                dynamic_axes=dynamic_axes,
                input_names=["image", "orig_h", "orig_w"],
                output_names=[
                    "format_logits",
                    "country_logits",
                    "standard_ctc",
                    "square_ctc",
                ],
                dynamo=False,
            )
        logger.info("ONNX exported to %s", output_path)

        if simplify:
            self._simplify(output_path)

        if embed_config:
            self._embed_metadata(output_path, plate_config, preprocessing)

        self._verify_onnx(wrapper, output_path)

    def _simplify(self, path: str) -> None:
        """Simplify ONNX graph with onnxsim."""
        try:
            import onnx  # type: ignore[import-untyped]
            import onnxsim  # type: ignore[import-untyped]  # noqa: F401

            model = onnx.load(path)
            simplified, check = onnxsim.simplify(model)
            if check:
                onnx.save(simplified, path)
                logger.info("ONNX simplified successfully")
            else:
                logger.warning("onnxsim check failed, keeping original")
        except ImportError:
            logger.warning("onnxsim not installed, skipping simplify")

    @staticmethod
    def _embed_metadata(
        path: str,
        plate_config: PlateConfig,
        preprocessing: dict | None = None,
    ) -> None:
        """Embed plate_config YAML and preprocessing params into ONNX model metadata."""
        import json

        import onnx  # type: ignore[import-untyped]

        onnx_model = onnx.load(path)
        yaml_str = plate_config.to_yaml_string()

        # Remove any existing entries to avoid duplicates
        keys_to_remove = {_METADATA_KEY, _PREPROCESS_KEY}
        onnx_model.metadata_props.extend(
            [
                p
                for p in onnx_model.metadata_props
                if p.key not in keys_to_remove
            ]
        )

        entry = onnx.StringStringEntryProto()
        entry.key = _METADATA_KEY
        entry.value = yaml_str
        onnx_model.metadata_props.append(entry)

        if preprocessing is not None:
            preproc_entry = onnx.StringStringEntryProto()
            preproc_entry.key = _PREPROCESS_KEY
            preproc_entry.value = json.dumps(preprocessing)
            onnx_model.metadata_props.append(preproc_entry)

        onnx.save(onnx_model, path)
        logger.info("Plate config embedded into ONNX metadata")

    def _verify_onnx(
        self,
        wrapper: ONNXWrapper,
        path: str,
    ) -> None:
        """Verify ONNX output matches PyTorch."""
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("onnxruntime not installed, skipping verification")
            return

        images = torch.randn(1, 3, 80, 192)
        orig_h = torch.tensor([80], dtype=torch.int64)
        orig_w = torch.tensor([192], dtype=torch.int64)

        with torch.no_grad():
            pt_out = wrapper(images, orig_h, orig_w)

        session = ort.InferenceSession(path)
        ort_out = session.run(
            None,
            {
                "image": images.numpy(),
                "orig_h": orig_h.numpy(),
                "orig_w": orig_w.numpy(),
            },
        )

        for i, (pt_t, ort_a) in enumerate(zip(pt_out, ort_out)):
            diff_arr = np.abs(pt_t.numpy() - ort_a)
            diff = float(np.nanmax(diff_arr)) if diff_arr.size else 0.0
            if diff > 1e-4:
                logger.warning(
                    "Output %d max diff: %.6f",
                    i,
                    diff,
                )
            else:
                logger.info("Output %d verified (diff=%.6f)", i, diff)


def read_plate_config_from_onnx(
    path: str,
) -> PlateConfig | None:
    """Read plate_config from ONNX metadata, or return None."""
    try:
        import onnx  # type: ignore[import-untyped]
    except ImportError:
        return None

    try:
        onnx_model = onnx.load(path)
    except Exception:
        return None

    for prop in onnx_model.metadata_props:
        if prop.key == _METADATA_KEY:
            return PlateConfig.from_yaml_string(prop.value)

    return None


def read_preprocess_from_onnx(
    path: str,
) -> dict | None:
    """Read preprocessing params from ONNX metadata, or return None.

    Returns a dict suitable for ``PreprocessPipeline(**params)``.
    """
    import json

    try:
        import onnx  # type: ignore[import-untyped]
    except ImportError:
        return None

    try:
        onnx_model = onnx.load(path)
    except Exception:
        return None

    for prop in onnx_model.metadata_props:
        if prop.key == _PREPROCESS_KEY:
            raw = json.loads(prop.value)
            return _preprocess_raw_to_pipeline_params(raw)

    return None


def _preprocess_raw_to_pipeline_params(
    raw: dict,
) -> dict:
    """Convert raw preprocessing config to PreprocessPipeline kwargs."""
    params: dict = {}
    if "canvas_height" in raw:
        params["canvas_height"] = raw["canvas_height"]
    if "canvas_width" in raw:
        params["canvas_width"] = raw["canvas_width"]
    if "pad_color" in raw:
        params["pad_color"] = raw["pad_color"]
    norm = raw.get("normalization", {})
    if "mean" in norm:
        params["mean"] = norm["mean"]
    if "std" in norm:
        params["std"] = norm["std"]
    return params
