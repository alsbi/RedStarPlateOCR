"""Recognizer protocol, PyTorch and ONNX implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor

from redstar_plate_ocr.data.transforms import PreprocessPipeline
from redstar_plate_ocr.nn.mask_table import MASK_VALUE
from redstar_plate_ocr.pipeline.utils import softmax
from redstar_plate_ocr.plate.config import PlateConfig
from redstar_plate_ocr.plate.postprocess import BeamSearchDecoder
from redstar_plate_ocr.plate.postprocessor import PostProcessor
from redstar_plate_ocr.plate.results import RawResult, RecognitionResult


def _stable_log_softmax(
    x: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """Numerically stable log_softmax via max-trick (M8)."""
    x_max = x.max(axis=axis, keepdims=True)
    shifted = x - x_max
    return shifted - np.log(
        np.sum(np.exp(shifted), axis=axis, keepdims=True),
    )


@runtime_checkable
class Recognizer(Protocol):
    """Protocol for plate recognizers."""

    def recognize(self, image: np.ndarray) -> RecognitionResult: ...


class _ModelOutput:
    """Normalized output from _run_model for _build_raw."""

    __slots__ = ("fmt_probs", "ctry_probs", "ctc_tensor")

    def __init__(
        self,
        fmt_probs: np.ndarray,
        ctry_probs: np.ndarray,
        ctc_tensor: Tensor,
    ) -> None:
        self.fmt_probs = fmt_probs
        self.ctry_probs = ctry_probs
        self.ctc_tensor = ctc_tensor


class _BaseRecognizer:
    """Base class for plate recognizers.

    Template method: recognize() calls
    preprocess → _run_model → _build_raw → postprocess.
    """

    def __init__(
        self,
        plate_config: PlateConfig,
        beam_width: int = 1,
        country_confidence_threshold: float = 0.7,
        preprocess_params: dict | None = None,
    ) -> None:
        self.plate_config = plate_config
        self.beam_width = beam_width
        self.country_confidence_threshold = country_confidence_threshold
        if preprocess_params is not None:
            self.preprocess = PreprocessPipeline(**preprocess_params)
        else:
            self.preprocess = PreprocessPipeline()
        self._postprocessor = PostProcessor(
            plate_config,
        )
        self._beam_decoder = BeamSearchDecoder(
            plate_config.union_alphabet,
            beam_width=beam_width,
        )
        self._pattern_decoders: dict[str, BeamSearchDecoder] = {}

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        """Recognize a plate from an image.

        Args:
            image: Input image as a numpy array. Expected shape: (H, W, 3)
                with uint8 dtype.

        Returns:
            RecognitionResult with recognized text and selected metadata.

        Raises:
            ValueError: If image has incorrect shape or dtype.
        """
        if image.ndim != 3:
            raise ValueError(
                "Expected image with 3 dimensions (H, W, C), "
                f"got {image.ndim}D array"
            )
        if image.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel RGB image, got {image.shape[2]} channels"
            )
        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 image, got {image.dtype}")
        tensor, orig_h, orig_w = self.preprocess(image)
        output = self._run_model(tensor, orig_h, orig_w)
        raw, hypotheses = self._build_raw(output)
        return self._postprocessor.process(raw, hypotheses)

    def _run_model(
        self,
        tensor: Tensor,
        orig_h: int,
        orig_w: int,
    ) -> _ModelOutput:
        raise NotImplementedError

    def _build_raw(
        self,
        output: _ModelOutput,
    ) -> tuple[RawResult, list[tuple[str, float]] | None]:
        """Build RawResult and hypotheses from normalized output."""
        country_list = self.plate_config.country_list
        fmt_idx = int(np.argmax(output.fmt_probs))
        ctry_idx = int(np.argmax(output.ctry_probs))
        ctry_conf = float(output.ctry_probs[ctry_idx])
        plate_type = "square" if fmt_idx == 1 else "standard"
        needs_review = ctry_conf < self.country_confidence_threshold
        country = country_list[ctry_idx]

        decoder = self._get_decoder(country)
        text, text_conf = decoder.decode(output.ctc_tensor)
        hypotheses = decoder.decode_n(
            output.ctc_tensor,
            n=max(self.beam_width, 1),
        )

        raw = RawResult(
            text=text,
            text_confidence=text_conf,
            country=country,
            country_confidence=ctry_conf,
            plate_type=plate_type,
            needs_review=needs_review,
        )
        return raw, hypotheses

    def _get_decoder(self, country: str) -> BeamSearchDecoder:
        """Get pattern-constrained decoder for country."""
        if country in self._pattern_decoders:
            return self._pattern_decoders[country]
        region = self.plate_config.regions.get(country)
        if region is None or not region.pattern:
            return self._beam_decoder
        decoder = BeamSearchDecoder(
            self.plate_config.union_alphabet,
            beam_width=self.beam_width,
            pattern=region.pattern[0],
            valid_letters=region.valid_chars.letters,
            valid_digits=region.valid_chars.digits,
        )
        self._pattern_decoders[country] = decoder
        return decoder


class PyTorchRecognizer(_BaseRecognizer):
    """PyTorch-based plate recognizer.

    Pipeline: preprocess → model → beam search →
              PostProcessor (forbidden filter + pattern validation).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        plate_config: PlateConfig,
        beam_width: int = 1,
        country_confidence_threshold: float = 0.7,
        device: torch.device | None = None,
        preprocess_params: dict | None = None,
    ) -> None:
        super().__init__(
            plate_config=plate_config,
            beam_width=beam_width,
            country_confidence_threshold=country_confidence_threshold,
            preprocess_params=preprocess_params,
        )
        self.model = model
        if device is not None:
            self.device = torch.device(device)
        else:
            # Auto-detect: CUDA → MPS → CPU (same logic as Trainer)
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        # Move real model to the target device
        if isinstance(self.model, torch.nn.Module):
            self.model = self.model.to(self.device)

    def _run_model(
        self,
        tensor: Tensor,
        orig_h: int,
        orig_w: int,
    ) -> _ModelOutput:
        """Run PyTorch model forward pass."""
        self.model.eval()
        batch = tensor.unsqueeze(0).to(self.device)
        h_tensor = torch.tensor([orig_h], device=self.device)
        w_tensor = torch.tensor([orig_w], device=self.device)
        with torch.no_grad():
            result = self.model(
                batch,
                h_tensor,
                w_tensor,
                scheduled_sampling_prob=0.0,
            )
        country_list = self.plate_config.country_list
        fmt_probs = (
            torch.softmax(
                result.format_logits,
                dim=-1,
            )[0]
            .cpu()
            .numpy()
        )
        ctry_logits = result.country_logits[:, : len(country_list)]
        ctry_probs = (
            torch.softmax(
                ctry_logits,
                dim=-1,
            )[0]
            .cpu()
            .numpy()
        )
        ctc_tensor = result.ctc_output[0].cpu()
        return _ModelOutput(
            fmt_probs=fmt_probs,
            ctry_probs=ctry_probs,
            ctc_tensor=ctc_tensor,
        )


class ONNXRecognizer(_BaseRecognizer):
    """Recognizer on ONNX Runtime (T7.3).

    If *plate_config* is not provided it will be read from the
    ONNX model metadata (embedded during export).  Raises
    ``ValueError`` when the config is neither passed explicitly
    nor found in the model metadata.
    """

    def __init__(
        self,
        model_path: str,
        plate_config: PlateConfig | None = None,
        beam_width: int = 1,
        country_confidence_threshold: float = 0.7,
        preprocess_params: dict | None = None,
    ) -> None:
        if plate_config is None:
            plate_config = self._read_config_from_metadata(model_path)
        # Try to read preprocessing params from ONNX metadata if not provided
        if preprocess_params is None:
            preprocess_params = self._read_preprocess_from_metadata(
                model_path,
            )
        super().__init__(
            plate_config=plate_config,
            beam_width=beam_width,
            country_confidence_threshold=country_confidence_threshold,
            preprocess_params=preprocess_params,
        )
        import onnxruntime as ort  # type: ignore[import-untyped]

        self.session = ort.InferenceSession(model_path)

    @staticmethod
    def _read_config_from_metadata(
        model_path: str,
    ) -> PlateConfig:
        """Read PlateConfig from ONNX metadata or raise ValueError."""
        from redstar_plate_ocr.pipeline.exporter import (
            read_plate_config_from_onnx,
        )

        cfg = read_plate_config_from_onnx(model_path)
        if cfg is None:
            raise ValueError(
                "plate_config not provided and not found in "
                "ONNX model metadata.  Either pass plate_config "
                "explicitly or re-export with embed_config=True."
            )
        return cfg

    @staticmethod
    def _read_preprocess_from_metadata(
        model_path: str,
    ) -> dict | None:
        """Read preprocessing params from ONNX metadata."""
        from redstar_plate_ocr.pipeline.exporter import (
            read_preprocess_from_onnx,
        )

        return read_preprocess_from_onnx(model_path)

    def _run_model(
        self,
        tensor: Tensor,
        orig_h: int,
        orig_w: int,
    ) -> _ModelOutput:
        """Run ONNX inference and process output."""
        batch = tensor.unsqueeze(0).numpy()
        h_arr = np.array([orig_h], dtype=np.int64)
        w_arr = np.array([orig_w], dtype=np.int64)

        outputs = self.session.run(
            None,
            {"image": batch, "orig_h": h_arr, "orig_w": w_arr},
        )
        fmt_logits: np.ndarray = outputs[0]  # type: ignore[assignment]
        ctry_logits: np.ndarray = outputs[1]  # type: ignore[assignment]
        std_ctc: np.ndarray = outputs[2]  # type: ignore[assignment]
        sq_ctc: np.ndarray = outputs[3]  # type: ignore[assignment]

        country_list = self.plate_config.country_list
        fmt_probs = softmax(fmt_logits[0])
        ctry_probs = softmax(ctry_logits[0, : len(country_list)])

        fmt_idx = int(np.argmax(fmt_probs))
        plate_type = "square" if fmt_idx == 1 else "standard"
        ctry_idx = int(np.argmax(ctry_probs))
        country = country_list[ctry_idx]

        ctc_logits = self._select_ctc(std_ctc, sq_ctc, plate_type)
        mask = self._build_numpy_mask(country)
        ctc_masked = ctc_logits + mask[np.newaxis, :]
        ctc_probs = _stable_log_softmax(ctc_masked, axis=-1)
        ctc_tensor = torch.from_numpy(ctc_probs)

        return _ModelOutput(
            fmt_probs=fmt_probs,
            ctry_probs=ctry_probs,
            ctc_tensor=ctc_tensor,
        )

    def _build_numpy_mask(self, country: str) -> np.ndarray:
        """Build numpy soft mask for ONNX inference."""
        allowed = self.plate_config.get_allowed_indices(country)
        mask = np.full(
            self.plate_config.union_alphabet_size,
            MASK_VALUE,
            dtype=np.float32,
        )
        mask[allowed] = 0.0
        return mask

    def _select_ctc(
        self,
        std_ctc: np.ndarray,
        sq_ctc: np.ndarray,
        plate_type: str,
    ) -> np.ndarray:
        """Select the right CTC output by plate type."""
        if plate_type == "square":
            return sq_ctc[0]
        return std_ctc[0]
