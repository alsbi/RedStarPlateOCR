"""Tests for detect_device() and get_onnx_providers()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from redstar_plate_ocr.pipeline.utils import (
    _resolve_cuda_backend,
    detect_device,
    get_onnx_providers,
)

# --- _resolve_cuda_backend ---


def test_resolve_cuda_backend_hip_none():
    """_resolve_cuda_backend returns 'cuda' when torch.version.hip is None."""
    with patch.object(torch.version, "hip", None):
        assert _resolve_cuda_backend() == "cuda"


def test_resolve_cuda_backend_hip_set():
    """_resolve_cuda_backend returns 'rocm' when torch.version.hip is set."""
    with patch.object(torch.version, "hip", "6.2.0"):
        assert _resolve_cuda_backend() == "rocm"


# --- detect_device: auto-detect ---


def test_detect_device_cpu():
    """detect_device returns (cpu, False, 'cpu') when no GPU available."""
    with (
        patch("redstar_plate_ocr.pipeline.utils.torch.cuda") as mock_cuda,
        patch(
            "redstar_plate_ocr.pipeline.utils.torch.backends",
        ) as mock_backends,
    ):
        mock_cuda.is_available.return_value = False
        mock_backends.mps.is_available.return_value = False
        device, amp, backend = detect_device(use_amp=True)
        assert device == torch.device("cpu")
        assert amp is False
        assert backend == "cpu"


def test_detect_device_cuda():
    """detect_device returns (cuda, True, 'cuda') on NVIDIA CUDA."""
    with (
        patch("redstar_plate_ocr.pipeline.utils.torch.cuda") as mock_cuda,
        patch.object(torch.version, "hip", None),
    ):
        mock_cuda.is_available.return_value = True
        device, amp, backend = detect_device(use_amp=True)
        assert device == torch.device("cuda")
        assert amp is True
        assert backend == "cuda"


def test_detect_device_rocm():
    """detect_device returns (cuda, True, 'rocm') on AMD ROCM."""
    with (
        patch("redstar_plate_ocr.pipeline.utils.torch.cuda") as mock_cuda,
        patch.object(torch.version, "hip", "6.2.0"),
    ):
        mock_cuda.is_available.return_value = True
        device, amp, backend = detect_device(use_amp=True)
        assert device == torch.device("cuda")
        assert amp is True
        assert backend == "rocm"


def test_detect_device_mps():
    """detect_device returns (mps, False, 'mps') on Apple MPS."""
    with (
        patch("redstar_plate_ocr.pipeline.utils.torch.cuda") as mock_cuda,
        patch(
            "redstar_plate_ocr.pipeline.utils.torch.backends",
        ) as mock_backends,
    ):
        mock_cuda.is_available.return_value = False
        mock_backends.mps.is_available.return_value = True
        device, amp, backend = detect_device(use_amp=True)
        assert device == torch.device("mps")
        assert amp is False
        assert backend == "mps"


def test_detect_device_amp_false():
    """detect_device with use_amp=False returns amp=False even on CUDA."""
    with (
        patch("redstar_plate_ocr.pipeline.utils.torch.cuda") as mock_cuda,
        patch.object(torch.version, "hip", None),
    ):
        mock_cuda.is_available.return_value = True
        device, amp, backend = detect_device(use_amp=False)
        assert device == torch.device("cuda")
        assert amp is False  # use_amp=False → amp=False
        assert backend == "cuda"


# --- detect_device: REDSTAR_DEVICE env var ---


def test_detect_device_env_cuda(monkeypatch):
    """REDSTAR_DEVICE=cuda forces CUDA."""
    monkeypatch.setenv("REDSTAR_DEVICE", "cuda")
    with (
        patch("redstar_plate_ocr.pipeline.utils.torch.cuda") as mock_cuda,
        patch.object(torch.version, "hip", None),
    ):
        mock_cuda.is_available.return_value = True
        device, amp, backend = detect_device(use_amp=True)
        assert device == torch.device("cuda")
        assert amp is True
        assert backend == "cuda"


def test_detect_device_env_cpu(monkeypatch):
    """REDSTAR_DEVICE=cpu forces CPU."""
    monkeypatch.setenv("REDSTAR_DEVICE", "cpu")
    device, amp, backend = detect_device(use_amp=True)
    assert device == torch.device("cpu")
    assert amp is False
    assert backend == "cpu"


def test_detect_device_env_mps(monkeypatch):
    """REDSTAR_DEVICE=mps forces MPS (when available)."""
    monkeypatch.setenv("REDSTAR_DEVICE", "mps")
    with (
        patch("redstar_plate_ocr.pipeline.utils.torch.cuda") as mock_cuda,
        patch(
            "redstar_plate_ocr.pipeline.utils.torch.backends",
        ) as mock_backends,
    ):
        mock_cuda.is_available.return_value = False
        mock_backends.mps.is_available.return_value = True
        device, amp, backend = detect_device(use_amp=True)
        assert device == torch.device("mps")
        assert amp is False
        assert backend == "mps"


def test_detect_device_env_cuda_unavailable_fallback(monkeypatch):
    """REDSTAR_DEVICE=cuda but no CUDA -> fallback with warning."""
    monkeypatch.setenv("REDSTAR_DEVICE", "cuda")
    with (
        patch("redstar_plate_ocr.pipeline.utils.torch.cuda") as mock_cuda,
        patch(
            "redstar_plate_ocr.pipeline.utils.torch.backends",
        ) as mock_backends,
    ):
        mock_cuda.is_available.return_value = False
        mock_backends.mps.is_available.return_value = False
        device, amp, backend = detect_device(use_amp=True)
        assert device == torch.device("cpu")
        assert amp is False
        assert backend == "cpu"


def test_detect_device_env_invalid_fallback(monkeypatch):
    """REDSTAR_DEVICE=invalid -> fallback with warning."""
    monkeypatch.setenv("REDSTAR_DEVICE", "nonsense")
    with (
        patch("redstar_plate_ocr.pipeline.utils.torch.cuda") as mock_cuda,
        patch(
            "redstar_plate_ocr.pipeline.utils.torch.backends",
        ) as mock_backends,
    ):
        mock_cuda.is_available.return_value = False
        mock_backends.mps.is_available.return_value = False
        device, amp, backend = detect_device(use_amp=True)
        assert device == torch.device("cpu")
        assert amp is False
        assert backend == "cpu"


# --- get_onnx_providers ---


def _mock_ort_import(providers: list[str]) -> MagicMock:
    """Helper: mock onnxruntime with given available providers."""
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = providers
    return mock_ort


def test_get_onnx_providers_no_ort():
    """get_onnx_providers returns ['CPUExecutionProvider'] when ORT missing."""

    def mock_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("not installed")
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        providers = get_onnx_providers()
        assert providers == ["CPUExecutionProvider"]


def test_get_onnx_providers_cpu_only():
    """get_onnx_providers only CPU when only CPU available."""
    mock_ort = _mock_ort_import(["CPUExecutionProvider"])

    def mock_import(name, *args, **kwargs):
        if name == "onnxruntime":
            return mock_ort
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        providers = get_onnx_providers()
        assert providers == ["CPUExecutionProvider"]


def test_get_onnx_providers_cuda():
    """get_onnx_providers returns CUDA+CPU when CUDA available."""
    mock_ort = _mock_ort_import(
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    def mock_import(name, *args, **kwargs):
        if name == "onnxruntime":
            return mock_ort
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        providers = get_onnx_providers()
        assert providers == [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]


def test_get_onnx_providers_rocm():
    """get_onnx_providers returns ROCM+CPU when ROCM available."""
    mock_ort = _mock_ort_import(
        ["ROCmExecutionProvider", "CPUExecutionProvider"],
    )

    def mock_import(name, *args, **kwargs):
        if name == "onnxruntime":
            return mock_ort
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        providers = get_onnx_providers()
        assert providers == [
            "ROCmExecutionProvider",
            "CPUExecutionProvider",
        ]


def test_get_onnx_providers_priority():
    """get_onnx_providers prefers ROCM > CUDA > CPU."""
    mock_ort = _mock_ort_import(
        [
            "ROCmExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    def mock_import(name, *args, **kwargs):
        if name == "onnxruntime":
            return mock_ort
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        providers = get_onnx_providers()
        assert providers == [
            "ROCmExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
