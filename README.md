# RedStarPlateOCR

License plate OCR engine with support for multiple region formats.

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.7+

### CPU

```bash
uv sync
```

### NVIDIA CUDA

```bash
uv add torch --index-url https://download.pytorch.org/whl/cu124
uv sync
```

### AMD ROCM

```bash
uv add torch --index-url https://download.pytorch.org/whl/rocm6.2
uv add onnxruntime-rocm  # optional, for GPU-backed ONNX inference
uv sync
```

### Apple Silicon (MPS)

PyTorch for macOS includes MPS support out of the box:

```bash
uv sync
```

No extra steps needed. MPS is auto-detected on Apple Silicon Macs.

## Quick Start

```bash
# Train a model
redstar-plate-ocr train \
    --config configs/model.yaml \
    --plate-config configs/plate.yaml \
    --data-dir /path/to/dataset

# Evaluate
redstar-plate-ocr evaluate \
    --checkpoint output/model.pt \
    --config configs/model.yaml \
    --plate-config configs/plate.yaml \
    --data-dir /path/to/dataset

# Predict a single image
redstar-plate-ocr predict \
    --checkpoint output/model.pt \
    --config configs/model.yaml \
    --plate-config configs/plate.yaml \
    --image /path/to/plate.jpg

# Export to ONNX
redstar-plate-ocr export \
    --checkpoint output/model.pt \
    --config configs/model.yaml \
    --plate-config configs/plate.yaml \
    --output model.onnx
```

## Hardware Support

### Auto-detection

RedStarPlateOCR automatically detects the best available device:

| Backend | Detection | Priority |
|---------|-----------|----------|
| NVIDIA CUDA | `torch.cuda.is_available()` | 1st |
| AMD ROCM | `torch.cuda.is_available()` + `torch.version.hip` | 1st (same as CUDA) |
| Apple MPS | `torch.backends.mps.is_available()` | 2nd |
| CPU | fallback | 3rd |

The detected backend is visible in logs: `backend=rocm`, `backend=cuda`, `backend=mps`, or `backend=cpu`.

### Device Override

You can override auto-detection with the `--device` CLI flag:

```bash
redstar-plate-ocr predict --device cpu ...
redstar-plate-ocr evaluate --device cuda ...
redstar-plate-ocr export --device mps ...
```

Or via the `REDSTAR_DEVICE` environment variable:

```bash
export REDSTAR_DEVICE=cpu   # force CPU
export REDSTAR_DEVICE=cuda   # force CUDA (also works on ROCM)
export REDSTAR_DEVICE=mps    # force MPS (macOS only)
```

If the requested device is not available, a warning is logged and auto-detection fallback is used.

### ROCM Notes

- ROCM uses the CUDA API surface internally — `torch.device("cuda")` works on AMD GPUs.
- Mixed precision (AMP) is fully supported via `torch.amp.GradScaler` and `torch.amp.autocast`.
- ONNX Runtime inference requires `onnxruntime-rocm` for GPU-backed execution. The engine auto-selects `ROCmExecutionProvider` when available.

### MPS Notes

- AMP is disabled on MPS (not supported by Apple's Metal Performance Shaders backend).
- CTC loss is automatically offloaded to CPU for MPS devices.
- ONNX export always runs on CPU (torch.onnx limitation).
- `pin_memory` is disabled in dataloaders for MPS.

## Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `REDSTAR_DEVICE` | `cuda`, `mps`, `cpu` | Override device auto-detection |

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Lint
uv run ruff check src/
uv run ruff format --check src/
```
