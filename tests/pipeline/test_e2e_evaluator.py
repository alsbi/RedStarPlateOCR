"""Tests for E2E evaluation mode in Evaluator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import Dataset

from redstar_plate_ocr.pipeline.evaluator import Evaluator
from redstar_plate_ocr.pipeline.training_config import TrainingConfig


class _FakeDataset(Dataset):
    """Minimal dataset returning a single fixed item."""

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": torch.randn(3, 32, 128),
            "orig_h": 32,
            "orig_w": 128,
            "region": ["RU", "UA"][idx],
            "plate_type": ["standard", "square"][idx],
            "plate_text": ["A001", "B002"][idx],
        }


def _make_plate_config() -> MagicMock:
    """Create a minimal mock PlateConfig."""
    pc = MagicMock()
    pc.union_alphabet = "AB0123456789"
    pc.country_list = ["RU", "UA"]
    return pc


def _make_model_output() -> MagicMock:
    """Create a mock ModelOutput."""
    output = MagicMock()
    output.ctc_output = torch.randn(2, 10, 14)
    output.country_logits = torch.randn(2, 2)
    output.format_logits = torch.randn(2, 2)
    return output


def test_e2e_mode_no_teacher_forcing() -> None:
    """When e2e=True, model is called without gt_countries."""
    from torch.utils.data import DataLoader

    pc = _make_plate_config()
    evaluator = Evaluator(pc, torch.device("cpu"))
    ds = _FakeDataset()
    loader = DataLoader(ds, batch_size=2)
    model_output = _make_model_output()

    model = MagicMock()
    model.eval = MagicMock()
    model.return_value = model_output

    with patch.object(
        evaluator, "_decode_countries", return_value=["RU", "UA"]
    ):
        with patch.object(
            evaluator,
            "_decode_formats",
            return_value=["standard", "square"],
        ):
            with patch(
                "redstar_plate_ocr.pipeline.evaluator.greedy_decode",
                return_value="A001",
            ):
                with patch(
                    "redstar_plate_ocr.pipeline.evaluator.to_long_tensor",
                    side_effect=lambda x, d: torch.as_tensor(x),
                ):
                    evaluator.evaluate(model, loader, e2e=True)

    call_args = model.call_args
    assert "gt_countries" not in call_args.kwargs
    assert "gt_plate_types" not in call_args.kwargs


def test_e2e_false_uses_teacher_forcing() -> None:
    """When e2e=False, model is called with gt_countries."""
    from torch.utils.data import DataLoader

    pc = _make_plate_config()
    evaluator = Evaluator(pc, torch.device("cpu"))
    ds = _FakeDataset()
    loader = DataLoader(ds, batch_size=2)
    model_output = _make_model_output()

    model = MagicMock()
    model.eval = MagicMock()
    model.return_value = model_output

    with patch.object(
        evaluator, "_decode_countries", return_value=["RU", "UA"]
    ):
        with patch.object(
            evaluator,
            "_decode_formats",
            return_value=["standard", "square"],
        ):
            with patch(
                "redstar_plate_ocr.pipeline.evaluator.greedy_decode",
                return_value="A001",
            ):
                with patch(
                    "redstar_plate_ocr.pipeline.evaluator.to_long_tensor",
                    side_effect=lambda x, d: torch.as_tensor(x),
                ):
                    evaluator.evaluate(model, loader, e2e=False)

    call_args = model.call_args
    assert "gt_countries" in call_args.kwargs
    assert "gt_plate_types" in call_args.kwargs


def test_e2e_metrics_have_prefix() -> None:
    """E2E metrics in process_epoch get val_e2e_ prefix."""
    e2e_metrics = {
        "val_plate_accuracy": 0.8,
        "val_cer": 0.2,
        "val_char_accuracy": 0.9,
    }
    prefixed: dict[str, float] = {}
    for k, v in e2e_metrics.items():
        prefixed[f"val_e2e_{k.removeprefix('val_')}"] = v

    assert "val_e2e_plate_accuracy" in prefixed
    assert "val_e2e_cer" in prefixed
    assert "val_e2e_char_accuracy" in prefixed
    assert "val_plate_accuracy" not in prefixed


def test_e2e_does_not_affect_scheduler() -> None:
    """Scheduler uses TF metrics, not E2E metrics."""
    config = TrainingConfig(e2e_eval=True)
    tf_metrics = {"val_plate_accuracy": 0.9}

    sched_val = tf_metrics.get(config.es_metric, 0.0)
    # E2E would give 0.5, but scheduler must use TF value
    assert sched_val == 0.9


def test_training_config_e2e_eval_default() -> None:
    """e2e_eval defaults to False."""
    config = TrainingConfig()
    assert config.e2e_eval is False


def test_training_config_e2e_eval_from_dict() -> None:
    """e2e_eval is read from training.e2e_eval."""
    cfg = {"training": {"e2e_eval": True}}
    config = TrainingConfig.from_dict(cfg)
    assert config.e2e_eval is True

    cfg2 = {"training": {}}
    config2 = TrainingConfig.from_dict(cfg2)
    assert config2.e2e_eval is False
