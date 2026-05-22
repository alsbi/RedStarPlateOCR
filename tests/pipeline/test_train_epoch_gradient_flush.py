"""Tests for gradient force-flush behaviour in _optimizer_step.

Validates critical scenarios:
1. When gradient_accumulation_steps > number_of_batches,
   optimizer.step() IS called via the force flush at epoch end.
2. When gradient_accumulation_steps == number_of_batches,
   the natural accumulation step fires and the subsequent
   force flush is SKIPPED (no phantom step on zero grads).
3. When batches % accum != 0, remaining gradients are flushed.
4. When batches % accum == 0, no redundant flush occurs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from redstar_plate_ocr.pipeline.train_epoch import _optimizer_step
from redstar_plate_ocr.pipeline.training_config import TrainingConfig


def _make_trainer_mock(
    gradient_accumulation_steps: int,
) -> MagicMock:
    """Create minimal mock Trainer for _optimizer_step."""
    config = TrainingConfig(
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    # Real parameter tensors so clip_grad_norm_ has something to iterate
    param = torch.nn.Parameter(torch.randn(2, 2))

    main_opt = MagicMock()
    main_opt.param_groups = [{"params": [param]}]

    ctry_opt = MagicMock()
    ctry_opt.param_groups = [{"params": [param]}]

    trainer = MagicMock(
        spec=[
            "config",
            "use_amp",
            "optimizer",
            "country_optimizer",
            "scaler",
        ]
    )
    trainer.config = config
    trainer.use_amp = False
    trainer.optimizer = main_opt
    trainer.country_optimizer = ctry_opt

    return trainer


@patch(
    "redstar_plate_ocr.pipeline.train_epoch.torch.nn.utils.clip_grad_norm_",
    return_value=torch.tensor(0.0),
)
class TestGradientFlushAccumExceedsBatches:
    """When gradient_accumulation_steps > num_batches, force flush fires."""

    def test_no_step_during_loop_when_accum_exceeds_batches(
        self,
        mock_clip: MagicMock,
    ) -> None:
        """No optimizer.step() during loop if accum > batches."""
        trainer = _make_trainer_mock(gradient_accumulation_steps=10)

        step = 0
        for _ in range(3):
            step, did_step, _ = _optimizer_step(trainer, step)
            assert not did_step

        assert trainer.optimizer.step.call_count == 0
        assert trainer.country_optimizer.step.call_count == 0

    def test_force_flush_calls_optimizer_step(
        self,
        mock_clip: MagicMock,
    ) -> None:
        """Force flush at epoch end calls optimizer.step() exactly once."""
        trainer = _make_trainer_mock(gradient_accumulation_steps=10)
        num_batches = 3

        step = 0
        for _ in range(num_batches):
            step, _, _ = _optimizer_step(trainer, step)

        step, did_step, grad_norm = _optimizer_step(trainer, step, force=True)

        assert did_step
        assert trainer.optimizer.step.call_count == 1
        assert trainer.country_optimizer.step.call_count == 1

    def test_force_flush_returns_nonzero_step_counter(
        self,
        mock_clip: MagicMock,
    ) -> None:
        """Force flush increments step counter past num_batches."""
        trainer = _make_trainer_mock(gradient_accumulation_steps=10)

        step = 0
        for _ in range(3):
            step, _, _ = _optimizer_step(trainer, step)

        new_step, did_step, _ = _optimizer_step(trainer, step, force=True)

        assert did_step
        assert new_step == 4  # 3 loop steps + 1 force flush

    def test_zero_grad_called_after_force_flush(
        self,
        mock_clip: MagicMock,
    ) -> None:
        """After force flush, zero_grad() is called on both optimizers."""
        trainer = _make_trainer_mock(gradient_accumulation_steps=10)

        step = 0
        for _ in range(3):
            step, _, _ = _optimizer_step(trainer, step)

        _optimizer_step(trainer, step, force=True)

        trainer.optimizer.zero_grad.assert_called_once()
        trainer.country_optimizer.zero_grad.assert_called_once()


@patch(
    "redstar_plate_ocr.pipeline.train_epoch.torch.nn.utils.clip_grad_norm_",
    return_value=torch.tensor(0.0),
)
class TestGradientFlushNoPhantomStep:
    """When accum divides batches exactly, force flush must NOT fire.

    This prevents phantom optimizer steps on zero/None gradients that
    would corrupt Adam momentum and inflate global_step / LR schedule.
    """

    def test_no_force_flush_when_exact_alignment(
        self,
        mock_clip: MagicMock,
    ) -> None:
        """accum=4, 4 batches → natural step on batch 3; force does nothing."""
        trainer = _make_trainer_mock(gradient_accumulation_steps=4)

        step = 0
        did_step_flags: list[bool] = []
        for _ in range(4):
            step, did_step, _ = _optimizer_step(trainer, step)
            did_step_flags.append(did_step)

        # Natural accumulation step fired once at batch index 3
        assert did_step_flags == [False, False, False, True]
        assert trainer.optimizer.step.call_count == 1

        # Force flush should NOT add another step — all grads already consumed
        step, did_step, _ = _optimizer_step(trainer, step, force=True)
        assert not did_step
        assert trainer.optimizer.step.call_count == 1  # unchanged

    def test_no_force_flush_when_accum_divides_batches_evenly(
        self,
        mock_clip: MagicMock,
    ) -> None:
        """accum=2, 6 batches → 3 natural steps; force adds nothing."""
        trainer = _make_trainer_mock(gradient_accumulation_steps=2)

        step = 0
        for _ in range(6):
            step, _, _ = _optimizer_step(trainer, step)

        natural_calls = trainer.optimizer.step.call_count
        assert natural_calls == 3  # at batch indices 1, 3, 5

        step, did_step, _ = _optimizer_step(trainer, step, force=True)
        assert not did_step
        assert trainer.optimizer.step.call_count == 3  # unchanged


@patch(
    "redstar_plate_ocr.pipeline.train_epoch.torch.nn.utils.clip_grad_norm_",
    return_value=torch.tensor(0.0),
)
class TestGradientFlushPartialRemainder:
    """When batches don't divide evenly by accum, remainder gets flushed."""

    def test_remainder_flushed_with_force(
        self,
        mock_clip: MagicMock,
    ) -> None:
        """accum=2, 5 batches → 2 natural + 1 force flush for last batch."""
        trainer = _make_trainer_mock(gradient_accumulation_steps=2)

        step = 0
        for _ in range(5):
            step, _, _ = _optimizer_step(trainer, step)

        natural_calls = trainer.optimizer.step.call_count
        assert natural_calls == 2  # at batch indices 1, 3

        step, did_step, _ = _optimizer_step(trainer, step, force=True)
        assert did_step
        assert trainer.optimizer.step.call_count == 3  # 2 + 1 flush


@patch(
    "redstar_plate_ocr.pipeline.train_epoch.torch.nn.utils.clip_grad_norm_",
    return_value=torch.tensor(0.0),
)
class TestGradientNormReturned:
    """Verify grad_norm is returned correctly from _optimizer_step."""

    def test_force_flush_returns_grad_norm(
        self,
        mock_clip: MagicMock,
    ) -> None:
        """Force flush returns computed combined gradient norm."""
        trainer = _make_trainer_mock(gradient_accumulation_steps=10)

        step = 0
        for _ in range(3):
            step, _, _ = _optimizer_step(trainer, step)

        _, _, grad_norm = _optimizer_step(trainer, step, force=True)

        # Mocked clip_grad_norm_ returns 0.0 for both optimizers;
        # combined norm = sqrt(0^2 + 0^2) = 0.0
        assert isinstance(grad_norm, float)
        assert grad_norm >= 0.0

    def test_no_step_returns_zero_grad_norm(
        self,
        mock_clip: MagicMock,
    ) -> None:
        """When no step is taken, grad_norm returns 0.0."""
        trainer = _make_trainer_mock(gradient_accumulation_steps=10)

        step, did_step, grad_norm = _optimizer_step(trainer, 0)
        assert not did_step
        assert grad_norm == 0.0
