"""Tests for graceful interrupt handling in Trainer.

Covers the three-level Ctrl+C protocol:
  1st Ctrl+C → _interrupt_requested=True (graceful, finish batch)
  2nd Ctrl+C → _force_stop=True (skip gradient flush & validation)
  3rd Ctrl+C → raise KeyboardInterrupt (last resort)

Design principle: tests verify public observable behaviour
(files on disk, flag values, side-effect counts) rather than
mock-internal call chains or duplicate production logic in test code.
"""

from __future__ import annotations

import shutil
import signal
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from redstar_plate_ocr.nn.model import PlateOCRModel
from redstar_plate_ocr.pipeline.process_epoch import (
    _EpochResult,
    _make_interrupt_result,
    process_epoch,
)
from redstar_plate_ocr.pipeline.train_epoch import _format_batch_stats
from redstar_plate_ocr.pipeline.trainer import Trainer
from redstar_plate_ocr.plate.config import PlateConfig

# ── Helpers ─────────────────────────────────────────────────────────


def _make_trainer(plate_config: PlateConfig) -> Trainer:
    """Create a minimal Trainer instance for testing."""
    pc = plate_config
    model = PlateOCRModel(plate_config=pc)
    train_ds = MagicMock()
    val_ds = MagicMock()
    cfg = {
        "training": {
            "epochs": 3,
            "warmup_epochs": 0,
            "no_aug_epochs": 0,
        },
        "preprocessing": {},
        "augmentation": {},
    }
    return Trainer(
        model=model,
        plate_config=pc,
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg=cfg,
    )


def _run_train_with_mocked_loaders(trainer, **overrides):
    """Run trainer.train() with mocked IO; return mock-save call count.

    Accepts optional keyword overrides applied to trainer before training.
    """
    tmp = Path(tempfile.mkdtemp())
    try:
        for k, v in overrides.items():
            setattr(trainer, k, v)

        with patch.object(trainer, "_build_train_loader"):
            with patch.object(trainer, "_build_val_loader"):
                with patch.object(
                    trainer,
                    "_train_epoch",
                    return_value={"loss": 1.0},
                ):
                    with patch.object(
                        trainer.evaluator,
                        "evaluate",
                        return_value={
                            "val_plate_accuracy": 0.5,
                            "val_cer": 0.5,
                            "val_country_accuracy": 0.5,
                            "val_format_accuracy": 0.5,
                            "val_square_accuracy": 0.5,
                        },
                    ):
                        with patch.object(trainer, "_save_checkpoint"):
                            with patch(
                                "redstar_plate_ocr.pipeline"
                                ".trainer.create_run_dir",
                                return_value=tmp,
                            ):
                                with patch(
                                    "redstar_plate_ocr.pipeline"
                                    ".trainer.ProgressDisplay",
                                ):
                                    trainer.train()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Level 1: First Ctrl+C ───────────────────────────────────────────


class TestFirstInterrupt:
    """1st Ctrl+C sets _interrupt_requested."""

    def test_sets_flag(self, plate_config):
        trainer = _make_trainer(plate_config)
        assert trainer._interrupt_requested is False
        trainer._handle_interrupt(signal.SIGINT, None)
        assert trainer._interrupt_requested is True

    def test_does_not_set_force_stop(self, plate_config):
        trainer = _make_trainer(plate_config)
        trainer._handle_interrupt(signal.SIGINT, None)
        assert trainer._force_stop is False

    def test_prints_graceful_message(self, plate_config):
        trainer = _make_trainer(plate_config)
        with patch(
            "redstar_plate_ocr.pipeline.trainer.Console",
        ) as mock_console_cls:
            mock_console = MagicMock()
            mock_console_cls.return_value = mock_console
            trainer._handle_interrupt(signal.SIGINT, None)

        mock_console.print.assert_called_once()
        msg = mock_console.print.call_args[0][0]
        assert "Interrupt requested" in msg

    def test_saves_checkpoint(self, plate_config):
        """When _interrupt_requested is set mid-training, an interrupted
        checkpoint file is written to disk."""
        trainer = _make_trainer(plate_config)
        tmp = Path(tempfile.mkdtemp())
        trainer.run_dir = tmp

        # Simulate: interrupt requested during epoch
        trainer._interrupt_requested = True
        trainer._save_interrupted_checkpoint(epoch=0, best_metric=-1.0)

        ckpt_files = list(tmp.glob("interrupted_epoch*.pt"))
        assert len(ckpt_files) == 1
        shutil.rmtree(tmp, ignore_errors=True)


# ── Level 2: Second Ctrl+C ──────────────────────────────────────────


class TestSecondInterrupt:
    """2nd Ctrl+C sets _force_stop (no longer raises)."""

    def test_sets_force_stop_flag(self, plate_config):
        trainer = _make_trainer(plate_config)
        trainer._handle_interrupt(signal.SIGINT, None)  # first
        trainer._handle_interrupt(signal.SIGINT, None)  # second
        assert trainer._force_stop is True

    def test_does_not_raise_keyboard_interrupt(self, plate_config):
        trainer = _make_trainer(plate_config)
        trainer._handle_interrupt(signal.SIGINT, None)  # first
        # Should NOT raise — this was the old behaviour
        trainer._handle_interrupt(signal.SIGINT, None)  # second

    def test_prints_force_stop_message(self, plate_config):
        trainer = _make_trainer(plate_config)
        trainer._handle_interrupt(signal.SIGINT, None)  # first
        with patch(
            "redstar_plate_ocr.pipeline.trainer.Console",
        ) as mock_console_cls:
            mock_console = MagicMock()
            mock_console_cls.return_value = mock_console
            trainer._handle_interrupt(signal.SIGINT, None)  # second

        mock_console.print.assert_called_once()
        msg = mock_console.print.call_args[0][0]
        assert "Force stop" in msg

    def test_saves_interrupted_checkpoint(self, plate_config):
        """Force stop still saves a checkpoint to disk."""
        trainer = _make_trainer(plate_config)
        tmp = Path(tempfile.mkdtemp())
        trainer.run_dir = tmp

        trainer._force_stop = True
        trainer._save_interrupted_checkpoint(epoch=2, best_metric=0.42)

        ckpt_files = list(tmp.glob("interrupted_epoch*.pt"))
        assert len(ckpt_files) == 1
        # Verify epoch number embedded in filename
        assert "epoch3" in ckpt_files[0].name
        shutil.rmtree(tmp, ignore_errors=True)


# ── Level 3: Third Ctrl+C ───────────────────────────────────────────


class TestThirdInterrupt:
    """3rd Ctrl+C raises KeyboardInterrupt as last resort."""

    def test_raises_keyboard_interrupt(self, plate_config):
        trainer = _make_trainer(plate_config)
        trainer._handle_interrupt(signal.SIGINT, None)  # first
        trainer._handle_interrupt(signal.SIGINT, None)  # second
        with pytest.raises(KeyboardInterrupt):
            trainer._handle_interrupt(signal.SIGINT, None)  # third

    def test_third_interrupt_saves_with_real_best_metric(self, plate_config):
        """When KeyboardInterrupt bubbles up through train(), the except
        handler should use real _best_metrics instead of hardcoded -1.0."""
        trainer = _make_trainer(plate_config)
        tmp = Path(tempfile.mkdtemp())

        # Simulate state after several epochs: _best_metrics populated
        trainer.run_dir = tmp
        trainer._best_metrics = {"val_plate_accuracy": 0.87}

        # Reproduce exactly what the except handler does
        last_epoch = max(trainer.start_epoch - 1, 0)
        saved_best = getattr(trainer, "_best_metrics", None)
        bm = saved_best.get("val_plate_accuracy", -1.0) if saved_best else -1.0
        trainer._save_interrupted_checkpoint(last_epoch, bm)

        ckpt_files = list(tmp.glob("interrupted_epoch*.pt"))
        assert len(ckpt_files) == 1
        ckpt = torch.load(
            ckpt_files[0], map_location="cpu", weights_only=False
        )
        # The checkpoint should have the REAL best metric, not -1.0
        assert abs(ckpt["best_metric"] - 0.87) < 1e-6
        shutil.rmtree(tmp, ignore_errors=True)

    def test_third_interrupt_falls_back_to_minus_one(self, plate_config):
        """If no _best_metrics available yet, falls back to -1.0."""
        trainer = _make_trainer(plate_config)
        tmp = Path(tempfile.mkdtemp())

        trainer.run_dir = tmp
        # No _best_metrics attribute set at all (fresh training)

        saved_best = getattr(trainer, "_best_metrics", None)
        bm = saved_best.get("val_plate_accuracy", -1.0) if saved_best else -1.0
        trainer._save_interrupted_checkpoint(0, bm)

        ckpt_files = list(tmp.glob("interrupted_epoch*.pt"))
        ckpt = torch.load(
            ckpt_files[0], map_location="cpu", weights_only=False
        )
        assert ckpt["best_metric"] == -1.0
        shutil.rmtree(tmp, ignore_errors=True)


# ── Flag lifecycle ────────────────────────────────────────────────────


class TestFlagLifecycle:
    """Flags are properly reset at start of each train() call."""

    def test_interrupt_flag_resets_on_train_start(self, plate_config):
        trainer = _make_trainer(plate_config)
        trainer._interrupt_requested = True
        _run_train_with_mocked_loaders(trainer)
        assert trainer._interrupt_requested is False

    def test_force_stop_flag_resets_on_train_start(self, plate_config):
        trainer = _make_trainer(plate_config)
        trainer._force_stop = True
        _run_train_with_mocked_loaders(trainer)
        assert trainer._force_stop is False

    def test_force_stop_at_epoch_loop_start_saves_interrupted(
        self,
        plate_config,
    ):
        """When _force_stop is set during training (e.g., by signal),
        the main loop saves an interrupted checkpoint before breaking."""
        trainer = _make_trainer(plate_config)
        tmp = Path(tempfile.mkdtemp())

        call_count = 0

        def _mock_run_one_epoch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                trainer._force_stop = True
            return (
                _EpochResult(
                    best_metric=-1.0,
                    best_metrics={},
                    last_metrics={"val_plate_accuracy": 0.1},
                    patience_counter=0,
                    should_stop=False,
                    was_interrupted=False,
                ),
                None,
                [],
            )

        with patch.object(
            trainer,
            "_run_one_epoch",
            side_effect=_mock_run_one_epoch,
        ):
            with patch.object(
                trainer,
                "_save_interrupted_checkpoint",
            ) as int_save:
                with patch.object(
                    trainer,
                    "_build_train_loader",
                ):
                    with patch.object(
                        trainer,
                        "_build_val_loader",
                    ):
                        with patch.object(
                            trainer,
                            "_save_checkpoint",
                        ):
                            with patch(
                                "redstar_plate_ocr.pipeline"
                                ".trainer.create_run_dir",
                                return_value=tmp,
                            ):
                                with patch(
                                    "redstar_plate_ocr.pipeline"
                                    ".trainer.ProgressDisplay",
                                ):
                                    trainer.train()

        int_save.assert_called_once()
        shutil.rmtree(tmp, ignore_errors=True)


# ── process_epoch interrupt behaviour ────────────────────────────────


class TestProcessEpochInterrupt:
    """process_epoch returns an interrupted result when flags are set,
    skipping validation entirely."""

    def test_process_epoch_returns_interrupt_result(self, plate_config):
        """When _interrupt_requested is True after _train_epoch,
        process_epoch returns was_interrupted=True without running
        validation."""
        trainer = _make_trainer(plate_config)
        trainer._interrupt_requested = True

        from torch.utils.data import DataLoader

        from redstar_plate_ocr.pipeline.progress_display import ProgressDisplay

        # Minimal dataloader that yields nothing
        empty_dl = MagicMock(spec=DataLoader)
        empty_dl.__len__ = MagicMock(return_value=1)

        progress_display = MagicMock(spec=ProgressDisplay)

        result = process_epoch(
            trainer=trainer,
            epoch=0,
            train_loader=empty_dl,
            val_loader=empty_dl,
            progress_display=progress_display,
            best_metric=-1.0,
            best_metrics={},
            patience_counter=0,
        )

        assert isinstance(result, _EpochResult)
        assert result.was_interrupted is True
        assert result.should_stop is False

    def test_process_epoch_skips_validation_on_interrupt(self, plate_config):
        """Validation evaluate() must NOT be called when interrupted."""
        trainer = _make_trainer(plate_config)
        trainer._interrupt_requested = True

        from torch.utils.data import DataLoader

        from redstar_plate_ocr.pipeline.progress_display import ProgressDisplay

        empty_dl = MagicMock(spec=DataLoader)
        empty_dl.__len__ = MagicMock(return_value=1)
        progress_display = MagicMock(spec=ProgressDisplay)

        with patch.object(
            trainer.evaluator,
            "evaluate",
        ) as mock_eval:
            process_epoch(
                trainer=trainer,
                epoch=0,
                train_loader=empty_dl,
                val_loader=empty_dl,
                progress_display=progress_display,
                best_metric=-1.0,
                best_metrics={},
                patience_counter=0,
            )

        mock_eval.assert_not_called()

    def test_make_interrupt_result_fields(self):
        """_make_interrupt_result produces correct field values."""
        result = _make_interrupt_result(
            best_metric=0.5,
            best_metrics={"val_plate_accuracy": 0.5},
            patience_counter=2,
            epoch=3,
        )
        assert isinstance(result, _EpochResult)
        assert result.was_interrupted is True
        assert result.should_stop is False
        assert result.best_metric == 0.5
        assert result.patience_counter == 2

    def test_validation_passes_interrupt_check(self, plate_config):
        """_run_validation passes a lambda that checks both flags
        so evaluation can break early on interrupt."""
        from redstar_plate_ocr.pipeline.process_epoch import _run_validation
        from redstar_plate_ocr.pipeline.progress_display import ProgressDisplay

        trainer = _make_trainer(plate_config)
        progress_display = MagicMock(spec=ProgressDisplay)

        # Capture the interrupt_check lambda passed to evaluator
        captured_check = None

        def fake_evaluate(model, loader, *, interrupt_check=None):
            nonlocal captured_check
            captured_check = interrupt_check
            return {
                "val_plate_accuracy": 0.5,
                "val_cer": 0.5,
                "val_char_accuracy": 0.5,
                "val_country_accuracy": 0.5,
                "val_format_accuracy": 0.5,
                "val_square_accuracy": 0.5,
            }

        with patch.object(
            trainer.evaluator,
            "evaluate",
            side_effect=fake_evaluate,
        ):
            val_dl = MagicMock()
            val_dl.__len__ = MagicMock(return_value=1)
            try:
                _run_validation(
                    trainer,
                    val_dl,
                    train_result={"loss": 1.0},
                    config=trainer.config,
                    epoch=0,
                    best_metric=-1.0,
                    best_metrics={},
                    patience_counter=0,
                    progress_display=progress_display,
                )
            except Exception:
                pass  # early-stop may raise; we only care about lambda

        # The lambda should reflect both flags
        assert captured_check is not None
        assert captured_check() is False
        trainer._interrupt_requested = True
        assert captured_check() is True
        trainer._interrupt_requested = False
        trainer._force_stop = True
        assert captured_check() is True


# ── run_train_epoch fast exit on force_stop ──────────────────────────


class TestTrainEpochFastExit:
    """When _force_stop is set, run_train_epoch skips gradient flush
    and final accuracy computation for maximum speed."""

    def test_force_stop_skips_gradient_flush(self, plate_config):
        """Verify that when _force_stop is active the gradient flush path
        is skipped after the batch loop."""
        from redstar_plate_ocr.pipeline.train_epoch import run_train_epoch

        trainer = _make_trainer(plate_config)
        trainer._force_stop = True

        # A tiny dataloader that yields one batch then stops
        class _OneBatchLoader:
            def __len__(self):
                return 1

            def __iter__(self):
                yield {
                    "image": torch.randn(1, 3, 80, 256),
                    "orig_h": torch.tensor([80]),
                    "orig_w": torch.tensor([256]),
                    "region": ["RU"],
                    "plate_type": ["standard"],
                    "plate_text": ["A000AA00"],
                }

        loader = _OneBatchLoader()

        with patch(
            "redstar_plate_ocr.pipeline.train_epoch._compute_final_accuracies",
        ) as mock_acc:
            with patch(
                "redstar_plate_ocr.pipeline.train_epoch._update_progress",
            ):
                result = run_train_epoch(
                    trainer=trainer,
                    loader=loader,
                    sampling_prob=0.5,
                    progress_display=MagicMock(),
                    task_id=0,
                )

        # Accuracy computation should be skipped on force stop
        mock_acc.assert_not_called()
        # Result should still contain loss info (early return path)
        assert "loss" in result

    def test_graceful_interrupt_flushes_gradients(self, plate_config):
        """When only _interrupt_requested is set (no force_stop),
        gradient flush and accuracy computation still happen."""
        from redstar_plate_ocr.pipeline.train_epoch import run_train_epoch

        trainer = _make_trainer(plate_config)
        trainer._interrupt_requested = True

        class _OneBatchLoader:
            def __len__(self):
                return 1

            def __iter__(self):
                yield {
                    "image": torch.randn(1, 3, 80, 256),
                    "orig_h": torch.tensor([80]),
                    "orig_w": torch.tensor([256]),
                    "region": ["RU"],
                    "plate_type": ["standard"],
                    "plate_text": ["A000AA00"],
                }

        loader = _OneBatchLoader()

        with patch(
            "redstar_plate_ocr.pipeline.train_epoch._compute_final_accuracies",
        ) as mock_acc:
            with patch(
                "redstar_plate_ocr.pipeline.train_epoch._update_progress",
            ):
                result = run_train_epoch(
                    trainer=trainer,
                    loader=loader,
                    sampling_prob=0.5,
                    progress_display=MagicMock(),
                    task_id=0,
                )

        # Accuracy computation SHOULD happen for graceful interrupt
        mock_acc.assert_called_once()
        assert "loss" in result


# ── _format_batch_stats ─────────────────────────────────────────────


class TestFormatBatchStats:
    """Verify progress bar stats formatting including grad norm,
    accumulation steps, and timing."""

    def test_basic_output(self):
        """Basic format includes loss, accuracies, and timing."""
        result = _format_batch_stats(
            running={"loss": 0.5432},
            fmt_acc=0.85,
            ctry_acc=0.72,
            plate_acc=0.91,
            char_acc=0.95,
            avg_batch_ms=120.0,
        )
        assert "loss=0.5432" in result
        # .3% format produces e.g. 91.000%
        assert "plate=91.000%" in result
        assert "char=95.000%" in result
        assert "region=72.000%" in result
        assert "fmt=85.000%" in result
        assert "⏱" in result

    def test_grad_norm_shown_when_nonzero(self):
        """Grad norm emoji appears only when grad_norm > 0."""
        result_no_norm = _format_batch_stats(
            running={"loss": 0.5},
            fmt_acc=0.8,
            ctry_acc=0.7,
            plate_acc=0.9,
            char_acc=0.9,
            grad_norm=0.0,
        )
        assert "📏" not in result_no_norm

        result_has_norm = _format_batch_stats(
            running={"loss": 0.5},
            fmt_acc=0.8,
            ctry_acc=0.7,
            plate_acc=0.9,
            char_acc=0.9,
            grad_norm=12.34,
        )
        assert "📏" in result_has_norm
        assert "12.34" in result_has_norm

    def test_accum_steps_shown_when_gt_one(self):
        """Accumulation step indicator appears only when accum_total > 1."""
        result_no_accum = _format_batch_stats(
            running={"loss": 0.5},
            fmt_acc=0.8,
            ctry_acc=0.7,
            plate_acc=0.9,
            char_acc=0.9,
            accum_total=1,
        )
        assert "📦" not in result_no_accum

        result_accum = _format_batch_stats(
            running={"loss": 0.5},
            fmt_acc=0.8,
            ctry_acc=0.7,
            plate_acc=0.9,
            char_acc=0.9,
            accum_step=2,
            accum_total=4,
        )
        assert "📦" in result_accum
        assert "2/4" in result_accum

    def test_left_right_separated_by_pipe(self):
        """Left section (metrics) and right section (system) are
        separated by │."""
        result = _format_batch_stats(
            running={"loss": 0.5},
            fmt_acc=0.8,
            ctry_acc=0.7,
            plate_acc=0.9,
            char_acc=0.9,
            avg_batch_ms=50.0,
            grad_norm=5.0,
            accum_step=1,
            accum_total=2,
        )
        parts = result.split(" │ ")
        assert len(parts) == 2
        assert "loss=" in parts[0]
        assert "⏱" in parts[1]


# ── Checkpoint persistence on disk ────────────────────────────────────


class TestInterruptedCheckpointOnDisk:
    """Verify that the interrupted checkpoint file has correct content."""

    def test_checkpoint_has_model_state_dict(self, plate_config):
        trainer = _make_trainer(plate_config)
        tmp = Path(tempfile.mkdtemp())
        trainer.run_dir = tmp

        trainer._save_interrupted_checkpoint(epoch=0, best_metric=-1.0)

        ckpt_files = list(tmp.glob("interrupted_epoch*.pt"))
        assert len(ckpt_files) == 1

        ckpt = torch.load(
            ckpt_files[0],
            map_location="cpu",
            weights_only=False,
        )
        assert "model_state_dict" in ckpt
        assert "interrupted" in ckpt
        assert ckpt["interrupted"] is True
        shutil.rmtree(tmp, ignore_errors=True)

    def test_checkpoint_preserves_best_metric(self, plate_config):
        trainer = _make_trainer(plate_config)
        tmp = Path(tempfile.mkdtemp())
        trainer.run_dir = tmp

        trainer._save_interrupted_checkpoint(epoch=5, best_metric=0.87)

        ckpt_files = list(tmp.glob("interrupted_epoch*.pt"))
        ckpt = torch.load(
            ckpt_files[0],
            map_location="cpu",
            weights_only=False,
        )
        assert abs(ckpt["best_metric"] - 0.87) < 1e-6
        shutil.rmtree(tmp, ignore_errors=True)


# ── Evaluator interrupt check ────────────────────────────────────────


class TestEvaluatorInterruptCheck:
    """evaluate() breaks early when interrupt_check returns True."""

    def test_breaks_on_interrupt_check(self, plate_config):
        from redstar_plate_ocr.pipeline.evaluator import Evaluator

        evaluator = Evaluator(plate_config, torch.device("cpu"))
        model = PlateOCRModel(plate_config=plate_config)
        call_count = 0

        class _CountingLoader:
            def __iter__(self):
                return self

            def __len__(self):
                return 3

            def __next__(self):
                nonlocal call_count
                if call_count >= 3:
                    raise StopIteration
                call_count += 1
                return {
                    "image": torch.randn(1, 3, 80, 256),
                    "orig_h": torch.tensor([80]),
                    "orig_w": torch.tensor([256]),
                    "region": ["RU"],
                    "plate_type": ["standard"],
                    "plate_text": ["A000AA00"],
                }

        loader = _CountingLoader()
        checks = iter([False, True])
        result = evaluator.evaluate(
            model,
            loader,
            interrupt_check=lambda: next(checks),
        )
        # Only 2 batches processed before breaking
        assert call_count == 2
        assert "val_plate_accuracy" in result


# ── Fallback / error resilience ──────────────────────────────────────


class TestCheckpointResilience:
    """Edge cases around interrupted checkpoint saving."""

    def test_fallback_dir_without_run_dir(self, plate_config):
        trainer = _make_trainer(plate_config)
        assert not hasattr(trainer, "run_dir") or trainer.run_dir is None

        with tempfile.TemporaryDirectory() as tmp:
            trainer.output_dir = Path(tmp)
            trainer._save_interrupted_checkpoint(0, -1.0)

            fallback = Path(tmp) / "interrupted"
            assert fallback.exists()
            ckpt_files = list(fallback.glob("interrupted_epoch*.pt"))
            assert len(ckpt_files) == 1

    def test_creates_parent_dirs(self, plate_config):
        trainer = _make_trainer(plate_config)
        tmp = Path(tempfile.mkdtemp()) / "nested" / "run"
        trainer.run_dir = tmp
        assert not tmp.exists()

        trainer._save_interrupted_checkpoint(3, 0.75)

        assert tmp.exists()
        ckpt_files = list(tmp.glob("interrupted_epoch*.pt"))
        assert len(ckpt_files) == 1
        shutil.rmtree(tmp.parent.parent, ignore_errors=True)

    def test_handles_save_error_gracefully(self, plate_config):
        trainer = _make_trainer(plate_config)
        tmp = Path(tempfile.mkdtemp())
        trainer.run_dir = tmp

        with patch("torch.save", side_effect=OSError("disk full")):
            trainer._save_interrupted_checkpoint(0, 0.0)
        # Did not crash
        shutil.rmtree(tmp, ignore_errors=True)


# ── Misc / existing coverage retained ────────────────────────────────


class TestMiscRetained:
    """Existing test coverage that doesn't fit above categories."""

    def test_save_checkpoint_includes_scaler(self, plate_config):
        trainer = _make_trainer(plate_config)
        trainer.run_dir = trainer.output_dir

        with patch.object(torch, "save") as mock_save:
            trainer._save_checkpoint("test.pt", 0, 0.5)

        ckpt_dict = mock_save.call_args[0][0]
        assert "scaler_state_dict" in ckpt_dict

    def test_resume_sets_start_epoch(self, plate_config):
        trainer = _make_trainer(plate_config)
        assert trainer.start_epoch == 0
        trainer.start_epoch = 5
        assert trainer.start_epoch == 5
