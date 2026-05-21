"""Tests for graceful interrupt handling in Trainer."""

from __future__ import annotations

import shutil
import signal
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from redstar_plate_ocr.nn.model import PlateOCRModel
from redstar_plate_ocr.pipeline.trainer import Trainer
from redstar_plate_ocr.plate.config import PlateConfig


def _make_trainer(plate_config: PlateConfig) -> Trainer:
    """Create a minimal Trainer instance for testing."""
    pc = plate_config
    model = PlateOCRModel(plate_config=pc)
    train_ds = MagicMock()
    val_ds = MagicMock()
    cfg = {
        "training": {
            "epochs": 2,
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


def test_handle_interrupt_sets_flag(plate_config: PlateConfig):
    """First SIGINT call sets _interrupt_requested flag."""
    trainer = _make_trainer(plate_config)
    assert trainer._interrupt_requested is False
    trainer._handle_interrupt(signal.SIGINT, None)
    assert trainer._interrupt_requested is True


def test_second_interrupt_raises(plate_config: PlateConfig):
    """Second SIGINT call raises KeyboardInterrupt."""
    trainer = _make_trainer(plate_config)
    trainer._handle_interrupt(signal.SIGINT, None)
    try:
        trainer._handle_interrupt(signal.SIGINT, None)
        raise AssertionError("Expected KeyboardInterrupt")
    except KeyboardInterrupt:
        pass


def test_interrupt_saves_checkpoint(plate_config: PlateConfig):
    """When interrupt flag is set, train saves interrupted ckpt."""
    trainer = _make_trainer(plate_config)
    tmp = Path(tempfile.mkdtemp())

    def set_interrupt(*args, **kwargs):
        trainer._interrupt_requested = True
        return {"loss": 1.0}

    with patch.object(
        trainer,
        "_save_interrupted_checkpoint",
    ) as mock_save:
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
                    "_train_epoch",
                    side_effect=set_interrupt,
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

    mock_save.assert_called_once()
    shutil.rmtree(tmp, ignore_errors=True)


def test_save_checkpoint_includes_scaler(plate_config: PlateConfig):
    """_save_checkpoint includes scaler_state_dict."""
    trainer = _make_trainer(plate_config)
    trainer.run_dir = trainer.output_dir

    with patch.object(torch, "save") as mock_save:
        trainer._save_checkpoint("test.pt", 0, 0.5)

    ckpt_dict = mock_save.call_args[0][0]
    assert "scaler_state_dict" in ckpt_dict


def test_resume_sets_start_epoch(plate_config: PlateConfig):
    """start_epoch defaults to 0 and can be set."""
    trainer = _make_trainer(plate_config)
    assert trainer.start_epoch == 0
    trainer.start_epoch = 5
    assert trainer.start_epoch == 5


def test_interrupt_flag_resets_on_train_start(plate_config: PlateConfig):
    """_interrupt_requested is reset at the start of train()."""
    trainer = _make_trainer(plate_config)
    trainer._interrupt_requested = True
    tmp = Path(tempfile.mkdtemp())

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
                                "redstar_plate_ocr.pipeline.trainer.ProgressDisplay",
                            ):
                                # Flag should be reset, so
                                # train should NOT save
                                # interrupted checkpoint
                                trainer.train()

    assert trainer._interrupt_requested is False
    shutil.rmtree(tmp, ignore_errors=True)


def test_save_interrupted_checkpoint_without_run_dir(
    plate_config: PlateConfig,
):
    """Fallback dir created when run_dir is None."""
    trainer = _make_trainer(plate_config)
    # run_dir is not set — simulates interrupt before train() creates it
    assert not hasattr(trainer, "run_dir") or trainer.run_dir is None

    with tempfile.TemporaryDirectory() as tmp:
        trainer.output_dir = Path(tmp)
        trainer._save_interrupted_checkpoint(0, -1.0)

        # Should have created fallback "interrupted" dir
        fallback = Path(tmp) / "interrupted"
        assert fallback.exists()
        ckpt_files = list(fallback.glob("interrupted_epoch*.pt"))
        assert len(ckpt_files) == 1


def test_save_interrupted_checkpoint_creates_parent(plate_config: PlateConfig):
    """_save_interrupted_checkpoint creates parent dirs if missing."""
    trainer = _make_trainer(plate_config)
    tmp = Path(tempfile.mkdtemp()) / "nested" / "run"
    trainer.run_dir = tmp
    # Parent dirs don't exist yet
    assert not tmp.exists()

    trainer._save_interrupted_checkpoint(3, 0.75)

    assert tmp.exists()
    ckpt_files = list(tmp.glob("interrupted_epoch*.pt"))
    assert len(ckpt_files) == 1
    shutil.rmtree(tmp.parent.parent, ignore_errors=True)


def test_save_interrupted_checkpoint_handles_save_error(
    plate_config: PlateConfig,
):
    """_save_interrupted_checkpoint logs error instead of crashing."""
    trainer = _make_trainer(plate_config)
    tmp = Path(tempfile.mkdtemp())
    trainer.run_dir = tmp

    with patch("torch.save", side_effect=OSError("disk full")):
        # Should NOT raise
        trainer._save_interrupted_checkpoint(0, 0.0)

    shutil.rmtree(tmp, ignore_errors=True)


def test_evaluator_interrupt_check_breaks_loop(plate_config: PlateConfig):
    """evaluate() breaks early when interrupt_check returns True."""
    from redstar_plate_ocr.pipeline.evaluator import Evaluator

    pc = plate_config
    evaluator = Evaluator(pc, torch.device("cpu"))
    model = PlateOCRModel(plate_config=pc)
    call_count = 0

    class _CountingLoader:
        """Dataloader stub that yields 3 batches."""

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
    # interrupt_check returns True after first batch
    checks = iter([False, True])
    result = evaluator.evaluate(
        model,
        loader,
        interrupt_check=lambda: next(checks),
    )
    # Should have processed only 2 batches (broke on 2nd check)
    assert call_count == 2
    assert "val_plate_accuracy" in result


def test_handle_interrupt_prints_rich_message(plate_config: PlateConfig):
    """_handle_interrupt prints Rich console message on first call."""
    trainer = _make_trainer(plate_config)
    with patch(
        "redstar_plate_ocr.pipeline.trainer.Console",
    ) as mock_console_cls:
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console
        trainer._handle_interrupt(signal.SIGINT, None)

    mock_console.print.assert_called_once()
    call_args = mock_console.print.call_args[0][0]
    assert "Interrupt requested" in call_args
