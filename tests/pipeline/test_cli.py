"""Tests for CLI commands."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from redstar_plate_ocr.presentation.cli import (
    _validate_dataset,
    app,
)
from redstar_plate_ocr.presentation.display import (
    _print_startup_panel,
)
from redstar_plate_ocr.presentation.logging import setup_logging

runner = CliRunner()


def test_cli_info_no_crash() -> None:
    """info command with valid plate-config does not crash."""
    result = runner.invoke(
        app,
        ["info", "--plate-config", "configs/plate.yaml"],
    )
    assert result.exit_code == 0


def test_cli_predict_no_checkpoint() -> None:
    """predict command fails gracefully without checkpoint."""
    result = runner.invoke(
        app,
        [
            "predict",
            "--checkpoint",
            "/nonexistent/best.pt",
            "--plate-config",
            "configs/plate.yaml",
            "--image",
            "nonexistent.jpg",
        ],
    )
    assert result.exit_code != 0


def test_cli_validate_ok() -> None:
    """validate command on clean dataset returns no errors."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "val.csv"
        csv_path.write_text(
            "image_path,plate_text,region,plate_type\n"
            "plates/A000AA00.png,A000AA00,RU,standard\n",
        )
        # Create the image so validation passes
        img_dir = Path(tmp) / "plates"
        img_dir.mkdir()
        img_path = img_dir / "A000AA00.png"
        import cv2
        import numpy as np

        img = np.zeros((32, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        result = runner.invoke(
            app,
            [
                "validate",
                "--plate-config",
                "configs/plate.yaml",
                "--data-dir",
                tmp,
                "--split",
                "val",
            ],
        )
        assert result.exit_code == 0


def test_cli_verbose_flag() -> None:
    """-v and -vv change logging level."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Default: WARNING
        setup_logging(output_dir=tmp_path, verbose=0)
        root = logging.getLogger()
        console_handler = root.handlers[0]
        assert console_handler.level == logging.WARNING

        # -v: INFO
        setup_logging(output_dir=tmp_path, verbose=1)
        root = logging.getLogger()
        console_handler = root.handlers[0]
        assert console_handler.level == logging.INFO

        # -vv: DEBUG
        setup_logging(output_dir=tmp_path, verbose=2)
        root = logging.getLogger()
        console_handler = root.handlers[0]
        assert console_handler.level == logging.DEBUG

        # File handler always DEBUG
        file_handler = root.handlers[1]
        assert file_handler.level == logging.DEBUG


def test_cli_train_creates_output_dir() -> None:
    """train command with 1 epoch creates output directory."""
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = os.path.join(tmp, "outputs")

        # Patch Trainer.train to run just 1 step
        with patch(
            "redstar_plate_ocr.pipeline.trainer.Trainer.train",
        ) as mock_train:
            mock_train.return_value = {
                "best": {"val_plate_accuracy": 0.5},
                "last": {},
            }
            result = runner.invoke(
                app,
                [
                    "train",
                    "--config",
                    "configs/model.yaml",
                    "--plate-config",
                    "configs/plate.yaml",
                    "--data-dir",
                    "data/",
                    "--output-dir",
                    out_dir,
                ],
            )

        # Should not crash
        assert result.exit_code == 0


def test_validate_dataset_missing_csv() -> None:
    """validate returns errors for missing CSV."""
    with tempfile.TemporaryDirectory() as tmp:
        errors, counts = _validate_dataset(
            "configs/plate.yaml",
            tmp,
            "nonexistent",
        )
        assert len(errors) > 0
        assert "CSV not found" in errors[0]


def test_validate_dataset_bad_region() -> None:
    """validate returns errors for unknown region."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "train.csv"
        csv_path.write_text(
            "image_path,plate_text,region,plate_type\n"
            "img.png,A123AA12,INVALID,standard\n",
        )
        errors, counts = _validate_dataset(
            "configs/plate.yaml",
            tmp,
            "train",
        )
        assert any("unknown region" in e for e in errors)


def test_cli_train_shows_all_metrics_with_labels() -> None:
    """train command shows all metrics in Rich Table."""
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = os.path.join(tmp, "outputs")

        with patch(
            "redstar_plate_ocr.pipeline.trainer.Trainer.train",
        ) as mock_train:
            mock_train.return_value = {
                "best": {
                    "val_plate_accuracy": 0.95,
                    "val_cer": 0.05,
                    "val_country_accuracy": 0.98,
                    "val_format_accuracy": 0.99,
                    "val_square_accuracy": 0.97,
                },
                "last": {
                    "val_plate_accuracy": 0.90,
                    "val_cer": 0.10,
                    "val_country_accuracy": 0.95,
                    "val_format_accuracy": 0.96,
                    "val_square_accuracy": 0.93,
                },
            }
            result = runner.invoke(
                app,
                [
                    "train",
                    "--config",
                    "configs/model.yaml",
                    "--plate-config",
                    "configs/plate.yaml",
                    "--data-dir",
                    "data/",
                    "--output-dir",
                    out_dir,
                ],
            )

        assert result.exit_code == 0
        output = result.output
        assert "Plate (exact match)" in output
        assert "CER" in output
        assert "Region" in output
        assert "Format" in output
        assert "Square" in output


def test_log_epoch_summary_includes_square() -> None:
    """_log_epoch_summary includes val_square_accuracy."""
    val_metrics = {
        "val_plate_accuracy": 0.9,
        "val_cer": 0.1,
        "val_country_accuracy": 0.8,
        "val_format_accuracy": 0.85,
        "val_square_accuracy": 0.75,
    }
    log_line = (
        f"plate="
        f"{val_metrics.get('val_plate_accuracy', 0):.4f} "
        f"cer={val_metrics.get('val_cer', 0):.4f} "
        f"ctry="
        f"{val_metrics.get('val_country_accuracy', 0):.4f} "
        f"fmt="
        f"{val_metrics.get('val_format_accuracy', 0):.4f} "
        f"square="
        f"{val_metrics.get('val_square_accuracy', 0):.4f}"
    )
    assert "plate=0.9000" in log_line
    assert "cer=0.1000" in log_line
    assert "ctry=0.8000" in log_line
    assert "fmt=0.8500" in log_line
    assert "square=0.7500" in log_line


def test_validate_dataset_quiet_returns_counts() -> None:
    """_validate_dataset with quiet=True returns (errors, counts)."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "train.csv"
        csv_path.write_text(
            "image_path,plate_text,region,plate_type\n"
            "img.png,A123AA12,RU,standard\n",
        )
        errors, counts = _validate_dataset(
            "configs/plate.yaml",
            tmp,
            "train",
            quiet=True,
        )
        assert isinstance(errors, list)
        assert isinstance(counts, dict)


def test_validate_dataset_quiet_no_print(
    capsys: object,
) -> None:
    """_validate_dataset with quiet=True does not print."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "train.csv"
        csv_path.write_text(
            "image_path,plate_text,region,plate_type\n"
            "img.png,A123AA12,RU,standard\n",
        )
        _validate_dataset(
            "configs/plate.yaml",
            tmp,
            "train",
            quiet=True,
        )
        captured = capsys.readouterr()
        assert "Dataset: train" not in captured.out


def test_validate_dataset_default_prints() -> None:
    """_validate_dataset without quiet prints dataset summary."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "train.csv"
        csv_path.write_text(
            "image_path,plate_text,region,plate_type\n"
            "img.png,A123AA12,RU,standard\n",
        )
        result = runner.invoke(
            app,
            [
                "validate",
                "--plate-config",
                "configs/plate.yaml",
                "--data-dir",
                tmp,
                "--split",
                "train",
            ],
        )
        assert "Dataset: train" in result.output


def test_print_startup_panel_renders() -> None:
    """_print_startup_panel renders unified panel."""
    trainer = MagicMock()
    trainer.epochs = 50
    trainer.base_lr = 0.001
    trainer.batch_size = 32
    trainer.device = "cpu"
    trainer.use_amp = False
    trainer.config.gradient_accumulation_steps = 1
    trainer.config.es_metric = "val_plate_accuracy"
    trainer.config.es_patience = 10
    trainer.start_epoch = 0

    cfg = {
        "training": {},
    }
    train_counts = {"RU": {"standard": 800, "square": 200}}
    val_counts = {"RU": {"standard": 150, "square": 40}}

    from io import StringIO

    from rich.console import Console

    buf = StringIO()
    con = Console(file=buf, force_terminal=True)
    _print_startup_panel(
        trainer,
        cfg,
        None,
        train_counts,
        val_counts,
        console=con,
    )
    output = buf.getvalue()
    assert "Training Configuration" in output
    assert "Epochs" in output
    assert "50" in output
    assert "Country" in output
    assert "Train" in output
    assert "Val" in output
    assert "800" in output
    assert "150" in output
