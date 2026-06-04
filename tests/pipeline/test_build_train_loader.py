"""Tests for Trainer._build_train_loader."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from redstar_plate_ocr.data.dataloader import _ConcatDatasetWithSamples
from redstar_plate_ocr.pipeline.trainer import Trainer
from redstar_plate_ocr.plate.config import PlateConfig


def _make_trainer(plate_config: PlateConfig) -> Trainer:
    """Create a minimal Trainer instance for testing."""
    from redstar_plate_ocr.nn.model import PlateOCRModel

    cfg = plate_config
    model = PlateOCRModel(plate_config=cfg)

    train_ds = MagicMock()
    train_ds.samples = [
        {
            "image_path": "a.jpg",
            "plate_text": "A000AA00",
            "region": "RU",
            "plate_type": "standard",
        },
    ]
    train_ds.csv_path = "mock.csv"
    train_ds.dataset_root = "/mock"
    val_ds = MagicMock()
    val_ds.samples = [
        {
            "image_path": "b.jpg",
            "plate_text": "A111BB11",
            "region": "RU",
            "plate_type": "standard",
        },
    ]

    training_cfg = {
        "training": {
            "epochs": 1,
            "lr": 1e-3,
            "batch_size": 1,
            "use_amp": False,
        },
        "augmentation": {
            "warmup": {"enable_warmup": False},
            "rotation": {"enabled": True, "limit": 5, "p": 0.5},
        },
    }

    with patch(
        "redstar_plate_ocr.pipeline.utils.detect_device",
        return_value=(torch.device("cpu"), False, "cpu"),
    ):
        return Trainer(
            model=model,
            plate_config=cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            cfg=training_cfg,
        )


def test_build_train_loader_phase_full(
    plate_config: PlateConfig,
):
    """phase='full' → ConcatDataset: 1 base + 1 single + 1 multi (default)."""
    trainer = _make_trainer(plate_config)

    with patch(
        "redstar_plate_ocr.pipeline.trainer.build_dataloader",
    ) as mock_build:
        mock_build.return_value = MagicMock()
        trainer._build_train_loader(phase="full")

        dataset = mock_build.call_args.args[0]
        assert isinstance(dataset, _ConcatDatasetWithSamples)
        # 1 base + 1 single + num_multi_aug(=1) = 3
        assert len(dataset.datasets) == 3
        assert dataset.datasets[0].transform.augmentation is None
        assert dataset.datasets[1].transform.augmentation is not None
        assert dataset.datasets[2].transform.augmentation is not None


def test_build_train_loader_phase_single(
    plate_config: PlateConfig,
):
    """phase='single' → ConcatDataset: 1 base + 1 single, no multi."""
    trainer = _make_trainer(plate_config)

    with patch(
        "redstar_plate_ocr.pipeline.trainer.build_dataloader",
    ) as mock_build:
        mock_build.return_value = MagicMock()
        trainer._build_train_loader(phase="single")

        dataset = mock_build.call_args.args[0]
        assert isinstance(dataset, _ConcatDatasetWithSamples)
        # 1 base + 1 single = 2
        assert len(dataset.datasets) == 2
        assert dataset.datasets[0].transform.augmentation is None
        assert dataset.datasets[1].transform.augmentation is not None


def test_build_train_loader_phase_none(
    plate_config: PlateConfig,
):
    """phase='none' → simple dataset without augmentation."""
    trainer = _make_trainer(plate_config)

    with patch(
        "redstar_plate_ocr.pipeline.trainer.build_dataloader",
    ) as mock_build:
        mock_build.return_value = MagicMock()
        trainer._build_train_loader(phase="none")

        dataset = mock_build.call_args.args[0]
        assert not isinstance(dataset, _ConcatDatasetWithSamples)
        assert dataset.transform is not None
        assert dataset.transform.augmentation is None


def test_build_train_loader_num_multi_aug_three(
    plate_config: PlateConfig,
):
    """num_multi_aug=3 + phase='full' → 1 base + 1 single + 3 multi = 5."""
    trainer = _make_trainer(plate_config)
    trainer.config.num_multi_aug = 3

    with patch(
        "redstar_plate_ocr.pipeline.trainer.build_dataloader",
    ) as mock_build:
        mock_build.return_value = MagicMock()
        trainer._build_train_loader(phase="full")

        dataset = mock_build.call_args.args[0]
        assert isinstance(dataset, _ConcatDatasetWithSamples)
        assert len(dataset.datasets) == 5
        assert dataset.datasets[0].transform.augmentation is None
        for ds in dataset.datasets[1:]:
            assert ds.transform.augmentation is not None


def test_compute_aug_phase(plate_config: PlateConfig):
    """Test _compute_aug_phase returns correct phase per epoch."""
    trainer = _make_trainer(plate_config)
    # warmup_epochs=2, single_aug_epochs=3, epochs=1 (but override)
    trainer.config.warmup_epochs = 2
    trainer.config.single_aug_epochs = 3
    trainer.config.no_aug_epochs = 5
    trainer.config.epochs = 50

    # Warmup: epochs 0-1 → "none"
    assert trainer._compute_aug_phase(0) == "none"
    assert trainer._compute_aug_phase(1) == "none"
    # SingleAug: epochs 2-4 → "single"
    assert trainer._compute_aug_phase(2) == "single"
    assert trainer._compute_aug_phase(3) == "single"
    assert trainer._compute_aug_phase(4) == "single"
    # Full: epochs 5-44 → "full"
    assert trainer._compute_aug_phase(5) == "full"
    assert trainer._compute_aug_phase(44) == "full"
    # NoAug: epochs 45-49 → "none"
    assert trainer._compute_aug_phase(45) == "none"
    assert trainer._compute_aug_phase(49) == "none"
