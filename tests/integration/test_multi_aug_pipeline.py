"""Интеграционные тесты multi-aug через ConcatDataset."""

from __future__ import annotations

from unittest.mock import patch

import albumentations as A
import numpy as np
import torch

from redstar_plate_ocr.data.dataloader import (
    _ConcatDatasetWithSamples,
    build_dataloader,
)
from redstar_plate_ocr.data.dataset import PlateDataset
from redstar_plate_ocr.data.transforms import PreprocessPipeline
from redstar_plate_ocr.nn.model import PlateOCRModel
from redstar_plate_ocr.pipeline.trainer import Trainer
from redstar_plate_ocr.plate.config import PlateConfig


def _make_image(h: int = 100, w: int = 400) -> np.ndarray:
    """Создаёт RGB-изображение uint8."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_base_dataset(n_samples: int = 4) -> PlateDataset:
    """Создаёт мок PlateDataset."""
    samples = [
        {
            "image_path": f"img_{i}.png",
            "plate_text": f"A{i:03d}BC99",
            "region": "RU",
            "plate_type": "standard",
        }
        for i in range(n_samples)
    ]
    return PlateDataset(
        csv_path="mock.csv",
        dataset_root="/mock",
        transform=PreprocessPipeline(),
        samples=samples,
    )


class TestConcatDatasetIntegration:
    """Интеграционные тесты ConcatDataset вместо pipelines."""

    def test_concat_2_datasets_batch_shape(
        self,
    ) -> None:
        """ConcatDataset из 2 датасетов → батч формы (B, 3, 80, 256)."""
        base = _make_base_dataset(n_samples=4)
        aug = A.Compose([A.Rotate(limit=3, p=1.0)])
        aug_pipeline = PreprocessPipeline(augmentation=aug)
        aug_ds = PlateDataset(
            csv_path="mock.csv",
            dataset_root="/mock",
            transform=aug_pipeline,
            samples=list(base.samples),
        )
        concat = _ConcatDatasetWithSamples([base, aug_ds])

        with patch.object(
            PlateDataset,
            "_read_image",
            return_value=_make_image(),
        ):
            loader = build_dataloader(
                concat,
                batch_size=2,
                num_workers=0,
                is_train=True,
            )
            batch = next(iter(loader))

        assert isinstance(batch["image"], torch.Tensor)
        assert batch["image"].shape[1:] == (3, 80, 256)

    def test_concat_3_datasets_batch_shape(
        self,
    ) -> None:
        """ConcatDataset из 3 датасетов → батч корректной формы."""
        base = _make_base_dataset(n_samples=6)
        aug = A.Compose([A.Rotate(limit=3, p=1.0)])
        aug_pipeline = PreprocessPipeline(augmentation=aug)
        aug_ds1 = PlateDataset(
            csv_path="mock.csv",
            dataset_root="/mock",
            transform=aug_pipeline,
            samples=list(base.samples),
        )
        aug_ds2 = PlateDataset(
            csv_path="mock.csv",
            dataset_root="/mock",
            transform=aug_pipeline,
            samples=list(base.samples),
        )
        concat = _ConcatDatasetWithSamples([base, aug_ds1, aug_ds2])

        with patch.object(
            PlateDataset,
            "_read_image",
            return_value=_make_image(),
        ):
            loader = build_dataloader(
                concat,
                batch_size=3,
                num_workers=0,
                is_train=True,
            )
            batch = next(iter(loader))

        assert isinstance(batch["image"], torch.Tensor)
        assert batch["image"].shape[1:] == (3, 80, 256)

    def test_concat_metadata(self) -> None:
        """Метаданные корректны при ConcatDataset."""
        base = _make_base_dataset(n_samples=2)
        aug = A.Compose([A.Rotate(limit=3, p=1.0)])
        aug_pipeline = PreprocessPipeline(augmentation=aug)
        aug_ds = PlateDataset(
            csv_path="mock.csv",
            dataset_root="/mock",
            transform=aug_pipeline,
            samples=list(base.samples),
        )
        concat = _ConcatDatasetWithSamples([base, aug_ds])

        with patch.object(
            PlateDataset,
            "_read_image",
            return_value=_make_image(),
        ):
            loader = build_dataloader(
                concat,
                batch_size=2,
                num_workers=0,
                is_train=True,
            )
            batch = next(iter(loader))

        assert len(batch["plate_text"]) == 2
        assert len(batch["region"]) == 2
        assert len(batch["plate_type"]) == 2

    def test_no_concat_uses_simple_collate(
        self,
    ) -> None:
        """Без ConcatDataset → обычный датасет."""
        base = _make_base_dataset(n_samples=2)

        with patch.object(
            PlateDataset,
            "_read_image",
            return_value=_make_image(),
        ):
            loader = build_dataloader(
                base,
                batch_size=2,
                num_workers=0,
                is_train=True,
            )
            batch = next(iter(loader))

        assert batch["image"].shape[0] == 2


class TestTrainerBuildTrainLoaderIntegration:
    """Интеграционный тест Trainer._build_train_loader."""

    def _make_trainer(self, plate_config: PlateConfig) -> Trainer:
        """Создаёт Trainer с мок-датасетами."""
        cfg = plate_config
        model = PlateOCRModel(plate_config=cfg)

        train_ds = _make_base_dataset(n_samples=4)
        val_ds = _make_base_dataset(n_samples=2)

        training_cfg = {
            "training": {
                "epochs": 1,
                "lr": 1e-3,
                "batch_size": 2,
                "use_amp": False,
            },
            "augmentation": {
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

    def test_trainer_build_train_loader_with_aug(
        self,
        plate_config: PlateConfig,
    ) -> None:
        """Trainer._build_train_loader(phase='full') → батч."""
        trainer = self._make_trainer(plate_config)

        from redstar_plate_ocr.data.dataloader import (
            build_dataloader as original_fn,
        )

        def _build_with_zero_workers(*args, **kwargs):
            kwargs["num_workers"] = 0
            return original_fn(*args, **kwargs)

        with (
            patch.object(
                PlateDataset,
                "_read_image",
                return_value=_make_image(),
            ),
            patch(
                "redstar_plate_ocr.pipeline.trainer.build_dataloader",
                side_effect=_build_with_zero_workers,
            ),
        ):
            loader = trainer._build_train_loader(phase="full")
            batch = next(iter(loader))

        assert isinstance(batch["image"], torch.Tensor)
        assert batch["image"].shape[1:] == (3, 80, 256)

    def test_trainer_build_train_loader_no_aug(
        self,
        plate_config: PlateConfig,
    ) -> None:
        """Trainer._build_train_loader(phase='none') → обычный батч."""
        trainer = self._make_trainer(plate_config)

        from redstar_plate_ocr.data.dataloader import (
            build_dataloader as original_fn,
        )

        def _build_with_zero_workers(*args, **kwargs):
            kwargs["num_workers"] = 0
            return original_fn(*args, **kwargs)

        with (
            patch.object(
                PlateDataset,
                "_read_image",
                return_value=_make_image(),
            ),
            patch(
                "redstar_plate_ocr.pipeline.trainer.build_dataloader",
                side_effect=_build_with_zero_workers,
            ),
        ):
            loader = trainer._build_train_loader(phase="none")
            batch = next(iter(loader))

        assert batch["image"].shape[0] == 2
