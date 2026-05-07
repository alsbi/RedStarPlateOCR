"""Тесты для PlateDataset."""

import os

import pytest

from redstar_plate_ocr.data.dataset import PlateDataset
from redstar_plate_ocr.data.transforms import PreprocessPipeline

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
VAL_CSV = os.path.join(DATASET_ROOT, "val", "val.csv")


class TestPlateDataset:
    """Тесты датасета номерных знаков."""

    @pytest.fixture()
    def dataset(self) -> PlateDataset:
        """Создаёт датасет с трансформом."""
        transform = PreprocessPipeline()
        return PlateDataset(
            csv_path=VAL_CSV,
            dataset_root=DATASET_ROOT,
            transform=transform,
        )

    def test_len(self, dataset: PlateDataset) -> None:
        """Датасет имеет ненулевую длину."""
        assert len(dataset) > 0

    def test_getitem(self, dataset: PlateDataset) -> None:
        """Загрузка сэмпла возвращает корректные ключи."""
        sample = dataset[0]
        assert "image" in sample
        assert "plate_text" in sample
        assert "region" in sample
        assert "plate_type" in sample
        assert "orig_h" in sample
        assert "orig_w" in sample

    def test_image_shape(self, dataset: PlateDataset) -> None:
        """Тензор изображения формы (3, 80, 192)."""
        sample = dataset[0]
        assert sample["image"].shape == (3, 80, 192)

    def test_plate_text_is_string(self, dataset: PlateDataset) -> None:
        """plate_text — строка."""
        sample = dataset[0]
        assert isinstance(sample["plate_text"], str)

    def test_orig_dims_positive(self, dataset: PlateDataset) -> None:
        """orig_h и orig_w положительные."""
        sample = dataset[0]
        assert sample["orig_h"] > 0
        assert sample["orig_w"] > 0

    def test_load_three_samples(self, dataset: PlateDataset) -> None:
        """Загрузка 3 сэмплов из val.csv → корректные данные."""
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            assert sample["image"].shape == (3, 80, 192)
            assert isinstance(sample["plate_text"], str)
            assert len(sample["plate_text"]) > 0
