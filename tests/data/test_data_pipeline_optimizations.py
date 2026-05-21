"""Тесты для оптимизаций data pipeline."""

from __future__ import annotations

import numpy as np

from redstar_plate_ocr.data.dataset import PlateDataset
from redstar_plate_ocr.data.transforms import PreprocessPipeline

# --- _normalize in-place ---


class TestNormalizeInplace:
    """Тесты in-place нормализации."""

    def test_normalize_produces_finite_values(self) -> None:
        """In-place нормализация даёт конечные значения."""
        pipe = PreprocessPipeline()
        img = np.random.randint(0, 255, (80, 256, 3), dtype=np.uint8)
        tensor, _, _ = pipe(img)
        assert tensor.isfinite().all()

    def test_normalize_shape_preserved(self) -> None:
        """Форма тензора корректна после in-place нормализации."""
        pipe = PreprocessPipeline()
        img = np.random.randint(0, 255, (80, 256, 3), dtype=np.uint8)
        tensor, _, _ = pipe(img)
        assert tensor.shape == (3, 80, 256)


# --- _oversample_square ---


class TestOversampleSquare:
    """Тесты _oversample_square."""

    def test_oversample_creates_new_dataset(self) -> None:
        """_oversample_square создаёт новый PlateDataset."""
        from redstar_plate_ocr.data.dataloader import _oversample_square

        samples = [
            {
                "image_path": f"s{i}.png",
                "plate_text": f"A{i:03d}AA00",
                "region": "RU",
                "plate_type": "standard",
            }
            for i in range(10)
        ] + [
            {
                "image_path": "q0.png",
                "plate_text": "A111BB11",
                "region": "RU",
                "plate_type": "square",
            },
        ]
        ds = PlateDataset(
            csv_path="mock.csv",
            dataset_root="/mock",
            transform=None,
            samples=samples,
        )

        result = _oversample_square(ds, 0.3)
        assert result is not ds
        assert isinstance(result, PlateDataset)
        assert len(result.samples) > len(ds.samples)
        # target_square = 4, original square = 1 → 10 + 4 = 14
        assert len(result.samples) == 14
