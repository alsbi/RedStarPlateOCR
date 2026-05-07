"""Тесты для DataLoader со stratified interleaving."""

import random

import pytest

from redstar_plate_ocr.data.dataloader import (
    _StratifiedBatchSampler,
    build_dataloader,
)
from redstar_plate_ocr.data.dataset import PlateDataset
from redstar_plate_ocr.data.transforms import PreprocessPipeline

DATASET_ROOT = "dataset"
VAL_CSV = "dataset/val/val.csv"


class TestStratifiedBatchSamplerInterleaving:
    """Тесты stratified interleaving в _StratifiedBatchSampler."""

    @pytest.fixture()
    def dataset(self) -> PlateDataset:
        """Создаёт датасет."""
        transform = PreprocessPipeline()
        return PlateDataset(
            csv_path=VAL_CSV,
            dataset_root=DATASET_ROOT,
            transform=transform,
        )

    def test_build_indices_mixed_types_in_batch(
        self, dataset: PlateDataset
    ) -> None:
        """Батчи содержат смесь standard и square (stratified)."""
        sampler = _StratifiedBatchSampler(
            dataset, batch_size=16, is_train=False
        )
        for batch in sampler._indices:
            types = {dataset.samples[i]["plate_type"] for i in batch}
            if len(types) > 1:
                return  # Нашли смешанный батч — ОК
        # Если все батчи однородные — проверяем что групп < 2
        groups = sampler._groups
        assert len(groups) < 2, (
            "Stratified interleaving должен смешивать типы в батчах"
        )

    def test_all_samples_present(self, dataset: PlateDataset) -> None:
        """Все сэмплы датасета присутствуют ровно 1 раз."""
        sampler = _StratifiedBatchSampler(
            dataset, batch_size=8, is_train=False
        )
        all_indices: list[int] = []
        for batch in sampler._indices:
            all_indices.extend(batch)
        assert sorted(all_indices) == list(range(len(dataset)))

    def test_no_duplicate_indices(self, dataset: PlateDataset) -> None:
        """Нет дубликатов индексов."""
        sampler = _StratifiedBatchSampler(
            dataset, batch_size=8, is_train=False
        )
        all_indices: list[int] = []
        for batch in sampler._indices:
            all_indices.extend(batch)
        assert len(all_indices) == len(set(all_indices))

    def test_interleaving_deterministic_when_seed_set(
        self, dataset: PlateDataset
    ) -> None:
        """С одинаковым seed результат одинаковый."""
        random.seed(42)
        s1 = _StratifiedBatchSampler(dataset, batch_size=8, is_train=True)
        random.seed(42)
        s2 = _StratifiedBatchSampler(dataset, batch_size=8, is_train=True)
        assert s1._indices == s2._indices

    def test_batch_size_respected(self, dataset: PlateDataset) -> None:
        """Размер батча не превышает заданный."""
        bs = 4
        sampler = _StratifiedBatchSampler(
            dataset, batch_size=bs, is_train=False
        )
        for batch in sampler._indices:
            assert len(batch) <= bs


class TestBuildDataLoader:
    """Тесты DataLoader с stratified interleaving."""

    @pytest.fixture()
    def dataset(self) -> PlateDataset:
        """Создаёт датасет."""
        transform = PreprocessPipeline()
        return PlateDataset(
            csv_path=VAL_CSV,
            dataset_root=DATASET_ROOT,
            transform=transform,
        )

    def test_dataloader_batch_mixed_types(self, dataset: PlateDataset) -> None:
        """Батчи содержат смесь plate_type (stratified interleaving)."""
        loader = build_dataloader(
            dataset,
            batch_size=16,
            num_workers=0,
            is_train=True,
        )
        for batch in loader:
            types = batch["plate_type"]
            if len(set(types)) > 1:
                return  # Смешанный батч — ОК
        # Если не нашли — значит только один тип в датасете
        groups = _StratifiedBatchSampler(
            dataset, batch_size=16, is_train=False
        )._groups
        assert len(groups) < 2
