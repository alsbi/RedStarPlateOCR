"""Тесты для _StratifiedBatchSampler."""

import random

import pytest

from redstar_plate_ocr.data.dataloader import _StratifiedBatchSampler


class _FakeDataset:
    """Минимальный mock датасета для тестов sampler."""

    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)


def _make_samples() -> list[dict]:
    """4 группы: (standard,RU), (square,RU), (standard,KZ), (square,KZ)."""
    samples: list[dict] = []
    for _ in range(8):
        samples.append({"plate_type": "standard", "region": "RU"})
    for _ in range(6):
        samples.append({"plate_type": "square", "region": "RU"})
    for _ in range(5):
        samples.append({"plate_type": "standard", "region": "KZ"})
    for _ in range(4):
        samples.append({"plate_type": "square", "region": "KZ"})
    return samples


class TestStratifiedBatchSamplerCompositeKey:
    """Тесты композитного ключа (plate_type, region)."""

    @pytest.fixture()
    def dataset(self) -> _FakeDataset:
        return _FakeDataset(_make_samples())

    def test_four_groups_with_composite_key(
        self, dataset: _FakeDataset
    ) -> None:
        """При stratify_keys=['plate_type','region'] — 4 группы."""
        sampler = _StratifiedBatchSampler(
            dataset,
            batch_size=4,
            is_train=False,
            stratify_keys=["plate_type", "region"],
        )
        assert len(sampler._groups) == 4

    def test_batch_contains_multiple_groups(
        self, dataset: _FakeDataset
    ) -> None:
        """Каждый батч содержит сэмплы из >=2 групп."""
        sampler = _StratifiedBatchSampler(
            dataset,
            batch_size=4,
            is_train=False,
            stratify_keys=["plate_type", "region"],
        )
        for batch in sampler._indices:
            groups_in_batch: set[tuple[str, ...]] = set()
            for idx in batch:
                s = dataset.samples[idx]
                key = (s["plate_type"], s["region"])
                groups_in_batch.add(key)
            if len(sampler._groups) >= 2:
                assert len(groups_in_batch) >= 2, (
                    f"Батч {batch} содержит сэмплы только из {groups_in_batch}"
                )

    def test_all_groups_represented_large_batch(
        self, dataset: _FakeDataset
    ) -> None:
        """При batch_size >= len(groups) все группы в батче."""
        sampler = _StratifiedBatchSampler(
            dataset,
            batch_size=10,
            is_train=False,
            stratify_keys=["plate_type", "region"],
        )
        first_batch = sampler._indices[0]
        groups_in_batch: set[tuple[str, ...]] = set()
        for idx in first_batch:
            s = dataset.samples[idx]
            groups_in_batch.add((s["plate_type"], s["region"]))
        assert len(groups_in_batch) == 4

    def test_empty_dataset_no_crash(self) -> None:
        """Empty dataset (n=0) does not crash sampler."""
        empty_ds = _FakeDataset([])
        sampler = _StratifiedBatchSampler(
            empty_ds,
            batch_size=4,
            is_train=False,
        )
        assert len(sampler._groups) == 0
        assert len(sampler._indices) == 0
        assert list(sampler) == []

    def test_all_samples_present(self, dataset: _FakeDataset) -> None:
        """Все сэмплы присутствуют ровно 1 раз."""
        sampler = _StratifiedBatchSampler(
            dataset,
            batch_size=4,
            is_train=False,
            stratify_keys=["plate_type", "region"],
        )
        all_indices: list[int] = []
        for batch in sampler._indices:
            all_indices.extend(batch)
        assert sorted(all_indices) == list(range(len(dataset)))

    def test_no_duplicate_indices(self, dataset: _FakeDataset) -> None:
        """Нет дубликатов индексов."""
        sampler = _StratifiedBatchSampler(
            dataset,
            batch_size=4,
            is_train=False,
            stratify_keys=["plate_type", "region"],
        )
        all_indices: list[int] = []
        for batch in sampler._indices:
            all_indices.extend(batch)
        assert len(all_indices) == len(set(all_indices))


class TestStratifiedBatchSamplerSingleKey:
    """Тесты с stratify_keys=['plate_type'] — как старый sampler."""

    @pytest.fixture()
    def dataset(self) -> _FakeDataset:
        return _FakeDataset(_make_samples())

    def test_two_groups_plate_type_only(self, dataset: _FakeDataset) -> None:
        """При stratify_keys=['plate_type'] — 2 группы."""
        sampler = _StratifiedBatchSampler(
            dataset,
            batch_size=4,
            is_train=False,
            stratify_keys=["plate_type"],
        )
        assert len(sampler._groups) == 2

    def test_batch_mixed_types_single_key(self, dataset: _FakeDataset) -> None:
        """Полные батчи содержат смесь plate_type."""
        sampler = _StratifiedBatchSampler(
            dataset,
            batch_size=4,
            is_train=False,
            stratify_keys=["plate_type"],
        )
        for batch in sampler._indices:
            if len(batch) < sampler.batch_size:
                continue  # последний неполный батч
            types_in_batch: set[str] = set()
            for idx in batch:
                types_in_batch.add(dataset.samples[idx]["plate_type"])
            if len(sampler._groups) >= 2:
                assert len(types_in_batch) >= 2, (
                    f"Батч {batch} содержит только {types_in_batch}"
                )

    def test_all_samples_present_single_key(
        self, dataset: _FakeDataset
    ) -> None:
        """Все сэмплы присутствуют ровно 1 раз."""
        sampler = _StratifiedBatchSampler(
            dataset,
            batch_size=4,
            is_train=False,
            stratify_keys=["plate_type"],
        )
        all_indices: list[int] = []
        for batch in sampler._indices:
            all_indices.extend(batch)
        assert sorted(all_indices) == list(range(len(dataset)))

    def test_deterministic_with_seed(self, dataset: _FakeDataset) -> None:
        """С одинаковым seed результат одинаковый."""
        random.seed(42)
        s1 = _StratifiedBatchSampler(
            dataset,
            batch_size=4,
            is_train=True,
            stratify_keys=["plate_type"],
        )
        random.seed(42)
        s2 = _StratifiedBatchSampler(
            dataset,
            batch_size=4,
            is_train=True,
            stratify_keys=["plate_type"],
        )
        assert s1._indices == s2._indices
