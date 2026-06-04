"""DataLoader со stratified batch sampling."""

from __future__ import annotations

import itertools
import platform
import random
from collections import defaultdict
from typing import Any, Protocol

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Sampler

from redstar_plate_ocr.data.dataset import PlateDataset


class _HasSamples(Protocol):
    """Протокол объекта с полем samples."""

    @property
    def samples(self) -> list[dict[str, str]]: ...

    def __len__(self) -> int: ...


class _ConcatDatasetWithSamples(ConcatDataset):
    """ConcatDataset с агрегированным .samples для stratified sampling.

    Обеспечивает совместимость с _HasSamples протоколом,
    который требует _StratifiedBatchSampler.
    """

    def __init__(self, datasets: list[PlateDataset]) -> None:
        super().__init__(datasets)
        self._samples: list[dict[str, str]] = []
        for ds in datasets:
            self._samples.extend(ds.samples)

    @property
    def samples(self) -> list[dict[str, str]]:
        return self._samples


class _StratifiedBatchSampler(Sampler[list[int]]):
    """BatchSampler с группировкой по композитному ключу.

    При ``original_prob < 1.0`` оригинальные сэмплы (индексы
    ``0 .. original_len-1``) включаются в батч с указанной
    вероятностью — каждый раз заново при каждой эпохе.
    """

    def __init__(
        self,
        dataset: _HasSamples,
        batch_size: int,
        is_train: bool = True,
        stratify_keys: list[str] | None = None,
        original_prob: float = 1.0,
        original_len: int = 0,
    ) -> None:
        if stratify_keys is None:
            stratify_keys = ["plate_type", "region"]
        self.batch_size = batch_size
        self.is_train = is_train
        self.stratify_keys = stratify_keys
        self.original_prob = original_prob
        self.original_len = original_len

        self._groups = self._build_groups(dataset)
        self._indices_built = not is_train
        if is_train:
            self._indices: list[list[int]] = []
        else:
            self._indices = self._build_indices()

    @staticmethod
    def _resolve_keys(
        samples: list[dict[str, str]],
        stratify_keys: list[str],
    ) -> list[str]:
        if not samples:
            return []
        available = set(samples[0].keys())
        return [k for k in stratify_keys if k in available]

    @staticmethod
    def _make_key(sample: dict[str, str], keys: list[str]) -> str:
        return str(tuple(sample[k] for k in keys))

    def _build_groups(self, dataset: _HasSamples) -> dict[str, list[int]]:
        """Группирует индексы по композитному ключу."""
        keys = self._resolve_keys(dataset.samples, self.stratify_keys)
        groups: dict[str, list[int]] = defaultdict(list)
        for i, sample in enumerate(dataset.samples):
            groups[self._make_key(sample, keys)].append(i)
        return dict(groups)

    @staticmethod
    def _interleave_groups(
        groups: dict[str, list[int]],
    ) -> list[int]:
        """Чередует индексы из групп по кругу."""
        iters = map(iter, groups.values())
        return [
            i
            for i in itertools.chain.from_iterable(
                itertools.zip_longest(*iters)
            )
            if i is not None
        ]

    def _shuffle_groups(self, groups: dict[str, list[int]]) -> None:
        """Перемешивает индексы внутри групп при обучении."""
        if not self.is_train:
            return
        for indices in groups.values():
            random.shuffle(indices)

    def _build_indices(self) -> list[list[int]]:
        """Строит батчи со stratified interleaving."""
        groups = {k: list(idx_list) for k, idx_list in self._groups.items()}
        self._shuffle_groups(groups)
        pool = self._interleave_groups(groups)
        pool = self._filter_original_indices(pool)
        batches = _chunk_batches(pool, self.batch_size)
        if self.is_train:
            random.shuffle(batches)
        return batches

    def _filter_original_indices(self, indices: list[int]) -> list[int]:
        """Исключает оригинальные сэмплы с вероятностью 1-original_prob.

        Работает только при is_train и original_prob < 1.0.
        Каждый вызов (каждая эпоха) рандомизирует заново.
        """
        if (
            not self.is_train
            or self.original_prob >= 1.0
            or self.original_len <= 0
        ):
            return indices
        if self.original_prob <= 0.0:
            return [i for i in indices if i >= self.original_len]
        return [
            i
            for i in indices
            if i >= self.original_len or random.random() < self.original_prob
        ]

    def __iter__(self):
        if self.is_train:
            self._indices = self._build_indices()
            self._indices_built = True
        return iter(self._indices)

    def __len__(self) -> int:
        if not self._indices_built:
            self._indices = self._build_indices()
            self._indices_built = True
        return len(self._indices)


def _pluck(batch: list[dict], key: str) -> list[Any]:
    return [item[key] for item in batch]


def _simple_collate_fn(batch: list[dict]) -> dict:
    """Кастомная функция сборки батча."""
    return {
        "image": torch.stack(_pluck(batch, "image")),
        "plate_text": _pluck(batch, "plate_text"),
        "region": _pluck(batch, "region"),
        "plate_type": _pluck(batch, "plate_type"),
        "orig_h": _pluck(batch, "orig_h"),
        "orig_w": _pluck(batch, "orig_w"),
    }


def seed_worker(worker_id: int) -> None:
    """Seed worker for reproducible augmentation.

    Also disables internal thread pools in worker processes to prevent
    thread explosion when multiple workers run concurrently — each worker
    is a separate process and only needs one thread for I/O + transforms.
    This is universally beneficial (macOS ``spawn``, Linux ``fork``).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def _effective_num_workers(requested: int) -> int:
    """Return safe number of workers for the current platform.

    On macOS ``spawn`` forces each worker to re-import the entire Python
    stack (torch, cv2, numpy, …) which adds 2–5 s overhead per worker
    and 17× pickle latency *per batch*.  For small-to-medium datasets
    single-process loading (0 workers) is significantly faster.
    Linux ``fork`` does copy-on-write so workers are cheap — keep
    the requested count there.
    """
    if requested > 0 and platform.system() == "Darwin":
        return 0
    return requested


def build_dataloader(
    dataset: PlateDataset | _ConcatDatasetWithSamples,
    batch_size: int = 32,
    num_workers: int = 4,
    is_train: bool = True,
    stratify_keys: list[str] | None = None,
    device: torch.device | None = None,
    original_prob: float = 1.0,
    original_len: int = 0,
) -> torch.utils.data.DataLoader:
    """Строит DataLoader со stratified batch sampling."""
    batch_sampler = _StratifiedBatchSampler(
        dataset,
        batch_size=batch_size,
        is_train=is_train,
        stratify_keys=stratify_keys,
        original_prob=original_prob,
        original_len=original_len,
    )

    nw = _effective_num_workers(num_workers)
    pin_memory = device is not None and device.type == "cuda"
    kwargs: dict = {"pin_memory": pin_memory}
    if nw > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 1
        kwargs["worker_init_fn"] = seed_worker
        if platform.system() == "Darwin":
            kwargs["multiprocessing_context"] = "spawn"

    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=nw,
        collate_fn=_simple_collate_fn,
        **kwargs,
    )


def _compute_target_square(
    standard_count: int,
    ratio: float,
    square_count: int,
) -> int | None:
    """Compute target square count, or None if no oversampling needed."""
    safe_ratio = max(0.001, min(0.999, ratio))
    target = int(standard_count * safe_ratio / (1 - safe_ratio))
    return target if target > square_count else None


def _maybe_compute_target(
    standard_count: int,
    square_count: int,
    ratio: float,
) -> int | None:
    """Return target square count, or None if oversampling not needed."""
    if standard_count == 0 or square_count == 0:
        return None
    return _compute_target_square(standard_count, ratio, square_count)


def _filter_indices(dataset: PlateDataset, plate_type: str) -> list[int]:
    """Возвращает индексы сэмплов заданного plate_type."""
    return [
        i
        for i in range(len(dataset))
        if dataset.samples[i]["plate_type"] == plate_type
    ]


def _build_extended_samples(
    dataset: PlateDataset,
    square: list[int],
    target: int,
) -> list[dict[str, str]]:
    """Создаёт новый список сэмплов с дублированием square."""
    extra_needed = target - len(square)
    extra = [square[i % len(square)] for i in range(extra_needed)]
    return list(dataset.samples) + [dataset.samples[i] for i in extra]


def _chunk_batches(pool: list[int], batch_size: int) -> list[list[int]]:
    """Разбивает pool на батчи заданного размера, пропуская пустые."""
    return [
        pool[i : i + batch_size]
        for i in range(0, len(pool), batch_size)
        if pool[i : i + batch_size]
    ]


def _balance_per_group(
    dataset: PlateDataset,
    group_keys: list[str] | None = None,
) -> PlateDataset:
    """Cap every group to the size of the smallest group.

    For warmup we want a perfectly balanced mini-dataset so that
    every country × format combination is seen equally.  Without
    balancing, the model sees 2852 RU-standard before it sees 391
    BY-square — causing GE/BY to lag at 0%.

    Returns a *new* PlateDataset where every (region, plate_type)
    group has exactly min_group_size samples.
    """
    if group_keys is None:
        group_keys = ["region", "plate_type"]

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for sample in dataset.samples:
        key = str(tuple(sample.get(k, "") for k in group_keys))
        groups[key].append(sample)

    if not groups:
        return dataset

    min_size = min(len(v) for v in groups.values())
    balanced = [s[:min_size] for s in groups.values()]
    flat = [s for grp in balanced for s in grp]

    if len(flat) == len(dataset.samples):
        return dataset

    return PlateDataset(
        csv_path=dataset.csv_path,
        dataset_root=dataset.dataset_root,
        transform=dataset.transform,
        samples=flat,
    )


def _oversample_square(
    dataset: PlateDataset,
    ratio: float,
) -> PlateDataset:
    """Дублирует square-сэмплы до целевой доли.

    Утилита для вызова из Trainer._build_train_loader().
    Работает только с PlateDataset — без isinstance-диспетчера.
    """
    standard = _filter_indices(dataset, "standard")
    square = _filter_indices(dataset, "square")

    target = _maybe_compute_target(len(standard), len(square), ratio)
    if target is None:
        return dataset

    new_samples = _build_extended_samples(dataset, square, target)

    return PlateDataset(
        csv_path=dataset.csv_path,
        dataset_root=dataset.dataset_root,
        transform=dataset.transform,
        samples=new_samples,
    )
