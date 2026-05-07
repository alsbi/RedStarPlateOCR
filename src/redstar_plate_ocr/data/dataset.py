"""Датасет номерных знаков из CSV."""

from __future__ import annotations

import csv
import os

import cv2
import numpy as np
import torch

from redstar_plate_ocr.data.transforms import PreprocessPipeline


class PlateDataset(torch.utils.data.Dataset):
    """Датасет номерных знаков из CSV."""

    def __init__(
        self,
        csv_path: str,
        dataset_root: str,
        transform: PreprocessPipeline | None = None,
        samples: list[dict[str, str]] | None = None,
        allowed_regions: list[str] | None = None,
    ) -> None:
        self.csv_path = csv_path
        self.dataset_root = dataset_root
        self.transform = transform
        self._allowed_regions = allowed_regions
        self.samples = self._resolve_samples(
            csv_path, samples, allowed_regions
        )

    @staticmethod
    def _any_filter(
        items: list[dict[str, str]],
        allowed: list[str] | None,
    ) -> list[dict[str, str]]:
        if allowed is None:
            return items
        return [s for s in items if s["region"] in allowed]

    def _resolve_samples(
        self,
        csv_path: str,
        samples: list[dict[str, str]] | None,
        allowed_regions: list[str] | None,
    ) -> list[dict[str, str]]:
        if samples is None:
            return self._load_csv(csv_path)
        return self._any_filter(samples, allowed_regions)

    def _load_csv(self, csv_path: str) -> list[dict[str, str]]:
        """Загружает данные из CSV."""
        raw: list[dict[str, str]] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw.append({
                    "image_path": row["image_path"],
                    "plate_text": row["plate_text"],
                    "region": row["region"],
                    "plate_type": row["plate_type"],
                })
        return self._any_filter(raw, self._allowed_regions)

    def __len__(self) -> int:
        """Количество сэмплов."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Возвращает словарь с данными сэмпла."""
        sample = self.samples[idx]
        img_path = os.path.join(self.dataset_root, sample["image_path"])
        image = self._read_image(img_path)

        content_h, content_w = image.shape[:2]
        if self.transform is not None:
            tensor, content_h, content_w = self.transform(image)
        else:
            tensor = self._default_transform(image)
        return {
            "image": tensor,
            "plate_text": sample["plate_text"],
            "region": sample["region"],
            "plate_type": sample["plate_type"],
            "orig_h": content_h,
            "orig_w": content_w,
        }

    @staticmethod
    def _read_image(
        path: str,
    ) -> np.ndarray:
        """Читает изображение и конвертирует в RGB."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _default_transform(
        image: np.ndarray,
    ) -> torch.Tensor:
        """Трансформ по умолчанию без предобработки."""
        img = image.astype(np.float32) / 255.0
        return torch.from_numpy(img.transpose(2, 0, 1).copy())
