"""Пайплайн предобработки: augment → unpad → scale → letterbox → normalize."""

from __future__ import annotations

import logging
from collections.abc import Callable

import albumentations as A
import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from redstar_plate_ocr.pipeline.enhancement import (
    build_enhancement_stack,
)

logger = logging.getLogger(__name__)


def auto_unpad(image: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Strip uniform-color padding from a pre-canvased image.

    Detects padding by checking that all four corners share the same
    uniform color, then crops from each edge until reaching non-padding
    pixels.  Returns the original image unchanged when no padding is
    detected (e.g. raw plate crops).

    This handles images that were already placed on a canvas (like
    256×256 with gray 114 padding) so that downstream scale+letterbox
    produces the correct content_mask instead of treating the padding
    as plate content.
    """
    h, w = image.shape[:2]
    if h < 8 or w < 8:
        return image

    corner_size = max(1, min(h, w) // 20)
    corner_size = min(corner_size, h // 4, w // 4)

    corners = np.concatenate(
        [
            image[:corner_size, :corner_size].reshape(-1, 3),
            image[:corner_size, -corner_size:].reshape(-1, 3),
            image[-corner_size:, :corner_size].reshape(-1, 3),
            image[-corner_size:, -corner_size:].reshape(-1, 3),
        ]
    )

    # If corners are not uniform -> not a pre-canvased image
    corner_std = corners.std()
    if corner_std > threshold:
        return image

    pad_color = corners.mean(axis=0)

    # Mask: pixel is padding if every channel is close to pad_color
    diff = np.abs(image.astype(np.float32) - pad_color)
    is_pad = diff.max(axis=2) < 15

    # Find content bounds (first/last non-padding row/col)
    row_has_content = ~is_pad.all(axis=1)
    col_has_content = ~is_pad.all(axis=0)
    content_rows = np.where(row_has_content)[0]
    content_cols = np.where(col_has_content)[0]

    if len(content_rows) == 0 or len(content_cols) == 0:
        return image

    top = content_rows[0]
    bottom = content_rows[-1] + 1
    left = content_cols[0]
    right = content_cols[-1] + 1

    # Only crop if there is significant padding (>= 5% of smallest dim)
    min_border = max(2, int(min(h, w) * 0.05))
    has_significant_padding = (
        top >= min_border
        or h - bottom >= min_border
        or left >= min_border
        or w - right >= min_border
    )
    if not has_significant_padding:
        return image

    logger.debug(
        "auto_unpad: %dx%d → %dx%d (crop rows %d:%d, cols %d:%d)",
        h,
        w,
        bottom - top,
        right - left,
        top,
        bottom,
        left,
        right,
    )
    return image[top:bottom, left:right]


class PreprocessPipeline:
    """Пайплайн предобработки: augment → unpad → scale → letterbox → normalize.

    The *unpad* step automatically strips uniform-color padding from
    images that were already placed on a canvas (e.g. 256×256 with
    gray borders).  Raw plate crops pass through unchanged.
    """

    def __init__(
        self,
        canvas_height: int = 80,
        canvas_width: int = 256,
        pad_color: int = 128,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        augmentation: A.Compose | None = None,
        enhancement_config: dict[str, object] | None = None,
        enhancement_enabled: bool = True,
        hooks: dict[str, Callable[[np.ndarray | torch.Tensor], None]]
        | None = None,
    ) -> None:
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.pad_color = pad_color
        self.mean = np.array(
            mean if mean is not None else [0.5, 0.5, 0.5],
            dtype=np.float32,
        )
        self.std = np.array(
            std if std is not None else [0.5, 0.5, 0.5],
            dtype=np.float32,
        )
        self.augmentation = augmentation
        self._hooks = hooks or {}
        self.enhancement_enabled = enhancement_enabled
        self._assessor, self._enhancer = build_enhancement_stack(
            enhancement_config
        )

    def get_aug_description(self) -> list[str]:
        """Возвращает список описаний трансформаций аугментации."""
        if self.augmentation is None:
            return []
        return [str(t) for t in self.augmentation.transforms]

    def _emit(self, event: str, data: np.ndarray | torch.Tensor) -> None:
        """Вызвать хук события, если зарегистрирован."""
        hook = self._hooks.get(event)
        if hook is not None:
            hook(data)

    def __call__(
        self, image: NDArray[np.uint8]
    ) -> tuple[torch.Tensor, int, int]:
        """Возвращает (tensor, scaled_h, scaled_w)."""
        if self.augmentation is not None:
            image = self.augmentation(image=image)["image"]
            self._emit("on_augmented", image)

        unpadded = auto_unpad(image)
        self._emit("on_unpadded", unpadded)

        scaled = self._scale(unpadded)
        self._emit("on_scaled", scaled)
        scaled_h, scaled_w = scaled.shape[:2]
        letterboxed = self._letterbox(scaled)
        self._emit("on_letterboxed", letterboxed)
        tensor = self._normalize(letterboxed)
        self._emit("on_normalized", tensor)
        return tensor, scaled_h, scaled_w

    def _scale(self, image: np.ndarray) -> np.ndarray:
        """Масштабирование с сохранением пропорций.

        Микро-кропы и низкокачественные изображения проходят через
        :class:`SmartEnhancer` перед финальным resize (Lanczos4).
        """
        img = image

        # Smart conditional enhancement
        if (
            self.enhancement_enabled
            and self._assessor is not None
            and self._enhancer is not None
        ):
            if self._assessor.needs_enhancement(img):
                logger.debug(
                    "Enhancing crop: %s",
                    self._assessor.describe(img),
                )
                img = self._enhancer.enhance(img)

        h, w = img.shape[:2]
        scale = min(self.canvas_width / w, self.canvas_height / h)
        new_w = round(w * scale)
        new_h = round(h * scale)
        scaled = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_LANCZOS4,
        )
        return scaled

    def _letterbox(self, image: np.ndarray) -> np.ndarray:
        """Вписывание в канву с выравниванием по левому верхнему углу."""
        h, w = image.shape[:2]
        canvas = np.full(
            (self.canvas_height, self.canvas_width, 3),
            self.pad_color,
            dtype=np.uint8,
        )
        canvas[:h, :w, :] = image
        return canvas

    def _normalize(self, image: np.ndarray) -> torch.Tensor:
        """Нормализация: float32/255, (x-mean)/std, HWC→CHW."""
        img = image.astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        return torch.from_numpy(img.transpose(2, 0, 1).copy())
