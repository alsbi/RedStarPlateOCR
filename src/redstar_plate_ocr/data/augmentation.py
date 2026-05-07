"""Билдер аугментаций из конфига."""

from __future__ import annotations

import random
from typing import Any, Callable

import albumentations as A
import cv2


def build_single_augmentation(
    config: dict,
    is_train: bool = True,
) -> A.Compose | None:
    """Ровно одна случайная аугментация из конфига.

    Выбирает одну из включённых трансформаций (p=1.0 внутри
    OneOf), что гарантирует применение ровно одной за вызов.
    """
    if not config:
        return None
    if not is_train:
        return None
    transforms = _build_transforms(config)
    if not transforms:
        return None
    # Каждая трансформация применяется с p=1.0 внутри OneOf,
    # а OneOf выбирает ровно одну.
    for t in transforms:
        t.p = 1.0
    composed = A.OneOf(transforms, p=1.0)
    return A.Compose([composed])


def build_multi_augmentation(
    config: dict,
    is_train: bool = True,
    min_aug: int = 2,
) -> A.Compose | None:
    """Рандомное количество (2+) аугментаций из конфига.

    Использует SomeOf для выбора случайного подмножества
    из доступных трансформаций (от min_aug до всех).
    Каждая выбранная трансформация применяется с p=1.0.
    Количество n рандомизируется при создании пайплайна.
    """
    if not config:
        return None
    if not is_train:
        return None
    transforms = _build_transforms(config)
    if not transforms:
        return None
    n_total = len(transforms)
    if n_total < min_aug:
        min_aug = n_total
    # Рандомное количество от min_aug до n_total
    n = random.randint(min_aug, n_total)
    # Каждая трансформация внутри SomeOf с p=1.0
    for t in transforms:
        t.p = 1.0
    composed = A.SomeOf(
        transforms,
        n=n,
        p=1.0,
        replace=False,
    )
    return A.Compose([composed])


def _is_enabled_config(value: Any) -> bool:
    return isinstance(value, dict) and value.get("enabled", False)


def _build_transforms(
    config: dict,
) -> list[A.BasicTransform]:
    """Создаёт список трансформаций: pixel-level → geometric.

    Pixel-level применяются к оригинальным пикселям,
    geometric — с fill=(128,128,128) для серого фона.
    """
    pixel_builders = {
        "brightness_contrast": _build_brightness_contrast,
        "motion_blur": _build_motion_blur,
        "gauss_noise": _build_gauss_noise,
        "jpeg_compression": _build_jpeg_compression,
        "coarse_dropout": _build_coarse_dropout,
        "dirt_spots": _build_dirt_spots,
        "defocus_blur": _build_defocus_blur,
    }
    geometric_builders = {
        "rotation": _build_rotation,
        "shift": _build_shift,
        "scale": _build_scale,
        "perspective": _build_perspective,
        "affine": _build_affine,
        "elastic": _build_elastic,
    }
    pixel_transforms: list[A.BasicTransform] = []
    geometric_transforms: list[A.BasicTransform] = []
    for key, cfg in config.items():
        if not _is_enabled_config(cfg):
            continue
        _append_transform(
            key, cfg, pixel_builders, geometric_builders,
            pixel_transforms, geometric_transforms,
        )
    return pixel_transforms + geometric_transforms


def _append_if_not_none(
    lst: list[A.BasicTransform],
    item: A.BasicTransform | None,
) -> None:
    if item is not None:
        lst.append(item)


def _append_transform(
    key: str,
    cfg: dict,
    pixel_builders: dict[str, Callable[[dict], A.BasicTransform | None]],
    geometric_builders: dict[
        str, Callable[[dict], A.BasicTransform | None]
    ],
    pixel_transforms: list[A.BasicTransform],
    geometric_transforms: list[A.BasicTransform],
) -> None:
    builder = pixel_builders.get(key)
    if builder is not None:
        _append_if_not_none(pixel_transforms, builder(cfg))
        return
    builder = geometric_builders.get(key)
    if builder is not None:
        _append_if_not_none(geometric_transforms, builder(cfg))


def _build_rotation(
    cfg: dict,
) -> A.Affine | None:
    """rotation → A.Affine (rotate) с fit_output."""
    return A.Affine(
        rotate=(-cfg["limit"], cfg["limit"]),
        border_mode=cv2.BORDER_CONSTANT,
        fill=(128, 128, 128),
        fit_output=True,
        p=cfg.get("p", 0.5),
    )


def _build_shift(
    cfg: dict,
) -> A.Affine | None:
    """shift → A.Affine."""
    tx = cfg.get("translate_x", 0.0)
    ty = cfg.get("translate_y", 0.0)
    return A.Affine(
        translate_percent={"x": (-tx, tx), "y": (-ty, ty)},
        scale=1.0,
        rotate=0.0,
        border_mode=cv2.BORDER_CONSTANT,
        fill=(128, 128, 128),
        fit_output=True,
        p=cfg.get("p", 0.5),
    )


def _build_scale(
    cfg: dict,
) -> A.Affine | None:
    """scale → A.Affine."""
    s_min = cfg["scale_min"]
    s_max = cfg["scale_max"]
    return A.Affine(
        scale=(s_min, s_max),
        rotate=0.0,
        border_mode=cv2.BORDER_CONSTANT,
        fill=(128, 128, 128),
        fit_output=True,
        p=cfg.get("p", 0.5),
    )


def _build_brightness_contrast(
    cfg: dict,
) -> A.RandomBrightnessContrast | None:
    """brightness_contrast → A.RandomBrightnessContrast."""
    return A.RandomBrightnessContrast(
        brightness_limit=cfg.get("brightness_limit", 0.1),
        contrast_limit=cfg.get("contrast_limit", 0.1),
        p=cfg.get("p", 0.5),
    )


def _build_motion_blur(
    cfg: dict,
) -> A.MotionBlur | None:
    """motion_blur → A.MotionBlur."""
    return A.MotionBlur(
        blur_limit=cfg.get("blur_limit", 3),
        p=cfg.get("p", 0.5),
    )


def _build_gauss_noise(
    cfg: dict,
) -> A.GaussNoise | None:
    """gauss_noise → A.GaussNoise."""
    try:
        return A.GaussNoise(p=cfg.get("p", 0.5))
    except AttributeError:
        # Older albumentations version without GaussNoise
        return None


def _build_jpeg_compression(
    cfg: dict,
) -> A.ImageCompression | None:
    """jpeg_compression → A.ImageCompression."""
    try:
        quality_range = cfg.get("quality_range", [85, 100])
        return A.ImageCompression(
            quality_range=quality_range,
            p=cfg.get("p", 0.5),
        )
    except AttributeError:
        # Older albumentations version without this parameter
        return None


def _build_perspective(
    cfg: dict,
) -> A.Perspective | None:
    """perspective → A.Perspective."""
    try:
        scale = cfg.get("scale", 0.05)
        if isinstance(scale, (list, tuple)):
            scale = tuple(scale)
        return A.Perspective(
            scale=scale,
            border_mode=cv2.BORDER_CONSTANT,
            fill=(128, 128, 128),
            fit_output=True,
            p=cfg.get("p", 0.5),
        )
    except AttributeError:
        # Older albumentations version without Perspective
        return None


def _build_coarse_dropout(
    cfg: dict,
    preset: str = "default",
) -> A.CoarseDropout | None:
    """coarse_dropout / dirt_spots → A.CoarseDropout."""
    defaults = {
        "default": {
            "max_holes": 8,
            "max_height": 8,
            "max_width": 8,
        },
        "dirt": {
            "max_holes": 3,
            "max_height": 4,
            "max_width": 4,
        },
    }
    d = defaults.get(preset, defaults["default"])
    max_holes = cfg.get("max_holes", d["max_holes"])
    max_height = cfg.get("max_height", d["max_height"])
    max_width = cfg.get("max_width", d["max_width"])
    return A.CoarseDropout(
        num_holes_range=(1, max_holes),
        hole_height_range=(1, max_height),
        hole_width_range=(1, max_width),
        p=cfg.get("p", 0.5),
    )


def _build_dirt_spots(
    cfg: dict,
) -> A.CoarseDropout | None:
    """dirt_spots → A.CoarseDropout (alias)."""
    return _build_coarse_dropout(cfg, preset="dirt")


def _build_defocus_blur(
    cfg: dict,
) -> A.Defocus | None:
    """defocus_blur → A.Defocus."""
    try:
        blur = cfg.get("blur_limit", 3)
        return A.Defocus(
            radius=(blur, blur),
            alias_blur=(0.1, 0.5),
            p=cfg.get("p", 0.5),
        )
    except AttributeError:
        # Older albumentations version without Defocus
        return None


def _build_affine(
    cfg: dict,
) -> A.Affine | None:
    """affine → A.Affine (shear + translate)."""
    try:
        return A.Affine(
            shear={
                "x": cfg.get("shear_x", (-9, 9)),
                "y": 0,
            },
            translate_percent={
                "x": cfg.get("translate_percent_x", (-0.09, 0.09)),
            },
            border_mode=cv2.BORDER_CONSTANT,
            fill=(128, 128, 128),
            fit_output=True,
            p=cfg.get("p", 0.45),
        )
    except AttributeError:
        # Older albumentations version without Affine shear
        return None


def _build_elastic(
    cfg: dict,
) -> A.ElasticTransform | None:
    """elastic → A.ElasticTransform."""
    try:
        return A.ElasticTransform(
            alpha=cfg.get("alpha", 1),
            sigma=cfg.get("sigma", 50),
            border_mode=cv2.BORDER_CONSTANT,
            fill=(128, 128, 128),
            p=cfg.get("p", 0.09),
        )
    except AttributeError:
        # Older albumentations version without ElasticTransform
        return None
