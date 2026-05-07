"""Debug visualization: save preprocessing stages per country/format."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import yaml

from redstar_plate_ocr.data.augmentation import build_single_augmentation
from redstar_plate_ocr.data.dataset import PlateDataset
from redstar_plate_ocr.nn.compression import compute_content_mask
from redstar_plate_ocr.pipeline.preprocess import PreprocessPipeline
from redstar_plate_ocr.plate.config import PlateConfig


def _denormalize(
    tensor: torch.Tensor,
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    """Convert normalized tensor back to displayable image."""
    arr = tensor.cpu().numpy().transpose(1, 2, 0)
    arr = arr * np.array(std) + np.array(mean)
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return arr


def _save_hook_factory(
    captured: dict[str, np.ndarray],
    stage: str,
) -> Callable:
    """Build a preprocessing hook that captures stage output."""
    def hook(data):
        if isinstance(data, torch.Tensor):
            captured[stage] = data.cpu().numpy()
        else:
            captured[stage] = data.copy()
    return hook


def _save_mask_vis(
    orig_h: int,
    orig_w: int,
    feat_h: int,
    feat_w: int,
    stride: int,
    canvas_h: int,
    canvas_w: int,
    out_dir: Path,
    prefix: str,
    lb: np.ndarray | None,
) -> None:
    """Save content mask visualization and overlay."""
    mask = compute_content_mask(
        torch.tensor([orig_h]),
        torch.tensor([orig_w]),
        feat_h, feat_w, stride,
    )
    mask_vis = mask[0, 0].cpu().numpy()
    mask_vis = (mask_vis * 255).astype(np.uint8)
    mask_vis = cv2.resize(
        mask_vis, (canvas_w, canvas_h),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imwrite(str(out_dir / f"{prefix}_content_mask.png"), mask_vis)
    if lb is not None and lb.ndim == 3:
        green = np.zeros_like(lb)
        green[:, :, 1] = mask_vis
        overlay = cv2.addWeighted(lb, 0.6, green, 0.4, 0)
        cv2.imwrite(
            str(out_dir / f"{prefix}_content_mask_overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )


def _save_stage_vis(
    stage: str,
    data: np.ndarray | torch.Tensor,
    tensor: torch.Tensor,
    mean: list[float],
    std: list[float],
    out_dir: Path,
    prefix: str,
) -> None:
    if stage == "tensor":
        vis = _denormalize(tensor, mean, std)
    elif isinstance(data, np.ndarray):
        vis = data
    else:
        return
    if vis.ndim == 3 and vis.shape[2] == 3:
        cv2.imwrite(
            str(out_dir / f"{prefix}_{stage}.png"),
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
        )


def _process_single_sample(
    sample: dict,
    pipeline: PreprocessPipeline,
    data_dir: str,
    out_dir: Path,
    prefix: str,
    canvas_h: int,
    canvas_w: int,
    feat_h: int,
    feat_w: int,
    stride: int,
    mean: list[float],
    std: list[float],
) -> bool:
    """Process one sample, save debug images. Returns True on success."""
    captured: dict[str, np.ndarray] = {}
    img_path = os.path.join(data_dir, sample["image_path"])
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        return False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    captured["original"] = image.copy()
    tensor, orig_h, orig_w = pipeline(image.astype(np.uint8))

    for stage, data in captured.items():
        _save_stage_vis(
            stage, data, tensor, mean, std, out_dir, prefix,
        )
    _save_mask_vis(
        orig_h, orig_w, feat_h, feat_w, stride,
        canvas_h, canvas_w, out_dir, prefix,
        captured.get("letterboxed"),
    )
    return True


def debug_samples(
    plate_config_path: str,
    config_path: str,
    augmentation_path: str | None,
    data_dir: str,
    output_dir: str = "debug/samples",
    num_per_group: int = 10,
    split: str = "train",
) -> None:
    """Save preprocessing stage images for each country × format."""
    plate_config = PlateConfig.from_yaml(plate_config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    preproc = cfg.get("preprocessing", {})
    canvas_h = preproc.get("canvas_height", 80)
    canvas_w = preproc.get("canvas_width", 192)
    pad_color = preproc.get("pad_color", 128)
    norm = preproc.get("normalization", {})
    mean = norm.get("mean", [0.485, 0.456, 0.406])
    std = norm.get("std", [0.229, 0.224, 0.225])

    aug = None
    if augmentation_path:
        with open(augmentation_path) as f:
            aug_cfg = yaml.safe_load(f)
        aug = build_single_augmentation(aug_cfg, is_train=True)

    stride = 4
    feat_h = canvas_h // stride
    feat_w = canvas_w // stride

    csv_path = os.path.join(data_dir, split, f"{split}.csv")
    dataset = PlateDataset(
        csv_path=csv_path,
        dataset_root=data_dir,
        allowed_regions=plate_config.country_list,
    )

    count: dict[str, int] = {}
    for sample in dataset.samples:
        country = sample["region"]
        fmt = sample["plate_type"]
        key = f"{country}/{fmt}"
        if count.get(key, 0) >= num_per_group:
            continue

        captured: dict[str, np.ndarray] = {}
        pipeline = PreprocessPipeline(
            canvas_height=canvas_h,
            canvas_width=canvas_w,
            pad_color=pad_color,
            mean=mean,
            std=std,
            augmentation=aug,
            hooks={
                "on_augmented": _save_hook_factory(
                    captured, "augmented"
                ),
                "on_scaled": _save_hook_factory(
                    captured, "scaled"
                ),
                "on_letterboxed": _save_hook_factory(
                    captured, "letterboxed"
                ),
                "on_normalized": _save_hook_factory(
                    captured, "tensor"
                ),
            },
        )

        out_dir = Path(output_dir) / country / fmt
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = count.get(key, 0)
        prefix = f"{idx:03d}"

        success = _process_single_sample(
            sample, pipeline, data_dir, out_dir,
            prefix, canvas_h, canvas_w, feat_h, feat_w,
            stride, mean, std,
        )
        if success:
            count[key] = idx + 1

    total = sum(count.values())
    print(f"Saved {total} sample groups to {output_dir}")
