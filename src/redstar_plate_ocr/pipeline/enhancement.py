"""Smart conditional image enhancement for micro-crops and low-quality images.

This module provides :class:`QualityAssessor` for lightweight image quality
evaluation (blur, contrast) and :class:`SmartEnhancer` that applies CLAHE +
mild unsharp mask only when needed.

The enhancement is deterministic — it is applied at inference *and* training
time (as part of :class:`PreprocessPipeline`) so there is no train⇔test
distribution shift.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class QualityAssessor:
    """Lightweight per-crop quality scorer.

    Decides whether a crop needs expensive enhancement steps (CLAHE,
    unsharp mask) based on:

    * **Minimum side**  – very small crops always get enhanced because
      Lanczos upscaling alone is too soft.
    * **Laplacian variance**  – low variance means the image is blurry.
    * **Michelson contrast**  – low contrast means the image is washed
      out.

    Parameters
    ----------
    blur_threshold
        Laplacian variance below which the crop is considered blurry.
    contrast_threshold
        Michelson contrast (0.0–1.0) below which the crop is considered
        low-contrast.
    min_side_threshold
        Crops whose shorter side (in pixels) is **strictly less** than this
        value are always enhanced.
    """

    def __init__(
        self,
        blur_threshold: float = 80.0,
        contrast_threshold: float = 0.25,
        min_side_threshold: int = 40,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.contrast_threshold = contrast_threshold
        self.min_side_threshold = min_side_threshold

    def needs_enhancement(self, image: NDArray[np.uint8]) -> bool:
        """Return ``True`` if the crop should be enhanced.

        The decision is based on the *minimum side* rule first (fast path).
        If the crop is large enough, lightweight blur/contrast checks are run.
        """
        h, w = image.shape[:2]
        if min(h, w) < self.min_side_threshold:
            return True

        # Fast quality metrics on luminance
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Laplacian variance (blur detection)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        blur_score = float(laplacian.var())
        if blur_score < self.blur_threshold:
            return True

        # Michelson contrast as a rough proxy for global contrast
        cmin = float(gray.min())
        cmax = float(gray.max())
        contrast = (cmax - cmin) / 255.0
        if contrast < self.contrast_threshold:
            return True

        return False

    def describe(self, image: NDArray[np.uint8]) -> str:
        """Return a human-readable quality report (useful for logging)."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        blur_score = float(laplacian.var())
        contrast = (float(gray.max()) - float(gray.min())) / 255.0
        return (
            f"size={h}x{w} min_side={min(h, w)} blur={blur_score:.1f} "
            f"contrast={contrast:.3f}"
        )


class SmartEnhancer:
    """Deterministic enhancement pipeline applied when
    :class:`QualityAssessor` signals it is needed.

    The pipeline consists of

    1. **Upscale** (if required) – bicubic to at least
       ``target_min_height × target_min_width``.
    2. **CLAHE** (in L* channel of LAB) – local contrast enhancement.
    3. **Mild unsharp mask** – edge sharpening without ringing.

    Parameters
    ----------
    target_min_side
        After optional upscaling the shorter side should be at least this
        many pixels.
    max_scale
        Clamp for the upscale factor (safety guard).
    clahe_clip_limit
        CLAHE clip-limit (higher = more aggressive contrast).
    clahe_tile_size
        CLAHE tile-grid size.
    sharpen_alpha
        Weight of the original image in the unsharp mask blend.
    sharpen_sigma
        Gaussian sigma for the unsharp mask blur kernel.
    """

    def __init__(
        self,
        target_min_side: int = 40,
        max_scale: float = 4.0,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: tuple[int, int] = (8, 8),
        sharpen_alpha: float = 1.3,
        sharpen_sigma: float = 1.0,
    ) -> None:
        self.target_min_side = target_min_side
        self.max_scale = max_scale
        self.sharpen_alpha = sharpen_alpha
        self.sharpen_sigma = sharpen_sigma
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_size,
        )

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------

    def enhance(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Apply the full enhancement pipeline to *image*.

        The image is assumed to be in RGB format and stays RGB.
        """
        img = image.copy()

        # 1. Up-scale micro-crops so downstream Lanczos has more to work with
        img = self._upscale_if_needed(img)

        # 2. Local contrast (only L* channel in LAB)
        img = self._apply_clahe(img)

        # 3. Mild sharpening
        img = self._apply_unsharp(img)

        return img

    # -----------------------------------------------------------------
    # internal helpers
    # -----------------------------------------------------------------

    def _upscale_if_needed(
        self, image: NDArray[np.uint8]
    ) -> NDArray[np.uint8]:
        h, w = image.shape[:2]
        min_side = min(h, w)
        if min_side >= self.target_min_side:
            return image

        scale = min(self.max_scale, self.target_min_side / min_side)
        new_w = round(w * scale)
        new_h = round(h * scale)
        logger.debug(
            "SmartEnhancer: upscaling %dx%d → %dx%d (scale %.2f)",
            w,
            h,
            new_w,
            new_h,
            scale,
        )
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    def _apply_clahe(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        l_channel = self._clahe.apply(l_channel)
        enhanced = cv2.cvtColor(
            cv2.merge((l_channel, a, b)), cv2.COLOR_LAB2RGB
        )
        return enhanced

    def _apply_unsharp(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        blurred = cv2.GaussianBlur(image, (0, 0), self.sharpen_sigma)
        sharpened = cv2.addWeighted(
            image,
            self.sharpen_alpha,
            blurred,
            1.0 - self.sharpen_alpha,
            0,
        )
        # Type safety: addWeighted returns float64; clamp and cast
        return np.clip(sharpened, 0, 255).astype(np.uint8)


def build_enhancement_stack(
    config: dict[str, object] | None,
) -> tuple[QualityAssessor, SmartEnhancer] | tuple[None, None]:
    """Factory used by :class:`PreprocessPipeline`.

    Parameters
    ----------
    config
        Sub-dictionary taken from ``preprocessing.smart_enhancement`` in
        the YAML config.  If ``None`` or ``{"enabled": False}`` the
        pipeline returns ``(None, None)`` and enhancement is bypassed.

    Returns
    -------
    assessor, enhancer
        Ready-to-use instances or ``(None, None)`` when disabled.
    """
    if config is None or not config.get("enabled", True):
        return None, None

    assessor = QualityAssessor(
        blur_threshold=config.get("blur_threshold", 80.0),
        contrast_threshold=config.get("contrast_threshold", 0.25),
        min_side_threshold=config.get("min_side_threshold", 40),
    )

    enhancer = SmartEnhancer(
        target_min_side=config.get("min_side_threshold", 40),
        max_scale=config.get("max_scale", 4.0),
        clahe_clip_limit=config.get("clahe_clip_limit", 2.0),
        clahe_tile_size=config.get("clahe_tile_size", (8, 8)),
        sharpen_alpha=config.get("sharpen_alpha", 1.3),
        sharpen_sigma=config.get("sharpen_sigma", 1.0),
    )

    return assessor, enhancer
