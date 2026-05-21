"""Tests for smart conditional image enhancement.

Covers :class:`QualityAssessor` (blur/contrast/size thresholds) and
:class:`SmartEnhancer` (upscale + CLAHE + unsharp).
"""

from __future__ import annotations

import numpy as np
import pytest

from redstar_plate_ocr.pipeline.enhancement import (
    QualityAssessor,
    SmartEnhancer,
    build_enhancement_stack,
)
from redstar_plate_ocr.pipeline.preprocess import PreprocessPipeline

# -----------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------


@pytest.fixture
def assessor() -> QualityAssessor:
    return QualityAssessor(
        blur_threshold=80.0,
        contrast_threshold=0.25,
        min_side_threshold=40,
    )


@pytest.fixture
def enhancer() -> SmartEnhancer:
    return SmartEnhancer(
        target_min_side=40,
        max_scale=4.0,
        clahe_clip_limit=2.0,
        sharpen_alpha=1.3,
    )


@pytest.fixture
def micro_crop() -> np.ndarray:
    """A 20×100 RGB crop that should trigger the micro-crop rule."""
    return np.random.randint(0, 255, size=(20, 100, 3), dtype=np.uint8)


@pytest.fixture
def large_sharp_crop() -> np.ndarray:
    """A 80×200 RGB crop that is sharp and well-contrasted."""
    # Use a checkerboard pattern for natural high variance
    crop = np.zeros((80, 200, 3), dtype=np.uint8)
    crop[:40, :100] = 255
    crop[40:, 100:] = 255
    return crop


# -----------------------------------------------------------------
# QualityAssessor
# -----------------------------------------------------------------


class TestQualityAssessor:
    def test_micro_crop_always_enhanced(
        self, assessor: QualityAssessor, micro_crop: np.ndarray
    ) -> None:
        assert assessor.needs_enhancement(micro_crop) is True

    def test_large_sharp_crop_not_enhanced(
        self,
        assessor: QualityAssessor,
        large_sharp_crop: np.ndarray,
    ) -> None:
        assert assessor.needs_enhancement(large_sharp_crop) is False

    def test_blurry_crop_needs_enhancement(
        self, assessor: QualityAssessor
    ) -> None:
        # Gaussian blur kills Laplacian variance
        sharp = np.random.randint(0, 255, size=(60, 160, 3), dtype=np.uint8)
        import cv2

        blurry = cv2.GaussianBlur(sharp, (15, 15), 5.0)
        assert assessor.needs_enhancement(blurry) is True

    def test_low_contrast_crop_needs_enhancement(
        self, assessor: QualityAssessor
    ) -> None:
        # Uniform mid-grey → zero contrast
        flat = np.full((60, 160, 3), 128, dtype=np.uint8)
        assert assessor.needs_enhancement(flat) is True

    def test_describe_returns_string(
        self, assessor: QualityAssessor, micro_crop: np.ndarray
    ) -> None:
        desc = assessor.describe(micro_crop)
        assert "size=20x100" in desc
        assert "blur=" in desc
        assert "contrast=" in desc


# -----------------------------------------------------------------
# SmartEnhancer
# -----------------------------------------------------------------


class TestSmartEnhancer:
    def test_upscales_micro_crop(self, enhancer: SmartEnhancer) -> None:
        crop = np.random.randint(0, 255, size=(20, 60, 3), dtype=np.uint8)
        enhanced = enhancer.enhance(crop)
        assert enhanced.shape[0] >= 40  # upscaled to 40+ px
        assert enhanced.shape[1] >= 120
        assert enhanced.dtype == np.uint8

    def test_no_upscale_for_large_crop(self, enhancer: SmartEnhancer) -> None:
        crop = np.random.randint(0, 255, size=(80, 200, 3), dtype=np.uint8)
        enhanced = enhancer.enhance(crop)
        assert enhanced.shape == (80, 200, 3)

    def test_clahe_changes_l_channel(self, enhancer: SmartEnhancer) -> None:
        # Create a low-contrast image
        crop = np.full((60, 160, 3), 128, dtype=np.uint8)
        crop[:30, :80] = 130  # very subtle difference
        enhanced = enhancer.enhance(crop)
        # Image should change (CLAHE should increase contrast)
        assert np.any(enhanced != crop)

    def test_output_clamped_uint8(self, enhancer: SmartEnhancer) -> None:
        crop = np.full((20, 50, 3), 255, dtype=np.uint8)
        enhanced = enhancer.enhance(crop)
        assert enhanced.max() <= 255
        assert enhanced.min() >= 0
        assert enhanced.dtype == np.uint8


# -----------------------------------------------------------------
# build_enhancement_stack
# -----------------------------------------------------------------


class TestBuildEnhancementStack:
    def test_none_returns_both_none(self) -> None:
        a, e = build_enhancement_stack(None)
        assert a is None and e is None

    def test_disabled_returns_both_none(self) -> None:
        a, e = build_enhancement_stack({"enabled": False})
        assert a is None and e is None

    def test_enabled_returns_instances(self) -> None:
        cfg = {
            "enabled": True,
            "blur_threshold": 90.0,
            "sharpen_alpha": 1.5,
        }
        a, e = build_enhancement_stack(cfg)
        assert isinstance(a, QualityAssessor)
        assert isinstance(e, SmartEnhancer)
        assert a.blur_threshold == 90.0
        assert e.sharpen_alpha == 1.5


# -----------------------------------------------------------------
# PreprocessPipeline integration
# -----------------------------------------------------------------


class TestPreprocessPipelineEnhancement:
    def test_enhancement_off_by_default(self) -> None:
        """Without config, enhancement should be disabled."""
        pipeline = PreprocessPipeline()
        assert pipeline._assessor is None
        assert pipeline._enhancer is None

    def test_enhancement_enabled_via_config(
        self, micro_crop: np.ndarray
    ) -> None:
        """Pipeline with smart enhancement should process micro-crops."""
        pipeline = PreprocessPipeline(
            enhancement_config={
                "enabled": True,
                "min_side_threshold": 40,
            }
        )
        assert pipeline._assessor is not None
        assert pipeline._enhancer is not None
        # Should not crash
        tensor, h, w = pipeline(micro_crop)
        assert tensor.shape == (3, 80, 256)
        assert 0 < h <= 80
        assert 0 < w <= 256

    def test_pipeline_without_enhancement_still_works(
        self, large_sharp_crop: np.ndarray
    ) -> None:
        """Backwards compatibility: pipeline without config works."""
        pipeline = PreprocessPipeline()
        tensor, h, w = pipeline(large_sharp_crop)
        assert tensor.shape == (3, 80, 256)

    def test_enhancement_does_not_change_canvas_size(
        self, micro_crop: np.ndarray
    ) -> None:
        """Canvas size must be the same regardless of enhancement."""
        pipeline_off = PreprocessPipeline()
        pipeline_on = PreprocessPipeline(
            enhancement_config={
                "enabled": True,
                "min_side_threshold": 40,
            }
        )
        tensor_off, _, _ = pipeline_off(micro_crop)
        tensor_on, _, _ = pipeline_on(micro_crop)
        assert tensor_off.shape == tensor_on.shape == (3, 80, 256)

    def test_enhancement_preserves_dtype(self) -> None:
        """Output must stay float32 tensor in CHW layout."""
        crop = np.random.randint(0, 255, size=(20, 80, 3), dtype=np.uint8)
        pipeline = PreprocessPipeline(
            enhancement_config={
                "enabled": True,
                "min_side_threshold": 40,
            }
        )
        tensor, _, _ = pipeline(crop)
        assert tensor.dtype == torch.float32
        assert tensor.shape[0] == 3  # channels first


# -----------------------------------------------------------------
# Edge cases & stress tests
# -----------------------------------------------------------------


class TestEdgeCases:
    def test_single_pixel_row(self) -> None:
        """Extreme aspect ratio 1×100 should not crash."""
        crop = np.random.randint(0, 255, size=(1, 100, 3), dtype=np.uint8)
        pipeline = PreprocessPipeline(
            enhancement_config={
                "enabled": True,
                "min_side_threshold": 40,
                "max_scale": 40,  # allow large scale for this test
            }
        )
        tensor, _, _ = pipeline(crop)
        assert tensor.shape == (3, 80, 256)

    def test_3x3_image(self) -> None:
        """Tiny square should not crash."""
        crop = np.random.randint(0, 255, size=(3, 3, 3), dtype=np.uint8)
        pipeline = PreprocessPipeline(
            enhancement_config={
                "enabled": True,
                "max_scale": 20,
            }
        )
        tensor, _, _ = pipeline(crop)
        assert tensor.shape == (3, 80, 256)

    def test_already_at_target_size(self) -> None:
        """Exact 40px min side should not upscale, only CLAHE."""
        crop = np.zeros((40, 160, 3), dtype=np.uint8)

        # Add some content so contrast check isn't triggered
        crop[:20, :] = 200
        crop[20:, :] = 50
        enhancer = SmartEnhancer(target_min_side=40)
        enhanced = enhancer.enhance(crop)
        # Should stay 40×160
        assert enhanced.shape == (40, 160, 3)


# need torch import for the last test
import torch  # noqa: E402
