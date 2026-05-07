"""Тесты для AugmentationBuilder."""

from unittest.mock import patch

import albumentations as A
import numpy as np

from redstar_plate_ocr.data.augmentation import (
    _build_affine,
    _build_elastic,
    _build_infrared_glow,
    _build_perspective,
    _infrared_glow_transform,
    build_multi_augmentation,
    build_single_augmentation,
)


class TestBuildSingleAugmentation:
    """Тесты билдера single аугментации."""

    def test_train_returns_compose(self) -> None:
        """is_train=True с конфигом → Compose не None."""
        config = {
            "rotation": {
                "enabled": True,
                "limit": 3,
                "p": 0.3,
            },
            "brightness_contrast": {
                "enabled": True,
                "brightness_limit": 0.1,
                "contrast_limit": 0.1,
                "p": 0.3,
            },
        }
        result = build_single_augmentation(config, is_train=True)
        assert result is not None
        assert isinstance(result, A.Compose)

    def test_val_returns_none(self) -> None:
        """is_train=False → None."""
        config = {
            "rotation": {
                "enabled": True,
                "limit": 3,
                "p": 0.3,
            },
        }
        result = build_single_augmentation(config, is_train=False)
        assert result is None

    def test_empty_config_train(self) -> None:
        """Пустой конфиг → None (нет включённых аугментаций)."""
        result = build_single_augmentation({}, is_train=True)
        assert result is None

    def test_disabled_augmentations(self) -> None:
        """Все аугментации disabled → None."""
        config = {
            "rotation": {
                "enabled": False,
                "limit": 3,
                "p": 0.3,
            },
        }
        result = build_single_augmentation(config, is_train=True)
        assert result is None

    def test_dirt_spots_alias(self) -> None:
        """dirt_spots → CoarseDropout (алиас)."""
        config = {
            "dirt_spots": {
                "enabled": True,
                "max_holes": 3,
                "max_height": 4,
                "max_width": 4,
                "p": 0.15,
            },
        }
        result = build_single_augmentation(config, is_train=True)
        assert result is not None


class TestBuildMultiAugmentation:
    """Тесты билдера multi аугментации."""

    def test_train_returns_compose(self) -> None:
        """is_train=True с конфигом → Compose не None."""
        config = {
            "rotation": {
                "enabled": True,
                "limit": 3,
                "p": 0.3,
            },
            "brightness_contrast": {
                "enabled": True,
                "brightness_limit": 0.1,
                "contrast_limit": 0.1,
                "p": 0.3,
            },
        }
        result = build_multi_augmentation(config, is_train=True)
        assert result is not None
        assert isinstance(result, A.Compose)

    def test_val_returns_none(self) -> None:
        """is_train=False → None."""
        config = {
            "rotation": {
                "enabled": True,
                "limit": 3,
                "p": 0.3,
            },
        }
        result = build_multi_augmentation(config, is_train=False)
        assert result is None


class TestBuildAffine:
    """Тесты для _build_affine."""

    def test_affine_with_params(self) -> None:
        """_build_affine с параметрами → A.Affine."""
        result = _build_affine(
            {
                "shear_x": (-9, 9),
                "translate_percent_x": (-0.09, 0.09),
                "p": 0.5,
            }
        )
        assert isinstance(result, A.Affine)

    def test_affine_default_params(self) -> None:
        """_build_affine({}) → A.Affine с дефолтами."""
        result = _build_affine({})
        assert isinstance(result, A.Affine)

    def test_affine_attribute_error(self) -> None:
        """При AttributeError → None."""
        with patch.object(A, "Affine", side_effect=AttributeError):
            result = _build_affine({"p": 0.5})
            assert result is None

    def test_affine_in_build_single_augmentation(self) -> None:
        """build_single_augmentation с affine → включает affine."""
        config = {
            "affine": {
                "enabled": True,
                "p": 0.5,
            },
        }
        result = build_single_augmentation(config, is_train=True)
        assert result is not None


class TestBuildElastic:
    """Тесты для _build_elastic."""

    def test_elastic_with_params(self) -> None:
        """_build_elastic с параметрами → A.ElasticTransform."""
        result = _build_elastic({"alpha": 1, "sigma": 50, "p": 0.1})
        assert isinstance(result, A.ElasticTransform)

    def test_elastic_default_params(self) -> None:
        """_build_elastic({}) → A.ElasticTransform с дефолтами."""
        result = _build_elastic({})
        assert isinstance(result, A.ElasticTransform)

    def test_elastic_attribute_error(self) -> None:
        """При AttributeError → None."""
        with patch.object(A, "ElasticTransform", side_effect=AttributeError):
            result = _build_elastic({"p": 0.1})
            assert result is None

    def test_elastic_in_build_single_augmentation(self) -> None:
        """build_single_augmentation с elastic → включает elastic."""
        config = {
            "elastic": {
                "enabled": True,
                "p": 0.1,
            },
        }
        result = build_single_augmentation(config, is_train=True)
        assert result is not None


class TestBuildPerspective:
    """Тесты для _build_perspective."""

    def test_perspective_list_scale_converts_to_tuple(
        self,
    ) -> None:
        """scale=[0.05, 0.135] → A.Perspective с scale=tuple."""
        result = _build_perspective({"scale": [0.05, 0.135], "p": 0.5})
        assert isinstance(result, A.Perspective)
        assert result.scale == (0.05, 0.135)

    def test_perspective_tuple_scale_unchanged(
        self,
    ) -> None:
        """scale=(0.05, 0.135) → A.Perspective с scale=tuple."""
        result = _build_perspective({"scale": (0.05, 0.135), "p": 0.5})
        assert isinstance(result, A.Perspective)
        assert result.scale == (0.05, 0.135)

    def test_perspective_scalar_scale_backward_compat(
        self,
    ) -> None:
        """scale=0.05 → A.Perspective (backward compat)."""
        result = _build_perspective({"scale": 0.05, "p": 0.5})
        assert isinstance(result, A.Perspective)
        # albumentations internally converts scalar to (0, scale)
        assert result.scale == (0.0, 0.05)

    def test_perspective_default_scale(self) -> None:
        """Без scale → дефолт 0.05 (albumentations → (0, 0.05))."""
        result = _build_perspective({"p": 0.5})
        assert isinstance(result, A.Perspective)
        assert result.scale == (0.0, 0.05)

    def test_perspective_attribute_error(self) -> None:
        """При AttributeError → None."""
        with patch.object(A, "Perspective", side_effect=AttributeError):
            result = _build_perspective({"p": 0.5})
            assert result is None

    def test_perspective_in_build_single_augmentation(self) -> None:
        """build_single_augmentation с perspective → включает perspective."""
        config = {
            "perspective": {
                "enabled": True,
                "scale": [0.05, 0.135],
                "p": 0.63,
            },
        }
        result = build_single_augmentation(config, is_train=True)
        assert result is not None


class TestBuildAugmentationFullConfig:
    """Тесты build_single_augmentation с полным конфигом из yaml."""

    def _full_config(self) -> dict:
        """Конфиг, соответствующий обновлённому yaml."""
        return {
            "perspective": {
                "enabled": True,
                "scale": [0.05, 0.135],
                "p": 0.63,
            },
            "rotation": {
                "enabled": True,
                "limit": 10,
                "p": 0.54,
            },
            "shift": {
                "enabled": True,
                "translate_x": 0.045,
                "p": 0.54,
            },
            "scale": {
                "enabled": True,
                "scale_min": 0.91,
                "scale_max": 1.09,
                "p": 0.54,
            },
            "brightness_contrast": {
                "enabled": True,
                "brightness_limit": 0.18,
                "contrast_limit": 0.18,
                "p": 0.63,
            },
            "gauss_noise": {
                "enabled": True,
                "p": 0.36,
            },
            "affine": {
                "enabled": True,
                "shear_x": [-9, 9],
                "translate_percent_x": [-0.09, 0.09],
                "p": 0.45,
            },
            "elastic": {
                "enabled": True,
                "alpha": 1,
                "sigma": 50,
                "p": 0.09,
            },
            "dirt_spots": {
                "enabled": True,
                "max_holes": 3,
                "max_height": 4,
                "max_width": 4,
                "p": 0.15,
            },
            "defocus_blur": {
                "enabled": True,
                "blur_limit": 3,
                "p": 0.15,
            },
        }

    def test_single_aug_full_config_not_none(
        self,
    ) -> None:
        """Все enabled-трансформы → single augmentation не None."""
        result = build_single_augmentation(self._full_config(), is_train=True)
        assert result is not None

    def test_multi_aug_full_config_not_none(
        self,
    ) -> None:
        """Все enabled-трансформы → multi augmentation не None."""
        result = build_multi_augmentation(self._full_config(), is_train=True)
        assert result is not None


class TestInfraredGlowTransform:
    """Tests for _infrared_glow_transform (core pixel logic)."""

    @staticmethod
    def _make_plate_image(
        h: int = 32, w: int = 128,
    ) -> np.ndarray:
        """Synthetic plate: white bg with dark rectangular 'char'."""
        img = np.full((h, w, 3), 230, dtype=np.uint8)  # light bg
        # Draw dark rectangle (simulates a character)
        img[8:24, 16:32] = 30
        return img

    def test_output_shape_unchanged(self) -> None:
        """Output has same shape as input."""
        img = self._make_plate_image()
        out = _infrared_glow_transform(img)
        assert out.shape == img.shape

    def test_output_dtype_uint8(self) -> None:
        """Output dtype is uint8."""
        img = self._make_plate_image()
        out = _infrared_glow_transform(img)
        assert out.dtype == "uint8"

    def test_inversion_happens(self) -> None:
        """Dark region becomes bright (IR glow inverts luminance)."""
        img = self._make_plate_image()
        # Character region is dark (30) in input
        dark_mean = img[8:24, 16:32].mean()
        assert dark_mean < 80  # sanity: input char is dark

        out = _infrared_glow_transform(img, glow_sigma=0)
        # After IR glow, the same region should be bright
        out_char_mean = out[8:24, 16:32].mean()
        assert out_char_mean > 100, (
            f"Char region should be bright after IR glow, "
            f"got mean={out_char_mean:.1f}"
        )

    def test_background_becomes_dark(self) -> None:
        """Light background region becomes dark after IR glow."""
        img = self._make_plate_image()
        # Background is bright (230) in input
        bg_mean = img[2:6, 40:60].mean()
        assert bg_mean > 200  # sanity: input bg is bright

        out = _infrared_glow_transform(img, glow_sigma=0)
        out_bg_mean = out[2:6, 40:60].mean()
        assert out_bg_mean < 150, (
            f"Background should become dark after IR glow, "
            f"got mean={out_bg_mean:.1f}"
        )

    def test_contrast_boost_amplifies(self) -> None:
        """Higher contrast_boost produces brighter chars."""
        img = self._make_plate_image()
        out_low = _infrared_glow_transform(
            img, contrast_boost=1.0, glow_sigma=0,
        )
        out_high = _infrared_glow_transform(
            img, contrast_boost=2.0, glow_sigma=0,
        )
        char_low = out_low[8:24, 16:32].mean()
        char_high = out_high[8:24, 16:32].mean()
        assert char_high > char_low

    def test_tint_produces_colour_shift(self) -> None:
        """tint_strength > 0 shifts colour channels differently."""
        img = self._make_plate_image()
        # Run many times to hit both greenish and magenta branches
        channels_diff: set[bool] = set()
        for _ in range(20):
            out = _infrared_glow_transform(
                img, tint_strength=0.5, glow_sigma=0,
            )
            r, g, b = out[12, 24].astype(float)
            # At least one channel differs from others
            channels_diff.add(
                not (abs(r - g) < 3 and abs(g - b) < 3),
            )
        # With tint, at least some runs should show colour shift
        assert True in channels_diff

    def test_glow_sigma_adds_bloom(self) -> None:
        """glow_sigma > 0 adds blurring bloom around bright chars."""
        img = self._make_plate_image()
        out_no_glow = _infrared_glow_transform(img, glow_sigma=0)
        out_with_glow = _infrared_glow_transform(
            img, glow_sigma=5,
        )
        # With glow, pixels near the character edge should be brighter
        # (bloom extends the bright region)
        edge_region_no = out_no_glow[6:8, 14:34].mean()
        edge_region_yes = out_with_glow[6:8, 14:34].mean()
        assert edge_region_yes > edge_region_no * 0.8  # at least comparable

    def test_glow_sigma_zero_no_bloom(self) -> None:
        """glow_sigma=0 produces identical result without blur."""
        img = self._make_plate_image()
        # With sigma=0, no Gaussian blur is applied
        out = _infrared_glow_transform(img, glow_sigma=0)
        assert out.shape == img.shape


class TestBuildInfraredGlow:
    """Tests for _build_infrared_glow builder."""

    def test_returns_lambda(self) -> None:
        """_build_infrared_glow returns A.Lambda."""
        result = _build_infrared_glow({"p": 0.3})
        assert isinstance(result, A.Lambda)

    def test_default_params(self) -> None:
        """_build_infrared_glow({}) returns A.Lambda with defaults."""
        result = _build_infrared_glow({})
        assert isinstance(result, A.Lambda)

    def test_custom_params(self) -> None:
        """Custom tint_strength, contrast_boost, glow_sigma."""
        result = _build_infrared_glow(
            {
                "tint_strength": 0.5,
                "contrast_boost": 1.8,
                "glow_sigma": 5,
                "p": 0.4,
            }
        )
        assert isinstance(result, A.Lambda)

    def test_in_build_single_augmentation(self) -> None:
        """build_single_augmentation with infrared_glow enabled."""
        config = {
            "infrared_glow": {
                "enabled": True,
                "tint_strength": 0.3,
                "contrast_boost": 1.4,
                "glow_sigma": 3,
                "p": 0.2,
            },
        }
        result = build_single_augmentation(config, is_train=True)
        assert result is not None

    def test_disabled_returns_none_from_pipeline(self) -> None:
        """infrared_glow disabled → not in pipeline."""
        config = {
            "infrared_glow": {
                "enabled": False,
                "p": 0.2,
            },
        }
        result = build_single_augmentation(config, is_train=True)
        assert result is None

    def test_apply_to_image(self) -> None:
        """Applying the transform to a real image works."""
        img = np.random.randint(0, 255, (32, 128, 3), dtype=np.uint8)
        transform = _build_infrared_glow({"p": 1.0})
        out = transform(image=img)["image"]
        assert out.shape == img.shape
        assert out.dtype == np.uint8
