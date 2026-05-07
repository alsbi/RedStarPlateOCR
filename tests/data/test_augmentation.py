"""Тесты для AugmentationBuilder."""

from unittest.mock import patch

import albumentations as A

from redstar_plate_ocr.data.augmentation import (
    _build_affine,
    _build_elastic,
    _build_perspective,
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
