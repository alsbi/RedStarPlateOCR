"""Тесты для PreprocessPipeline и auto_unpad."""

import numpy as np
import torch

from redstar_plate_ocr.data.transforms import PreprocessPipeline, auto_unpad


def _make_image(h: int, w: int) -> np.ndarray:
    """Создаёт RGB-изображение uint8 заданного размера."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 255, (h, w, 3), dtype=np.uint8)


class TestPreprocessPipeline:
    """Тесты пайплайна предобработки."""

    def test_standard_shape(self) -> None:
        """Изображение 400×100 → тензор (3,80,192)."""
        pipe = PreprocessPipeline()
        img = _make_image(100, 400)
        tensor, scaled_h, scaled_w = pipe(img)
        assert tensor.shape == (3, 80, 192)
        # scale = min(192/400, 80/100) = 0.48 → (48, 192)
        assert scaled_h == 48
        assert scaled_w == 192

    def test_square_shape(self) -> None:
        """Квадратное изображение 100×100 → тензор (3,80,192)."""
        pipe = PreprocessPipeline()
        img = _make_image(100, 100)
        tensor, scaled_h, scaled_w = pipe(img)
        assert tensor.shape == (3, 80, 192)
        # scale = min(192/100, 80/100) = 0.8 → (80, 80)
        assert scaled_h == 80
        assert scaled_w == 80

    def test_preserves_scaled_dims(self) -> None:
        """scaled_h и scaled_w — размеры после масштабирования."""
        pipe = PreprocessPipeline()
        img = _make_image(50, 200)
        _, scaled_h, scaled_w = pipe(img)
        # scale = min(192/200, 80/50) = 0.96 → (48, 192)
        assert scaled_h == 48
        assert scaled_w == 192

    def test_letterbox_left_top(self) -> None:
        """Изображение выровнено по левому верхнему углу."""
        pipe = PreprocessPipeline()
        # Белое изображение 50×200
        img = np.full((50, 200, 3), 255, dtype=np.uint8)
        tensor, _, _ = pipe(img)
        # После letterbox нижняя часть холста — серый
        # Проверяем, что нижний ряд — серый (pad)
        bottom_row = tensor[0, -1, :]
        assert bottom_row.mean() < 0.6  # серый ~= 0.502

    def test_normalize_range(self) -> None:
        """После нормализации значения в разумном диапазоне."""
        pipe = PreprocessPipeline()
        img = _make_image(80, 200)
        tensor, _, _ = pipe(img)
        assert tensor.dtype == torch.float32
        # Не должно быть NaN или Inf
        assert tensor.isfinite().all()

    def test_custom_mean_std(self) -> None:
        """Кастомные mean/std корректно применяются."""
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        pipe = PreprocessPipeline(mean=mean, std=std)
        img = _make_image(80, 200)
        tensor, _, _ = pipe(img)
        assert tensor.shape == (3, 80, 192)

    def test_with_augmentation(self) -> None:
        """Пайплайн работает с аугментацией (augment → scale)."""
        import albumentations as A

        aug = A.Compose([A.Rotate(limit=3, p=1.0)])
        pipe = PreprocessPipeline(augmentation=aug)
        img = _make_image(100, 400)
        tensor, scaled_h, scaled_w = pipe(img)
        assert tensor.shape == (3, 80, 192)
        # Rotate на полном размере, затем scale — размеры те же
        assert scaled_h == 48
        assert scaled_w == 192

    def test_augment_before_scale(self) -> None:
        """Аугментация применяется до масштабирования."""
        import albumentations as A

        # Аугментация, которая точно меняет изображение
        aug = A.Compose([A.HorizontalFlip(p=1.0)])
        pipe = PreprocessPipeline(augmentation=aug)
        img = _make_image(100, 400)
        tensor, scaled_h, scaled_w = pipe(img)
        assert tensor.shape == (3, 80, 192)
        # После flip+scale размеры те же, что и без аугментации
        assert scaled_h == 48
        assert scaled_w == 192

    def test_no_augmentation_same_as_before(self) -> None:
        """Без аугментации порядок augment→scale≡scale."""
        pipe = PreprocessPipeline()
        img = _make_image(100, 400)
        tensor, scaled_h, scaled_w = pipe(img)
        assert tensor.shape == (3, 80, 192)
        assert scaled_h == 48
        assert scaled_w == 192

    def test_scale_ringing_replaced_with_pad_color(self) -> None:
        """LANCZOS4 ringing у границ серого → точный pad_color."""
        # Изображение: серый 128, но с контрастной полосой
        img = np.full((80, 192, 3), 128, dtype=np.uint8)
        # Белая полоса в центре — на границе будет ringing
        img[30:50, 60:130, :] = 255
        pipe = PreprocessPipeline(pad_color=128)
        scaled = pipe._scale(img)
        # Все пиксели со значением близким к 128 должны быть точно 128
        near_128 = np.abs(scaled.astype(np.int16) - 128) < 20
        all_near_128_gray = np.all(near_128, axis=2)
        near_pixels = scaled[all_near_128_gray]
        if len(near_pixels) > 0:
            assert np.all(near_pixels == 128), (
                "Ringing pixels near pad_color should be exact 128"
            )

    def test_scale_preserves_content_pixels(self) -> None:
        """Масштабирование сохраняет контент-пиксели (не серые)."""
        img = np.full((80, 192, 3), 128, dtype=np.uint8)
        img[30:50, 60:130, :] = 255
        pipe = PreprocessPipeline(pad_color=128)
        scaled = pipe._scale(img)
        # Белые пиксели в центре должны остаться яркими
        center = scaled[35:45, 70:120, :]
        assert center.mean() > 200, "Content pixels should be preserved"


class TestAutoUnpad:
    """Тесты для auto_unpad — автообрезка серого паддинга."""

    def test_noop_on_raw_plate(self) -> None:
        """Raw plate crop (no padding) passes through unchanged."""
        rng = np.random.RandomState(0)
        img = rng.randint(0, 255, (30, 117, 3), dtype=np.uint8)
        result = auto_unpad(img)
        np.testing.assert_array_equal(result, img)

    def test_strips_gray_padding(self) -> None:
        """Pre-canvased 256×256 with gray 114 padding is stripped."""
        canvas = np.full((256, 256, 3), 114, dtype=np.uint8)
        # Place a plate-like content in the center
        rng = np.random.RandomState(1)
        canvas[100:156, :, :] = rng.randint(
            0, 255, (56, 256, 3), dtype=np.uint8
        )
        result = auto_unpad(canvas)
        assert result.shape == (56, 256, 3)

    def test_strips_different_pad_color(self) -> None:
        """Works with any uniform pad color (e.g. white 255)."""
        canvas = np.full((200, 200, 3), 255, dtype=np.uint8)
        rng = np.random.RandomState(2)
        canvas[40:80, 30:170, :] = rng.randint(
            0, 200, (40, 140, 3), dtype=np.uint8
        )
        result = auto_unpad(canvas)
        assert result.shape[0] <= 80  # at most content height + small margins
        assert result.shape[1] <= 170

    def test_preserves_content(self) -> None:
        """Content pixels are preserved after unpadding."""
        canvas = np.full((256, 256, 3), 114, dtype=np.uint8)
        rng = np.random.RandomState(3)
        content = rng.randint(0, 255, (56, 256, 3), dtype=np.uint8)
        canvas[100:156, :, :] = content
        result = auto_unpad(canvas)
        np.testing.assert_array_equal(result, content)

    def test_asymmetric_padding(self) -> None:
        """Works when padding is only on top/bottom (no left/right)."""
        canvas = np.full((200, 100, 3), 100, dtype=np.uint8)
        rng = np.random.RandomState(4)
        canvas[50:150, :, :] = rng.randint(
            0, 255, (100, 100, 3), dtype=np.uint8
        )
        result = auto_unpad(canvas)
        assert result.shape == (100, 100, 3)

    def test_tiny_image_noop(self) -> None:
        """Very small images (< 8px) are returned as-is."""
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        result = auto_unpad(img)
        np.testing.assert_array_equal(result, img)

    def test_corner_mismatch_noop(self) -> None:
        """If corners are not the same color, no unpadding."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        # Make one corner different
        img[0, 0] = [0, 0, 0]
        result = auto_unpad(img)
        np.testing.assert_array_equal(result, img)


class TestPreprocessWithAutoUnpad:
    """Тесты что PreprocessPipeline корректно работает с auto_unpad."""

    def test_raw_plate_unchanged(self) -> None:
        """Raw plate crop produces same result with auto_unpad."""
        pipe = PreprocessPipeline()
        rng = np.random.RandomState(10)
        img = rng.randint(0, 255, (30, 117, 3), dtype=np.uint8)
        tensor, h, w = pipe(img)
        assert tensor.shape == (3, 80, 192)
        assert h == 49
        assert w == 192

    def test_pre_canvased_image_stripped(self) -> None:
        """Pre-canvased image stripped with correct content_mask dims."""
        pipe = PreprocessPipeline()
        # Simulate dataset2-style image: 256x256 with gray 114 padding
        canvas = np.full((256, 256, 3), 114, dtype=np.uint8)
        rng = np.random.RandomState(11)
        # Place a plate-like rectangle
        canvas[100:156, :, :] = rng.randint(
            0, 255, (56, 256, 3), dtype=np.uint8
        )
        tensor, h, w = pipe(canvas)
        assert tensor.shape == (3, 80, 192)
        # After unpad: 56x256, scale=min(192/256,80/56)
        # = min(0.75, 1.428) = 0.75
        # So 56*0.75=42, 256*0.75=192
        assert h == 42
        assert w == 192

    def test_pre_canvased_same_as_raw(self) -> None:
        """Pre-canvased and raw images of the same plate produce same dims."""
        pipe = PreprocessPipeline()
        rng = np.random.RandomState(12)
        # Raw plate
        raw = rng.randint(0, 255, (56, 256, 3), dtype=np.uint8)
        _, h_raw, w_raw = pipe(raw)

        # Same plate on canvas
        canvas = np.full((256, 256, 3), 114, dtype=np.uint8)
        canvas[100:156, :, :] = raw
        _, h_canvas, w_canvas = pipe(canvas)

        assert h_raw == h_canvas
        assert w_raw == w_canvas
