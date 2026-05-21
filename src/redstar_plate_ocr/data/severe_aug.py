"""Модуль экстремальной аугментации и адаптивного планировщика.

Содержит пайплайн severe_aug для генерации «плохих» изображений
и класс SevereAugScheduler для управления интенсивностью аугментации
в процессе обучения.
"""

from __future__ import annotations

import cv2
import numpy as np

# --- Константы максимальных параметров (при intensity=100) ----------------

MAX_GAUSS_NOISE_SIGMA: float = 26.6
MAX_DISTORTION_AMPLITUDE: float = 0.81
MAX_GAUSSIAN_BLUR_KSIZE: int = 15
MAX_GAUSSIAN_BLUR_SIGMA: float = 2.66
MAX_MOTION_BLUR_KERNEL: int = 7
MAX_PIXEL_CORRUPTION_RATIO: float = 0.088
MAX_BLOCK_CORRUPTION_RATIO: float = 0.054
MAX_BRIGHTNESS_SHIFT: int = 52
MAX_CONTRAST_HALF_RANGE: float = 0.32
MIN_JPEG_QUALITY: int = 16


# --- Вспомогательные функции ------------------------------------------------


def _to_odd(ksize: int) -> int:
    """Гарантировать нечётное значение ≥ 1."""
    if ksize < 1:
        return 1
    return ksize if ksize % 2 == 1 else ksize + 1


def _clamp_odd(value: int, max_val: int) -> int:
    """Ограничить нечётное значение до max_val.

    Возвращает ближайшее нечётное число ≤ max_val,
    но не менее 1.
    """
    if max_val < 1:
        return 1
    value = min(value, max_val)
    if value % 2 == 0:
        value -= 1
    return max(1, value)


# --- Функции аугментации ---------------------------------------------------


def apply_brightness_contrast(
    image: np.ndarray,
    shift: float,
    contrast_half_range: float,
) -> np.ndarray:
    """Изменение яркости и контраста.

    Args:
        image: Изображение (H, W, C), BGR.
        shift: Максимальный сдвиг яркости; фактический сдвиг
            выбирается равномерно из ``[-shift, +shift]``.
        contrast_half_range: Половина диапазона множителя контраста;
            множитель выбирается из ``[1-range, 1+range]``.

    Returns:
        Изображение с изменённой яркостью/контрастом.
    """
    if contrast_half_range > 0:
        low = 1.0 - contrast_half_range
        high = 1.0 + contrast_half_range
        alpha = np.random.uniform(low, high)
    else:
        alpha = 1.0
    beta = np.random.uniform(-shift, shift) if shift > 0 else 0.0
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def apply_gaussian_noise(
    image: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Добавление гауссова шума.

    Args:
        image: Изображение (H, W, C), BGR.
        sigma: Стандартное отклонение шума.

    Returns:
        Зашумлённое изображение.
    """
    if sigma <= 0:
        return image
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_light_distortion(
    image: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    """Лёгкие оптические искривления через ``cv2.remap``.

    Args:
        image: Изображение (H, W, C), BGR.
        amplitude: Амплитуда синусоидального искажения в пикселях.

    Returns:
        Искажённое изображение.
    """
    if amplitude <= 0:
        return image
    h, w = image.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)

    freq_y = 2 * np.pi / h * np.random.uniform(2, 5)
    freq_x = 2 * np.pi / w * np.random.uniform(2, 5)
    phase_y = np.random.uniform(0, 2 * np.pi)
    phase_x = np.random.uniform(0, 2 * np.pi)

    map_x = x + amplitude * np.sin(freq_y * y + phase_x)
    map_y = y + amplitude * np.sin(freq_x * x + phase_y)

    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def apply_gaussian_blur(
    image: np.ndarray,
    ksize: int,
    sigma: float,
) -> np.ndarray:
    """Гауссово размытие.

    Args:
        image: Изображение (H, W, C), BGR.
        ksize: Размер ядра (нечётный).
        sigma: Стандартное отклонение ядра Гаусса.

    Returns:
        Размытое изображение.
    """
    if ksize <= 1 and sigma <= 0:
        return image
    h, w = image.shape[:2]
    min_dim = min(h, w)
    ksize = _to_odd(ksize)
    ksize = _clamp_odd(ksize, min_dim)
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def apply_motion_blur(
    image: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    """Горизонтальный motion blur.

    Args:
        image: Изображение (H, W, C), BGR.
        kernel_size: Размер ядра (нечётный).

    Returns:
        Размытое изображение.
    """
    if kernel_size <= 1:
        return image
    h, w = image.shape[:2]
    min_dim = min(h, w)
    kernel_size = _to_odd(kernel_size)
    kernel_size = _clamp_odd(kernel_size, min_dim)
    if kernel_size <= 1:
        return image
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def apply_pixel_corruption(
    image: np.ndarray,
    pixel_ratio: float,
    block_ratio: float,
) -> np.ndarray:
    """Пиксельные помехи: salt-and-pepper + цветовые артефакты в блоках 8×8.

    Args:
        image: Изображение (H, W, C), BGR.
        pixel_ratio: Доля повреждённых пикселей.
        block_ratio: Доля повреждённых блоков 8×8.

    Returns:
        Повреждённое изображение.
    """
    if pixel_ratio <= 0 and block_ratio <= 0:
        return image
    out = image.copy()
    h, w = out.shape[:2]

    # Salt-and-pepper
    if pixel_ratio > 0:
        num_pixels = max(1, int(h * w * pixel_ratio))
        ys = np.random.randint(0, h, size=num_pixels)
        xs = np.random.randint(0, w, size=num_pixels)
        values = np.random.randint(
            0,
            256,
            size=(num_pixels, 3),
            dtype=np.uint8,
        )
        out[ys, xs] = values

    # Цветовые артефакты в блоках 8×8
    if block_ratio > 0:
        block_h = max(1, h // 8)
        block_w = max(1, w // 8)
        num_blocks = max(1, int(block_h * block_w * block_ratio))
        by = np.random.randint(0, block_h, size=num_blocks) * 8
        bx = np.random.randint(0, block_w, size=num_blocks) * 8
        for y, x in zip(by, bx, strict=True):
            y_end = min(y + 8, h)
            x_end = min(x + 8, w)
            channel = np.random.randint(0, 3)
            shift_val = np.random.randint(-40, 41)
            patch = out[y:y_end, x:x_end, channel].astype(
                np.int16,
            )
            patch = np.clip(patch + shift_val, 0, 255).astype(
                np.uint8,
            )
            out[y:y_end, x:x_end, channel] = patch

    return out


def apply_jpeg_compression(
    image: np.ndarray,
    quality: int,
) -> np.ndarray:
    """Эмуляция JPEG-сжатия с заданным качеством.

    Args:
        image: Изображение (H, W, C), BGR.
        quality: Качество JPEG (1–100). При quality ≥ 95
            изображение возвращается без изменений.

    Returns:
        Изображение после JPEG-сжатия и декодирования.
    """
    if quality >= 95:
        return image
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, encoded = cv2.imencode(".jpg", image, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


# --- Пайплайн severe_aug ----------------------------------------------------


def severe_aug(image: np.ndarray, intensity: float) -> np.ndarray:
    """Применить пайплайн экстремальной аугментации.

    Args:
        image: numpy array (H, W, C), формат BGR.
        intensity: Интенсивность от 0.0 до 100.0.
            0 — без аугментации (возвращается копия).
            100 — максимальный эффект.

    Returns:
        Аугментированное изображение (того же размера, что и вход).
    """
    if intensity <= 0:
        return image.copy()

    img = image.copy()
    factor = intensity / 100.0

    # Масштабированные параметры с минимальными значениями
    noise_sigma = max(MAX_GAUSS_NOISE_SIGMA * factor, 1.0)
    distortion_amp = max(MAX_DISTORTION_AMPLITUDE * factor, 0.05)

    blur_ksize_raw = round(MAX_GAUSSIAN_BLUR_KSIZE * factor)
    blur_ksize = max(_to_odd(max(blur_ksize_raw, 1)), 3)
    blur_sigma = max(MAX_GAUSSIAN_BLUR_SIGMA * factor, 0.3)

    motion_kernel_raw = round(MAX_MOTION_BLUR_KERNEL * factor)
    if motion_kernel_raw <= 0:
        motion_kernel = 0
    elif motion_kernel_raw < 3:
        motion_kernel = 3
    else:
        motion_kernel = _to_odd(motion_kernel_raw)

    pixel_ratio = max(MAX_PIXEL_CORRUPTION_RATIO * factor, 0.005)
    block_ratio = max(MAX_BLOCK_CORRUPTION_RATIO * factor, 0.005)

    # Яркость/контраст НЕ масштабируются по intensity — полный диапазон
    bright_shift = MAX_BRIGHTNESS_SHIFT
    contrast_half_range = MAX_CONTRAST_HALF_RANGE

    jpeg_quality = round(95 - (95 - MIN_JPEG_QUALITY) * factor)

    # Применяем пайплайн
    img = apply_brightness_contrast(img, bright_shift, contrast_half_range)
    img = apply_gaussian_noise(img, noise_sigma)
    img = apply_light_distortion(img, distortion_amp)
    img = apply_gaussian_blur(img, blur_ksize, blur_sigma)
    if motion_kernel >= 3:
        img = apply_motion_blur(img, motion_kernel)
    img = apply_pixel_corruption(img, pixel_ratio, block_ratio)
    img = apply_jpeg_compression(img, jpeg_quality)

    return img


# --- Адаптивный планировщик -------------------------------------------------


class SevereAugScheduler:
    """Адаптивный планировщик разогрева экстремальной аугментации.

    Управляет переходом severe_aug → std_aug по мере улучшения модели.
    Активен только при ``enable_warmup=True`` в конфигурации.

    Attributes:
        severe_severity: Текущая интенсивность severe (0.0–1.0).
        std_severity: Текущая интенсивность стандартной аугментации.
        best_word_acc: Лучшая достигнутая точность по словам.
        preprocessing_enabled: Включена ли предобработка.
    """

    def __init__(
        self,
        initial_severity: float = 1.0,
        threshold_disable_severe: float = 0.3,
        severe_step: float = 0.01,
        patience_severe: int = 10,
        severe_threshold_std_start: float = 0.3,
        severe_midpoint: float = 0.15,
        early_stop_patience: int = 15,
    ) -> None:
        self.severe_severity: float = initial_severity
        self.std_severity: float = 0.0
        self.best_word_acc: float = 0.0
        self.threshold_disable_severe: float = threshold_disable_severe
        self.severe_step: float = severe_step
        self.patience_severe: int = patience_severe
        self.epochs_without_improvement: int = 0
        self.preprocessing_enabled: bool = False
        self.early_stop_patience: int = early_stop_patience
        self.early_stop_counter: int = 0

        # Параметры кривой перехода
        self.severe_threshold_std_start: float = severe_threshold_std_start
        self.severe_midpoint: float = severe_midpoint

    def update_schedule(self, word_acc: float, epoch: int) -> None:
        """Обновить расписание после эпохи валидации.

        Логика:

        1. Если word_acc улучшился → уменьшить severe_severity на
           ``severe_step``.
        2. Если нет улучшения ``patience_severe`` эпох → принудительно
           уменьшить.
        3. Если best_word_acc >= threshold → отключить severe,
           установить std=1.0.
        4. Рассчитать std_severity на основе текущего severe_severity.
        5. Включить предобработку при std_severity >= 0.5.

        Args:
            word_acc: Точность по словам на текущей эпохе.
            epoch: Номер текущей эпохи.
        """
        # 1. Обновление best_word_acc и управление severe_severity
        if word_acc > self.best_word_acc:
            self.best_word_acc = word_acc
            self.epochs_without_improvement = 0
            if (
                self.severe_severity > 0.0
                and self.best_word_acc < self.threshold_disable_severe
            ):
                self.severe_severity = max(
                    0.0,
                    self.severe_severity - self.severe_step,
                )
        else:
            self.epochs_without_improvement += 1
            # Fallback: если severe > 0 и нет прогресса
            # patience_severe эпох
            if (
                self.severe_severity > 0.0
                and self.epochs_without_improvement >= self.patience_severe
            ):
                self.severe_severity = max(
                    0.0,
                    self.severe_severity - self.severe_step,
                )
                self.epochs_without_improvement = 0

        # Пороговое отключение severe по точности
        if self.best_word_acc >= self.threshold_disable_severe:
            self.severe_severity = 0.0
            self.std_severity = 1.0
        else:
            # 2. Расчёт std_severity на основе текущего severe_severity
            if self.severe_severity >= self.severe_threshold_std_start:
                self.std_severity = 0.0
            elif self.severe_severity >= self.severe_midpoint:
                # От 0.3 до 0.15: линейно от 0 до 0.3
                self.std_severity = 2.0 * (
                    self.severe_threshold_std_start - self.severe_severity
                )
            else:
                # От 0.15 до 0: линейно от 0.3 до 1.0
                self.std_severity = 0.3 + (0.7 / self.severe_midpoint) * (
                    self.severe_midpoint - self.severe_severity
                )

        # 3. Предобработка включается при std_severity >= 0.5
        self.preprocessing_enabled = self.std_severity >= 0.5

    def should_stop_early(self) -> bool:
        """Проверить условие раннего останова.

        Ранний останов ОТКЛЮЧЁН пока severe_severity > 0.
        После обнуления severe активируется стандартный ранний
        останов с patience=early_stop_patience.

        Returns:
            Всегда ``False``; реальная проверка через
            :meth:`check_early_stop`.
        """
        if self.severe_severity > 0:
            return False
        return False

    def check_early_stop(self, word_acc: float) -> bool:
        """Проверить, следует ли остановить обучение досрочно.

        Ранний останов активен только после отключения severe.

        Args:
            word_acc: Точность по словам на текущей эпохе.

        Returns:
            ``True``, если обучение следует остановить.
        """
        if self.severe_severity > 0:
            return False

        if word_acc > self.best_word_acc:
            self.best_word_acc = word_acc
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1

        return self.early_stop_counter >= self.early_stop_patience

    def get_intensity(self) -> float:
        """Вернуть текущую интенсивность severe_aug (шкала 0–100)."""
        return self.severe_severity * 100.0

    @property
    def active_augmentation(self) -> str:
        """Какая аугментация сейчас активна.

        Returns:
            ``'severe'``, ``'std'`` или ``'none'``.
        """
        if self.severe_severity > 0:
            return "severe"
        elif self.std_severity > 0:
            return "std"
        else:
            return "none"

    def state_dict(self) -> dict:
        """Вернуть состояние для чекпоинта."""
        return {
            "severe_severity": self.severe_severity,
            "std_severity": self.std_severity,
            "best_word_acc": self.best_word_acc,
            "epochs_without_improvement": (self.epochs_without_improvement),
            "preprocessing_enabled": self.preprocessing_enabled,
            "early_stop_counter": self.early_stop_counter,
        }

    def load_state_dict(self, state: dict) -> None:
        """Загрузить состояние из чекпоинта.

        Args:
            state: Словарь, полученный из :meth:`state_dict`.
        """
        self.severe_severity = state.get(
            "severe_severity",
            self.severe_severity,
        )
        self.std_severity = state.get(
            "std_severity",
            self.std_severity,
        )
        self.best_word_acc = state.get(
            "best_word_acc",
            self.best_word_acc,
        )
        self.epochs_without_improvement = state.get(
            "epochs_without_improvement",
            0,
        )
        self.preprocessing_enabled = state.get(
            "preprocessing_enabled",
            False,
        )
        self.early_stop_counter = state.get(
            "early_stop_counter",
            0,
        )
