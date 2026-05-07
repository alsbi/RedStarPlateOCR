"""Тесты для single_aug_epochs и фаз аугментации."""

from redstar_plate_ocr.pipeline.training_config import TrainingConfig


def test_single_aug_epochs_default() -> None:
    """По умолчанию single_aug_epochs=3."""
    cfg = TrainingConfig()
    assert cfg.single_aug_epochs == 3


def test_single_aug_epochs_from_dict() -> None:
    """from_dict читает single_aug_epochs из конфига."""
    cfg = TrainingConfig.from_dict(
        {"training": {"single_aug_epochs": 5}},
    )
    assert cfg.single_aug_epochs == 5


def test_single_aug_epochs_from_dict_default() -> None:
    """from_dict без ключа → дефолт 3."""
    cfg = TrainingConfig.from_dict({"training": {}})
    assert cfg.single_aug_epochs == 3


def test_no_aug_epochs_default() -> None:
    """По умолчанию no_aug_epochs=5."""
    cfg = TrainingConfig()
    assert cfg.no_aug_epochs == 5


def test_num_multi_aug_default() -> None:
    """По умолчанию num_multi_aug=1."""
    cfg = TrainingConfig()
    assert cfg.num_multi_aug == 1
