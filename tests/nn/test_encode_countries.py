"""Test PlateOCRModel.encode_countries() method."""

import pytest
import torch

from redstar_plate_ocr.nn.model import PlateOCRModel
from redstar_plate_ocr.plate.config import (
    PlateConfig,
    RegionConfig,
    ValidChars,
)


@pytest.fixture
def plate_config() -> PlateConfig:
    return PlateConfig(
        regions={
            "RU": RegionConfig(
                pattern=["X000XX00"],
                valid_chars=ValidChars(
                    letters="ABEKMHOPCTYX",
                    digits="0123456789",
                ),
            ),
            "KZ": RegionConfig(
                pattern=["000XXX00"],
                valid_chars=ValidChars(
                    letters="ABCDEFGHIJKLMNOPRSTUVWXYZ",
                    digits="0123456789",
                ),
            ),
        }
    )


def test_encode_countries_known(plate_config):
    """Known countries map to correct indices (alphabetical order)."""
    model = PlateOCRModel(plate_config)
    result = model.encode_countries(["RU", "KZ"])
    assert result.dtype == torch.long
    # country_list is sorted alphabetically: ['KZ', 'RU']
    assert result[0].item() == 1  # RU is second in alphabetical order
    assert result[1].item() == 0  # KZ is first in alphabetical order


def test_encode_countries_unknown(plate_config):
    """Unknown country raises ValueError."""
    model = PlateOCRModel(plate_config)
    with pytest.raises(ValueError, match="Unknown country"):
        model.encode_countries(["XX"])


def test_encode_countries_empty(plate_config):
    """Empty list returns empty tensor."""
    model = PlateOCRModel(plate_config)
    result = model.encode_countries([])
    assert result.shape[0] == 0


def test_encode_countries_does_not_expose_internal(
    plate_config,
):
    """encode_countries does not expose _country_to_idx directly."""
    model = PlateOCRModel(plate_config)
    result = model.encode_countries(["RU"])
    # Verify it's a Tensor, not a dict
    assert isinstance(result, torch.Tensor)
