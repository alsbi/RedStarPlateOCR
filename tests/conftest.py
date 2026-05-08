"""Shared test fixtures."""

import pathlib

import pytest

from redstar_plate_ocr.plate.config import (
    PlateConfig,
)


def _load_plate_config() -> PlateConfig:
    """Load production plate config from YAML."""
    yaml_path = pathlib.Path(__file__).parent.parent / "configs" / "plate.yaml"
    with yaml_path.open("r", encoding="utf-8") as fh:
        return PlateConfig.from_yaml_string(fh.read())


@pytest.fixture
def plate_config() -> PlateConfig:
    """Create a PlateConfig from production YAML for testing."""
    return _load_plate_config()
