"""Shared test fixtures."""

import pytest

from redstar_plate_ocr.plate.config import (
    PlateConfig,
    RegionConfig,
    ValidChars,
)


@pytest.fixture
def plate_config() -> PlateConfig:
    """Create a PlateConfig with all enabled countries for testing.

    Matches the structure of configs/plate.yaml but without
    forbidden_combos (not needed by most unit tests).
    """
    return PlateConfig(
        regions={
            "RU": RegionConfig(
                pattern=["X000XX00o"],
                valid_chars=ValidChars(
                    letters="ABEKMHOPCTYX",
                    digits="0123456789",
                ),
            ),
            "KZ": RegionConfig(
                pattern=["000XXX00", "00000XXX"],
                valid_chars=ValidChars(
                    letters="ABCDEFGHIJKLMNOPRSTUVWXYZ",
                    digits="0123456789",
                ),
            ),
            "BY": RegionConfig(
                pattern=["0000XX-0"],
                valid_chars=ValidChars(
                    letters="ABEKMHOPCTYX-",
                    digits="0123456789",
                ),
            ),
            "UA": RegionConfig(
                pattern=["XX0000XX"],
                valid_chars=ValidChars(
                    letters="ABEKMHOPCTYX",
                    digits="0123456789",
                ),
            ),
            "UZ": RegionConfig(
                pattern=["00X000XX", "X000XX00"],
                valid_chars=ValidChars(
                    letters="ABCDEFGHIJKLMNOPRSTUVWXYZ",
                    digits="0123456789",
                ),
            ),
            "KG": RegionConfig(
                pattern=["X0000XX", "00000XXX"],
                valid_chars=ValidChars(
                    letters="ABCDEFGHIJKLMNOPRSTUVWXYZ",
                    digits="0123456789",
                ),
            ),
            "GE": RegionConfig(
                pattern=["XX-000-XX"],
                valid_chars=ValidChars(
                    letters="ABCDEFGHIJKLMNOPRSTUVWXYZ-",
                    digits="0123456789",
                ),
            ),
        }
    )
