"""Tests for RegionConfig.get_alphabet()."""

from redstar_plate_ocr.plate.config import (
    PlateConfig,
    RegionConfig,
    ValidChars,
)


def _make_region() -> RegionConfig:
    """Create a RegionConfig for testing."""
    return RegionConfig(
        pattern="X000XX00",
        valid_chars=ValidChars(
            letters="ABEKMHOPCTYX",
            digits="0123456789",
        ),
    )


class TestRegionConfigGetAlphabet:
    """Tests for RegionConfig.get_alphabet()."""

    def test_returns_raw_alphabet(self) -> None:
        """get_alphabet() returns raw_alphabet()."""
        region = _make_region()
        result = region.get_alphabet()
        assert result == region.raw_alphabet()


class TestPlateConfigGetAlphabet:
    """Tests for PlateConfig.get_alphabet delegation."""

    def test_delegates_to_region_get_alphabet(self) -> None:
        """PlateConfig.get_alphabet delegates to RegionConfig."""
        region = RegionConfig(
            pattern="X000XX00",
            valid_chars=ValidChars(
                letters="ABEKMHOPCTYX",
                digits="0123456789",
            ),
        )
        config = PlateConfig(regions={"RU": region})
        assert config.get_alphabet("RU") == region.raw_alphabet()

    def test_unknown_country_returns_hardcoded_alphabet(self) -> None:
        """Unknown country returns hardcoded base alphabet."""
        config = PlateConfig(
            regions={
                "RU": RegionConfig(
                    pattern="X000XX00",
                    valid_chars=ValidChars(
                        letters="ABEKMHOPCTYX",
                        digits="0123456789",
                    ),
                ),
            },
        )
        result = config.get_alphabet("XX")
        assert result == "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
