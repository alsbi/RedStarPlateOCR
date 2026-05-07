"""T0.1: Verify country_list matches expected set (alphabetical order)."""

# Expected country list: enabled countries sorted by ISO code
_EXPECTED_COUNTRIES = [
    "BY",
    "GE",
    "KG",
    "KZ",
    "RU",
    "UA",
    "UZ",
]


def test_all_expected_countries_in_config(plate_config):
    """All expected countries are in plate_config.country_list."""
    for country in _EXPECTED_COUNTRIES:
        assert country in plate_config.country_list


def test_country_list_sorted_alphabetically(plate_config):
    """country_list is sorted by ISO code (alphabetical)."""
    assert plate_config.country_list == _EXPECTED_COUNTRIES
