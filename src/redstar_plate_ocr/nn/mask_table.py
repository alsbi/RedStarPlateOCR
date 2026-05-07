"""Build positional country-conditioned logit mask table."""

import torch

from redstar_plate_ocr.plate.config import PlateConfig

MASK_VALUE: float = -15.0


def _char_allowed(
    ch: str,
    region_letters: set[str],
    region_digits: set[str],
) -> set[str]:
    dispatch = {
        "X": region_letters,
        "0": region_digits,
        "o": region_letters | region_digits,
        "-": {"-"},
    }
    return dispatch.get(ch, set())


def _allowed_for_position(
    pos: int,
    patterns: list[str],
    region_letters: set[str],
    region_digits: set[str],
) -> set[str]:
    """Determine allowed characters at a given position."""
    allowed: set[str] = set()
    for pat in patterns:
        if pos >= len(pat):
            allowed |= region_letters | region_digits
        else:
            allowed |= _char_allowed(
                pat[pos], region_letters, region_digits
            )
    return allowed


def build_positional_mask_table(
    plate_config: PlateConfig,
    max_seq_len: int,
    mask_value: float = MASK_VALUE,
) -> torch.Tensor:
    """Build (num_countries, max_seq_len, union_alphabet_size) mask.

    Position-aware: each position in the sequence gets its own
    mask based on the pattern character at that position.

    Pattern characters:
        X → letters allowed, digits forbidden
        0 → digits allowed, letters forbidden
        o → both letters and digits allowed
        - → hyphen allowed

    For countries with multiple patterns: union of allowed
    characters across all patterns per position.

    Blank (last index) is always 0.0.
    Positions beyond all pattern lengths are fully allowed.
    """
    union = plate_config.union_alphabet
    num_c = plate_config.num_countries
    union_size = plate_config.union_alphabet_size
    blank_idx = union_size - 1

    table = torch.zeros(num_c, max_seq_len, union_size)

    for c_idx, c_name in enumerate(plate_config.country_list):
        region = plate_config.regions[c_name]
        region_letters = set(region.valid_chars.letters)
        region_digits = set(region.valid_chars.digits)
        patterns = region.pattern

        for pos in range(max_seq_len):
            allowed_chars = _allowed_for_position(
                pos, patterns, region_letters, region_digits
            )

            for u_idx, char in enumerate(union):
                if char not in allowed_chars:
                    table[c_idx, pos, u_idx] = mask_value

            table[c_idx, pos, blank_idx] = 0.0

    return table


def build_mask_table(
    plate_config: PlateConfig,
    mask_value: float = MASK_VALUE,
) -> torch.Tensor:
    """Build (num_countries, union_alphabet_size) flat mask table.

    Fallback: same mask for all positions.
    0.0 = allowed, mask_value = disallowed.
    Blank (last index) is always 0.0.
    """
    union = plate_config.union_alphabet
    num_c = plate_config.num_countries
    union_size = plate_config.union_alphabet_size

    table = torch.full(
        (num_c, union_size),
        mask_value,
    )

    for c_idx, c_name in enumerate(plate_config.country_list):
        region = plate_config.regions[c_name]
        alphabet = region.get_alphabet()
        for char in alphabet:
            if char in union:
                pos = union.index(char)
                table[c_idx, pos] = 0.0
        # Blank is always allowed (last index = union_alphabet_size - 1)
        table[c_idx, -1] = 0.0

    return table
