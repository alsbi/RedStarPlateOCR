"""Tests for _char_allowed and build_positional_mask_table.

Verifies that optional pattern characters (x, o) produce
correct positional masks and that the mask is consistent
with PatternValidator semantics.
"""

from __future__ import annotations

from redstar_plate_ocr.nn.mask_table import (
    MASK_VALUE,
    _allowed_for_position,
    _char_allowed,
    build_positional_mask_table,
)
from redstar_plate_ocr.plate.config import PlateConfig

# ── _char_allowed ─────────────────────────────────────────────


class TestCharAllowed:
    """Tests for _char_allowed dispatch."""

    LETTERS = set("ABCD")
    DIGITS = set("0123")

    def test_mandatory_letter_x(self) -> None:
        """X → letters allowed."""
        assert _char_allowed("X", self.LETTERS, self.DIGITS) == self.LETTERS

    def test_mandatory_digit_0(self) -> None:
        """0 → digits allowed."""
        assert _char_allowed("0", self.LETTERS, self.DIGITS) == self.DIGITS

    def test_optional_letter_x(self) -> None:
        """x → letters allowed (same as X)."""
        assert _char_allowed("x", self.LETTERS, self.DIGITS) == self.LETTERS

    def test_optional_digit_o(self) -> None:
        """o → digits only (not letters + digits)."""
        result = _char_allowed("o", self.LETTERS, self.DIGITS)
        assert result == self.DIGITS
        # Explicitly: letters are NOT allowed at 'o' positions
        assert not result & self.LETTERS

    def test_literal_hyphen(self) -> None:
        """- → only hyphen allowed."""
        assert _char_allowed("-", self.LETTERS, self.DIGITS) == {"-"}

    def test_unknown_returns_empty(self) -> None:
        """Unknown pattern char → empty set."""
        assert _char_allowed("Z", self.LETTERS, self.DIGITS) == set()


# ── _allowed_for_position ─────────────────────────────────────


class TestAllowedForPosition:
    """Tests for _allowed_for_position."""

    LETTERS = set("AB")
    DIGITS = set("01")

    def test_position_within_pattern(self) -> None:
        """Position 0 of 'X0' → letters."""
        result = _allowed_for_position(
            0, ["X0"], self.LETTERS, self.DIGITS
        )
        assert result == self.LETTERS

    def test_position_beyond_pattern(self) -> None:
        """Position beyond pattern length → all allowed."""
        result = _allowed_for_position(
            2, ["X0"], self.LETTERS, self.DIGITS
        )
        assert result == self.LETTERS | self.DIGITS

    def test_multiple_patterns_union(self) -> None:
        """Multiple patterns → union of allowed chars."""
        # pos 0: 'X' (letters) from pat1, '0' (digits) from pat2
        result = _allowed_for_position(
            0, ["X0", "0X"], self.LETTERS, self.DIGITS
        )
        assert result == self.LETTERS | self.DIGITS

    def test_optional_digit_position(self) -> None:
        """Position with 'o' → only digits."""
        result = _allowed_for_position(
            1, ["Xo"], self.LETTERS, self.DIGITS
        )
        assert result == self.DIGITS

    def test_optional_letter_position(self) -> None:
        """Position with 'x' → only letters."""
        result = _allowed_for_position(
            1, ["0x"], self.LETTERS, self.DIGITS
        )
        assert result == self.LETTERS


# ── build_positional_mask_table ───────────────────────────────


class TestPositionalMaskTable:
    """Tests for build_positional_mask_table."""

    def test_shape(self, plate_config: PlateConfig) -> None:
        """Table has shape (num_countries, max_seq_len, union_size)."""
        table = build_positional_mask_table(plate_config, max_seq_len=12)
        assert table.shape == (
            plate_config.num_countries,
            12,
            plate_config.union_alphabet_size,
        )

    def test_blank_always_allowed(
        self, plate_config: PlateConfig
    ) -> None:
        """Blank index is always 0.0 for all positions."""
        table = build_positional_mask_table(plate_config, max_seq_len=12)
        blank_idx = plate_config.union_alphabet_size - 1
        for c_idx in range(table.shape[0]):
            for pos in range(table.shape[1]):
                assert table[c_idx, pos, blank_idx].item() == 0.0

    def test_ru_pattern_x000xx00o(
        self, plate_config: PlateConfig
    ) -> None:
        """RU pattern 'X000XX00o': position-wise mask is correct.

        Position 0 (X): letters only
        Position 1-3 (0): digits only
        Position 4-5 (X): letters only
        Position 6-7 (0): digits only
        Position 8 (o): digits only (NOT letters+digits)
        """
        table = build_positional_mask_table(
            plate_config, max_seq_len=10
        )
        union = plate_config.union_alphabet
        ru_idx = plate_config.country_list.index("RU")
        ru_letters = set(plate_config.regions["RU"].valid_chars.letters)
        ru_digits = set(plate_config.regions["RU"].valid_chars.digits)

        pattern = "X000XX00o"
        for pos, pc in enumerate(pattern):
            for u_idx, ch in enumerate(union):
                is_masked = (
                    table[ru_idx, pos, u_idx].item() == MASK_VALUE
                )
                if pc in ("X", "x"):
                    # Letter position: digits must be masked
                    if ch in ru_digits:
                        assert is_masked, (
                            f"RU pos={pos} pc={pc}: digit "
                            f"'{ch}' should be masked"
                        )
                    elif ch in ru_letters:
                        assert not is_masked, (
                            f"RU pos={pos} pc={pc}: letter "
                            f"'{ch}' should be allowed"
                        )
                elif pc in ("0", "o"):
                    # Digit position: letters must be masked
                    if ch in ru_letters:
                        assert is_masked, (
                            f"RU pos={pos} pc={pc}: letter "
                            f"'{ch}' should be masked"
                        )
                    elif ch in ru_digits:
                        assert not is_masked, (
                            f"RU pos={pos} pc={pc}: digit "
                            f"'{ch}' should be allowed"
                        )

    def test_beyond_pattern_all_allowed(
        self, plate_config: PlateConfig
    ) -> None:
        """Positions beyond pattern length allow all region chars."""
        table = build_positional_mask_table(
            plate_config, max_seq_len=20
        )
        union = plate_config.union_alphabet
        ru_idx = plate_config.country_list.index("RU")
        ru_letters = set(plate_config.regions["RU"].valid_chars.letters)
        ru_digits = set(plate_config.regions["RU"].valid_chars.digits)

        # RU pattern length is 9 (X000XX00o)
        for pos in range(9, 20):
            for u_idx, ch in enumerate(union):
                is_masked = (
                    table[ru_idx, pos, u_idx].item() == MASK_VALUE
                )
                if ch in ru_letters | ru_digits:
                    assert not is_masked, (
                        f"RU pos={pos} (beyond pattern): "
                        f"'{ch}' should be allowed"
                    )

    def test_o_position_no_letters(
        self, plate_config: PlateConfig
    ) -> None:
        """CRITICAL: 'o' position does NOT allow letters.

        Regression test: previously _char_allowed('o', ...)
        returned letters | digits, allowing the model to
        predict letters at optional-digit positions.
        """
        table = build_positional_mask_table(
            plate_config, max_seq_len=10
        )
        union = plate_config.union_alphabet
        ru_idx = plate_config.country_list.index("RU")
        ru_letters = set(plate_config.regions["RU"].valid_chars.letters)

        # Position 8 of RU pattern 'X000XX00o' is 'o'
        for u_idx, ch in enumerate(union):
            if ch in ru_letters:
                assert (
                    table[ru_idx, 8, u_idx].item() == MASK_VALUE
                ), (
                    f"RU pos=8 ('o'): letter '{ch}' must be masked"
                )

    def test_x_position_allows_letters(
        self, plate_config: PlateConfig
    ) -> None:
        """CRITICAL: 'x' position allows letters (not empty).

        Regression test: previously _char_allowed('x', ...)
        returned set(), blocking all characters.
        """
        # Build a synthetic config with 'x' in pattern
        # We'll test via _char_allowed directly since
        # no real config uses 'x'
        letters = set("ABCD")
        digits = set("0123")
        result = _char_allowed("x", letters, digits)
        assert result == letters, (
            "'x' pattern should allow letters"
        )

    def test_kz_multi_pattern_union(
        self, plate_config: PlateConfig
    ) -> None:
        """KZ has two patterns: union of allowed chars per pos."""
        table = build_positional_mask_table(
            plate_config, max_seq_len=10
        )
        union = plate_config.union_alphabet
        kz_idx = plate_config.country_list.index("KZ")
        kz_letters = set(plate_config.regions["KZ"].valid_chars.letters)
        kz_digits = set(plate_config.regions["KZ"].valid_chars.digits)

        # KZ patterns: '000XXX00' and '00000XXX'
        # Position 3: pat1='X' (letters), pat2='0' (digits)
        # → union: letters + digits
        for u_idx, ch in enumerate(union):
            is_masked = (
                table[kz_idx, 3, u_idx].item() == MASK_VALUE
            )
            if ch in kz_letters | kz_digits:
                assert not is_masked, (
                    f"KZ pos=3: '{ch}' should be allowed (union)"
                )

        # Position 0: both patterns have '0' → digits only
        for u_idx, ch in enumerate(union):
            is_masked = (
                table[kz_idx, 0, u_idx].item() == MASK_VALUE
            )
            if ch in kz_letters:
                assert is_masked, (
                    f"KZ pos=0: letter '{ch}' should be masked"
                )
