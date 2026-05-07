"""Tests for PatternValidator."""

from __future__ import annotations

from redstar_plate_ocr.plate.pattern import PatternValidator


class TestPatternValidatorFixed:
    """Validation against fixed patterns (X/0/x/o)."""

    def test_valid_ru_pattern_passes(self) -> None:
        """Valid RU plate text passes validation."""
        validator = PatternValidator("X000XX00o")
        result = validator.validate("A123BC99")
        assert result.text == "A123BC99"
        assert result.corrected is False

    def test_invalid_digit_where_letter_expected_corrected(
        self,
    ) -> None:
        """Digit where letter expected is corrected."""
        validator = PatternValidator("X000XX00o")
        result = validator.validate("0123BC99")
        assert result.corrected is True

    def test_invalid_letter_where_digit_expected_corrected(
        self,
    ) -> None:
        """Letter where digit expected is corrected."""
        validator = PatternValidator("X000XX00o")
        result = validator.validate("AA23BC99")
        assert result.corrected is True

    def test_optional_digit_present_passes(self) -> None:
        """Optional digit (o) present passes."""
        validator = PatternValidator("X000XX00o")
        result = validator.validate("A123BC995")
        assert result.corrected is False

    def test_optional_digit_absent_passes(self) -> None:
        """Optional digit (o) absent passes."""
        validator = PatternValidator("X000XX00o")
        result = validator.validate("A123BC99")
        assert result.corrected is False

    def test_text_shorter_than_mandatory_fails(
        self,
    ) -> None:
        """Text shorter than mandatory positions is flagged."""
        validator = PatternValidator("X000XX00o")
        result = validator.validate("A12")
        assert result.corrected is True

    def test_text_longer_than_pattern_truncated(
        self,
    ) -> None:
        """Text longer than pattern is truncated."""
        validator = PatternValidator("X000XX00o")
        result = validator.validate("A123BC99123")
        assert len(result.text) <= 9

    def test_kz_pattern_valid(self) -> None:
        """KZ pattern validation works."""
        validator = PatternValidator("000XXX00")
        result = validator.validate("123ABC00")
        assert result.corrected is False

    def test_by_pattern_4digit(self) -> None:
        """BY 4-digit pattern validates."""
        validator = PatternValidator("0000XX-0")
        result = validator.validate("1234AB-5")
        assert result.corrected is False

    def test_by_pattern_5digit_fails(self) -> None:
        """BY 5-digit number is too long for 0000XX-0 pattern."""
        validator = PatternValidator("0000XX-0")
        result = validator.validate("12345AB-5")
        assert result.corrected is True


class TestPatternValidatorCorrection:
    """Correction logic tests."""

    def test_correction_uses_valid_chars(
        self,
    ) -> None:
        """Correction replaces with valid chars."""
        validator = PatternValidator(
            "X00",
            valid_letters="AB",
            valid_digits="01",
        )
        result = validator.validate("C01")
        assert result.text[0] in "AB"
        assert result.corrected is True

    def test_no_correction_needed_returns_original(
        self,
    ) -> None:
        """No correction needed returns original text."""
        validator = PatternValidator(
            "X00",
            valid_letters="ABC",
            valid_digits="012",
        )
        result = validator.validate("A01")
        assert result.text == "A01"
        assert result.corrected is False


class TestValidateMulti:
    """validate_multi() — multi-pattern validation."""

    def test_validate_multi_perfect_match_first(self) -> None:
        """7-char text matches first pattern (X0000XX)."""
        validator = PatternValidator("X0000XX")
        result = validator.validate_multi("E2695BP", ["X0000XX", "000000XXX"])
        assert result.text == "E2695BP"
        assert result.corrected is False

    def test_validate_multi_perfect_match_second(self) -> None:
        """9-char text matches second pattern (000000XXX)."""
        validator = PatternValidator("X0000XX")
        result = validator.validate_multi(
            "069759DLI", ["X0000XX", "000000XXX"]
        )
        assert result.text == "069759DLI"
        assert result.corrected is False

    def test_validate_multi_no_match(self) -> None:
        """No pattern matches — best effort with corrections."""
        validator = PatternValidator("X0000XX")
        result = validator.validate_multi("ABC", ["X0000XX", "000000XXX"])
        assert result.corrected is True

    def test_validate_multi_empty_patterns(self) -> None:
        """Empty patterns list falls back to self.validate()."""
        validator = PatternValidator("X0000XX")
        result = validator.validate_multi("E2695BP", [])
        assert result.text == "E2695BP"
        assert result.corrected is False
