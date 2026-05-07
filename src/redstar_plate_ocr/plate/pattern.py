"""Pattern validation and correction for plate text."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    """Result of pattern validation."""

    text: str
    corrected: bool


def _resolve_mandatory(
    ch: str,
    valid_set: str,
    default: str,
) -> tuple[str, bool, int]:
    """Resolve mandatory slot (X or 0): match or default."""
    return (ch, False, 1) if ch in valid_set else (default, True, 1)


def _resolve_optional(
    ch: str,
    valid_set: str,
) -> tuple[str, bool, int]:
    """Resolve optional slot (x or o): match or skip."""
    return (ch, False, 1) if ch in valid_set else ("", False, 0)


def _resolve_literal(ch: str, pc: str) -> tuple[str, bool, int]:
    """Resolve literal separator: match or correct."""
    return (ch, False, 1) if ch == pc else (pc, True, 1)


def _classify_char(
    ch: str,
    pc: str,
    valid_letters: str,
    valid_digits: str,) -> tuple[str, bool, int]:
    """Classify input char against pattern slot.

    Returns (output_char, was_corrected, text_index_increment).
    """
    if pc == "X":
        return _resolve_mandatory(ch, valid_letters, valid_letters[0])
    if pc == "0":
        return _resolve_mandatory(ch, valid_digits, valid_digits[0])
    if pc == "x":
        return _resolve_optional(ch, valid_letters)
    if pc == "o":
        return _resolve_optional(ch, valid_digits)
    return _resolve_literal(ch, pc)


def _process_pattern_slot(
    text: str,
    ti: int,
    pc: str,
    valid_letters: str,
    valid_digits: str,
) -> tuple[str, bool, int]:
    """Process one pattern slot against text at position ti.

    Returns (output_char, was_corrected, new_text_index).
    """
    if ti >= len(text):
        if pc in ("o", "x"):
            return "", False, ti
        return "", True, ti
    ch = text[ti]
    out, corr, inc = _classify_char(ch, pc, valid_letters, valid_digits)
    return out, corr, ti + inc


class PatternValidator:
    """Validate and correct plate text against a pattern.

    Pattern syntax:
        X = mandatory letter
        0 = mandatory digit
        x = optional letter
        o = optional digit
    Any other character (e.g. '-') is a literal separator.
    """

    def __init__(
        self,
        pattern: str,
        valid_letters: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        valid_digits: str = "0123456789",
    ) -> None:
        self.pattern = pattern
        self.valid_letters = valid_letters
        self.valid_digits = valid_digits
        self._multi_cache: dict[str, PatternValidator] = {}

    def validate(self, text: str) -> ValidationResult:
        """Validate text against pattern, attempt correction."""
        return self._validate_fixed(text)

    def validate_multi(
        self,
        text: str,
        patterns: list[str],
    ) -> ValidationResult:
        """Validate text against multiple patterns, pick best.

        Strategy: try each pattern, return the result with
        corrected=False if any matches perfectly. Otherwise
        return the result with fewest corrections (longest
        text).  Falls back to self.validate() when patterns
        is empty.
        """
        if not patterns:
            return self.validate(text)
        best: ValidationResult | None = None
        for p in patterns:
            result = self._get_cached(p).validate(text)
            if not result.corrected:
                return result
            best = self._pick_best(best, result)
        return best or ValidationResult(text=text, corrected=True)

    def _pick_best(
        self,
        best: ValidationResult | None,
        result: ValidationResult,
    ) -> ValidationResult:
        """Return the better of two validation results."""
        if best is None or len(result.text) > len(best.text):
            return result
        return best

    def _get_cached(self, pattern: str) -> PatternValidator:
        """Get or create a cached validator for a pattern."""
        if pattern == self.pattern:
            return self
        if pattern not in self._multi_cache:
            self._multi_cache[pattern] = PatternValidator(
                pattern=pattern,
                valid_letters=self.valid_letters,
                valid_digits=self.valid_digits,
            )
        return self._multi_cache[pattern]

    def _validate_fixed(self, text: str) -> ValidationResult:
        """Validate against fixed pattern with correction."""
        if len(text) < self._mandatory_length():
            return ValidationResult(text=text, corrected=True)
        result, corrected, ti = self._apply_pattern(text)
        corrected = corrected or self._check_truncate(ti, text)
        return ValidationResult(text="".join(result), corrected=corrected)

    def _apply_pattern(self, text: str) -> tuple[list[str], bool, int]:
        """Apply pattern slots to text."""
        result: list[str] = []
        corrected = False
        ti = 0
        for pc in self.pattern:
            out, corr, ti = _process_pattern_slot(
                text, ti, pc, self.valid_letters, self.valid_digits,
            )
            if out:
                result.append(out)
            if corr:
                corrected = True
        return result, corrected, ti

    def _check_truncate(self, ti: int, text: str) -> bool:
        """Return True if text was truncated after applying pattern."""
        return ti < len(text)

    def _mandatory_length(self) -> int:
        """Count mandatory positions in pattern."""
        count = 0
        for c in self.pattern:
            if c in ("X", "0"):
                count += 1
            elif c not in ("x", "o"):
                count += 1  # separator is mandatory
        return count
