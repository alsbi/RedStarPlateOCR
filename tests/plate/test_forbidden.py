"""Tests for ForbiddenFilter (T6.2)."""

from __future__ import annotations

from redstar_plate_ocr.plate.forbidden import ForbiddenFilter


class TestForbiddenFilter:
    """Forbidden combo filtering for beam search hypotheses."""

    def test_no_forbidden_combos_returns_best(
        self,
    ) -> None:
        """If no forbidden combos, best hypothesis is returned."""
        filt = ForbiddenFilter(forbidden_combos=[])
        hypotheses = [("A123BC99", 0.9), ("A124BC99", 0.8)]
        text, review = filt.filter(hypotheses)
        assert text == "A123BC99"
        assert review is False

    def test_best_contains_forbidden_returns_next(
        self,
    ) -> None:
        """If best contains forbidden, next valid is returned."""
        filt = ForbiddenFilter(forbidden_combos=["SEX", "LOX"])
        hypotheses = [
            ("01SEX00", 0.9),
            ("01ABC00", 0.7),
        ]
        text, review = filt.filter(hypotheses)
        assert text == "01ABC00"
        assert review is False

    def test_all_hypotheses_forbidden_returns_best_with_review(
        self,
    ) -> None:
        """If all hypotheses contain forbidden, best + needs_review."""
        filt = ForbiddenFilter(forbidden_combos=["SEX", "LOX"])
        hypotheses = [
            ("01SEX00", 0.9),
            ("02LOX00", 0.7),
        ]
        text, review = filt.filter(hypotheses)
        assert text == "01SEX00"
        assert review is True

    def test_forbidden_is_substring_match(
        self,
    ) -> None:
        """Forbidden combo is matched as substring."""
        filt = ForbiddenFilter(forbidden_combos=["SEX"])
        hypotheses = [("AASEX00", 0.9), ("AAABC00", 0.5)]
        text, review = filt.filter(hypotheses)
        assert text == "AAABC00"
        assert review is False

    def test_empty_hypotheses_returns_empty(
        self,
    ) -> None:
        """Empty hypothesis list returns empty result."""
        filt = ForbiddenFilter(forbidden_combos=["SEX"])
        text, review = filt.filter([])
        assert text == ""
        assert review is False

    def test_single_valid_hypothesis(self) -> None:
        """Single valid hypothesis is returned as-is."""
        filt = ForbiddenFilter(forbidden_combos=["SEX"])
        hypotheses = [("A123BC99", 0.9)]
        text, review = filt.filter(hypotheses)
        assert text == "A123BC99"
        assert review is False

    def test_case_sensitive_matching(
        self,
    ) -> None:
        """Forbidden matching is case-sensitive."""
        filt = ForbiddenFilter(forbidden_combos=["SEX"])
        # "sex" (lowercase) should NOT match "SEX"
        hypotheses = [("01sex00", 0.9)]
        text, review = filt.filter(hypotheses)
        assert text == "01sex00"
        assert review is False
