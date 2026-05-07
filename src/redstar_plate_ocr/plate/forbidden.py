"""Forbidden combo filter for beam search hypotheses."""

from __future__ import annotations

import re


class ForbiddenFilter:
    """Filter beam search hypotheses by forbidden combos.

    If the best hypothesis contains a forbidden combo,
    try the next one. If none are valid, return the best
    with needs_review=True.
    """

    def __init__(self, forbidden_combos: list[str]) -> None:
        self.forbidden_combos = forbidden_combos
        if forbidden_combos:
            self._pattern = re.compile(
                "|".join(map(re.escape, forbidden_combos))
            )
        else:
            self._pattern = None

    def filter(
        self,
        hypotheses: list[tuple[str, float]],
    ) -> tuple[str, bool]:
        """Filter hypotheses, returning (text, needs_review).

        Args:
            hypotheses: list of (text, confidence) sorted
                by confidence descending.

        Returns:
            (text, needs_review) — best valid hypothesis.
        """
        if not hypotheses:
            return "", False
        if self._pattern is None:
            return hypotheses[0][0], False
        for text, _ in hypotheses:
            if not self.contains_forbidden(text):
                return text, False
        return hypotheses[0][0], True

    def contains_forbidden(self, text: str) -> bool:
        """Check if text contains any forbidden combo."""
        if self._pattern is None:
            return False
        return bool(self._pattern.search(text))
