"""Metrics: CER, CharAcc, Accuracy."""

from __future__ import annotations

from rapidfuzz.distance import Levenshtein


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    return Levenshtein.distance(s1, s2)


class CharacterErrorRate:
    """CER via Levenshtein distance."""

    def __init__(self) -> None:
        self._total_dist: int = 0
        self._total_len: int = 0

    def update(
        self,
        predictions: list[str],
        targets: list[str],
    ) -> None:
        """Accumulate distances and target lengths."""
        for pred, tgt in zip(predictions, targets):
            self._total_dist += levenshtein_distance(pred, tgt)
            self._total_len += max(len(tgt), 1)

    def compute(self) -> float:
        """Return CER = total_distance / total_target_length."""
        if self._total_len == 0:
            return 0.0
        return self._total_dist / self._total_len

    def reset(self) -> None:
        """Reset accumulated state."""
        self._total_dist = 0
        self._total_len = 0


class Accuracy:
    """Fraction of exact matches between predictions and targets."""

    def __init__(self) -> None:
        self._correct: int = 0
        self._total: int = 0

    def update(
        self,
        predictions: list[str],
        targets: list[str],
    ) -> None:
        """Count exact matches."""
        for pred, tgt in zip(predictions, targets):
            self._total += 1
            if pred == tgt:
                self._correct += 1

    def compute(self) -> float:
        """Return accuracy = correct / total."""
        if self._total == 0:
            return 0.0
        return self._correct / self._total

    def reset(self) -> None:
        """Reset accumulated state."""
        self._correct = 0
        self._total = 0


class CharacterAccuracy:
    """Per-character accuracy via Levenshtein distance."""

    def __init__(self) -> None:
        self._total_correct: int = 0
        self._total_chars: int = 0

    def update(
        self,
        predictions: list[str],
        targets: list[str],
    ) -> None:
        """Accumulate correct chars and total target chars."""
        for pred, tgt in zip(predictions, targets):
            dist = levenshtein_distance(pred, tgt)
            self._total_correct += max(len(tgt) - dist, 0)
            self._total_chars += max(len(tgt), 1)

    def compute(self) -> float:
        """Return character accuracy."""
        if self._total_chars == 0:
            return 0.0
        return self._total_correct / self._total_chars

    def reset(self) -> None:
        """Reset accumulated state."""
        self._total_correct = 0
        self._total_chars = 0
