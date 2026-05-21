"""Metrics: CER, CharAcc, Accuracy, NED, BSR, ATR."""

from __future__ import annotations

from collections import defaultdict

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
        for pred, tgt in zip(predictions, targets, strict=True):
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
        for pred, tgt in zip(predictions, targets, strict=True):
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
        for pred, tgt in zip(predictions, targets, strict=True):
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


class NormalizedEditDistance:
    """NED = 1 - edit_distance(pred, gt) / max(len(pred), len(gt))."""

    def __call__(
        self,
        predictions: list[str],
        targets: list[str],
    ) -> float:
        """Return mean NED across all prediction-target pairs."""
        if not predictions:
            return 0.0
        total = 0.0
        for pred, tgt in zip(predictions, targets, strict=True):
            denom = max(len(pred), len(tgt))
            if denom == 0:
                total += 1.0
            else:
                dist = levenshtein_distance(pred, tgt)
                total += 1.0 - dist / denom
        return total / len(predictions)


class BigramSwapRate:
    """Fraction of examples where a critical bigram is swapped."""

    DEFAULT_BIGRAMS = ["CX", "XC", "KH", "HK", "PH", "HP", "CK", "KC"]

    def __init__(self, bigrams: list[str] | None = None) -> None:
        self.bigrams = bigrams or self.DEFAULT_BIGRAMS

    def __call__(
        self,
        predictions: list[str],
        targets: list[str],
    ) -> float:
        """Return BSR = swapped_examples / total_examples."""
        if not predictions:
            return 0.0
        swaps = 0
        for pred, tgt in zip(predictions, targets, strict=True):
            found = False
            for bg in self.bigrams:
                rev = bg[::-1]
                limit = min(len(tgt), len(pred))
                for i in range(limit - 1):
                    if tgt[i : i + 2] == bg and pred[i : i + 2] == rev:
                        found = True
                        break
                if found:
                    break
            if found:
                swaps += 1
        return swaps / len(predictions)


class AdjacentTranspositionRate:
    """Fraction of errors that are a swap of exactly two adjacent chars."""

    def __call__(
        self,
        predictions: list[str],
        targets: list[str],
    ) -> float:
        """Return ATR = transpositions / total_with_errors."""
        transpositions = 0
        total_with_errors = 0
        for pred, tgt in zip(predictions, targets, strict=True):
            if pred == tgt:
                continue
            total_with_errors += 1
            if _is_adjacent_transposition(pred, tgt):
                transpositions += 1
        if total_with_errors == 0:
            return 0.0
        return transpositions / total_with_errors


def _is_adjacent_transposition(pred: str, tgt: str) -> bool:
    """Check if *pred* differs from *tgt* by one adjacent swap."""
    if len(pred) != len(tgt):
        return False
    if levenshtein_distance(pred, tgt) != 2:
        return False
    diff_positions = [i for i in range(len(tgt)) if pred[i] != tgt[i]]
    if len(diff_positions) != 2:
        return False
    i, j = diff_positions
    return j == i + 1 and pred[i] == tgt[j] and pred[j] == tgt[i]


def compute_per_group_metrics(
    predictions: list[str],
    targets: list[str],
    group_labels: list[str],
) -> dict[str, dict[str, float]]:
    """Compute CER and PlateAcc for each group.

    Returns:
        ``{'group_name': {'cer': ..., 'plate_acc': ...}}``
    """
    groups: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"preds": [], "tgts": []}
    )
    for pred, tgt, label in zip(
        predictions, targets, group_labels, strict=True
    ):
        groups[label]["preds"].append(pred)
        groups[label]["tgts"].append(tgt)

    result: dict[str, dict[str, float]] = {}
    for label in sorted(groups):
        data = groups[label]
        cer = CharacterErrorRate()
        acc = Accuracy()
        cer.update(data["preds"], data["tgts"])
        acc.update(data["preds"], data["tgts"])
        result[label] = {
            "cer": cer.compute(),
            "plate_acc": acc.compute(),
        }
    return result
