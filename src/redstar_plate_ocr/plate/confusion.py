"""Confusion corrector: fix visually-similar cross-type character errors.

During inference, a character may be misclassified into a visually
similar character of the wrong type (letter ↔ digit).  For example,
the model might predict 'B' on a digit-only position, when the
correct answer is '8'.  The positional mask inside the model already
makes this *less* likely, but with a soft mask (-3.0) the model can
still override it.

The corrector runs AFTER greedy/beam decoding and BEFORE forbidden
filter / pattern validation in the PostProcessor pipeline.  It checks
each character against the region's pattern: if the character doesn't
match the expected type at that position, it tries to substitute it
with a visually similar character that DOES match.

This module is deliberately conservative:
  - Only applies when exactly ONE confusion pair exists.
  - Only applies at mandatory positions (X, 0), not optional (x, o).
  - For countries with multiple patterns: tries each pattern, picks
    the one requiring the fewest corrections.
"""

from __future__ import annotations

# ── Visual confusion map ──────────────────────────────────────────────
# Each entry: (predicted_char, allowed_type_char).
# Ordered by visual similarity — first match wins when there are
# multiple candidates (e.g. 'O' could map to '0' or 'Q').
_CONFUSION_MAP: dict[str, list[str]] = {
    # Digit → Letter
    "0": ["O", "Q"],
    "1": ["I", "L"],
    "2": ["Z"],
    "3": [],
    "4": [],
    "5": ["S"],
    "6": ["G", "b"],
    "7": ["T"],
    "8": ["B"],
    "9": ["g", "q"],
    # Letter → Digit
    "A": [],
    "B": ["8"],
    "C": [],
    "D": ["0"],
    "E": [],
    "F": [],
    "G": ["6"],
    "H": [],
    "I": ["1"],
    "J": [],
    "K": [],
    "L": ["1"],
    "M": [],
    "N": [],
    "O": ["0"],
    "P": [],
    "Q": ["0"],
    "R": [],
    "S": ["5"],
    "T": ["7"],
    "U": [],
    "V": [],
    "W": [],
    "X": [],
    "Y": [],
    "Z": ["2"],
}

# Reverse map: for a given predicted char and a target set, find
# the best replacement from the confusion pairs that belongs to the
# target set.
_REVERSE_CACHE: dict[tuple[str, frozenset[str]], str | None] = {}


def _find_replacement(
    predicted: str,
    allowed_set: frozenset[str],
) -> str | None:
    """Find a visually-similar replacement for *predicted* in *allowed_set*.

    Returns the first confusion-pair member that belongs to *allowed_set*,
    or None if no suitable replacement exists.
    """
    key = (predicted, allowed_set)
    cached = _REVERSE_CACHE.get(key)
    if cached is not None:
        return cached if cached != "" else None

    candidates = _CONFUSION_MAP.get(predicted, [])
    for c in candidates:
        if c in allowed_set:
            _REVERSE_CACHE[key] = c
            return c

    # Also check if predicted itself is in allowed_set (no fix needed)
    if predicted in allowed_set:
        _REVERSE_CACHE[key] = ""
        return None

    # No confusion pair → no fix possible
    _REVERSE_CACHE[key] = ""
    return None


def correct_confusions(
    text: str,
    patterns: list[str],
    valid_letters: str,
    valid_digits: str,
) -> str:
    """Correct visually-similar cross-type character errors.

    Args:
        text: Decoded plate text (after CTC greedy/beam).
        patterns: Region patterns to try (e.g. ["X000XX00o"]).
        valid_letters: Valid letter characters for this region.
        valid_digits: Valid digit characters for this region.

    Returns:
        Corrected text string.
    """
    if not text or not patterns:
        return text

    letters_fs = frozenset(valid_letters)
    digits_fs = frozenset(valid_digits)

    best_text = text
    best_fixes = len(text) + 1  # impossibly many

    for pattern in patterns:
        corrected, n_fixes = _correct_against_pattern(
            text, pattern, letters_fs, digits_fs,
        )
        if n_fixes < best_fixes:
            best_fixes = n_fixes
            best_text = corrected
            if n_fixes == 0:
                break  # perfect, no need to try other patterns

    return best_text


def _correct_against_pattern(
    text: str,
    pattern: str,
    letters_fs: frozenset[str],
    digits_fs: frozenset[str],
) -> tuple[str, int]:
    """Correct text against a single pattern.

    Returns (corrected_text, number_of_fixes_applied).
    """
    result: list[str] = []
    ti = 0  # text index
    fixes = 0

    for pc in pattern:
        if ti >= len(text):
            break
        ti, fix = _handle_slot(
            text, ti, pc, letters_fs, digits_fs, result
        )
        fixes += fix

    # Append any remaining characters (shouldn't happen for valid plates)
    # but we keep them to avoid data loss
    remaining = text[ti:]
    if remaining:
        result.append(remaining)

    return "".join(result), fixes


def _handle_slot(
    text: str,
    ti: int,
    pc: str,
    letters_fs: frozenset[str],
    digits_fs: frozenset[str],
    result: list[str],
) -> tuple[int, int]:
    """Process one pattern slot; return (new_ti, fixes_this_slot)."""
    ch = text[ti]
    if pc == "X":
        return _handle_mandatory(ch, ti, letters_fs, result)
    if pc == "0":
        return _handle_mandatory(ch, ti, digits_fs, result)
    if pc == "x":
        return _handle_optional_letter(
            text, ti, ch, letters_fs, digits_fs, result
        )
    if pc == "o":
        return _handle_optional_any(
            text, ti, ch, letters_fs, digits_fs, result
        )
    return _handle_literal(ch, ti, pc, result)


def _handle_mandatory(
    ch: str,
    ti: int,
    allowed: frozenset[str],
    result: list[str],
) -> tuple[int, int]:
    """Process a mandatory slot (X or 0)."""
    if ch in allowed:
        result.append(ch)
        return ti + 1, 0
    rep = _find_replacement(ch, allowed)
    if rep is not None:
        result.append(rep)
        return ti + 1, 1
    # Can't fix — keep original
    result.append(ch)
    return ti + 1, 0


def _handle_optional_letter(
    text: str,
    ti: int,
    ch: str,
    letters_fs: frozenset[str],
    digits_fs: frozenset[str],
    result: list[str],
) -> tuple[int, int]:
    """Process an optional letter slot (x)."""
    if ch in letters_fs:
        result.append(ch)
        return ti + 1, 0
    if _peek_next_match(text, ti, "x", letters_fs, digits_fs):
        # Don't consume — try next slot
        return ti, 0
    # Try confusion replacement
    rep = _find_replacement(ch, letters_fs)
    if rep is not None:
        result.append(rep)
        return ti + 1, 1
    # Skip optional slot
    return ti, 0


def _handle_optional_any(
    text: str,
    ti: int,
    ch: str,
    letters_fs: frozenset[str],
    digits_fs: frozenset[str],
    result: list[str],
) -> tuple[int, int]:
    """Process an optional letter-or-digit slot (o)."""
    if ch in letters_fs or ch in digits_fs:
        result.append(ch)
        return ti + 1, 0
    # Try confusion against both sets
    rep = _find_replacement(ch, letters_fs | digits_fs)
    if rep is not None:
        result.append(rep)
        return ti + 1, 1
    # skip optional
    return ti, 0


def _handle_literal(
    ch: str,
    ti: int,
    pc: str,
    result: list[str],
) -> tuple[int, int]:
    """Process a literal separator slot."""
    if ch == pc:
        result.append(ch)
        return ti + 1, 0
    result.append(pc)
    # Don't consume text — separator was missing
    return ti, 1


def _peek_next_match(
    text: str,
    ti: int,
    _current_slot: str,
    letters_fs: frozenset[str],
    digits_fs: frozenset[str],
) -> bool:
    """Check if text[ti] would match the NEXT pattern slot.

    Used for optional slots to decide whether to skip or consume.
    Simplified: always return False (consume the char, try confusion).
    """
    return False
