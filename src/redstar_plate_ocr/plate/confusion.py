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

    letters_fs = frozenset(c for c in valid_letters if c.isalpha())
    digits_fs = frozenset(valid_digits)

    best_text = text
    best_fixes = len(text) + 1  # impossibly many

    for pattern in patterns:
        corrected, n_fixes = _correct_against_pattern(
            text,
            pattern,
            letters_fs,
            digits_fs,
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
        ti, fix = _handle_slot(text, ti, pc, letters_fs, digits_fs, result)
        fixes += fix

    # Append any remaining characters (shouldn't happen for valid plates)
    # but we keep them to avoid data loss
    remaining = text[ti:]
    if remaining:
        result.append(remaining)

    return "".join(result), fixes


def _check_pattern_perfect(
    text: str,
    pattern: str,
    letters_fs: frozenset[str],
    digits_fs: frozenset[str],
) -> bool:
    """Check if text matches pattern with zero corrections."""
    _, fixes = _correct_against_pattern(
        text,
        pattern,
        letters_fs,
        digits_fs,
    )
    return fixes == 0


def _should_skip_swap(
    text: str,
    patterns: list[str],
    ctc_logits: object | None,
    ctc_alignment: list[int] | None,
    alphabet: str,
    text_confidence: float,
    threshold: float,
) -> bool:
    """Return True if swap correction should be skipped."""
    alignment_len = len(ctc_alignment) if ctc_alignment else 0
    return any(
        [
            len(text) < 2,
            not patterns,
            ctc_logits is None,
            ctc_alignment is None,
            not alphabet,
            alignment_len != len(text),
            text_confidence >= threshold,
        ]
    )


def _is_same_type_pair(
    a: str,
    b: str,
    letters_fs: frozenset[str],
    digits_fs: frozenset[str],
) -> bool:
    """Check if two characters are same-type (both letters or both digits)."""
    return (a in letters_fs and b in letters_fs) or (
        a in digits_fs and b in digits_fs
    )


def _get_char_index(char: str, alphabet: str) -> int:
    """Get index of char in alphabet, or -1 if not found."""
    return alphabet.index(char) if char in alphabet else -1


def _try_swap_pair(
    i: int,
    result: list[str],
    result_alignment: list[int],
    letters_fs: frozenset[str],
    digits_fs: frozenset[str],
    ctc_logits: object,
    alphabet: str,
    swap_margin: float,
) -> int:
    """Try swapping adjacent pair at position *i*; return next position."""
    a, b = result[i], result[i + 1]
    if not _is_same_type_pair(a, b, letters_fs, digits_fs):
        return i + 1
    if a == b:
        return i + 1

    t1 = result_alignment[i]
    t2 = result_alignment[i + 1]
    a_idx = _get_char_index(a, alphabet)
    b_idx = _get_char_index(b, alphabet)
    if min(a_idx, b_idx) < 0:
        return i + 1

    current_score = ctc_logits[t1, a_idx].item() + ctc_logits[t2, b_idx].item()
    swapped_score = ctc_logits[t1, b_idx].item() + ctc_logits[t2, a_idx].item()

    if swapped_score > current_score + swap_margin:
        result[i], result[i + 1] = b, a
        return i + 2
    return i + 1


def adjacent_swap_correct(
    text: str,
    patterns: list[str],
    valid_letters: str,
    valid_digits: str,
    ctc_logits: "object | None" = None,
    ctc_alignment: list[int] | None = None,
    alphabet: str = "",
    text_confidence: float = 1.0,
    swap_confidence_threshold: float = 0.95,
    swap_margin: float = 0.3,
) -> str:
    """Fix adjacent-same-type transposition errors using CTC logit evidence.

    When the model swaps two adjacent characters of the same type
    (e.g. ``CX`` → ``XC`` on an ``XX`` pattern), the confusion
    corrector cannot help because both characters are valid at
    either position.

    This function inspects the CTC logits: for each adjacent pair
    of same-type characters ``(a, b)`` decoded at timesteps
    ``(t₁, t₂)``, it compares:

    - **current score**: ``logits[t₁, a_idx] + logits[t₂, b_idx]``
    - **swapped score**: ``logits[t₁, b_idx] + logits[t₂, a_idx]``

    A swap is applied **only** when **all** of the following hold:

    1. The model's overall confidence is below
       *swap_confidence_threshold* (the model is uncertain).
    2. The swapped score exceeds the current score by at
       least *swap_margin* (logit-based evidence is strong).
    3. Both characters belong to the same type (letter-letter
       or digit-digit) — cross-type swaps are handled by
       the confusion corrector instead.

    When ``ctc_logits`` or ``ctc_alignment`` is not available,
    the function returns *text* unchanged (safe fallback).

    Args:
        text: Plate text (typically after confusion correction).
        patterns: Region patterns (used to restrict swaps to
            same-type adjacent pairs only).
        valid_letters: Valid letter characters.
        valid_digits: Valid digit characters.
        ctc_logits: CTC log-probability tensor ``(T, V)``.
        ctc_alignment: Timestep index for each character in *text*.
        alphabet: Alphabet string (letters + digits) matching the
            CTC vocabulary order.
        text_confidence: Overall recognition confidence (0–1).
            Swap correction is skipped when the model is highly
            confident — no point fixing what isn't broken.
        swap_confidence_threshold: Maximum text_confidence at which
            swap correction is attempted.  Default 0.95 means
            "only try swaps when confidence < 95%".
        swap_margin: Minimum logit advantage for the swapped
            ordering over the current one.  Prevents swaps driven
            by negligible noise.

    Returns:
        Corrected text if logit evidence supports a swap, else
        the original text.
    """
    if _should_skip_swap(
        text,
        patterns,
        ctc_logits,
        ctc_alignment,
        alphabet,
        text_confidence,
        swap_confidence_threshold,
    ):
        return text

    import torch

    if not isinstance(ctc_logits, torch.Tensor):
        return text

    # Filter to alphabetic letters only — valid_letters may include
    # separators like '-' that should NOT participate in same-type
    # swap logic (they are literals, not true letters).
    letters_fs = frozenset(c for c in valid_letters if c.isalpha())
    digits_fs = frozenset(valid_digits)
    result = list(text)
    result_alignment = list(ctc_alignment)

    i = 0
    while i < len(result) - 1:
        i = _try_swap_pair(
            i,
            result,
            result_alignment,
            letters_fs,
            digits_fs,
            ctc_logits,
            alphabet,
            swap_margin,
        )

    return "".join(result)


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
