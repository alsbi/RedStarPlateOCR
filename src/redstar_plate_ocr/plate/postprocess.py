"""CTC beam search decoder and greedy decoding."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


def _compute_confidence(probs: "Sequence[float]") -> float:
    """Geometric mean confidence from a sequence of probabilities."""
    if len(probs) == 0:
        return 1.0
    log_sum = sum(math.log(p) for p in probs)
    return math.exp(log_sum / len(probs))


def _decode_gpu(
    indices: torch.Tensor,
    max_probs: torch.Tensor,
    blank_idx: int,
    alphabet: str,
) -> tuple[str, float]:
    """CTC decode on GPU tensor."""
    pad = torch.full(
        (1,), -1, device=indices.device, dtype=indices.dtype,
    )
    shifted = torch.cat([pad, indices[:-1]])
    keep = (indices != blank_idx) & (indices != shifted)
    chars_idx = indices[keep]
    used_probs = max_probs[keep]
    valid = chars_idx < len(alphabet)
    chars_idx = chars_idx[valid]
    used_probs = used_probs[valid]
    text = "".join(alphabet[i] for i in chars_idx.tolist())
    if used_probs.numel() == 0:
        return text, 1.0
    log_sum = torch.log(used_probs).sum().item()
    conf = math.exp(log_sum / used_probs.numel())
    return text, conf


def _decode_cpu(
    indices: torch.Tensor,
    max_probs: torch.Tensor,
    blank_idx: int,
    alphabet: str,
) -> tuple[str, float]:
    """CTC decode on CPU via numpy."""
    idx_np = indices.cpu().numpy()
    prob_np = max_probs.cpu().numpy()
    non_blank = idx_np != blank_idx
    shifted = np.empty_like(idx_np)
    shifted[0] = -1
    shifted[1:] = idx_np[:-1]
    keep = non_blank & (idx_np != shifted)
    chars_idx = idx_np[keep]
    used_probs = prob_np[keep]
    valid = chars_idx < len(alphabet)
    text = "".join(alphabet[i] for i in chars_idx[valid])
    used = used_probs[valid]
    conf = _compute_confidence(used.tolist())
    return text, conf


def greedy_decode_with_conf(
    logits: Tensor,
    alphabet: str,
) -> tuple[str, float]:
    """CTC greedy decode with confidence from log-probabilities.

    Args:
        logits: (T, alphabet_size) LOG-PROBABILITIES
            (already log_softmax'd).
        alphabet: character set string.

    Returns:
        (text, confidence) — best hypothesis with geometric
        mean confidence.
    """
    text, conf, _alignment = greedy_decode_with_alignment(logits, alphabet)
    return text, conf


def greedy_decode_with_alignment(
    logits: Tensor,
    alphabet: str,
) -> tuple[str, float, list[int]]:
    """CTC greedy decode returning per-character timestep alignment.

    Args:
        logits: (T, alphabet_size) LOG-PROBABILITIES
            (already log_softmax'd).
        alphabet: character set string.

    Returns:
        (text, confidence, alignment) — decoded text, geometric
        mean confidence, and a list of timestep indices — one per
        character in *text*, indicating the CTC timestep where
        that character was first emitted.
    """
    blank_idx = len(alphabet)
    probs = logits.exp()
    max_probs, indices = probs.max(dim=1)

    # Collapse CTC: remove blanks + repeated chars
    idx_np = indices.cpu().numpy()
    prob_np = max_probs.cpu().numpy()
    T = len(idx_np)

    non_blank = idx_np != blank_idx
    shifted = np.empty_like(idx_np)
    shifted[0] = -1
    shifted[1:] = idx_np[:-1]
    keep = non_blank & (idx_np != shifted)

    chars_idx = idx_np[keep]
    used_probs = prob_np[keep]
    timesteps = np.arange(T)[keep]  # which timestep each kept entry came from

    valid = chars_idx < len(alphabet)
    chars_idx = chars_idx[valid]
    used_probs = used_probs[valid]
    timesteps = timesteps[valid]

    text = "".join(alphabet[i] for i in chars_idx.tolist())
    alignment = timesteps.tolist()

    if len(used_probs) == 0:
        return text, 1.0, alignment

    conf = _compute_confidence(used_probs.tolist())
    return text, conf, alignment


def _alphabet_indices_for(
    pattern_char: str,
    alphabet: str,
    valid_letters: set[str],
    valid_digits: set[str],
) -> set[int]:
    """Return allowed alphabet indices for a pattern character."""
    if pattern_char in ("X", "x"):
        allowed = valid_letters
    elif pattern_char in ("0", "o"):
        allowed = valid_digits
    else:
        allowed = {pattern_char}
    return {i for i, ch in enumerate(alphabet) if ch in allowed}


class BeamSearchDecoder:
    """CTC prefix beam search decoder with greedy fallback.

    Alphabet order: letters + digits.
    Blank token is at index len(alphabet).

    Supports optional pattern constraint: at each output
    position, only characters matching the pattern slot
    (letter, digit, or any) are allowed during beam
    expansion.
    """

    def __init__(
        self,
        alphabet: str,
        beam_width: int = 1,
        pattern: str | None = None,
        valid_letters: str = "",
        valid_digits: str = "0123456789",
    ) -> None:
        self.alphabet = alphabet
        self.beam_width = beam_width
        self.blank_idx = len(alphabet)
        self.pattern = pattern
        self.valid_letters = valid_letters
        self.valid_digits = valid_digits
        self._pos_mask = self._build_position_mask()

    def _build_position_mask(
        self,
    ) -> list[set[int]] | None:
        """Build per-position allowed character index sets."""
        if self.pattern is None:
            return None
        valid_l = {c for c in self.valid_letters if c.isalpha()}
        valid_d = set(self.valid_digits)
        return [
            _alphabet_indices_for(pc, self.alphabet, valid_l, valid_d)
            for pc in self.pattern
        ]

    def decode(self, logits: Tensor) -> tuple[str, float]:
        """Decode CTC logits to text and confidence."""
        results = self.decode_n(logits, n=1)
        return results[0]

    def decode_n(
        self,
        logits: Tensor,
        n: int = 1,
    ) -> list[tuple[str, float]]:
        """Decode CTC logits returning top-N hypotheses."""
        if self.beam_width <= 1:
            text, conf = self._greedy_decode(logits)
            return [(text, conf)] * min(n, 1)
        beams = self._beam_search(logits)
        return beams[:n]

    def _greedy_decode(self, logits: Tensor) -> tuple[str, float]:
        """Greedy CTC decoding."""
        return greedy_decode_with_conf(logits, self.alphabet)

    def _beam_search(self, logits: Tensor) -> list[tuple[str, float]]:
        """CTC prefix beam search in log-space.

        Standard two-state tracking (p_blank, p_non_blank)
        per prefix for correct handling of repeated chars
        separated by blank.

        If pattern is set, beam expansion is constrained:
        at output position k only characters matching
        pattern slot k are allowed.
        """
        T, V = logits.shape
        log_probs = logits
        beam: dict[str, tuple[float, float]] = {
            "": (0.0, float("-inf")),
        }
        for t in range(T):
            new_beam: dict[str, tuple[float, float]] = {}
            step_lp = log_probs[t].tolist()
            for prefix, (p_b, p_nb) in beam.items():
                self._expand_prefix(
                    prefix, p_b, p_nb, step_lp, V, new_beam
                )
            beam = _prune_beams(new_beam, self.beam_width)
        return self._finalize_beam(beam)

    def _expand_prefix(
        self,
        prefix: str,
        p_b: float,
        p_nb: float,
        step_lp: list[float],
        vocab_size: int,
        new_beam: dict[str, tuple[float, float]],
    ) -> None:
        """Expand a single prefix with all possible next characters."""
        p_total = _log_sum_exp(p_b, p_nb)
        for c in range(vocab_size):
            lp = step_lp[c]
            if self._is_blank(c):
                self._expand_blank(prefix, p_b, p_nb, lp, new_beam)
                continue
            if not self._is_allowed(c, len(prefix)):
                continue
            self._expand_character(
                prefix, c, lp, p_total, p_b, p_nb, new_beam,
            )

    def _is_blank(self, c: int) -> bool:
        """Check if character index is the blank token."""
        return c == self.blank_idx

    def _is_allowed(self, c: int, pos_k: int) -> bool:
        """Check if character is allowed at position k."""
        if self._pos_mask is None:
            return True
        if pos_k >= len(self._pos_mask):
            return False
        return c in self._pos_mask[pos_k]

    def _expand_blank(
        self,
        prefix: str,
        p_b: float,
        p_nb: float,
        lp: float,
        new_beam: dict[str, tuple[float, float]],
    ) -> None:
        """Update beam for blank token."""
        new_p_b = _log_sum_exp(p_b + lp, p_nb + lp)
        _update_beam(new_beam, prefix, new_p_b, float("-inf"))

    def _expand_character(
        self,
        prefix: str,
        c: int,
        lp: float,
        p_total: float,
        p_b: float,
        p_nb: float,
        new_beam: dict[str, tuple[float, float]],
    ) -> None:
        """Expand prefix with a non-blank character."""
        ch = self.alphabet[c]
        is_repeat = prefix and prefix[-1] == ch
        if is_repeat:
            self._expand_same_char(prefix, ch, lp, p_b, p_nb, new_beam)
        else:
            self._expand_new_char(prefix, ch, lp, p_total, new_beam)

    def _expand_same_char(
        self,
        prefix: str,
        ch: str,
        lp: float,
        p_b: float,
        p_nb: float,
        new_beam: dict[str, tuple[float, float]],
    ) -> None:
        """Handle repeated character case."""
        _update_beam(new_beam, prefix + ch, float("-inf"), p_b + lp)
        _update_beam(new_beam, prefix, float("-inf"), p_nb + lp)

    def _expand_new_char(
        self,
        prefix: str,
        ch: str,
        lp: float,
        p_total: float,
        new_beam: dict[str, tuple[float, float]],
    ) -> None:
        """Handle new character case."""
        _update_beam(
            new_beam, prefix + ch, float("-inf"), p_total + lp,
        )

    @staticmethod
    def _finalize_beam(
        beam: dict[str, tuple[float, float]],
    ) -> list[tuple[str, float]]:
        """Normalize beam scores and return top hypotheses."""
        results: list[tuple[str, float]] = []
        for prefix, (p_b, p_nb) in beam.items():
            score = _log_sum_exp(p_b, p_nb)
            results.append((prefix, math.exp(score)))
        results.sort(key=lambda x: x[1], reverse=True)
        total = sum(s for _, s in results) or 1.0
        return [(text, score / total) for text, score in results]


def _update_beam(
    beam: dict[str, tuple[float, float]],
    key: str,
    p_b: float,
    p_nb: float,
) -> None:
    """Update beam entry with new p_blank and/or p_non_blank."""
    if key in beam:
        old_b, old_nb = beam[key]
        if p_b != float("-inf"):
            new_b = _log_sum_exp(old_b, p_b)
        else:
            new_b = old_b
        if p_nb != float("-inf"):
            new_nb = _log_sum_exp(old_nb, p_nb)
        else:
            new_nb = old_nb
        beam[key] = (new_b, new_nb)
    else:
        beam[key] = (p_b, p_nb)


def _log_sum_exp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    mx = max(a, b)
    return mx + math.log1p(math.exp(min(a, b) - mx))


def _prune_beams(
    beam: dict[str, tuple[float, float]],
    beam_width: int,
) -> dict[str, tuple[float, float]]:
    """Keep top beam_width entries by total log-prob."""
    scored = sorted(
        beam.items(),
        key=lambda x: _log_sum_exp(x[1][0], x[1][1]),
        reverse=True,
    )
    return dict(scored[:beam_width])
