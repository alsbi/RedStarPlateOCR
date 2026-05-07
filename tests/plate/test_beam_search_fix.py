"""Tests for beam search: blank/non-blank tracking and pattern constraint."""

from __future__ import annotations

import torch

from redstar_plate_ocr.plate.postprocess import BeamSearchDecoder


class TestBeamSearchBlankNonBlank:
    """BUG-5 fix: beam search tracks blank/non-blank states."""

    def test_beam_search_handles_repeated_chars(
        self,
    ) -> None:
        """AB-blank-AB produces 'ABAB' with beam_width > 1."""
        alphabet = "AB0"
        decoder = BeamSearchDecoder(alphabet, beam_width=10)
        logits = torch.full((6, 4), -20.0)
        logits[0, 0] = 0.0  # A
        logits[1, 1] = 0.0  # B
        logits[2, 3] = 0.0  # blank
        logits[3, 0] = 0.0  # A
        logits[4, 1] = 0.0  # B
        logits[5, 3] = 0.0  # blank
        logits = torch.log_softmax(logits, dim=-1)
        text, _ = decoder.decode(logits)
        assert text == "ABAB"

    def test_repeated_chars_without_blank_collapsed(
        self,
    ) -> None:
        """Repeated A (no blank between) produces 'A'."""
        alphabet = "A0"
        decoder = BeamSearchDecoder(alphabet, beam_width=5)
        logits = torch.full((3, 3), -20.0)
        logits[0, 0] = 0.0
        logits[1, 0] = 0.0
        logits[2, 0] = 0.0
        logits = torch.log_softmax(logits, dim=-1)
        text, _ = decoder.decode(logits)
        assert text == "A"

    def test_beam_search_multiple_hypotheses(
        self,
    ) -> None:
        """decode_n returns multiple valid hypotheses."""
        alphabet = "AB0"
        decoder = BeamSearchDecoder(alphabet, beam_width=5)
        logits = torch.full((4, 4), -5.0)
        logits = torch.log_softmax(logits, dim=-1)
        results = decoder.decode_n(logits, n=3)
        assert len(results) >= 1
        for text, conf in results:
            assert isinstance(text, str)
            assert 0.0 <= conf <= 1.0


class TestBeamSearchPatternConstrained:
    """Pattern-constrained beam search."""

    def test_pattern_allows_correct_types(
        self,
    ) -> None:
        """Pattern X0 allows letter at pos 0, digit at pos 1."""
        alphabet = "AB012"
        decoder = BeamSearchDecoder(
            alphabet,
            beam_width=5,
            pattern="X0",
            valid_letters="AB",
            valid_digits="012",
        )
        logits = torch.full((2, len(alphabet) + 1), -20.0)
        logits[0, 0] = 0.0  # A at pos 0 (letter = OK)
        logits[1, 2] = 0.0  # '0' at pos 1 (digit = OK)
        logits = torch.log_softmax(logits, dim=-1)
        text, _ = decoder.decode(logits)
        assert text == "A0"

    def test_pattern_blocks_digit_at_letter_pos(
        self,
    ) -> None:
        """Pattern X0: digit at position 0 is blocked."""
        alphabet = "AB012"
        decoder = BeamSearchDecoder(
            alphabet,
            beam_width=5,
            pattern="X0",
            valid_letters="AB",
            valid_digits="012",
        )
        logits = torch.full((2, len(alphabet) + 1), -5.0)
        logits[0, 2] = 5.0  # '0' at pos 0 (digit, NOT letter)
        logits[1, 0] = 5.0  # A at pos 1 (letter, NOT digit)
        logits = torch.log_softmax(logits, dim=-1)
        text, _ = decoder.decode(logits)
        assert not (len(text) >= 1 and text[0] in "012")

    def test_no_pattern_allows_all(
        self,
    ) -> None:
        """Without pattern, all chars allowed."""
        alphabet = "A0"
        decoder = BeamSearchDecoder(alphabet, beam_width=5)
        logits = torch.full((2, 3), -20.0)
        logits[0, 0] = 0.0  # A
        logits[1, 1] = 0.0  # 0
        logits = torch.log_softmax(logits, dim=-1)
        text, _ = decoder.decode(logits)
        assert text == "A0"


class TestBeamSearchLogSpace:
    """BUG-6 fix: beam search operates in log-space."""

    def test_long_sequence_no_underflow(
        self,
    ) -> None:
        """48-step sequence doesn't cause underflow."""
        alphabet = "A0"
        decoder = BeamSearchDecoder(alphabet, beam_width=3)
        T = 48
        V = len(alphabet) + 1
        logits = torch.full((T, V), -5.0)
        for t in range(T):
            if t % 3 == 0:
                logits[t, 0] = 5.0
            elif t % 3 == 1:
                logits[t, 1] = 5.0
            else:
                logits[t, 2] = 5.0
        logits = torch.log_softmax(logits, dim=-1)
        text, conf = decoder.decode(logits)
        assert len(text) > 0
        assert 0.0 <= conf <= 1.0

    def test_beam_search_confidence_valid(
        self,
    ) -> None:
        """Beam search confidence is a valid probability."""
        alphabet = "AB0"
        decoder = BeamSearchDecoder(alphabet, beam_width=5)
        logits = torch.full((4, 4), -5.0)
        logits = torch.log_softmax(logits, dim=-1)
        _, conf = decoder.decode(logits)
        assert 0.0 <= conf <= 1.0
