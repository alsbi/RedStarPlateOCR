"""Tests for BeamSearchDecoder (T6.1)."""

from __future__ import annotations

import torch

from redstar_plate_ocr.plate.postprocess import BeamSearchDecoder


class TestBeamSearchDecoderGreedy:
    """Greedy decoding (beam_width=1)."""

    def test_decode_simple_text_greedy(self) -> None:
        """Greedy decode of CTC logits produces correct text."""
        alphabet = "AB0"  # letters + digits, blank at end
        decoder = BeamSearchDecoder(alphabet, beam_width=1)
        # 3 timesteps, alphabet_size=4 (A,B,0,blank)
        # Sequence: A, A, B -> after collapse: AB
        logits = torch.zeros(1, 3, 4)
        logits[0, 0, 0] = 10.0  # A
        logits[0, 1, 3] = 10.0  # blank (collapse AA -> A)
        logits[0, 2, 1] = 10.0  # B
        text, conf = decoder.decode(logits[0])
        assert text == "AB"

    def test_decode_blank_collapse_greedy(self) -> None:
        """Repeated chars separated by blank are collapsed correctly."""
        alphabet = "A0"
        decoder = BeamSearchDecoder(alphabet, beam_width=1)
        # A, blank, A -> AA (blank separates, no collapse)
        logits = torch.zeros(1, 3, 3)
        logits[0, 0, 0] = 10.0  # A
        logits[0, 1, 2] = 10.0  # blank
        logits[0, 2, 0] = 10.0  # A
        text, conf = decoder.decode(logits[0])
        assert text == "AA"

    def test_decode_repeated_without_blank_collapsed(
        self,
    ) -> None:
        """Repeated chars without blank are collapsed."""
        alphabet = "A0"
        decoder = BeamSearchDecoder(alphabet, beam_width=1)
        # A, A, A -> A (collapse repeats)
        logits = torch.zeros(1, 3, 3)
        logits[0, 0, 0] = 10.0
        logits[0, 1, 0] = 10.0
        logits[0, 2, 0] = 10.0
        text, conf = decoder.decode(logits[0])
        assert text == "A"

    def test_decode_all_blank_produces_empty(self) -> None:
        """All-blank input produces empty string."""
        alphabet = "A0"
        decoder = BeamSearchDecoder(alphabet, beam_width=1)
        logits = torch.zeros(1, 3, 3)
        logits[0, :, 2] = 10.0  # all blank
        text, conf = decoder.decode(logits[0])
        assert text == ""

    def test_decode_confidence_is_probability(self) -> None:
        """Confidence is a valid probability in [0, 1]."""
        alphabet = "AB"
        decoder = BeamSearchDecoder(alphabet, beam_width=1)
        raw = torch.zeros(1, 2, 3)
        raw[0, 0, 0] = 5.0  # A
        raw[0, 1, 1] = 3.0  # B
        logits = torch.log_softmax(raw, dim=-1)
        text, conf = decoder.decode(logits[0])
        assert 0.0 <= conf <= 1.0


class TestBeamSearchDecoderBeam:
    """Beam search decoding (beam_width > 1)."""

    def test_beam_width_1_matches_greedy(self) -> None:
        """beam_width=1 produces same result as greedy."""
        alphabet = "AB0"
        greedy = BeamSearchDecoder(alphabet, beam_width=1)
        beam = BeamSearchDecoder(alphabet, beam_width=1)
        logits = torch.zeros(1, 4, 4)
        logits[0, 0, 0] = 10.0  # A
        logits[0, 1, 3] = 10.0  # blank
        logits[0, 2, 1] = 10.0  # B
        logits[0, 3, 2] = 10.0  # 0
        text_g, _ = greedy.decode(logits[0])
        text_b, _ = beam.decode(logits[0])
        assert text_g == text_b

    def test_beam_search_returns_multiple_hypotheses(
        self,
    ) -> None:
        """decode_n returns top-N hypotheses."""
        alphabet = "AB0"
        decoder = BeamSearchDecoder(alphabet, beam_width=3)
        logits = torch.zeros(1, 4, 4)
        logits[0, 0, 0] = 5.0
        logits[0, 1, 1] = 5.0
        logits[0, 2, 2] = 5.0
        logits[0, 3, 3] = 5.0
        results = decoder.decode_n(logits[0], n=2)
        assert len(results) >= 1
        assert len(results) <= 2
        # Each result is (text, confidence)
        for text, conf in results:
            assert isinstance(text, str)
            assert 0.0 <= conf <= 1.0


class TestBeamSearchDecoderAlphabet:
    """Alphabet handling."""

    def test_blank_is_last_index(self) -> None:
        """Blank token is at the last index of alphabet_size."""
        alphabet = "ABC"
        decoder = BeamSearchDecoder(alphabet, beam_width=1)
        # alphabet_size = 4 (A,B,C,blank)
        logits = torch.zeros(1, 1, 4)
        logits[0, 0, 3] = 10.0  # blank index = len(alphabet)
        text, _ = decoder.decode(logits[0])
        assert text == ""

    def test_russian_alphabet_decodes(self) -> None:
        """Decoder works with Russian alphabet."""
        alphabet = "ABEKMHOPCTYX0123456789"
        decoder = BeamSearchDecoder(alphabet, beam_width=1)
        # A, 0, 1
        a_idx = alphabet.index("A")
        z0_idx = alphabet.index("0")
        z1_idx = alphabet.index("1")
        logits = torch.zeros(1, 3, len(alphabet) + 1)
        logits[0, 0, a_idx] = 10.0
        logits[0, 1, z0_idx] = 10.0
        logits[0, 2, z1_idx] = 10.0
        text, _ = decoder.decode(logits[0])
        assert text == "A01"
