"""T0.2: Verify greedy_decode equivalence between pipeline.utils
and postprocess."""

import torch

from redstar_plate_ocr.pipeline.utils import greedy_decode
from redstar_plate_ocr.plate.postprocess import BeamSearchDecoder

ALPHABET = "ABEKMHOPCTYX0123456789"


def _make_random_logits(
    seq_len: int,
    alphabet_size: int,
    seed: int,
) -> torch.Tensor:
    """Generate random log-probabilities for testing."""
    gen = torch.Generator().manual_seed(seed)
    raw = torch.randn(seq_len, alphabet_size, generator=gen)
    return torch.log_softmax(raw, dim=-1)


class TestGreedyDecodeEquivalence:
    """Both implementations must produce identical text."""

    def test_simple_text(self):
        logits = _make_random_logits(20, len(ALPHABET) + 1, seed=42)
        text_eval = greedy_decode(logits, ALPHABET)
        decoder = BeamSearchDecoder(ALPHABET, beam_width=1)
        text_pp, _conf = decoder._greedy_decode(logits)
        assert text_eval == text_pp

    def test_all_blank(self):
        """All-blank logits produce empty string from both."""
        alphabet_size = len(ALPHABET) + 1
        logits = torch.zeros(10, alphabet_size)
        logits[:, -1] = 100.0  # blank dominates
        text_eval = greedy_decode(logits, ALPHABET)
        decoder = BeamSearchDecoder(ALPHABET, beam_width=1)
        text_pp, _conf = decoder._greedy_decode(logits)
        assert text_eval == text_pp

    def test_repeated_chars(self):
        """Repeated characters should be collapsed identically."""
        alphabet_size = len(ALPHABET) + 1
        logits = torch.zeros(8, alphabet_size)
        logits[0, 0] = 50.0  # A
        logits[1, 0] = 50.0  # A (repeat)
        logits[2, 0] = 50.0  # A (repeat)
        logits[3, -1] = 50.0  # blank
        logits[4, 1] = 50.0  # B
        logits[5, -1] = 50.0  # blank
        logits[6, 2] = 50.0  # E
        logits[7, -1] = 50.0  # blank
        text_eval = greedy_decode(logits, ALPHABET)
        decoder = BeamSearchDecoder(ALPHABET, beam_width=1)
        text_pp, _conf = decoder._greedy_decode(logits)
        assert text_eval == text_pp

    def test_multiple_seeds(self):
        """Test with multiple random seeds for robustness."""
        for seed in range(10):
            logits = _make_random_logits(
                15,
                len(ALPHABET) + 1,
                seed=seed,
            )
            text_eval = greedy_decode(logits, ALPHABET)
            decoder = BeamSearchDecoder(ALPHABET, beam_width=1)
            text_pp, _conf = decoder._greedy_decode(logits)
            assert text_eval == text_pp, (
                f"Seed {seed}: '{text_eval}' != '{text_pp}'"
            )

    def test_confidence_is_valid(self):
        """BeamSearchDecoder._greedy_decode returns valid
        confidence."""
        logits = _make_random_logits(10, len(ALPHABET) + 1, seed=42)
        decoder = BeamSearchDecoder(ALPHABET, beam_width=1)
        _text, conf = decoder._greedy_decode(logits)
        assert 0.0 <= conf <= 1.0
