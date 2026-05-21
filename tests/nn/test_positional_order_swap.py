"""Tests for sinusoidal positional encoding, order-penalty loss,
and logit-based adjacent-swap correction.

These three mechanisms work together to fix adjacent same-type
character transpositions (e.g. CX → XC on an XX pattern slot):

1. **SinusoidalPositionalEncoding** — gives the LSTM an explicit
   absolute-position signal so it can distinguish horizontal order.
2. **Order-penalty loss** — training signal that penalises wrong
   temporal order of emission peaks for same-type adjacent pairs.
3. **adjacent_swap_correct** — post-processing that flips swapped
   pairs when CTC logit evidence supports it.
"""

from __future__ import annotations

import math

import torch

from redstar_plate_ocr.nn.losses import CombinedLoss
from redstar_plate_ocr.nn.positional import SinusoidalPositionalEncoding
from redstar_plate_ocr.nn.types import ModelOutput
from redstar_plate_ocr.plate.config import (
    PlateConfig,
    RegionConfig,
    ValidChars,
)
from redstar_plate_ocr.plate.confusion import adjacent_swap_correct

# ── Helpers ───────────────────────────────────────────────────────


def _make_plate_config() -> PlateConfig:
    """Minimal plate config with RU-like patterns."""
    return PlateConfig(
        regions={
            "RU": RegionConfig(
                pattern=["X000XX00"],
                valid_chars=ValidChars(
                    letters="ABEKMHOPCTYX",
                    digits="0123456789",
                ),
            ),
        }
    )


# ── SinusoidalPositionalEncoding tests ────────────────────────────


class TestSinusoidalPositionalEncoding:
    """Tests for SinusoidalPositionalEncoding."""

    def test_output_shape(self):
        """PE preserves input shape."""
        d = 384
        pe = SinusoidalPositionalEncoding(d_model=d, max_len=256)
        x = torch.randn(2, 48, d)
        out = pe(x)
        assert out.shape == (2, 48, d)

    def test_additive_not_multiplicative(self):
        """PE adds to input, not overwrites it."""
        d = 64
        pe = SinusoidalPositionalEncoding(d_model=d, max_len=256, dropout=0.0)
        x = torch.zeros(1, 10, d)
        out = pe(x)
        # Output should be exactly the PE table (since x=0)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_different_positions_different_encoding(self):
        """Different timesteps receive different PE."""
        d = 64
        pe = SinusoidalPositionalEncoding(d_model=d, max_len=256, dropout=0.0)
        x = torch.zeros(1, 20, d)
        out = pe(x)
        # Each timestep should be different
        for i in range(19):
            assert not torch.allclose(out[0, i], out[0, i + 1], atol=1e-6), (
                f"PE at t={i} and t={i + 1} are identical"
            )

    def test_same_position_same_encoding(self):
        """Same position across batch gets the same PE."""
        d = 64
        pe = SinusoidalPositionalEncoding(d_model=d, max_len=256, dropout=0.0)
        x = torch.zeros(4, 10, d)
        out = pe(x)
        # All batch items at position 3 should have the same encoding
        for b in range(1, 4):
            assert torch.allclose(out[0, 3], out[b, 3], atol=1e-6)

    def test_sinusoidal_values(self):
        """Verify actual sin/cos values at known positions."""
        d = 8
        pe_module = SinusoidalPositionalEncoding(
            d_model=d, max_len=10, dropout=0.0
        )
        pe_table = pe_module._pe  # (10, 8)

        # Position 0: sin(0*freq) and cos(0*freq) for each pair
        for j in range(d // 2):
            freq = 10000.0 ** (-2 * j / d)
            # Even index = sin
            assert abs(pe_table[0, 2 * j].item() - math.sin(0 * freq)) < 1e-5
            # Odd index = cos
            cos_val = pe_table[0, 2 * j + 1].item()
            assert abs(cos_val - math.cos(0 * freq)) < 1e-5

    def test_gradient_flows_through(self):
        """PE is additive and doesn't block gradients."""
        d = 64
        pe = SinusoidalPositionalEncoding(d_model=d, max_len=256, dropout=0.0)
        x = torch.randn(1, 10, d, requires_grad=True)
        out = pe(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_dropout_enabled(self):
        """With dropout > 0, outputs differ across calls in training mode."""
        d = 64
        pe = SinusoidalPositionalEncoding(d_model=d, max_len=256, dropout=0.5)
        pe.train()
        x = torch.randn(1, 10, d)
        out1 = pe(x)
        out2 = pe(x)
        # With 50% dropout, outputs should differ (probabilistically)
        # Not guaranteed for every call, but extremely likely with 640 elements
        assert not torch.allclose(out1, out2, atol=1e-7)

    def test_no_dropout_in_eval(self):
        """No dropout in eval mode."""
        d = 64
        pe = SinusoidalPositionalEncoding(d_model=d, max_len=256, dropout=0.5)
        pe.eval()
        x = torch.randn(1, 10, d)
        out1 = pe(x)
        out2 = pe(x)
        assert torch.allclose(out1, out2, atol=1e-7)

    def test_longer_sequence_than_max_len(self):
        """Handles T > max_len gracefully (clamps PE table)."""
        d = 64
        pe = SinusoidalPositionalEncoding(d_model=d, max_len=10, dropout=0.0)
        x = torch.randn(1, 20, d)
        out = pe(x)
        # Should still produce output of correct shape
        assert out.shape == (1, 20, d)
        # Positions 0-9 should have PE, 10-19 should have no PE added
        # (falls back to zero PE beyond max_len)


# ── Order-penalty loss tests ─────────────────────────────────────


class TestOrderPenalty:
    """Tests for the differentiable order-penalty component of CombinedLoss."""

    def test_zero_weight_no_penalty(self):
        """With order_weight=0, penalty is always 0."""
        cfg = _make_plate_config()
        loss_fn = CombinedLoss(
            cfg,
            order_weight=0.0,
            order_margin=1.0,
        )
        T, V = 48, cfg.union_alphabet_size
        ctc_logits = torch.randn(1, T, V)
        input_lengths = torch.tensor([T])
        result = loss_fn._compute_order_penalty(
            ctc_logits, ["AB"], input_lengths
        )
        assert result.item() == 0.0

    def test_correct_order_no_penalty(self):
        """When soft peaks are in correct order, no penalty."""
        cfg = _make_plate_config()
        loss_fn = CombinedLoss(
            cfg,
            order_weight=1.0,
            order_margin=1.0,
        )
        # Build logits where 'A' peaks at t=5, 'B' peaks at t=10
        union = cfg.union_alphabet
        a_idx = union.index("A")
        b_idx = union.index("B")
        T = 48
        logits = torch.full((1, T, cfg.union_alphabet_size), -5.0)
        logits[0, 5, a_idx] = 10.0  # A peaks at t=5
        logits[0, 10, b_idx] = 10.0  # B peaks at t=10
        input_lengths = torch.tensor([T])

        result = loss_fn._compute_order_penalty(logits, ["AB"], input_lengths)
        # soft_peak_A ≈ 5, soft_peak_B ≈ 10
        # diff = 5 - 10 = -5, penalty = max(-5 - 1, 0) = 0
        assert result.item() == 0.0

    def test_swapped_order_penalty(self):
        """When soft peaks are in wrong order, penalty is positive."""
        cfg = _make_plate_config()
        loss_fn = CombinedLoss(
            cfg,
            order_weight=1.0,
            order_margin=1.0,
        )
        union = cfg.union_alphabet
        a_idx = union.index("A")
        b_idx = union.index("B")
        T = 48
        logits = torch.full((1, T, cfg.union_alphabet_size), -5.0)
        logits[0, 10, a_idx] = 10.0  # A peaks at t=10
        logits[0, 5, b_idx] = 10.0  # B peaks at t=5 (wrong order!)
        input_lengths = torch.tensor([T])

        result = loss_fn._compute_order_penalty(logits, ["AB"], input_lengths)
        # soft_peak_A ≈ 10, soft_peak_B ≈ 5
        # diff = 10 - 5 = 5, penalty = max(5 - 1, 0) = 4.0
        assert result.item() > 0.0
        # Verify approximate magnitude (should be ~4)
        assert 3.0 < result.item() < 5.0

    def test_cross_type_no_penalty(self):
        """Cross-type pairs (letter-digit) are not penalised."""
        cfg = _make_plate_config()
        loss_fn = CombinedLoss(
            cfg,
            order_weight=1.0,
            order_margin=1.0,
        )
        union = cfg.union_alphabet
        a_idx = union.index("A")
        d_idx = union.index("0")
        T = 48
        # A (letter) at t=10, 0 (digit) at t=5 — wrong order but different type
        logits = torch.full((1, T, cfg.union_alphabet_size), -5.0)
        logits[0, 10, a_idx] = 10.0
        logits[0, 5, d_idx] = 10.0
        input_lengths = torch.tensor([T])

        result = loss_fn._compute_order_penalty(logits, ["A0"], input_lengths)
        assert result.item() == 0.0

    def test_identical_chars_no_penalty(self):
        """Identical adjacent characters (AA) are not penalised."""
        cfg = _make_plate_config()
        loss_fn = CombinedLoss(
            cfg,
            order_weight=1.0,
            order_margin=1.0,
        )
        T = 48
        logits = torch.randn(1, T, cfg.union_alphabet_size)
        input_lengths = torch.tensor([T])

        result = loss_fn._compute_order_penalty(logits, ["AA"], input_lengths)
        assert result.item() == 0.0

    def test_penalty_is_differentiable(self):
        """Order penalty has gradients flowing back to ctc_output."""
        cfg = _make_plate_config()
        loss_fn = CombinedLoss(
            cfg,
            order_weight=1.0,
            order_margin=1.0,
        )
        union = cfg.union_alphabet
        a_idx = union.index("A")
        b_idx = union.index("B")
        T = 48
        V = cfg.union_alphabet_size

        # Swapped peaks — will produce nonzero penalty
        logits = torch.full((1, T, V), -5.0, requires_grad=True)
        logits_data = logits.data
        logits_data[0, 10, a_idx] = 10.0
        logits_data[0, 5, b_idx] = 10.0
        input_lengths = torch.tensor([T])

        penalty = loss_fn._compute_order_penalty(logits, ["AB"], input_lengths)
        assert penalty.item() > 0.0

        # Gradient must flow through
        penalty.backward()
        assert logits.grad is not None
        # At least some gradients should be nonzero
        assert logits.grad.abs().sum().item() > 0.0

    def test_margin_tolerance(self):
        """Small violations within margin are not penalised."""
        cfg = _make_plate_config()
        loss_fn = CombinedLoss(
            cfg,
            order_weight=1.0,
            order_margin=5.0,  # large margin
        )
        union = cfg.union_alphabet
        a_idx = union.index("A")
        b_idx = union.index("B")
        T = 48
        # A peaks at t=8, B peaks at t=5 → diff ≈ 3 < margin 5
        logits = torch.full((1, T, cfg.union_alphabet_size), -5.0)
        logits[0, 8, a_idx] = 10.0
        logits[0, 5, b_idx] = 10.0
        input_lengths = torch.tensor([T])

        result = loss_fn._compute_order_penalty(logits, ["AB"], input_lengths)
        # diff ≈ 3, margin = 5 → penalty = max(3 - 5, 0) = 0
        assert result.item() == 0.0

    def test_penalty_in_forward(self):
        """Order penalty appears in CombinedLoss.forward() output."""
        cfg = _make_plate_config()
        loss_fn = CombinedLoss(
            cfg,
            order_weight=0.5,
            order_margin=1.0,
        )
        union = cfg.union_alphabet
        a_idx = union.index("A")
        b_idx = union.index("B")
        T = 48
        V = cfg.union_alphabet_size

        # Fake model output with swapped peaks
        logits_swapped = torch.full((1, T, V), -5.0)
        logits_swapped[0, 10, a_idx] = 10.0
        logits_swapped[0, 5, b_idx] = 10.0

        model_output = ModelOutput(
            format_logits=torch.randn(1, 3),
            country_logits=torch.randn(1, 1),
            ctc_output=logits_swapped,
            content_mask=torch.ones(1, T, dtype=torch.long),
            plate_types=["standard"],
            char_aux_logits=None,
        )
        result = loss_fn(
            model_output,
            gt_format=torch.tensor([0]),
            gt_country=torch.tensor([0]),
            gt_texts=["AB"],
            input_lengths=torch.tensor([T]),
        )
        assert "order" in result
        assert result["order"].item() > 0.0

    def test_separator_not_treated_as_letter(self):
        """Non-alpha chars in letters (e.g. '-') are not same-type
        with real letters, so no order penalty is applied."""
        cfg = PlateConfig(
            regions={
                "BY": RegionConfig(
                    pattern=["0000XX-0"],
                    valid_chars=ValidChars(
                        letters="ABEKMHOPCTYX-",
                        digits="0123456789",
                    ),
                ),
            }
        )
        loss_fn = CombinedLoss(
            cfg,
            order_weight=1.0,
            order_margin=1.0,
        )
        # Pair (A, -) should NOT be same-type
        assert not loss_fn._same_type_pairs.get(("A", "-"), False)
        # Pair (-, -) should NOT be same-type either
        assert not loss_fn._same_type_pairs.get(("-", "-"), False)
        # Pair (A, B) should still be same-type
        assert loss_fn._same_type_pairs.get(("A", "B"), False)

        # Also verify via _compute_order_penalty that a text with '-'
        # adjacent to a letter doesn't trigger a penalty
        union = cfg.union_alphabet
        a_idx = union.index("A")
        dash_idx = union.index("-")
        T = 48
        # A at t=10, - at t=5 (reversed) — but NOT same-type
        logits = torch.full((1, T, cfg.union_alphabet_size), -5.0)
        logits[0, 10, a_idx] = 10.0
        logits[0, 5, dash_idx] = 10.0
        input_lengths = torch.tensor([T])
        result = loss_fn._compute_order_penalty(logits, ["A-"], input_lengths)
        assert result.item() == 0.0


# ── Adjacent-swap corrector tests ────────────────────────────────


class TestAdjacentSwapCorrect:
    """Tests for adjacent_swap_correct postprocessor."""

    def test_no_logits_returns_unchanged(self):
        """Without CTC logits, text is returned unchanged."""
        result = adjacent_swap_correct(
            "XC",
            patterns=["XX00"],
            valid_letters="ABCX",
            valid_digits="0123",
            ctc_logits=None,
            ctc_alignment=[0, 5],
            alphabet="ABCX0123",
        )
        assert result == "XC"

    def test_no_alignment_returns_unchanged(self):
        """Without alignment, text is returned unchanged."""
        logits = torch.randn(10, 10)
        result = adjacent_swap_correct(
            "XC",
            patterns=["XX00"],
            valid_letters="ABCX",
            valid_digits="0123",
            ctc_logits=logits,
            ctc_alignment=None,
            alphabet="ABCX0123",
        )
        assert result == "XC"

    def test_high_confidence_no_swap(self):
        """High-confidence predictions are not corrected."""
        alphabet = "ABCX0123"
        logits = torch.zeros(10, len(alphabet))
        c_idx = alphabet.index("C")
        x_idx = alphabet.index("X")
        # Make swapped score higher
        logits[0, x_idx] = 2.0
        logits[5, c_idx] = 2.0
        logits[0, c_idx] = 3.0
        logits[5, x_idx] = 3.0

        result = adjacent_swap_correct(
            "XC",
            patterns=["XX"],
            valid_letters="ABCX",
            valid_digits="0123",
            ctc_logits=logits,
            ctc_alignment=[0, 5],
            alphabet=alphabet,
            text_confidence=0.99,  # too confident
        )
        assert result == "XC"

    def test_swap_applied_when_logits_support(self):
        """Swap applied when logits strongly prefer swapped order."""
        alphabet = "ABCX0123"
        logits = torch.zeros(10, len(alphabet))
        c_idx = alphabet.index("C")
        x_idx = alphabet.index("X")
        # Current: X at t=0, C at t=5 → score = logits[0,X] + logits[5,C]
        logits[0, x_idx] = 1.0
        logits[5, c_idx] = 1.0
        # Swapped: C at t=0, X at t=5 → score = logits[0,C] + logits[5,X]
        logits[0, c_idx] = 5.0
        logits[5, x_idx] = 5.0
        # Swapped score (10) >> current score (2) + margin (0.3)

        result = adjacent_swap_correct(
            "XC",
            patterns=["XX"],
            valid_letters="ABCX",
            valid_digits="0123",
            ctc_logits=logits,
            ctc_alignment=[0, 5],
            alphabet=alphabet,
            text_confidence=0.5,
            swap_margin=0.3,
        )
        assert result == "CX"

    def test_no_swap_when_margin_not_met(self):
        """Swap not applied when logit advantage is below margin."""
        alphabet = "ABCX0123"
        logits = torch.zeros(10, len(alphabet))
        c_idx = alphabet.index("C")
        x_idx = alphabet.index("X")
        # Swapped score only slightly higher
        logits[0, x_idx] = 2.0
        logits[5, c_idx] = 2.0
        logits[0, c_idx] = 2.5
        logits[5, x_idx] = 2.5

        result = adjacent_swap_correct(
            "XC",
            patterns=["XX"],
            valid_letters="ABCX",
            valid_digits="0123",
            ctc_logits=logits,
            ctc_alignment=[0, 5],
            alphabet=alphabet,
            text_confidence=0.5,
            swap_margin=2.0,  # high margin
        )
        assert result == "XC"

    def test_cross_type_not_swapped(self):
        """Cross-type pairs (letter-digit) are not swapped."""
        alphabet = "ABCX0123"
        logits = torch.zeros(10, len(alphabet))
        a_idx = alphabet.index("A")
        d_idx = alphabet.index("0")
        logits[0, a_idx] = 1.0
        logits[5, d_idx] = 1.0
        logits[0, d_idx] = 5.0
        logits[5, a_idx] = 5.0

        result = adjacent_swap_correct(
            "A0",
            patterns=["X0"],
            valid_letters="ABCX",
            valid_digits="0123",
            ctc_logits=logits,
            ctc_alignment=[0, 5],
            alphabet=alphabet,
            text_confidence=0.5,
        )
        assert result == "A0"  # not swapped

    def test_identical_chars_not_swapped(self):
        """Identical adjacent characters are not swapped."""
        alphabet = "ABCX0123"
        logits = torch.zeros(10, len(alphabet))
        x_idx = alphabet.index("X")
        logits[0, x_idx] = 5.0
        logits[5, x_idx] = 5.0

        result = adjacent_swap_correct(
            "XX",
            patterns=["XX"],
            valid_letters="ABCX",
            valid_digits="0123",
            ctc_logits=logits,
            ctc_alignment=[0, 5],
            alphabet=alphabet,
            text_confidence=0.5,
        )
        assert result == "XX"

    def test_alignment_length_mismatch(self):
        """Returns unchanged when alignment length != text length."""
        alphabet = "ABCX0123"
        logits = torch.zeros(10, len(alphabet))
        result = adjacent_swap_correct(
            "XC",
            patterns=["XX"],
            valid_letters="ABCX",
            valid_digits="0123",
            ctc_logits=logits,
            ctc_alignment=[0],  # only 1 entry for 2-char text
            alphabet=alphabet,
        )
        assert result == "XC"

    def test_short_text_unchanged(self):
        """Single-character text is returned unchanged."""
        alphabet = "ABCX"
        result = adjacent_swap_correct(
            "X",
            patterns=["X"],
            valid_letters="ABCX",
            valid_digits="0123",
            ctc_logits=torch.zeros(10, len(alphabet)),
            ctc_alignment=[0],
            alphabet=alphabet,
        )
        assert result == "X"

    def test_multiple_swaps_in_sequence(self):
        """Multiple swap opportunities can be fixed in one pass."""
        alphabet = "ABCX"
        logits = torch.zeros(20, len(alphabet))
        c_idx = alphabet.index("C")
        x_idx = alphabet.index("X")
        # First pair: XC → should become CX
        logits[0, x_idx] = 1.0
        logits[5, c_idx] = 1.0
        logits[0, c_idx] = 5.0
        logits[5, x_idx] = 5.0
        # Second pair: XA → should become AX
        a_idx = alphabet.index("A")
        logits[6, x_idx] = 1.0
        logits[10, a_idx] = 1.0
        logits[6, a_idx] = 5.0
        logits[10, x_idx] = 5.0

        result = adjacent_swap_correct(
            "XCXA",
            patterns=["XXXX"],
            valid_letters="ABCX",
            valid_digits="0123",
            ctc_logits=logits,
            ctc_alignment=[0, 5, 6, 10],
            alphabet=alphabet,
            text_confidence=0.5,
        )
        assert result == "CXAX"
