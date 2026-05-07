"""Tests for CombinedLoss and text_to_indices."""

import torch
import torch.nn.functional as F

from redstar_plate_ocr.nn.losses import CombinedLoss, text_to_indices
from redstar_plate_ocr.nn.model import ModelOutput
from redstar_plate_ocr.plate.config import PlateConfig


def _make_model_output(
    cfg: PlateConfig,
    B: int,
    T: int,
) -> ModelOutput:
    """Create a fake ModelOutput for testing."""
    union_size = cfg.union_alphabet_size
    fmt_logits = torch.randn(
        B,
        2,
        requires_grad=True,
    )
    ctry_logits = torch.randn(
        B,
        cfg.num_countries + 1,
        requires_grad=True,
    )
    ctc_in = torch.randn(
        B,
        T,
        union_size,
        requires_grad=True,
    )
    ctc_out = torch.log_softmax(ctc_in, dim=-1)
    return ModelOutput(
        format_logits=fmt_logits,
        country_logits=ctry_logits,
        ctc_output=ctc_out,
        content_mask=torch.ones(B, 1, 1, 1),
        plate_types=["standard"] * B,
    )


class TestTextToIndices:
    """Tests for text_to_indices helper."""

    def test_text_to_indices_basic(self):
        alphabet = "ABEKMHOPCTYX0123456789"
        result = text_to_indices("A123", alphabet)
        # A=0, 1=13, 2=14, 3=15
        assert result == [0, 13, 14, 15]

    def test_text_to_indices_skip_unknown(self):
        alphabet = "ABEKMHOPCTYX0123456789"
        result = text_to_indices("AZ1", alphabet)
        # A=0, Z not in alphabet (skipped), 1=13
        assert result == [0, 13]

    def test_text_to_indices_empty(self):
        result = text_to_indices("", "ABC")
        assert result == []


class TestCombinedLossFormat:
    """Tests for L_format component."""

    def test_format_loss_positive(self, plate_config: PlateConfig):
        cfg = plate_config
        loss_fn = CombinedLoss(cfg)
        B, T = 4, 48
        out = _make_model_output(cfg, B, T)
        gt_format = torch.tensor([0, 1, 0, 1])
        gt_country = torch.tensor([0, 0, 1, 1])
        gt_texts = ["A123AA12", "B456BB34", "K789CC56", "M012KK78"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        losses = loss_fn(
            out,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )

        assert losses["format"] > 0

    def test_format_loss_gradients(self, plate_config: PlateConfig):
        cfg = plate_config
        loss_fn = CombinedLoss(cfg)
        B, T = 2, 48
        out = _make_model_output(cfg, B, T)
        gt_format = torch.tensor([0, 1])
        gt_country = torch.tensor([0, 1])
        gt_texts = ["A123AA12", "K567BB34"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        losses = loss_fn(
            out,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )
        losses["total"].backward()

        assert out.format_logits.grad is not None
        assert out.format_logits.grad.abs().sum() > 0


class TestCombinedLossCountry:
    """Tests for L_country component."""

    def test_country_loss_positive(self, plate_config: PlateConfig):
        cfg = plate_config
        loss_fn = CombinedLoss(cfg, label_smoothing=0.01)
        B, T = 4, 48
        out = _make_model_output(cfg, B, T)
        gt_format = torch.tensor([0, 1, 0, 1])
        gt_country = torch.tensor([0, 0, 1, 1])
        gt_texts = ["A123AA12", "B456BB34", "K789CC56", "M012KK78"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        losses = loss_fn(
            out,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )

        assert losses["country"] > 0

    def test_country_loss_with_label_smoothing(
        self, plate_config: PlateConfig
    ):
        cfg = plate_config
        loss_fn = CombinedLoss(cfg, label_smoothing=0.1)
        B, T = 2, 48
        out = _make_model_output(cfg, B, T)
        gt_format = torch.tensor([0, 0])
        gt_country = torch.tensor([0, 1])
        gt_texts = ["A123AA12", "K567BB34"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        losses = loss_fn(
            out,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )

        assert losses["country"] > 0


class TestCombinedLossCTC:
    """Tests for L_ctc component."""

    def test_ctc_loss_positive(self, plate_config: PlateConfig):
        cfg = plate_config
        loss_fn = CombinedLoss(cfg)
        B, T = 2, 48
        out = _make_model_output(cfg, B, T)
        gt_format = torch.tensor([0, 0])
        gt_country = torch.tensor([0, 0])
        gt_texts = ["A123AA12", "K567BB34"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        losses = loss_fn(
            out,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )

        assert losses["ctc"] > 0

    def test_ctc_loss_gradients(self, plate_config: PlateConfig):
        cfg = plate_config
        loss_fn = CombinedLoss(cfg)
        B, T = 2, 48
        out = _make_model_output(cfg, B, T)
        gt_format = torch.tensor([0, 0])
        gt_country = torch.tensor([0, 0])
        gt_texts = ["A123AA12", "K567BB34"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        losses = loss_fn(
            out,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )
        out.ctc_output.retain_grad()
        losses["total"].backward()

        # Real gradient flows through ctc_output
        assert out.ctc_output.grad is not None
        assert out.ctc_output.grad.abs().sum() > 0


class TestCombinedLossTotal:
    """Tests for total loss composition."""

    def test_total_equals_weighted_sum(self, plate_config: PlateConfig):
        cfg = plate_config
        alpha, beta, gamma = 1.0, 0.7, 1.0
        loss_fn = CombinedLoss(
            cfg,
            format_weight=alpha,
            country_weight=beta,
            ctc_weight=gamma,
        )
        B, T = 2, 48
        out = _make_model_output(cfg, B, T)
        gt_format = torch.tensor([0, 1])
        gt_country = torch.tensor([0, 1])
        gt_texts = ["A123AA12", "K567BB34"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        losses = loss_fn(
            out,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )

        expected = (
            alpha * losses["format"]
            + beta * losses["country"]
            + gamma * losses["ctc"]
        )
        assert torch.isclose(losses["total"], expected)


class TestPerSampleCTC:
    """Tests for per-sample CTC loss."""

    def test_compute_ctc_loss_returns_tuple(self, plate_config: PlateConfig):
        cfg = plate_config
        loss_fn = CombinedLoss(cfg)
        B, T = 2, 48
        out = _make_model_output(cfg, B, T)
        gt_texts = ["A123AA12", "K567BB34"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        result = loss_fn._compute_ctc_loss(out, gt_texts, input_lengths)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_per_sample_shape_matches_batch(self, plate_config: PlateConfig):
        cfg = plate_config
        loss_fn = CombinedLoss(cfg)
        B, T = 4, 48
        out = _make_model_output(cfg, B, T)
        gt_texts = [
            "A123AA12",
            "K567BB34",
            "M789CC56",
            "H012DD78",
        ]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        mean_loss, per_sample = loss_fn._compute_ctc_loss(
            out,
            gt_texts,
            input_lengths,
        )

        assert per_sample.shape == (B,)

    def test_mean_equals_per_sample_mean(self, plate_config: PlateConfig):
        cfg = plate_config
        loss_fn = CombinedLoss(cfg)
        B, T = 3, 48
        out = _make_model_output(cfg, B, T)
        gt_texts = ["A123AA12", "K567BB34", "M789CC56"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        mean_loss, per_sample = loss_fn._compute_ctc_loss(
            out,
            gt_texts,
            input_lengths,
        )

        assert torch.isclose(mean_loss, per_sample.mean())

    def test_per_sample_values_positive(self, plate_config: PlateConfig):
        cfg = plate_config
        loss_fn = CombinedLoss(cfg)
        B, T = 2, 48
        out = _make_model_output(cfg, B, T)
        gt_texts = ["A123AA12", "K567BB34"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        _, per_sample = loss_fn._compute_ctc_loss(
            out,
            gt_texts,
            input_lengths,
        )

        assert (per_sample >= 0).all()

    def test_ctc_per_sample_all_finite(self, plate_config: PlateConfig):
        """No inf/nan in per_sample CTC losses."""
        cfg = plate_config
        loss_fn = CombinedLoss(cfg)
        B, T = 2, 48
        out = _make_model_output(cfg, B, T)
        gt_texts = ["A123AA12", "K567BB34"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        _, per_sample = loss_fn._compute_ctc_loss(
            out,
            gt_texts,
            input_lengths,
        )

        assert torch.isfinite(per_sample).all()


class TestSynergyBonus:
    """Tests for synergy bonus."""

    def test_synergy_weight_zero_backward_compat(
        self, plate_config: PlateConfig
    ):
        """With synergy_weight=0, total equals weighted sum."""
        cfg = plate_config
        alpha, beta, gamma = 1.0, 0.7, 1.0
        loss_fn = CombinedLoss(
            cfg,
            format_weight=alpha,
            country_weight=beta,
            ctc_weight=gamma,
            synergy_weight=0.0,
        )
        B, T = 2, 48
        out = _make_model_output(cfg, B, T)
        gt_format = torch.tensor([0, 1])
        gt_country = torch.tensor([0, 1])
        gt_texts = ["A123AA12", "K567BB34"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        losses = loss_fn(
            out,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )

        expected = (
            alpha * losses["format"]
            + beta * losses["country"]
            + gamma * losses["ctc"]
        )
        assert torch.isclose(losses["total"], expected)
        assert torch.isclose(losses["synergy"], torch.tensor(0.0))

    def test_synergy_key_present(self, plate_config: PlateConfig):
        """Output dict contains 'synergy' key."""
        cfg = plate_config
        loss_fn = CombinedLoss(cfg, synergy_weight=0.1)
        B, T = 2, 48
        out = _make_model_output(cfg, B, T)
        gt_format = torch.tensor([0, 1])
        gt_country = torch.tensor([0, 1])
        gt_texts = ["A123AA12", "K567BB34"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        losses = loss_fn(
            out,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )

        assert "synergy" in losses

    def test_synergy_bonus_positive(self, plate_config: PlateConfig):
        """Synergy bonus is positive when weight > 0."""
        cfg = plate_config
        loss_fn = CombinedLoss(cfg, synergy_weight=0.5)
        B, T = 2, 48
        out = _make_model_output(cfg, B, T)
        gt_format = torch.tensor([0, 1])
        gt_country = torch.tensor([0, 1])
        gt_texts = ["A123AA12", "K567BB34"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        losses = loss_fn(
            out,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )

        assert losses["synergy"] > 0

    def test_total_equals_weighted_sum_minus_synergy(
        self, plate_config: PlateConfig
    ):
        """Total equals weighted_sum minus synergy bonus."""
        cfg = plate_config
        sw = 0.5
        loss_fn = CombinedLoss(cfg, synergy_weight=sw)
        B, T = 2, 48
        out = _make_model_output(cfg, B, T)
        gt_format = torch.tensor([0, 1])
        gt_country = torch.tensor([0, 1])
        gt_texts = ["A123AA12", "K567BB34"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        losses = loss_fn(
            out,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )

        weighted_sum = (
            loss_fn.format_weight * losses["format"]
            + loss_fn.country_weight * losses["country"]
            + loss_fn.ctc_weight * losses["ctc"]
        )
        expected_total = weighted_sum - losses["synergy"]

        assert torch.isclose(losses["total"], expected_total)

    def test_compute_synergy_bonus_formula(self, plate_config: PlateConfig):
        """Verify _compute_synergy_bonus matches expected formula."""
        cfg = plate_config
        loss_fn = CombinedLoss(cfg, synergy_weight=1.0)
        B, T = 2, 48
        out = _make_model_output(cfg, B, T)
        gt_format = torch.tensor([0, 1])
        gt_country = torch.tensor([0, 1])
        gt_texts = ["A123AA12", "K567BB34"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        _, per_sample_ctc = loss_fn._compute_ctc_loss(
            out,
            gt_texts,
            input_lengths,
        )

        bonus = loss_fn._compute_synergy_bonus(
            out,
            gt_format,
            gt_country,
            per_sample_ctc,
        )

        # Manually compute expected
        fmt_probs = F.softmax(out.format_logits, dim=-1)
        p_fmt = fmt_probs.gather(1, gt_format.unsqueeze(1)).squeeze(1)
        ctry_probs = F.softmax(out.country_logits, dim=-1)
        p_ctry = ctry_probs.gather(1, gt_country.unsqueeze(1)).squeeze(1)
        clamped = torch.clamp(per_sample_ctc, max=50.0)
        p_text = torch.exp(-clamped)
        expected = (p_fmt * p_ctry * p_text).mean()

        assert torch.isclose(bonus, expected, atol=1e-5)

    def test_synergy_gradient_flows(self, plate_config: PlateConfig):
        """Gradients flow through synergy bonus."""
        cfg = plate_config
        loss_fn = CombinedLoss(cfg, synergy_weight=0.5)
        B, T = 2, 48
        out = _make_model_output(cfg, B, T)
        gt_format = torch.tensor([0, 1])
        gt_country = torch.tensor([0, 1])
        gt_texts = ["A123AA12", "K567BB34"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        losses = loss_fn(
            out,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )
        out.ctc_output.retain_grad()
        losses["total"].backward()

        assert out.format_logits.grad is not None
        assert out.country_logits.grad is not None
        assert out.ctc_output.grad is not None

    def test_synergy_bonus_all_correct_high_prob(
        self, plate_config: PlateConfig
    ):
        """All p≈1 → bonus ≈ synergy_weight."""
        cfg = plate_config
        sw = 0.5
        loss_fn = CombinedLoss(cfg, synergy_weight=sw)
        B, T = 2, 48
        out = _make_model_output(cfg, B, T)
        gt_format = torch.tensor([0, 1])
        gt_country = torch.tensor([0, 1])
        gt_texts = ["A123AA12", "K567BB34"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        # Make format/country logits very confident on correct class
        out.format_logits.data.zero_()
        out.format_logits.data.scatter_(
            1,
            gt_format.unsqueeze(1),
            10.0,
        )
        out.country_logits.data.zero_()
        out.country_logits.data.scatter_(
            1,
            gt_country.unsqueeze(1),
            10.0,
        )
        losses = loss_fn(
            out,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )

        # With high probs, synergy bonus should be close to sw
        assert losses["synergy"].item() > 0
        # Upper bound: bonus <= sw (since each synergy_i <= 1)
        assert losses["synergy"].item() <= sw + 1e-4

    def test_synergy_bonus_one_wrong_low_prob(self, plate_config: PlateConfig):
        """One p≈0 → bonus ≈ 0."""
        cfg = plate_config
        sw = 0.5
        loss_fn = CombinedLoss(cfg, synergy_weight=sw)
        B, T = 2, 48
        out = _make_model_output(cfg, B, T)
        gt_format = torch.tensor([0, 1])
        gt_country = torch.tensor([0, 1])
        gt_texts = ["A123AA12", "K567BB34"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        # Make format logits very wrong for sample 0
        out.format_logits.data.zero_()
        out.format_logits.data[0, gt_format[0]] = -10.0
        out.format_logits.data[0, 1 - gt_format[0]] = 10.0
        out.format_logits.data[1, gt_format[1]] = 10.0

        losses = loss_fn(
            out,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )

        # With one very wrong sample, bonus should be small
        assert losses["synergy"].item() < sw * 0.5

    def test_synergy_bonus_range(self, plate_config: PlateConfig):
        """Each synergy_i ∈ [0, 1], so bonus ∈ [0, sw]."""
        cfg = plate_config
        sw = 0.3
        loss_fn = CombinedLoss(cfg, synergy_weight=sw)
        B, T = 2, 48
        out = _make_model_output(cfg, B, T)
        gt_format = torch.tensor([0, 1])
        gt_country = torch.tensor([0, 1])
        gt_texts = ["A123AA12", "K567BB34"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        losses = loss_fn(
            out,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )

        assert losses["synergy"].item() >= 0
        assert losses["synergy"].item() <= sw + 1e-6

    def test_synergy_bonus_increases_with_agreement(
        self, plate_config: PlateConfig
    ):
        """Higher p → higher bonus."""
        cfg = plate_config
        sw = 0.5
        loss_fn = CombinedLoss(cfg, synergy_weight=sw)
        B, T = 2, 48
        gt_format = torch.tensor([0, 1])
        gt_country = torch.tensor([0, 1])
        gt_texts = ["A123AA12", "K567BB34"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        # Low confidence
        out_low = _make_model_output(cfg, B, T)
        out_low.format_logits.data.zero_()
        out_low.format_logits.data.scatter_(
            1,
            gt_format.unsqueeze(1),
            1.0,
        )
        out_low.country_logits.data.zero_()
        out_low.country_logits.data.scatter_(
            1,
            gt_country.unsqueeze(1),
            1.0,
        )
        losses_low = loss_fn(
            out_low,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )

        # High confidence
        out_high = _make_model_output(cfg, B, T)
        out_high.format_logits.data.zero_()
        out_high.format_logits.data.scatter_(
            1,
            gt_format.unsqueeze(1),
            10.0,
        )
        out_high.country_logits.data.zero_()
        out_high.country_logits.data.scatter_(
            1,
            gt_country.unsqueeze(1),
            10.0,
        )
        losses_high = loss_fn(
            out_high,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )

        assert losses_high["synergy"] > losses_low["synergy"]

    def test_forward_total_non_negative(self, plate_config: PlateConfig):
        """Total ≥ 0 with synergy_weight=0.1 (clamp)."""
        cfg = plate_config
        loss_fn = CombinedLoss(cfg, synergy_weight=0.1)
        B, T = 2, 48
        out = _make_model_output(cfg, B, T)
        gt_format = torch.tensor([0, 1])
        gt_country = torch.tensor([0, 1])
        gt_texts = ["A123AA12", "K567BB34"]
        input_lengths = torch.full((B,), T, dtype=torch.long)

        losses = loss_fn(
            out,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )

        assert losses["total"].item() >= 0
