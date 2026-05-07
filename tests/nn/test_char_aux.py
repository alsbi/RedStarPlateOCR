"""Tests for P1: character-level auxiliary loss."""

import torch

from redstar_plate_ocr.nn.char_aux import CharAuxHead, build_char_targets
from redstar_plate_ocr.nn.losses import CombinedLoss
from redstar_plate_ocr.nn.model import ModelOutput
from redstar_plate_ocr.plate.config import PlateConfig


class TestBuildCharTargets:
    """Tests for build_char_targets helper."""

    def test_empty_text_all_blank(self) -> None:
        alphabet = "AB12"
        blank_idx = len(alphabet)
        targets = build_char_targets(
            "", alphabet, width=8, blank_idx=blank_idx
        )
        assert targets == [blank_idx] * 8

    def test_single_char_fills_all(self) -> None:
        alphabet = "AB12"
        targets = build_char_targets("A", alphabet, width=4, blank_idx=4)
        assert targets == [0, 0, 0, 0]

    def test_two_chars_half_half(self) -> None:
        alphabet = "AB12"
        targets = build_char_targets("AB", alphabet, width=4, blank_idx=4)
        assert targets == [0, 0, 1, 1]

    def test_six_chars_across_24(self) -> None:
        alphabet = "ABEKMHOPCTYX0123456789"
        # A=0, B=1, E=2, K=3, M=4, H=5, O=6, P=7, C=8
        # 0=12, 1=13, 2=14, 3=15
        targets = build_char_targets(
            "A123BC",
            alphabet,
            width=24,
            blank_idx=24,
        )
        # 6 chars across 24 positions: 4 each
        assert targets[0:4] == [0, 0, 0, 0]  # A
        assert targets[4:8] == [13, 13, 13, 13]  # 1
        assert targets[8:12] == [14, 14, 14, 14]  # 2
        assert targets[12:16] == [15, 15, 15, 15]  # 3
        assert targets[16:20] == [1, 1, 1, 1]  # B
        assert targets[20:24] == [8, 8, 8, 8]  # C

    def test_unknown_chars_skipped(self) -> None:
        alphabet = "AB"
        # Z not in alphabet, skipped
        targets = build_char_targets("AZB", alphabet, width=6, blank_idx=2)
        # Only A and B, 2 chars across 6: 3 each
        assert targets == [0, 0, 0, 1, 1, 1]


class TestCharAuxHead:
    """Tests for CharAuxHead module."""

    def test_output_shape(self) -> None:
        head = CharAuxHead(in_channels=256, max_alphabet_size=30)
        features = torch.randn(2, 256, 5, 24)
        out = head(features)
        assert out.shape == (2, 24, 30)

    def test_gradient_flows(self) -> None:
        head = CharAuxHead(in_channels=256, max_alphabet_size=30)
        features = torch.randn(2, 256, 5, 24, requires_grad=True)
        out = head(features)
        out.sum().backward()
        assert features.grad is not None
        assert features.grad.abs().sum() > 0

    def test_different_width(self) -> None:
        head = CharAuxHead(in_channels=256, max_alphabet_size=30)
        features = torch.randn(1, 256, 5, 12)
        out = head(features)
        assert out.shape == (1, 12, 30)


class TestCharAuxLoss:
    """Tests for char aux loss in CombinedLoss."""

    def _make_output_with_char_aux(
        self,
        cfg: PlateConfig,
        B: int,
        T: int,
        char_aux_logits: torch.Tensor | None = None,
    ) -> ModelOutput:
        union_size = cfg.union_alphabet_size
        fmt_logits = torch.randn(B, 2, requires_grad=True)
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
            char_aux_logits=char_aux_logits,
        )

    def test_char_aux_loss_computed(self, plate_config: PlateConfig) -> None:
        cfg = plate_config
        loss_fn = CombinedLoss(
            cfg,
            char_aux_weight=0.3,
        )
        B, T = 2, 20
        char_aux = torch.randn(
            B,
            24,
            cfg.union_alphabet_size,
            requires_grad=True,
        )
        output = self._make_output_with_char_aux(
            cfg,
            B,
            T,
            char_aux_logits=char_aux,
        )
        gt_format = torch.tensor([0, 0])
        gt_country = torch.tensor([0, 0])
        gt_texts = ["A123BC", "K456MX"]
        input_lengths = torch.tensor([20, 20])
        result = loss_fn(
            output,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )
        assert "char_aux" in result
        assert result["char_aux"].item() > 0

    def test_char_aux_zero_weight_no_loss(
        self, plate_config: PlateConfig
    ) -> None:
        cfg = plate_config
        loss_fn = CombinedLoss(
            cfg,
            char_aux_weight=0.0,
        )
        B, T = 2, 20
        char_aux = torch.randn(
            B,
            24,
            cfg.union_alphabet_size,
            requires_grad=True,
        )
        output = self._make_output_with_char_aux(
            cfg,
            B,
            T,
            char_aux_logits=char_aux,
        )
        gt_format = torch.tensor([0, 0])
        gt_country = torch.tensor([0, 0])
        gt_texts = ["A123BC", "K456MX"]
        input_lengths = torch.tensor([20, 20])
        result = loss_fn(
            output,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )
        assert "char_aux" not in result or result["char_aux"].item() == 0.0

    def test_char_aux_none_no_loss(self, plate_config: PlateConfig) -> None:
        cfg = plate_config
        loss_fn = CombinedLoss(
            cfg,
            char_aux_weight=0.3,
        )
        B, T = 2, 20
        output = self._make_output_with_char_aux(
            cfg,
            B,
            T,
            char_aux_logits=None,
        )
        gt_format = torch.tensor([0, 0])
        gt_country = torch.tensor([0, 0])
        gt_texts = ["A123BC", "K456MX"]
        input_lengths = torch.tensor([20, 20])
        result = loss_fn(
            output,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )
        # No char_aux_logits → no char_aux loss
        assert "char_aux" not in result or result["char_aux"].item() == 0.0
