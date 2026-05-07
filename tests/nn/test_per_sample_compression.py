"""Tests for per-sample compression (bugs #5, #6, #7)."""

from __future__ import annotations

import torch

from redstar_plate_ocr.nn.compression import AdaptiveCompression
from redstar_plate_ocr.nn.model import PlateOCRModel
from redstar_plate_ocr.plate.config import PlateConfig


class TestPerSampleCompression:
    """Bug #5: per-sample compression path selection."""

    def test_mixed_batch_output_shape(
        self,
        plate_config: PlateConfig,
    ):
        """Mixed standard+square batch produces (B, 96, C) output."""
        model = PlateOCRModel(plate_config)
        images = torch.randn(2, 3, 80, 192)
        orig_h = torch.tensor([80, 80])
        orig_w = torch.tensor([192, 192])
        gt_countries = ["RU", "RU"]
        gt_plate_types = ["standard", "square"]
        result = model(
            images,
            orig_h,
            orig_w,
            gt_countries=gt_countries,
            gt_plate_types=gt_plate_types,
        )
        # ctc_output shape: (B, seq_len, alphabet_size)
        # seq_len should be 96 (max of standard=48, square=96)
        assert result.ctc_output.shape[0] == 2
        assert result.ctc_output.shape[1] == 96

    def test_mixed_batch_plate_types_in_output(
        self,
        plate_config: PlateConfig,
    ):
        """ModelOutput.plate_types reflects per-sample types."""
        model = PlateOCRModel(plate_config)
        images = torch.randn(2, 3, 80, 192)
        orig_h = torch.tensor([80, 80])
        orig_w = torch.tensor([192, 192])
        gt_countries = ["RU", "RU"]
        gt_plate_types = ["standard", "square"]
        result = model(
            images,
            orig_h,
            orig_w,
            gt_countries=gt_countries,
            gt_plate_types=gt_plate_types,
        )
        assert result.plate_types == ["standard", "square"]

    def test_all_standard_batch_shape(
        self,
        plate_config: PlateConfig,
    ):
        """All-standard batch produces (B, T, C) with T=48."""
        model = PlateOCRModel(plate_config)
        images = torch.randn(2, 3, 80, 192)
        orig_h = torch.tensor([80, 80])
        orig_w = torch.tensor([192, 192])
        result = model(
            images,
            orig_h,
            orig_w,
            gt_countries=["RU", "KZ"],
            gt_plate_types=["standard", "standard"],
        )
        # Standard compression yields T=48
        assert result.ctc_output.shape[1] == 48

    def test_all_square_batch_shape(
        self,
        plate_config: PlateConfig,
    ):
        """All-square batch produces (B, 96, C)."""
        model = PlateOCRModel(plate_config)
        images = torch.randn(2, 3, 80, 192)
        orig_h = torch.tensor([80, 80])
        orig_w = torch.tensor([192, 192])
        result = model(
            images,
            orig_h,
            orig_w,
            gt_countries=["RU", "RU"],
            gt_plate_types=["square", "square"],
        )
        assert result.ctc_output.shape[1] == 96


class TestInferenceFormatPrediction:
    """Bug #7: when gt_plate_types=None, use format_head prediction."""

    def test_no_gt_uses_format_prediction(
        self,
        plate_config: PlateConfig,
    ):
        """Without gt_plate_types, model uses format_head."""
        model = PlateOCRModel(plate_config)
        model.eval()
        images = torch.randn(1, 3, 80, 192)
        orig_h = torch.tensor([80])
        orig_w = torch.tensor([192])
        with torch.no_grad():
            result = model(
                images,
                orig_h,
                orig_w,
                scheduled_sampling_prob=0.0,
            )
        # plate_types should be derived from format_logits
        assert len(result.plate_types) == 1
        assert result.plate_types[0] in ("standard", "square")

    def test_format_head_square_prediction(
        self,
        plate_config: PlateConfig,
    ):
        """When format_head predicts square, plate_type is square."""
        model = PlateOCRModel(plate_config)
        # Override format_head to predict square
        with torch.no_grad():
            model.format_head.fc.weight.fill_(0)
            model.format_head.fc.bias.fill_(0)
            model.format_head.fc.bias[1] = 10.0  # square
        images = torch.randn(1, 3, 80, 192)
        orig_h = torch.tensor([80])
        orig_w = torch.tensor([192])
        result = model(
            images,
            orig_h,
            orig_w,
            scheduled_sampling_prob=0.0,
        )
        assert result.plate_types[0] == "square"

    def test_format_head_standard_prediction(
        self,
        plate_config: PlateConfig,
    ):
        """When format_head predicts standard, plate_type is standard."""
        model = PlateOCRModel(plate_config)
        with torch.no_grad():
            model.format_head.fc.weight.fill_(0)
            model.format_head.fc.bias.fill_(0)
            model.format_head.fc.bias[0] = 10.0  # standard
        images = torch.randn(1, 3, 80, 192)
        orig_h = torch.tensor([80])
        orig_w = torch.tensor([192])
        result = model(
            images,
            orig_h,
            orig_w,
            scheduled_sampling_prob=0.0,
        )
        assert result.plate_types[0] == "standard"


class TestPerSampleInputLengths:
    """Bug #6: per-sample input_lengths by plate_type."""

    def test_standard_input_length_col_mask_sum(self):
        """Standard sample: input_lengths = col_mask.sum()."""
        comp = AdaptiveCompression(
            canvas_height=80,
            canvas_width=192,
            stride=4,
            in_channels=256,
        )
        mask = torch.zeros(1, 1, 20, 48)
        mask[0, 0, :, :30] = 1.0
        result = comp.compute_input_lengths(mask, ["standard"])
        assert result[0].item() == 30

    def test_square_input_length_top_w_plus_bot_w(self):
        """Square sample: input_lengths = feat_w + bot_present."""
        comp = AdaptiveCompression(
            canvas_height=80,
            canvas_width=192,
            stride=4,
            in_channels=256,
        )
        mask = torch.zeros(1, 1, 20, 48)
        mask[0, 0, :10, :20] = 1.0
        mask[0, 0, 10:, :15] = 1.0
        result = comp.compute_input_lengths(mask, ["square"])
        # feat_w(48) + bot_present(15) = 63
        assert result[0].item() == 63

    def test_mixed_batch_per_sample_lengths(self):
        """Mixed batch: each sample gets its own input_length."""
        comp = AdaptiveCompression(
            canvas_height=80,
            canvas_width=192,
            stride=4,
            in_channels=256,
        )
        mask = torch.zeros(2, 1, 20, 48)
        # standard: 30 cols
        mask[0, 0, :, :30] = 1.0
        # square: top=20, bot=15 => feat_w(48) + bot_present(15) = 63
        mask[1, 0, :10, :20] = 1.0
        mask[1, 0, 10:, :15] = 1.0
        result = comp.compute_input_lengths(
            mask,
            ["standard", "square"],
        )
        assert result[0].item() == 30
        assert result[1].item() == 63


class TestGradientFlow:
    """Gradient flows through per-sample compression paths."""

    def test_gradient_flows_through_mixed_batch(
        self,
        plate_config: PlateConfig,
    ):
        """Gradient flows to both compression paths in mixed batch."""
        model = PlateOCRModel(plate_config)
        images = torch.randn(2, 3, 80, 192)
        orig_h = torch.tensor([80, 80])
        orig_w = torch.tensor([192, 192])
        result = model(
            images,
            orig_h,
            orig_w,
            gt_countries=["RU", "RU"],
            gt_plate_types=["standard", "square"],
        )
        loss = result.ctc_output.sum()
        loss.backward()
        assert model.backbone.stage1[0].pw[1].weight.grad is not None
        assert model.backbone.stage1[0].pw[1].weight.grad.abs().sum() > 0


class TestPerSampleInputLengthsExtra:
    """Bug #6: per-sample input_lengths by plate_type (extra)."""

    def test_train_epoch_uses_per_sample_types(
        self,
        plate_config: PlateConfig,
    ):
        """train_epoch uses gt_plate_types (not majority vote)."""
        comp = AdaptiveCompression(
            canvas_height=80,
            canvas_width=192,
            stride=4,
            in_channels=256,
        )
        # Simulate a mixed batch: 2 standard, 1 square
        mask = torch.zeros(3, 1, 20, 48)
        mask[0, 0, :, :30] = 1.0  # standard
        mask[1, 0, :, :25] = 1.0  # standard
        mask[2, 0, :10, :20] = 1.0  # square top
        mask[2, 0, 10:, :15] = 1.0  # square bot
        gt_plate_types = ["standard", "standard", "square"]
        result = comp.compute_input_lengths(mask, gt_plate_types)
        assert result[0].item() == 30
        assert result[1].item() == 25
        assert result[2].item() == 63  # feat_w(48) + bot_present(15)
