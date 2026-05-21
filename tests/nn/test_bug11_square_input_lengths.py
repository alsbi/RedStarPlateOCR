"""Bug #11: compute_input_lengths for square — bottom half invisible to CTC.

The square layout is [top_feat_w_cols | bot_feat_w_cols], so
input_length must be at least feat_w + bot_present to include
the bottom half's content.
"""

from __future__ import annotations

import torch

from redstar_plate_ocr.nn.compression import AdaptiveCompression


class TestBug11SquareInputLengths:
    """Square input_lengths must cover both top and bottom halves."""

    def test_square_input_lengths_covers_both_halves(
        self,
    ):
        """Square input_length = feat_w + bot_present (both halves visible)."""
        comp = AdaptiveCompression(
            canvas_height=80,
            canvas_width=256,
            stride=4,
            in_channels=256,
        )
        # feat_w = 256 // 4 = 64
        # Square plate 80x80: top half uses 20 cols,
        # bottom half uses 20 cols
        mask = torch.zeros(1, 1, 20, 64)
        mask[0, 0, :10, :20] = 1.0  # top half content
        mask[0, 0, 10:, :20] = 1.0  # bottom half content
        result = comp.compute_input_lengths(mask, ["square"])
        # feat_w(64) + bot_present(20) + safety_margin(4) = 88
        assert result[0].item() == 64 + 20 + 4

    def test_square_input_lengths_specific_value(
        self,
    ):
        """Canvas 80x256, stride=4, square 80x80: input_length=88."""
        comp = AdaptiveCompression(
            canvas_height=80,
            canvas_width=256,
            stride=4,
            in_channels=256,
        )
        # feat_w = 64, feat_h = 20
        # Square plate 80x80: top=10 rows x 20 cols, bot=10 rows x 20 cols
        mask = torch.zeros(1, 1, 20, 64)
        mask[0, 0, :10, :20] = 1.0
        mask[0, 0, 10:, :20] = 1.0
        result = comp.compute_input_lengths(mask, ["square"])
        # feat_w(64) + bot_present(20) + safety_margin(4) = 88
        assert result[0].item() == 88

    def test_standard_input_lengths_unchanged(
        self,
    ):
        """Standard path input_lengths not affected by the fix."""
        comp = AdaptiveCompression(
            canvas_height=80,
            canvas_width=256,
            stride=4,
            in_channels=256,
        )
        mask = torch.zeros(1, 1, 20, 64)
        mask[0, 0, :, :30] = 1.0
        result = comp.compute_input_lengths(mask, ["standard"])
        # col_present(30) + safety_margin(4) = 34
        assert result[0].item() == 34
