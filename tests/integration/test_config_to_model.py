"""Integration: PlateConfig -> PlateOCRModel."""

from __future__ import annotations

import torch

from redstar_plate_ocr.nn.model import PlateOCRModel


def test_config_creates_model(plate_config):
    """PlateConfig.from_yaml -> PlateOCRModel creates model."""
    plate_config = plate_config
    model = PlateOCRModel(plate_config)

    # UnifiedCTCHead replaces CTCHeadCollection
    assert hasattr(model, "ctc_head")
    assert model.ctc_head.fc.out_features == (plate_config.union_alphabet_size)

    # Country head output = num_countries (no sentinel)
    # PositionAwareCountryHead uses fc3 as final layer
    country_fc = model.country_head.fc3
    assert country_fc.out_features == plate_config.num_countries

    # Format head output = 2 (standard, square)
    assert model.format_head.fc.out_features == 2


def test_model_forward_with_real_config(plate_config):
    """Full forward pass with real config."""
    plate_config = plate_config
    model = PlateOCRModel(plate_config)
    model.eval()

    images = torch.randn(2, 3, 80, 192)
    orig_h = torch.tensor([80, 60])
    orig_w = torch.tensor([192, 120])
    gt_countries = ["RU", "KZ"]
    gt_plate_types = ["standard", "standard"]

    with torch.no_grad():
        output = model(
            images,
            orig_h,
            orig_w,
            gt_countries=gt_countries,
            gt_plate_types=gt_plate_types,
        )

    assert output.format_logits.shape == (2, 2)
    assert output.country_logits.shape == (
        2,
        plate_config.num_countries,
    )
    assert output.ctc_output.shape[0] == 2
    assert output.ctc_output.dim() == 3
    assert output.ctc_output.shape[2] == plate_config.union_alphabet_size
