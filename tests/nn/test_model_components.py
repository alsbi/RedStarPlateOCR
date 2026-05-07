"""Tests for nn components (T3.1-T3.6)."""

from __future__ import annotations

import torch

from redstar_plate_ocr.nn.backbone import PlateBackbone
from redstar_plate_ocr.nn.compression import (
    AdaptiveCompression,
    compute_content_mask,
)
from redstar_plate_ocr.nn.heads import (
    CountryHead,
    FormatHead,
    PositionAwareCountryHead,
    UnifiedCTCHead,
)
from redstar_plate_ocr.nn.lstm import PlateBiLSTM
from redstar_plate_ocr.nn.model import ModelOutput, PlateOCRModel
from redstar_plate_ocr.plate.config import PlateConfig

# T3.1: PlateBackbone


def test_backbone_shape():
    """(2,3,80,192) -> BackboneOutput with final (2,256,20,48)."""
    model = PlateBackbone()
    x = torch.randn(2, 3, 80, 192)
    out = model(x)
    assert out.final.shape == (2, 256, 20, 48)


def test_backbone_param_count():
    """~0.4M parameters (±10%)."""
    model = PlateBackbone()
    n = sum(p.numel() for p in model.parameters())
    assert 360_000 <= n <= 440_000, f"Got {n} params"


# T3.2: Classification Heads


def _make_mask(
    orig_h: torch.Tensor, orig_w: torch.Tensor
) -> torch.Tensor:
    """Shortcut to compute content_mask with default feat params."""
    return compute_content_mask(
        orig_h, orig_w, feat_h=20, feat_w=48, stride=4
    )


def test_format_head_shape():
    """FormatHead uses content_mask + plate dims, not visual features."""
    head = FormatHead(in_channels=256)
    x = torch.randn(2, 256, 20, 48)
    orig_h = torch.tensor([80, 60], dtype=torch.int64)
    orig_w = torch.tensor([192, 192], dtype=torch.int64)
    cmask = _make_mask(orig_h, orig_w)
    out = head(
        x, content_mask=cmask, orig_h=orig_h, orig_w=orig_w
    )
    assert out.shape == (2, 2)


def test_format_head_ignores_visual_features():
    """FormatHead output depends only on plate dims/shape, not pixels."""
    head = FormatHead(in_channels=256)
    head.eval()
    orig_h = torch.tensor([80], dtype=torch.int64)
    orig_w = torch.tensor([192], dtype=torch.int64)
    cmask = _make_mask(orig_h, orig_w)

    x1 = torch.randn(1, 256, 20, 48)
    x2 = torch.randn(1, 256, 20, 48)  # different pixels

    with torch.no_grad():
        out1 = head(
            x1, content_mask=cmask, orig_h=orig_h, orig_w=orig_w
        )
        out2 = head(
            x2, content_mask=cmask, orig_h=orig_h, orig_w=orig_w
        )

    # Same plate dims → same logits regardless of features
    assert torch.allclose(out1, out2, atol=1e-6), (
        "FormatHead should depend only on shape/dims"
    )


def test_format_head_different_dims_different_output():
    """Different plate dimensions produce different format predictions."""
    head = FormatHead(in_channels=256)
    head.eval()
    x = torch.randn(1, 256, 20, 48)

    orig_h_std = torch.tensor([80], dtype=torch.int64)
    orig_w_std = torch.tensor([192], dtype=torch.int64)
    orig_h_sq = torch.tensor([80], dtype=torch.int64)
    orig_w_sq = torch.tensor([80], dtype=torch.int64)

    mask_std = _make_mask(orig_h_std, orig_w_std)
    mask_sq = _make_mask(orig_h_sq, orig_w_sq)

    with torch.no_grad():
        out_std = head(
            x, content_mask=mask_std,
            orig_h=orig_h_std, orig_w=orig_w_std,
        )
        out_sq = head(
            x, content_mask=mask_sq,
            orig_h=orig_h_sq, orig_w=orig_w_sq,
        )

    assert not torch.allclose(out_std, out_sq, atol=1e-6), (
        "Different plate dims should produce different format logits"
    )


def test_country_head_ignores_padding():
    """CountryHead with content_mask: padding region does
    not affect logits."""
    head = CountryHead(in_channels=256, num_countries=7)
    head.eval()

    B, C, H, W = 1, 256, 8, 12
    features = torch.randn(B, C, H, W)
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, :, :6] = 1.0

    with torch.no_grad():
        out_masked = head(features, content_mask=mask)

    corrupted = features.clone()
    corrupted[:, :, :, 6:] = 1000.0

    with torch.no_grad():
        out_corrupted = head(corrupted, content_mask=mask)

    assert torch.allclose(out_masked, out_corrupted, atol=1e-5), (
        "Padding values should not affect CountryHead output"
    )


def test_country_head_shape():
    """(2,256,20,48) -> (2,7)."""
    head = CountryHead(in_channels=256, num_countries=7)
    x = torch.randn(2, 256, 20, 48)
    out = head(x)
    assert out.shape == (2, 7)


def test_country_head_pos_aware_ignores_padding():
    """PositionAwareCountryHead with content_mask: padding region does
    not affect logits.  Changing padding values must not change output."""
    head = PositionAwareCountryHead(
        in_channels=64, num_countries=7, conv_channels=32,
        grid_rows=2, grid_cols=3, hidden_size=64,
    )
    head.eval()

    B, C, H, W = 1, 64, 8, 12
    features = torch.randn(B, C, H, W)
    # Content occupies left half (width 0..5), right half is padding
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, :, :6] = 1.0

    with torch.no_grad():
        out_masked = head(features, content_mask=mask)

    # Corrupt padding region with huge values
    corrupted = features.clone()
    corrupted[:, :, :, 6:] = 1000.0

    with torch.no_grad():
        out_corrupted = head(corrupted, content_mask=mask)

    # Both outputs must be identical — the mask zeroes out padding
    assert torch.allclose(out_masked, out_corrupted, atol=1e-5), (
        "Padding values should not affect country head output"
    )


# T3.3: AdaptiveCompression


def test_compression_standard():
    """(2,256,20,48) + orig dims -> (2,48,256)."""
    comp = AdaptiveCompression(in_channels=256)
    features = torch.randn(2, 256, 20, 48)
    orig_h = torch.tensor([80, 80])
    orig_w = torch.tensor([192, 192])
    out = comp.forward_standard(features, orig_h, orig_w)
    assert out.shape == (2, 48, 256)


def test_compression_square():
    """(2,256,20,48) + orig dims -> (2,96,256)."""
    comp = AdaptiveCompression(in_channels=256)
    features = torch.randn(2, 256, 20, 48)
    orig_h = torch.tensor([80, 80])
    orig_w = torch.tensor([192, 192])
    out = comp.forward_square(features, orig_h, orig_w)
    assert out.shape == (2, 96, 256)


def test_content_mask():
    """orig_h=80, orig_w=192 -> mask all 1.0."""
    orig_h = torch.tensor([80, 80])
    orig_w = torch.tensor([192, 192])
    mask = compute_content_mask(
        orig_h,
        orig_w,
        feat_h=20,
        feat_w=48,
        stride=4,
    )
    assert mask.shape == (2, 1, 20, 48)
    assert (mask == 1.0).all()


# T3.4: PlateBiLSTM


def test_bilstm_standard():
    """(2,48,256) -> (2,48,512)."""
    model = PlateBiLSTM(input_size=256, hidden_size=256)
    x = torch.randn(2, 48, 256)
    out = model(x)
    assert out.shape == (2, 48, 512)


def test_bilstm_square():
    """(2,96,256) -> (2,96,512)."""
    model = PlateBiLSTM(input_size=256, hidden_size=256)
    x = torch.randn(2, 96, 256)
    out = model(x)
    assert out.shape == (2, 96, 512)


# T3.5: UnifiedCTCHead


def test_unified_ctc_head_forward(plate_config: PlateConfig):
    """UnifiedCTCHead forward produces correct shape."""
    from redstar_plate_ocr.nn.mask_table import build_mask_table

    mask_table = build_mask_table(plate_config)
    head = UnifiedCTCHead(
        input_size=512,
        union_alphabet_size=plate_config.union_alphabet_size,
    )
    x = torch.randn(2, 48, 512)
    country_idx = torch.tensor([0, 1])
    mask = mask_table[country_idx]
    out = head(x, mask)
    assert out.shape == (2, 48, plate_config.union_alphabet_size)


# T3.6: PlateOCRModel


def test_model_forward(plate_config: PlateConfig):
    """Full forward pass -> ModelOutput with correct shapes."""
    model = PlateOCRModel(plate_config)
    images = torch.randn(2, 3, 80, 192)
    orig_h = torch.tensor([80, 80])
    orig_w = torch.tensor([192, 192])
    gt_countries = ["RU", "KZ"]
    gt_plate_types = ["standard", "standard"]
    result = model(
        images,
        orig_h,
        orig_w,
        gt_countries=gt_countries,
        gt_plate_types=gt_plate_types,
    )
    assert isinstance(result, ModelOutput)
    assert result.format_logits.shape == (2, 2)
    assert result.country_logits.shape == (
        2,
        plate_config.num_countries,
    )
    assert result.ctc_output.dim() == 3
    assert result.ctc_output.shape[0] == 2
    assert result.ctc_output.shape[2] == plate_config.union_alphabet_size
    assert result.content_mask.shape == (2, 1, 20, 48)


def test_model_scheduled_sampling(plate_config: PlateConfig):
    """With prob=0.0 uses GT (no sampling)."""
    model = PlateOCRModel(plate_config)
    images = torch.randn(2, 3, 80, 192)
    orig_h = torch.tensor([80, 80])
    orig_w = torch.tensor([192, 192])
    gt_countries = ["RU", "KZ"]
    gt_plate_types = ["standard", "standard"]
    result = model(
        images,
        orig_h,
        orig_w,
        gt_countries=gt_countries,
        gt_plate_types=gt_plate_types,
        scheduled_sampling_prob=0.0,
    )
    assert result.plate_types == ["standard", "standard"]


def test_model_scheduled_sampling_mixed_types_no_assert(
    plate_config: PlateConfig,
):
    """Mixed plate_types from scheduled sampling must not assert.

    When scheduled_sampling_prob=1.0 and the model is untrained,
    vectorized resolve logic may predict 'standard' for all samples while
    GT has 'square'. Per-sample compression runs both paths and
    selects per-sample result, so mixed types are handled
    correctly.
    """
    model = PlateOCRModel(plate_config)
    images = torch.randn(2, 3, 80, 192)
    orig_h = torch.tensor([80, 80])
    orig_w = torch.tensor([192, 192])
    gt_countries = ["RU", "RU"]
    gt_plate_types = ["square", "square"]
    # prob=1.0 forces prediction, which defaults to 'standard'
    # for untrained model, but compression must use GT
    result = model(
        images,
        orig_h,
        orig_w,
        gt_countries=gt_countries,
        gt_plate_types=gt_plate_types,
        scheduled_sampling_prob=1.0,
    )
    assert isinstance(result, ModelOutput)


# mask_disable_warmup: no mask during warmup epochs


def test_mask_disable_warmup_no_mask_table(plate_config: PlateConfig):
    """Model with mask_disable_warmup has _no_mask_table (all zeros)."""
    cfg = {
        "unified_ctc_head": {
            "mask_disable_warmup": True,
            "mask_ramp_warmup": 3,
            "mask_ramp_epochs": 8,
        },
    }
    model = PlateOCRModel(plate_config, classification_cfg=cfg)
    assert hasattr(model, "_no_mask_table")
    assert model._no_mask_table.shape == model._flat_mask_table.shape
    assert model._no_mask_table.abs().sum().item() == 0


def test_mask_disable_warmup_ramp_values(plate_config: PlateConfig):
    """Ramp returns -1.0 during warmup, 0.0 at ramp start, 1.0 at end."""
    cfg = {
        "unified_ctc_head": {
            "mask_disable_warmup": True,
            "mask_ramp_warmup": 3,
            "mask_ramp_epochs": 8,
        },
    }
    model = PlateOCRModel(plate_config, classification_cfg=cfg)
    model.train()
    assert model._compute_mask_ramp(0) == -1.0
    assert model._compute_mask_ramp(2) == -1.0
    assert model._compute_mask_ramp(3) == 0.0
    assert model._compute_mask_ramp(7) == 0.5
    assert model._compute_mask_ramp(11) == 1.0


def test_mask_disable_warmup_eval_always_full(plate_config: PlateConfig):
    """Eval mode always uses full positional mask regardless of epoch."""
    cfg = {
        "unified_ctc_head": {
            "mask_disable_warmup": True,
            "mask_ramp_warmup": 3,
            "mask_ramp_epochs": 8,
        },
    }
    model = PlateOCRModel(plate_config, classification_cfg=cfg)
    model.eval()
    assert model._compute_mask_ramp(0) == 1.0
    assert model._compute_mask_ramp(2) == 1.0


def test_mask_disable_warmup_false_old_behavior(plate_config: PlateConfig):
    """mask_disable_warmup=False gives old ramp behavior (flat at warmup)."""
    cfg = {
        "unified_ctc_head": {
            "mask_disable_warmup": False,
            "mask_ramp_warmup": 3,
            "mask_ramp_epochs": 8,
        },
    }
    model = PlateOCRModel(plate_config, classification_cfg=cfg)
    model.train()
    assert model._compute_mask_ramp(0) == 0.0
    assert model._compute_mask_ramp(2) == 0.0
    assert model._compute_mask_ramp(3) == 0.0


def test_mask_disable_warmup_forward_no_mask(plate_config: PlateConfig):
    """Forward at epoch 0 with disable_warmup produces unconstrained output."""
    cfg = {
        "unified_ctc_head": {
            "mask_disable_warmup": True,
            "mask_ramp_warmup": 3,
            "mask_ramp_epochs": 8,
            "mask_value": -10.0,
        },
    }
    model = PlateOCRModel(plate_config, classification_cfg=cfg)
    model.train()

    images = torch.randn(1, 3, 80, 192)
    orig_h = torch.tensor([80])
    orig_w = torch.tensor([192])
    gt_countries = ["RU"]
    gt_plate_types = ["standard"]

    out_no_mask = model(
        images, orig_h, orig_w,
        gt_countries=gt_countries, gt_plate_types=gt_plate_types, epoch=0,
    )
    out_flat = model(
        images, orig_h, orig_w,
        gt_countries=gt_countries, gt_plate_types=gt_plate_types, epoch=3,
    )
    # Same shape (same plate type), different values (no-mask vs flat-mask)
    assert out_no_mask.ctc_output.shape == out_flat.ctc_output.shape
    assert not torch.allclose(out_no_mask.ctc_output, out_flat.ctc_output)
