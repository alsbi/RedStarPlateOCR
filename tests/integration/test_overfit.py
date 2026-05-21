"""Integration: model overfits a small synthetic batch."""

from __future__ import annotations

import pytest
import torch

from redstar_plate_ocr.nn.losses import CombinedLoss
from redstar_plate_ocr.nn.model import PlateOCRModel


@pytest.mark.timeout(30)
def test_model_overfits_small_batch(plate_config):
    """5 samples, 10 steps -> loss must decrease."""
    plate_config = plate_config
    model = PlateOCRModel(plate_config)
    combined_loss = CombinedLoss(plate_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Synthetic data
    images = torch.randn(5, 3, 80, 256)
    orig_h = torch.tensor([80] * 5)
    orig_w = torch.tensor([256] * 5)
    gt_countries = ["RU"] * 5
    gt_plate_types = ["standard"] * 5
    gt_format = torch.tensor([0] * 5)  # standard
    gt_country = torch.tensor([0] * 5)  # RU
    gt_texts = ["A123BC99"] * 5
    input_lengths = torch.tensor([64] * 5)

    model.train()
    losses: list[float] = []
    for _ in range(10):
        optimizer.zero_grad()
        output = model(
            images,
            orig_h,
            orig_w,
            gt_countries=gt_countries,
            gt_plate_types=gt_plate_types,
        )
        loss_dict = combined_loss(
            output,
            gt_format,
            gt_country,
            gt_texts,
            input_lengths,
        )
        loss = loss_dict["total"]
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
