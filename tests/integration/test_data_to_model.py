"""Integration: PlateDataset -> PreprocessPipeline -> PlateOCRModel."""

from __future__ import annotations

import os

import pytest
import torch

from redstar_plate_ocr.data.dataset import PlateDataset
from redstar_plate_ocr.data.transforms import PreprocessPipeline
from redstar_plate_ocr.nn.model import PlateOCRModel

CSV_PATH = "data/val/val.csv"
DATASET_ROOT = "data/"


def _val_csv_exists() -> bool:
    return os.path.isfile(CSV_PATH)


@pytest.mark.skipif(
    not _val_csv_exists(),
    reason="val.csv not found",
)
def test_dataset_to_model(plate_config):
    """3 samples from val.csv through model."""
    plate_config = plate_config
    transform = PreprocessPipeline()
    dataset = PlateDataset(
        csv_path=CSV_PATH,
        dataset_root=DATASET_ROOT,
        transform=transform,
    )
    model = PlateOCRModel(plate_config)
    model.eval()

    n_samples = min(3, len(dataset))
    for i in range(n_samples):
        sample = dataset[i]
        image = sample["image"].unsqueeze(0)
        orig_h = torch.tensor([sample["orig_h"]])
        orig_w = torch.tensor([sample["orig_w"]])

        with torch.no_grad():
            output = model(image, orig_h, orig_w)

        assert output.format_logits.shape[0] == 1
        assert output.country_logits.shape[0] == 1
        assert output.ctc_output.shape[0] == 1
