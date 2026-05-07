"""T0.3: Characterization test — full training cycle (1 epoch)."""

from pathlib import Path

import pytest


@pytest.fixture
def train_dataset():
    csv_path = Path("data/train.csv")
    if not csv_path.exists():
        pytest.skip("No training data available")
    from redstar_plate_ocr.data.dataset import PlateDataset

    return PlateDataset(
        csv_path=str(csv_path),
        dataset_root="data",
    )


@pytest.fixture
def val_dataset():
    csv_path = Path("data/val.csv")
    if not csv_path.exists():
        pytest.skip("No validation data available")
    from redstar_plate_ocr.data.dataset import PlateDataset

    return PlateDataset(
        csv_path=str(csv_path),
        dataset_root="data",
    )


def test_training_one_epoch(
    plate_config,
    train_dataset,
    val_dataset,
    tmp_path,
):
    """Full training cycle: 1 epoch must produce valid metrics."""
    from redstar_plate_ocr.nn.model import PlateOCRModel
    from redstar_plate_ocr.pipeline.trainer import Trainer

    model = PlateOCRModel(plate_config=plate_config)
    cfg = {
        "training": {
            "epochs": 1,
            "lr": 0.001,
            "batch_size": 4,
            "warmup_epochs": 0,
            "no_aug_epochs": 0,
            "gradient_accumulation_steps": 1,
            "early_stopping": {
                "patience": 15,
                "metric": "val_cer",
                "mode": "min",
            },
            "scheduler": {"patience": 5, "factor": 0.5},
        },
        "preprocessing": {
            "canvas_height": 80,
            "canvas_width": 192,
            "pad_color": 128,
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
    }
    trainer = Trainer(
        model=model,
        plate_config=plate_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=cfg,
        output_dir=tmp_path,
    )
    result = trainer.train()

    # Must have best and last metrics
    assert "best" in result
    assert "last" in result

    # Best metrics must contain expected keys
    best = result["best"]
    if best:  # May be empty if interrupted
        assert "val_plate_accuracy" in best
        assert "val_cer" in best
        assert "val_country_accuracy" in best
        assert "val_format_accuracy" in best
        # CER must be non-negative
        assert best["val_cer"] >= 0.0
        # Plate accuracy must be between 0 and 1
        assert 0.0 <= best["val_plate_accuracy"] <= 1.0
