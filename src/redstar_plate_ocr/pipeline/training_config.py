"""Training configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings.

    Augmentation phases (by epoch):
        0 .. warmup_epochs-1              → no augmentation
        warmup_epochs .. +single_aug_ep-1 → single random augmentation
        single_aug_end .. -no_aug_epochs  → full (single + multi)
        last no_aug_epochs                → no augmentation (final polish)
    """

    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    single_aug_epochs: int = 3
    no_aug_epochs: int = 5
    gradient_accumulation_steps: int = 2
    scheduled_sampling_max_prob: float = 0.3
    scheduled_sampling_ramp_epochs: int = 5
    batch_size: int = 32
    gradient_clip: float = 1.0
    update_every_n_batches: int = 3
    num_multi_aug: int = 1
    multi_aug_min: int = 2
    square_oversample_ratio: float = 0.0
    original_prob: float = 1.0
    num_workers: int = 4

    # Validation frequency
    val_every_n_epochs: int = 1

    # Early stopping
    es_patience: int = 15
    es_metric: str = "val_plate_accuracy"
    es_mode: str = "max"

    # Scheduler
    sched_patience: int = 5
    sched_factor: float = 0.5
    final_lr_factor: float = 0.0

    # AMP
    use_amp: bool = True

    # Synergy loss
    synergy_weight: float = 0.0

    # Character-level auxiliary loss
    char_aux_weight: float = 0.3

    # Loss weights
    format_weight: float = 1.0
    country_weight: float = 1.5
    ctc_weight: float = 1.2

    # Country branch gradient scaling
    country_grad_scale: float = 5.0

    # E2E evaluation (no teacher forcing)
    e2e_eval: bool = False

    @classmethod
    def from_dict(cls, cfg: dict) -> TrainingConfig:
        """Create TrainingConfig from config dict."""
        training = cfg.get("training", cfg)
        es = training.get("early_stopping", {})
        sched = training.get("scheduler", {})

        return cls(
            epochs=training.get("epochs", 10),
            lr=training.get("lr", 1e-3),
            weight_decay=training.get("weight_decay", 0.01),
            warmup_epochs=training.get("warmup_epochs", 2),
            single_aug_epochs=training.get("single_aug_epochs", 3),
            no_aug_epochs=training.get("no_aug_epochs", 5),
            gradient_accumulation_steps=training.get(
                "gradient_accumulation_steps", 2
            ),
            scheduled_sampling_max_prob=training.get(
                "scheduled_sampling_max_prob", 0.3
            ),
            scheduled_sampling_ramp_epochs=training.get(
                "scheduled_sampling_ramp_epochs", 5
            ),
            batch_size=training.get("batch_size", 32),
            gradient_clip=training.get("gradient_clip", 1.0),
            update_every_n_batches=training.get("update_every_n_batches", 3),
            val_every_n_epochs=training.get("val_every_n_epochs", 1),
            es_patience=es.get("patience", 15),
            es_metric=es.get("metric", "val_plate_accuracy"),
            es_mode=es.get("mode", "max"),
            sched_patience=sched.get("patience", 5),
            sched_factor=sched.get("factor", 0.5),
            final_lr_factor=sched.get("final_lr_factor", 0.0),
            use_amp=training.get("use_amp", True),
            synergy_weight=training.get("synergy_weight", 0.0),
            char_aux_weight=training.get("char_aux_weight", 0.3),
            format_weight=training.get("format_weight", 1.0),
            country_weight=training.get("country_weight", 1.5),
            ctc_weight=training.get("ctc_weight", 1.2),
            country_grad_scale=training.get("country_grad_scale", 5.0),
            e2e_eval=training.get("e2e_eval", False),
            num_multi_aug=training.get("num_multi_aug", 1),
            multi_aug_min=training.get("multi_aug_min", 2),
            square_oversample_ratio=training.get(
                "square_oversample_ratio", 0.0
            ),
            original_prob=training.get("original_prob", 1.0),
            num_workers=training.get("num_workers", 4),
        )
