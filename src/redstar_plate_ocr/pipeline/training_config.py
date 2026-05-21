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
    log_interval: int = 20
    log_grad_interval: int = 100
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
    char_aux_peak_weight: float | None = None
    char_aux_ramp_epochs: int = 0

    # Order-penalty loss (adjacent same-type character order)
    order_weight: float = 0.0
    order_margin: float = 1.0

    # Length-consistency loss (penalise under-emission of chars)
    length_weight: float = 0.0

    # Warmup schedule parameters
    enable_warmup: bool = True
    initial_severity: float = 1.0
    threshold_disable_severe: float = 0.15
    severe_step: float = 0.01
    patience_severe: int = 10
    severe_threshold_std_start: float = 0.3
    severe_midpoint: float = 0.15
    early_stop_patience: int = 15

    # Loss weights
    format_weight: float = 1.0
    country_weight: float = 1.5
    ctc_weight: float = 1.2

    # Country branch gradient scaling
    country_grad_scale: float = 5.0

    # E2E evaluation (no teacher forcing)
    e2e_eval: bool = False

    # TrackIO metrics tracking
    trackio_enabled: bool = True
    trackio_project: str = "redstar-plate-ocr"
    trackio_name: str | None = None
    trackio_dashboard: bool = False  # Auto-launch TrackIO dashboard

    @classmethod
    def from_dict(cls, cfg: dict) -> TrainingConfig:
        """Create TrainingConfig from config dict."""
        training = cfg.get("training", cfg)
        es = training.get("early_stopping", {})
        sched = training.get("scheduler", {})
        warmup = cfg.get("augmentation", {}).get("warmup", {})
        tracking = cfg.get("tracking", {})

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
            log_interval=training.get("log_interval", 20),
            log_grad_interval=training.get("log_grad_interval", 100),
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
            char_aux_peak_weight=training.get("char_aux_peak_weight"),
            char_aux_ramp_epochs=training.get("char_aux_ramp_epochs", 0),
            order_weight=training.get("order_weight", 0.0),
            order_margin=training.get("order_margin", 1.0),
            length_weight=training.get("length_weight", 0.0),
            enable_warmup=warmup.get("enable_warmup", False),
            initial_severity=warmup.get("initial_severity", 1.0),
            threshold_disable_severe=warmup.get(
                "threshold_disable_severe", 0.3
            ),
            severe_step=warmup.get("severe_step", 0.01),
            patience_severe=warmup.get("patience_severe", 10),
            severe_threshold_std_start=warmup.get(
                "severe_threshold_std_start", 0.3
            ),
            severe_midpoint=warmup.get("severe_midpoint", 0.15),
            early_stop_patience=warmup.get("early_stop_patience", 15),
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
            trackio_enabled=tracking.get("enabled", True),
            trackio_project=tracking.get("project", "redstar-plate-ocr"),
            trackio_name=tracking.get("name"),
            trackio_dashboard=tracking.get("dashboard", False),
        )
