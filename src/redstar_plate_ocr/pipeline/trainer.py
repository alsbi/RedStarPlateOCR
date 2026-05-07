"""Trainer: training loop for PlateOCRModel."""

from __future__ import annotations

import logging
import shutil
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import TaskID
from torch.utils.data import DataLoader

from redstar_plate_ocr.data.augmentation import (
    build_multi_augmentation,
    build_single_augmentation,
)
from redstar_plate_ocr.data.dataloader import build_dataloader
from redstar_plate_ocr.data.dataset import PlateDataset
from redstar_plate_ocr.pipeline.preprocess import PreprocessPipeline
from redstar_plate_ocr.nn.losses import CombinedLoss
from redstar_plate_ocr.nn.model import PlateOCRModel
from redstar_plate_ocr.pipeline.checkpoint import (
    build_checkpoint,
    save_checkpoint,
)
from redstar_plate_ocr.pipeline.evaluator import Evaluator
from redstar_plate_ocr.pipeline.process_epoch import (
    _EpochResult,
    process_epoch,
)
from redstar_plate_ocr.pipeline.progress_display import ProgressDisplay
from redstar_plate_ocr.pipeline.train_epoch import run_train_epoch
from redstar_plate_ocr.pipeline.training_config import TrainingConfig
from redstar_plate_ocr.pipeline.utils import format_duration
from redstar_plate_ocr.plate.config import PlateConfig

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

logger = logging.getLogger(__name__)


def get_device_and_amp(use_amp: bool) -> tuple[torch.device, bool]:
    """Auto-detect device and AMP compatibility.

    The ``REDSTAR_DEVICE`` environment variable overrides auto-detection.
    Set it to ``cpu``, ``cuda``, or ``mps`` to force a specific device.
    """
    import os

    env_device = os.environ.get("REDSTAR_DEVICE", "").lower()
    if env_device:
        return torch.device(env_device), use_amp and env_device == "cuda"
    if torch.cuda.is_available():
        return torch.device("cuda"), use_amp
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), False
    return torch.device("cpu"), False


def create_run_dir(output_dir: Path) -> Path:
    """Create timestamped run directory."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_snapshot(
    run_dir: Path,
    config_path: str,
    plate_config_path: str,
    augmentation_path: str | None = None,
) -> None:
    """Copy config files into run directory."""
    snap = run_dir / "config_snapshot"
    snap.mkdir(exist_ok=True)
    shutil.copy2(config_path, snap / "model.yaml")
    shutil.copy2(plate_config_path, snap / "plate.yaml")
    if augmentation_path and Path(augmentation_path).exists():
        shutil.copy2(augmentation_path, snap / "augmentation.yaml")


class Trainer:
    """Training loop for PlateOCRModel."""

    @staticmethod
    def _resolve_output_dir(output_dir: Path | None) -> Path:
        """Return output directory or default."""
        return output_dir if output_dir is not None else Path("output")

    @staticmethod
    def _split_parameters(
        model: PlateOCRModel,
    ) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        """Separate country head params from main model params."""
        country_ids = {id(p) for p in model.country_head.parameters()}
        main_params = [
            p for p in model.parameters() if id(p) not in country_ids
        ]
        return main_params, list(model.country_head.parameters())

    def __init__(
        self,
        model: PlateOCRModel,
        plate_config: PlateConfig,
        train_dataset: PlateDataset,
        val_dataset: PlateDataset,
        cfg: dict,
        output_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.plate_config = plate_config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = self._resolve_output_dir(output_dir)

        self.config = TrainingConfig.from_dict(cfg)
        self.device, self.use_amp = get_device_and_amp(
            self.config.use_amp,
        )
        self.model = self.model.to(self.device)

        main_params, country_params = self._split_parameters(model)

        self.optimizer = torch.optim.AdamW(
            main_params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        country_lr = self.config.lr * self.config.country_grad_scale
        self.country_optimizer = torch.optim.AdamW(
            country_params,
            lr=country_lr,
            weight_decay=self.config.weight_decay,
        )
        min_lr = (
            self.config.lr * self.config.final_lr_factor
            if self.config.final_lr_factor > 0.0
            else 0.0
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=self.config.es_mode,  # type: ignore[arg-type]
            patience=self.config.sched_patience,
            factor=self.config.sched_factor,
            min_lr=min_lr,
        )
        self.combined_loss = CombinedLoss(
            plate_config,
            format_weight=self.config.format_weight,
            country_weight=self.config.country_weight,
            ctc_weight=self.config.ctc_weight,
            synergy_weight=self.config.synergy_weight,
            char_aux_weight=self.config.char_aux_weight,
        )
        self.evaluator = Evaluator(
            plate_config, self.device, beam_width=1,
        )
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.use_amp,
        )

        preproc = cfg.get("preprocessing", {})
        self.canvas_h = preproc.get("canvas_height", 80)
        self.canvas_w = preproc.get("canvas_width", 192)
        self.pad_color = preproc.get("pad_color", 128)
        norm = preproc.get("normalization", {})
        self.mean = norm.get("mean", _IMAGENET_MEAN)
        self.std = norm.get("std", _IMAGENET_STD)
        self.aug_cfg = cfg.get("augmentation", {})

        self._interrupt_requested: bool = False
        self._save_thread: threading.Thread | None = None
        self.start_epoch: int = 0

    @property
    def epochs(self) -> int:
        """Number of training epochs."""
        return self.config.epochs

    @property
    def base_lr(self) -> float:
        """Base learning rate."""
        return self.config.lr

    @property
    def batch_size(self) -> int:
        """Batch size."""
        return self.config.batch_size

    def _build_base_pipeline(self) -> PreprocessPipeline:
        """Create base preprocessing pipeline."""
        return PreprocessPipeline(
            canvas_height=self.canvas_h,
            canvas_width=self.canvas_w,
            pad_color=self.pad_color,
            mean=self.mean,
            std=self.std,
        )

    def _build_single_aug_pipeline(
        self,
    ) -> PreprocessPipeline:
        """Create pipeline with exactly 1 random augmentation."""
        augmentation = build_single_augmentation(
            self.aug_cfg,
            is_train=True,
        )
        return PreprocessPipeline(
            canvas_height=self.canvas_h,
            canvas_width=self.canvas_w,
            pad_color=self.pad_color,
            mean=self.mean,
            std=self.std,
            augmentation=augmentation,
        )

    def _build_multi_aug_pipeline(
        self,
    ) -> PreprocessPipeline:
        """Create pipeline with random 2+ augmentations."""
        augmentation = build_multi_augmentation(
            self.aug_cfg,
            is_train=True,
            min_aug=self.config.multi_aug_min,
        )
        return PreprocessPipeline(
            canvas_height=self.canvas_h,
            canvas_width=self.canvas_w,
            pad_color=self.pad_color,
            mean=self.mean,
            std=self.std,
            augmentation=augmentation,
        )

    def _make_filtered_dataset(
        self,
        transform: PreprocessPipeline,
        allowed_regions: list[str],
        balance: bool = False,
    ) -> PlateDataset:
        """Create PlateDataset with optional oversampling and balancing."""
        from redstar_plate_ocr.data.dataloader import (
            _balance_per_group,
            _oversample_square,
        )
        ds = PlateDataset(
            csv_path=self.train_dataset.csv_path,
            dataset_root=self.train_dataset.dataset_root,
            transform=transform,
            samples=list(self.train_dataset.samples),
            allowed_regions=allowed_regions,
        )
        if balance:
            ds = _balance_per_group(ds)
        if self.config.square_oversample_ratio > 0:
            ds = _oversample_square(
                ds, self.config.square_oversample_ratio,
            )
        return ds

    def _build_train_loader(
        self,
        phase: str = "full",
    ) -> DataLoader:
        """Build train dataloader for the given augmentation phase.

        Phases:
            "none"   → only originals (warmup / final polish)
            "single" → originals + 1 copy with exactly 1 random aug
            "full"   → originals + single + num_multi_aug multi-aug copies
        """
        allowed = self.plate_config.country_list
        base_pipeline = self._build_base_pipeline()

        # Warmup: balance per group for equal exposure
        do_balance = phase == "none"

        if phase == "none":
            train_ds = self._make_filtered_dataset(
                base_pipeline, allowed, balance=do_balance,
            )
            return build_dataloader(
                train_ds,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                is_train=True,
                device=self.device,
            )

        from redstar_plate_ocr.data.dataloader import (
            _ConcatDatasetWithSamples,
        )

        datasets = [self._make_filtered_dataset(base_pipeline, allowed)]

        # 1 copy: exactly 1 random augmentation
        single_pipeline = self._build_single_aug_pipeline()
        if single_pipeline.augmentation is not None:
            logger.info(
                "Augmentations (single): %s",
                single_pipeline.get_aug_description(),
            )
            datasets.append(
                self._make_filtered_dataset(
                    single_pipeline, allowed,
                ),
            )

        # num_multi_aug copies: random 2+ augmentations each
        if phase == "full":
            for _ in range(self.config.num_multi_aug):
                multi_pipeline = self._build_multi_aug_pipeline()
                if multi_pipeline.augmentation is not None:
                    logger.info(
                        "Augmentations (multi): %s",
                        multi_pipeline.get_aug_description(),
                    )
                    datasets.append(
                        self._make_filtered_dataset(
                            multi_pipeline, allowed,
                        ),
                    )

        concat_ds = _ConcatDatasetWithSamples(datasets)
        return build_dataloader(
            concat_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            is_train=True,
            device=self.device,
        )

    def _build_val_loader(self) -> DataLoader:
        """Build validation dataloader without augmentation."""
        transform = PreprocessPipeline(
            canvas_height=self.canvas_h,
            canvas_width=self.canvas_w,
            pad_color=self.pad_color,
            mean=self.mean,
            std=self.std,
            augmentation=None,
        )
        self.val_dataset.transform = transform
        return build_dataloader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            is_train=False,
            device=self.device,
        )

    def _handle_interrupt(self, signum: int, frame: object) -> None:
        """Handle SIGINT: first call sets flag, second raises."""
        if self._interrupt_requested:
            raise KeyboardInterrupt
        self._interrupt_requested = True
        logger.info("Interrupt requested, finishing current batch...")
        try:
            Console().print(
                "[bold yellow]⏹ Interrupt requested — "
                "finishing current batch... "
                "Press Ctrl+C again to force stop.[/bold yellow]"
            )
        except Exception:
            pass  # Don't crash in signal handler

    def _save_interrupted_checkpoint(
        self,
        epoch: int,
        best_metric: float,
    ) -> None:
        """Save checkpoint on interrupt."""
        if not hasattr(self, "run_dir") or self.run_dir is None:
            self.run_dir = Path(self.output_dir) / "interrupted"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        ckpt = build_checkpoint(
            epoch=epoch,
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            scaler_state=self.scaler.state_dict(),
            best_metric=best_metric,
            plate_config_yaml=self.plate_config.to_yaml_string(),
            interrupted=True,
            country_optimizer_state_dict=self.country_optimizer.state_dict(),
        )
        path = self.run_dir / f"interrupted_epoch{epoch + 1}.pt"
        try:
            torch.save(ckpt, path)
            logger.info("Interrupted checkpoint saved: %s", path)
        except Exception as e:
            logger.error("Failed to save interrupted checkpoint: %s", e)

    def _log_training_config(self) -> None:
        """Log training configuration at start."""
        logger.info(
            "Training config: epochs=%d, lr=%.6f, "
            "batch_size=%d, device=%s, amp=%s",
            self.config.epochs,
            self.config.lr,
            self.config.batch_size,
            self.device,
            self.use_amp,
        )
        logger.info(
            "Warmup: %d epochs, SingleAug: %d epochs, "
            "NoAug: %d epochs, num_multi_aug=%d",
            self.config.warmup_epochs,
            self.config.single_aug_epochs,
            self.config.no_aug_epochs,
            self.config.num_multi_aug,
        )
        logger.info(
            "grad_accum=%d, update_every=%d batches",
            self.config.gradient_accumulation_steps,
            self.config.update_every_n_batches,
        )
        logger.info(
            "Early stopping: metric=%s, patience=%d, mode=%s",
            self.config.es_metric,
            self.config.es_patience,
            self.config.es_mode,
        )

    def train(self) -> dict:
        """Run training loop, return best metrics."""
        self._interrupt_requested = False
        run_dir = create_run_dir(self.output_dir)
        self.run_dir = run_dir
        logger.info("Run directory: %s", run_dir)

        file_handler = logging.FileHandler(run_dir / "train.log")
        file_handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        file_handler.setFormatter(fmt)
        logging.getLogger().addHandler(file_handler)
        self._log_training_config()
        signal.signal(signal.SIGINT, self._handle_interrupt)

        try:
            best_metrics, last_metrics = self._run_main_training()
            if best_metrics:
                logger.warning(
                    "Training complete. Best plate=%.4f cer=%.4f",
                    best_metrics.get("val_plate_accuracy", 0.0),
                    best_metrics.get("val_cer", 0.0),
                )
        finally:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
            signal.signal(
                signal.SIGINT,
                signal.default_int_handler,
            )

        if self._save_thread is not None:
            self._save_thread.join()
        return {"best": best_metrics, "last": last_metrics}

    def _compute_aug_phase(self, epoch: int) -> str:
        """Compute augmentation phase for the given epoch.

        Returns one of: "none", "single", "full".
        """
        # Final polish: no augmentation
        if epoch >= self.config.epochs - self.config.no_aug_epochs:
            return "none"
        # Warmup: no augmentation
        if epoch < self.config.warmup_epochs:
            return "none"
        # Transition: originals + single augmentation only
        single_aug_end = (
            self.config.warmup_epochs + self.config.single_aug_epochs
        )
        if epoch < single_aug_end:
            return "single"
        # Full augmentation
        return "full"

    def _adjust_warmup_lr(self, epoch: int) -> None:
        """Adjust learning rate during warmup phase."""
        if epoch < self.config.warmup_epochs:
            lr = (
                self.config.lr
                * (epoch + 1)
                / max(self.config.warmup_epochs, 1)
            )
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def _apply_epoch_result(
        self,
        result: "_EpochResult",
        epoch: int,
        best_metric: float,
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        int,
        bool,
    ]:
        """Update trainer state from epoch result.

        Returns (best_metrics, last_metrics, patience_counter, should_stop).
        """
        self._best_metrics = result.best_metrics
        if result.was_interrupted:
            self._save_interrupted_checkpoint(epoch, best_metric)
            return (
                result.best_metrics,
                result.last_metrics,
                result.patience_counter,
                True,
            )
        return (
            result.best_metrics,
            result.last_metrics,
            result.patience_counter,
            result.should_stop,
        )

    def _run_one_epoch(
        self,
        epoch: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        progress_display: ProgressDisplay,
        best_metric: float,
        best_metrics: dict[str, float],
        patience_counter: int,
        epoch_times: list[float],
    ) -> tuple["_EpochResult", float, list[float]]:
        """Run a single epoch and update tracking state.

        Returns (result, epoch_start_time_for_tracking, updated_epoch_times).
        """
        self._adjust_warmup_lr(epoch)
        epoch_start = time.monotonic()
        result = self._process_epoch(
            epoch,
            train_loader,
            val_loader,
            progress_display,
            best_metric,
            best_metrics,
            patience_counter,
            epoch_start=epoch_start,
        )
        epoch_times.append(time.monotonic() - epoch_start)
        progress_display.update_epoch_summary(result.epoch_stats)
        self._update_eta(
            epoch, epoch_times, result.epoch_stats, progress_display,
        )
        return result, epoch_start, epoch_times

    def _maybe_log_resume(self) -> None:
        if self.start_epoch > 0:
            logger.info(
                "Resuming from epoch %d/%d",
                self.start_epoch,
                self.config.epochs,
            )

    def _resolve_train_loader(
        self,
        epoch: int,
        loader: DataLoader | None,
        current_phase: str | None,
    ) -> tuple[DataLoader, str]:
        new_phase = self._compute_aug_phase(epoch)
        if new_phase == current_phase:
            return loader, current_phase
        loader = self._build_train_loader(phase=new_phase)
        return loader, new_phase

    def _should_stop_epoch(
        self,
        result: "_EpochResult",
        epoch: int,
        best_metric: float,
    ) -> bool:
        """Handle epoch interrupt / early stop. Return True to break."""
        if result.was_interrupted:
            self._save_interrupted_checkpoint(epoch, best_metric)
            return True
        return result.should_stop

    def _run_main_training(
        self,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Run main training loop, return (best_metrics, last_metrics)."""
        self._maybe_log_resume()
        train_loader: DataLoader | None = None
        _loader_phase: str | None = None
        val_loader = self._build_val_loader()
        best_metric: float = (
            -1.0 if self.config.es_mode == "max" else float("inf")
        )
        patience_counter = 0
        best_metrics: dict[str, float] = {}
        last_metrics: dict[str, float] = {}

        progress_display = ProgressDisplay(
            total_epochs=self.config.epochs,
        )

        with progress_display:
            epoch_times: list[float] = []
            for epoch in range(self.start_epoch, self.config.epochs):
                train_loader, _loader_phase = self._resolve_train_loader(
                    epoch, train_loader, _loader_phase,
                )
                assert train_loader is not None
                result, _, epoch_times = self._run_one_epoch(
                    epoch,
                    train_loader,
                    val_loader,
                    progress_display,
                    best_metric,
                    best_metrics,
                    patience_counter,
                    epoch_times,
                )
                best_metric = result.best_metric
                best_metrics = result.best_metrics
                last_metrics = result.last_metrics
                patience_counter = result.patience_counter
                self._best_metrics = best_metrics

                if self._should_stop_epoch(result, epoch, best_metric):
                    break

        return best_metrics, last_metrics

    def _update_eta(
        self,
        epoch: int,
        epoch_times: list[float],
        epoch_stats: str,
        progress_display: ProgressDisplay,
    ) -> None:
        """Compute and display ETA using sliding window."""
        remaining = self.config.epochs - (epoch + 1)
        if remaining <= 0 or not epoch_times:
            return
        window = epoch_times[-3:]
        avg_epoch = sum(window) / len(window)
        eta_str = format_duration(avg_epoch * remaining)
        logger.info(
            "ETA: %s (%d epochs left, avg_epoch=%.1fs)",
            eta_str,
            remaining,
            avg_epoch,
        )
        if epoch_stats:
            progress_display.update_epoch_summary(
                f"{epoch_stats} ETA:{eta_str}",
            )

    def _process_epoch(
        self,
        epoch: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        progress_display: ProgressDisplay,
        best_metric: float,
        best_metrics: dict[str, float],
        patience_counter: int,
        epoch_start: float = 0.0,
    ) -> _EpochResult:
        """Process one training epoch — delegates to process_epoch module."""
        return process_epoch(
            self,
            epoch,
            train_loader,
            val_loader,
            progress_display,
            best_metric,
            best_metrics,
            patience_counter,
            epoch_start=epoch_start,
        )

    def _train_epoch(
        self,
        loader: DataLoader,
        sampling_prob: float,
        progress_display: ProgressDisplay,
        task_id: "TaskID",
        epoch_start: float = 0.0,
        current_epoch: int = 0,
    ) -> dict[str, float]:
        """Run one training epoch — delegates to train_epoch module."""
        return run_train_epoch(
            self,
            loader,
            sampling_prob,
            progress_display,
            task_id,
            epoch_start=epoch_start,
            current_epoch=current_epoch,
        )

    def _save_checkpoint(
        self,
        name: str,
        epoch: int,
        metric: float,
    ) -> None:
        """Save model checkpoint to run directory (async)."""
        ckpt = build_checkpoint(
            epoch=epoch,
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            scaler_state=self.scaler.state_dict(),
            best_metric=metric,
            plate_config_yaml=self.plate_config.to_yaml_string(),
            country_optimizer_state_dict=self.country_optimizer.state_dict(),
        )
        run_dir = getattr(self, "run_dir", self.output_dir)
        path = run_dir / name
        self._save_thread = save_checkpoint(
            ckpt,
            path,
            self._save_thread,
        )
