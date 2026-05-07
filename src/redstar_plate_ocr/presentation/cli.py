"""CLI for RedStarPlateOCR — full Typer app."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import typer
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from redstar_plate_ocr.pipeline.dataset_validator import validate_dataset
from redstar_plate_ocr.pipeline.utils import find_csv
from redstar_plate_ocr.presentation.config import (
    _load_plate_config,
    _model_kwargs_from_cfg,
)
from redstar_plate_ocr.presentation.debug_samples import (
    debug_samples as _debug_samples,
)
from redstar_plate_ocr.presentation.display import (
    _METRIC_LABELS,
    _print_dataset_summary,
    _print_startup_panel,
    _print_training_results,
)
from redstar_plate_ocr.presentation.logging import setup_logging

if TYPE_CHECKING:
    from redstar_plate_ocr.nn.model import PlateOCRModel
    from redstar_plate_ocr.pipeline.trainer import Trainer
    from redstar_plate_ocr.plate.config import PlateConfig

app = typer.Typer(
    name="redstar-plate-ocr",
    help="RedStarPlateOCR — License plate OCR",
    no_args_is_help=True,
)
console = Console()

_verbose_count: int = 0


@app.callback()
def main(
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Verbosity: -v=INFO, -vv=DEBUG",
    ),
) -> None:
    """RedStarPlateOCR — License plate OCR."""
    global _verbose_count
    _verbose_count = verbose


@app.command()
def train(
    config: str = typer.Option(..., help="Model config YAML"),
    plate_config: str = typer.Option(..., help="Plate config YAML"),
    data_dir: str = typer.Option(..., help="Dataset directory"),
    output_dir: str = typer.Option(
        "output/",
        help="Output directory",
    ),
    checkpoint: str | None = typer.Option(
        None,
        help="Checkpoint to resume from",
    ),
    augmentation: str | None = typer.Option(
        None,
        help="Augmentation config YAML",
    ),
    original_prob: float = typer.Option(
        1.0,
        "--original-prob",
        help="Probability of including original (non-augmented) "
        "images in training batches (0.0-1.0)",
    ),
) -> None:
    """Start training."""

    from redstar_plate_ocr.nn.model import PlateOCRModel
    from redstar_plate_ocr.pipeline.trainer import Trainer

    pc = _load_plate_config(plate_config)
    cfg = _load_model_config(config, augmentation)

    # CLI override for original_prob
    if "training" not in cfg:
        cfg["training"] = {}
    cfg["training"]["original_prob"] = original_prob

    out = Path(output_dir)
    setup_logging(verbose=_verbose_count)

    train_counts, val_counts = _validate_train_val(
        plate_config, data_dir
    )

    model = PlateOCRModel(
        plate_config=pc, **_model_kwargs_from_cfg(cfg)
    )
    ckpt_state = _load_checkpoint(model, pc, checkpoint)

    if ckpt_state is not None:
        _validate_checkpoint_compat(ckpt_state, pc)

    train_ds, val_ds = _build_datasets(data_dir)

    trainer = Trainer(
        model=model,
        plate_config=pc,
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg=cfg,
        output_dir=out,
    )

    _print_startup_panel(
        trainer,
        cfg,
        checkpoint,
        train_counts,
        val_counts,
    )

    if ckpt_state is not None:
        _apply_resume_state(trainer, ckpt_state)

    result = trainer.train()

    _save_training_artifacts(
        trainer,
        config,
        plate_config,
        augmentation,
        checkpoint,
    )

    _print_training_results(result)


@app.command()
def evaluate(
    checkpoint: str = typer.Option(..., help="Model checkpoint"),
    config: str = typer.Option(..., help="Model config YAML"),
    plate_config: str = typer.Option(..., help="Plate config YAML"),
    data_dir: str = typer.Option(..., help="Dataset directory"),
    split: str = typer.Option("val", help="Data split"),
    e2e: bool = typer.Option(
        False,
        "--e2e",
        help="E2E mode (no teacher forcing)",
    ),
) -> None:
    """Evaluate model on a dataset."""

    from redstar_plate_ocr.data.dataset import PlateDataset
    from redstar_plate_ocr.pipeline.preprocess import PreprocessPipeline
    from redstar_plate_ocr.pipeline.evaluator import Evaluator

    setup_logging(verbose=_verbose_count)

    model, pc = _load_model_for_inference(config, plate_config, checkpoint)

    preproc_params = _preprocess_params_from_config(config)
    csv_path = find_csv(data_dir, split)
    preproc = PreprocessPipeline(**preproc_params)
    ds = PlateDataset(
        csv_path=csv_path,
        dataset_root=data_dir,
        transform=preproc,
    )

    from redstar_plate_ocr.data.dataloader import build_dataloader

    loader = build_dataloader(ds, batch_size=32, is_train=False, num_workers=0)
    from redstar_plate_ocr.pipeline.trainer import get_device_and_amp

    device, _ = get_device_and_amp(False)
    model = model.to(device)
    evaluator = Evaluator(pc, device)
    metrics = evaluator.evaluate(model, loader, e2e=e2e)

    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for k, v in metrics.items():
        label = _METRIC_LABELS.get(k, k)
        table.add_row(label, f"{v:.4f}")

    console.print(table)


@app.command()
def validate(
    plate_config: str = typer.Option(..., help="Plate config YAML"),
    data_dir: str = typer.Option(..., help="Dataset directory"),
    split: str = typer.Option("train", help="Data split"),
) -> None:
    """Validate dataset integrity."""
    setup_logging(verbose=_verbose_count)
    errors, _ = _validate_dataset(plate_config, data_dir, split)
    if not errors:
        console.print(
            "[bold green]Validation passed![/bold green]",
        )
    else:
        console.print(
            f"[bold red]{len(errors)} errors found[/bold red]",
        )
        for e in errors[:20]:
            console.print(f"  [red]•[/red] {e}")
        if len(errors) > 20:
            console.print(
                f"  ... and {len(errors) - 20} more",
            )
        raise typer.Exit(code=1)


@app.command()
def predict(
    checkpoint: str = typer.Option(
        ...,
        help="Model checkpoint (.pt) or ONNX model (.onnx)",
    ),
    config: str | None = typer.Option(
        None,
        help="Model config YAML (not needed for ONNX)",
    ),
    plate_config: str | None = typer.Option(
        None,
        help="Plate config YAML (optional for ONNX with embedded config)",
    ),
    image: str = typer.Option(..., help="Image path"),
) -> None:
    """Recognize a single plate image."""
    import cv2

    setup_logging(verbose=_verbose_count)

    is_onnx = checkpoint.lower().endswith(".onnx")

    if is_onnx:
        from redstar_plate_ocr.pipeline.recognizer import ONNXRecognizer

        pc: PlateConfig | None = None
        if plate_config:
            pc = _load_plate_config(plate_config)
        recognizer = ONNXRecognizer(
            model_path=checkpoint,
            plate_config=pc,
        )
    else:
        from redstar_plate_ocr.pipeline.recognizer import PyTorchRecognizer

        if not config or not plate_config:
            console.print(
                "[red]--config and --plate-config are required "
                "for PyTorch checkpoints[/red]"
            )
            raise typer.Exit(code=1)
        model, pc = _load_model_for_inference(config, plate_config, checkpoint)
        preproc_params = _preprocess_params_from_config(config)
        device = next(model.parameters()).device
        recognizer = PyTorchRecognizer(
            model=model,
            plate_config=pc,
            preprocess_params=preproc_params,
            device=device,
        )

    img = cv2.imread(image)
    if img is None:
        console.print(
            f"[red]Cannot read image: {image}[/red]",
        )
        raise typer.Exit(code=1)

    # cv2.imread returns BGR, model expects RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = recognizer.recognize(img)

    table = Table(title="Recognition Result")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Plate", result.text)
    table.add_row(
        "Text Confidence",
        f"{result.text_confidence:.4f}",
    )
    table.add_row("Country", result.country)
    table.add_row(
        "Country Confidence",
        f"{result.country_confidence:.4f}",
    )
    table.add_row("Format", result.plate_type)
    table.add_row(
        "Needs Review",
        "Yes" if result.needs_review else "No",
    )
    console.print(table)


@app.command()
def export(
    checkpoint: str = typer.Option(..., help="Model checkpoint"),
    config: str = typer.Option(..., help="Model config YAML"),
    plate_config: str = typer.Option(..., help="Plate config YAML"),
    output: str = typer.Option(
        "model.onnx",
        help="Output ONNX path",
    ),
) -> None:
    """Export model to ONNX format."""

    from redstar_plate_ocr.pipeline.exporter import Exporter

    setup_logging(verbose=_verbose_count)

    model, _pc = _load_model_for_inference(config, plate_config, checkpoint)
    preproc_params = _preprocess_params_from_config(config)
    # Build raw preprocessing config for embedding into ONNX metadata
    with open(config) as f:
        raw_cfg = yaml.safe_load(f)
    preprocessing = raw_cfg.get("preprocessing")
    exporter = Exporter()
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Exporting to ONNX...",
            total=None,
        )
        exporter.export_onnx(
            model, output, plate_config=_pc, preprocessing=preprocessing,
        )
        progress.update(task, completed=True)
    console.print(f"[green]✓ Exported to {output}[/green]")


@app.command()
def info(
    plate_config: str = typer.Option(..., help="Plate config YAML"),
    config: str | None = typer.Option(
        None,
        help="Model config YAML (optional)",
    ),
) -> None:
    """Show model configuration info."""

    setup_logging(verbose=_verbose_count)

    pc = _load_plate_config(plate_config)
    cfg = _load_model_config(config) if config else None

    from redstar_plate_ocr.nn.model import PlateOCRModel

    model = PlateOCRModel(plate_config=pc, **_model_kwargs_from_cfg(cfg))

    table = _build_info_table(pc, model)
    console.print(table)


@app.command()
def debug_samples(
    plate_config: str = typer.Option(..., help="Plate config YAML"),
    config: str = typer.Option(..., help="Model config YAML"),
    augmentation: str | None = typer.Option(
        None, help="Augmentation config YAML"
    ),
    data_dir: str = typer.Option(..., help="Dataset directory"),
    output: str = typer.Option("debug/samples", help="Output directory"),
    num_per_group: int = typer.Option(10, help="Samples per country/format"),
    split: str = typer.Option("train", help="Data split"),
) -> None:
    """Save preprocessing stage images for debugging."""
    _debug_samples(
        plate_config_path=plate_config,
        config_path=config,
        augmentation_path=augmentation,
        data_dir=data_dir,
        output_dir=output,
        num_per_group=num_per_group,
        split=split,
    )


def _load_model_for_inference(
    config: str,
    plate_config: str,
    checkpoint: str,
    device: torch.device | None = None,
) -> tuple[PlateOCRModel, PlateConfig]:
    """Load plate config, model config, create model, load checkpoint.

    When *device* is ``None`` the function auto-detects the best
    available device (CUDA → MPS → CPU), matching the logic used by
    :class:`Trainer` and :class:`PyTorchRecognizer`.
    """
    from redstar_plate_ocr.nn.model import PlateOCRModel

    if device is None:
        from redstar_plate_ocr.pipeline.trainer import get_device_and_amp

        device, _ = get_device_and_amp(False)

    pc = _load_plate_config(plate_config)
    cfg = _load_model_config(config)
    model = PlateOCRModel(plate_config=pc, **_model_kwargs_from_cfg(cfg))
    state = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, pc


def _preprocess_params_from_config(config: str) -> dict:
    """Extract PreprocessPipeline kwargs from model config YAML.

    Reads preprocessing section (canvas_height, canvas_width, pad_color,
    normalization.mean, normalization.std) and returns a dict suitable
    for ``PreprocessPipeline(**params)``.
    """
    from redstar_plate_ocr.pipeline.exporter import _preprocess_raw_to_pipeline_params

    with open(config) as f:
        cfg = yaml.safe_load(f)
    raw = cfg.get("preprocessing", {})
    return _preprocess_raw_to_pipeline_params(raw)


def _validate_dataset(
    plate_config_path: str,
    data_dir: str,
    split: str,
    *,
    quiet: bool = False,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Validate dataset, return (errors, counts).

    If quiet=True, skip printing the dataset summary.
    """
    errors, counts = validate_dataset(
        plate_config_path,
        data_dir,
        split,
    )
    if not quiet:
        _print_dataset_summary(counts, split)
    return errors, counts


def _load_checkpoint(
    model: PlateOCRModel,
    plate_config: PlateConfig,
    checkpoint: str | None,
) -> dict | None:
    """Load checkpoint into model, return state or None."""
    if not checkpoint:
        return None

    ckpt_state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if ckpt_state is None:
        raise RuntimeError("Checkpoint state is None")

    _migrate_model_state(ckpt_state, model, plate_config)
    _maybe_migrate_optimizer_state(ckpt_state, model, plate_config)

    console.print(f"[green]Resumed from {checkpoint}[/green]")
    return ckpt_state


def _apply_resume_state(
    trainer: Trainer,
    ckpt_state: dict,
) -> None:
    """Apply checkpoint state to trainer for resuming."""
    trainer.start_epoch = ckpt_state["epoch"] + 1
    if trainer.start_epoch >= trainer.epochs:
        _raise_resume_error(trainer.start_epoch, trainer.epochs)

    optional_loaders = {
        "optimizer_state_dict": (
            trainer.optimizer,
            "optimizer_state_dict missing, skipping optimizer restore",
        ),
        "country_optimizer_state_dict": (
            trainer.country_optimizer,
            "country_optimizer_state_dict missing, skipping restore",
        ),
        # Legacy key from before naming fix
        "country_optimizer_state": (
            trainer.country_optimizer,
            "country_optimizer_state (legacy) missing, skipping restore",
        ),
        "scheduler_state_dict": (
            trainer.scheduler,
            "scheduler_state_dict missing, skipping scheduler restore",
        ),
    }

    loaded_country_opt = False
    for key, (target, warn_msg) in optional_loaders.items():
        state = ckpt_state.get(key)
        if state is not None:
            # Skip legacy key if new key already loaded
            if key == "country_optimizer_state" and loaded_country_opt:
                continue
            target.load_state_dict(state)
            if key.startswith("country_optimizer"):
                loaded_country_opt = True
        else:
            console.print(f"[dim]Info: {warn_msg}[/dim]")

    scaler_state = ckpt_state.get("scaler_state_dict")
    if scaler_state is not None:
        trainer.scaler.load_state_dict(scaler_state)


def _load_model_config(
    config: str,
    augmentation: str | None = None,
) -> dict:
    """Load model config from YAML, optionally merging augmentation."""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    if augmentation:
        with open(augmentation) as af:
            cfg["augmentation"] = yaml.safe_load(af)

    return cfg


def _validate_train_val(
    plate_config: str,
    data_dir: str,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Validate train and val datasets quietly, return counts."""
    _, train_counts = _validate_dataset(
        plate_config, data_dir, "train", quiet=True
    )
    _, val_counts = _validate_dataset(
        plate_config, data_dir, "val", quiet=True
    )
    return train_counts, val_counts


def _validate_checkpoint_compat(
    ckpt_state: dict,
    pc: PlateConfig,
) -> None:
    """Validate checkpoint compatibility, exit on mismatch."""
    from redstar_plate_ocr.pipeline.checkpoint import (
        validate_checkpoint_compat,
    )

    try:
        warns = validate_checkpoint_compat(ckpt_state, pc)
        for w in warns:
            console.print(f"[yellow]⚠ {w}[/yellow]")
    except ValueError as e:
        console.print(
            f"[bold red]Checkpoint incompatible:[/bold red] {e}"
        )
        raise typer.Exit(code=1)


def _build_datasets(
    data_dir: str,
) -> tuple:
    """Build train and validation PlateDatasets."""
    from redstar_plate_ocr.data.dataset import PlateDataset

    train_csv = find_csv(data_dir, "train")
    val_csv = find_csv(data_dir, "val")
    train_ds = PlateDataset(
        csv_path=train_csv,
        dataset_root=data_dir,
    )
    val_ds = PlateDataset(
        csv_path=val_csv,
        dataset_root=data_dir,
    )
    return train_ds, val_ds


def _save_training_artifacts(
    trainer: Trainer,
    config: str,
    plate_config: str,
    augmentation: str | None,
    checkpoint: str | None,
) -> None:
    """Save config snapshot and copy source checkpoint if resuming."""
    from redstar_plate_ocr.pipeline.trainer import save_config_snapshot

    if hasattr(trainer, "run_dir"):
        save_config_snapshot(
            trainer.run_dir,
            config,
            plate_config,
            augmentation_path=augmentation,
        )

    if checkpoint and hasattr(trainer, "run_dir"):
        src = Path(checkpoint)
        dst = trainer.run_dir / "resume_from.pt"
        shutil.copy2(src, dst)
        console.print(
            f"[dim]Source checkpoint copied to {dst}[/dim]",
        )


def _build_info_table(
    pc: PlateConfig,
    model: PlateOCRModel,
) -> Table:
    """Build Rich table with configuration overview."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    table = Table(title="Plate Configuration")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Countries", str(pc.country_list))
    table.add_row("Num countries", str(pc.num_countries))
    table.add_row(
        "Union alphabet size",
        str(pc.union_alphabet_size),
    )
    table.add_row("Total params", f"{total_params:,}")
    table.add_row("Trainable params", f"{trainable_params:,}")

    for country in pc.country_list:
        alpha = pc.get_alphabet(country)
        table.add_row(
            f"  {country}",
            f"alphabet={len(alpha)} ({alpha})",
        )

    return table


def _migrate_model_state(
    ckpt_state: dict,
    model: PlateOCRModel,
    plate_config: PlateConfig,
) -> None:
    """Migrate and load model state dict from checkpoint."""
    from redstar_plate_ocr.pipeline.checkpoint import migrate_checkpoint

    warnings = migrate_checkpoint(
        ckpt_state["model_state_dict"],
        plate_config,
    )
    for w in warnings:
        console.print(f"[yellow]Migration: {w}[/yellow]")
    model.load_state_dict(ckpt_state["model_state_dict"])


def _maybe_migrate_optimizer_state(
    ckpt_state: dict,
    model: PlateOCRModel,
    plate_config: PlateConfig,
) -> None:
    """Migrate optimizer state if present in checkpoint."""
    from redstar_plate_ocr.pipeline.checkpoint import migrate_optimizer_state

    opt_sd = ckpt_state.get("optimizer_state_dict")
    if opt_sd is None:
        return

    warnings = migrate_optimizer_state(
        opt_sd,
        model,
        plate_config,
    )
    for w in warnings:
        console.print(f"[yellow]Migration: {w}[/yellow]")


def _raise_resume_error(
    start_epoch: int,
    total_epochs: int,
) -> None:
    """Raise typer.Exit when checkpoint epoch exceeds configured epochs."""
    console.print(
        f"[bold red]Cannot resume:[/bold red] checkpoint is at "
        f"epoch {start_epoch}, but training is "
        f"configured for only {total_epochs} epochs. "
        f"Increase --epochs in config to continue."
    )
    raise typer.Exit(code=1)
