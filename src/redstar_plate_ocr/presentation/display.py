"""Rich display helpers for CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from redstar_plate_ocr.pipeline.trainer import Trainer

console = Console()

_METRIC_LABELS = {
    "val_plate_accuracy": "Plate (exact match)",
    "val_cer": "CER",
    "val_char_accuracy": "Char Accuracy",
    "val_country_accuracy": "Region",
    "val_format_accuracy": "Format",
    "val_standard_accuracy": "Standard",
    "val_square_accuracy": "Square",
}


def _print_dataset_summary(
    counts: dict[str, dict[str, int]],
    split: str,
) -> None:
    """Print dataset summary table."""
    table = Table(title=f"Dataset: {split}")
    table.add_column("Country", style="cyan")
    table.add_column("Format", style="cyan")
    table.add_column("Count", style="green")
    for region, types in sorted(counts.items()):
        for pt, cnt in sorted(types.items()):
            table.add_row(region, pt, str(cnt))
    console.print(table)


def _all_country_format_keys(
    train_counts: dict[str, dict[str, int]],
    val_counts: dict[str, dict[str, int]],
) -> list[tuple[str, str]]:
    keys = set()
    for counts in (train_counts, val_counts):
        for country, types in counts.items():
            for fmt in types:
                keys.add((country, fmt))
    return sorted(keys)


def _print_startup_panel(
    trainer: Trainer,
    cfg: dict,
    checkpoint: str | None,
    train_counts: dict[str, dict[str, int]],
    val_counts: dict[str, dict[str, int]],
    *,
    console: Console | None = None,
) -> None:
    """Show unified startup panel with config and dataset."""
    con = console if console is not None else Console()

    table = Table(show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Col3", style="green")
    table.add_column("Col4", style="green")

    # Config section (uses first 2 columns)
    table.add_row("Epochs", str(trainer.epochs))
    table.add_row("Learning Rate", f"{trainer.base_lr:.6f}")
    table.add_row("Batch Size", str(trainer.batch_size))
    table.add_row("Device", str(trainer.device))
    table.add_row("AMP", str(trainer.use_amp))
    table.add_row(
        "Grad Accumulation",
        str(trainer.config.gradient_accumulation_steps),
    )
    table.add_row(
        "Early Stopping",
        f"{trainer.config.es_metric} patience={trainer.config.es_patience}",
    )
    if checkpoint:
        table.add_row("Resuming From", checkpoint)
        table.add_row("Start Epoch", str(trainer.start_epoch))

    # Dataset section separator
    table.add_section()
    table.add_row(
        "[bold]Country[/bold]",
        "[bold]Format[/bold]",
        "[bold]Train[/bold]",
        "[bold]Val[/bold]",
    )

    for country, fmt in _all_country_format_keys(train_counts, val_counts):
        tr = train_counts.get(country, {}).get(fmt, 0)
        va = val_counts.get(country, {}).get(fmt, 0)
        table.add_row(country, fmt, str(tr), str(va))

    con.print(
        Panel(
            table,
            title="Training Configuration",
            border_style="blue",
        ),
    )


def _format_metric_value(container: dict, key: str) -> str:
    return f"{container.get(key, 0.0):.4f}" if container else "-"


def _region_metric_keys(best: dict, last: dict) -> list[str]:
    return [
        k for k in sorted(set(best) | set(last)) if k.startswith("val_region_")
    ]


def _print_training_results(result: dict) -> None:
    """Show training results as Rich Table."""
    best = result.get("best", {})
    last = result.get("last", {})

    console.print("[bold green]Training complete![/bold green]")

    results_table = Table(
        title="Training Results",
        show_lines=True,
    )
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Best", style="green")
    results_table.add_column("Last", style="yellow")

    for key, label in _METRIC_LABELS.items():
        results_table.add_row(
            label,
            _format_metric_value(best, key),
            _format_metric_value(last, key),
        )

    for k in _region_metric_keys(best, last):
        region_name = k[len("val_region_") :]
        results_table.add_row(
            f"  {region_name}",
            _format_metric_value(best, k),
            _format_metric_value(last, k),
        )

    console.print(results_table)
