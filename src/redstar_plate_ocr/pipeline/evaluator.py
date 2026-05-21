"""Evaluator: validation loop for PlateOCRModel."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from redstar_plate_ocr.nn.metrics import (
    Accuracy,
    AdjacentTranspositionRate,
    BigramSwapRate,
    CharacterAccuracy,
    CharacterErrorRate,
    NormalizedEditDistance,
    compute_per_group_metrics,
)
from redstar_plate_ocr.nn.model import PlateOCRModel
from redstar_plate_ocr.pipeline.utils import (
    greedy_decode,
    to_long_tensor,
)
from redstar_plate_ocr.plate.config import PlateConfig
from redstar_plate_ocr.plate.postprocess import BeamSearchDecoder

logger = logging.getLogger(__name__)


class Evaluator:
    """Validation of PlateOCRModel."""

    def __init__(
        self,
        plate_config: PlateConfig,
        device: torch.device,
        beam_width: int = 1,
    ) -> None:
        self.plate_config = plate_config
        self.device = device
        self.beam_width = beam_width
        if beam_width > 1:
            self._beam_decoder = BeamSearchDecoder(
                plate_config.union_alphabet,
                beam_width=beam_width,
            )

    def _batch_predictions(
        self,
        ctc: Tensor,
        input_lengths: Tensor | None = None,
    ) -> list[str]:
        """Decode CTC output to text predictions."""
        bsz = ctc.shape[0]
        union_alphabet = self.plate_config.union_alphabet
        preds: list[str] = []
        for i in range(bsz):
            inp_len = (
                int(input_lengths[i]) if input_lengths is not None else None
            )
            if self.beam_width > 1:
                pred, _ = self._beam_decoder.decode(ctc[i])
            else:
                pred = greedy_decode(
                    ctc[i], union_alphabet, input_length=inp_len
                )
            preds.append(pred)
        return preds

    def _update_country_data(
        self,
        output,
        gt_regions: list[str],
        country_acc: Accuracy,
        pred_countries_total: list[str],
        gt_countries_total: list[str],
        country_conf_total: list[float],
    ) -> None:
        """Decode and accumulate country predictions."""
        pred_countries = self._decode_countries(output.country_logits)
        country_acc.update(pred_countries, gt_regions)
        pred_countries_total.extend(pred_countries)
        gt_countries_total.extend(gt_regions)
        probs = F.softmax(output.country_logits, dim=1)
        max_probs = probs.max(dim=1).values.tolist()
        country_conf_total.extend(max_probs)

    def _update_type_metrics(
        self,
        preds: list[str],
        gt_texts: list[str],
        gt_plate_types: list[str],
        square_acc: Accuracy,
        standard_acc: Accuracy,
    ) -> None:
        """Update per-plate-type accuracy metrics."""
        sq_p, sq_t = self._filter_by_type(
            preds, gt_texts, gt_plate_types, "square"
        )
        if sq_p:
            square_acc.update(sq_p, sq_t)
        std_p, std_t = self._filter_by_type(
            preds, gt_texts, gt_plate_types, "standard"
        )
        if std_p:
            standard_acc.update(std_p, std_t)

    def _update_region_accs(
        self,
        preds: list[str],
        gt_texts: list[str],
        gt_regions: list[str],
        region_accs: dict[str, Accuracy],
    ) -> None:
        """Update per-region accuracy trackers."""
        for pred, tgt, region in zip(preds, gt_texts, gt_regions, strict=True):
            if region not in region_accs:
                region_accs[region] = Accuracy()
            region_accs[region].update([pred], [tgt])

    @staticmethod
    def _forward(
        model: PlateOCRModel,
        images: Tensor,
        orig_h: Tensor,
        orig_w: Tensor,
        gt_regions: list[str],
        gt_plate_types: list[str],
        e2e: bool,
    ):
        """Forward pass with or without teacher forcing."""
        if e2e:
            return model(images, orig_h, orig_w)
        return model(
            images,
            orig_h,
            orig_w,
            gt_countries=gt_regions,
            gt_plate_types=gt_plate_types,
            scheduled_sampling_prob=0.0,
        )

    def evaluate(
        self,
        model: PlateOCRModel,
        dataloader: DataLoader,
        interrupt_check: Callable[[], bool] | None = None,
        e2e: bool = False,
    ) -> dict[str, float]:
        """Run validation, return metrics dict with per-region stats.

        Args:
            model: PlateOCRModel to evaluate.
            dataloader: Validation DataLoader.
            interrupt_check: Optional callable to check interruption.
            e2e: If True, run without teacher forcing (no gt_countries,
                no gt_plate_types). Metrics use same keys.
        """
        cer = CharacterErrorRate()
        char_acc = CharacterAccuracy()
        plate_acc = Accuracy()
        country_acc = Accuracy()
        format_acc = Accuracy()
        square_acc = Accuracy()
        standard_acc = Accuracy()

        # Per-region plate accuracy trackers
        region_accs: dict[str, Accuracy] = {}

        # Country diagnostics accumulators
        pred_countries_total: list[str] = []
        gt_countries_total: list[str] = []
        country_conf_total: list[float] = []

        # Accumulate all predictions/targets for extended metrics
        all_preds: list[str] = []
        all_targets: list[str] = []
        all_regions: list[str] = []
        all_plate_types: list[str] = []

        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                orig_h = to_long_tensor(batch["orig_h"], self.device)
                orig_w = to_long_tensor(batch["orig_w"], self.device)
                gt_regions = batch["region"]
                gt_plate_types = batch["plate_type"]
                gt_texts = batch["plate_text"]

                output = self._forward(
                    model,
                    images,
                    orig_h,
                    orig_w,
                    gt_regions,
                    gt_plate_types,
                    e2e,
                )
                ctc = output.ctc_output

                # Compute input_lengths to clip padded timesteps
                per_sample_types = list(gt_plate_types)
                input_lengths = model.compression.compute_input_lengths(
                    output.content_mask, per_sample_types
                )

                preds = self._batch_predictions(ctc, input_lengths)
                cer.update(preds, gt_texts)
                char_acc.update(preds, gt_texts)
                plate_acc.update(preds, gt_texts)

                # Accumulate for extended metrics
                all_preds.extend(preds)
                all_targets.extend(gt_texts)
                all_regions.extend(gt_regions)
                all_plate_types.extend(gt_plate_types)

                self._update_country_data(
                    output,
                    gt_regions,
                    country_acc,
                    pred_countries_total,
                    gt_countries_total,
                    country_conf_total,
                )

                pred_formats = self._decode_formats(output.format_logits)
                format_acc.update(pred_formats, gt_plate_types)

                self._update_type_metrics(
                    preds,
                    gt_texts,
                    gt_plate_types,
                    square_acc,
                    standard_acc,
                )
                self._update_region_accs(
                    preds,
                    gt_texts,
                    gt_regions,
                    region_accs,
                )

                if interrupt_check is not None and interrupt_check():
                    break

        result: dict[str, float] = {
            "val_plate_accuracy": plate_acc.compute(),
            "val_cer": cer.compute(),
            "val_char_accuracy": char_acc.compute(),
            "val_country_accuracy": country_acc.compute(),
            "val_format_accuracy": format_acc.compute(),
            "val_square_accuracy": square_acc.compute(),
            "val_standard_accuracy": standard_acc.compute(),
        }

        # Add per-region plate accuracy
        for region, acc in sorted(region_accs.items()):
            result[f"val_region_{region}"] = acc.compute()

        # Extended metrics: NED, BSR, ATR
        if all_preds:
            ned_calc = NormalizedEditDistance()
            bsr_calc = BigramSwapRate()
            atr_calc = AdjacentTranspositionRate()

            result["val_ned"] = ned_calc(all_preds, all_targets)
            result["val_bigram_swap_rate"] = bsr_calc(
                all_preds,
                all_targets,
            )
            result["val_adjacent_transposition_rate"] = atr_calc(
                all_preds,
                all_targets,
            )

            # Per-country CER and plate accuracy
            per_country = compute_per_group_metrics(
                all_preds,
                all_targets,
                all_regions,
            )
            for country, metrics in per_country.items():
                result[f"val_cer_{country}"] = metrics["cer"]
                result[f"val_plateacc_{country}"] = metrics["plate_acc"]

            # Per-format CER and plate accuracy
            per_format = compute_per_group_metrics(
                all_preds,
                all_targets,
                all_plate_types,
            )
            for fmt, metrics in per_format.items():
                result[f"val_cer_fmt_{fmt}"] = metrics["cer"]
                result[f"val_plateacc_fmt_{fmt}"] = metrics["plate_acc"]

        self._log_country_diagnostics(
            country_acc,
            pred_countries_total,
            gt_countries_total,
            country_conf_total,
        )

        return result

    def _log_country_diagnostics(
        self,
        country_acc: Accuracy,
        pred_countries: list[str],
        gt_countries: list[str],
        country_conf: list[float],
    ) -> None:
        """Log country prediction diagnostics."""
        total = len(pred_countries)
        if total == 0:
            return
        accuracy = country_acc.compute()
        correct = int(accuracy * total)
        pred_dist = Counter(pred_countries)
        gt_dist = Counter(gt_countries)
        avg_conf = sum(country_conf) / len(country_conf)
        logger.debug(
            "Country eval: %d/%d correct (%.1f%%), "
            "avg_conf=%.3f, pred_dist=%s, gt_dist=%s",
            correct,
            total,
            accuracy * 100,
            avg_conf,
            dict(pred_dist.most_common()),
            dict(gt_dist.most_common()),
        )
        confusion: dict[str, Counter] = {}
        for pred, gt in zip(pred_countries, gt_countries, strict=True):
            confusion.setdefault(gt, Counter())[pred] += 1
        for gt in sorted(confusion):
            parts = []
            for pred, cnt in confusion[gt].most_common(3):
                parts.append(f"{pred}={cnt}")
            logger.debug("  GT=%s → %s", gt, ", ".join(parts))

    def _decode_countries(
        self,
        country_logits: Tensor,
    ) -> list[str]:
        """Decode country logits to country names."""
        country_list = self.plate_config.country_list
        n = len(country_list)
        indices = country_logits[:, :n].argmax(dim=1).tolist()
        return [country_list[i] for i in indices]

    def _decode_formats(
        self,
        format_logits: Tensor,
    ) -> list[str]:
        """Decode format logits to plate type strings."""
        indices = format_logits.argmax(dim=1)
        return [
            "square" if int(idx.item()) == 1 else "standard" for idx in indices
        ]

    @staticmethod
    def _filter_by_type(
        preds: list[str],
        targets: list[str],
        plate_types: list[str],
        target_type: str,
    ) -> tuple[list[str], list[str]]:
        """Filter predictions and targets by plate type."""
        filtered_p, filtered_t = [], []
        for p, t, pt in zip(preds, targets, plate_types, strict=True):
            if pt == target_type:
                filtered_p.append(p)
                filtered_t.append(t)
        return filtered_p, filtered_t
