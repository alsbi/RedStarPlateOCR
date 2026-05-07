"""Combined loss for plate OCR: format + country + CTC + synergy."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from redstar_plate_ocr.nn.char_aux import build_char_targets
from redstar_plate_ocr.nn.model import ModelOutput
from redstar_plate_ocr.plate.config import PlateConfig

logger = logging.getLogger(__name__)


def _log_dropped(dropped: set[str], text: str) -> None:
    if dropped:
        logger.debug(
            "Dropped characters %s from text '%s' — not in alphabet.",
            sorted(dropped),
            text,
        )


def _resolve_map(
    alphabet: str,
    char_to_idx: dict[str, int] | None = None,
) -> dict[str, int]:
    return char_to_idx if char_to_idx is not None else {
        c: i for i, c in enumerate(alphabet)
    }


def _extract_indices(
    text: str,
    mapping: dict[str, int],
) -> tuple[list[int], set[str]]:
    indices: list[int] = []
    dropped: set[str] = set()
    for c in text:
        idx = mapping.get(c)
        if idx is not None:
            indices.append(idx)
        else:
            dropped.add(c)
    return indices, dropped


def text_to_indices(
    text: str,
    alphabet: str,
    char_to_idx: dict[str, int] | None = None,
) -> list[int]:
    """Convert text to list of indices by alphabet.

    If char_to_idx is provided, uses O(1) dict lookup
    instead of O(n) str.index.
    Characters not in alphabet are silently (!) dropped
    — callers should validate input beforehand.
    """
    mapping = _resolve_map(alphabet, char_to_idx)
    indices, dropped = _extract_indices(text, mapping)
    _log_dropped(dropped, text)
    return indices


class CombinedLoss(nn.Module):
    """Combined loss: α·L_fmt + β·L_ctry + γ·L_ctc − δ·synergy."""

    def __init__(
        self,
        plate_config: PlateConfig,
        format_weight: float = 1.0,
        country_weight: float = 0.7,
        ctc_weight: float = 1.0,
        label_smoothing: float = 0.01,
        country_label_smoothing: float = 0.1,
        synergy_weight: float = 0.0,
        char_aux_weight: float = 0.0,
    ):
        super().__init__()
        self.plate_config = plate_config
        self.format_weight = format_weight
        self.country_weight = country_weight
        self.ctc_weight = ctc_weight
        self.synergy_weight = synergy_weight
        self.char_aux_weight = char_aux_weight
        self.format_loss = nn.CrossEntropyLoss()
        self.country_loss = nn.CrossEntropyLoss(
            label_smoothing=country_label_smoothing
        )
        # Pre-computed dict for O(1) char→index lookup
        self._char_to_idx: dict[str, int] = {
            c: i for i, c in enumerate(plate_config.union_alphabet)
        }

    def forward(
        self,
        model_output: ModelOutput,
        gt_format: Tensor,
        gt_country: Tensor,
        gt_texts: list[str],
        input_lengths: Tensor,
    ) -> dict[str, Tensor]:
        """Compute combined loss and return components."""
        fmt_logits = model_output.format_logits
        l_fmt = self.format_loss(fmt_logits, gt_format)
        l_ctry = self.country_loss(
            model_output.country_logits,
            gt_country,
        )
        l_ctc, per_sample_ctc = self._compute_ctc_loss(
            model_output,
            gt_texts,
            input_lengths,
        )

        l_char_aux = self._compute_char_aux_loss(
            model_output,
            gt_texts,
        )

        weighted_sum = (
            self.format_weight * l_fmt
            + self.country_weight * l_ctry
            + self.ctc_weight * l_ctc
        )
        if l_char_aux is not None:
            weighted_sum = weighted_sum + self.char_aux_weight * l_char_aux

        if self.synergy_weight > 0:
            bonus = self.synergy_weight * self._compute_synergy_bonus(
                model_output,
                gt_format,
                gt_country,
                per_sample_ctc,
            )
        else:
            bonus = torch.tensor(
                0.0,
                device=weighted_sum.device,
            )

        total = torch.clamp(weighted_sum - bonus, min=0.0)

        result = {
            "total": total,
            "format": l_fmt,
            "country": l_ctry,
            "ctc": l_ctc,
            "synergy": bonus,
        }
        if l_char_aux is not None:
            result["char_aux"] = l_char_aux
        return result

    def _compute_ctc_loss(
        self,
        model_output: ModelOutput,
        gt_texts: list[str],
        input_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute CTC loss using union_alphabet.

        Returns:
            Tuple of (mean_loss, per_sample_losses) where
            per_sample_losses has shape (B,).
        """
        union_alphabet = self.plate_config.union_alphabet
        blank = len(union_alphabet)  # index after last char

        ctc = model_output.ctc_output

        targets_list: list[int] = []
        target_lengths_list: list[int] = []
        for text in gt_texts:
            indices_t = text_to_indices(
                text, union_alphabet, self._char_to_idx
            )
            targets_list.extend(indices_t)
            target_lengths_list.append(len(indices_t))

        # Guard against empty targets (all texts contain only
        # characters outside the alphabet).
        per_sample_losses: Tensor
        if not targets_list:
            batch_size = ctc.shape[0]
            zero = torch.zeros(batch_size, device=ctc.device)
            return zero.mean(), zero

        # CTCLoss expects (T, N, C), ctc is (N, T, C)
        log_probs = ctc.permute(1, 0, 2)

        # MPS does not implement aten::_ctc_loss —
        # compute CTC on CPU, then move result back.
        orig_device = log_probs.device
        compute_device = torch.device("cpu")
        needs_offload = orig_device.type == "mps"

        if needs_offload:
            log_probs = log_probs.to(compute_device)
            input_lengths = input_lengths.to(compute_device)

        targets = torch.tensor(
            targets_list,
            dtype=torch.long,
            device=compute_device,
        )
        target_lengths = torch.tensor(
            target_lengths_list,
            dtype=torch.long,
            device=compute_device,
        )

        loss_per_sample = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            zero_infinity=True,  # M2
            reduction="none",
        )

        if needs_offload:
            loss_per_sample = loss_per_sample.to(orig_device)

        per_sample_losses = loss_per_sample

        mean_loss = per_sample_losses.mean()
        return mean_loss, per_sample_losses

    def _compute_synergy_bonus(
        self,
        model_output: ModelOutput,
        gt_format: Tensor,
        gt_country: Tensor,
        per_sample_ctc: Tensor,
    ) -> Tensor:
        """Compute soft synergy bonus from joint correctness."""
        fmt_probs = F.softmax(model_output.format_logits, dim=-1)
        p_fmt = fmt_probs.gather(1, gt_format.unsqueeze(1)).squeeze(1)

        ctry_probs = F.softmax(
            model_output.country_logits,
            dim=-1,
        )
        p_ctry = ctry_probs.gather(
            1,
            gt_country.unsqueeze(1),
        ).squeeze(1)

        # Clamp CTC loss to prevent exp(-inf) = inf.
        # CTC loss > 50 means completely wrong prediction;
        # exp(-50) ≈ 1.9e-22 is effectively zero.
        clamped = torch.clamp(per_sample_ctc, max=50.0)
        p_text = torch.exp(-clamped)

        synergy = p_fmt * p_ctry * p_text
        return synergy.mean()

    def _compute_char_aux_loss(
        self,
        model_output: ModelOutput,
        gt_texts: list[str],
    ) -> Tensor | None:
        """Compute character-level auxiliary cross-entropy loss.

        Returns None if char_aux_logits is not present or
        char_aux_weight is 0.
        """
        if self.char_aux_weight <= 0:
            return None
        logits = model_output.char_aux_logits
        if logits is None:
            return None

        B, W, _ = logits.shape
        union_alphabet = self.plate_config.union_alphabet
        blank_idx = len(union_alphabet)  # 36

        losses = []
        for i in range(B):
            targets = build_char_targets(
                gt_texts[i],
                union_alphabet,
                width=W,
                blank_idx=blank_idx,
                char_to_idx=self._char_to_idx,
            )
            t = torch.tensor(
                targets,
                dtype=torch.long,
                device=logits.device,
            )
            # Slice logits to union_alphabet_size + 1 (blank)
            vocab_size = min(logits.shape[-1], blank_idx + 1)
            sample_logits = logits[i, :, :vocab_size]
            losses.append(
                F.cross_entropy(
                    sample_logits,
                    t[:W],
                    ignore_index=-100,
                )
            )

        if not losses:
            return None
        return torch.stack(losses).mean()
