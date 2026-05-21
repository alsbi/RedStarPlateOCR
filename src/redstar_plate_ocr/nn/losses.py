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
    return (
        char_to_idx
        if char_to_idx is not None
        else {c: i for i, c in enumerate(alphabet)}
    )


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


def _collect_alpha_letters(plate_config: PlateConfig) -> set[str]:
    """Collect true alphabetic letters from all regions.

    valid_chars.letters may contain '-' or other separators
    that are *not* real letters; pairing them with letters
    should NOT trigger the order penalty.
    """
    letters: set[str] = set()
    for rc in plate_config.regions.values():
        for c in rc.valid_chars.letters:
            if c.isalpha():
                letters.add(c)
    return letters


def _build_same_type_pairs(
    union_alphabet: str,
    letters_set: set[str],
) -> dict[tuple[str, str], bool]:
    """Pre-compute same-type lookup for order penalty."""
    digits_set = set("0123456789")
    pairs: dict[tuple[str, str], bool] = {}
    for c1 in union_alphabet:
        for c2 in union_alphabet:
            if c1 == c2:
                continue
            both_digits = c1 in digits_set and c2 in digits_set
            both_letters = c1 in letters_set and c2 in letters_set
            pairs[(c1, c2)] = both_digits or both_letters
    return pairs


class CombinedLoss(nn.Module):
    """Combined loss: α·L_fmt + β·L_ctry + γ·L_ctc + ε·L_order − δ·synergy."""

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
        order_weight: float = 0.0,
        order_margin: float = 1.0,
        length_weight: float = 0.0,
    ):
        super().__init__()
        self.plate_config = plate_config
        self.format_weight = format_weight
        self.country_weight = country_weight
        self.ctc_weight = ctc_weight
        self.synergy_weight = synergy_weight
        self.char_aux_weight = char_aux_weight
        self.order_weight = order_weight
        self.order_margin = order_margin
        self.length_weight = length_weight
        self.format_loss = nn.CrossEntropyLoss()
        self.country_loss = nn.CrossEntropyLoss(
            label_smoothing=country_label_smoothing
        )
        self._char_to_idx: dict[str, int] = {
            c: i for i, c in enumerate(plate_config.union_alphabet)
        }
        letters_set = _collect_alpha_letters(plate_config)
        self._same_type_pairs = _build_same_type_pairs(
            plate_config.union_alphabet, letters_set
        )

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

        l_order = self._compute_order_penalty(
            model_output.ctc_output,
            gt_texts,
            input_lengths,
        )

        weighted_sum = (
            self.format_weight * l_fmt
            + self.country_weight * l_ctry
            + self.ctc_weight * l_ctc
        )
        if l_char_aux is not None:
            weighted_sum = weighted_sum + self.char_aux_weight * l_char_aux
        if self.order_weight > 0:
            weighted_sum = weighted_sum + self.order_weight * l_order

        l_length = self._compute_length_loss(
            model_output.ctc_output,
            gt_texts,
            input_lengths,
        )
        if l_length is not None:
            weighted_sum = weighted_sum + self.length_weight * l_length

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
            "order": l_order,
            "synergy": bonus,
        }
        if l_char_aux is not None:
            result["char_aux"] = l_char_aux
        if l_length is not None:
            result["length"] = l_length
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

    def _compute_order_penalty(
        self,
        ctc_output: Tensor,
        gt_texts: list[str],
        input_lengths: Tensor,
    ) -> Tensor:
        """Penalise wrong temporal order of adjacent same-type characters.

        For every pair of adjacent characters ``(a, c_next)`` in a GT
        text where both are the same type (both letters or both
        digits), compute the soft expected timestep of each
        character's emission and check the left-to-right order.

        The **soft peak** is the probability-weighted average
        timestep:

            soft_peak(c) = Σ_t  softmax(log_probs[:, c])[t] · t

        This is fully differentiable — gradients flow back through
        the softmax to the CTC log-probabilities, allowing the
        model to learn correct temporal ordering.

        Correct order: character ``a`` (earlier in the text) should
        peak at an earlier timestep than ``c_next`` (later in the
        text), i.e. ``soft_peak_a < soft_peak_c``.  When reversed,
        a penalty proportional to the violation is applied:

            penalty += ReLU(soft_peak_a - soft_peak_c - margin)

        The *margin* parameter (default 1) allows small timestep
        overlaps without penalty — CTC alignments are not pixel-
        perfect and a 1-step tolerance is reasonable.

        Returns 0.0 tensor when ``order_weight`` is 0 or no
        same-type pairs exist in the batch.
        """
        if self.order_weight <= 0:
            return torch.tensor(0.0, device=ctc_output.device)

        char_to_idx = self._char_to_idx
        margin = self.order_margin

        penalties: list[Tensor] = []
        B = ctc_output.shape[0]

        for b in range(B):
            text = gt_texts[b]
            T = input_lengths[b].item()
            sample_logits = ctc_output[b, :T]  # (T, V)

            # Timestep indices as a tensor for soft peak computation
            t_range = torch.arange(
                T, dtype=torch.float32, device=ctc_output.device
            )

            sample_penalty = torch.tensor(0.0, device=ctc_output.device)
            count = 0

            for k in range(len(text) - 1):
                a, c_next = text[k], text[k + 1]
                # Only penalise same-type adjacent pairs
                if not self._same_type_pairs.get((a, c_next), False):
                    continue

                a_idx = char_to_idx.get(a)
                c_idx = char_to_idx.get(c_next)
                if a_idx is None or c_idx is None:
                    continue

                # Soft peak: probability-weighted average timestep.
                # softmax over T dimension gives a probability
                # distribution over timesteps for each character.
                probs_a = torch.softmax(sample_logits[:, a_idx], dim=0)
                probs_c = torch.softmax(sample_logits[:, c_idx], dim=0)
                soft_peak_a = (probs_a * t_range).sum()
                soft_peak_c = (probs_c * t_range).sum()

                # Penalty: a (earlier in text) should peak at an
                # earlier timestep than c_next.  Correct order:
                # soft_peak_a < soft_peak_c.  When reversed the
                # model emits the later character first.
                diff = soft_peak_a - soft_peak_c  # positive = wrong
                # ReLU with margin: small violations are tolerated
                penalty = torch.clamp(diff - margin, min=0.0)
                sample_penalty = sample_penalty + penalty
                count += 1

            if count > 0:
                sample_penalty = sample_penalty / count
            penalties.append(sample_penalty)

        if not penalties:
            return torch.tensor(0.0, device=ctc_output.device)

        batch_penalty = torch.stack(penalties).mean()
        return batch_penalty

    def _compute_length_loss(
        self,
        ctc_output: Tensor,
        gt_texts: list[str],
        input_lengths: Tensor,
    ) -> Tensor | None:
        """Differentiable length consistency loss.

        Censures enough non-blank emission probability along the
        sequence to cover every character in the ground-truth text.

        For each sample we compute the expected number of emitted
        non-blank symbols (sum of marginal non-blank probabilities
        over valid positions) and compare it against the true
        number of characters.  When this expectation drops below the
        target the model is penalised — forcing it to emit, not
        blank, in timesteps that should contain characters.

        This especially matters for optional trailing symbols (RU `o`)
        which CTC can silently skip without any loss penalty.

        Returns None when length_weight == 0.
        """
        if self.length_weight <= 0:
            return None

        union_alphabet = self.plate_config.union_alphabet
        blank_idx = len(union_alphabet)
        B = ctc_output.shape[0]
        losses: list[Tensor] = []

        for b in range(B):
            T = int(input_lengths[b].item())
            sample_logits = ctc_output[b, :T]  # (T, V)
            target_len = len(gt_texts[b])
            if target_len == 0:
                continue

            # Soft non-blank emission count:
            # For each timestep, probability mass on non-blank chars.
            probs = F.softmax(sample_logits, dim=-1)  # (T, V)
            non_blank_probs = probs[:, :blank_idx].sum(dim=-1)  # (T,)
            expected_count = non_blank_probs.sum()  # scalar

            # Under-emission penalty: if expected non-blank < target
            diff = target_len - expected_count
            if diff > 0:
                losses.append(diff * diff)

        if not losses:
            return torch.tensor(0.0, device=ctc_output.device)
        return torch.stack(losses).mean()
