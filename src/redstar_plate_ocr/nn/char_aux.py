"""Character-level auxiliary head for backbone gradient signal."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class CharAuxHead(nn.Module):
    """Conv1x1 on backbone features → per-position char classification.

    Gives backbone direct gradient signal before compression/LSTM.
    Input: (B, C, H, W) backbone features.
    Output: (B, W, max_alphabet_size) per-position logits.

    When *content_mask* is provided, padding regions are zeroed out
    before pooling so the head cannot infer plate dimensions from
    the spatial layout of padding values.
    """

    def __init__(
        self,
        in_channels: int,
        max_alphabet_size: int,
    ):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, None))  # pool H→1
        self.proj = nn.Conv1d(in_channels, max_alphabet_size, 1)

    def forward(
        self, features: Tensor, content_mask: Tensor | None = None
    ) -> Tensor:
        """(B, C, H, W) → (B, W, max_alphabet_size)."""
        if content_mask is not None:
            features = features * content_mask
            # Masked average over H per (b, w) column
            # content_mask: (B, 1, H, W) -> count per col: (B, 1, W)
            count = content_mask.sum(dim=2).clamp(min=1.0)  # (B, 1, W)
            summed = features.sum(dim=2)                     # (B, C, W)
            x = summed / count                                # (B, C, W)
        else:
            x = self.gap(features).squeeze(2)  # (B, C, W)
        x = self.proj(x)  # (B, max_alphabet, W)
        return x.permute(0, 2, 1)  # (B, W, max_alphabet)


def _resolve_char_indices(
    text: str,
    alphabet: str,
    char_to_idx: dict[str, int] | None = None,
) -> list[int]:
    """Map text characters to alphabet indices."""
    lookup = char_to_idx or {c: i for i, c in enumerate(alphabet)}
    return [lookup[c] for c in text if c in lookup]


def _fill_targets(
    chars: list[int],
    width: int,
    blank_idx: int,
) -> list[int]:
    """Spread characters evenly across width positions."""
    n = len(chars)
    per_char = width / n
    targets = [blank_idx] * width
    for i, ch_idx in enumerate(chars):
        start = int(i * per_char)
        end = int((i + 1) * per_char)
        for pos in range(start, end):
            if 0 <= pos < width:
                targets[pos] = ch_idx
    return targets


def build_char_targets(
    text: str,
    alphabet: str,
    width: int,
    blank_idx: int,
    char_to_idx: dict[str, int] | None = None,
) -> list[int]:
    """Build per-position char targets for auxiliary loss.

    Characters are evenly spread across width positions.
    Positions not covered by any char get blank_idx.

    If char_to_idx is provided, uses O(1) dict lookup
    instead of O(n) str.index.
    """
    chars = _resolve_char_indices(text, alphabet, char_to_idx)
    if not chars:
        return [blank_idx] * width
    return _fill_targets(chars, width, blank_idx)
