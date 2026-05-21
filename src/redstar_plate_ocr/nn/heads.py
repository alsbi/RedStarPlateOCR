"""Classification and CTC heads for plate recognition."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# Default canvas dimensions used in FormatHead for normalising orig_h/w.
_FORMAT_CANVAS_H = 80
_FORMAT_CANVAS_W = 256


def _grid_pool(
    x: Tensor,
    grid_rows: int,
    grid_cols: int,
    gap: nn.Module,
    mask: Tensor | None = None,
) -> Tensor:
    """Adaptive grid pooling shared by attention modules.

    Falls back to global pooling when the feature map is too small
    for the configured grid. Extracted from the two original
    ``AttentionCompression``/``SeparateCountryBranch`` copies.

    When *mask* is provided (shape ``(B, 1, H, W)``), uses masked
    average pooling — sums only content pixels and divides by the
    content count.  This strips spatial layout information so the
    head cannot cheat by reading the padding pattern.
    """
    B, C, H, W = x.shape

    if mask is not None and mask.shape[-2:] != (H, W):
        mask = None  # shape mismatch — fall back to unmasked

    if H < grid_rows or W < grid_cols:
        if mask is not None:
            return _masked_global_pool(x, mask, grid_rows * grid_cols)
        return gap(x).flatten(1).repeat(1, grid_rows * grid_cols)

    rh = H // grid_rows
    rw = W // grid_cols
    pools: list[Tensor] = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            region = x[:, :, i * rh : (i + 1) * rh, j * rw : (j + 1) * rw]
            if mask is not None:
                region_mask = mask[
                    :, :, i * rh : (i + 1) * rh, j * rw : (j + 1) * rw
                ]
                pooled = _masked_pool_region(region, region_mask)
            else:
                pooled = gap(region).flatten(1)
            pools.append(pooled)
    return torch.cat(pools, dim=1)


def _masked_pool_region(x: Tensor, mask: Tensor) -> Tensor:
    """Masked average pool over a single grid region.

    Args:
        x:     ``(B, C, H, W)`` feature region.
        mask:  ``(B, 1, H, W)`` binary mask (1=content, 0=padding).
    """
    # Sum over spatial dims, divide by content-pixel count (per sample)
    summed = (x * mask).sum(dim=(-2, -1))  # (B, C)
    count = mask.sum(dim=(-2, -1)).clamp(min=1.0)  # (B, 1)
    return summed / count  # (B, C)


def _masked_global_pool(x: Tensor, mask: Tensor, repeat: int) -> Tensor:
    """Global masked average pool, repeated for grid compatibility."""
    summed = (x * mask).sum(dim=(-2, -1))  # (B, C)
    count = mask.sum(dim=(-2, -1)).clamp(min=1.0)  # (B, 1)
    pooled = summed / count  # (B, C)
    return pooled.repeat(1, repeat)


class FormatHead(nn.Module):
    """Plate shape -> format logits (standard=0, square=1).

    Format is a geometric property — it depends only on the plate's
    shape, not on its visual content.  The head therefore works in
    two stages:

    1. **Shape encoder**: a tiny CNN runs on the *content_mask*
       (a binary H×W map showing where the plate is).  This
       captures the plate's contour — where content begins/ends,
       the aspect ratio of the content region — without seeing
       any pixel values inside the plate.

    2. **Dimension features**: normalised ``orig_h`` and ``orig_w``
       are concatenated as explicit features.

    No backbone feature channels are used, eliminating any
    possibility of the head reading character content.
    """

    def __init__(
        self,
        in_channels: int,  # kept for API compat, unused
        dropout: float = 0.1,
        hidden_size: int | None = None,
    ):
        super().__init__()
        # Shape encoder: tiny CNN on content_mask (1-channel binary map)
        # feat_h=20, feat_w=64 → small enough for cheap conv
        self.shape_enc = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # → (B, 8, 4, 4)
        )
        _SHAPE_ENC_CHANNELS = 8
        _SHAPE_ENC_POOL = 4  # AdaptiveAvgPool2d output spatial size
        shape_dim = _SHAPE_ENC_CHANNELS * _SHAPE_ENC_POOL**2  # 128
        self._shape_dim = shape_dim
        # +2 for normalised h, w
        fc_in = shape_dim + 2

        self.drop = nn.Dropout(dropout)
        if hidden_size is not None:
            self.fc = nn.Sequential(
                nn.Linear(fc_in, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 2),
            )
        else:
            self.fc = nn.Linear(fc_in, 2)

    def forward(
        self,
        features: Tensor,  # kept for API compat, unused
        content_mask: Tensor | None = None,
        orig_h: Tensor | None = None,
        orig_w: Tensor | None = None,
    ) -> Tensor:
        b = features.shape[0]
        device = features.device

        # Encode shape from content_mask
        if content_mask is not None:
            x = self.shape_enc(content_mask)  # (B, 8, 4, 4)
            x = x.flatten(1)  # (B, 128)
        else:
            # Fallback: zeros (e.g. ONNX with fixed dims)
            x = torch.zeros(b, self._shape_dim, device=device)

        # Append normalised plate dimensions
        if orig_h is not None and orig_w is not None:
            h_ratio = orig_h.float().unsqueeze(1) / _FORMAT_CANVAS_H
            w_ratio = orig_w.float().unsqueeze(1) / _FORMAT_CANVAS_W
            x = torch.cat([x, h_ratio, w_ratio], dim=1)
        else:
            x = torch.cat([x, torch.ones(b, 2, device=device)], dim=1)

        x = self.drop(x)
        return self.fc(x)


class CountryHead(nn.Module):
    """Masked GAP -> Linear(C, num_countries) -> logits.

    Uses *content_mask* to strip spatial layout information so the
    head cannot cheat by reading the padding / aspect-ratio pattern.
    """

    def __init__(
        self,
        in_channels: int,
        num_countries: int,
        dropout: float = 0.1,
        hidden_size: int | None = None,
    ):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        if hidden_size is not None:
            self.fc = nn.Sequential(
                nn.Linear(in_channels, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_countries),
            )
        else:
            self.fc = nn.Linear(in_channels, num_countries)

    def forward(
        self, features: Tensor, content_mask: Tensor | None = None
    ) -> Tensor:
        if content_mask is not None:
            features = features * content_mask
            summed = features.sum(dim=(-2, -1))
            count = content_mask.sum(dim=(-2, -1)).clamp(min=1.0)
            x = summed / count
        else:
            x = self.gap(features).flatten(1)
        x = self.drop(x)
        return self.fc(x)


class PositionAwareCountryHead(nn.Module):
    """Conv + Masked Grid GAP -> MLP -> country logits.

    Uses *content_mask* to strip spatial layout information:
    features are zeroed in padding regions **before** the conv,
    and grid pooling uses masked average so it cannot infer the
    content / padding boundary.
    """

    @property
    def final_layer(self) -> nn.Module:
        """Return the last linear layer for test access."""
        return self.fc3

    def __init__(
        self,
        in_channels: int,
        num_countries: int,
        conv_channels: int = 144,
        grid_rows: int = 2,
        grid_cols: int = 3,
        hidden_size: int = 288,
        dropout: float = 0.3,
        pos_aware: bool = True,
    ) -> None:
        super().__init__()
        self._grid_rows = grid_rows
        self._grid_cols = grid_cols
        self._pos_aware = pos_aware
        self.gap = nn.AdaptiveAvgPool2d(1)
        if pos_aware:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, conv_channels, 3, padding=1),
                nn.BatchNorm2d(conv_channels),
                nn.SiLU(),
            )
            grid_dim = grid_rows * grid_cols * conv_channels
            self.fc1 = nn.Sequential(
                nn.Linear(grid_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.fc2 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.fc3 = nn.Linear(hidden_size, num_countries)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, conv_channels, 3, padding=1),
                nn.BatchNorm2d(conv_channels),
                nn.SiLU(),
            )
            self.fc1 = nn.Sequential(
                nn.Linear(conv_channels, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.fc2 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.fc3 = nn.Linear(hidden_size, num_countries)

    def forward(
        self, features: Tensor, content_mask: Tensor | None = None
    ) -> Tensor:
        # Zero out padding regions so conv never sees spatial layout
        if content_mask is not None:
            features = features * content_mask
        x = self.conv(features)
        if self._pos_aware:
            x = _grid_pool(
                x,
                self._grid_rows,
                self._grid_cols,
                self.gap,
                mask=content_mask,
            )
        else:
            if content_mask is not None:
                summed = (x * content_mask).sum(dim=(-2, -1))
                count = content_mask.sum(dim=(-2, -1)).clamp(min=1.0)
                x = summed / count
            else:
                x = self.gap(x).flatten(1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class UnifiedCTCHead(nn.Module):
    """Single CTC head with union alphabet + country-conditioned masking.

    Architecture (when hidden_size is set):
        Linear(input_size → hidden_size) → LayerNorm → ReLU → Dropout
        + residual (if input_size == hidden_size)
        → Linear(hidden_size → union_alphabet_size)

    The residual connection preserves the full BiLSTM signal so the
    projection layer only needs to learn the *delta* — making it
    both faster to converge and less likely to lose information.
    """

    mask_table: Tensor

    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int | None = None,
        union_alphabet_size: int = 37,
        mask_table: Tensor | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size is not None and hidden_size > 0:
            self.proj = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.fc = nn.Linear(hidden_size, union_alphabet_size)
            self._use_residual = input_size == hidden_size
        else:
            # Fallback: no hidden, project in-place then classify
            self.proj = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.ReLU(),
            )
            self.fc = nn.Linear(input_size, union_alphabet_size)
            self._use_residual = True

        if mask_table is not None:
            self.register_buffer("mask_table", mask_table)
        else:
            self.register_buffer(
                "mask_table",
                torch.zeros(1, union_alphabet_size),
            )

    def forward(
        self,
        lstm_out: Tensor,
        effective_mask: Tensor,
    ) -> Tensor:
        """proj→(+residual)→fc→mask→log_softmax.

        Args:
            lstm_out: BiLSTM output, shape (B, T, H).
            effective_mask: Pre-computed mask, shape (B, C),
                (B, 1, C), or (B, T, C).
        """
        logits = self.forward_raw(lstm_out)
        if effective_mask.dim() == 2:
            # (B, C) -> (B, 1, C) for broadcast over T
            effective_mask = effective_mask.unsqueeze(1)
        logits = logits + effective_mask
        return torch.log_softmax(logits, dim=-1)

    def forward_raw(self, lstm_out: Tensor) -> Tensor:
        """proj→(+residual)→fc, no masking — for ONNX."""
        x = self.proj(lstm_out)
        if self._use_residual:
            x = x + lstm_out
        return self.fc(x)
