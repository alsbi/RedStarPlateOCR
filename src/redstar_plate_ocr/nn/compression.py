"""Adaptive compression: attention-based pooling for standard/square."""

from __future__ import annotations

import torch
from torch import Tensor, nn

_ATTN_MASK_VALUE = -1e4  # Large negative for attention masking


class AttentionPool(nn.Module):
    """Learnable attention-based pooling over height dimension."""

    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        mid = max(in_channels // reduction, 1)
        self.attn_proj = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
        )

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        """Pool over height using learned attention.

        Args:
            features: (B, C, H, W)
            mask: (B, 1, H, W) — 1 for content, 0 for padding

        Returns:
            (B, C, W) — pooled features
        """
        attn_logits = self.attn_proj(features)
        attn_logits = attn_logits + (1.0 - mask) * _ATTN_MASK_VALUE
        attn_weights = attn_logits.softmax(dim=2)
        return (features * attn_weights).sum(dim=2)


class AdaptiveCompression(nn.Module):
    """Adaptive compression: standard -> (B,48,C), square -> (B,96,C)."""

    def __init__(
        self,
        canvas_height: int = 80,
        canvas_width: int = 192,
        stride: int = 4,
        in_channels: int = 256,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.feat_h = canvas_height // stride
        self.feat_w = canvas_width // stride
        self.attn_pool = AttentionPool(in_channels)

    def compute_input_lengths(
        self,
        content_mask: Tensor,
        plate_types: list[str],
    ) -> Tensor:
        """Compute input_lengths from content_mask.

        Vectorized: no Python loop, no .item() GPU sync.
        For standard plates: count columns with any content.
        For square plates: sum top-half and bottom-half
        column counts.

        Args:
            content_mask: (B, 1, feat_h, feat_w)
            plate_types: list of plate_type for each
                sample in batch (length B)

        Returns:
            (B,) long tensor with input_lengths
        """
        col_present = (content_mask[:, 0].sum(dim=1) > 0).sum(dim=1)
        mid = content_mask.shape[2] // 2
        top_present = (content_mask[:, 0, :mid, :].sum(dim=1) > 0).sum(dim=1)
        bot_present = (content_mask[:, 0, mid:, :].sum(dim=1) > 0).sum(dim=1)
        # Square layout: [top_feat_w_cols | bot_feat_w_cols]
        # top occupies positions 0..feat_w-1, bot starts at feat_w.
        # Must include at least feat_w + bot_present to cover bottom content.
        sq_lengths = self.feat_w + bot_present
        is_sq = torch.tensor(
            [pt == "square" for pt in plate_types],
            device=content_mask.device,
        )
        result = torch.where(is_sq, sq_lengths, col_present)
        return result.clamp(min=1)

    def compute_content_mask(
        self,
        batch_orig_h: Tensor,
        batch_orig_w: Tensor,
    ) -> Tensor:
        """Compute content mask using stored feat dims."""
        return compute_content_mask(
            batch_orig_h,
            batch_orig_w,
            feat_h=self.feat_h,
            feat_w=self.feat_w,
            stride=self.stride,
        )

    def _pool(self, features: Tensor, mask: Tensor) -> Tensor:
        """Attention-based pooling over height."""
        return self.attn_pool(features, mask)

    def forward_standard(
        self,
        features: Tensor,
        orig_h: Tensor,
        orig_w: Tensor,
        content_mask: Tensor | None = None,
    ) -> Tensor:
        """Attention pool over height.

        features: (B, C, H, W)
        orig_h, orig_w: (B,) original image dims
        content_mask: optional pre-computed (B, 1, feat_h, feat_w)
        -> (B, W, C) = (B, 48, C)
        """
        if content_mask is None:
            content_mask = self.compute_content_mask(orig_h, orig_w)
        result = self._pool(features, content_mask)
        return result.permute(0, 2, 1)

    def forward_square(
        self,
        features: Tensor,
        orig_h: Tensor,
        orig_w: Tensor,
        content_mask: Tensor | None = None,
    ) -> Tensor:
        """Split by height -> top/bot attention pool -> concat.

        features: (B, C, H, W)
        orig_h, orig_w: (B,) original image dims
        content_mask: optional pre-computed (B, 1, feat_h, feat_w)
        -> (B, 2*W, C) = (B, 96, C)
        """
        if content_mask is None:
            content_mask = self.compute_content_mask(orig_h, orig_w)
        mid = features.shape[2] // 2
        top_f = features[:, :, :mid, :]
        bot_f = features[:, :, mid:, :]
        top_m = content_mask[:, :, :mid, :]
        bot_m = content_mask[:, :, mid:, :]
        top_mean = self._pool(top_f, top_m)
        bot_mean = self._pool(bot_f, bot_m)
        combined = torch.cat([top_mean, bot_mean], dim=2)
        return combined.permute(0, 2, 1)


def compute_content_mask(
    batch_orig_h: Tensor,
    batch_orig_w: Tensor,
    feat_h: int,
    feat_w: int,
    stride: int,
) -> Tensor:
    """Compute content mask from original dimensions.

    Returns (B, 1, feat_h, feat_w) mask.

    Vectorized: no Python loop, no .item() GPU sync.
    """
    h_feat = (batch_orig_h // stride).clamp(max=feat_h)
    w_feat = (batch_orig_w // stride).clamp(max=feat_w)
    y = torch.arange(feat_h, device=batch_orig_h.device)
    x = torch.arange(feat_w, device=batch_orig_h.device)
    mask = (
        (y[None, :, None] < h_feat[:, None, None])
        & (x[None, None, :] < w_feat[:, None, None])
    ).to(dtype=torch.float32)
    return mask.unsqueeze(1)
