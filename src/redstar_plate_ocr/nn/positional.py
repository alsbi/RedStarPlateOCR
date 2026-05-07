"""Sinusoidal positional encoding for sequence models.

Adds absolute position information to the LSTM input so the
model can distinguish the horizontal order of characters —
critical for avoiding adjacent same-type transpositions
(e.g. ``CX`` vs ``XC``) that pure CTC cannot resolve.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017).

    Injects absolute position information into a sequence by
    adding a fixed (non-learned) sinusoidal signal.  Because
    the encoding is deterministic, it requires no additional
    training parameters and generalises to any sequence length.

    For CTC-based OCR this is particularly important: the
    BiLSTM sees a sequence of compressed column features but
    has no explicit signal about which column came from which
    horizontal position.  Without positional encoding, the
    LSTM can confuse the order of adjacent characters of the
    same type (e.g. ``C`` vs ``X`` on an ``XX`` pattern slot)
    because their visual features are very similar and the
    only distinguishing factor is *where* on the plate they
    appear.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 256,
        dropout: float = 0.0,
    ) -> None:
        """Initialise positional encoding table.

        Args:
            d_model: Feature dimension (must match LSTM input_size).
            max_len: Maximum sequence length to pre-compute.
                For standard plates ≈ 48, for square ≈ 96.
                256 is a generous upper bound.
            dropout: Optional dropout after adding PE.
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        pe = self._build_pe_table(d_model, max_len)
        self.register_buffer("_pe", pe, persistent=False)
        self._pe: Tensor

    @staticmethod
    def _build_pe_table(d_model: int, max_len: int) -> Tensor:
        """Build the (max_len, d_model) sinusoidal table."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to input sequence.

        Args:
            x: (B, T, d_model) — compressed sequence features.

        Returns:
            (B, T, d_model) — features with positional encoding
            added (and optional dropout applied).
        """
        T = x.size(1)
        pe_len = self._pe.size(0)
        if T <= pe_len:
            pe = self._pe[:T]
        else:
            # Extend PE table on-the-fly for long sequences (rare)
            extra = self._build_pe_table(self.d_model, T - pe_len)
            # Offset positions so the continuation is seamless
            pos_offset = torch.arange(
                pe_len, T, dtype=torch.float32
            ).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, dtype=torch.float32)
                * (-math.log(10000.0) / self.d_model)
            )
            extra[:, 0::2] = torch.sin(pos_offset * div_term)
            extra[:, 1::2] = torch.cos(pos_offset * div_term)
            pe = torch.cat([self._pe, extra], dim=0)
        return self.dropout(x + pe)
