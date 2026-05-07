"""PlateBiLSTM: bidirectional LSTM for sequence modeling."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class PlateBiLSTM(nn.Module):
    """Bidirectional LSTM for plate sequence processing."""

    def __init__(
        self,
        input_size: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    @property
    def hidden_size(self) -> int:
        """LSTM hidden size."""
        return self.lstm.hidden_size

    def forward(self, x: Tensor) -> Tensor:
        """Process sequence.

        x: (B, L, input_size)
        -> (B, L, hidden_size*2)
        """
        out, _ = self.lstm(x)
        return out
