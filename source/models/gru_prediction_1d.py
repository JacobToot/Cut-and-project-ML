from __future__ import annotations

import torch
import torch.nn as nn


class CausalGRUNet(nn.Module):
    """
    Simple GRU baseline for next-symbol prediction.
    Returns logits for every sequence position: (B, L, vocab_size).
    """

    def __init__(
        self,
        vocab_size: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        """
        x_ids: (B, L) long tokens
        return: (B, L, vocab_size)
        """
        x = self.embedding(x_ids)
        x = self.input_dropout(x)
        out, _ = self.gru(x)
        out = self.norm(out)
        logits = self.head(out)
        return logits
