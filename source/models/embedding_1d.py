from __future__ import annotations

import torch
import torch.nn as nn


class Embedding_NPE(nn.Module):

    def __init__(self, seq_len, kernel_size=9, stride=1, padding=4,
                 dropout_conv=0, pooling_ratio=4, d_out=128):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.AvgPool1d(kernel_size=pooling_ratio, stride=pooling_ratio),
            nn.Dropout1d(p=dropout_conv),

            nn.Conv1d(16, 32, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.AvgPool1d(kernel_size=pooling_ratio, stride=pooling_ratio),
            nn.Dropout1d(p=dropout_conv),

            nn.Conv1d(32, 32, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.AvgPool1d(kernel_size=pooling_ratio, stride=pooling_ratio),
            nn.Dropout1d(p=dropout_conv),
        )

        # +2 for the mean and log-std statistics appended before the head
        self.head = nn.Linear(32 + 2, d_out)

    def forward(self, x):
        # Accept (seq_len,), (B, seq_len), or (B, 1, seq_len).
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        mean = torch.mean(x, dim=-1)
        std = torch.std(x, dim=-1).clamp(min=1e-8)
        stats = torch.cat([mean, torch.log(std)], dim=1)

        z = self.layers(x)
        z = z.mean(dim=-1)                     # global average pool
        z = torch.cat([z, stats], dim=1)

        return self.head(z)


class ResBlock1D(nn.Module):

    def __init__(self, channels, kernel_size=9, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class EmbeddingResNet(nn.Module):

    def __init__(self, seq_len, kernel_size=9, stride=1, padding=4,
                 dropout_conv=0.1, pooling_ratio=4, d_out=128,
                 n_res_blocks=4):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size, padding=padding),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_conv),
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock1D(128, kernel_size, dropout_conv)
              for _ in range(n_res_blocks)]
        )

        self.pool = nn.AdaptiveAvgPool1d(pooling_ratio)
        self.fc = nn.Linear(128 * pooling_ratio, d_out)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)