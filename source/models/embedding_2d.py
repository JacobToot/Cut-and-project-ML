"""embedding_2d.py
================

2D embedding networks used by the 2D NPE model and the 2D
dimension classifier.

Classes
-------
Shared:
    ResBlock2D
        Pre-activation 2D residual block used by DiffractionConditioner.

For the NPE model (``npe_2d``):
    DiffractionConditioner
    EdgeHistogramConditioner

For the classifier model (``dim_classifier_2d``):
    DiffractionBackbone.
    """

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Shared 2D residual block
# ---------------------------------------------------------------------

class ResBlock2D(nn.Module):
    """Pre-activation 2D residual block.

    Layout: BN -> ReLU -> Conv3x3 -> Dropout2d -> BN -> ReLU -> Conv3x3,
    added back to the input. All convolutions are 3x3, padding 1,
    bias=False (BN handles the bias). Preserves spatial resolution.
    """

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=1, bias=False)
        self.dropout = (nn.Dropout2d(dropout) if dropout > 0
                        else nn.Identity())

    def forward(self, x):
        h = F.relu(self.bn1(x))
        h = self.conv1(h)
        h = self.dropout(h)
        h = F.relu(self.bn2(h))
        h = self.conv2(h)
        return x + h


# ---------------------------------------------------------------------
# NPE 2D diffraction + edge histogram
# ---------------------------------------------------------------------

class DiffractionConditioner(nn.Module):
    """ResNet CNN over diffraction images: (B, 1, H, W) -> (B, d_out).
    """

    def __init__(
        self, *,
        in_channels: int = 1,
        cnn_width: int = 32,
        d_out: int = 128,
        n_res_per_stage: int = 2,
        n_head_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        w = cnn_width
        widths = [w, 2 * w, 4 * w, 8 * w]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )

        stages: list[nn.Module] = []
        prev = widths[0]
        for ch in widths[1:]:
            stage: list[nn.Module] = [
                nn.Conv2d(prev, ch, kernel_size=3, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ]
            for _ in range(n_res_per_stage):
                stage.append(ResBlock2D(ch, dropout=dropout))
            stages.append(nn.Sequential(*stage))
            prev = ch
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)

        head_layers: list[nn.Module] = []
        in_dim = widths[-1]
        for _ in range(max(0, n_head_layers - 1)):
            head_layers.extend([
                nn.Linear(in_dim, d_out),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = d_out
        head_layers.extend([
            nn.Linear(in_dim, d_out),
            nn.LayerNorm(d_out),
        ])
        self.head = nn.Sequential(*head_layers)

        self.d_out = d_out

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.stem(img)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class EdgeHistogramConditioner(nn.Module):
    """1D CNN over an edge-length histogram.

    Input: (B, n_bins) or (B, 1, n_bins).
    Output: (B, d_out).
    """

    def __init__(
        self, *,
        n_bins: int = 64,
        base_width: int = 32,
        d_out: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        w = base_width
        self.net = nn.Sequential(
            nn.Conv1d(1, w, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(w),
            nn.ReLU(inplace=True),

            nn.Conv1d(w, 2 * w, kernel_size=5, stride=2,
                      padding=2, bias=False),
            nn.BatchNorm1d(2 * w),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(2 * w, 2 * w, kernel_size=5, stride=2,
                      padding=2, bias=False),
            nn.BatchNorm1d(2 * w),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(2 * w, d_out),
            nn.LayerNorm(d_out),
        )
        self.d_out = d_out

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        if hist.dim() == 2:
            hist = hist.unsqueeze(1)   # (B, n_bins) -> (B, 1, n_bins)
        x = self.net(hist)
        x = self.pool(x).flatten(1)
        return self.head(x)


# ---------------------------------------------------------------------
# Classifier for dim_classifier_2d
# ---------------------------------------------------------------------

class DiffractionBackbone(nn.Module):
    """CNN for the 2D dimension classifier: (B, 1, H, W) -> (B, d_summary).
    """

    def __init__(self, in_ch: int = 1, cnn_width: int = 32,
                 d_summary: int = 128, dropout: float = 0.1):
        super().__init__()
        w = cnn_width
        widths = [w, w * 2, w * 4, w * 8]

        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, widths[0], 3, 1, 1),
            nn.BatchNorm2d(widths[0]),
            nn.GELU(),
        ]
        c = widths[0]
        for nxt in widths[1:]:
            layers += [
                nn.Conv2d(c, nxt, 3, 1, 1),
                nn.BatchNorm2d(nxt),
                nn.GELU(),
                nn.MaxPool2d(2),
            ]
            c = nxt

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c, d_summary),
            nn.GELU(),
            nn.Linear(d_summary, d_summary),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(self.features(image)))