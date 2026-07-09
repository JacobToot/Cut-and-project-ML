"""
diffraction_2d.py
=================

Diffraction-only baseline classifier for quasicrystal parent-lattice
dimension. A CNN reads the diffraction pattern (structure factor) computed
ON THE FLY from the (augmented / cropped) tiling vertices via the type-1
NUFFT in `nufft2d.py`.

Exposes the same interface as the other models in this project
(histogram_2d, gnn_2d, deepsets_2d):

    DimensionalityClassifier(...).forward(points, mask) -> logits
    .predict(points, mask) -> dimension labels
    .summary(points, mask) -> (B, d_summary) feature vector  [reused by fusion]

The diffraction image is treated as fixed data (no autograd through the
NUFFT); gradients flow only through the CNN.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

try:                                            # models/ on sys.path
    from utils.nufft2d import DiffractionImager, DiffractionConfig
except ImportError:                             # repo-root import style
    from utils.nufft2d import DiffractionImager, DiffractionConfig


class DiffractionBackbone(nn.Module):
    """Small CNN: (B,1,H,W) -> (B, d_summary).

    Stem at full resolution, then three stride-2 (max-pool) stages doubling
    channels, global average pool, project to d_summary. Resolution-agnostic
    thanks to the adaptive pool, so grid_size can change without code edits."""

    def __init__(self, in_ch: int = 1, cnn_width: int = 32,
                 d_summary: int = 128, dropout: float = 0.1):
        super().__init__()
        w = cnn_width
        widths = [w, w * 2, w * 4, w * 8]

        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, widths[0], 3, 1, 1),
            nn.BatchNorm2d(widths[0]), nn.GELU(),
        ]
        c = widths[0]
        for nxt in widths[1:]:
            layers += [
                nn.Conv2d(c, nxt, 3, 1, 1),
                nn.BatchNorm2d(nxt), nn.GELU(),
                nn.MaxPool2d(2),
            ]
            c = nxt

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c, d_summary), nn.GELU(),
            nn.Linear(d_summary, d_summary),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(self.features(image)))


class DimensionalityClassifier(nn.Module):
    """Diffraction-only classifier.

    Parameters
    ----------
    d_min, d_max : inclusive range of dimension classes.
    d_summary    : CNN summary / head hidden width.
    n_layers     : depth of the classification head.
    cnn_width    : base channel width of the CNN (channels: w, 2w, 4w, 8w).
    dropout      : dropout in CNN projection and head.
    diffraction  : DiffractionConfig (or dict) controlling the on-the-fly
                   pattern: grid_size, q_max, backend ("nufft"/"direct"),
                   normalization, log/standardize, DC suppression.

    Extra keyword arguments (e.g. d_node, k from a shared training script)
    are accepted and ignored so this model is a drop-in for the others.
    """

    def __init__(
        self,
        d_min: int = 4,
        d_max: int = 9,
        d_summary: int = 128,
        n_layers: int = 2,
        cnn_width: int = 32,
        dropout: float = 0.1,
        diffraction=None,
        **ignored,
    ):
        super().__init__()
        self.d_min = d_min
        self.d_max = d_max
        self.n_classes = d_max - d_min + 1

        if isinstance(diffraction, dict):
            diffraction = DiffractionConfig(**diffraction)
        self.imager = DiffractionImager(diffraction or DiffractionConfig())

        self.backbone = DiffractionBackbone(
            in_ch=1, cnn_width=cnn_width, d_summary=d_summary, dropout=dropout)

        head: list[nn.Module] = []
        for _ in range(max(n_layers, 1)):
            head += [nn.Linear(d_summary, d_summary), nn.GELU(),
                     nn.Dropout(dropout)]
        head.append(nn.Linear(d_summary, self.n_classes))
        self.head = nn.Sequential(*head)

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        image = self.imager(points, mask)        # (B,1,H,W); no grad inside
        return self.head(self.backbone(image))

    def predict(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.forward(points, mask).argmax(dim=-1) + self.d_min

    def summary(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.imager(points, mask))