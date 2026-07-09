"""
npe_diffraction_2d.py
=====================

NPE for d=5 cut-and-project basis vectors with a two-channel conditioner:

    points -> NUFFT diffraction image -> 2D ResNet  -> diff feats  (B, 128)
          +--> Delaunay edge histogram -> 1D CNN    -> hist feats  (B,  64)
                                          log_mean_nn               (B,   1)
                                                                       |
                                                       concat + Linear+LN
                                                                       |
                                                                       v
                                                                   ctx (B, 128) -> flow
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 2D ResNet over diffraction images
# ----------------------------------------------------------------------------

class ResBlock2D(nn.Module):
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


class DiffractionConditioner(nn.Module):
    def __init__(
        self, *, in_channels=1, cnn_width=32, d_out=128,
        n_res_per_stage=2, n_head_layers=2, dropout=0.1,
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
        stages = []
        prev = widths[0]
        for ch in widths[1:]:
            stage = [
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
        head_layers.extend([nn.Linear(in_dim, d_out),
                            nn.LayerNorm(d_out)])
        self.head = nn.Sequential(*head_layers)
        self.d_out = d_out

    def forward(self, img):
        x = self.stem(img)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


# ----------------------------------------------------------------------------
# NEW: 1D CNN over the edge-length histogram
# ----------------------------------------------------------------------------

class EdgeHistogramConditioner(nn.Module):
    """Small 1D CNN over a fixed-bin edge-length histogram.

    For n_bins=64, resolution goes 64 -> 32 -> 16 before global average pool.
    """

    def __init__(self, *, n_bins=64, base_width=32, d_out=64, dropout=0.1):
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

    def forward(self, hist):
        if hist.dim() == 2:
            hist = hist.unsqueeze(1)        # (B, 1, n_bins)
        x = self.net(hist)
        x = self.pool(x).flatten(1)
        return self.head(x)


# ----------------------------------------------------------------------------
# Flow masks
# ----------------------------------------------------------------------------

def _alternating_halves_masks(theta_dim, n_steps):
    assert theta_dim % 2 == 0, f"theta_dim must be even; got {theta_dim}"
    half = theta_dim // 2
    masks = []
    for i in range(n_steps):
        m = torch.zeros(theta_dim, dtype=torch.bool)
        if i % 2 == 0:
            m[:half] = True
        else:
            m[half:] = True
        masks.append(m)
    return masks


# ----------------------------------------------------------------------------
# NPE model
# ----------------------------------------------------------------------------

class NPEDiffraction2D(nn.Module):
    """Conditional NSF for canonical (u, v) given a 2D tiling.

    Up to three parallel conditioning paths:
      1. Diffraction CNN          (always on)
      2. Edge-histogram CNN       (if use_edge_hist=True)
      3. log_mean_nn scalar       (if use_log_mean_nn=True)
    """

    def __init__(
        self, *,
        diff_image_module: nn.Module,
        # Diffraction conditioner
        cnn_width: int = 32,
        n_res_per_stage: int = 2,
        n_head_layers: int = 2,
        dropout: float = 0.1,
        context_dim: int = 128,
        # Edge-histogram conditioner
        use_edge_hist: bool = True,
        hist_n_bins: int = 64,
        hist_feature_dim: int = 64,
        hist_cnn_width: int = 32,
        # Scalar scale path
        use_log_mean_nn: bool = True,
        # Output
        theta_dim: int = 10,
        K: int = 8,
        B: int = 3,
        n_flow_steps: int = 8,
        hidden_dim_conditioner: int = 128,
        num_conditioner_blocks: int = 2,
    ):
        super().__init__()
        assert theta_dim % 2 == 0, f"theta_dim must be even; got {theta_dim}"

        self.theta_dim = theta_dim
        self.context_dim = context_dim
        self.use_edge_hist = use_edge_hist
        self.use_log_mean_nn = use_log_mean_nn

        self.diff_image_module = diff_image_module
        self.conditioner = DiffractionConditioner(
            in_channels=1, cnn_width=cnn_width, d_out=context_dim,
            n_res_per_stage=n_res_per_stage, n_head_layers=n_head_layers,
            dropout=dropout,
        )

        if use_edge_hist:
            self.hist_conditioner = EdgeHistogramConditioner(
                n_bins=hist_n_bins, base_width=hist_cnn_width,
                d_out=hist_feature_dim, dropout=dropout,
            )
            hist_dim = hist_feature_dim
        else:
            self.hist_conditioner = None
            hist_dim = 0

        scale_dim = 1 if use_log_mean_nn else 0
        fused_in_dim = context_dim + hist_dim + scale_dim

        if fused_in_dim != context_dim:
            self.fuse_proj = nn.Sequential(
                nn.Linear(fused_in_dim, context_dim),
                nn.LayerNorm(context_dim),
            )
        else:
            self.fuse_proj = nn.Identity()

        # Flow (imports deferred so models/ is on sys.path first)
        from models.nsf import ResNetConditioner
        from models.nsf import TransformRQS
        from models.nsf import Flow

        transforms = []
        for mask in _alternating_halves_masks(theta_dim, n_flow_steps):
            transforms.append(TransformRQS(
                dAB=theta_dim,
                K=K, B=B,
                dC=context_dim,
                mask=mask,
                conditioning_NN=ResNetConditioner,
                hidden_dim_conditioner=hidden_dim_conditioner,
                min_height=1e-3, min_derivative=1e-3, min_width=1e-3,
                num_blocks=num_conditioner_blocks,
            ))
        self.flow = Flow(transforms=transforms, dim=theta_dim,
                         base="standard_normal")

    # ----- context construction -----------------------------------------

    def _context(self, points, mask, edge_hist=None, log_mean_nn=None):
        img = self.diff_image_module(points, mask)
        parts = [self.conditioner(img)]                  # (B, context_dim)

        if self.use_edge_hist:
            if edge_hist is None:
                raise ValueError(
                    "edge_hist required when use_edge_hist=True.")
            parts.append(self.hist_conditioner(edge_hist))   # (B, hist_d)

        if self.use_log_mean_nn:
            if log_mean_nn is None:
                raise ValueError(
                    "log_mean_nn required when use_log_mean_nn=True.")
            if log_mean_nn.dim() == 1:
                log_mean_nn = log_mean_nn.unsqueeze(-1)
            parts.append(log_mean_nn)                        # (B, 1)

        fused = torch.cat(parts, dim=-1)
        return self.fuse_proj(fused)

    # ----- training / inference -----------------------------------------

    def compute_loss(self, points, mask, theta,
                     edge_hist=None, log_mean_nn=None):
        ctx = self._context(points, mask, edge_hist=edge_hist,
                            log_mean_nn=log_mean_nn)
        z, log_det = self.flow.inverse(theta, ctx)
        log_prob_z = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(-1)
        return -(log_prob_z + log_det).mean()

    @torch.no_grad()
    def sample(self, points, mask, n_samples,
               edge_hist=None, log_mean_nn=None):
        if points.size(0) != 1:
            raise ValueError(
                f"sample() expects batch size 1, got {points.size(0)}.")
        ctx = self._context(points, mask, edge_hist=edge_hist,
                            log_mean_nn=log_mean_nn)
        ctx = ctx.expand(n_samples, -1)
        z = self.flow.distribution.sample((n_samples,)).to(ctx.device)
        theta, _ = self.flow.forward(z, ctx)
        return theta

    def log_prob(self, points, mask, theta,
                 edge_hist=None, log_mean_nn=None):
        ctx = self._context(points, mask, edge_hist=edge_hist,
                            log_mean_nn=log_mean_nn)
        z, log_det = self.flow.inverse(theta, ctx)
        log_prob_z = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(-1)
        return log_prob_z + log_det