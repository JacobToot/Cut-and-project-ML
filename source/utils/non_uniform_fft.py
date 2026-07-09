"""
nufft2d.py
==========

On-the-fly 2D diffraction patterns (structure factors) for point sets.

For a set of points r_j (a tiling, possibly augmented / cropped), the
kinematic diffraction intensity sampled on a uniform reciprocal grid is

        S(q) = | sum_j  w_j * exp(-i q . r_j) |^2 ,

with w_j a per-point weight (here a 0/1 padding mask). Evaluating this on a
uniform grid of q from non-uniform r_j is exactly a *type-1* NUFFT
(non-uniform sources -> uniform frequency grid).

This module provides TWO interchangeable backends behind one interface:

  * "direct"  -- exact, batched, chunked direct DFT. O(N * M) but trivially
                 correct, dependency-free, GPU-native. Recommended at the
                 scales in this project (N ~ 1e3 points, M ~ 1e4-1e5 pixels).
                 Also serves as the ground-truth oracle for the NUFFT.

  * "nufft"   -- type-1 NUFFT via Kaiser-Bessel gridding + oversampled FFT +
                 analytic de-apodization. O(N + M log M). Use when N or M grow
                 large enough that the direct method is the bottleneck.

Neither backend is differentiated through (points are data, not parameters),
so both are plain tensor ops with no autograd requirement.

The diffraction *intensity* is invariant to a global phase and to any
per-frequency phase of unit modulus, which is why the NUFFT does not need the
usual (-1)^k half-grid-shift correction: it cancels under |.|^2. Only the
real-valued de-apodization (kernel Fourier transform) affects the magnitude
and is applied.

Author: (generated scaffold — validate q_max / resolution on your own data)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

TWO_PI = 2.0 * math.pi



def _kb_beta(width: int, sigma: float) -> float:
    """Optimal KB shape parameter for kernel support `width` (grid points)
    and oversampling factor `sigma` = G / N_grid."""
    # Beatty 2005, eq. 5
    val = (width / sigma) ** 2 * (sigma - 0.5) ** 2 - 0.8
    val = max(val, 1e-8)
    return math.pi * math.sqrt(val)


def _kb_kernel(dist: torch.Tensor, width: int, beta: float) -> torch.Tensor:
    """KB kernel evaluated at distances `dist` (in oversampled-grid units).
    Support is |dist| <= width/2; zero outside. Returns same shape as `dist`."""
    half = width / 2.0
    z = 1.0 - (dist / half) ** 2
    z = z.clamp(min=0.0)                       
    val = torch.special.i0(beta * torch.sqrt(z))
    val = torch.where(dist.abs() <= half, val, torch.zeros_like(val))
    return val


def _kb_apodization(n_out: int, grid_os: int, width: int, beta: float,
                    device, dtype) -> torch.Tensor:
    """Exact 1D de-apodization factor: the *discrete* DFT of the gridding
    kernel sampled on the oversampled grid, evaluated at output frequencies
    m in [-n_out/2, n_out/2). This is the exact inverse of the gridding step
    (Poisson summation), so it carries the correct overall scale -- unlike the
    continuous analytic KB Fourier transform, which is shape-correct but off
    by a kernel-dependent constant. Returns shape (n_out,)."""
    p = torch.arange(grid_os, device=device, dtype=dtype) - grid_os // 2
    ker = _kb_kernel(p, width, beta)                         
    apod = torch.fft.fftshift(
        torch.fft.fft(torch.fft.ifftshift(ker))).real       
    lo = grid_os // 2 - n_out // 2
    return apod[lo:lo + n_out]


@torch.no_grad()
def structure_factor_direct(
    points: torch.Tensor,      # (B, N, 2) 
    strengths: torch.Tensor,   # (B, N)     
    q_grid: torch.Tensor,      # (M, 2)     
    q_chunk: int = 4096,
) -> torch.Tensor:
    """Exact F(q) = sum_j strengths_j exp(-i q . r_j) on arbitrary q points.

    Returns complex tensor (B, M). Chunked over q to bound memory; the only
    large intermediate is (B, N, q_chunk)."""
    B, N, _ = points.shape
    M = q_grid.shape[0]
    out = torch.empty(B, M, dtype=torch.complex64, device=points.device)
    s = strengths.to(points.dtype)             # (B, N)
    for start in range(0, M, q_chunk):
        qc = q_grid[start:start + start_step(start, q_chunk, M)]  # (Mc, 2)
        # phase[b, j, m] = q_m . r_bj
        phase = torch.einsum("bnd,md->bnm", points, qc)           # (B, N, Mc)
        e = torch.polar(s.unsqueeze(-1).expand_as(phase),
                        -phase)                                   # complex
        out[:, start:start + qc.shape[0]] = e.sum(dim=1)
    return out


def start_step(start: int, q_chunk: int, M: int) -> int:
    return min(q_chunk, M - start)


@torch.no_grad()
def structure_factor_nufft(
    points_scaled: torch.Tensor,   # (B, N, 2) 
    strengths: torch.Tensor,       # (B, N)
    n_grid: int,                   
    sigma: float = 2.0,
    width: int = 6,
) -> torch.Tensor:
    """Type-1 NUFFT: returns complex F on an (n_grid x n_grid) frequency grid,
    centred at zero frequency. `points_scaled` MUST already be wrapped into
    [-pi, pi) per dimension (see DiffractionImager, which does the q-scaling
    and wrapping). Frequencies correspond to integer k in [-n_grid/2, n_grid/2)."""
    B, N, _ = points_scaled.shape
    device = points_scaled.device
    dtype = points_scaled.dtype

    G = int(math.ceil(sigma * n_grid))
    G += G % 2                                 
    beta = _kb_beta(width, G / n_grid)

    u = (points_scaled + math.pi) / TWO_PI * G  # (B, N, 2)

    offsets = torch.arange(-(width // 2) + 1, width // 2 + 1,
                           device=device, dtype=torch.long)        # (W,)
    base = torch.floor(u).to(torch.long)                           # (B, N, 2)
    # neighbour indices per dim: (B, N, 2, W)
    idx = base.unsqueeze(-1) + offsets.view(1, 1, 1, width)
    dist = u.unsqueeze(-1) - idx.to(dtype)                         # (B, N, 2, W)
    w = _kb_kernel(dist, width, beta)                              # (B, N, 2, W)
    idx = idx % G                                                  # periodic wrap

    wx, wy = w[:, :, 0], w[:, :, 1]            # (B, N, W) each
    ix, iy = idx[:, :, 0], idx[:, :, 1]        # (B, N, W) each

    w2 = (wx.unsqueeze(-1) * wy.unsqueeze(-2))                     # (B, N, W, W)
    w2 = w2 * strengths.to(dtype).view(B, N, 1, 1)
    flat = (ix.unsqueeze(-1) * G + iy.unsqueeze(-2))               # (B, N, W, W)

    grid = torch.zeros(B, G * G, dtype=dtype, device=device)
    grid.scatter_add_(1, flat.reshape(B, -1), w2.reshape(B, -1))
    grid = grid.reshape(B, G, G)

    F = torch.fft.fftshift(torch.fft.fft2(grid), dim=(-2, -1))     # (B, G, G)

    # crop central n_grid x n_grid block (low frequencies around DC)
    lo = G // 2 - n_grid // 2
    F = F[:, lo:lo + n_grid, lo:lo + n_grid]

    apod = _kb_apodization(n_grid, G, width, beta, device, dtype)  # (n_grid,)
    deapod = (apod.view(-1, 1) * apod.view(1, -1)).clamp(min=1e-8)
    F = F / deapod
    return F


@dataclass
class DiffractionConfig:
    grid_size: int = 128          # output image is grid_size x grid_size
    q_max: float = 4.0 * math.pi  # half-extent of q-grid (physical units)
    backend: str = "direct"       # "direct" | "nufft"
    normalize: str = "per_atom"   # "none" | "per_atom" (divide |F|^2 by N)
    log1p: bool = True            # apply log1p to intensity
    standardize: bool = True      # per-sample zero-mean unit-std after log
    suppress_dc: bool = True      # zero the forward-scattering peak (|q|~0)
    dc_radius: int = 1            # radius (pixels) of the DC region to zero
    # nufft-only knobs
    nufft_sigma: float = 2.0
    nufft_width: int = 6


class DiffractionImager(nn.Module):
    """points (B, N, 2), mask (B, N) -> intensity image (B, 1, H, W).

    Handles q-grid construction, the [-pi, pi) wrapping required by the NUFFT
    backend, per-atom normalisation (important: N varies per sample under
    augmentation/subsampling, so raw |F|^2 ~ N^2 would be inconsistent), DC
    suppression, and log/standardise preprocessing for the CNN."""

    def __init__(self, cfg: DiffractionConfig | None = None, **kw):
        super().__init__()
        self.cfg = cfg or DiffractionConfig(**kw)
        g = self.cfg.grid_size
        ax = torch.linspace(-self.cfg.q_max, self.cfg.q_max, g + 1)[:-1]
        qx, qy = torch.meshgrid(ax, ax, indexing="ij")
        self.register_buffer("q_grid", torch.stack([qx, qy], -1).reshape(-1, 2))
        self.dq = 2.0 * self.cfg.q_max / g

    @torch.no_grad()
    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B = points.shape[0]
        g = cfg.grid_size
        device = points.device
        strengths = mask.to(points.dtype)

        if cfg.backend == "direct":
            F = structure_factor_direct(points, strengths,
                                        self.q_grid.to(device))      # (B, M)
            F = F.reshape(B, g, g)
        elif cfg.backend == "nufft":
            x = points * self.dq
            x = torch.remainder(x + math.pi, TWO_PI) - math.pi
            F = structure_factor_nufft(x, strengths, g,
                                       sigma=cfg.nufft_sigma,
                                       width=cfg.nufft_width)         # (B,g,g)
        else:
            raise ValueError(f"unknown backend {cfg.backend!r}")

        intensity = (F.real ** 2 + F.imag ** 2)                      # (B, g, g)

        if cfg.normalize == "per_atom":
            n = strengths.sum(dim=1).clamp(min=1.0).view(B, 1, 1)
            intensity = intensity / n

        if cfg.suppress_dc:
            c = g // 2
            r = cfg.dc_radius
            intensity[:, c - r:c + r + 1, c - r:c + r + 1] = 0.0

        if cfg.log1p:
            intensity = torch.log1p(intensity)

        if cfg.standardize:
            mu = intensity.mean(dim=(-2, -1), keepdim=True)
            sd = intensity.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
            intensity = (intensity - mu) / sd

        return intensity.unsqueeze(1)                                # (B,1,H,W)


def _square_lattice(n_side: int, jitter: float = 0.0, seed: int = 0):
    g = torch.arange(n_side, dtype=torch.float32)
    xx, yy = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], -1)
    pts = pts - pts.mean(0)
    if jitter > 0:
        torch.manual_seed(seed)
        pts = pts + jitter * torch.randn_like(pts)
    return pts


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / (a.norm() + 1e-12)
    b = b / (b.norm() + 1e-12)
    return float((a - b).norm() / (b.norm() + 1e-12))


def _self_test():
    import time
    torch.manual_seed(0)
    g = 96
    qmax = 4.0 * math.pi

    cases = {
        "random_diffuse": torch.rand(1, 400, 2) * 30.0 - 15.0,
        "square_lattice": _square_lattice(18).unsqueeze(0),          
        "jittered_lattice": _square_lattice(18, jitter=0.08).unsqueeze(0),
    }

    direct = DiffractionImager(DiffractionConfig(
        grid_size=g, q_max=qmax, backend="direct",
        log1p=False, standardize=False, suppress_dc=False,
        normalize="none"))
    nufft = DiffractionImager(DiffractionConfig(
        grid_size=g, q_max=qmax, backend="nufft",
        log1p=False, standardize=False, suppress_dc=False,
        normalize="none", nufft_sigma=2.0, nufft_width=6))

    print(f"{'case':18s}  {'rel_err(shape)':>14s}  {'scale_ratio':>11s}")
    for name, pts in cases.items():
        mask = torch.ones(pts.shape[0], pts.shape[1])
        Id = direct(pts, mask)[0, 0]
        In = nufft(pts, mask)[0, 0]
        err = _rel_err(In, Id)
        ratio = float(In.sum() / (Id.sum() + 1e-12))
        flag = "OK " if err < 5e-3 else "!! "
        print(f"{flag}{name:15s}  {err:14.2e}  {ratio:11.4f}")

    print("\n[timing] B=16, N=2048, grid=128")
    pts = torch.rand(16, 2048, 2) * 60 - 30
    mask = torch.ones(16, 2048)
    for backend in ("direct", "nufft"):
        im = DiffractionImager(DiffractionConfig(
            grid_size=128, q_max=qmax, backend=backend))
        im(pts, mask)  
        t0 = time.time()
        for _ in range(3):
            im(pts, mask)
        dt = (time.time() - t0) / 3
        print(f"  {backend:7s}: {dt * 1e3:7.1f} ms / batch")


if __name__ == "__main__":
    _self_test()