"""
npe_diffraction_dataset.py
==========================

Dataset for d=5 cut-and-project NPE training.

Augmentation
------------
Two modes of point-cloud corruption are supported. Training uses *random*
rates drawn per sample; evaluation uses *exact* rates for deterministic
noise sweeps.

Corruption operations (applied to the cropped, centred point cloud
*before* mean-NN rescaling):
  drop   : remove a fraction p_drop of points
  insert : add  a fraction p_insert of spurious points drawn uniformly
           from the bounding box of the cropped cloud

Rates are relative to the *pre-corruption* point count; so a sample
starting with N points and corrupted with (p_drop=0.1, p_insert=0.1)
ends up with roughly N points but ~10% replaced with noise.

Mode selection
--------------
1. Exact (eval): pass exact_drop_rate and/or exact_insert_rate (float).
2. Random (train): set drop_rate_max > 0 and/or insert_rate_max > 0.
3. Off (default): all four defaults are 0.0, no corruption.

Returns per item
----------------
    points       : (n, 2) float32  -- cropped, optionally corrupted,
                                       rescaled to mean_NN = 1
    edge_hist    : (n_bins,) float32 -- Delaunay edge-length histogram
                                          on rescaled points
    log_mean_nn  : float32 scalar    -- log mean nearest-neighbour distance
                                          BEFORE the rescaling
    theta        : (2d,) float32     -- canonical [u | v] concatenation
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import Delaunay, cKDTree
from torch.utils.data import Dataset


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _circular_crop(points, n_target, circle_frac=0.95):
    if len(points) < 3:
        return points
    center_pt = points.mean(axis=0)
    extent = points.max(axis=0) - points.min(axis=0)
    L = float(extent.min())
    r_max = circle_frac * L / 2.0
    density = len(points) / max(float(np.prod(extent)), 1e-12)
    r_target = (np.sqrt(n_target / (np.pi * density))
                if density > 0 else r_max)
    r = min(r_target, r_max)
    d2 = ((points - center_pt) ** 2).sum(axis=1)
    return points[d2 <= r * r]


def _extract_uv(entry, dimension):
    if "u" in entry and "v" in entry:
        u = np.asarray(entry["u"], dtype=np.float64).copy()
        v = np.asarray(entry["v"], dtype=np.float64).copy()
    elif "components" in entry:
        comps = np.asarray(entry["components"], dtype=np.float64).ravel()
        if len(comps) != 2 * dimension:
            raise ValueError(
                f"components has length {len(comps)}, expected "
                f"2 * dimension = {2 * dimension}")
        u = comps[:dimension].copy()
        v = comps[dimension:].copy()
    else:
        raise KeyError(
            "Tiling entry has neither ('u', 'v') keys nor a 'components' "
            f"key. Available keys: {sorted(entry.keys())}")
    return u, v


def canonicalize_basis(u, v):
    u = np.asarray(u, dtype=np.float64).copy()
    v = np.asarray(v, dtype=np.float64).copy()
    needs_flip = (v < 0) | ((v == 0) & (u < 0))
    u[needs_flip] *= -1.0
    v[needs_flip] *= -1.0
    angles = np.arctan2(v, u)
    order = np.argsort(angles, kind="stable")
    return u[order], v[order]


def compute_edge_length_histogram(points, n_bins=64,
                                  hist_min=0.0, hist_max=5.0):
    if len(points) < 4:
        return np.zeros(n_bins, dtype=np.float32)
    try:
        tri = Delaunay(np.asarray(points, dtype=np.float64))
    except Exception:
        return np.zeros(n_bins, dtype=np.float32)
    edges = set()
    for s in tri.simplices:
        for i in range(3):
            a, b = int(s[i]), int(s[(i + 1) % 3])
            edges.add((min(a, b), max(a, b)))
    if not edges:
        return np.zeros(n_bins, dtype=np.float32)
    edge_arr = np.array(list(edges))
    diffs = points[edge_arr[:, 0]] - points[edge_arr[:, 1]]
    lengths = np.sqrt((diffs ** 2).sum(axis=1))
    hist, _ = np.histogram(lengths, bins=n_bins,
                           range=(hist_min, hist_max))
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist


def _apply_corruption(points, p_drop, p_insert, rng):
    """Drop fraction p_drop, insert fraction p_insert (rel. to pre-corruption N).

    Spurious points are sampled uniformly from the axis-aligned bounding box of
    the input cloud (computed before drop). At least 4 points are retained so
    Delaunay can still be computed.
    """
    if len(points) == 0 or (p_drop <= 0.0 and p_insert <= 0.0):
        return points

    n_orig = len(points)

    # Bounding box used for insertion: pre-drop, so it reflects the
    # original cropped extent.
    if p_insert > 0.0:
        bb_min = points.min(axis=0)
        bb_max = points.max(axis=0)
    else:
        bb_min = bb_max = None

    # Drop
    if p_drop > 0.0:
        n_drop = int(round(p_drop * n_orig))
        # Keep at least 4 points so Delaunay can triangulate
        n_drop = min(n_drop, n_orig - 4)
        if n_drop > 0:
            keep_idx = rng.choice(n_orig, size=n_orig - n_drop,
                                  replace=False)
            points = points[keep_idx]

    # Insert
    if p_insert > 0.0:
        n_insert = int(round(p_insert * n_orig))
        if n_insert > 0 and bb_min is not None:
            spurious = rng.uniform(bb_min, bb_max,
                                   size=(n_insert, 2)).astype(points.dtype)
            points = np.concatenate([points, spurious], axis=0)
            # Shuffle so spurious points are not all at the tail
            shuf = rng.permutation(len(points))
            points = points[shuf]

    return points


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class NPEDiffraction2D(Dataset):
    """Tilings + canonical (u, v) labels + edge-length histogram, with
    optional drop/insert augmentation."""

    def __init__(
        self,
        tilings,
        *,
        expected_dim: int = 5,
        n_min: int = 1024,
        n_max: int = 2048,
        circle_frac: float = 0.95,
        normalize: bool = True,
        canonicalize: bool = True,
        compute_hist: bool = True,
        hist_n_bins: int = 64,
        hist_min: float = 0.0,
        hist_max: float = 5.0,
        drop_rate_min: float = 0.0,
        drop_rate_max: float = 0.0,
        insert_rate_min: float = 0.0,
        insert_rate_max: float = 0.0,
        clean_frac: float = 0.0,                   
        exact_drop_rate: float | None = None,
        exact_insert_rate: float | None = None,
        seed: int | None = None,
    ):
        
        self.tilings = [t for t in tilings
                        if int(t["dimension"]) == int(expected_dim)]
        if len(self.tilings) == 0:
            raise ValueError(
                f"No tilings with dimension={expected_dim} found "
                f"(out of {len(tilings)} total).")
        _ = _extract_uv(self.tilings[0], expected_dim)
        self.clean_frac = float(clean_frac)
        self.expected_dim = expected_dim
        self.n_min = n_min
        self.n_max = n_max
        self.circle_frac = circle_frac
        self.normalize = normalize
        self.canonicalize = canonicalize
        self.compute_hist = compute_hist
        self.hist_n_bins = hist_n_bins
        self.hist_min = hist_min
        self.hist_max = hist_max

        self.drop_rate_min = float(drop_rate_min)
        self.drop_rate_max = float(drop_rate_max)
        self.insert_rate_min = float(insert_rate_min)
        self.insert_rate_max = float(insert_rate_max)
        self.exact_drop_rate = (None if exact_drop_rate is None
                                else float(exact_drop_rate))
        self.exact_insert_rate = (None if exact_insert_rate is None
                                  else float(exact_insert_rate))

        if self.drop_rate_max < self.drop_rate_min:
            raise ValueError("drop_rate_max must be >= drop_rate_min")
        if self.insert_rate_max < self.insert_rate_min:
            raise ValueError("insert_rate_max must be >= insert_rate_min")

        self._base_seed = seed

    def __len__(self):
        return len(self.tilings)

    def _select_rates(self, rng):
        """Return (p_drop, p_insert) for this sample."""
        if (self.exact_drop_rate is not None
                or self.exact_insert_rate is not None):
            p_drop = float(self.exact_drop_rate or 0.0)
            p_insert = float(self.exact_insert_rate or 0.0)
        elif self.drop_rate_max > 0.0 or self.insert_rate_max > 0.0:
            p_drop = float(rng.uniform(self.drop_rate_min,
                                       self.drop_rate_max))
            p_insert = float(rng.uniform(self.insert_rate_min,
                                         self.insert_rate_max))
        else:
            p_drop = 0.0
            p_insert = 0.0
        return p_drop, p_insert

    def __getitem__(self, idx):
        entry = self.tilings[idx]
        points = np.asarray(entry["points"], dtype=np.float32).copy()
        u, v = _extract_uv(entry, self.expected_dim)
        if len(u) != self.expected_dim or len(v) != self.expected_dim:
            raise ValueError(
                f"Item {idx}: u/v have wrong length for dim "
                f"{self.expected_dim}: u={len(u)}, v={len(v)}")

        rng = (np.random.default_rng(self._base_seed + idx)
               if self._base_seed is not None
               else np.random.default_rng())

        n_target = int(rng.integers(self.n_min, self.n_max + 1))
        points = _circular_crop(points, n_target, self.circle_frac)
        if len(points) > 0:
            points = points - points.mean(axis=0)

        using_exact = (self.exact_drop_rate is not None
               or self.exact_insert_rate is not None)

        if using_exact:
           
            p_drop, p_insert = self._select_rates(rng)
            if p_drop > 0.0 or p_insert > 0.0:
                points = _apply_corruption(points, p_drop, p_insert, rng)
        else:
            
            if rng.random() >= self.clean_frac:
                p_drop, p_insert = self._select_rates(rng)
                if p_drop > 0.0 or p_insert > 0.0:
                    points = _apply_corruption(points, p_drop, p_insert, rng)

        if len(points) >= 2:
            tree = cKDTree(points)
            dists, _ = tree.query(points, k=2)
            mean_nn = float(dists[:, 1].mean())
        else:
            mean_nn = 1.0
        log_mean_nn = float(np.log(max(mean_nn, 1e-8)))

        # Rescale
        if self.normalize and mean_nn > 1e-8:
            points = points / mean_nn

        # Edge histogram on rescaled points
        if self.compute_hist:
            edge_hist = compute_edge_length_histogram(
                points, n_bins=self.hist_n_bins,
                hist_min=self.hist_min, hist_max=self.hist_max)
        else:
            edge_hist = np.zeros(self.hist_n_bins, dtype=np.float32)

        # Canonical labels
        if self.canonicalize:
            u, v = canonicalize_basis(u, v)
        theta = np.concatenate([u, v]).astype(np.float32)

        return (points.astype(np.float32), edge_hist, log_mean_nn, theta)


def collate_npe_2d(batch):
    points_list, hist_list, log_mean_nns, thetas = zip(*batch)
    max_n = max(len(p) for p in points_list)
    B = len(points_list)
    points = torch.zeros(B, max_n, 2, dtype=torch.float32)
    mask = torch.zeros(B, max_n, dtype=torch.float32)
    for i, pts in enumerate(points_list):
        n = len(pts)
        if n > 0:
            points[i, :n] = torch.as_tensor(pts, dtype=torch.float32)
            mask[i, :n] = 1.0
    edge_hist = torch.stack(
        [torch.as_tensor(h, dtype=torch.float32) for h in hist_list])
    log_mean_nn = torch.tensor(log_mean_nns, dtype=torch.float32)
    theta = torch.stack(
        [torch.as_tensor(t, dtype=torch.float32) for t in thetas])
    return points, mask, edge_hist, log_mean_nn, theta