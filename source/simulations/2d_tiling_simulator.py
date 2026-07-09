"""multigrid_simulator.py
=======================

2D cut-and-project quasicrystal tiling generator via the de Bruijn
multigrid method. N dim lattice projected onto span(u,v), two vectors from R^N.

The canonicalise option sorts 

Returns a dict rather than a positional tuple; extend safely by adding
keys. Edges and tiles, when requested, are computed on the CLEAN lattice
(before noise). 
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree



def _generate_primes(limit: int) -> list[int]:
    """Sieve of Eratosthenes up to *limit* inclusive."""
    prime = [True] * (limit + 1)
    prime[0] = prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if prime[i]:
            for j in range(i * i, limit + 1, i):
                prime[j] = False
    return [i for i in range(2, limit + 1) if prime[i]]


_PRIMES = _generate_primes(10 ** 6)
_SQRT_PRIMES = np.array([np.sqrt(p) for p in _PRIMES], dtype=np.float64)


# ---------------------------------------------------------------------
# Basis generation
# ---------------------------------------------------------------------

def _canonical_basis(
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    u_k = cos(2*pi*k/d), v_k = sin(2*pi*k/d) for k in 0..d-1.
    Returns (u, v, thetas, radii).
    """
    theta = 2.0 * np.pi * np.arange(dimension, dtype=np.float64) / dimension
    u = np.cos(theta)
    v = np.sin(theta)
    return u, v, theta, np.ones(dimension, dtype=np.float64)


def _random_basis(
    dimension: int,
    alpha_min: float,
    r_min: float,
    r_max: float,
    snap_range: float,
    rng: np.random.Generator,
    max_restarts: int = 1000,
    max_attempts_per_point: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random basis with minimum angular separation and optional snapping.

    Angles theta_k are drawn from U(0, pi) via rejection sampling to
    enforce |theta_i - theta_j| >= alpha_min for all i != j; if placement
    fails the whole set is restarted (up to ``max_restarts`` times).
    Radii r_k ~ U(r_min, r_max) independently.

    If snap_range > 0, each Cartesian component of the resulting basis is
    snapped to the nearest value of (sqrt(prime) mod snap_range) from the
    module-level prime table, preserving sign. Set snap_range=0 to
    disable snapping.
    """
    for _ in range(max_restarts):
        angles: list[float] = []
        placed_all = True
        for _ in range(dimension):
            placed = False
            for _ in range(max_attempts_per_point):
                theta = float(rng.uniform(0.0, math.pi))
                if all(abs(theta - a) >= alpha_min for a in angles):
                    angles.append(theta)
                    placed = True
                    break
            if not placed:
                placed_all = False
                break
        if placed_all:
            thetas = np.sort(np.asarray(angles, dtype=np.float64))
            break
    else:
        raise RuntimeError(
            f"Could not place {dimension} angles with "
            f"alpha_min={alpha_min} after {max_restarts} restarts."
        )

    radii = rng.uniform(r_min, r_max, size=dimension)
    u = radii * np.cos(thetas)
    v = radii * np.sin(thetas)

    if snap_range > 0.0:
        mapped = _SQRT_PRIMES % snap_range

        def _snap(val: float) -> float:
            sign = -1.0 if val < 0 else 1.0
            idx = int(np.argmin(np.abs(mapped - abs(val))))
            return sign * float(mapped[idx])

        u = np.array([_snap(x) for x in u], dtype=np.float64)
        v = np.array([_snap(x) for x in v], dtype=np.float64)

    return u, v, thetas, radii


def canonicalize_basis(
    u: np.ndarray, v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Put (u, v) into a canonical representative form.

    Removes the two discrete symmetries of the basis label (total equivalent representations 2^N(N!)):
      1. Sign flip per column so v_k >= 0 (break tie v_k=0 on sign of u_k)
      2. Permute columns by ascending order in atan2(v_k, u_k)

    After this, any two bases that describe the same quasicrystal up to
    column sign flips and column permutations map to identical (u, v).
    """
    u = np.asarray(u, dtype=np.float64).copy()
    v = np.asarray(v, dtype=np.float64).copy()
    needs_flip = (v < 0) | ((v == 0) & (u < 0))
    u[needs_flip] *= -1.0
    v[needs_flip] *= -1.0
    order = np.argsort(np.arctan2(v, u), kind="stable")
    return u[order], v[order]



def _cut_and_project(
    dimension: int,
    u: np.ndarray,
    v: np.ndarray,
    physical_extent: float,
    track_tiles: bool = False,
):
    """Multigrid projection: (u, v) in R^d -> 2D tiling on |x|,|y| <= L.

    For each pair (i, j) with i < j of parent-lattice directions, iterate
    over intersections of the (i)-family and (j)-family grid lines in
    physical space. Each intersection corresponds to a rhombic tile whose
    four Z^d-corners (base + shifts of 0 or 1 in the i,j slots) are
    candidate lattice points; corners whose physical projection falls
    inside the window are accepted.

    Returns
    -------
    accepted : dict[tuple[int, ...], np.ndarray]
        Z^d integer key -> 2D physical position (float64, shape (2,)).
    edge_set : set[tuple[tuple, tuple]]
        Unordered accepted-key pairs joined by a rhombus edge.
    tiles : list[tuple[int, np.ndarray]] or None
        (face_index, (4, 2) polygon) per rhombic tile fully inside the
        window, ordered as one traverses the rhombus. None if
        track_tiles=False.
    face_labels : list[tuple[int, int]] or None
        (i, j) label for each face_index (position in the C(d, 2) list of
        non-degenerate face types encountered), or None.
    """
    A = np.column_stack([u, v])                # (d, 2)
    M = np.linalg.inv(A.T @ A) @ A.T           # (2, d) projection

    L = physical_extent
    pad = L + 2                                # small overshoot buffer

    accepted: dict[tuple, np.ndarray] = {}
    edge_set: set = set()
    tiles: list | None = [] if track_tiles else None
    face_labels: list | None = [] if track_tiles else None

    face_idx = 0
    for i in range(dimension):
        for j in range(i + 1, dimension):
            det = u[i] * v[j] - u[j] * v[i]
            if abs(det) < 1e-12:
                # basis directions i, j are parallel -> no tiles from
                # this face; skip.
                if track_tiles:
                    face_idx += 1
                continue
            if track_tiles:
                face_labels.append((i, j))
            inv_det = 1.0 / det

            max_zi = pad * (abs(u[i]) + abs(v[i]))
            max_zj = pad * (abs(u[j]) + abs(v[j]))
            mi_vals = np.arange(
                int(np.floor(-max_zi - 0.5)),
                int(np.ceil(max_zi - 0.5)) + 1,
            )
            mj_vals = np.arange(
                int(np.floor(-max_zj - 0.5)),
                int(np.ceil(max_zj - 0.5)) + 1,
            )
            MI, MJ = np.meshgrid(mi_vals, mj_vals)
            mi_flat = MI.ravel()
            mj_flat = MJ.ravel()

            # Cramer's rule: intersect the (mi + 0.5)-th grid line of
            # direction i with the (mj + 0.5)-th grid line of direction j.
            c1 = ((mi_flat + 0.5) * v[j] - (mj_flat + 0.5) * v[i]) * inv_det
            c2 = (u[i] * (mj_flat + 0.5) - u[j] * (mi_flat + 0.5)) * inv_det
            mask = (np.abs(c1) <= pad) & (np.abs(c2) <= pad)
            c1, c2 = c1[mask], c2[mask]
            mi_flat, mj_flat = mi_flat[mask], mj_flat[mask]

            if len(c1) == 0:
                if track_tiles:
                    face_idx += 1
                continue

            z_all = c1[:, None] * u[None, :] + c2[:, None] * v[None, :]
            base_all = np.round(z_all).astype(int)

            for g in range(len(c1)):
                base = base_all[g]
                mi, mj = int(mi_flat[g]), int(mj_flat[g])

                corners: list = []
                for di in (0, 1):
                    for dj in (0, 1):
                        x = base.copy()
                        x[i] = mi + di
                        x[j] = mj + dj
                        key = tuple(x)
                        if key not in accepted:
                            c_par = M @ x.astype(np.float64)
                            if (abs(c_par[0]) <= pad
                                    and abs(c_par[1]) <= pad):
                                accepted[key] = c_par
                        corners.append(key)

                # 4 rhombus edges: (0,1)-(0,0), (1,0)-(0,0),
                # (1,1)-(0,1), (1,1)-(1,0)
                for a_idx, b_idx in ((0, 1), (0, 2), (1, 3), (2, 3)):
                    ka, kb = corners[a_idx], corners[b_idx]
                    if ka in accepted and kb in accepted:
                        edge_set.add((min(ka, kb), max(ka, kb)))

                if track_tiles and all(c in accepted for c in corners):
                    # corners traversal order: (0,0) -> (0,1) -> (1,1) -> (1,0)
                    poly = np.array([
                        accepted[corners[0]],
                        accepted[corners[1]],
                        accepted[corners[3]],
                        accepted[corners[2]],
                    ])
                    if np.all(np.abs(poly) <= L):
                        tiles.append((face_idx, poly))

            if track_tiles:
                face_idx += 1

    return accepted, edge_set, tiles, face_labels


def _apply_dropout(
    points: np.ndarray, frac: float, rng: np.random.Generator,
) -> np.ndarray:
    if frac <= 0.0 or len(points) == 0:
        return points
    keep = max(1, int(round(len(points) * (1.0 - frac))))
    idx = rng.choice(len(points), size=keep, replace=False)
    return points[np.sort(idx)]


def _apply_insertion(
    points: np.ndarray,
    frac: float,
    n_reference: int,
    bbox: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    if frac <= 0.0 or n_reference == 0:
        return points, 0.0
    n_insert = max(1, int(round(n_reference * frac)))
    lo, hi = bbox
    area = float(np.prod(np.maximum(hi - lo, 1e-12)))
    rate = n_insert / area
    spurious = rng.uniform(lo, hi, size=(n_insert, 2))
    return np.concatenate([points, spurious], axis=0), rate


def multigrid_tiling(
    dimension: int,
    *,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
    canonical: bool = False,
    alpha_min: float = 0.0,
    r_min: float = 1.0,
    r_max: float = 1.0,
    snap_range: float = 0.0,
    canonicalize: bool = False,
    physical_extent: float = 15.0,
    dropout: float = 0.0,
    insertion: float = 0.0,
    normalize: bool = False,
    seed: int | None = None,
    return_edges: bool = False,
    return_tiles: bool = False,
) -> dict[str, Any]:
    """Generate one 2D cut-and-project tiling.

    Parameters
    ----------
    dimension : int
        Parent-lattice dimension d. Must be >= 4 (need >= 2 irrational
        components per basis vector for a non-trivial 2D tiling).
    u, v : (d,) arrays, optional
        Explicit basis vectors. If both provided, other basis-selection
        arguments (canonical, alpha_min, r_min, r_max, snap_range) are
        ignored.
    canonical : bool
        If True (and u, v not given), use the N-fold symmetric basis
        u_k = cos(2*pi*k/d), v_k = sin(2*pi*k/d).
    alpha_min : float
        Minimum pairwise angular separation for random angles (radians).
        Default 0 (no constraint).
    r_min, r_max : float
        Uniform range for random radii. Default both 1.0 (unit-radius
        basis with random angles). Set r_min=0, r_max=1 for maximally
        random radii in the unit disc.
    snap_range : float
        If > 0, snap each Cartesian component of a randomly drawn basis
        to the nearest value of (sqrt(prime) mod snap_range), preserving
        sign. Default 0 (no snapping).
    canonicalize : bool
        If True, apply ``canonicalize_basis`` to (u, v) BEFORE projection,
        so the returned (u, v) is the canonical representative. Off by
        default; the dataset layer may prefer to do this itself.
    physical_extent : float
        Half-width of the square viewing window in physical space. Points
        with |x| > physical_extent or |y| > physical_extent are discarded.
    dropout : float in [0, 1)
        Fraction of vertices randomly removed.
    insertion : float >= 0
        Fraction (relative to the PRE-noise vertex count) of spurious
        vertices added uniformly inside the pre-noise bounding box.
    normalize : bool
        If True, rescale points so their mean nearest-neighbour distance
        is 1. Edges and tiles (when returned) are rescaled by the same
        factor. Rescale is by the POST-noise mean-NN so it matches what
        the model sees at inference.
    seed : int, optional
        RNG seed for reproducibility.
    return_edges : bool
        Include a per-edge array of 2D endpoints (E, 2, 2) computed on
        the clean lattice.
    return_tiles : bool
        Include per-tile face indices and 4x2 rhombus polygons, plus
        (i, j) face labels. Also computed on the clean lattice.

    Returns
    -------
    dict with keys (always present):
        points          : (N, 2) float64
        u, v            : (d,) float64        (after canonicalise)
        dimension       : int
        physical_extent : float
        n_points_clean  : int                 (vertex count before noise)
        insertion_rate  : float               (0 when insertion == 0)
        mean_nn         : float or None       (mean-NN of POST-noise
                                                cloud, BEFORE any
                                                normalisation; None when
                                                fewer than 2 points)

    Extra keys when the basis was auto-generated (canonical or random):
        thetas, radii   : (d,) float64        (design parameters)

    Extra keys when return_edges=True:
        edges           : (E, 2, 2) float64   (empty array if no edges)

    Extra keys when return_tiles=True:
        tiles           : list of (face_idx, (4, 2) polygon)
        face_labels     : list of (i, j)
    """
    if not (0.0 <= dropout < 1.0):
        raise ValueError(f"dropout must be in [0, 1). Got {dropout}.")
    if insertion < 0.0:
        raise ValueError(f"insertion must be >= 0. Got {insertion}.")
    if dimension < 4:
        raise ValueError(
            f"dimension must be >= 4 for a non-trivial 2D "
            f"cut-and-project. Got {dimension}."
        )
    if (u is None) != (v is None):
        raise ValueError("Provide both u and v, or neither.")

    rng = np.random.default_rng(seed=seed)

    # -- basis selection ------------------------------------------------
    thetas: np.ndarray | None = None
    radii: np.ndarray | None = None
    if u is not None:
        u = np.asarray(u, dtype=np.float64).copy()
        v = np.asarray(v, dtype=np.float64).copy()
        if len(u) != dimension or len(v) != dimension:
            raise ValueError(
                f"u/v have lengths {len(u)}/{len(v)} but "
                f"dimension={dimension}."
            )
    elif canonical:
        u, v, thetas, radii = _canonical_basis(dimension)
    else:
        u, v, thetas, radii = _random_basis(
            dimension, alpha_min, r_min, r_max, snap_range, rng,
        )

    if canonicalize:
        u, v = canonicalize_basis(u, v)

    # -- projection -----------------------------------------------------
    accepted, edge_set, tiles, face_labels = _cut_and_project(
        dimension, u, v, physical_extent, track_tiles=return_tiles,
    )

    L = physical_extent
    clean_keys = [k for k, p in accepted.items()
                  if abs(p[0]) <= L and abs(p[1]) <= L]
    clean_points = (np.array([accepted[k] for k in clean_keys],
                             dtype=np.float64)
                    if clean_keys else np.empty((0, 2), dtype=np.float64))
    n_points_clean = len(clean_points)

    # -- edges (clean lattice, inside window) ---------------------------
    edges_array: np.ndarray | None = None
    if return_edges:
        visible = [
            (ka, kb) for ka, kb in edge_set
            if ka in accepted and kb in accepted
            and abs(accepted[ka][0]) <= L and abs(accepted[ka][1]) <= L
            and abs(accepted[kb][0]) <= L and abs(accepted[kb][1]) <= L
        ]
        edges_array = (
            np.array([(accepted[ka], accepted[kb]) for ka, kb in visible])
            if visible
            else np.empty((0, 2, 2), dtype=np.float64)
        )

    # -- noise ---------------------------------------------------------
    points = clean_points.copy()
    n_pre_noise = len(points)
    if n_pre_noise > 0:
        bbox = (points.min(axis=0), points.max(axis=0))
    else:
        bbox = (np.zeros(2), np.zeros(2))
    insertion_rate = 0.0
    if dropout > 0.0:
        points = _apply_dropout(points, dropout, rng)
    if insertion > 0.0:
        points, insertion_rate = _apply_insertion(
            points, insertion, n_pre_noise, bbox, rng,
        )

    # -- diagnostic: mean-NN of post-noise cloud -----------------------
    mean_nn: float | None = None
    if len(points) >= 2:
        tree = cKDTree(points)
        d_nn, _ = tree.query(points, k=2)
        mean_nn = float(d_nn[:, 1].mean())

    # -- optional normalization ----------------------------------------
    if normalize and mean_nn is not None and mean_nn > 1e-12:
        points = points / mean_nn
        if edges_array is not None and len(edges_array) > 0:
            edges_array = edges_array / mean_nn
        if tiles:
            tiles = [(fi, poly / mean_nn) for fi, poly in tiles]

    # -- output --------------------------------------------------------
    result: dict[str, Any] = {
        "points": points,
        "u": u,
        "v": v,
        "dimension": int(dimension),
        "physical_extent": float(physical_extent),
        "n_points_clean": int(n_points_clean),
        "insertion_rate": float(insertion_rate),
        "mean_nn": mean_nn,
    }
    if thetas is not None:
        result["thetas"] = thetas
        result["radii"] = radii
    if return_edges:
        result["edges"] = edges_array
    if return_tiles:
        result["tiles"] = tiles
        result["face_labels"] = face_labels
    return result