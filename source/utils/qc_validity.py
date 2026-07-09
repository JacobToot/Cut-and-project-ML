from __future__ import annotations
import math
import numpy as np

def compute_internal_gaps_from_tiles(
    tile_lengths,
    slope: float,
    W: float,
    search_radius: int = 500,
) -> np.ndarray:
    
    """
    Compute internal tile candidates for a given set of tile lengths, slope and acceptance window. 

    The exact computation from 'quasi_crystal(...,internal_tiles=True)' in module simulations.quasi_crystal is preferred as it uses an exact method,
    opposed to the index grid search for candidates used here.

    This function is exclusively a fallback for when internal tiles are unavailable.

    Parameters
    ----------
    tile_lengths : array-like, length 2 or 3 — ascending physical tile lengths
    slope : float
    W : float  — used only for search validation
    search_radius : int
    """
    
    tiles = np.asarray(tile_lengths, dtype=np.float64).ravel()
    if tiles.size not in (2, 3):
        raise ValueError(f"Expected 2 or 3 tile lengths, got {tiles.size}.")

    norm = math.sqrt(1.0 + slope * slope)
    internal_gaps = np.empty(tiles.size, dtype=np.float64)

    for k, t in enumerate(tiles):

        best_g, best_err = None, float("inf")

        for i in range(-search_radius, search_radius + 1):
            j_approx = (t * norm - i) / slope if abs(slope) > 1e-12 else 0.0
            for j in range(int(j_approx) - 2, int(j_approx) + 3):
                err = abs((i + j * slope) / norm - t)
                if err < best_err:
                    best_err = err
                    best_g = (j - i * slope) / norm
        if best_g is None:
            raise ValueError(f"No lattice vector found for tile {t:.6f}.")
        internal_gaps[k] = best_g

    return internal_gaps


def _interval_for_gstar(g: float, W: float) -> tuple[float, float]:

    """
    Return [a, b) such that x_k* in [a,b) guarantees both x_k* and
    x_{k+1}* = x_k* + g are in (-W, W).

    a = max(-W, -W - g)
    b = min( W,  W - g)
    This is the fundamental geometric constraint.

    """

    return max(-W, -W - g), min(W, W - g)


def compute_valid_interval(
    symbols,
    internal_gaps,
    W: float,
    tol: float = 1e-10,
) -> tuple[float, float]:
    
    """
    Compute the feasible interval [x1_min, x1_max) for x_1*.

    x1_min > x1_max means infeasible.
    """

    ds = np.asarray(internal_gaps, dtype=np.float64).ravel()
    syms = np.asarray(symbols, dtype=np.int64).ravel()
    n = int(syms.size)

    if n == 0:
        return float(-W), float(W)

    if np.any((syms < 0) | (syms >= ds.size)):
        raise ValueError(f"Symbol ids out of range [0, {ds.size}).")

    deltas = ds[syms]
    C = np.empty(n, dtype=np.float64)
    C[0] = 0.0
    if n > 1:
        C[1:] = np.cumsum(deltas[:-1])

    lows  = np.empty(n, dtype=np.float64)
    highs = np.empty(n, dtype=np.float64)
    for k in range(n):
        a, b = _interval_for_gstar(float(deltas[k]), W)
        lows[k]  = a - C[k]
        highs[k] = b - C[k]

    return float(lows.max()), float(highs.min())


def is_valid_sequence(
    symbols,
    internal_gaps,
    W: float,
    tol: float = 1e-10,
) -> bool:
    
    """Return True iff the sequence is consistent with a cut-and-project sequence."""

    x1_min, x1_max = compute_valid_interval(symbols, internal_gaps, W, tol=tol)
    return x1_min <= x1_max + tol


def next_symbol_probabilities(
    symbols,
    internal_gaps,
    W: float,
    tol: float = 1e-10,
) -> np.ndarray:
    
    """
    Probability distribution over the next symbol.

    Assumes x_1* uniform over [x1_min, x1_max).
    P(next=s) = overlap(x_{n+1}* range, [a_s, b_s)) / valid_width.

    Raises ValueError if the sequence is infeasible.
    """

    ds = np.asarray(internal_gaps, dtype=np.float64).ravel()
    n_syms = int(ds.size)

    x1_min, x1_max = compute_valid_interval(symbols, ds, W, tol=tol)
    if x1_min > x1_max + tol:
        raise ValueError(
            f"Infeasible: x1_min={x1_min:.9g} > x1_max={x1_max:.9g}."
        )
    x1_max = max(x1_max, x1_min)

    syms = np.asarray(symbols, dtype=np.int64).ravel()
    C_next = float(ds[syms].sum()) if syms.size > 0 else 0.0

    xn_min = x1_min + C_next
    xn_max = x1_max + C_next

    probs = np.zeros(n_syms, dtype=np.float64)
    for s in range(n_syms):
        a_s, b_s = _interval_for_gstar(float(ds[s]), W)
        probs[s] = max(0.0, min(xn_max, b_s) - max(xn_min, a_s))

    total = float(probs.sum())
    if total < tol:
        return np.full(n_syms, 1.0 / n_syms, dtype=np.float64)
    return probs / total