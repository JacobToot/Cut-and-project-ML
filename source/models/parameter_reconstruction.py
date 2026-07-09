import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d


def find_gap_peaks_exact_rounded(sequence, decimals=4):
    """Fast exact peak finder for nearly clean sequences.

    Rounds positive finite gaps and checks for exactly 2 or 3 unique values.
    """
    seq = np.asarray(sequence, dtype=np.float64)
    seq = seq[np.isfinite(seq) & (seq > 0)]

    if len(seq) < 3:
        raise ValueError(f"Sequence too short ({len(seq)} valid gaps)")

    rounded = np.round(seq, decimals=decimals)
    peak_positions, peak_counts = np.unique(rounded, return_counts=True)

    if len(peak_positions) not in (2, 3):
        raise ValueError(
            f"Rounded sequence has {len(peak_positions)} unique values, not 2 or 3."
        )

    peak_freqs = peak_counts / peak_counts.sum()
    order = np.argsort(peak_positions)

    return (
        peak_positions[order].astype(np.float64),
        peak_freqs[order].astype(np.float64),
        peak_counts[order].astype(int),
    )


# ── Peak detection: KDE / histogram fallback (from detect_modes) ─────────────

def _x_to_index_units(val, dx):
    if val is None:
        return None
    if np.isscalar(val):
        return max(1, int(round(val / dx)))
    if isinstance(val, (tuple, list)) and len(val) == 2:
        a = None if val[0] is None else max(1, int(round(val[0] / dx)))
        b = None if val[1] is None else max(1, int(round(val[1] / dx)))
        return (a, b)
    raise ValueError("Expected scalar, tuple, list, or None.")


def find_gap_peaks_kde(
    sequence,
    use_kde=True,
    kde_points=2500,
    kde_bw="scott",
    bins=300,
    smooth_sigma=2.0,
    prominence_frac=0.08,
    height_frac=0.0,
    min_distance_x=None,
    width_x=None,
    rel_height=0.5,
):
    """Detect gap peaks via KDE (or histogram) + scipy find_peaks.

    Returns same (peak_positions, peak_freqs, peak_counts) interface.
    """
    seq = np.asarray(sequence, dtype=np.float64)
    seq = seq[np.isfinite(seq) & (seq > 0)]

    if len(seq) < 10:
        raise ValueError("Sequence too short for KDE peak detection")

    xmin, xmax = seq.min(), seq.max()

    if use_kde:
        x = np.linspace(xmin, xmax, kde_points)
        y_raw = gaussian_kde(seq, bw_method=kde_bw)(x)
        y = gaussian_filter1d(y_raw, smooth_sigma) if smooth_sigma > 0 else y_raw.copy()
    else:
        counts, edges = np.histogram(seq, bins=bins, range=(xmin, xmax))
        x = 0.5 * (edges[:-1] + edges[1:])
        y_raw = counts.astype(float)
        y = gaussian_filter1d(y_raw, smooth_sigma) if smooth_sigma > 0 else y_raw.copy()

    dx = x[1] - x[0]

    prominence = prominence_frac * np.max(y)
    height = height_frac * np.max(y) if height_frac > 0 else None
    distance = _x_to_index_units(min_distance_x, dx)
    width = _x_to_index_units(width_x, dx)

    peaks, props = find_peaks(
        y, prominence=prominence, height=height, distance=distance, width=width,
    )

    if len(peaks) == 0:
        raise ValueError("KDE peak detection found 0 peaks")

    peak_positions = x[peaks]
    peak_heights = y[peaks]

    dists = np.abs(seq[:, None] - peak_positions[None, :])
    assignments = np.argmin(dists, axis=1)
    peak_counts = np.bincount(assignments, minlength=len(peak_positions))
    peak_freqs = peak_counts / peak_counts.sum()

    order = np.argsort(peak_positions)
    return (
        peak_positions[order].astype(np.float64),
        peak_freqs[order].astype(np.float64),
        peak_counts[order].astype(int),
    )



def find_gap_peaks(
    sequence,
    rounded_decimals=4,
    try_exact_rounded=True,
    **kde_kwargs,
):
    """Find dominant gap lengths: exact rounded first, KDE fallback."""
    if try_exact_rounded:
        try:
            return find_gap_peaks_exact_rounded(sequence, decimals=rounded_decimals)
        except ValueError:
            pass

    return find_gap_peaks_kde(sequence, **kde_kwargs)



def acceptance_window_from_density(sequence):
    total_length = np.sum(sequence)
    n = len(sequence)
    return (n / total_length) / 2.0


def acceptance_window_from_peaks(peak_positions, peak_freqs):
    mean_gap = np.sum(peak_positions * peak_freqs)
    if mean_gap <= 0:
        raise ValueError(f"Invalid mean gap: {mean_gap}")
    return (1.0 / mean_gap) / 2.0



def solve_slope_for_ij(i, j, delta_k):
    solutions = []
    if j == 0:
        if abs(i) <= delta_k:
            return []
        val = np.sqrt(i**2 / delta_k**2 - 1)
        if val > 0:
            solutions.append(val)
        return solutions

    a_coeff = j**2 - delta_k**2
    b_coeff = 2 * i * j
    c_coeff = i**2 - delta_k**2

    if abs(a_coeff) < 1e-12:
        if abs(b_coeff) < 1e-12:
            return []
        val = -c_coeff / b_coeff
        if val > 0:
            solutions.append(val)
        return solutions

    disc = b_coeff**2 - 4 * a_coeff * c_coeff
    if disc < 0:
        return []

    sqrt_disc = np.sqrt(disc)
    for sign in [1, -1]:
        val = (-b_coeff + sign * sqrt_disc) / (2 * a_coeff)
        if val > 0:
            solutions.append(val)
    return solutions


def _internal_projection(alpha, i, j):
    """Internal (perpendicular) projection of lattice gap (i, j) at slope α."""
    return (-alpha * i + j) / np.sqrt(1 + alpha**2)


def find_candidate_slopes(peak_positions, W_hat, slope_range=(1.0, 5.0),
                          tolerance=2e-2):
    """Recover candidate slopes from two detected tile lengths.

    For each (i,j) in the search grid, solves
        |i + α·j| / √(1 + α²) = δ_k
    for α.  Only keeps proposals whose internal projection
    |−αi + j| / √(1+α²) ≤ 2W, ensuring (i,j) is a physically realisable
    gap within the acceptance window.

    Cross-matches solutions for δ₁ and δ₂, keeping pairs whose slopes
    agree within `tolerance`.  Returns candidates sorted by ascending
    residual (|α₁ − α₂|).
    """
    if len(peak_positions) < 2:
        raise ValueError(f"Need ≥2 tile lengths, got {len(peak_positions)}")

    delta_1, delta_2 = peak_positions[0], peak_positions[1]
    slope_lo, slope_hi = slope_range

    max_ij = max(int(np.ceil(W_hat * slope_hi)) + 2, 10)

    proposals_1, proposals_2 = [], []
    for i in range(-max_ij, max_ij + 1):
        for j in range(-max_ij, max_ij + 1):
            if i == 0 and j == 0:
                continue
            for alpha in solve_slope_for_ij(i, j, delta_1):
                if slope_lo <= alpha <= slope_hi:
                    if abs(_internal_projection(alpha, i, j)) <= 2 * W_hat:
                        proposals_1.append((alpha, i, j))
            for alpha in solve_slope_for_ij(i, j, delta_2):
                if slope_lo <= alpha <= slope_hi:
                    if abs(_internal_projection(alpha, i, j)) <= 2 * W_hat:
                        proposals_2.append((alpha, i, j))

    if not proposals_1 or not proposals_2:
        return []

    alphas_1 = np.array([p[0] for p in proposals_1])
    alphas_2 = np.array([p[0] for p in proposals_2])

    order_2 = np.argsort(alphas_2)
    alphas_2_sorted = alphas_2[order_2]

    candidates = []
    for idx1, (alpha_1, i1, j1) in enumerate(proposals_1):
        lo = np.searchsorted(alphas_2_sorted, alpha_1 - tolerance, side="left")
        hi = np.searchsorted(alphas_2_sorted, alpha_1 + tolerance, side="right")
        for k in range(lo, hi):
            idx2 = order_2[k]
            alpha_2, i2, j2 = proposals_2[idx2]
            residual = abs(alpha_1 - alpha_2)
            candidates.append({
                "slope": (alpha_1 + alpha_2) / 2,
                "residual": residual,
                "ij_1": (i1, j1),
                "ij_2": (i2, j2),
                "delta_1": delta_1,
                "delta_2": delta_2,
            })

    candidates.sort(key=lambda c: c["residual"])

    if len(candidates) > 1:
        deduped = [candidates[0]]
        for c in candidates[1:]:
            if all(abs(c["slope"] - d["slope"]) > tolerance for d in deduped):
                deduped.append(c)
        candidates = deduped

    return candidates


def reconstruct_parameters(
    sequence,
    rounded_decimals=4,
    try_exact_rounded=True,
    slope_range=(1.0, 5.0),
    tolerance=2e-2,
    use_peak_density=True,
    **kde_kwargs,
):
    """Full parameter reconstruction pipeline.

    1. Exact rounded peaks (2 or 3 unique values)
    2. KDE fallback
    3. Acceptance window estimation
    4. Slope recovery
    """
    peak_positions, peak_freqs, peak_counts = find_gap_peaks(
        sequence,
        rounded_decimals=rounded_decimals,
        try_exact_rounded=try_exact_rounded,
        **kde_kwargs,
    )

    W_density = acceptance_window_from_density(sequence)
    W_peaks = acceptance_window_from_peaks(peak_positions, peak_freqs)
    W_hat = W_peaks if use_peak_density else W_density

    candidates = find_candidate_slopes(
        peak_positions, W_hat, slope_range=slope_range, tolerance=tolerance,
    )

    best_slope = candidates[0]["slope"] if candidates else None

    return {
        "W_hat": W_hat,
        "W_hat_density": W_density,
        "W_hat_peaks": W_peaks,
        "peak_positions": peak_positions,
        "peak_freqs": peak_freqs,
        "peak_counts": peak_counts,
        "candidates": candidates,
        "best_slope": best_slope,
    }


def reconstruct_slope_from_tiles(tile_lengths, W_hat, slope_range=(1.0, 5.0),
                                 tolerance=2e-2):
    tile_lengths = np.asarray(tile_lengths, dtype=np.float64)
    candidates = find_candidate_slopes(
        tile_lengths, W_hat, slope_range=slope_range, tolerance=tolerance,
    )
    return candidates[0]["slope"] if candidates else None