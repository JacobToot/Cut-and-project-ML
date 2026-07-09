import numpy as np
from utils.math_utils import generate_irrational


def compute_internal_coordinates(ij_pairs, slope):
    
    """
    Compute the internal coordinate for each lattice point.

    Arguments:
    ij_pairs : np.ndarray, shape (N, 2)
        The list of Z2 indices sorted by their physical coordinate.
    slope : float
        The irrational slope defining the projection direction y = slope·x.

    Returns:
    star_coords : np.ndarray of float64, shape (N,)
        Internal coordinate x* for each point, in sequence order.
    """

    ij = np.asarray(ij_pairs, dtype=np.float64)
    norm = np.sqrt(1.0 + slope ** 2)
    return (ij[:, 1] - ij[:, 0] * slope) / norm


def _find_tiles(delta_ij, slope):
    
    """
    Identify tile types from lattice gaps in Z2 from a sorted list (according to projected physical coordinates)
    of indices appearing in the cut-and-project acceptance window

    arguments:
    delta_ij : np.ndarray of int, shape (N_gaps, 2)
        Lattice step (Δi, Δj) for each consecutive gap, i.e.
        ``np.diff(ij_array, axis=0)``.
    slope : float
        The irrational slope defining the projection.

    returns:
    physical_tiles : np.ndarray of float64, length 2 or 3
        Distinct physical gap lengths, sorted ascending.
    internal_tiles : np.ndarray of float64, same length
        Internal (star-space) gap for each tile, same order as
        physical_tiles.
    symbols : np.ndarray of int64, length N_gaps
        Tile index for each gap, indexing into physical_tiles /
        internal_tiles.
    """

    unique_steps, inverse = np.unique(delta_ij, axis=0, return_inverse=True)
    n_tiles = len(unique_steps)
    assert n_tiles in (2, 3), (
        f"Expected 2 or 3 tile types, got {n_tiles}."
    )

    norm = np.sqrt(1.0 + slope ** 2)
    physical_tiles = np.empty(n_tiles, dtype=np.float64)
    internal_tiles = np.empty(n_tiles, dtype=np.float64)

    for k in range(n_tiles):
        di = int(unique_steps[k, 0])
        dj = int(unique_steps[k, 1])
        physical_tiles[k] = (di + dj * slope) / norm
        internal_tiles[k] = (dj - di * slope) / norm

    # Sort everything by ascending physical gap length
    order = np.argsort(physical_tiles)
    physical_tiles = physical_tiles[order]
    internal_tiles = internal_tiles[order]

    remap = np.empty(n_tiles, dtype=np.int64)
    remap[order] = np.arange(n_tiles)
    symbols = remap[inverse]

    return physical_tiles, internal_tiles, symbols


def quasi_crystal(
    slope=0,
    slope_upper=5,
    slope_lower=0,
    acceptance_window=0,
    acceptance_window_lower=2,
    acceptance_window_upper=10,
    lattice_spacing=1,
    number_of_points=1000,
    poisson_ratio=0,
    gaussian_ratio=0,
    dropout=0,
    seed=None,
    internal_tiles=False,
):
    
    """
    Generate a cut-and-project sequence on the line y = slope * x.

    Lattice points (i, j) in Z2 whose orthogonal distance from the line
    y = slope * x is within the acceptance window are projected onto that
    line.

    Noise modes (insertions, deletions and gaussian noise) can be
    applied independently and are controlled by the ratio parameters

    arguments:
    slope : float
        Irrational slope. If 0, one is drawn from [slope_lower, slope_upper].
    slope_upper, slope_lower : float
        Range for random slope selection when slope == 0.
    acceptance_window : float
        Half-width of the strip around y = slope·x (in orthogonal units).
        If 0, drawn uniformly from [acceptance_window_lower, acceptance_window_upper].
    acceptance_window_lower, acceptance_window_upper : float
        Range for random acceptance window selection.
    lattice_spacing : float
        Spacing of the underlying square lattice (default 1).
    number_of_points : int
        Number of projected points to retain in the output.
    poisson_ratio : float
        Fraction of clutter points added as a Poisson process. 0 = no clutter.
    gaussian_ratio : float
        Standard deviation of Gaussian jitter as a fraction of mean spacing. 0 = no jitter.
    dropout : float
        Fraction of points randomly removed. Must be in [0, 1).
    seed : int or None
        Random seed for reproducibility.
    internal_tiles : bool
        If True, also return the internal (star-space) gap values
        and the per-gap symbol sequence, computed from exact lattice
        arithmetic.

    returns:
    spacings : np.ndarray of float64
        Gap sequence (differences between consecutive projected points).
    distances : np.ndarray of float64
        Sorted projected positions along the line.
    ij_pairs : np.ndarray of int, shape (N, 2)
        Lattice index pairs (i, j) for each point, sorted by ascending
        physical distance.
    slope : float
        The slope used (useful when randomly generated).
    acceptance_window : float
        The acceptance window used.
    poisson_rate : float
        The Poisson rate used (0 if poisson_ratio == 0).
    noise_std : float
        The Gaussian noise standard deviation used (0 if gaussian_ratio == 0).
    number_tiles : int or None
        Number of distinct tile types (None if number_of_points < 200).
    tiles : np.ndarray of float64 or None
        Distinct physical tile lengths (ascending), or None.

    When *internal_tiles=True*, two additional fields are appended:

    internal_tile_coords : np.ndarray of float64 or None
        Internal gap for each tile, same order as *tiles*.
    symbol_sequence : np.ndarray of int64 or None
        Per-gap tile index (into *tiles* / *internal_tile_coords*).
        Length = number_of_points - 1 (clean sequence before noise).
    """

    rng = np.random.default_rng(seed=seed)

    if not (0 <= dropout < 1):
        raise ValueError(f"dropout must be in [0, 1). Got {dropout}")

    # ---- parameter generation ----
    if slope == 0:
        slope = generate_irrational(
            upper_limit=slope_upper, lower_limit=slope_lower, rng=rng,
        )
    if acceptance_window == 0:
        acceptance_window = rng.uniform(acceptance_window_lower, acceptance_window_upper)

    # ---- lattice projection ----
    sqrt_slope = np.sqrt(1.0 + slope ** 2)
    lower_index = int(-acceptance_window)
    upper_index = number_of_points
    len_lim = (number_of_points * (2.0 / (1.0 - dropout))
               if dropout > 0 else 2 * number_of_points)

    proj_values = []
    ij_list = []

    for i in range(lower_index, upper_index):
        x_coord = i * lattice_spacing
        lower_j = int(np.ceil(
            (slope * x_coord - acceptance_window * sqrt_slope) / lattice_spacing
        ))
        upper_j = int(np.floor(
            (slope * x_coord + acceptance_window * sqrt_slope) / lattice_spacing
        ))

        j_indices = np.arange(lower_j, upper_j + 1, dtype=np.int64)
        y_coords = j_indices * lattice_spacing

        x_proj = (x_coord + y_coords * slope) / (1.0 + slope ** 2)
        mask = x_proj >= 0.0

        proj_values.extend((x_proj[mask] * sqrt_slope).tolist())
        ij_list.extend(
            zip([i] * int(mask.sum()), j_indices[mask].tolist())
        )

        if len(proj_values) > len_lim:
            break

    # ---- sort by ascending physical distance ----
    proj_values = np.array(proj_values, dtype=np.float64)
    ij_array = np.array(ij_list, dtype=np.int64)
    order = np.argsort(proj_values)
    proj_values = proj_values[order]
    ij_array = ij_array[order]

    # ---- tile detection on clean (pre-noise) sequence ----
    tiles = None
    number_tiles = None
    internal_tile_coords = None
    symbol_sequence = None

    if number_of_points >= 200:
        delta_ij = np.diff(ij_array[:number_of_points], axis=0)
        physical_tiles, internal_gaps, symbols = _find_tiles(
            delta_ij, float(slope),
        )
        tiles = physical_tiles
        number_tiles = int(tiles.size)
        if internal_tiles:
            internal_tile_coords = internal_gaps
            symbol_sequence = symbols

    # ---- noise ----
    poisson_rate = 0.0
    noise_std = 0.0

    if dropout > 0:
        keep = rng.uniform(0, 1, len(proj_values)) > dropout
        proj_values = proj_values[keep]

    if poisson_ratio > 0:
        end = proj_values[-1]
        poisson_rate = (number_of_points * poisson_ratio) / end
        value, gaps = 0.0, []
        while value < end:
            value += rng.exponential(scale=1.0 / poisson_rate)
            gaps.append(value)
        proj_values = np.concatenate([proj_values, np.array(gaps)])

    if gaussian_ratio > 0:
        clean_spacings = np.diff(proj_values[:number_of_points])
        noise_std = float(np.mean(clean_spacings)) * gaussian_ratio
        proj_values += rng.normal(0.0, noise_std, size=proj_values.size)

    proj_values = np.sort(proj_values)
    distances = proj_values[:number_of_points + 1]
    spacings = np.diff(distances)

    ij_out = ij_array[:number_of_points + 1]

    if internal_tiles:
        return (
            spacings,
            distances,
            ij_out,
            slope,
            acceptance_window,
            poisson_rate,
            noise_std,
            number_tiles,
            tiles,
            internal_tile_coords,
            symbol_sequence,
        )

    return (
        spacings,
        distances,
        ij_out,
        slope,
        acceptance_window,
        poisson_rate,
        noise_std,
        number_tiles,
        tiles,
    )