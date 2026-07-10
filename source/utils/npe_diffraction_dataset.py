from __future__ import annotations

import argparse
import math
import pickle
import pprint
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import Delaunay, cKDTree
from torch.utils.data import Dataset


def find_root(root: str = "cut-and-project-ML") -> Path:
    cwd = Path.cwd()
    for parent in (cwd, *cwd.parents):
        if parent.name == root:
            return parent
    raise RuntimeError(f"Specified root '{root}' not found")

_root = find_root()
_source = _root / "source"
if str(_source) not in sys.path:
    sys.path.insert(0, str(_source))

from simulations.tiling_2d import _SQRT_PRIMES, multigrid_tiling


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


def generate_controlled_basis(
        dimension: int,
        alpha_min: float,
        r_lo: float,
        r_hi: float,
        snap_range: float,
        rng: np.random.Generator,
        max_restarts=1000,
        max_attempts_per_point=1000):  # NOTE: currently unused; kept for signature compatibility

    for _ in range(max_restarts):

        intervals = [(0.0, math.pi)]
        angles = []

        success = True

        for _ in range(dimension):

            lengths = np.array([b - a for a, b in intervals])
            total = lengths.sum()

            if total <= 0:
                success = False
                break

            x = rng.uniform(0.0, total)

            acc = 0.0
            chosen = None

            for i, L in enumerate(lengths):
                if acc + L >= x:
                    chosen = i
                    break
                acc += L

            if chosen is None:
                success = False
                break

            a, b = intervals[chosen]

            theta = a + (x - acc)

            left = (a, theta - alpha_min)
            right = (theta + alpha_min, b)

            intervals.pop(chosen)

            if left[1] > left[0]:
                intervals.append(left)
            if right[1] > right[0]:
                intervals.append(right)

            intervals = sorted(intervals)
            angles.append(theta)

        if success and len(angles) == dimension:
            angles = np.sort(np.array(angles))
            break
    else:
        return None, None, None, None

    radii = rng.uniform(r_lo, r_hi, size=dimension)

    u_ideal = radii * np.cos(angles)
    v_ideal = radii * np.sin(angles)

    mapped = (_SQRT_PRIMES % snap_range)

    def snap(val):
        sign = -1.0 if val < 0 else 1.0
        idx = np.argmin(np.abs(mapped - abs(val)))
        return sign * float(mapped[idx])

    u = np.array([snap(x) for x in u_ideal], dtype=np.float64)
    v = np.array([snap(x) for x in v_ideal], dtype=np.float64)

    return u, v, angles, radii


def build_tilings(
    num_tilings: int,
    dimensions: list[int],
    physical_extent: float,
    seed: int,
    alpha_min: float,
    r_lo: float,
    r_hi: float,
    snap_range: float,
    target_points: int = 4096,
    circle_frac: float = 0.95,
    attempt_multiplier: int = 10,
) -> list[dict]:
    """Build a list of tiling dicts, each containing:
      - points      : (n, 2) float32, centered, circularly cropped to
                       ~target_points points, no noise applied
      - dimension   : int, cut-and-project dimension d
      - components  : list[float], length 2*dimension, [u | v] concatenation
      - n_raw       : int, number of points in the raw (uncropped) tiling
      - n_cropped   : int, number of points retained after the circular crop
      - density     : float, raw point density (points / raw bounding-box area)
      - seed        : int, the per-tiling RNG seed used

    No edge histogram is stored: it depends on runtime corruption and
    rescaling (see NPEDiffraction2D), so it can't be precomputed here.

    Basis vectors are generated here (generate_controlled_basis) and
    passed to multigrid_tiling as explicit u=/v=, rather than letting
    multigrid_tiling generate its own -- this keeps the (already
    None-safe) basis sampler as the single source of basis-generation
    failure handling, and avoids depending on the simulator's internal
    _random_basis, which uses a different sampling algorithm and raises
    RuntimeError rather than returning None on failure. Similarly,
    dropout/insertion/normalize are intentionally left at their
    multigrid_tiling defaults (0.0 / False): noise and normalization
    stay exclusively the Dataset's job at __getitem__ time.
    """
    rng = np.random.default_rng(seed)

    n_dims = len(dimensions)
    per_dim = num_tilings // n_dims
    extra = num_tilings % n_dims

    print(f"[INFO] Sampling: positive interval, "
          f"alpha_min={alpha_min}, r_lo={r_lo}, r_hi={r_hi}, "
          f"{per_dim}+ tilings/dim")

    tilings: list[dict] = []
    counts = {d: 0 for d in dimensions}

    for dim_idx, dim in enumerate(dimensions):
        dim_target = per_dim + (1 if dim_idx < extra else 0)
        dim_built = 0
        max_attempts = dim_target * attempt_multiplier

        for attempt in range(max_attempts):
            if dim_built >= dim_target:
                break

            u_comps, v_comps, _, _ = generate_controlled_basis(
                dim, alpha_min, r_lo, r_hi, snap_range, rng,
            )
            if u_comps is None or v_comps is None:
                continue
            components = np.concatenate([u_comps, v_comps])

            tiling_seed = int(rng.integers(0, 2**31))
            try:
                out = multigrid_tiling(
                    dimension=dim,
                    u=u_comps,
                    v=v_comps,
                    physical_extent=physical_extent,
                    seed=tiling_seed,
                )
            except (ValueError, RuntimeError):
                continue

            points = out["points"]
            n_raw = len(points)
            if n_raw < target_points:
                continue

            bbox_extent = points.max(axis=0) - points.min(axis=0)
            bbox_area = float(np.prod(bbox_extent))
            if bbox_area < 1e-8:
                continue
            density = n_raw / bbox_area

            # Circular (isotropic) crop, done once here at generation
            # time. NPEDiffraction2D crops again to a smaller, randomly
            # sized window per __getitem__ call; it works on this cloud
            # rather than a square one, so both crops use a consistent
            # radius-from-density estimate.
            cropped = _circular_crop(points, target_points, circle_frac)
            if len(cropped) < 100:
                continue

            cropped = cropped - cropped.mean(axis=0)

            tilings.append(dict(
                points=cropped.astype(np.float32),
                dimension=dim,
                components=components.tolist(),
                n_raw=n_raw,
                n_cropped=len(cropped),
                density=float(density),
                seed=tiling_seed))
            counts[dim] += 1
            dim_built += 1
            if dim_built % 50 == 0:
                print(f"  dim={dim}: {dim_built}/{dim_target}")

        if dim_built < dim_target:
            print(f"  [WARN] dim={dim}: built {dim_built}/{dim_target} "
                  f"after {max_attempts} attempts")

    print(f"[INFO] Tilings per dimension: {counts}")
    print(f"[INFO] Total tilings built: {len(tilings)}")
    return tilings


def save_tilings(path: Path, tilings: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(tilings, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_tilings(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


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

    if p_insert > 0.0:
        bb_min = points.min(axis=0)
        bb_max = points.max(axis=0)
    else:
        bb_min = bb_max = None

    if p_drop > 0.0:
        n_drop = int(round(p_drop * n_orig))
        n_drop = min(n_drop, n_orig - 4)
        if n_drop > 0:
            keep_idx = rng.choice(n_orig, size=n_orig - n_drop,
                                  replace=False)
            points = points[keep_idx]

    if p_insert > 0.0:
        n_insert = int(round(p_insert * n_orig))
        if n_insert > 0 and bb_min is not None:
            spurious = rng.uniform(bb_min, bb_max,
                                   size=(n_insert, 2)).astype(points.dtype)
            points = np.concatenate([points, spurious], axis=0)
            shuf = rng.permutation(len(points))
            points = points[shuf]

    return points


class NPEDiffraction2D(Dataset):
    """Tilings + canonical (u, v) labels + edge-length histogram, with
    optional drop/insert augmentation.

    Expects tilings produced by `build_tilings` / loaded via
    `load_tilings`: each entry's `points` are already centered and
    circularly cropped, with no noise applied. This class:
      1. draws a further, randomly-sized circular sub-window per item
         (n_min..n_max), for augmentation diversity,
      2. optionally corrupts it (drop/insert),
      3. rescales to unit mean nearest-neighbour distance,
      4. computes the Delaunay edge-length histogram on the *final*
         (corrupted, rescaled) points -- this is why the histogram is
         never precomputed at generation time.

    Mode selection for corruption rates:
      1. Exact (eval): pass exact_drop_rate and/or exact_insert_rate.
      2. Random (train): set drop_rate_max > 0 and/or insert_rate_max > 0.
      3. Off (default): all rate defaults are 0.0, no corruption.
    """

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

        if self.normalize and mean_nn > 1e-8:
            points = points / mean_nn

        if self.compute_hist:
            edge_hist = compute_edge_length_histogram(
                points, n_bins=self.hist_n_bins,
                hist_min=self.hist_min, hist_max=self.hist_max)
        else:
            edge_hist = np.zeros(self.hist_n_bins, dtype=np.float32)

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


# ----------------------------------------------------------------------------
# generation only. Creates a dataset of tilings for use by all models that work on 2d tilings. 
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate and save quasicrystal tiling datasets.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--npe_dataset", type=int, default=5,
                        help="Dimension to use when --multi_dim is not "
                             "set (single-dimension generation, the "
                             "default). Saved under "
                             "<dataset_dir>/npe_2d/.")
    parser.add_argument("--multi_dim", type=int, default=0, choices=[0, 1]) # 1 is truth 0 is false

    parser.add_argument("--train_num_tilings", type=int, default=18)
    parser.add_argument("--val_num_tilings", type=int, default=18)
    parser.add_argument("--physical_extent", type=float, default=40.0)
    parser.add_argument("--d_min", type=int, default=4)
    parser.add_argument("--d_max", type=int, default=9)
    parser.add_argument("--target_points", type=int, default=4096)
    parser.add_argument("--circle_frac", type=float, default=0.95)
    parser.add_argument("--attempt_multiplier", type=int, default=10)
    parser.add_argument("--alpha_min", type=float, default=0.2) 
    parser.add_argument("--r_lo", type=float, default=0.25)
    parser.add_argument("--r_hi", type=float, default=0.8)
    parser.add_argument("--snap_range", type=float, default=0.9) # sets to which values the sqrt_primes will be regularised to. The operations performed on the root primes preserve the badly approximable property.

    args = parser.parse_args()

    if not args.multi_dim:
        if args.npe_dataset < 4:
            raise ValueError(
                f"--npe_dataset must be >= 4 (multigrid_tiling requires "
                f"dimension >= 4). Got {args.npe_dataset}."
            )
        dataset_dir = _root / args.dataset_dir / f"npe_2d_dim={args.npe_dataset}" 
        dimensions = [args.npe_dataset]
    else:
        dataset_dir = _root / args.dataset_dir
        dimensions = list(range(args.d_min, args.d_max + 1))

    print(f"[INFO] Mode: {'multi_dim sweep ' + str(dimensions) if args.multi_dim else f'single-dimension (dim={args.npe_dataset})'}")
    print(f"[INFO] Output: {dataset_dir}")

    train_dir = dataset_dir / "training"
    val_dir = dataset_dir / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_seed = args.seed
    val_seed = args.seed + 1

    print("[INFO] Building training tilings ...")
    t0 = time.time()
    train_tilings = build_tilings(
        num_tilings=args.train_num_tilings,
        dimensions=dimensions,
        physical_extent=args.physical_extent,
        alpha_min=args.alpha_min,
        r_lo=args.r_lo,
        r_hi=args.r_hi,
        snap_range=args.snap_range,
        seed=train_seed,
        target_points=args.target_points,
        circle_frac=args.circle_frac,
        attempt_multiplier=args.attempt_multiplier)
    train_time = time.time() - t0
    
    print(f"[INFO] Training tilings: {len(train_tilings)}  "
          f"({train_time:.1f}s)")

    print("[INFO] Building validation tilings ...")
    t0 = time.time()
    val_tilings = build_tilings(
        num_tilings=args.val_num_tilings,
        dimensions=dimensions,
        physical_extent=args.physical_extent,
        alpha_min=args.alpha_min,
        r_lo=args.r_lo,
        r_hi=args.r_hi,
        snap_range=args.snap_range,
        seed=val_seed,
        target_points=args.target_points,
        circle_frac=args.circle_frac,
        attempt_multiplier=args.attempt_multiplier)
    val_time = time.time() - t0
    print(f"[INFO] Validation tilings: {len(val_tilings)}  "
          f"({val_time:.1f}s)")

    train_path = train_dir / f"quasiperiodic_2d_seed_{train_seed}.pkl"
    val_path = val_dir / f"quasiperiodic_2d_seed_{val_seed}.pkl"

    save_tilings(train_path, train_tilings)
    print(f"[OK] Saved training data: {train_path}")

    save_tilings(val_path, val_tilings)
    print(f"[OK] Saved validation data: {val_path}")

    setup = dict(
        seed=args.seed,
        train_seed=train_seed,
        val_seed=val_seed,
        train_num_tilings=len(train_tilings),
        val_num_tilings=len(val_tilings),
        physical_extent=args.physical_extent,
        alpha_min=args.alpha_min,
        r_lo=args.r_lo,
        r_hi=args.r_hi,
        snap_range=args.snap_range,
        multi_dim=args.multi_dim,
        dimension_range=([args.d_min, args.d_max] if args.multi_dim
                        else [args.npe_dataset, args.npe_dataset]),
        dimensions=dimensions,
        target_points=args.target_points,
        circle_frac=args.circle_frac,
        attempt_multiplier=args.attempt_multiplier,
        train_file=str(train_path.relative_to(dataset_dir)),
        val_file=str(val_path.relative_to(dataset_dir)),
        generation_time_seconds=dict(
            train=round(train_time, 1),
            val=round(val_time, 1)),
    )

    setup_path = dataset_dir / "data_setup.txt"
    with open(setup_path, "w") as f:
        pprint.pprint(setup, stream=f)
    print(f"[OK] Saved setup: {setup_path}")


if __name__ == "__main__":
    main()