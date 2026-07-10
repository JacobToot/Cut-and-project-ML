from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np

def find_root(root="cut-and-project-ML"):
    cwd = Path.cwd()
    for parent in (cwd, *cwd.parents):
        if parent.name == root:
            return parent
    raise RuntimeError(f"'{root}' not found above {cwd}")

root = find_root()
sys.path.insert(0, str(root / "source"))
sys.path.insert(0, str(root / "models"))

from simulations.tiling_2d import _SQRT_PRIMES

import math
import numpy as np

def generate_controlled_basis(
        dimension,
        alpha_min,
        r_lo,
        r_hi,
        snap_range,
        rng,
        max_restarts=1000,
        max_attempts_per_point=1000):


    for _ in range(max_restarts):

        angles = []

        success = True

        for _ in range(dimension):

            placed = False

            for _ in range(max_attempts_per_point):

                theta = rng.uniform(0.0, math.pi)

                if all(abs(theta - a) >= alpha_min for a in angles):
                    angles.append(theta)
                    placed = True
                    break

            if not placed:
                success = False
                break

        if success:
            thetas = np.sort(np.array(angles))
            break

    else:
        raise RuntimeError(
            "Could not place all angles with requested separation."
        )


    radii = rng.uniform(r_lo, r_hi, size=dimension)

    u_ideal = radii * np.cos(thetas)
    v_ideal = radii * np.sin(thetas)

    mapped = (_SQRT_PRIMES % snap_range)

    def snap(val):
        sign = -1.0 if val < 0 else 1.0
        idx = np.argmin(np.abs(mapped - abs(val)))
        return sign * float(mapped[idx])

    u = np.array([snap(x) for x in u_ideal], dtype=np.float64)
    v = np.array([snap(x) for x in v_ideal], dtype=np.float64)

    return u, v, thetas, radii


def generate_tiling_with_tiles(dimension, u, v, physical_extent):
    """
    Run the multigrid algorithm and return vertices, edges, and
    colored tiles. Each tile records its face index (which pair i,j).
    """
    A = np.column_stack([u, v])
    M = np.linalg.inv(A.T @ A) @ A.T

    L = physical_extent
    pad = L + 2

    accepted = {}
    edge_set = set()
    tiles = []         
    face_labels = []    

    face_idx = 0
    for i in range(dimension):
        for j in range(i + 1, dimension):
            det = u[i] * v[j] - u[j] * v[i]
            if abs(det) < 1e-12:
                face_idx += 1
                continue

            face_labels.append((i, j))
            inv_det = 1.0 / det
            max_zi = pad * (abs(u[i]) + abs(v[i]))
            max_zj = pad * (abs(u[j]) + abs(v[j]))

            mi_vals = np.arange(int(np.floor(-max_zi - 0.5)),
                                int(np.ceil(max_zi - 0.5)) + 1)
            mj_vals = np.arange(int(np.floor(-max_zj - 0.5)),
                                int(np.ceil(max_zj - 0.5)) + 1)

            MI, MJ = np.meshgrid(mi_vals, mj_vals)
            mi_flat, mj_flat = MI.ravel(), MJ.ravel()

            c1 = ((mi_flat + 0.5) * v[j] - (mj_flat + 0.5) * v[i]) * inv_det
            c2 = (u[i] * (mj_flat + 0.5) - u[j] * (mi_flat + 0.5)) * inv_det

            mask = (np.abs(c1) <= pad) & (np.abs(c2) <= pad)
            c1, c2 = c1[mask], c2[mask]
            mi_flat, mj_flat = mi_flat[mask], mj_flat[mask]

            if len(c1) == 0:
                face_idx += 1
                continue

            z_all = c1[:, None] * u[None, :] + c2[:, None] * v[None, :]
            base_all = np.round(z_all).astype(int)

            for g in range(len(c1)):
                base = base_all[g]
                mi, mj = int(mi_flat[g]), int(mj_flat[g])

                corners = []
                for di in (0, 1):
                    for dj in (0, 1):
                        x = base.copy()
                        x[i] = mi + di
                        x[j] = mj + dj
                        key = tuple(x)
                        if key not in accepted:
                            c_par = M @ x.astype(np.float64)
                            if abs(c_par[0]) <= pad and abs(c_par[1]) <= pad:
                                accepted[key] = c_par
                        corners.append(key)

                for a_idx, b_idx in ((0, 1), (0, 2), (1, 3), (2, 3)):
                    ka, kb = corners[a_idx], corners[b_idx]
                    if ka in accepted and kb in accepted:
                        edge_set.add((min(ka, kb), max(ka, kb)))

                if all(c in accepted for c in corners):
                    poly = np.array([accepted[corners[0]],
                                     accepted[corners[1]],
                                     accepted[corners[3]],
                                     accepted[corners[2]]])
                    if np.all(np.abs(poly) <= L):
                        tiles.append((face_idx, poly))

            face_idx += 1

    clean_keys = [k for k, p in accepted.items()
                  if abs(p[0]) <= L and abs(p[1]) <= L]
    points = np.array([accepted[k] for k in clean_keys], dtype=np.float64)

    edges_array = None
    if edge_set:
        visible = [(ka, kb) for ka, kb in edge_set
                   if ka in accepted and kb in accepted
                   and abs(accepted[ka][0]) <= L and abs(accepted[ka][1]) <= L
                   and abs(accepted[kb][0]) <= L and abs(accepted[kb][1]) <= L]
        if visible:
            edges_array = np.array([(accepted[ka], accepted[kb])
                                    for ka, kb in visible])

    return points, edges_array, tiles, face_labels


def apply_corruption(points, dropout, insertion, rng):
    """Random vertex dropout + uniform spurious insertion (point set only).

    Removes a `dropout` fraction of vertices and adds `insertion` * N_original
    spurious points drawn uniformly over the current bounding box. Matches
    TilingDataset / evaluate_robustness corruption."""
    pts = points.copy()
    n0 = len(pts)
    if dropout > 0 and n0 > 1:
        keep = max(1, int(round(n0 * (1.0 - dropout))))
        idx = rng.choice(n0, size=keep, replace=False)
        pts = pts[idx]
    if insertion > 0 and len(pts) > 0:
        n_ins = max(1, int(round(n0 * insertion)))
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        spurious = rng.uniform(lo, hi, size=(n_ins, 2))
        pts = np.concatenate([pts, spurious], axis=0)
    return pts


def circular_crop(points, edges, tiles, n_target, circle_frac=0.95):
    """Crop to a DISK centred on the patch, containing ~n_target points.

    Radius is sized so pi * r^2 * density = n_target, capped at
    circle_frac * (min extent)/2 so the disk is always fully inside the data
    (orientation-free boundary). Points/edges/tiles are recentred at origin.
    `edges` may be None and `tiles` may be empty (augmented mode)."""
    if len(points) < 10:
        raise RuntimeError("Tiling too small.")
    center = points.mean(axis=0)
    extent = points.max(axis=0) - points.min(axis=0)
    L = float(extent.min())                          
    r_max = circle_frac * L / 2.0
    density = len(points) / max(float(np.prod(extent)), 1e-12)
    r_target = (np.sqrt(n_target / (np.pi * density))
                if density > 0 else r_max)
    r = min(r_target, r_max)
    r2 = r * r

    pmask = ((points - center) ** 2).sum(axis=1) <= r2
    cropped_pts = points[pmask] - center

    cropped_edges = None
    if edges is not None and len(edges) > 0:
        shifted = edges - center[None, None, :]               
        emask = ((shifted ** 2).sum(axis=2) <= r2).all(axis=1)
        cropped_edges = shifted[emask]

    cropped_tiles = []
    for face_idx, poly in tiles:
        shifted = poly - center
        if np.all((shifted ** 2).sum(axis=1) <= r2):
            cropped_tiles.append((face_idx, shifted))

    return cropped_pts, cropped_edges, cropped_tiles


def rotate_all(pts, edges, tiles, rng):
    """Apply one random rotation about the origin to points, edges, tiles."""
    theta = rng.uniform(0.0, 2.0 * np.pi)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    if edges is not None and len(edges) > 0:
        edges = edges @ R.T                          
    tiles = [(fi, poly @ R.T) for fi, poly in tiles]
    return pts, edges, tiles, theta


def compute_diffraction(points, q_max, n_q=512, backend="nufft"):
    """2D structure factor S(q) = |Σ_i exp(-i q·r_i)|^2 on a regular q-grid
    in [-q_max, q_max]^2.

    backend='nufft'  -- type-1 NUFFT from models/nufft2d.py, O(N + M log M);
                        fast even at n_q=512.
    backend='direct' -- exact brute-force DFT, O(N·M); kept as a reference.

    Returns (S, qx, qy) with q_x along the horizontal (column) axis.
    """
    if backend == "nufft":
        return _diffraction_nufft(points, q_max, n_q)
    return _diffraction_direct(points, q_max, n_q)


def _diffraction_nufft(points, q_max, n_q):
    import torch
    from utils.nufft2d import DiffractionImager, DiffractionConfig


    cfg = DiffractionConfig(
        grid_size=n_q, q_max=q_max, backend="nufft",
        normalize="none", log1p=False, standardize=False, suppress_dc=False)
    imager = DiffractionImager(cfg)

    pts = np.asarray(points, dtype=np.float32)
    pts = pts - pts.mean(axis=0)
    t = torch.from_numpy(pts)[None]                       
    mask = torch.ones(1, t.shape[1])
    with torch.no_grad():
        S = imager(t, mask)[0, 0].cpu().numpy()
    S = S.T
    ax = np.linspace(-q_max, q_max, n_q + 1)[:-1]
    return S, ax, ax


def _diffraction_direct(points, q_max, n_q, chunk=2048):
    """Exact brute-force DFT reference (same q-grid as the NUFFT path)."""
    pts = np.asarray(points, dtype=np.float64)
    pts = pts - pts.mean(axis=0)
    ax = np.linspace(-q_max, q_max, n_q + 1)[:-1]
    QX, QY = np.meshgrid(ax, ax, indexing="xy")
    Q_flat = np.column_stack([QX.ravel(), QY.ravel()])    
    F_flat = np.zeros(n_q * n_q, dtype=np.complex128)
    for start in range(0, len(pts), chunk):
        end = min(start + chunk, len(pts))
        phase = Q_flat @ pts[start:end].T                
        F_flat += np.exp(-1j * phase).sum(axis=1)
    S = (F_flat * F_flat.conj()).real.reshape(n_q, n_q)
    return S, ax, ax


def plot_diffraction(points, peak_alpha, ax=None, save_path=None, q_max=None,
                     n_q=512, log_floor=1.0, backend="nufft"):
    """Render the diffraction pattern as a log-intensity image."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    d_nn, _ = tree.query(points, k=2)
    mean_nn = float(d_nn[:, 1].mean())

    if q_max is None:
        q_max = 6.0 * math.pi / mean_nn

    S, qx, qy = compute_diffraction(points, q_max=q_max, n_q=n_q,
                                    backend=backend)
    S_log = np.log10(S + log_floor)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 7))

    extent = [qx[0], qx[-1], qy[0], qy[-1]]
   
    vmin = np.percentile(S_log, peak_alpha)     
    vmax = S_log.max()

    im = ax.imshow(S_log, origin="lower", extent=extent,
               cmap="inferno", aspect="equal",
               vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlabel(r"$q_x$")
    ax.set_ylabel(r"$q_y$")
    ax.set_title(r"Diffraction pattern")
    plt.colorbar(im, ax=ax, shrink=0.8, label=r"$\log_{10}\,S$")

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)

    return S, qx, qy


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a quasicrystal tiling with colored tiles.")
    parser.add_argument("--u", type=list, default=np.array([1,
                                                   np.cos((2*np.pi)/5),
                                                   np.cos((4*np.pi)/5),
                                                   np.cos((6*np.pi)/5),
                                                   np.cos((8*np.pi)/5)]))
    parser.add_argument("--v", type=list, default=np.array([0,
                                                   np.sin((2*np.pi)/5),
                                                   np.sin((4*np.pi)/5),
                                                   np.sin((6*np.pi)/5),
                                                   np.sin((8*np.pi)/5)]))
    parser.add_argument("--random", type=bool, default=False)
    parser.add_argument("--dimension", type=int, default=5)
    parser.add_argument("--n_target", type=int, default=1024)
    parser.add_argument("--physical_extent", type=float, default=15.0)
    parser.add_argument("--alpha_min", type=float, default=0.18)
    parser.add_argument("--r_lo", type=float, default=0.3)
    parser.add_argument("--r_hi", type=float, default=0.8)
    parser.add_argument("--snap_range", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--peak_alpha", type=float, default=60)


    parser.add_argument("--augment", type=int, default=0, choices=[0, 1],
                        help="1: apply dropout+insertion and show ONLY the "
                             "point cloud (edges/tiles dropped). "
                             "0: clean tiling with colored tiles and edges.")
    parser.add_argument("--dropout", type=float, default=0.00000000005,
                        help="Fraction of vertices removed when --augment 1.")
    parser.add_argument("--insertion", type=float, default=0.5,
                        help="Fraction of spurious vertices added (--augment 1).")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 1],
                        help="Apply a random global rotation (as the network "
                             "sees it). Rotates points, edges and tiles.")
    parser.add_argument("--circle_frac", type=float, default=0.95,
                        help="Disk radius fraction of (min extent)/2; <1 keeps "
                             "the crop fully inside the data.")

    parser.add_argument("--diffraction", action="store_true", default=True,
                    help="Also compute and show the diffraction pattern.")
    parser.add_argument("--diffraction_backend", type=str, default="nufft",
                        choices=["nufft", "direct"],
                        help="Structure-factor backend: 'nufft' (fast, "
                             "default) or 'direct' (exact brute-force).")
    parser.add_argument("--diffraction_save", type=str, default=None,
                        help="If set, save the diffraction plot to this path.")
    parser.add_argument("--diffraction_qmax", type=float, default=None,
                        help="Manual q_max; defaults to 6π/d_nn.")
    parser.add_argument("--diffraction_nq", type=int, default=512,
                        help="Grid resolution for the diffraction pattern.")
    args = parser.parse_args()

    d = args.dimension
    n_faces = d * (d - 1) // 2

    print(f"Generating d={d} tiling with ~{args.n_target} vertices, "
          f"{n_faces} face types ...")

    rng = np.random.default_rng(args.seed)

    if args.random:
        u, v, thetas, radii = generate_controlled_basis(
            d, args.alpha_min, args.r_lo, args.r_hi, args.snap_range, rng)
        
    else:
        u,v = args.u, args.v
        thetas, radii = [0]*args.dimension, [0]*args.dimension
        assert len(u) == args.dimension and len(v) == args.dimension, "Length of u and v must match the dimension of the lattice."


    points, edges, tiles, face_labels = generate_tiling_with_tiles(
        d, u, v, args.physical_extent)

    # --- augmentation: corrupt the vertex set, drop the tiling structure ---
    if args.augment:
        points = apply_corruption(points, args.dropout, args.insertion, rng)
        edges, tiles = None, []        # corrupted vertices are not a tiling
        print(f"Augmented: dropout={args.dropout}, insertion={args.insertion} "
              f"-> {len(points)} vertices (edges/tiles disabled)")

    # --- circular crop (points, and edges/tiles when present) ---
    pts, edges, tiles = circular_crop(
        points, edges, tiles, args.n_target, args.circle_frac)

    # --- optional random rotation (what the network sees) ---
    rot_theta = None
    if args.rotate:
        pts, edges, tiles, rot_theta = rotate_all(pts, edges, tiles, rng)

    print(f"Cropped to {len(pts)} vertices, {len(tiles)} tiles.")

    A = np.column_stack([u, v])
    M = np.linalg.inv(A.T @ A) @ A.T
    e_stars = M @ np.eye(d)
    rhombus_angles = []
    for i in range(d):
        for j in range(i + 1, d):
            dot = (e_stars[0, i] * e_stars[0, j]
                   + e_stars[1, i] * e_stars[1, j])
            ni = math.sqrt(e_stars[0, i]**2 + e_stars[1, i]**2)
            nj = math.sqrt(e_stars[0, j]**2 + e_stars[1, j]**2)
            cos_val = np.clip(abs(dot / (ni * nj + 1e-12)), 0, 1)
            rhombus_angles.append(math.acos(cos_val))

    cmap = plt.cm.get_cmap("tab20" if n_faces <= 20 else "hsv", n_faces)

    if args.diffraction:
        fig, (ax, ax_diff) = plt.subplots(1, 2, figsize=(16, 8))
    else:
        fig, ax = plt.subplots(figsize=(9, 9),dpi=200)
        ax_diff = None

    if tiles:
        patches = []
        colors = []
        for face_idx, poly in tiles:
            patches.append(Polygon(poly, closed=True))
            colors.append(cmap(face_idx % cmap.N))
        pc = PatchCollection(patches, facecolors=colors,
                             edgecolors="0.4", linewidths=0.3, alpha=0.55)
        ax.add_collection(pc)

    if edges is not None and len(edges) > 0:
        segments = [[(e[0, 0], e[0, 1]), (e[1, 0], e[1, 1])] for e in edges]
        lc = mc.LineCollection(segments, linewidths=0.3, colors="0.3",
                               zorder=2)
        ax.add_collection(lc)

    ax.scatter(pts[:, 0], pts[:, 1], s=2, c="black", zorder=3)

    ax.set_aspect("equal")
    # title = f"Tiling with N = {d}"
    # ax.set_title(title, fontsize=11)

    margin = 0.05 * max(pts[:, 0].ptp(), pts[:, 1].ptp())
    ax.set_xlim(pts[:, 0].min() - margin, pts[:, 0].max() + margin)
    ax.set_ylim(pts[:, 1].min() - margin, pts[:, 1].max() + margin)
    ax.set_axis_off()
    from matplotlib.patches import Patch
    legend_handles = []
    for idx, (i, j) in enumerate(face_labels):
        legend_handles.append(
            Patch(facecolor=cmap(idx % cmap.N), alpha=0.55,
                  edgecolor="0.4", label=f"({i},{j})"))
    # ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
    #           ncol=max(1, n_faces // 10 + 1), title="Face (i,j)")

    u_str = ", ".join(f"{x:.3f}" for x in u)
    v_str = ", ".join(f"{x:.3f}" for x in v)
    theta_str = ", ".join(f"{math.degrees(t):.1f}" for t in thetas)

    if ax_diff is not None:
        plot_diffraction(
            pts,
            peak_alpha=args.peak_alpha,
            ax=ax_diff,
            q_max=args.diffraction_qmax,
            n_q=args.diffraction_nq,
            backend=args.diffraction_backend,
        )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.12 + 0.015 * d)

    if args.save:
        plt.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()

    print(f"\n{'='*60}")
    print(f"Dimension:          {d}")
    print(f"Augmented:          {bool(args.augment)}")
    if args.augment:
        print(f"  dropout:          {args.dropout}")
        print(f"  insertion:        {args.insertion}")
    print(f"Rotated:            {bool(args.rotate)}")
    print(f"Vertices:           {len(pts)}")
    print(f"Tiles:              {len(tiles)}")
    print(f"Face types:         {n_faces}")
    print(f"Design angles:      [{theta_str}] deg")
    print(f"Min rhombus angle:  {math.degrees(min(rhombus_angles)):.2f} deg")
    print(f"u = [{u_str}]")
    print(f"v = [{v_str}]")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()