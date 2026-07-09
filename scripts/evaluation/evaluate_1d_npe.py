"""evaluate_1d_npe.py
======================

Evaluate a trained 1D NPE model on a noise-rate grid.

For every combination of (gaussian_ratio, poisson_ratio, dropout) in
``noise_levels ** 3`` (default {0, 0.1, 0.2}^3 = 27 conditions),
generates fresh quasicrystal test observations, draws posterior
samples, and saves diagnostic plots in a subfolder named ``{g}_{p}_{d}``.

Per subfolder:
    calibration.png            true vs posterior-mean with error bars
    posterior_grid.png         grid of 2D posterior scatter plots
    marginal_histograms.png    posterior histograms for slope & aw

Once at the top level:
    loss_curve_eval.png        training / val loss (log scale)

Model reconstruction reads ``parameters.txt`` (or the older
``parameter.txt`` / ``model.txt`` for backward compat). Handles both
the current format (dict with a ``model_cfg`` key) and the older
format (the whole file IS the model_cfg dict).
"""

from __future__ import annotations

import argparse
import ast
import itertools
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

def find_root(root_name: str = "cut-and-project-ML") -> Path:
    cwd = Path.cwd()
    for parent in (cwd, *cwd.parents):
        if parent.name == root_name:
            return parent
    for parent in Path(__file__).resolve().parents:
        if parent.name == root_name:
            return parent
    raise RuntimeError(f"Repo root '{root_name}' not found from {cwd}")


_root = find_root()
_source = _root / "source"
if str(_source) not in sys.path:
    sys.path.insert(0, str(_source))

from models.npe_1d import NPEModel


import numpy as np
import torch
from torch.utils.data import Dataset
from simulations.quasi_crystal import quasi_crystal


class NPEDataset(Dataset):
    def __init__(
        self,
        n_samples: int,
        seq_len: int = 8192,
        slope_range: tuple[float, float] = (1.0, 5.0),
        aw_range: tuple[float, float] = (0.5, 5.0),
        gaussian_range: tuple[float, float] = (0.0, 0.3),
        poisson_range: tuple[float, float] = (0.0, 0.2),
        dropout_range: tuple[float, float] = (0.0, 0.2),
        seed: int = 42,
        normalize_spacings: bool = True,
        stats: dict | None = None,
    ):
        rng = np.random.default_rng(seed)
        spacings_list, theta_list = [], []
        attempt = 0

        while len(spacings_list) < n_samples:
            gr = rng.uniform(gaussian_range[0], gaussian_range[1])
            pr = rng.uniform(poisson_range[0], poisson_range[1])
            do = rng.uniform(dropout_range[0], dropout_range[1])
            
            attempt += 1
            try:
                result = quasi_crystal(
                    slope=0,
                    slope_lower=slope_range[0],
                    slope_upper=slope_range[1],
                    acceptance_window=0,
                    acceptance_window_lower=aw_range[0],
                    acceptance_window_upper=aw_range[1],
                    number_of_points=seq_len,
                    gaussian_ratio=gr,
                    poisson_ratio=pr,
                    dropout=do,
                    seed=seed + attempt,
                )
                
                s = result[0]
                sl = result[3]
                aw = result[4]

                if len(s) >= seq_len and np.all(np.isfinite(s[:seq_len])):
                    spacings_list.append(s[:seq_len].astype(np.float32))
                    theta_list.append([sl, aw])
            except Exception:
                pass

            if attempt > n_samples * 20:
                raise RuntimeError("Too many retries generating dataset.")

        spacings = torch.from_numpy(np.stack(spacings_list)) # (N, seq_len)
        theta = torch.tensor(theta_list, dtype=torch.float32) # (N, 2)

        mean_gap = spacings.mean(dim=-1).clamp(min=1e-8) # (N,)
        self.log_mean_gap = mean_gap.log().float() # (N,)

        if normalize_spacings:
            spacings = spacings / mean_gap.unsqueeze(-1)
        self.spacings = spacings # (N, seq_len)

        if stats is None:
            theta_mean = theta.mean(dim=0) # (2,)
            theta_std = theta.std(dim=0).clamp(min=1e-8) # (2,)
        else:
            theta_mean = stats["theta_mean"].float() # (2,)
            theta_std = stats["theta_std"].float().clamp(min=1e-8) # (2,)

        self.theta = (theta - theta_mean) / theta_std # (N, 2)
        self.theta_mean = theta_mean # (2,)
        self.theta_std = theta_std # (2,)
        self.normalize_spacings = normalize_spacings

    def __len__(self) -> int:
        return len(self.spacings)

    def __getitem__(self, idx: int):
        return (
            self.spacings[idx],
            self.log_mean_gap[idx],
            self.theta[idx],
        )


PARAM_NAMES = ["slope", "acceptance window"]


# ---------------------------------------------------------------------
# Test-data generation
# ---------------------------------------------------------------------

def generate_test_crystals(
    n_test, seq_len, slope_range, aw_range,
    gaussian_ratio, poisson_ratio, dropout, seed,
):
    """Generate ``n_test`` test observations with fixed noise rates and
    uniformly sampled (slope, acceptance_window)."""
    rng = np.random.default_rng(seed)
    spacings_list, theta_list = [], []
    attempt = 0

    while len(spacings_list) < n_test:
        slope = rng.uniform(*slope_range)
        aw = rng.uniform(*aw_range)
        try:
            result = quasi_crystal(
                slope=slope,
                acceptance_window=aw,
                number_of_points=seq_len,
                gaussian_ratio=gaussian_ratio,
                poisson_ratio=poisson_ratio,
                dropout=dropout,
                seed=seed + attempt,
            )
            s = result[0]
            if len(s) >= seq_len and np.all(np.isfinite(s[:seq_len])):
                spacings_list.append(s[:seq_len].astype(np.float32))
                theta_list.append([slope, aw])
        except Exception:
            pass
        attempt += 1
        if attempt > n_test * 25:
            raise RuntimeError(
                f"Too many retries generating test crystals "
                f"(g={gaussian_ratio}, p={poisson_ratio}, d={dropout})"
            )

    spacings = torch.from_numpy(np.stack(spacings_list))
    theta = torch.tensor(theta_list, dtype=torch.float32)
    return spacings, theta


@torch.no_grad()
def get_posteriors(model, spacings, theta_mean, theta_std, n_samples, device):
    """Draw posterior samples for each observation.

    Matches the preprocessing done by NPEDataset at training time:
      * divide spacings by their per-sample mean (so the CNN sees mean 1),
      * pass log(raw mean) as log_mean_gap conditioning.
    Then undoes the theta standardisation with (theta_mean, theta_std).

    Returns
    -------
    (n_obs, n_samples, theta_dim) tensor of unstandardised theta samples.
    """
    model.eval()
    all_samples = []
    for idx in range(len(spacings)):
        s = spacings[idx]                                          # (seq_len,)
        mean_gap = s.mean().clamp(min=1e-8)
        lmg = torch.tensor([mean_gap.log().item()],
                           dtype=torch.float32, device=device)
        s = (s / mean_gap).unsqueeze(0).to(device)                 # (1, seq_len)
        samples_std = model.sample(s, n_samples, log_mean_gap=lmg)
        samples_std = samples_std.view(n_samples, -1).cpu()
        samples = samples_std * theta_std + theta_mean
        all_samples.append(samples)
    return torch.stack(all_samples)


def plot_calibration(true_theta, all_samples, save_path, subtitle=""):
    """True vs posterior-mean scatter with ±1 std error bars."""
    post_mean = all_samples.mean(dim=1).numpy()
    post_std = all_samples.std(dim=1).numpy()
    true_np = true_theta.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        lo = min(true_np[:, i].min(), post_mean[:, i].min()) * 0.9
        hi = max(true_np[:, i].max(), post_mean[:, i].max()) * 1.1
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="perfect")
        ax.errorbar(
            true_np[:, i], post_mean[:, i], yerr=post_std[:, i],
            fmt="o", ms=3, alpha=0.5, elinewidth=0.5, capsize=1.5,
            color="steelblue",
        )
        ax.set_xlabel(f"true {name}")
        ax.set_ylabel(f"posterior mean {name}")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(f"Calibration{subtitle}", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_posterior_grid(true_theta, all_samples, n_grid, save_path, subtitle=""):
    n_grid = min(n_grid, len(true_theta))
    ncols = int(np.ceil(np.sqrt(n_grid)))
    nrows = int(np.ceil(n_grid / ncols))
    indices = np.linspace(0, len(true_theta) - 1, n_grid, dtype=int)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows), squeeze=False,
    )
    for panel, data_idx in enumerate(indices):
        row, col = divmod(panel, ncols)
        ax = axes[row, col]
        samples = all_samples[data_idx].numpy()
        true_val = true_theta[data_idx].numpy()
        ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.15,
                   c="steelblue", rasterized=True)
        ax.scatter(true_val[0], true_val[1], s=120, marker="*",
                   c="crimson", edgecolors="k", linewidths=0.5,
                   zorder=5, label="true")
        ax.set_xlabel("slope"); ax.set_ylabel("acceptance window")
        ax.set_title(f"test #{data_idx}", fontsize=9)
    for panel in range(n_grid, nrows * ncols):
        row, col = divmod(panel, ncols)
        axes[row, col].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle(f"Posterior samples vs true parameters{subtitle}",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_marginal_histograms(true_theta, all_samples, n_hist, save_path,
                             subtitle=""):
    n_hist = min(n_hist, len(true_theta))
    indices = np.linspace(0, len(true_theta) - 1, n_hist, dtype=int)

    fig, axes = plt.subplots(n_hist, 2, figsize=(10, 2.4 * n_hist),
                              squeeze=False)
    for row, data_idx in enumerate(indices):
        samples = all_samples[data_idx].numpy()
        true_val = true_theta[data_idx].numpy()
        for col, name in enumerate(PARAM_NAMES):
            ax = axes[row, col]
            ax.hist(samples[:, col], bins=40, density=True,
                    color="steelblue", alpha=0.6, edgecolor="white",
                    linewidth=0.3)
            ax.axvline(true_val[col], color="crimson", linewidth=2,
                       label="true")
            if row == 0:
                ax.set_title(name, fontsize=10)
            ax.set_ylabel(f"test #{data_idx}", fontsize=8)
            if row == 0:
                ax.legend(fontsize=7)
    fig.suptitle(f"Marginal posterior histograms{subtitle}",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(train_dir, save_path):
    metrics_path = train_dir / "metrics_table.tsv"
    if not metrics_path.exists():
        print(f"  [WARN] {metrics_path} not found; skipping loss curve")
        return

    data = np.loadtxt(metrics_path, skiprows=1, delimiter="\t")
    if data.ndim == 1:
        data = data[None, :]
    epochs = data[:, 0].astype(int)
    tr, va = data[:, 1], data[:, 2]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, tr, label="train", linewidth=1.5)
    ax.plot(epochs, va, label="val", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("NLL (log)")
    ax.set_title("1D NPE training loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200); plt.close(fig)
    print(f"  saved {save_path}")


# ---------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------

def load_model_cfg(train_dir: Path) -> dict:
    """Load the model config from parameters.txt / parameter.txt / model.txt.

    Handles both the current format (dict with a 'model_cfg' key) and
    the older format (the whole dict IS the model_cfg).
    """
    for name in ("parameters.txt", "parameter.txt", "model.txt"):
        p = train_dir / name
        if p.exists():
            params = ast.literal_eval(p.read_text())
            if isinstance(params, dict) and "model_cfg" in params:
                return dict(params["model_cfg"])
            return dict(params)
    raise FileNotFoundError(
        f"No parameters.txt / parameter.txt / model.txt in {train_dir}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="results/npe_1d/npe_1d_thesis",
                        help="Training run directory containing "
                             "best_weights.pt + parameters.txt.")
    parser.add_argument("--n_test", type=int, default=200,
                        help="Test crystals per noise combination.")
    parser.add_argument("--n_posterior_samples", type=int, default=1000)
    parser.add_argument("--n_grid", type=int, default=16,
                        help="Panels in posterior_grid.png.")
    parser.add_argument("--n_hist", type=int, default=8,
                        help="Rows in marginal_histograms.png.")
    parser.add_argument("--slope_lower", type=float, default=1.0)
    parser.add_argument("--slope_upper", type=float, default=5.0)
    parser.add_argument("--aw_lower", type=float, default=0.5)
    parser.add_argument("--aw_upper", type=float, default=5.0)
    parser.add_argument("--noise_levels", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2],
                        help="Values used for the (g, p, d)^3 grid.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    train_dir = Path(args.train_dir)
    if not train_dir.is_absolute():
        train_dir = _root / train_dir

    # ---- model reconstruction --------------------------------------
    model_cfg = load_model_cfg(train_dir)
    model = NPEModel(model_cfg).to(device)

    weights_path = train_dir / "best_weights.pt"
    if not weights_path.exists():
        weights_path = train_dir / "weights.pt"
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] device      : {device}")
    print(f"[INFO] weights     : {weights_path}")
    print(f"[INFO] model params: {n_params:,}")
    print(f"[INFO] model_cfg   : {model_cfg}")

    # ---- theta standardisation -------------------------------------
    stats_path = train_dir / "stats.pt"
    stats = torch.load(stats_path, weights_only=True)
    theta_mean = stats["theta_mean"].float()
    theta_std = stats["theta_std"].float()
    seq_len = int(model_cfg["seq_len"])

    # ---- loss curve (once) -----------------------------------------
    print("\n[0] Loss curve ...")
    plot_loss_curve(train_dir, train_dir / "loss_curve_eval.png")

    # ---- sweep -----------------------------------------------------
    combos = list(itertools.product(args.noise_levels, repeat=3))
    print(f"\n[INFO] Sweeping {len(combos)} noise conditions "
          f"(levels = {args.noise_levels})\n")

    for ci, (g, p, d) in enumerate(combos, 1):
        tag = f"{g}_{p}_{d}"
        subtitle = f"  (gauss={g}, poisson={p}, dropout={d})"
        sub_dir = train_dir / tag
        sub_dir.mkdir(exist_ok=True)

        print(f"[{ci}/{len(combos)}] {tag}")
        print(f"  generating {args.n_test} test crystals ...")
        spacings, true_theta = generate_test_crystals(
            n_test=args.n_test, seq_len=seq_len,
            slope_range=(args.slope_lower, args.slope_upper),
            aw_range=(args.aw_lower, args.aw_upper),
            gaussian_ratio=g, poisson_ratio=p, dropout=d,
            seed=args.seed + ci * 10000,
        )

        print(f"  sampling {args.n_posterior_samples} posterior draws "
              f"per obs ...")
        all_samples = get_posteriors(
            model, spacings, theta_mean, theta_std,
            args.n_posterior_samples, device,
        )

        print("  plotting ...")
        plot_calibration(
            true_theta, all_samples,
            sub_dir / "calibration.png", subtitle,
        )
        plot_posterior_grid(
            true_theta, all_samples, args.n_grid,
            sub_dir / "posterior_grid.png", subtitle,
        )
        plot_marginal_histograms(
            true_theta, all_samples, args.n_hist,
            sub_dir / "marginal_histograms.png", subtitle,
        )
        print(f"  -> {sub_dir}")

    print(f"\n[OK] Evaluation complete "
          f"({len(combos)} noise conditions in {train_dir})")


if __name__ == "__main__":
    main()