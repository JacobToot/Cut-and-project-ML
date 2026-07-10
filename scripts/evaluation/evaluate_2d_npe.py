from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import ast
import math
import pickle
import pprint
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def find_root(root: str = "cut-and-project-ML") -> Path:
    cwd = Path.cwd()
    for parent in (cwd, *cwd.parents):
        if parent.name == root:
            return parent
    raise RuntimeError(f"Specified root '{root}' not found")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # if torch.backends.mps.is_available():
    #     return "mps"
    return "cpu"


def load_tilings(directory: Path) -> list:
    pkl_files = sorted(directory.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {directory}")
    path = pkl_files[0]
    print(f"[INFO] Loading {path} ...")
    with open(path, "rb") as f:
        tilings = pickle.load(f)
    print(f"[INFO] Loaded {len(tilings)} tilings from {path.name}")
    return tilings


def load_parameters(params_path: Path) -> dict:
    return ast.literal_eval(params_path.read_text())


def parse_noise_levels(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


# ----------------------------------------------------------------------------
# Model reconstruction
# ----------------------------------------------------------------------------

def build_model_from_params(params, weights_path, device):
    expected_dim = int(params["expected_dim"])
    theta_dim = int(params["theta_dim"])
    mc = dict(params["model_cfg"])
    dc = dict(params["diffraction_cfg"])

    from utils.nufft2d import DiffractionConfig
    from utils.nufft2d import DiffractionImager


    diff_cfg = DiffractionConfig(
        grid_size=int(dc["grid_size"]),
        q_max=float(dc["q_max_factor"]) * math.pi,
        backend=dc.get("backend", "nufft"),
        normalize="per_atom",
        log1p=bool(dc.get("log1p", True)),
        standardize=True,
        suppress_dc=bool(dc.get("suppress_dc", True)),
        dc_radius=int(dc.get("dc_radius", 1)),
        nufft_sigma=float(dc.get("nufft_sigma", 2.0)),
        nufft_width=int(dc.get("nufft_width", 6)),
    )
    diff_image_module = DiffractionImager(diff_cfg)

    from models.npe_diffraction_2d import NPEDiffraction2D

    model = NPEDiffraction2D(
        diff_image_module=diff_image_module,
        cnn_width=mc.get("cnn_width", 32),
        n_res_per_stage=mc.get("n_res_per_stage", 2),
        n_head_layers=mc.get("n_head_layers", 2),
        dropout=mc.get("dropout", 0.1),
        context_dim=mc.get("context_dim", 128),
        use_edge_hist=mc.get("use_edge_hist", True),
        hist_n_bins=mc.get("hist_n_bins", 64),
        hist_feature_dim=mc.get("hist_feature_dim", 64),
        hist_cnn_width=mc.get("hist_cnn_width", 32),
        use_log_mean_nn=mc.get("use_log_mean_nn", True),
        theta_dim=theta_dim,
        K=mc.get("K", 8), B=mc.get("B", 3),
        n_flow_steps=mc.get("n_flow_steps", 8),
        hidden_dim_conditioner=mc.get("hidden_dim_conditioner", 128),
        num_conditioner_blocks=mc.get("num_conditioner_blocks", 2),
    ).to(device)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, expected_dim, theta_dim


# ----------------------------------------------------------------------------
# Sampling + metrics
# ----------------------------------------------------------------------------

def draw_posterior_samples(model, dataset, indices, n_samples, device,
                           use_edge_hist=True, verbose=True):
    N = len(indices)
    samples = []
    truth = []
    log_nn = []
    nll = []

    for k, idx in enumerate(indices):
        item = dataset[idx]
        pts_np, edge_hist_np, lnn, theta_true_np = item

        pts_t = torch.as_tensor(pts_np, dtype=torch.float32
                                ).unsqueeze(0).to(device)
        mask_t = torch.ones(1, len(pts_np), dtype=torch.float32,
                            device=device)
        lnn_t = torch.tensor([lnn], dtype=torch.float32, device=device)
        theta_true_t = torch.as_tensor(theta_true_np, dtype=torch.float32
                                       ).unsqueeze(0).to(device)
        eh_t = (torch.as_tensor(edge_hist_np, dtype=torch.float32
                                ).unsqueeze(0).to(device)
                if use_edge_hist else None)

        with torch.no_grad():
            s = model.sample(pts_t, mask_t, n_samples,
                             edge_hist=eh_t, log_mean_nn=lnn_t)
            lp = model.log_prob(pts_t, mask_t, theta_true_t,
                                edge_hist=eh_t, log_mean_nn=lnn_t)

        samples.append(s.cpu().numpy())
        truth.append(theta_true_np)
        log_nn.append(lnn)
        nll.append(-float(lp.item()))

        if verbose and ((k + 1) % max(1, N // 5) == 0 or (k + 1) == N):
            print(f"    [{k + 1}/{N}] sampled "
                  f"(mean nll so far: {np.mean(nll):.3f})")

    return (np.stack(samples).astype(np.float32),
            np.stack(truth).astype(np.float32),
            np.array(log_nn, dtype=np.float32),
            np.array(nll, dtype=np.float32))


COVERAGE_LEVELS = [0.50, 0.68, 0.90, 0.95]


def compute_metrics(samples, truth):
    pred_mean = samples.mean(axis=1)
    pred_std = samples.std(axis=1)
    err = pred_mean - truth
    rmse = np.sqrt((err ** 2).mean(axis=0))
    mae = np.abs(err).mean(axis=0)
    sharp = pred_std.mean(axis=0)

    coverage = {}
    for level in COVERAGE_LEVELS:
        alpha = (1 - level) / 2
        lo = np.quantile(samples, alpha, axis=1)
        hi = np.quantile(samples, 1 - alpha, axis=1)
        coverage[level] = ((truth >= lo) & (truth <= hi)).mean(axis=0)

    return dict(pred_mean=pred_mean, pred_std=pred_std, err=err,
                rmse=rmse, mae=mae, sharpness=sharp, coverage=coverage)


def compute_coverage_curve(samples, truth, n_points=19):
    """Calibration curve: nominal vs empirical coverage per component."""
    nominal = np.linspace(0.05, 0.95, n_points)
    theta_dim = truth.shape[1]
    empirical = np.zeros((len(nominal), theta_dim))
    for k, level in enumerate(nominal):
        alpha = (1 - level) / 2
        lo = np.quantile(samples, alpha, axis=1)
        hi = np.quantile(samples, 1 - alpha, axis=1)
        empirical[k] = ((truth >= lo) & (truth <= hi)).mean(axis=0)
    return nominal, empirical


# ----------------------------------------------------------------------------
# Output writers — all .tsv
# ----------------------------------------------------------------------------

def write_metrics_tsv(m, theta_dim, result_dir):
    d = theta_dim // 2
    cov_levels = sorted(m['coverage'].keys())
    with open(result_dir / "metrics.tsv", "w") as f:
        cols = ["component", "rmse", "mae", "posterior_std"] + \
               [f"coverage_{int(L * 100)}" for L in cov_levels]
        f.write("\t".join(cols) + "\n")
        for i in range(theta_dim):
            name = f"u_{i}" if i < d else f"v_{i - d}"
            row = [name,
                   f"{m['rmse'][i]:.6f}",
                   f"{m['mae'][i]:.6f}",
                   f"{m['sharpness'][i]:.6f}"]
            for L in cov_levels:
                row.append(f"{m['coverage'][L][i]:.4f}")
            f.write("\t".join(row) + "\n")


def write_per_sample_tsv(samples, truth, log_nn, nll, indices, result_dir):
    pred_mean = samples.mean(axis=1)
    pred_std = samples.std(axis=1)
    theta_dim = truth.shape[1]
    d = theta_dim // 2
    cols = ["sample_idx_in_split", "log_mean_nn", "nll"]
    for i in range(theta_dim):
        name = f"u_{i}" if i < d else f"v_{i - d}"
        cols.extend([f"true_{name}", f"pred_{name}", f"std_{name}"])
    with open(result_dir / "per_sample.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for k in range(len(truth)):
            vals = [str(indices[k]), f"{log_nn[k]:.6f}", f"{nll[k]:.6f}"]
            for i in range(theta_dim):
                vals.extend([f"{truth[k, i]:.6f}",
                             f"{pred_mean[k, i]:.6f}",
                             f"{pred_std[k, i]:.6f}"])
            f.write("\t".join(vals) + "\n")


def write_coverage_curves_tsv(nominal, empirical, theta_dim, result_dir):
    d = theta_dim // 2
    cols = ["nominal_level"] + [
        f"emp_coverage_u_{i}" for i in range(d)
    ] + [
        f"emp_coverage_v_{i}" for i in range(d)
    ] + ["emp_coverage_mean"]
    mean_emp = empirical.mean(axis=1)
    with open(result_dir / "coverage_curves.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for k in range(len(nominal)):
            row = [f"{nominal[k]:.4f}"]
            row.extend(f"{empirical[k, i]:.4f}" for i in range(theta_dim))
            row.append(f"{mean_emp[k]:.4f}")
            f.write("\t".join(row) + "\n")


def write_summary_tsv(summary_rows, path):
    if not summary_rows:
        return
    cols = list(summary_rows[0].keys())
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for row in summary_rows:
            f.write("\t".join(
                f"{row[c]:.6f}" if isinstance(row[c], float) else str(row[c])
                for c in cols
            ) + "\n")


# ----------------------------------------------------------------------------
# Per-level plots (reuse previous style)
# ----------------------------------------------------------------------------

def plot_prediction_vs_truth_uv(samples, truth, theta_dim, result_dir):
    d = theta_dim // 2
    pred_mean = samples.mean(axis=1)
    pred_std = samples.std(axis=1)
    fig, axes = plt.subplots(2, d, figsize=(2.6 * d, 5.2), sharex='col')
    for i in range(theta_dim):
        row, col = divmod(i, d)
        ax = axes[row, col]
        ax.errorbar(truth[:, i], pred_mean[:, i], yerr=pred_std[:, i],
                    fmt='o', ms=2.5, alpha=0.45, capsize=1.5,
                    color='steelblue', ecolor='lightsteelblue')
        lo = min(truth[:, i].min(), pred_mean[:, i].min())
        hi = max(truth[:, i].max(), pred_mean[:, i].max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.6)
        ax.set_title(f"u_{col}" if row == 0 else f"v_{col}", fontsize=9)
        ax.grid(True, alpha=0.2)
    for col in range(d):
        axes[1, col].set_xlabel("true", fontsize=8)
    axes[0, 0].set_ylabel("predicted (mean ± std)", fontsize=8)
    axes[1, 0].set_ylabel("predicted (mean ± std)", fontsize=8)
    fig.suptitle("Per-component posterior mean vs ground truth", fontsize=11)
    plt.tight_layout()
    plt.savefig(result_dir / "prediction_vs_truth_uv.png", dpi=200)
    plt.close()


def plot_prediction_vs_truth_rtheta(samples, truth, expected_dim, result_dir):
    d = expected_dim
    sa_u = samples[:, :, :d]; sa_v = samples[:, :, d:]
    sa_r = np.sqrt(sa_u ** 2 + sa_v ** 2)
    sa_th = np.arctan2(sa_v, sa_u)
    tr_r = np.sqrt(truth[:, :d] ** 2 + truth[:, d:] ** 2)
    tr_th = np.arctan2(truth[:, d:], truth[:, :d])
    pred_r_mean = sa_r.mean(axis=1); pred_r_std = sa_r.std(axis=1)
    pred_th_mean = sa_th.mean(axis=1); pred_th_std = sa_th.std(axis=1)

    fig, axes = plt.subplots(2, d, figsize=(2.6 * d, 5.2), sharex='col')
    for i in range(d):
        ax = axes[0, i]
        ax.errorbar(tr_r[:, i], pred_r_mean[:, i], yerr=pred_r_std[:, i],
                    fmt='o', ms=2.5, alpha=0.45, capsize=1.5,
                    color='darkorange', ecolor='peachpuff')
        lo, hi = tr_r[:, i].min(), tr_r[:, i].max()
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.6)
        ax.set_title(f"r_{i}", fontsize=9); ax.grid(True, alpha=0.2)

        ax = axes[1, i]
        ax.errorbar(tr_th[:, i], pred_th_mean[:, i], yerr=pred_th_std[:, i],
                    fmt='o', ms=2.5, alpha=0.45, capsize=1.5,
                    color='seagreen', ecolor='palegreen')
        lo, hi = tr_th[:, i].min(), tr_th[:, i].max()
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.6)
        ax.set_title(f"θ_{i}", fontsize=9); ax.grid(True, alpha=0.2)
        axes[1, i].set_xlabel("true", fontsize=8)
    axes[0, 0].set_ylabel("predicted r", fontsize=8)
    axes[1, 0].set_ylabel("predicted θ", fontsize=8)
    fig.suptitle("Polar (r, θ) reconstruction per axis", fontsize=11)
    plt.tight_layout()
    plt.savefig(result_dir / "prediction_vs_truth_rtheta.png", dpi=200)
    plt.close()


def plot_coverage(nominal, empirical, theta_dim, result_dir):
    d = theta_dim // 2
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap_u = plt.cm.Blues(np.linspace(0.4, 0.9, d))
    cmap_v = plt.cm.Reds(np.linspace(0.4, 0.9, d))
    for i in range(theta_dim):
        label = f"u_{i}" if i < d else f"v_{i - d}"
        color = cmap_u[i] if i < d else cmap_v[i - d]
        ax.plot(nominal, empirical[:, i], lw=1.0, alpha=0.85,
                color=color, label=label)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label="ideal")
    ax.set_xlabel("nominal credible level")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Marginal calibration (per component)")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.25); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(result_dir / "coverage.png", dpi=200)
    plt.close()


def plot_nll_histogram(nll, result_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(nll, bins=30, edgecolor='black', alpha=0.75, color='steelblue')
    ax.axvline(nll.mean(), color='red', lw=1.5,
               label=f"mean = {nll.mean():.3f}")
    ax.axvline(np.median(nll), color='orange', lw=1.5, linestyle="--",
               label=f"median = {np.median(nll):.3f}")
    ax.set_xlabel("NLL of true θ"); ax.set_ylabel("count")
    ax.set_title("Per-tiling NLL distribution"); ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(result_dir / "nll_histogram.png", dpi=200)
    plt.close()


def plot_posterior_example(samples_one, truth_one, idx_label, expected_dim,
                           result_dir,
                           credible_levels=(0.50, 0.68, 0.90, 0.95),
                           show_scatter=True):
    """Posterior over each (u_i, v_i) pair as HPD contours from a 2D KDE.

    Each contour at credible level L is the iso-density boundary of the
    smallest region containing L*100% of the posterior samples
    (highest posterior density region). Density is estimated by Gaussian
    KDE with Scott's bandwidth.

    If show_scatter=True, a very faint scatter of the raw samples is
    drawn beneath the contours so any out-of-bulk artifacts (e.g.
    spline-boundary clusters) remain visible.
    """
    d = expected_dim
    u_s = samples_one[:, :d]
    v_s = samples_one[:, d:]
    u_t = truth_one[:d]
    v_t = truth_one[d:]

    # Sort credible levels descending so the outermost contour is drawn
    # first; darker line for the innermost HPD region.
    credible_levels = tuple(sorted(credible_levels, reverse=True))
    cmap = plt.cm.Blues
    colors = cmap(np.linspace(0.95, 0.45, len(credible_levels)))

    fig, axes = plt.subplots(1, d, figsize=(2.7 * d, 3.0))
    if d == 1:
        axes = [axes]

    for i in range(d):
        ax = axes[i]
        data = np.vstack([u_s[:, i], v_s[:, i]])

        if show_scatter:
            ax.scatter(u_s[:, i], v_s[:, i], s=1, alpha=0.07,
                       color='steelblue', rasterized=True, zorder=1)

        try:
            kde = gaussian_kde(data)
            dens_at_samples = kde(data)
            sorted_dens = np.sort(dens_at_samples)[::-1]
            n_s = len(sorted_dens)

            # HPD threshold for each credible level: the density of the
            # k-th most concentrated sample, where k = floor(level * n).
            thresholds = []
            for level in credible_levels:
                k = max(1, min(int(n_s * level), n_s))
                thresholds.append(sorted_dens[k - 1])

            # Evaluation grid spans samples + truth, with a small margin.
            margin = 0.3
            x_min = min(u_s[:, i].min(), u_t[i]) - margin
            x_max = max(u_s[:, i].max(), u_t[i]) + margin
            y_min = min(v_s[:, i].min(), v_t[i]) - margin
            y_max = max(v_s[:, i].max(), v_t[i]) + margin
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

            for level, thr, color in zip(credible_levels, thresholds,
                                          colors):
                ax.contour(xx, yy, zz, levels=[thr], colors=[color],
                           linewidths=1.6, zorder=5)
            for level, color in zip(credible_levels, colors):
                ax.plot([], [], color=color, lw=1.6,
                        label=f"{int(level * 100)}% HPD")
        except Exception:
            ax.scatter(u_s[:, i], v_s[:, i], s=2, alpha=0.2,
                       color='steelblue')

        ax.scatter(u_t[i], v_t[i], marker='x', color='red', s=80, lw=2,
                   label='true', zorder=10)
        ax.set_xlabel(f"u_{i}")
        ax.set_ylabel(f"v_{i}")
        ax.grid(True, alpha=0.25)
        ax.axhline(0, color='k', lw=0.5, alpha=0.4)

        if i == 0:
            ax.legend(fontsize=7, loc='best')

    fig.suptitle(f"Posterior over (u_i, v_i) — tiling #{idx_label}",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(result_dir / f"posterior_tiling_{idx_label}.png", dpi=200)
    plt.close()


# ----------------------------------------------------------------------------
# Cross-noise plots
# ----------------------------------------------------------------------------

def plot_coverage_overlay(coverage_by_level, result_dir):
    """Mean-component calibration curve, one per noise level."""
    levels = sorted(coverage_by_level.keys())
    cmap = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(levels), 1)))
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for col, level in zip(cmap, levels):
        nominal, emp = coverage_by_level[level]
        mean_emp = emp.mean(axis=1)
        ax.plot(nominal, mean_emp, lw=1.6, color=col,
                label=f"noise = {level:.2f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.7, label="ideal")
    ax.set_xlabel("nominal credible level")
    ax.set_ylabel("empirical coverage (mean over components)")
    ax.set_title("Calibration overlay across noise levels")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(result_dir / "coverage_overlay.png", dpi=200)
    plt.close()


def plot_coverage_at_levels_vs_noise(summary_rows, result_dir):
    """For each canonical credible level (50/68/90/95), plot mean coverage
    against noise."""
    levels = [row["noise_level"] for row in summary_rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    styles = [("50", 'o-', '#1f77b4'),
              ("68", 's-', '#ff7f0e'),
              ("90", '^-', '#2ca02c'),
              ("95", 'd-', '#d62728')]
    for key, fmt, color in styles:
        ys = [row[f"coverage_{key}"] for row in summary_rows]
        ax.plot(levels, ys, fmt, color=color, lw=1.6, ms=6,
                label=f"empirical @ {key}%")
        ax.axhline(float(key) / 100.0, color=color, ls=":", lw=0.8,
                   alpha=0.55)
    ax.set_xlabel("noise level (drop = insert rate)")
    ax.set_ylabel("empirical coverage (mean over components)")
    ax.set_title("Coverage degradation with noise")
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(result_dir / "coverage_at_levels_vs_noise.png", dpi=200)
    plt.close()


def plot_metrics_vs_noise(summary_rows, result_dir):
    levels = [row["noise_level"] for row in summary_rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(levels, [row["mean_rmse"] for row in summary_rows],
                 'o-', color='steelblue', lw=1.6, ms=6)
    axes[0].set_title("Mean RMSE per component vs noise")
    axes[0].set_xlabel("noise level"); axes[0].set_ylabel("RMSE")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(levels, [row["mean_posterior_std"] for row in summary_rows],
                 'o-', color='darkorange', lw=1.6, ms=6)
    axes[1].set_title("Mean posterior std vs noise")
    axes[1].set_xlabel("noise level"); axes[1].set_ylabel("posterior std")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(levels, [row["mean_nll"] for row in summary_rows],
                 'o-', color='seagreen', lw=1.6, ms=6,
                 label="mean NLL")
    axes[2].plot(levels, [row["median_nll"] for row in summary_rows],
                 's--', color='seagreen', lw=1.2, ms=5, alpha=0.7,
                 label="median NLL")
    axes[2].set_title("NLL vs noise")
    axes[2].set_xlabel("noise level"); axes[2].set_ylabel("NLL of true θ")
    axes[2].grid(True, alpha=0.25); axes[2].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(result_dir / "metrics_vs_noise.png", dpi=200)
    plt.close()


# ----------------------------------------------------------------------------
# One-noise-level pipeline
# ----------------------------------------------------------------------------

def evaluate_at_noise_level(model, tilings, params, level, indices,
                             n_samples, n_examples, expected_dim,
                             theta_dim, device, seed, sub_dir):
    from utils.npe_diffraction_dataset import NPEDiffraction2D as NPEDataset

    ds_cfg = dict(params["dataset_cfg"])
    eval_ds = NPEDataset(
        tilings,
        expected_dim=expected_dim,
        n_min=ds_cfg.get("n_min", 1024),
        n_max=ds_cfg.get("n_max", 2048),
        circle_frac=ds_cfg.get("circle_frac", 0.95),
        normalize=ds_cfg.get("normalize", True),
        canonicalize=bool(params.get("canonicalize", True)),
        compute_hist=ds_cfg.get("compute_hist", True),
        hist_n_bins=ds_cfg.get("hist_n_bins", 64),
        hist_min=ds_cfg.get("hist_min", 0.0),
        hist_max=ds_cfg.get("hist_max", 5.0),
        # exact noise for this evaluation run
        exact_drop_rate=level,
        exact_insert_rate=level,
        seed=seed,
    )

    use_edge_hist = params["model_cfg"].get("use_edge_hist", True)
    samples, truth, log_nn, nll = draw_posterior_samples(
        model, eval_ds, indices, n_samples, device,
        use_edge_hist=use_edge_hist)

    sub_dir.mkdir(parents=True, exist_ok=True)

    # Metrics + coverage
    metrics = compute_metrics(samples, truth)
    nominal, empirical = compute_coverage_curve(samples, truth)

    # TSV outputs
    write_metrics_tsv(metrics, theta_dim, sub_dir)
    write_per_sample_tsv(samples, truth, log_nn, nll, indices, sub_dir)
    write_coverage_curves_tsv(nominal, empirical, theta_dim, sub_dir)

    # Plots
    plot_coverage(nominal, empirical, theta_dim, sub_dir)
    plot_prediction_vs_truth_uv(samples, truth, theta_dim, sub_dir)
    plot_prediction_vs_truth_rtheta(samples, truth, expected_dim, sub_dir)
    plot_nll_histogram(nll, sub_dir)

    # Posterior examples
    (sub_dir / "posterior_examples").mkdir(exist_ok=True)
    n_ex = min(n_examples, len(truth))
    if n_ex > 0:
        rng = np.random.default_rng(seed)
        ex_pos = sorted(rng.choice(len(truth), size=n_ex,
                                   replace=False).tolist())
        for k in ex_pos:
            plot_posterior_example(samples[k], truth[k], indices[k],
                                   expected_dim,
                                   sub_dir / "posterior_examples")

    # Raw outputs for replotting
    np.savez_compressed(
        sub_dir / "raw_outputs.npz",
        samples=samples, truth=truth, log_mean_nn=log_nn, nll=nll,
        indices=np.array(indices),
        nominal=nominal, empirical=empirical,
        noise_level=float(level))

    # Summary row for this level
    summary_row = dict(
        noise_level=float(level),
        drop_rate=float(level),
        insert_rate=float(level),
        mean_nll=float(nll.mean()),
        median_nll=float(np.median(nll)),
        mean_rmse=float(metrics["rmse"].mean()),
        mean_mae=float(metrics["mae"].mean()),
        mean_posterior_std=float(metrics["sharpness"].mean()),
        coverage_50=float(metrics["coverage"][0.50].mean()),
        coverage_68=float(metrics["coverage"][0.68].mean()),
        coverage_90=float(metrics["coverage"][0.90].mean()),
        coverage_95=float(metrics["coverage"][0.95].mean()),
    )
    return summary_row, (nominal, empirical)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, default="results/npe_2d/npe_2d_thesis")
    p.add_argument("--weights", type=str, default="weights_best.pt")
    p.add_argument("--params", type=str, default="parameters.txt")
    p.add_argument("--eval_label", type=str, default="")
    p.add_argument("--dataset_dir", type=str, default="dataset/npe_2d_dim=5")
    p.add_argument("--split", type=str, default="validation",
                   choices=["training", "validation"])
    p.add_argument("--n_eval", type=int, default=200)
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--n_examples", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--noise_levels", type=str,
        default="0.0,0.05,0.10,0.15,0.20,0.25,0.30",
        help="Comma-separated list of exact noise levels. Each value is "
             "applied to BOTH drop and insert rate.")
    args = p.parse_args()

    root = find_root("cut-and-project-ML")
    for sp in (str(root), str(root / "models"), str(root / "source"),
               str(root / "utils")):
        if Path(sp).exists() and sp not in sys.path:
            sys.path.insert(0, sp)

    device = pick_device()
    print(f"[INFO] Device: {device}")

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    weights_path = run_dir / args.weights
    params_path = run_dir / args.params

    params = load_parameters(params_path)
    model, expected_dim, theta_dim = build_model_from_params(
        params, weights_path, device)
    print(f"[INFO] Loaded model with theta_dim={theta_dim}, "
          f"use_edge_hist="
          f"{params['model_cfg'].get('use_edge_hist', True)}")

    dataset_dir = root / params.get("dataset_dir", args.dataset_dir)
    if not dataset_dir.is_dir():
        dataset_dir = root / args.dataset_dir
    tilings = load_tilings(dataset_dir / args.split)

    from utils.npe_diffraction_dataset import NPEDiffraction2D as NPEDataset

    ds_cfg = dict(params["dataset_cfg"])
    probe_ds = NPEDataset(
        tilings, expected_dim=expected_dim,
        n_min=ds_cfg.get("n_min", 1024),
        n_max=ds_cfg.get("n_max", 2048),
        circle_frac=ds_cfg.get("circle_frac", 0.95),
        normalize=ds_cfg.get("normalize", True),
        canonicalize=bool(params.get("canonicalize", True)),
        compute_hist=False,
        hist_n_bins=ds_cfg.get("hist_n_bins", 64),
        seed=args.seed,
    )
    full_n = len(probe_ds)
    print(f"[INFO] Eval split size: {full_n}")
    del probe_ds

    n_eval = args.n_eval if args.n_eval > 0 else full_n
    n_eval = min(n_eval, full_n)
    rng = np.random.default_rng(args.seed)
    if n_eval < full_n:
        indices = sorted(rng.choice(full_n, size=n_eval,
                                    replace=False).tolist())
    else:
        indices = list(range(full_n))
    print(f"[INFO] Evaluating {n_eval} samples per noise level.")

    # Parse noise levels
    noise_levels = parse_noise_levels(args.noise_levels)
    print(f"[INFO] Noise levels: {noise_levels}")

    # Output root
    eval_root = run_dir / "evaluation"
    if args.eval_label:
        result_dir = eval_root / args.eval_label
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_dir = eval_root / f"npe_eval_{ts}"
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results: {result_dir}")

    # ----- Loop over noise levels -----
    summary_rows = []
    coverage_by_level = {}

    for level in noise_levels:
        sub_label = f"noise_{level:.3f}".rstrip("0").rstrip(".")
        # ensure unique non-empty label even for level=0
        if not sub_label.startswith("noise_"):
            sub_label = f"noise_{level:.3f}"
        sub_dir = result_dir / sub_label
        print(f"\n[INFO] === Noise level {level:.3f} -> {sub_label} ===")

        summary_row, cov = evaluate_at_noise_level(
            model, tilings, params, level, indices,
            args.n_samples, args.n_examples,
            expected_dim, theta_dim, device,
            seed=args.seed,
            sub_dir=sub_dir,
        )
        summary_rows.append(summary_row)
        coverage_by_level[level] = cov

        print(f"    mean NLL: {summary_row['mean_nll']:.4f}  "
              f"mean RMSE: {summary_row['mean_rmse']:.4f}  "
              f"cov@68: {summary_row['coverage_68']:.3f}  "
              f"cov@90: {summary_row['coverage_90']:.3f}")

    # ----- Cross-noise outputs -----
    write_summary_tsv(summary_rows, result_dir / "summary_across_noise.tsv")
    plot_coverage_overlay(coverage_by_level, result_dir)
    plot_coverage_at_levels_vs_noise(summary_rows, result_dir)
    plot_metrics_vs_noise(summary_rows, result_dir)

    cfg = dict(
        run_dir=str(run_dir), weights=str(weights_path),
        params=str(params_path), split=args.split,
        n_eval=n_eval, n_samples=args.n_samples,
        n_examples=int(args.n_examples), seed=args.seed,
        theta_dim=theta_dim, expected_dim=expected_dim,
        noise_levels=noise_levels,
        timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        n_noise_levels_evaluated=len(noise_levels),
    )
    with open(result_dir / "eval_config.txt", "w") as f:
        pprint.pprint(cfg, stream=f)

    print(f"\n[OK] Swept {len(noise_levels)} noise levels.")
    print(f"[OK] Summary: {result_dir / 'summary_across_noise.tsv'}")
    print(f"[OK] All outputs: {result_dir}")


if __name__ == "__main__":
    main()