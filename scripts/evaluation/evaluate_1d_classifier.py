from __future__ import annotations

import argparse
import ast
import sys
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

def find_root(root_name: str = "cut-and-project-ML") -> Path:
    cwd = Path.cwd()
    for parent in (cwd, *cwd.parents):
        if parent.name == root_name:
            return parent
    raise RuntimeError(f"Repo root '{root_name}' not found from {cwd}")


_root = find_root()
_source = _root / "source"
if str(_source) not in sys.path:
    sys.path.insert(0, str(_source))
print("cwd:", Path.cwd())
print("root:", _root)
print("source:", _source)
print("exists:", _source.exists())
print("sys.path[0]:", sys.path[0])

from models.classifier_1d import Conv1DClassifier
from simulations.quasi_crystal import quasi_crystal
from simulations.gaussian_step_1d import gaussian_step_1d
from simulations.random_step_1d import random_step_1d
from utils.permutation import permutation_1d


# ---------------------------------------------------------------------
# Data-type registry (must match training)
# ---------------------------------------------------------------------

DATA_TYPES = [
    ("crystal_d1",   "Crystal (d=1)",   0, "Periodic"),
    ("crystal_d2",   "Crystal (d=2)",   0, "Periodic"),
    ("qc",           "Quasi-Crystal",   1, "Quasiperiodic"),
    ("perm1024",     "Perm 1024",       2, "Aperiodic"),
    ("perm2048",     "Perm 2048",       2, "Aperiodic"),
    ("gaussian",     "Gaussian",        2, "Aperiodic"),
    ("random_step",  "Random step",     2, "Aperiodic"),
    ("qc_freq",      "QC freq. sample", 2, "Aperiodic"),
]

SUBCLASS_NAMES = [dt[1] for dt in DATA_TYPES]
NUM_SUBCLASSES = len(SUBCLASS_NAMES)
_GROUPS = sorted({(dt[2], dt[3]) for dt in DATA_TYPES})
GROUP_NAMES = [name for _, name in _GROUPS]
NUM_GROUPS = len(GROUP_NAMES)

GROUP_TO_SUBCLASSES: dict[int, list[int]] = {}
for i, dt in enumerate(DATA_TYPES):
    GROUP_TO_SUBCLASSES.setdefault(dt[2], []).append(i)


# ---------------------------------------------------------------------
# Sequence helpers (must match training)
# ---------------------------------------------------------------------

def spacings_to_positions(spacings):
    return np.concatenate([[0], np.cumsum(spacings)])


def apply_poisson_to_positions(positions, poisson_ratio):
    if poisson_ratio <= 0:
        return positions
    start, end = positions[0], positions[-1]
    span = end - start
    if span <= 0:
        return positions
    rate = (len(positions) * poisson_ratio) / span
    value, extra = start, []
    while value < end:
        value += np.random.exponential(1.0 / rate)
        if value < end:
            extra.append(value)
    if extra:
        positions = np.sort(np.append(positions, extra))
    return positions


def apply_dropout_to_positions(positions, dropout_ratio):
    if dropout_ratio <= 0:
        return positions
    mask = np.random.uniform(0, 1, len(positions)) > dropout_ratio
    mask[0] = True
    return positions[mask]


def normalize_sequence(seq):
    avg = np.mean(seq)
    return seq / avg if not np.isclose(avg, 0.0) else seq


def normalize_signed_sequence(seq):
    scale = np.mean(np.abs(seq))
    return seq / scale if not np.isclose(scale, 0.0) else seq


def apply_spacing_noise(spacings, seq_len, gauss, poiss, drop,
                        permute_pairs=None):
    pos = spacings_to_positions(spacings)
    pos = apply_poisson_to_positions(pos, poiss)
    pos = apply_dropout_to_positions(pos, drop)
    gaps = np.diff(pos)
    if len(gaps) < seq_len:
        return None
    gaps = gaps[:seq_len]
    if permute_pairs is not None:
        gaps = permutation_1d(gaps, number_of_pairs=permute_pairs)
    avg = np.mean(gaps)
    gaps = gaps + np.random.normal(0, avg * gauss, len(gaps))
    return normalize_sequence(gaps)


def apply_step_noise(steps, seq_len, gauss, poiss, drop):
    positions = np.concatenate([[0], np.cumsum(steps)])
    n = len(positions)
    if poiss > 0:
        n_extra = np.random.poisson(n * poiss)
        if n_extra > 0:
            t_extra = np.sort(np.random.uniform(0, n - 1, size=n_extra))
            t_orig = np.arange(n, dtype=float)
            pos_extra = np.interp(t_extra, t_orig, positions)
            t_all = np.concatenate([t_orig, t_extra])
            pos_all = np.concatenate([positions, pos_extra])
            order = np.argsort(t_all, kind="stable")
            positions = pos_all[order]
    if drop > 0:
        mask = np.random.uniform(0, 1, len(positions)) > drop
        mask[0] = True
        positions = positions[mask]
    steps = np.diff(positions)
    if len(steps) < seq_len:
        return None
    steps = steps[:seq_len]
    avg = np.mean(np.abs(steps))
    steps = steps + np.random.normal(0, avg * gauss, len(steps))
    return normalize_signed_sequence(steps)


def n_gen_for(seq_len, dropout_ratio):
    return int(seq_len / max(1e-6, 1.0 - dropout_ratio)) + 200


# ---------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------

def make_generators(seq_len: int):
    """Same generators as the training script, but taking (g, p, d) as
    positional args rather than a config dict (evaluation uses exact
    noise values, not sampled ranges)."""
    QC_KWARGS = dict(poisson_ratio=0, gaussian_ratio=0, dropout=0)

    def _qc_safe(**kw):
        try:
            return quasi_crystal(**kw)
        except (AssertionError, ValueError, IndexError):
            return None

    def gen_crystal(decimals):
        def _gen(g, p, d):
            slope = np.random.uniform(0.5, 5.0)
            slope = np.trunc(slope * (10 ** decimals)) / (10 ** decimals)
            r = _qc_safe(slope=slope, number_of_points=n_gen_for(seq_len, d),
                         **QC_KWARGS)
            if r is None:
                return None
            return apply_spacing_noise(r[0], seq_len, g, p, d)
        return _gen

    def gen_qc(g, p, d):
        r = _qc_safe(number_of_points=n_gen_for(seq_len, d), **QC_KWARGS)
        if r is None:
            return None
        return apply_spacing_noise(r[0], seq_len, g, p, d)

    def gen_perm(n_pairs):
        def _gen(g, p, d):
            r = _qc_safe(number_of_points=n_gen_for(seq_len, d), **QC_KWARGS)
            if r is None:
                return None
            return apply_spacing_noise(r[0], seq_len, g, p, d,
                                       permute_pairs=n_pairs)
        return _gen

    def gen_gaussian(g, p, d):
        std_dev = np.random.uniform(0.05, 0.5)
        steps = gaussian_step_1d(step_size=1, std_dev=std_dev,
                                 number_of_points=n_gen_for(seq_len, d))
        return apply_step_noise(steps, seq_len, g, p, d)

    def gen_random(g, p, d):
        pos_prob = np.random.uniform(0.2, 0.5)
        steps = random_step_1d(step_size=1, positive_probability=pos_prob,
                               number_of_points=n_gen_for(seq_len, d))
        return apply_step_noise(steps, seq_len, g, p, d)

    def gen_qc_freq(g, p, d):
        n_pts = n_gen_for(seq_len, d)
        r = _qc_safe(number_of_points=n_pts, **QC_KWARGS)
        if r is None:
            return None
        clean_sp = r[0]
        tiles = r[8]
        if tiles is not None and len(tiles) > 0:
            tile_idx = np.argmin(
                np.abs(clean_sp[:n_pts, None] - tiles[None, :]), axis=1,
            )
            unique, counts = np.unique(tile_idx, return_counts=True)
            probs = np.zeros(len(tiles))
            for i, c in zip(unique, counts):
                probs[i] = c
            probs = probs / probs.sum()
            sampled = np.random.choice(tiles, size=n_pts, p=probs)
        else:
            sampled = np.random.permutation(clean_sp[:n_pts])
        return apply_spacing_noise(sampled, seq_len, g, p, d)

    return {
        "crystal_d1":  gen_crystal(1),
        "crystal_d2":  gen_crystal(2),
        "qc":          gen_qc,
        "perm1024":    gen_perm(1024),
        "perm2048":    gen_perm(2048),
        "gaussian":    gen_gaussian,
        "random_step": gen_random,
        "qc_freq":     gen_qc_freq,
    }


# ---------------------------------------------------------------------
# Model reconstruction from parameters.txt
# ---------------------------------------------------------------------

def load_parameters(params_path: Path) -> dict:
    """Read a parameters.txt written by pprint.pprint."""
    return ast.literal_eval(params_path.read_text())


def build_model_from_params(params: dict, weights_path: Path,
                            device: str) -> Conv1DClassifier:
    """Reconstruct a Conv1DClassifier from a saved parameters.txt.

    Missing kwargs fall through to the class defaults, which matches
    the behaviour of the older training script (which built the model
    from a 5-key model_cfg without passing padding / pooling_channels).
    """
    model_cfg = dict(params["model_cfg"])
    model = Conv1DClassifier(**model_cfg).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------
# Dataset for one condition
# ---------------------------------------------------------------------

def split_group_count(total: int, k: int) -> list[int]:
    base = total // k
    rem = total % k
    return [base + (1 if i < rem else 0) for i in range(k)]


def build_condition_dataset(seq_len, gauss, poiss, drop, samples_per_group,
                            generators):
    """Generate ``samples_per_group`` sequences per group (equally split
    across subclasses) at exact noise levels (g, p, d).

    Returns
    -------
    X          : torch.FloatTensor of shape (N, 1, seq_len)
    grp_labels : np.int64 array of shape (N,)
    sub_labels : np.int64 array of shape (N,)
    """
    X, grp_labels, sub_labels = [], [], []
    for group_idx, sc_idxs in GROUP_TO_SUBCLASSES.items():
        counts = split_group_count(samples_per_group, len(sc_idxs))
        for sc_idx, n_samples in zip(sc_idxs, counts):
            arg_key = DATA_TYPES[sc_idx][0]
            gen = generators[arg_key]
            done = 0
            while done < n_samples:
                seq = gen(gauss, poiss, drop)
                if seq is None:
                    continue
                X.append(seq)
                grp_labels.append(group_idx)
                sub_labels.append(sc_idx)
                done += 1
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(1)
    return X, np.array(grp_labels), np.array(sub_labels)


# ---------------------------------------------------------------------
# Inference & metrics
# ---------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, X, device, batch_size=64):
    model.eval()
    preds_list, probs_list = [], []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        preds_list.append(preds.cpu().numpy())
        probs_list.append(probs.cpu().numpy())
    return (np.concatenate(preds_list),
            np.concatenate(probs_list, axis=0))


def subclass_preds_to_group_preds(subclass_preds):
    """Map raw subclass predictions to their group. Predictions
    that fall on the reserved 'Unknown' class map to -1."""
    out = np.full_like(subclass_preds, fill_value=-1)
    for i, dt in enumerate(DATA_TYPES):
        out[subclass_preds == i] = dt[2]
    return out


def compute_metrics(y_true_grp, y_pred_grp, qc_group=1):
    valid = (y_pred_grp >= 0)
    if valid.any():
        acc = 100.0 * float(np.mean(y_true_grp[valid] == y_pred_grp[valid]))
    else:
        acc = 0.0
    qc_t = y_true_grp == qc_group
    qc_p = y_pred_grp == qc_group
    tp = int((qc_t & qc_p).sum())
    fp = int((~qc_t & qc_p).sum())
    fn = int((qc_t & ~qc_p).sum())
    rec = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pre = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0
    return dict(acc=acc, qc_precision=pre, qc_recall=rec, qc_f1=f1)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def _noise_subtitle(g, d, p):
    parts = []
    if g > 0: parts.append(f"gauss={g}")
    if d > 0: parts.append(f"drop={d}")
    if p > 0: parts.append(f"poiss={p}")
    return "clean" if not parts else "  ".join(parts)


def plot_group_cm(y_true, y_pred, gauss, drop, poiss, save_path):
    valid = (y_pred >= 0)
    cm = confusion_matrix(y_true[valid], y_pred[valid],
                          labels=list(range(NUM_GROUPS)))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=GROUP_NAMES, yticklabels=GROUP_NAMES,
                vmin=0, vmax=1, ax=ax)
    ax.set_xlabel("Predicted Group"); ax.set_ylabel("True Group")
    ax.set_title(f"Group Confusion Matrix\n{_noise_subtitle(gauss, drop, poiss)}")
    plt.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)


def plot_subclass_cm(y_pred, sub_labels, gauss, drop, poiss, save_path):
    """Rows are subclasses, columns are groups. Predictions on the
    reserved Unknown class are simply not counted anywhere.
    """
    cm = np.zeros((NUM_SUBCLASSES, NUM_GROUPS), dtype=int)
    for sc in range(NUM_SUBCLASSES):
        mask = sub_labels == sc
        if not mask.any():
            continue
        for g in range(NUM_GROUPS):
            cm[sc, g] = int((y_pred[mask] == g).sum())
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=GROUP_NAMES, yticklabels=SUBCLASS_NAMES,
                vmin=0, vmax=1, ax=ax)
    ax.set_xlabel("Predicted Group"); ax.set_ylabel("True Subclass")
    ax.set_title(f"Subclass -> Group Confusion (norm. per subclass)\n"
                 f"{_noise_subtitle(gauss, drop, poiss)}")
    plt.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="results/classifier_1d/classifier_1d_thesis",
                        help="Training run directory containing "
                             "parameters.txt and best_weights.pt.")
    parser.add_argument("--weights", type=str, default="best_weights.pt")
    parser.add_argument("--params", type=str, default="model.txt")
    parser.add_argument("--samples", type=int, default=1024,
                        help="Samples per group per condition.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gauss_levels", type=float, nargs="+", default=[0.0, 0.1],
        help="Space-separated list of gaussian noise levels to sweep.",
    )
    parser.add_argument(
        "--drop_levels", type=float, nargs="+", default=[0.0, 0.05],
    )
    parser.add_argument(
        "--poiss_levels", type=float, nargs="+", default=[0.0, 0.05],
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = _root / run_dir
    weights_path = run_dir / args.weights
    params_path = run_dir / args.params
    eval_dir = run_dir / "Evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"[INFO] device       : {device}")
    print(f"[INFO] run dir      : {run_dir}")
    print(f"[INFO] weights      : {weights_path}")
    print(f"[INFO] parameters   : {params_path}")
    print(f"[INFO] output dir   : {eval_dir}")

    params = load_parameters(params_path)
    seq_len = int(params.get("seq_len", params["model_cfg"]["seq_len"]))

    model = build_model_from_params(params, weights_path, device)
    print(f"[INFO] model_cfg    : {params['model_cfg']}\n")

    generators = make_generators(seq_len)
    conditions = list(product(args.gauss_levels, args.drop_levels,
                              args.poiss_levels))
    print(f"[INFO] sweeping {len(conditions)} noise conditions\n")

    all_metrics = []
    tsv_rows = []

    for idx, (gauss, drop, poiss) in enumerate(conditions, start=1):
        tag = f"g{gauss:.2f}_d{drop:.2f}_p{poiss:.2f}".replace(".", "p")
        label = f"gauss={gauss}  drop={drop}  poiss={poiss}"
        print(f"-- Condition {idx}/{len(conditions)}  [{label}]")

        X, grp_labels, sub_labels = build_condition_dataset(
            seq_len, gauss, poiss, drop, args.samples, generators,
        )
        print(f"   generated {len(X)} samples")

        preds, probs = run_inference(model, X, device,
                                      batch_size=args.batch_size)
        grp_preds = subclass_preds_to_group_preds(preds)

        m = compute_metrics(grp_labels, grp_preds)
        m.update(gauss=gauss, drop=drop, poiss=poiss, tag=tag, n=len(X))
        all_metrics.append(m)

        print(f"   acc={m['acc']:.2f}%  QC P/R/F1 = "
              f"{m['qc_precision']:.1f}% / {m['qc_recall']:.1f}% / "
              f"{m['qc_f1']:.1f}%")

        plot_group_cm(grp_labels, grp_preds, gauss, drop, poiss,
                      eval_dir / f"cm_group_{tag}.png")
        plot_subclass_cm(grp_preds, sub_labels, gauss, drop, poiss,
                         eval_dir / f"cm_subclass_{tag}.png")

        np.savez(
            eval_dir / f"predictions_{tag}.npz",
            preds=preds, group_preds=grp_preds,
            group_labels=grp_labels, sub_labels=sub_labels, probs=probs,
            gauss=np.float32(gauss), drop=np.float32(drop),
            poiss=np.float32(poiss),
        )

        tsv_rows.append([tag, gauss, drop, poiss,
                         m["acc"], m["qc_precision"], m["qc_recall"],
                         m["qc_f1"]])

    # ---- summary TSV -----------------------------------------------
    tsv_path = eval_dir / "metrics_summary.tsv"
    with open(tsv_path, "w") as f:
        f.write("tag\tgauss\tdrop\tpoiss\tacc\tqc_precision\tqc_recall\tqc_f1\n")
        for row in tsv_rows:
            f.write(
                f"{row[0]}\t{row[1]:.4f}\t{row[2]:.4f}\t{row[3]:.4f}\t"
                f"{row[4]:.4f}\t{row[5]:.4f}\t{row[6]:.4f}\t{row[7]:.4f}\n"
            )
    print(f"\n[OK] metrics_summary.tsv -> {tsv_path}")

    # ---- human summary ---------------------------------------------
    summary_path = eval_dir / "eval_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write(f"Evaluation summary -- {run_dir.name}\n")
        f.write(f"Weights  : {weights_path}\n")
        f.write(f"Params   : {params_path}\n")
        f.write(f"Samples  : {args.samples} per group per condition\n")
        f.write(f"Seq len  : {seq_len}\n")
        f.write("=" * 72 + "\n\n")
        for m in all_metrics:
            noise_str = _noise_subtitle(m["gauss"], m["drop"], m["poiss"])
            f.write(f"[{m['tag']}]  {noise_str}\n"
                    f"  n={m['n']}  acc={m['acc']:.2f}%  "
                    f"QC P/R/F1 = {m['qc_precision']:.1f}% / "
                    f"{m['qc_recall']:.1f}% / {m['qc_f1']:.1f}%\n\n")
    print(f"[OK] eval_summary.txt    -> {summary_path}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()