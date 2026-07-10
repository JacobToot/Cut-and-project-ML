"""
Training script for the 1-dimensional Sequence classifier. Will receive 8 different subclasses which are grouped together:"""

from __future__ import annotations

import argparse
import pprint
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Subset

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

from models.classifier_1d import Conv1DClassifier
from simulations.quasi_crystal import quasi_crystal
from simulations.gaussian_step_1d import gaussian_step_1d
from simulations.random_step_1d import random_step_1d
from utils.permutation import permutation_1d


# ---------------------------------------------------------------------
# Data-type registry
# ---------------------------------------------------------------------
# (arg_key, display_name, group_index, group_name)

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
UNKNOWN_CLASS = NUM_SUBCLASSES
NUM_CLASSES = NUM_SUBCLASSES + 1

SUBCLASS_TO_GROUP = {i: dt[2] for i, dt in enumerate(DATA_TYPES)}
_GROUPS = sorted({(dt[2], dt[3]) for dt in DATA_TYPES})
GROUP_NAMES = [name for _, name in _GROUPS]
NUM_GROUPS = len(GROUP_NAMES)

GROUP_TO_SUBCLASSES: dict[int, list[int]] = {}
for i, dt in enumerate(DATA_TYPES):
    GROUP_TO_SUBCLASSES.setdefault(dt[2], []).append(i)

VARIANT_NAMES = ["clean", "noisy"]
ALL_CLASS_NAMES = SUBCLASS_NAMES + ["Unknown"]


# ---------------------------------------------------------------------
# Group -> subclass count distribution
# ---------------------------------------------------------------------

def split_group_count(total: int, n_subclasses: int) -> list[int]:
    """Split ``total`` equally across ``n_subclasses``.

    Remainder goes to the earliest subclasses so counts are stable and
    reproducible.
    """
    if n_subclasses == 0 or total == 0:
        return [0] * n_subclasses
    base = total // n_subclasses
    rem = total % n_subclasses
    return [base + (1 if i < rem else 0) for i in range(n_subclasses)]


# ---------------------------------------------------------------------
# Noise config + sampling
# ---------------------------------------------------------------------

def make_noise_config(gauss_low, gauss_high, poiss_low, poiss_high,
                      drop_low, drop_high) -> dict:
    return dict(
        gauss_low=gauss_low, gauss_high=gauss_high,
        poiss_low=poiss_low, poiss_high=poiss_high,
        drop_low=drop_low, drop_high=drop_high,
    )


def sample_noise_params(cfg: dict):
    return (
        np.random.uniform(cfg["gauss_low"], cfg["gauss_high"]),
        np.random.uniform(cfg["poiss_low"], cfg["poiss_high"]),
        np.random.uniform(cfg["drop_low"], cfg["drop_high"]),
    )


# ---------------------------------------------------------------------
# Sequence helpers
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
    mask[0] = True                                  # keep origin
    return positions[mask]


def normalize_sequence(seq):
    avg = np.mean(seq)
    return seq / avg if not np.isclose(avg, 0.0) else seq


def normalize_signed_sequence(seq):
    scale = np.mean(np.abs(seq))
    return seq / scale if not np.isclose(scale, 0.0) else seq


def apply_spacing_noise(spacings, seq_len, gauss, poiss, drop,
                        permute_pairs=None):
    """Position-domain noise (poisson clutter + dropout) then gaussian
    jitter on the resulting gap sequence. Returns None if the sequence
    is too short after noise -- the caller retries."""
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
    """Signed-step variant used by gaussian_step_1d and random_step_1d.

    Interpolates the poisson insertions on the (index -> position)
    interpolant so 'extra' points fall between existing ones without
    disturbing the sign pattern.
    """
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
    """Points to generate before noise so seq_len survives dropout."""
    return int(seq_len / max(1e-6, 1.0 - dropout_ratio)) + 200


# ---------------------------------------------------------------------
# Generators (one per subclass)
# ---------------------------------------------------------------------
# Each returns a length-seq_len float32 array, or None on failure
# (the retry loop in _generate_n handles that transparently).

def make_generators(seq_len: int):
    QC_KWARGS = dict(poisson_ratio=0, gaussian_ratio=0, dropout=0)

    def _quasi_crystal_safe(**kwargs):
        """Wrap quasi_crystal to swallow _find_tiles assertion / index errors
        that can fire for some rational slope + window combinations."""
        try:
            return quasi_crystal(**kwargs)
        except (AssertionError, ValueError, IndexError):
            return None

    def gen_crystal(decimals):
        def _gen(cfg):
            g, p, d = sample_noise_params(cfg)
            slope = np.random.uniform(0.5, 5.0)
            slope = np.trunc(slope * (10 ** decimals)) / (10 ** decimals)
            result = _quasi_crystal_safe(
                slope=slope,
                number_of_points=n_gen_for(seq_len, d),
                **QC_KWARGS,
            )
            if result is None:
                return None
            return apply_spacing_noise(result[0], seq_len, g, p, d)
        return _gen

    def gen_qc(cfg):
        g, p, d = sample_noise_params(cfg)
        result = _quasi_crystal_safe(
            number_of_points=n_gen_for(seq_len, d), **QC_KWARGS,
        )
        if result is None:
            return None
        return apply_spacing_noise(result[0], seq_len, g, p, d)

    def gen_perm(n_pairs):
        def _gen(cfg):
            g, p, d = sample_noise_params(cfg)
            result = _quasi_crystal_safe(
                number_of_points=n_gen_for(seq_len, d), **QC_KWARGS,
            )
            if result is None:
                return None
            return apply_spacing_noise(result[0], seq_len, g, p, d,
                                       permute_pairs=n_pairs)
        return _gen

    def gen_gaussian(cfg):
        g, p, d = sample_noise_params(cfg)
        std_dev = np.random.uniform(0.05, 0.5)
        steps = gaussian_step_1d(
            step_size=1, std_dev=std_dev,
            number_of_points=n_gen_for(seq_len, d),
        )
        return apply_step_noise(steps, seq_len, g, p, d)

    def gen_random(cfg):
        g, p, d = sample_noise_params(cfg)
        pos_prob = np.random.uniform(0.2, 0.5)
        steps = random_step_1d(
            step_size=1, positive_probability=pos_prob,
            number_of_points=n_gen_for(seq_len, d),
        )
        return apply_step_noise(steps, seq_len, g, p, d)

    def gen_qc_freq(cfg):
        g, p, d = sample_noise_params(cfg)
        n_pts = n_gen_for(seq_len, d)
        result = _quasi_crystal_safe(number_of_points=n_pts, **QC_KWARGS)
        if result is None:
            return None
        clean_spacings = result[0]
        tiles = result[8]
        if tiles is not None and len(tiles) > 0:
            tile_idx = np.argmin(
                np.abs(clean_spacings[:n_pts, None] - tiles[None, :]),
                axis=1,
            )
            unique, counts = np.unique(tile_idx, return_counts=True)
            probs = np.zeros(len(tiles))
            for i, c in zip(unique, counts):
                probs[i] = c
            probs = probs / probs.sum()
            sampled = np.random.choice(tiles, size=n_pts, p=probs)
        else:
            sampled = np.random.permutation(clean_spacings[:n_pts])
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



class GroupedDataset:
    """In-memory dataset accumulating sequences with subclass, group, and
    variant labels.

    ``__getitem__`` returns ``(x, subclass_label)`` so it plugs into
    standard cross-entropy training against the ``subclass_labels``.
    ``group_labels`` and ``variant_labels`` are used for split
    stratification and per-slice evaluation.
    """

    def __init__(self):
        self.data = []
        self.group_labels = []
        self.subclass_labels = []
        self.variant_labels = []

    def append(self, sequence, group_label, subclass_label, variant_label):
        self.data.append(sequence)
        self.group_labels.append(group_label)
        self.subclass_labels.append(subclass_label)
        self.variant_labels.append(variant_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return (
            torch.tensor(self.data[i], dtype=torch.float32).unsqueeze(0),
            torch.tensor(self.subclass_labels[i], dtype=torch.long),
        )


def _generate_n(gen_fn, n, group_label, subclass_label, variant_label,
                dataset, class_name):
    i = 0
    while i < n:
        seq = gen_fn()
        if seq is None:
            continue
        dataset.append(seq, group_label, subclass_label, variant_label)
        i += 1
        if i % 1000 == 0 or i == n:
            print(f"  {class_name}: {i}/{n}")


def build_dataset(seq_len, subclass_counts, clean_cfg, noisy_cfg):
    """Assemble the full training dataset.

    Args
    ----
    subclass_counts : dict mapping arg_key (e.g. 'crystal_d1') to
        (n_clean, n_noisy).
    """
    generators = make_generators(seq_len)
    dataset = GroupedDataset()

    print("Building dataset")
    print(f"  clean noise config : {clean_cfg}")
    print(f"  noisy noise config : {noisy_cfg}")
    for arg_key, name, group_idx, _ in DATA_TYPES:
        nc, nn_ = subclass_counts.get(arg_key, (0, 0))
        if nc + nn_ > 0:
            print(f"  {name}: {nc + nn_} ({nc} clean + {nn_} noisy)")

    for i, (arg_key, name, group_idx, _) in enumerate(DATA_TYPES):
        nc, nn_ = subclass_counts.get(arg_key, (0, 0))
        gen = generators[arg_key]
        if nc > 0:
            print(f"Generating {name} clean ({nc}) ...")
            _generate_n(lambda: gen(clean_cfg), nc, group_idx, i, 0,
                        dataset, f"{name} clean")
        if nn_ > 0:
            print(f"Generating {name} noisy ({nn_}) ...")
            _generate_n(lambda: gen(noisy_cfg), nn_, group_idx, i, 1,
                        dataset, f"{name} noisy")

    return dataset


# ---------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------

def split_per_subclass_variant(dataset, split_ratio):
    """Stratified split preserving subclass x variant proportions."""
    sub_arr = np.array(dataset.subclass_labels)
    var_arr = np.array(dataset.variant_labels)
    train, val = [], []
    for sc in sorted(np.unique(sub_arr)):
        for v in sorted(np.unique(var_arr)):
            idx = np.where((sub_arr == sc) & (var_arr == v))[0]
            if len(idx) == 0:
                continue
            idx = np.random.permutation(idx)
            n_tr = int(len(idx) * split_ratio)
            train += idx[:n_tr].tolist()
            val += idx[n_tr:].tolist()
    np.random.shuffle(train)
    np.random.shuffle(val)
    return train, val


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def subclass_preds_to_group(subclass_preds):
    out = np.full_like(subclass_preds, fill_value=-1)
    for sc, g in SUBCLASS_TO_GROUP.items():
        out[subclass_preds == sc] = g
    return out


def subclass_probs_to_group_probs(probs):
    out = np.zeros((probs.shape[0], NUM_GROUPS), dtype=probs.dtype)
    for sc, g in SUBCLASS_TO_GROUP.items():
        out[:, g] += probs[:, sc]
    return out


def _epoch_metrics(preds, labels, qc_group=1):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    sub_acc = 100.0 * float(np.mean(preds == labels))

    grp_preds = subclass_preds_to_group(preds)
    grp_labels = np.array([SUBCLASS_TO_GROUP.get(int(l), -1) for l in labels])
    valid = (grp_preds >= 0) & (grp_labels >= 0)
    grp_acc = (100.0 * float(np.mean(grp_preds[valid] == grp_labels[valid]))
               if valid.any() else 0.0)

    tp = int(((grp_preds == qc_group) & (grp_labels == qc_group)).sum())
    qc_true = int((grp_labels == qc_group).sum())
    qc_pred = int((grp_preds == qc_group).sum())
    rec = 100.0 * tp / qc_true if qc_true > 0 else 0.0
    pre = 100.0 * tp / qc_pred if qc_pred > 0 else 0.0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0

    return dict(subclass_acc=sub_acc, group_acc=grp_acc,
                qc_precision=pre, qc_recall=rec, qc_f1=f1)


def compute_basic_metrics(y_true, y_pred, qc_group=1):
    m = _epoch_metrics(y_pred, y_true, qc_group=qc_group)
    m["n"] = len(y_true)
    return m


def print_split_metrics(name, m):
    print(f"[{name}] n={m['n']} | "
          f"sub_acc={m['subclass_acc']:.2f}% | "
          f"grp_acc={m['group_acc']:.2f}% | "
          f"QC P/R/F1={m['qc_precision']:.1f}% / "
          f"{m['qc_recall']:.1f}% / {m['qc_f1']:.1f}%")


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------

def train_classifier(model, train_loader, val_loader, device, *,
                     epochs, lr, eta_min_ratio, class_weights, result_dir):
    """Constant-LR for the first half of training, then cosine anneal
    down to ``eta_min_ratio * lr``. Saves ``best_weights.pt`` whenever
    val loss improves.
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    opt = optim.Adam(model.parameters(), lr=lr)
    t_start = epochs // 2
    eta_min = eta_min_ratio * lr
    sched = optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=t_start),
            optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=epochs - t_start, eta_min=eta_min,
            ),
        ],
        milestones=[t_start],
    )

    hist = dict(
        tr_loss=[], va_loss=[],
        tr_subclass_acc=[], va_subclass_acc=[],
        tr_group_acc=[], va_group_acc=[],
        tr_qc_recall=[], va_qc_recall=[],
        tr_qc_precision=[], va_qc_precision=[],
        tr_qc_f1=[], va_qc_f1=[],
    )
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss_sum = 0.0
        tr_preds, tr_labels = [], []
        for xb, yb in train_loader:
            xb = xb.to(device, dtype=torch.float32).contiguous()
            yb = yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            tr_loss_sum += loss.item()
            tr_preds.extend(out.argmax(1).cpu().numpy())
            tr_labels.extend(yb.cpu().numpy())
        tr_loss = tr_loss_sum / len(train_loader)
        tr_m = _epoch_metrics(tr_preds, tr_labels)

        model.eval()
        va_loss_sum = 0.0
        va_preds, va_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, dtype=torch.float32).contiguous()
                yb = yb.to(device)
                out = model(xb)
                va_loss_sum += criterion(out, yb).item()
                va_preds.extend(out.argmax(1).cpu().numpy())
                va_labels.extend(yb.cpu().numpy())
        va_loss = va_loss_sum / len(val_loader)
        va_m = _epoch_metrics(va_preds, va_labels)

        sched.step()

        hist["tr_loss"].append(tr_loss); hist["va_loss"].append(va_loss)
        hist["tr_subclass_acc"].append(tr_m["subclass_acc"])
        hist["va_subclass_acc"].append(va_m["subclass_acc"])
        hist["tr_group_acc"].append(tr_m["group_acc"])
        hist["va_group_acc"].append(va_m["group_acc"])
        hist["tr_qc_recall"].append(tr_m["qc_recall"])
        hist["va_qc_recall"].append(va_m["qc_recall"])
        hist["tr_qc_precision"].append(tr_m["qc_precision"])
        hist["va_qc_precision"].append(va_m["qc_precision"])
        hist["tr_qc_f1"].append(tr_m["qc_f1"])
        hist["va_qc_f1"].append(va_m["qc_f1"])

        cur_lr = opt.param_groups[0]["lr"]
        print(f"epoch {epoch:03d}/{epochs} | lr {cur_lr:.2e} | "
              f"loss {tr_loss:.4f}/{va_loss:.4f} | "
              f"sub_acc {tr_m['subclass_acc']:.2f}%/{va_m['subclass_acc']:.2f}% | "
              f"grp_acc {tr_m['group_acc']:.2f}%/{va_m['group_acc']:.2f}% | "
              f"QC P/R/F1 {va_m['qc_precision']:.1f}%/"
              f"{va_m['qc_recall']:.1f}%/{va_m['qc_f1']:.1f}%")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), result_dir / "best_weights.pt")

    return hist, best_val


# ---------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    preds, labels, probs = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, dtype=torch.float32).contiguous()
        yb = yb.to(device)
        logits = model(xb)
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(yb.cpu().numpy())
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return (np.array(preds), np.array(labels),
            np.concatenate(probs, axis=0))


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, class_names, save_path,
                          title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    size = max(8, len(class_names) * 1.2)
    plt.figure(figsize=(size, size))
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()


def plot_subclass_to_group_confusion(subclass_preds, subclass_labels,
                                     save_path):
    grp_preds = subclass_preds_to_group(subclass_preds)
    sub_arr = np.array(subclass_labels)
    active = sorted(np.unique(sub_arr))
    active_names = [SUBCLASS_NAMES[sc] for sc in active]

    cm = np.zeros((len(active), NUM_GROUPS), dtype=int)
    for row, sc in enumerate(active):
        mask = sub_arr == sc
        for g in range(NUM_GROUPS):
            cm[row, g] = int((grp_preds[mask] == g).sum())

    row_sums = cm.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    plt.figure(figsize=(10, max(6, len(active) * 1.5)))
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=GROUP_NAMES, yticklabels=active_names,
                vmin=0, vmax=1)
    plt.xlabel("Predicted Group"); plt.ylabel("True Subclass")
    plt.title("Subclass -> Group Confusion (normalized per subclass)")
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()


def plot_roc_curves(y_true, probs, n_classes, class_names, save_path,
                    title="ROC curves"):
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fpr, tpr, roc_auc = {}, {}, {}
    for c in range(n_classes):
        fpr[c], tpr[c], _ = roc_curve(y_bin[:, c], probs[:, c])
        roc_auc[c] = auc(fpr[c], tpr[c])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[c] for c in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(9, 7))
    plt.plot(fpr["micro"], tpr["micro"], linestyle=":", linewidth=3,
             label=f"micro-avg (AUC={roc_auc['micro']:.3f})")
    plt.plot(fpr["macro"], tpr["macro"], linestyle=":", linewidth=3,
             label=f"macro-avg (AUC={roc_auc['macro']:.3f})")
    for c in range(n_classes):
        plt.plot(fpr[c], tpr[c],
                 label=f"{class_names[c]} (AUC={roc_auc[c]:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend(loc="lower right", fontsize=7)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()


def plot_training_curves(hist, result_dir):
    ep = np.arange(1, len(hist["tr_loss"]) + 1)

    plt.figure()
    plt.plot(ep, hist["tr_loss"], label="train loss")
    plt.plot(ep, hist["va_loss"], label="val loss")
    plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("loss (log)")
    plt.legend(); plt.tight_layout()
    plt.savefig(result_dir / "loss_curve.png", dpi=200); plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(ep, hist["tr_subclass_acc"], label="train sub acc")
    plt.plot(ep, hist["va_subclass_acc"], label="val sub acc")
    plt.plot(ep, hist["tr_group_acc"], label="train grp acc", linestyle="--")
    plt.plot(ep, hist["va_group_acc"], label="val grp acc", linestyle="--")
    plt.plot(ep, hist["va_qc_recall"], label="val QC recall", linestyle=":")
    plt.plot(ep, hist["va_qc_precision"], label="val QC precision", linestyle=":")
    plt.plot(ep, hist["va_qc_f1"], label="val QC F1", linestyle="-.")
    plt.xlabel("epoch"); plt.ylabel("accuracy (%)"); plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(result_dir / "accuracy_curve.png", dpi=200); plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    # ---- data ------------------------------------------------------
    parser.add_argument("--seq_len", type=int, default=16384)
    parser.add_argument("--periodic_clean_samples",  type=int, default=16384)
    parser.add_argument("--periodic_noisy_samples",  type=int, default=16384)
    parser.add_argument("--qc_clean_samples",        type=int, default=8192)
    parser.add_argument("--qc_noisy_samples",        type=int, default=8192)
    parser.add_argument("--aperiodic_clean_samples", type=int, default=4096)
    parser.add_argument("--aperiodic_noisy_samples", type=int, default=4096)

    parser.add_argument("--clean_gauss_low",  type=float, default=0.0)
    parser.add_argument("--clean_gauss_high", type=float, default=0.0)
    parser.add_argument("--clean_poiss_low",  type=float, default=0.0)
    parser.add_argument("--clean_poiss_high", type=float, default=0.0)
    parser.add_argument("--clean_drop_low",   type=float, default=0.0)
    parser.add_argument("--clean_drop_high",  type=float, default=0.0)

    parser.add_argument("--noisy_gauss_low",  type=float, default=0.05)
    parser.add_argument("--noisy_gauss_high", type=float, default=0.1)
    parser.add_argument("--noisy_poiss_low",  type=float, default=0.025)
    parser.add_argument("--noisy_poiss_high", type=float, default=0.075)
    parser.add_argument("--noisy_drop_low",   type=float, default=0.025)
    parser.add_argument("--noisy_drop_high",  type=float, default=0.075)

    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    # ---- model (Conv1DClassifier kwargs) ---------------------------
    parser.add_argument("--kernel_size", type=int, default=17)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dropout_conv", type=float, default=0.15)
    parser.add_argument("--dropout_linear", type=float, default=0.2)
    parser.add_argument("--unknown_weight", type=float, default=0.2,
                        help="Cross-entropy weight for the unused "
                             "'Unknown' output class.")

    # ---- optimiser -------------------------------------------------
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eta_min_ratio", type=float, default=0.05)

    # ---- output ----------------------------------------------------
    parser.add_argument("--result_folder", type=str, default="classifier_1d")
    parser.add_argument(
        "--weights_name", type=str, default=None,
        help="If set, copy best_weights.pt + model.txt to "
             "<root>/weights/<weights_name>/ at end of training.",
    )
    args = parser.parse_args()

    # ---- setup -----------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    script = Path(__file__).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"classifier_1d_{timestamp}"
    result_dir = _root / "results" / args.result_folder / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] device       : {device}")
    print(f"[INFO] result dir   : {result_dir}")

    # ---- noise configs --------------------------------------------
    clean_cfg = make_noise_config(
        args.clean_gauss_low, args.clean_gauss_high,
        args.clean_poiss_low, args.clean_poiss_high,
        args.clean_drop_low, args.clean_drop_high,
    )
    noisy_cfg = make_noise_config(
        args.noisy_gauss_low, args.noisy_gauss_high,
        args.noisy_poiss_low, args.noisy_poiss_high,
        args.noisy_drop_low, args.noisy_drop_high,
    )

    # ---- subclass counts from per-group args ----------------------
    subclass_counts: dict[str, tuple[int, int]] = {}
    for group_idx, sc_idxs in GROUP_TO_SUBCLASSES.items():
        group_name = GROUP_NAMES[group_idx]
        if group_name == "Periodic":
            gc, gn = args.periodic_clean_samples, args.periodic_noisy_samples
        elif group_name == "Quasiperiodic":
            gc, gn = args.qc_clean_samples, args.qc_noisy_samples
        else:                                             # Aperiodic
            gc, gn = args.aperiodic_clean_samples, args.aperiodic_noisy_samples
        cs = split_group_count(gc, len(sc_idxs))
        ns = split_group_count(gn, len(sc_idxs))
        for sc_idx, c, n in zip(sc_idxs, cs, ns):
            arg_key = DATA_TYPES[sc_idx][0]
            subclass_counts[arg_key] = (c, n)

    # ---- build dataset --------------------------------------------
    dataset = build_dataset(args.seq_len, subclass_counts, clean_cfg, noisy_cfg)
    train_idx, val_idx = split_per_subclass_variant(dataset, args.split_ratio)

    train_loader = DataLoader(Subset(dataset, train_idx),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx),
                            batch_size=args.batch_size, shuffle=False)

    print(f"[INFO] train samples : {len(train_idx)}")
    print(f"[INFO] val samples   : {len(val_idx)}")

    # ---- model ----------------------------------------------------
    model_cfg = dict(
        seq_len=args.seq_len,
        classes=NUM_CLASSES,
        kernel_size=args.kernel_size,
        dropout_conv=args.dropout_conv,
        dropout_linear=args.dropout_linear,
    )
    model = Conv1DClassifier(**model_cfg).to(device)

    class_weights = torch.ones(NUM_CLASSES, dtype=torch.float32)
    class_weights[UNKNOWN_CLASS] = args.unknown_weight

    print(f"[INFO] classes ({NUM_CLASSES}) : {ALL_CLASS_NAMES}")
    print(f"[INFO] class weights          : {class_weights.tolist()}")
    print(f"[INFO] starting training      "
          f"(cosine anneal from epoch {args.epochs // 2}, "
          f"eta_min = {args.eta_min_ratio * args.lr:.2e})")

    # ---- train ----------------------------------------------------
    hist, best_val = train_classifier(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr,
        eta_min_ratio=args.eta_min_ratio,
        class_weights=class_weights, result_dir=result_dir,
    )

    # ---- persist per-epoch metrics --------------------------------
    ep = np.arange(1, len(hist["tr_loss"]) + 1)
    metrics = np.column_stack([
        ep,
        hist["tr_loss"], hist["va_loss"],
        hist["tr_subclass_acc"], hist["va_subclass_acc"],
        hist["tr_group_acc"], hist["va_group_acc"],
        hist["tr_qc_recall"], hist["va_qc_recall"],
        hist["tr_qc_precision"], hist["va_qc_precision"],
        hist["tr_qc_f1"], hist["va_qc_f1"],
    ])
    np.savetxt(
        result_dir / "metrics_table.tsv", metrics,
        fmt=["%d"] + ["%.8f"] * 12, delimiter="\t",
        header=("epoch\ttrain_loss\tval_loss\t"
                "train_subclass_acc\tval_subclass_acc\t"
                "train_group_acc\tval_group_acc\t"
                "train_qc_recall\tval_qc_recall\t"
                "train_qc_precision\tval_qc_precision\t"
                "train_qc_f1\tval_qc_f1"),
        comments="",
    )
    plot_training_curves(hist, result_dir)

    # ---- reload best & evaluate -----------------------------------
    best_ckpt = result_dir / "best_weights.pt"
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f"[OK] loaded best val checkpoint: {best_ckpt}")

    preds, labels, probs = collect_predictions(model, val_loader, device)
    val_variants = np.array([dataset.variant_labels[i] for i in val_idx])
    clean_mask = val_variants == 0
    noisy_mask = val_variants == 1

    clean_m = compute_basic_metrics(labels[clean_mask], preds[clean_mask])
    noisy_m = compute_basic_metrics(labels[noisy_mask], preds[noisy_mask])
    print_split_metrics("val clean", clean_m)
    print_split_metrics("val noisy", noisy_m)

    with open(result_dir / "val_variant_metrics.txt", "w") as f:
        pprint.pprint({"clean": clean_m, "noisy": noisy_m}, stream=f)

    # ---- confusion matrices ---------------------------------------
    plot_confusion_matrix(labels, preds, SUBCLASS_NAMES,
                          result_dir / "confusion_matrix_subclass.png",
                          title="Confusion Matrix (subclasses)")
    plot_confusion_matrix(
        labels[clean_mask], preds[clean_mask], SUBCLASS_NAMES,
        result_dir / "confusion_matrix_subclass_clean.png",
        title="Subclass Confusion Matrix (clean only)",
    )
    plot_confusion_matrix(
        labels[noisy_mask], preds[noisy_mask], SUBCLASS_NAMES,
        result_dir / "confusion_matrix_subclass_noisy.png",
        title="Subclass Confusion Matrix (noisy only)",
    )
    grp_preds = subclass_preds_to_group(preds)
    grp_labels = np.array([SUBCLASS_TO_GROUP.get(int(l), -1) for l in labels])
    valid = (grp_preds >= 0) & (grp_labels >= 0)
    plot_confusion_matrix(
        grp_labels[valid], grp_preds[valid], GROUP_NAMES,
        result_dir / "confusion_matrix_group.png",
        title="Group Confusion Matrix",
    )
    plot_confusion_matrix(
        grp_labels[valid & clean_mask], grp_preds[valid & clean_mask],
        GROUP_NAMES,
        result_dir / "confusion_matrix_group_clean.png",
        title="Group Confusion Matrix (clean only)",
    )
    plot_confusion_matrix(
        grp_labels[valid & noisy_mask], grp_preds[valid & noisy_mask],
        GROUP_NAMES,
        result_dir / "confusion_matrix_group_noisy.png",
        title="Group Confusion Matrix (noisy only)",
    )
    val_sub_labels = [dataset.subclass_labels[i] for i in val_idx]
    plot_subclass_to_group_confusion(
        preds, val_sub_labels,
        result_dir / "confusion_subclass_to_group.png",
    )

    # ---- ROC curves ----------------------------------------------
    probs_no_unk = probs[:, :NUM_SUBCLASSES]
    probs_no_unk = probs_no_unk / probs_no_unk.sum(axis=1, keepdims=True)
    plot_roc_curves(labels, probs_no_unk, NUM_SUBCLASSES, SUBCLASS_NAMES,
                    result_dir / "roc_curves_subclass.png",
                    title="ROC curves (subclass level)")

    grp_probs = subclass_probs_to_group_probs(probs)
    grp_probs = grp_probs / grp_probs.sum(axis=1, keepdims=True)
    plot_roc_curves(grp_labels[valid], grp_probs[valid],
                    NUM_GROUPS, GROUP_NAMES,
                    result_dir / "roc_curves_group.png",
                    title="ROC curves (group level)")

    # ---- save predictions ----------------------------------------
    np.savez(
        result_dir / "predictions.npz",
        subclass_preds=preds, subclass_labels=labels,
        group_preds=grp_preds, group_labels=grp_labels,
        subclass_probs=probs, group_probs=grp_probs,
        variant_labels=val_variants,
    )

    # ---- save model.txt (full config) -----------------------
    parameters = dict(
        script=script,
        timestamp=timestamp,
        device=device,
        seed=args.seed,
        seq_len=args.seq_len,
        best_val_loss=float(best_val),
        train_samples=len(train_idx),
        val_samples=len(val_idx),
        num_classes=NUM_CLASSES,
        num_subclasses=NUM_SUBCLASSES,
        class_names=ALL_CLASS_NAMES,
        group_names=GROUP_NAMES,
        subclass_names=SUBCLASS_NAMES,
        subclass_to_group=SUBCLASS_TO_GROUP,
        variant_names=VARIANT_NAMES,
        subclass_counts=subclass_counts,
        clean_noise_cfg=dict(clean_cfg),
        noisy_noise_cfg=dict(noisy_cfg),
        split_ratio=args.split_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        eta_min_ratio=args.eta_min_ratio,
        unknown_weight=args.unknown_weight,
        model_cfg=model_cfg,
    )
    with open(result_dir / "model.txt", "w") as f:
        pprint.pprint(parameters, stream=f)

    # ---- final full-weight save + optional weights/ copy ---------
    torch.save(model.state_dict(), result_dir / "weights.pt")
    print(f"[OK] best   -> {result_dir / 'best_weights.pt'}")
    print(f"[OK] final  -> {result_dir / 'weights.pt'}")
    print(f"[OK] params -> {result_dir / 'model.txt'}")

    if args.weights_name:
        w_dir = _root / "weights" / args.weights_name
        w_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(result_dir / "best_weights.pt", w_dir / "weights.pt")
        shutil.copyfile(result_dir / "model.txt", w_dir / "model.txt")
        print(f"[OK] weights/ copy -> {w_dir}")


if __name__ == "__main__":
    main()