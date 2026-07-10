from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import math
import pickle
import pprint
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def find_root(root: str = "cut-and-project-ML") -> Path:
    cwd = Path.cwd()
    for parent in (cwd, *cwd.parents):
        if parent.name == root:
            return parent
    raise RuntimeError(f"Specified root '{root}' not found")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # if torch.backends.mps.is_available():
    #     return "mps"
    return "cpu"


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_tilings(directory: Path) -> list[dict]:
    """Load the first .pkl file found in *directory*."""
    pkl_files = sorted(directory.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {directory}")
    path = pkl_files[0]
    print(f"[INFO] Loading {path} ...")
    with open(path, "rb") as f:
        tilings = pickle.load(f)
    print(f"[INFO] Loaded {len(tilings)} tilings from {path.name}")
    return tilings


def _apply_corruption(points, dropout, insertion, rng):
    pts = points.copy()
    if dropout > 0 and len(pts) > 1:
        keep = max(1, int(len(pts) * (1 - dropout)))
        idx = rng.choice(len(pts), size=keep, replace=False)
        pts = pts[idx]
    if insertion > 0 and len(pts) > 0:
        n_ins = max(1, int(len(points) * insertion))
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        spurious = rng.uniform(lo, hi, size=(n_ins, 2)).astype(np.float32)
        pts = np.concatenate([pts, spurious], axis=0)
    return pts


def _rotate(points, rng):
    """Random rotation about the origin (orientation augmentation)."""
    theta = rng.uniform(0.0, 2.0 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=points.dtype)
    return points @ R.T


def _circular_crop(points, n_target, circle_frac=0.95):

    """Crops data into a circular region to prevent the model from seeing a rotation. Model should be rotationally invariant."""

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


class TilingDataset(Dataset):

    """
    Creates the pattern used for training the diffraction model. Does not 
    """

    def __init__(self, tilings, augment=False, max_dropout=0.15,
                 max_insertion=0.15, n_min=512, n_max=2048,
                 rotate=True, circle_frac=0.95):
        self.tilings = tilings
        self.augment = augment
        self.max_dropout = max_dropout
        self.max_insertion = max_insertion
        self.n_min = n_min
        self.n_max = n_max
        self.rotate = rotate
        self.circle_frac = circle_frac

    def __len__(self):
        return len(self.tilings)

    def __getitem__(self, idx):
        entry = self.tilings[idx]
        points = entry["points"].copy()
        dim = entry["dimension"]

        rng = np.random.default_rng()

        if self.augment:
            drop = rng.uniform(0, self.max_dropout)
            ins = rng.uniform(0, self.max_insertion)
            points = _apply_corruption(points, drop, ins, rng)

        n_target = int(rng.integers(self.n_min, self.n_max + 1))
        cropped = _circular_crop(points, n_target, self.circle_frac)

        cropped = cropped - cropped.mean(axis=0)

        if self.rotate and len(cropped) >= 2:
            cropped = _rotate(cropped, rng)

        if len(cropped) >= 2:
            tree = cKDTree(cropped)
            dists, _ = tree.query(cropped, k=2)
            mean_nn = float(dists[:, 1].mean())
            if mean_nn > 1e-8:
                cropped = cropped / mean_nn

        return cropped, dim


def collate_point_clouds(batch):
    points_list, dims_list = zip(*batch)
    max_n = max(len(p) for p in points_list)
    B = len(points_list)

    points = torch.zeros(B, max_n, 2)
    mask = torch.zeros(B, max_n)

    for i, pts in enumerate(points_list):
        n = len(pts)
        points[i, :n] = torch.as_tensor(pts, dtype=torch.float32)
        mask[i, :n] = 1.0

    dims = torch.tensor(dims_list, dtype=torch.long)
    return points, mask, dims


def train_one_epoch(model, loader, optimizer, criterion, device,
                    grad_clip, d_min):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for points, mask, dims in loader:
        points = points.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        targets = (dims - d_min).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(points, mask)
        loss = criterion(logits, targets)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        preds = logits.argmax(dim=-1)
        total_correct += int((preds == targets).sum().item())
        total_count += int(targets.numel())
        total_loss += float(loss.item()) * int(targets.numel())

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, d_min):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for points, mask, dims in loader:
        points = points.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        targets = (dims - d_min).to(device, non_blocking=True)

        logits = model(points, mask)
        loss = criterion(logits, targets)

        preds = logits.argmax(dim=-1)
        total_correct += int((preds == targets).sum().item())
        total_count += int(targets.numel())
        total_loss += float(loss.item()) * int(targets.numel())

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def write_metrics_table(result_dir, train_losses, val_losses,
                        train_accs, val_accs):
    if not train_losses:
        return
    epochs = np.arange(1, len(train_losses) + 1)
    data = np.column_stack([epochs, train_losses, val_losses,
                            train_accs, val_accs])
    np.savetxt(
        result_dir / "metrics_table.tsv", data, fmt="%.6f", delimiter="\t",
        header="epoch\ttrain_loss\tval_loss\ttrain_acc\tval_acc", comments="")


def save_curves(result_dir, train_losses, val_losses, train_accs, val_accs):
    if not train_losses:
        return
    epochs = np.arange(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, train_losses, label="train CE")
    ax1.plot(epochs, val_losses, label="val CE")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax1.set_title("Cross-Entropy Loss"); ax1.legend(); ax1.grid(True, alpha=0.25)
    ax2.plot(epochs, train_accs, label="train acc")
    ax2.plot(epochs, val_accs, label="val acc")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(result_dir / "training_curves.png", dpi=200)
    plt.close()


def save_confusion_matrix(model, loader, device, d_min, dimensions, result_dir):
    model.eval()
    n_classes = len(dimensions)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    with torch.no_grad():
        for points, mask, dims in loader:
            points = points.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            targets = (dims - d_min).numpy()
            preds = model(points, mask).argmax(-1).cpu().numpy()
            for t, p in zip(targets, preds):
                cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels([f"d={d}" for d in dimensions])
    ax.set_yticklabels([f"d={d}" for d in dimensions])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (validation)")
    for i in range(n_classes):
        for j in range(n_classes):
            colour = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=colour, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(result_dir / "confusion_matrix.png", dpi=200)
    plt.close()


def build_scheduler(optimizer, args):
    if args.scheduler == "constant":
        return None, False
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr), False
    if args.scheduler == "cosine_warmup":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0,
            total_iters=args.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs,
            eta_min=args.min_lr)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine],
            milestones=[args.warmup_epochs]), False
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.plateau_factor,
            patience=args.plateau_patience, min_lr=args.min_lr,
            verbose=True), True
    if args.scheduler == "constant+cosine":
        constant = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=args.constant_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.constant_epochs,
            eta_min=args.min_lr)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[constant, cosine],
            milestones=[args.constant_epochs]), False
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a DIFFRACTION-ONLY baseline to classify "
                    "quasicrystal parent-lattice dimension from 2D tilings.")

    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--d_min", type=int, default=4)
    parser.add_argument("--d_max", type=int, default=9)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result_folder", type=str,
                        default="classifier_dimension_2d")

    parser.add_argument("--n_min", type=int, default=1024)
    parser.add_argument("--n_max", type=int, default=2048)

    parser.add_argument("--augment", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max_dropout", type=float, default=0.15)
    parser.add_argument("--max_insertion", type=float, default=0.0)
    parser.add_argument("--rotate", type=int, default=1, choices=[0, 1],
                        help="Random global rotation (orientation aug). "
                             "Applied to train AND val so the task is "
                             "orientation-agnostic.")
    parser.add_argument("--circle_frac", type=float, default=0.95,
                        help="Disk radius fraction of (min extent)/2; <1 keeps "
                             "the crop fully inside the data.")

    parser.add_argument("--d_summary", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--cnn_width", type=int, default=32,
                        help="Base CNN channel width; channels are "
                             "[w, 2w, 4w, 8w].")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d_node", type=int, default=128,
                        help="(ignored by the diffraction model)")
    parser.add_argument("--k", type=int, default=10,
                        help="(ignored by the diffraction model)")

    parser.add_argument("--grid_size", type=int, default=128,
                        help="Diffraction image is grid_size x grid_size.")
    parser.add_argument("--q_max_factor", type=float, default=4.0,
                        help="q-grid half-extent = q_max_factor * pi "
                             "(assumes mean-NN-distance = 1 rescaling).")
    parser.add_argument("--diff_backend", type=str, default="nufft",
                        choices=["nufft", "direct"])
    parser.add_argument("--nufft_width", type=int, default=6)
    parser.add_argument("--nufft_sigma", type=float, default=2.0)
    parser.add_argument("--suppress_dc", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dc_radius", type=int, default=1)
    parser.add_argument("--diff_log", type=int, default=1, choices=[0, 1])

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--scheduler", type=str, default="cosine_warmup",
                        choices=["cosine", "cosine_warmup", "plateau",
                                 "constant", "constant+cosine"])
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--constant_epochs", type=int, default=150)
    parser.add_argument("--plateau_patience", type=int, default=15)
    parser.add_argument("--plateau_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--retrain", type=str, default=None,
                        help="Path to a .pt checkpoint to initialise from.")

    args = parser.parse_args()

    root = find_root("cut-and-project-ML")
    models_dir = root / "source"
    for p in (str(root), str(models_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    seed_everything(args.seed)
    device = pick_device()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = (root / "results" / args.result_folder
                  / f"classifier_dimension_2d_{timestamp}")
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Results: {result_dir}")

    dimensions = list(range(args.d_min, args.d_max + 1))

    dataset_dir = root / args.dataset_dir
    train_tilings = load_tilings(dataset_dir / "training")
    val_tilings = load_tilings(dataset_dir / "validation")

    train_ds = TilingDataset(
        train_tilings, augment=bool(args.augment),
        max_dropout=args.max_dropout, max_insertion=args.max_insertion,
        n_min=args.n_min, n_max=args.n_max,
        rotate=bool(args.rotate), circle_frac=args.circle_frac)
    val_ds = TilingDataset(
        val_tilings, augment=bool(args.augment), n_min=args.n_min, n_max=args.n_max,
        rotate=bool(args.rotate), circle_frac=args.circle_frac)

    print(f"[INFO] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=collate_point_clouds, num_workers=args.num_workers,
        pin_memory=(device == "cuda"))
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        collate_fn=collate_point_clouds, num_workers=args.num_workers,
        pin_memory=(device == "cuda"))

    from models.classifier_dimension_2d import DimensionalityClassifier
    from utils.nufft2d import DiffractionConfig


    diff_cfg = DiffractionConfig(
        grid_size=args.grid_size,
        q_max=args.q_max_factor * math.pi,
        backend=args.diff_backend,
        normalize="per_atom",
        log1p=bool(args.diff_log),
        standardize=True,
        suppress_dc=bool(args.suppress_dc),
        dc_radius=args.dc_radius,
        nufft_sigma=args.nufft_sigma,
        nufft_width=args.nufft_width,
    )

    model = DimensionalityClassifier(
        d_min=args.d_min, d_max=args.d_max,
        d_summary=args.d_summary, n_layers=args.n_layers,
        cnn_width=args.cnn_width, dropout=args.dropout,
        diffraction=diff_cfg,
    ).to(device)

    if args.retrain is not None:
        state = torch.load(args.retrain, map_location=device)
        model.load_state_dict(state)
        print(f"[INFO] Loaded weights from {args.retrain}")

    n_params = count_parameters(model)
    print(f"[INFO] Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler, is_plateau = build_scheduler(optimizer, args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            args.grad_clip, args.d_min)
        va_loss, va_acc = evaluate(
            model, val_loader, criterion, device, args.d_min)

        if scheduler is not None:
            if is_plateau:
                scheduler.step(va_loss)
            else:
                scheduler.step()

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_accs.append(tr_acc);    val_accs.append(va_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:03d} | "
              f"train CE {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val CE {va_loss:.4f} acc {va_acc:.4f} | "
              f"lr {lr_now:.2e}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), result_dir / "weights_best.pt")

        write_metrics_table(result_dir, train_losses, val_losses,
                            train_accs, val_accs)
        save_curves(result_dir, train_losses, val_losses,
                    train_accs, val_accs)

    torch.save(model.state_dict(), result_dir / "weights_last.pt")

    model.load_state_dict(torch.load(result_dir / "weights_best.pt",
                                     map_location=device))
    save_confusion_matrix(model, val_loader, device, args.d_min,
                          dimensions, result_dir)

    save_dict = dict(
        script="diffraction_classification",
        timestamp=timestamp,
        device=device,
        best_val_acc=float(best_val_acc),
        train_num_samples=len(train_ds),
        val_num_samples=len(val_ds),
        dataset_dir=str(dataset_dir),
        model_cfg=dict(
            d_min=args.d_min, d_max=args.d_max,
            d_summary=args.d_summary, n_layers=args.n_layers,
            cnn_width=args.cnn_width, dropout=args.dropout),
        diffraction_cfg=dict(
            grid_size=args.grid_size,
            q_max_factor=args.q_max_factor,
            backend=args.diff_backend,
            nufft_width=args.nufft_width, nufft_sigma=args.nufft_sigma,
            suppress_dc=bool(args.suppress_dc), dc_radius=args.dc_radius,
            log1p=bool(args.diff_log), normalize="per_atom",
            standardize=True),
        dataset_cfg=dict(
            n_min=args.n_min, n_max=args.n_max,
            augment=bool(args.augment),
            max_dropout=args.max_dropout,
            max_insertion=args.max_insertion,
            rotate=bool(args.rotate),
            circle_frac=args.circle_frac),
        optimizer_cfg=dict(
            lr=args.lr, weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            scheduler=args.scheduler,
            warmup_epochs=args.warmup_epochs,
            constant_epochs=args.constant_epochs,
            plateau_patience=args.plateau_patience,
            plateau_factor=args.plateau_factor,
            min_lr=args.min_lr, grad_clip=args.grad_clip),
        param_count=n_params,
        args=vars(args))
    with open(result_dir / "parameters.txt", "w") as f:
        pprint.pprint(save_dict, stream=f)

    print(f"[OK] Best val acc: {best_val_acc:.4f}")
    print(f"[OK] Saved to: {result_dir}")


if __name__ == "__main__":
    main()