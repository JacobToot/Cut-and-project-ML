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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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

        spacings = torch.from_numpy(np.stack(spacings_list))  # (N, seq_len)
        theta = torch.tensor(theta_list, dtype=torch.float32)  # (N, 2)

        mean_gap = spacings.mean(dim=-1).clamp(min=1e-8)  # (N,)
        self.log_mean_gap = mean_gap.log().float()  # (N,)

        if normalize_spacings:
            spacings = spacings / mean_gap.unsqueeze(-1)
        self.spacings = spacings  # (N, seq_len)

        if stats is None:
            theta_mean = theta.mean(dim=0)  # (2,)
            theta_std = theta.std(dim=0).clamp(min=1e-8)  # (2,)
        else:
            theta_mean = stats["theta_mean"].float()  # (2,)
            theta_std = stats["theta_std"].float().clamp(min=1e-8)  # (2,)

        self.theta = (theta - theta_mean) / theta_std  # (N, 2)
        self.theta_mean = theta_mean  # (2,)
        self.theta_std = theta_std  # (2,)
        self.normalize_spacings = normalize_spacings

    def __len__(self) -> int:
        return len(self.spacings)

    def __getitem__(self, idx: int):
        return (
            self.spacings[idx],
            self.log_mean_gap[idx],
            self.theta[idx],
        )


def train_npe(
    model, train_loader, val_loader, device, *,
    epochs: int, start_schedule: int, lr: float, wd: float,
    clip: float | None, result_dir: Path, patience: int,
):
    """Constant LR for the first ``start_schedule`` epochs, then cosine
    anneal. Saves ``best_weights.pt`` whenever val loss improves, and
    ``weights.pt`` every epoch as a last-epoch fallback.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs - start_schedule),
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    tr_hist, va_hist = [], []

    for ep in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for spacings, log_mean_gap, theta in train_loader:
            spacings = spacings.to(device)
            log_mean_gap = log_mean_gap.to(device)
            theta = theta.to(device)

            opt.zero_grad(set_to_none=True)
            loss = model.compute_loss(
                spacings, theta, log_mean_gap=log_mean_gap,
            )
            loss.backward()
            if clip is not None and clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            batch_losses.append(loss.item())

        if ep > start_schedule:
            sched.step()

        tr_loss = float(np.mean(batch_losses))

        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for spacings, log_mean_gap, theta in val_loader:
                spacings = spacings.to(device)
                log_mean_gap = log_mean_gap.to(device)
                theta = theta.to(device)
                loss = model.compute_loss(
                    spacings, theta, log_mean_gap=log_mean_gap,
                )
                val_batch_losses.append(loss.item())
        va_loss = float(np.mean(val_batch_losses))

        tr_hist.append(tr_loss)
        va_hist.append(va_loss)

        cur_lr = opt.param_groups[0]["lr"]
        print(f"epoch {ep:03d}/{epochs} | lr {cur_lr:.2e} | "
              f"train {tr_loss:.4f} | val {va_loss:.4f}")

        # Save every epoch so partial runs are recoverable.
        torch.save(model.state_dict(), result_dir / "weights.pt")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), result_dir / "best_weights.pt")
            print(f"  -> new best (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {ep} "
                      f"(no improvement for {patience} epochs)")
                break

    return tr_hist, va_hist, best_val_loss


def main():
    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument("--slope_lower", type=float, default=1.0)
    parser.add_argument("--slope_upper", type=float, default=5.0)
    parser.add_argument("--aw_lower", type=float, default=0.5)
    parser.add_argument("--aw_upper", type=float, default=5.0)
    parser.add_argument("--gauss_lower", type=float, default=0.0)
    parser.add_argument("--gauss_upper", type=float, default=0.3)
    parser.add_argument("--poisson_lower", type=float, default=0.0)
    parser.add_argument("--poisson_upper", type=float, default=0.2)
    parser.add_argument("--dropout_lower", type=float, default=0.0)
    parser.add_argument("--dropout_upper", type=float, default=0.2)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--number_samples", type=int, default=8192,
                        help="Number of training samples to generate.")
    parser.add_argument("--val_samples", type=int, default=None,
                        help="Number of validation samples to generate. "
                             "Defaults to number_samples // 4.")

    # flow args
    parser.add_argument("--context_dim", type=int, default=256)
    parser.add_argument("--theta_dim", type=int, default=2)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--B", type=int, default=3)
    parser.add_argument("--n_flow_steps", type=int, default=10)
    parser.add_argument("--hidden_dim_conditioner", type=int, default=128)
    parser.add_argument("--num_conditioner_blocks", type=int, default=2)
    parser.add_argument("--kernel_size", type=int, default=9)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--pooling_ratio", type=int, default=2)
    parser.add_argument("--dropout_conv", type=float, default=0.1)
    parser.add_argument("--embedding_type", type=str, default="cnn",
                        help="Kept for config compatibility; "
                             "NPEModel currently only supports 'cnn'.")

    # training loop args
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--start_schedule", type=int, default=20,
                        help="Constant LR for this many epochs, then cosine.")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)

    # general args
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result_folder", type=str, default="npe_1d")
    parser.add_argument("--weights_name", type=str, default=None,
                        help="If set, copy best_weights.pt + "
                             "parameters.txt to <root>/weights/<weights_name>/.")

    args = parser.parse_args()

    if args.val_samples is None:
        args.val_samples = max(256, args.number_samples // 4)

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
    run_name = f"npe_1d_{timestamp}"
    result_dir = _root / "results" / args.result_folder / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] device      : {device}")
    print(f"[INFO] result dir  : {result_dir}")

    # ---- data ------------------------------------------------------
    slope_range = (args.slope_lower, args.slope_upper)
    aw_range = (args.aw_lower, args.aw_upper)
    gauss_range = (args.gauss_lower, args.gauss_upper)
    poisson_range = (args.poisson_lower, args.poisson_upper)
    dropout_range = (args.dropout_lower, args.dropout_upper)

    train_ds = NPEDataset(
        n_samples=args.number_samples,
        seq_len=args.seq_len,
        slope_range=slope_range,
        aw_range=aw_range,
        gaussian_range=gauss_range,
        poisson_range=poisson_range,
        dropout_range=dropout_range,
        seed=args.seed,
        normalize_spacings=True,
        stats=None,
    )
    torch.save(
        {"theta_mean": train_ds.theta_mean, "theta_std": train_ds.theta_std},
        result_dir / "stats.pt",
    )
    print(f"[OK] stats -> {result_dir / 'stats.pt'}")

    # NPEDataset is a synthetic generator (no on-disk data/split to load),
    # so val is a second, independently-sampled draw. It reuses train's
    # theta_mean/theta_std so both sets live in the same normalized theta
    # space -- otherwise val loss would not be comparable to train loss.
    val_ds = NPEDataset(
        n_samples=args.val_samples,
        seq_len=args.seq_len,
        slope_range=slope_range,
        aw_range=aw_range,
        gaussian_range=gauss_range,
        poisson_range=poisson_range,
        dropout_range=dropout_range,
        seed=args.seed + 1_000_000,  # disjoint stream from train
        normalize_spacings=True,
        stats={"theta_mean": train_ds.theta_mean, "theta_std": train_ds.theta_std},
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"[INFO] train samples : {len(train_ds)}")
    print(f"[INFO] val samples   : {len(val_ds)}")

    model_cfg = {
        "seq_len": args.seq_len,
        "context_dim": args.context_dim,
        "theta_dim": args.theta_dim,
        "K": args.K,
        "B": args.B,
        "n_flow_steps": args.n_flow_steps,
        "hidden_dim_conditioner": args.hidden_dim_conditioner,
        "num_conditioner_blocks": args.num_conditioner_blocks,
        "kernel_size": args.kernel_size,
        "padding": args.padding,
        "stride": args.stride,
        "pooling_ratio": args.pooling_ratio,
        "dropout_conv": args.dropout_conv,
        "embedding_type": args.embedding_type,
    }
    model = NPEModel(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] model params  : {n_params:,}")
    print(f"[INFO] model_cfg     : {model_cfg}")

    tr_hist, va_hist, best_val = train_npe(
        model, train_loader, val_loader, device,
        epochs=args.epochs,
        start_schedule=args.start_schedule,
        lr=args.lr,
        wd=args.weight_decay,
        clip=args.grad_clip,
        result_dir=result_dir,
        patience=args.patience,
    )

    # ---- metrics + loss curve --------------------------------------
    with open(result_dir / "metrics_table.tsv", "w") as f:
        f.write("epoch\ttrain_loss\tval_loss\n")
        for i, (t, v) in enumerate(zip(tr_hist, va_hist), 1):
            f.write(f"{i}\t{t:.6f}\t{v:.6f}\n")

    ep_axis = np.arange(1, len(tr_hist) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep_axis, tr_hist, label="train")
    ax.plot(ep_axis, va_hist, label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("NLL")
    ax.set_title("1D NPE training loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(result_dir / "loss_curve.png", dpi=200)
    plt.close(fig)

    # ---- parameters.txt (full config) ------------------------------
    parameters = dict(
        script=script,
        timestamp=timestamp,
        device=device,
        seed=args.seed,
        slope_range=list(slope_range),
        aw_range=list(aw_range),
        gaussian_range=list(gauss_range),
        poisson_range=list(poisson_range),
        dropout_range=list(dropout_range),
        train_num_samples=len(train_ds),
        val_num_samples=len(val_ds),
        best_val_loss=float(best_val),
        param_count=int(n_params),
        optimizer_cfg=dict(
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            start_schedule=args.start_schedule,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
        ),
        model_cfg=model_cfg,
    )
    with open(result_dir / "parameters.txt", "w") as f:
        pprint.pprint(parameters, stream=f)

    print(f"\n[OK] best   -> {result_dir / 'best_weights.pt'}")
    print(f"[OK] final  -> {result_dir / 'weights.pt'}")
    print(f"[OK] params -> {result_dir / 'parameters.txt'}")

    if args.weights_name:
        w_dir = _root / "weights" / args.weights_name
        w_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(result_dir / "best_weights.pt", w_dir / "weights.pt")
        shutil.copyfile(result_dir / "parameters.txt", w_dir / "parameters.txt")
        shutil.copyfile(result_dir / "stats.pt", w_dir / "stats.pt")
        print(f"[OK] weights/ copy -> {w_dir}")


if __name__ == "__main__":
    main()