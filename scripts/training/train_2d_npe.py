"""
train_2d_npe.py
========================

Train an NPE model (diffraction + edge histogram) on d=5 cut-and-project
tilings, optionally with drop/insert augmentation on the training set.

Augmentation is controlled by four parameters:
  --drop_rate_min   --drop_rate_max
  --insert_rate_min --insert_rate_max

Each training sample draws p_drop ~ U(drop_rate_min, drop_rate_max)
and p_insert ~ U(insert_rate_min, insert_rate_max) independently per
__getitem__ call. The validation dataset is always built without
augmentation so val NLL stays comparable across runs.
"""

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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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


def train_one_epoch(model, loader, optimizer, device, grad_clip):
    model.train()
    losses = []
    for points, mask, edge_hist, log_mean_nn, theta in loader:
        points = points.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        edge_hist = edge_hist.to(device, non_blocking=True)
        log_mean_nn = log_mean_nn.to(device, non_blocking=True)
        theta = theta.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(points, mask, theta,
                                  edge_hist=edge_hist,
                                  log_mean_nn=log_mean_nn)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    for points, mask, edge_hist, log_mean_nn, theta in loader:
        points = points.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        edge_hist = edge_hist.to(device, non_blocking=True)
        log_mean_nn = log_mean_nn.to(device, non_blocking=True)
        theta = theta.to(device, non_blocking=True)
        loss = model.compute_loss(points, mask, theta,
                                  edge_hist=edge_hist,
                                  log_mean_nn=log_mean_nn)
        losses.append(loss.item())
    return float(np.mean(losses))


def save_loss_curves(result_dir, train_losses, val_losses):
    if not train_losses:
        return
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="train NLL")
    ax.plot(epochs, val_losses, label="val NLL")
    ax.set_xlabel("epoch"); ax.set_ylabel("NLL")
    ax.set_title("NPE training (diffraction + edge histogram)")
    ax.legend(); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(result_dir / "loss_curves.png", dpi=200)
    plt.close()


def write_metrics_table(result_dir, train_losses, val_losses):
    if not train_losses:
        return
    epochs = np.arange(1, len(train_losses) + 1)
    data = np.column_stack([epochs, train_losses, val_losses])
    np.savetxt(result_dir / "metrics_table.tsv", data, fmt="%.6f",
               delimiter="\t",
               header="epoch\ttrain_loss\tval_loss", comments="")


def build_scheduler(optimizer, args):
    if args.scheduler == "constant":
        return None
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr)
    if args.scheduler == "cosine_warmup":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0,
            total_iters=args.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs,
            eta_min=args.min_lr)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine],
            milestones=[args.warmup_epochs])
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


def main():
    p = argparse.ArgumentParser(description=__doc__)

    # Data
    p.add_argument("--dataset_dir", type=str, default="dataset/npe_2d_dim=5")
    p.add_argument("--expected_dim", type=int, default=5)
    p.add_argument("--n_min", type=int, default=1024)
    p.add_argument("--n_max", type=int, default=2048)
    p.add_argument("--circle_frac", type=float, default=0.95)
    p.add_argument("--canonicalize", type=int, default=1, choices=[0, 1])

    # Edge histogram
    p.add_argument("--use_edge_hist", type=int, default=1, choices=[0, 1])
    p.add_argument("--hist_n_bins", type=int, default=64)
    p.add_argument("--hist_min", type=float, default=0.0)
    p.add_argument("--hist_max", type=float, default=5.0)
    p.add_argument("--hist_feature_dim", type=int, default=64)
    p.add_argument("--hist_cnn_width", type=int, default=32)

    # Scale scalar
    p.add_argument("--use_log_mean_nn", type=int, default=1, choices=[0, 1])

    # ----- Augmentation (training only) ------------------------------
    p.add_argument("--drop_rate_min", type=float, default=0.0)
    p.add_argument("--drop_rate_max", type=float, default=0.0)
    p.add_argument("--insert_rate_min", type=float, default=0.0)
    p.add_argument("--insert_rate_max", type=float, default=0.0)
    p.add_argument("--clean_frac",type=float, default=0.0)

    # Seed / output
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--result_folder", type=str, default="npe_2d")

    # Diffraction (NUFFT)
    p.add_argument("--grid_size", type=int, default=128)
    p.add_argument("--q_max_factor", type=float, default=4.0)
    p.add_argument("--diff_backend", type=str, default="nufft",
                   choices=["nufft", "direct"])
    p.add_argument("--nufft_width", type=int, default=6)
    p.add_argument("--nufft_sigma", type=float, default=2.0)
    p.add_argument("--suppress_dc", type=int, default=1, choices=[0, 1])
    p.add_argument("--dc_radius", type=int, default=1)
    p.add_argument("--diff_log", type=int, default=1, choices=[0, 1])

    # Diffraction conditioner
    p.add_argument("--context_dim", type=int, default=128)
    p.add_argument("--cnn_width", type=int, default=32)
    p.add_argument("--n_res_per_stage", type=int, default=2)
    p.add_argument("--n_head_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)

    # Flow
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--B", type=int, default=3)
    p.add_argument("--n_flow_steps", type=int, default=8)
    p.add_argument("--hidden_dim_conditioner", type=int, default=128)
    p.add_argument("--num_conditioner_blocks", type=int, default=2)

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--scheduler", type=str, default="cosine_warmup",
                   choices=["constant", "cosine", "cosine_warmup"])
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--num_workers", type=int, default=0)

    args = p.parse_args()

    root = find_root("cut-and-project-ML")
    for sp in (str(root), str(root / "models"), str(root / "source"),
               str(root / "utils")):
        if sp not in sys.path:
            sys.path.insert(0, sp)

    seed_everything(args.seed)
    device = pick_device()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = (root / "results" / args.result_folder
                  / f"npe_diffraction_2d_{timestamp}")
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Results: {result_dir}")

    dataset_dir = root / args.dataset_dir
    train_tilings = load_tilings(dataset_dir / "training")
    val_tilings = load_tilings(dataset_dir / "validation")

    from utils.npe_diffraction_dataset import (
        NPEDiffraction2D as NPEDataset,
        collate_npe_2d,
    )

    common_ds = dict(
        expected_dim=args.expected_dim,
        n_min=args.n_min, n_max=args.n_max,
        circle_frac=args.circle_frac,
        normalize=True,
        canonicalize=bool(args.canonicalize),
        compute_hist=bool(args.use_edge_hist),
        hist_n_bins=args.hist_n_bins,
        hist_min=args.hist_min,
        hist_max=args.hist_max,
    )

    train_ds = NPEDataset(
        train_tilings,
        drop_rate_min=args.drop_rate_min,
        drop_rate_max=args.drop_rate_max,
        insert_rate_min=args.insert_rate_min,
        insert_rate_max=args.insert_rate_max,
        clean_frac=args.clean_frac,
        **common_ds,
    )

    # Val must NOT receive the training augmentation rates -- passing
    # them through here (as the code previously did) silently corrupted
    # validation with the same random drop/insert rates as training,
    # contradicting this script's own stated design (see module
    # docstring and the "NOT used at eval time" comment near
    # save_dict below). Leaving these at the class defaults (0.0) keeps
    # val clean and comparable across differently-augmented runs.
    val_ds = NPEDataset(
        val_tilings,
        seed=args.seed,
        **common_ds,
    )

    print(f"[INFO] Train samples (dim={args.expected_dim}): {len(train_ds)}")
    print(f"[INFO] Val samples (dim={args.expected_dim}):   {len(val_ds)}")
    if args.drop_rate_max > 0 or args.insert_rate_max > 0:
        print(f"[INFO] Training augmentation: "
              f"drop ~ U({args.drop_rate_min}, {args.drop_rate_max}), "
              f"insert ~ U({args.insert_rate_min}, {args.insert_rate_max})")
    else:
        print("[INFO] No training augmentation (drop/insert rates are zero).")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=collate_npe_2d, num_workers=args.num_workers,
        pin_memory=(device == "cuda"))
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        collate_fn=collate_npe_2d, num_workers=args.num_workers,
        pin_memory=(device == "cuda"))

    from utils.nufft2d import DiffractionConfig
    from utils.nufft2d import DiffractionImager


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
    diff_image_module = DiffractionImager(diff_cfg)

    # ---- model ---------------------------------------------------------
    theta_dim = 2 * args.expected_dim
    from models.npe_diffraction_2d import NPEDiffraction2D as NPEModel

    model = NPEModel(
        diff_image_module=diff_image_module,
        cnn_width=args.cnn_width,
        n_res_per_stage=args.n_res_per_stage,
        n_head_layers=args.n_head_layers,
        dropout=args.dropout,
        context_dim=args.context_dim,
        use_edge_hist=bool(args.use_edge_hist),
        hist_n_bins=args.hist_n_bins,
        hist_feature_dim=args.hist_feature_dim,
        hist_cnn_width=args.hist_cnn_width,
        use_log_mean_nn=bool(args.use_log_mean_nn),
        theta_dim=theta_dim,
        K=args.K, B=args.B,
        n_flow_steps=args.n_flow_steps,
        hidden_dim_conditioner=args.hidden_dim_conditioner,
        num_conditioner_blocks=args.num_conditioner_blocks,
    ).to(device)

    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"[INFO] Model parameters: {n_params:,}")
    print(f"[INFO] theta_dim       : {theta_dim}")
    print(f"[INFO] use_edge_hist   : {bool(args.use_edge_hist)}")
    print(f"[INFO] use_log_mean_nn : {bool(args.use_log_mean_nn)}")

    # ---- optimiser + scheduler -----------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args)

    # ---- training loop -------------------------------------------------
    train_losses, val_losses = [], []
    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device,
                                   args.grad_clip)
        va_loss = evaluate(model, val_loader, device)
        if scheduler is not None:
            scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:03d} | train NLL {tr_loss:.4f} | "
              f"val NLL {va_loss:.4f} | lr {lr_now:.2e}")

        if va_loss < best_val:
            best_val = va_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), result_dir / "weights_best.pt")
        else:
            epochs_no_improve += 1

        write_metrics_table(result_dir, train_losses, val_losses)
        save_loss_curves(result_dir, train_losses, val_losses)

        if epochs_no_improve >= args.patience:
            print(f"[INFO] Early stopping at epoch {epoch}.")
            break

    torch.save(model.state_dict(), result_dir / "weights_last.pt")

    save_dict = dict(
        script="npe_diffraction_2d",
        timestamp=timestamp,
        device=device,
        best_val_nll=float(best_val),
        train_num_samples=len(train_ds),
        val_num_samples=len(val_ds),
        dataset_dir=str(dataset_dir),
        expected_dim=args.expected_dim,
        theta_dim=theta_dim,
        canonicalize=bool(args.canonicalize),
        model_cfg=dict(
            context_dim=args.context_dim,
            cnn_width=args.cnn_width,
            n_res_per_stage=args.n_res_per_stage,
            n_head_layers=args.n_head_layers,
            dropout=args.dropout,
            use_log_mean_nn=bool(args.use_log_mean_nn),
            use_edge_hist=bool(args.use_edge_hist),
            hist_n_bins=args.hist_n_bins,
            hist_feature_dim=args.hist_feature_dim,
            hist_cnn_width=args.hist_cnn_width,
            K=args.K, B=args.B,
            n_flow_steps=args.n_flow_steps,
            hidden_dim_conditioner=args.hidden_dim_conditioner,
            num_conditioner_blocks=args.num_conditioner_blocks),
        diffraction_cfg=dict(
            grid_size=args.grid_size,
            q_max_factor=args.q_max_factor,
            backend=args.diff_backend,
            nufft_width=args.nufft_width, nufft_sigma=args.nufft_sigma,
            suppress_dc=bool(args.suppress_dc), dc_radius=args.dc_radius,
            log1p=bool(args.diff_log)),
        dataset_cfg=dict(
            n_min=args.n_min, n_max=args.n_max,
            circle_frac=args.circle_frac,
            normalize=True,
            compute_hist=bool(args.use_edge_hist),
            hist_n_bins=args.hist_n_bins,
            hist_min=args.hist_min,
            hist_max=args.hist_max,
            # training augmentation rates (NOT used at eval time)
            drop_rate_min=args.drop_rate_min,
            drop_rate_max=args.drop_rate_max,
            insert_rate_min=args.insert_rate_min,
            insert_rate_max=args.insert_rate_max),
        optimizer_cfg=dict(
            lr=args.lr, weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            scheduler=args.scheduler,
            warmup_epochs=args.warmup_epochs,
            min_lr=args.min_lr,
            patience=args.patience),
        param_count=n_params,
        args=vars(args),
    )
    with open(result_dir / "parameters.txt", "w") as f:
        pprint.pprint(save_dict, stream=f)

    print(f"[OK] Best val NLL: {best_val:.4f}")
    print(f"[OK] Saved to: {result_dir}")


if __name__ == "__main__":
    main()