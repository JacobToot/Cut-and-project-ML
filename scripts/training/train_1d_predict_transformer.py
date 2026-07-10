from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import pprint
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
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
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_crystals(
    num_crystals: int,
    len_crystal: int,
    slope_lower: float,
    slope_upper: float,
    acceptance_window_lower: float,
    acceptance_window_upper: float,
    seed: int,
    attempt_multiplier: int = 50,
    min_tiles: int = 2,
    max_tiles: int = 3,
) -> list[dict]:
    """
    Build a list of crystal dicts, each containing:
      - sym_ids        : np.ndarray of int64, the symbol sequence
      - regime         : int, number of distinct tile types (2 or 3)
      - acceptance_window : float, the half-width W
      - internal_gaps  : np.ndarray of float64, star-space gap per tile type
      - tiles          : np.ndarray of float64, physical tile lengths (ascending)
      - slope          : float

    Uses quasi_crystal(..., internal_tiles=True) so internal gaps are exact.
    """
    from simulations.quasi_crystal import quasi_crystal
    from utils.complexity_utils import wordify

    crystals = []
    rng = np.random.default_rng(seed)
    max_attempts = num_crystals * attempt_multiplier

    for attempt in range(max_attempts):
        if len(crystals) >= num_crystals:
            break

        crystal_seed = int(rng.integers(0, 2**31))

        result = quasi_crystal(
            number_of_points=len_crystal + 64,
            seed=crystal_seed,
            slope_lower=slope_lower,
            slope_upper=slope_upper,
            acceptance_window_lower=acceptance_window_lower,
            acceptance_window_upper=acceptance_window_upper,
            internal_tiles=True,
        )

        (
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
        ) = result

        if tiles is None or number_tiles is None:
            continue
        if not (min_tiles <= number_tiles <= max_tiles):
            continue

        try:
            symbols, _centers = wordify(
                                        spacings, centers=tiles, alphabet=tuple(range(number_tiles)),
                                    )
        except (ValueError, IndexError):
            continue

        symbols = np.asarray(symbols, dtype=np.int64)
        if symbols.size < len_crystal:
            continue

        symbols = symbols[:len_crystal]

        # Validate: internal_tile_coords should have one entry per tile type
        if internal_tile_coords is None or internal_tile_coords.size != number_tiles:
            continue

        crystals.append(dict(
            sym_ids=symbols,
            regime=number_tiles,
            acceptance_window=float(acceptance_window),
            internal_gaps=internal_tile_coords.astype(np.float64),
            tiles=tiles.astype(np.float64),
            slope=float(slope),
        ))

    if len(crystals) < num_crystals:
        print(f"[WARN] Only built {len(crystals)}/{num_crystals} crystals "
              f"after {max_attempts} attempts.")
    return crystals


class QCWindowDataset(Dataset):
    """
    Each item is a (context, targets, mask) triple:
      - context : (L,) int64   — the symbol window  [s_0, s_1, ..., s_{L-1}]
      - targets : (L,) int64   — shifted symbols     [s_1, s_2, ..., s_L]
      - mask    : (L,) bool    — True at positions k >= window_min

    So logits[:, k, :] is trained to predict targets[:, k] = s_{k+1},
    but only for k >= window_min.

    Windows are extracted sequentially from each crystal with stride 1,
    up to `num_windows` per crystal.
    """

    def __init__(
        self,
        crystals: list[dict],
        window_length: int,
        num_windows: int,
        window_min: int = 10,
    ):
        self.samples = []
        self.window_min = window_min

        for crystal in crystals:
            sym = crystal["sym_ids"]
            max_start = sym.size - window_length - 1
            if max_start < 0:
                continue
            n_win = min(num_windows, max_start + 1)
            starts = np.linspace(0, max_start, n_win, dtype=np.int64)
            for s in starts:
                s = int(s)
                context = sym[s: s + window_length].copy()
                targets = sym[s + 1: s + window_length + 1].copy()
                self.samples.append((context, targets))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, targets = self.samples[idx]
        mask = np.zeros(context.shape[0], dtype=np.bool_)
        mask[self.window_min:] = True
        return (
            torch.from_numpy(context),
            torch.from_numpy(targets),
            torch.from_numpy(mask),
        )


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip,
                    window_min):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for context, targets, mask in loader:
        context = context.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(context)  # (B, L, V)

        logits_flat = logits[mask]
        targets_flat = targets[mask]

        if logits_flat.size(0) == 0:
            continue

        loss = criterion(logits_flat, targets_flat)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        preds = logits_flat.argmax(dim=-1)
        total_correct += int((preds == targets_flat).sum().item())
        total_count += int(targets_flat.numel())
        total_loss += float(loss.item()) * int(targets_flat.numel())

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, window_min):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for context, targets, mask in loader:
        context = context.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        logits = model(context)

        logits_flat = logits[mask]
        targets_flat = targets[mask]

        if logits_flat.size(0) == 0:
            continue

        loss = criterion(logits_flat, targets_flat)

        preds = logits_flat.argmax(dim=-1)
        total_correct += int((preds == targets_flat).sum().item())
        total_count += int(targets_flat.numel())
        total_loss += float(loss.item()) * int(targets_flat.numel())

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


def write_metrics_table(result_dir, train_losses, val_losses, train_accs, val_accs):
    if not train_losses:
        return
    epochs = np.arange(1, len(train_losses) + 1)
    data = np.column_stack([epochs, train_losses, val_losses, train_accs, val_accs])
    np.savetxt(
        result_dir / "metrics_table.tsv", data, fmt="%.6f", delimiter="\t",
        header="epoch\ttrain_loss\tval_loss\ttrain_acc\tval_acc", comments="",
    )


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


def main():
    parser = argparse.ArgumentParser(
        description="Train a causal Transformer next-symbol predictor on "
                    "quasi-crystal windows."
    )

    # Reproducibility
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--result_folder", type=str, default="Predict_Symbol_Transformer")

    # Crystal generation
    parser.add_argument("--train_num_crystals", type=int, default=2048)
    parser.add_argument("--val_num_crystals", type=int, default=256)
    parser.add_argument("--len_crystal", type=int, default=4096)
    parser.add_argument("--slope_lower", type=float, default=1.0)
    parser.add_argument("--slope_upper", type=float, default=5.0)
    parser.add_argument("--acceptance_window_lower", type=float, default=0.5)
    parser.add_argument("--acceptance_window_upper", type=float, default=5.0)
    parser.add_argument("--attempt_multiplier", type=int, default=50)
    parser.add_argument("--min_tiles", type=int, default=2)
    parser.add_argument("--max_tiles", type=int, default=3)

    # Windowing
    parser.add_argument("--window_length", type=int, default=512)
    parser.add_argument("--num_windows", type=int, default=512)
    parser.add_argument("--window_min", type=int, default=10,
                        help="First position index included in the loss. "
                             "Positions 0..window_min-1 are context-only.")

    # Model — Transformer-specific hyperparameters
    parser.add_argument("--vocab_size", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=128,
                        help="Embedding / hidden dimension (analogous to "
                             "hidden_size in the GRU).")
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of transformer decoder layers.")
    parser.add_argument("--d_ff", type=int, default=256,
                        help="Feed-forward inner dimension.")
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--tie_embeddings", type=int, default=1, choices=[0, 1],
                        help="Tie input embeddings with the output projection.")

    # Optimisation
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--use_cosine_lr", type=int, default=1, choices=[0, 1])
    parser.add_argument("--warmup_steps", type=int, default=200,
                        help="Linear warmup steps before cosine decay. "
                             "Set to 0 to disable warmup.")
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    # ── Setup ──
    root = find_root("cut-and-project-ML")
    source_dir = root / "source"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    from transformer_prediction_1d import CausalTransformer

    seed_everything(args.seed)
    device = pick_device()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = root / "results" / "prediction_1d"/ "transformer" / f"predict_transformer_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Results: {result_dir}")

    # ── Build crystals ──
    print("[INFO] Building training crystals...")
    train_crystals = build_crystals(
        num_crystals=args.train_num_crystals,
        len_crystal=args.len_crystal,
        slope_lower=args.slope_lower, slope_upper=args.slope_upper,
        acceptance_window_lower=args.acceptance_window_lower,
        acceptance_window_upper=args.acceptance_window_upper,
        seed=args.seed,
        attempt_multiplier=args.attempt_multiplier,
        min_tiles=args.min_tiles, max_tiles=args.max_tiles,
    )
    print(f"[INFO] Training crystals: {len(train_crystals)}")

    print("[INFO] Building validation crystals...")
    val_crystals = build_crystals(
        num_crystals=args.val_num_crystals,
        len_crystal=args.len_crystal,
        slope_lower=args.slope_lower, slope_upper=args.slope_upper,
        acceptance_window_lower=args.acceptance_window_lower,
        acceptance_window_upper=args.acceptance_window_upper,
        seed=args.seed + 1,
        attempt_multiplier=args.attempt_multiplier,
        min_tiles=args.min_tiles, max_tiles=args.max_tiles,
    )
    print(f"[INFO] Validation crystals: {len(val_crystals)}")

    # ── Datasets & loaders ──
    train_ds = QCWindowDataset(train_crystals, args.window_length,
                               args.num_windows, args.window_min)
    val_ds = QCWindowDataset(val_crystals, args.window_length,
                             args.num_windows, args.window_min)
    print(f"[INFO] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    # ── Model ──
    model = CausalTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=args.window_length,
        dropout=args.dropout,
        tie_embeddings=bool(args.tie_embeddings),
        param_dim=0,
        param_tokens=0,
    ).to(device)
    n_params = count_parameters(model)
    print(f"[INFO] Transformer parameters: {n_params:,}")

    # ── Optimiser ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # Warmup + cosine schedule (transformers generally benefit from warmup)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(args.warmup_steps, total_steps // 4)

    if args.use_cosine_lr:
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        step_scheduler_per_batch = True
    else:
        scheduler = None
        step_scheduler_per_batch = False

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ── Training loop ──
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        ep_loss = 0.0
        ep_correct = 0
        ep_count = 0

        for context, targets, mask in train_loader:
            context = context.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(context)

            logits_flat = logits[mask]
            targets_flat = targets[mask]

            if logits_flat.size(0) == 0:
                continue

            loss = criterion(logits_flat, targets_flat)
            loss.backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            if step_scheduler_per_batch and scheduler is not None:
                scheduler.step()

            preds = logits_flat.argmax(dim=-1)
            ep_correct += int((preds == targets_flat).sum().item())
            ep_count += int(targets_flat.numel())
            ep_loss += float(loss.item()) * int(targets_flat.numel())

        tr_loss = ep_loss / max(ep_count, 1)
        tr_acc = ep_correct / max(ep_count, 1)

        # --- validate ---
        va_loss, va_acc = evaluate(model, val_loader, criterion, device,
                                   args.window_min)

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_accs.append(tr_acc); val_accs.append(va_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:03d} | "
              f"train CE {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val CE {va_loss:.4f} acc {va_acc:.4f} | "
              f"lr {current_lr:.2e}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), result_dir / "weights_best.pt")

        write_metrics_table(result_dir, train_losses, val_losses, train_accs, val_accs)
        save_curves(result_dir, train_losses, val_losses, train_accs, val_accs)

    torch.save(model.state_dict(), result_dir / "weights_last.pt")

    # ── Save parameters ──
    dataset_cfg = dict(
        window_length=args.window_length,
        window_min=args.window_min,
        num_windows=args.num_windows,
        len_crystal=args.len_crystal,
        slope_range=[args.slope_lower, args.slope_upper],
        W_range=[args.acceptance_window_lower, args.acceptance_window_upper],
        min_tiles=args.min_tiles,
        max_tiles=args.max_tiles,
    )
    save_dict = dict(
        script="predict_transformer",
        timestamp=timestamp,
        device=device,
        best_val_acc=float(best_val_acc),
        train_num_samples=len(train_ds),
        val_num_samples=len(val_ds),
        model_cfg=dict(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            max_len=args.window_length,
            dropout=args.dropout,
            tie_embeddings=bool(args.tie_embeddings),
        ),
        dataset_cfg=dataset_cfg,
        optimizer_cfg=dict(
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            use_cosine_lr=args.use_cosine_lr,
            warmup_steps=warmup_steps,
            grad_clip=args.grad_clip,
        ),
        param_count=n_params,
        args=vars(args),
    )
    with open(result_dir / "parameters.txt", "w") as f:
        pprint.pprint(save_dict, stream=f)

    print(f"[OK] Best val acc: {best_val_acc:.4f}")
    print(f"[OK] Saved to: {result_dir}")


if __name__ == "__main__":
    main()