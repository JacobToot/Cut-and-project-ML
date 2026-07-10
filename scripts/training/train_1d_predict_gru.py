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


class CausalGRU(nn.Module):
    """
    Embedding → multi-layer GRU → linear head.

    forward(x)  x: (B, L) long  →  logits: (B, L, vocab_size)

    The logits at position k correspond to the prediction for position k+1,
    i.e. the model is trained so that logits[:, k, :] predicts x[:, k+1].
    """

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int,
                 dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)              # (B, L, H)
        h, _ = self.gru(h)             # (B, L, H)
        return self.head(h)            # (B, L, V)


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


# ────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────

class QCWindowDataset(Dataset):
    """
    Each item is a (context, targets, mask) triple:
      - context : (L,) int64   — the symbol window  [s_0, s_1, ..., s_{L-1}]
      - targets : (L,) int64   — shifted symbols     [s_1, s_2, ..., s_L]
      - mask    : (L,) bool    — True at positions k >= window_min
                                 (only those positions contribute to the loss)

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
            # We need window_length + 1 symbols to form (context, target) pairs
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


# ────────────────────────────────────────────────────────────────────
# Training / evaluation loops
# ────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip,
                    window_min):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for context, targets, mask in loader:
        context = context.to(device, non_blocking=True)  # (B, L)
        targets = targets.to(device, non_blocking=True)  # (B, L)
        mask = mask.to(device, non_blocking=True)         # (B, L)

        optimizer.zero_grad(set_to_none=True)

        logits = model(context)  # (B, L, V)

        # Flatten only the masked positions
        # mask is (B, L), logits is (B, L, V), targets is (B, L)
        logits_flat = logits[mask]    # (N_masked, V)
        targets_flat = targets[mask]  # (N_masked,)

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


# ────────────────────────────────────────────────────────────────────
# Plotting / saving
# ────────────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train a causal GRU next-symbol predictor on quasi-crystal windows."
    )

    # Reproducibility
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--result_folder", type=str, default="prediction_1d/gru")

    # Crystal generation
    parser.add_argument("--train_num_crystals", type=int, default=20)
    parser.add_argument("--val_num_crystals", type=int, default=20)
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

    # Model
    parser.add_argument("--vocab_size", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)

    # Optimisation
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--use_cosine_lr", type=int, default=1, choices=[0, 1])
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    # ── Setup ──
    root = find_root("cut-and-project-ML")
    source_dir = root / "source"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    seed_everything(args.seed)
    device = pick_device()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = root / "results" / args.result_folder / f"prediction_1d_gru_{timestamp}"
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
    model = CausalGRU(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = count_parameters(model)
    print(f"[INFO] GRU parameters: {n_params:,}")

    # ── Optimiser ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.use_cosine_lr else None
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ── Training loop ──
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            args.grad_clip, args.window_min,
        )
        va_loss, va_acc = evaluate(model, val_loader, criterion, device,
                                   args.window_min)
        if scheduler is not None:
            scheduler.step()

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_accs.append(tr_acc); val_accs.append(va_acc)

        print(f"epoch {epoch:03d} | "
              f"train CE {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val CE {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), result_dir / "weights_best.pt")

        write_metrics_table(result_dir, train_losses, val_losses, train_accs, val_accs)
        save_curves(result_dir, train_losses, val_losses, train_accs, val_accs)

    torch.save(model.state_dict(), result_dir / "weights_last.pt")

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
        script="train_1d_predict_gru",
        timestamp=timestamp,
        device=device,
        best_val_acc=float(best_val_acc),
        train_num_samples=len(train_ds),
        val_num_samples=len(val_ds),
        model_cfg=dict(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ),
        dataset_cfg=dataset_cfg,
        optimizer_cfg=dict(
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            use_cosine_lr=args.use_cosine_lr,
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