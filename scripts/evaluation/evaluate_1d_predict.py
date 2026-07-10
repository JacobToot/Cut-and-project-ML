from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import ast
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


def find_root(root: str = "cut-and-project-ML") -> Path:
    cwd = Path.cwd()
    for parent in (cwd, *cwd.parents):
        if parent.name == root:
            return parent
    raise RuntimeError(f"Specified root '{root}' not found")


def load_parameters(run_dir: Path) -> dict:
    text = (run_dir / "parameters.txt").read_text()
    return ast.literal_eval(text)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def wilson_ci(k: np.ndarray, n: np.ndarray, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    n_safe = np.maximum(n, 1).astype(np.float64)
    p = k / n_safe
    denom = 1.0 + z * z / n_safe
    center = (p + z * z / (2.0 * n_safe)) / denom
    halfwidth = (z * np.sqrt(p * (1 - p) / n_safe + z * z / (4.0 * n_safe * n_safe))) / denom
    lo = np.where(n > 0, center - halfwidth, np.nan)
    hi = np.where(n > 0, center + halfwidth, np.nan)
    return lo, hi


# ─────────────────────────────────────── model loader

def load_model(run_dir: Path, model_kind: str, device: str):
    params = load_parameters(run_dir)
    cfg = params["model_cfg"]

    if str(Path(__file__).parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).parent))

    if model_kind == "gru":
        from models.gru_prediction_1d import CausalGRUNet
        model = CausalGRUNet(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg.get("dropout", 0.0),
        ).to(device)
    elif model_kind == "transformer":
        from models.transformer_prediction_1d import CausalTransformer
        model = CausalTransformer(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            d_ff=cfg["d_ff"],
            max_len=cfg["max_len"],
            dropout=cfg.get("dropout", 0.0),
            tie_embeddings=cfg.get("tie_embeddings", True),
            param_dim=0,
            param_tokens=0,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    weights = run_dir / "weights_best.pt"
    if not weights.exists():
        weights = run_dir / "weights_last.pt"
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, params

@torch.no_grad()
def per_position_accuracy(
    model, crystals, window_length, window_min, device,
    batch_size=32, max_windows_per_crystal=8,
):
    """
    For each validation window of length `window_length`:
      - model prediction at every position k in [window_min, window_length-1]
      - oracle Bayes-optimal prediction using `next_symbol_probabilities`
        with the crystal's true (internal_gaps, W)

    Returns per-window correctness matrices for downstream aggregation.
    """
    from utils.math_utils import next_symbol_probabilities

    L = window_length
    all_windows = []

    for cid, crystal in enumerate(crystals):
        sym = np.asarray(crystal["sym_ids"], dtype=np.int64)
        regime = int(crystal["regime"])
        if regime not in (2, 3):
            continue
        max_start = sym.size - L - 1
        if max_start < 0:
            continue
        n_win = min(max_windows_per_crystal, max_start + 1)
        starts = np.linspace(0, max_start, n_win).astype(np.int64)

        W = float(crystal["acceptance_window"])  # scalar half-width
        internal_gaps = np.asarray(crystal["internal_gaps"], dtype=np.float64)

        for s in starts:
            s = int(s)
            all_windows.append(dict(
                sym=sym[s: s + L].copy(),
                targets=sym[s + 1: s + L + 1].copy(),
                crystal_id=cid,
                regime=regime,
                W=W,
                internal_gaps=internal_gaps,
            ))

    N = len(all_windows)
    print(f"[INFO] Per-position accuracy on {N} windows")
    if N == 0:
        return None

    model_correct = np.zeros((N, L), dtype=np.int8)
    oracle_correct = np.zeros((N, L), dtype=np.int8)
    valid_mask = np.zeros((N, L), dtype=np.bool_)
    regime_arr = np.zeros(N, dtype=np.int64)
    crystal_id_arr = np.zeros(N, dtype=np.int64)

    # ── Model forward in batches ──
    for bstart in range(0, N, batch_size):
        bend = min(bstart + batch_size, N)
        x_batch = np.stack([w["sym"] for w in all_windows[bstart:bend]])
        x_t = torch.from_numpy(x_batch).long().to(device)
        logits = model(x_t)                       # (B, L, V)
        preds = logits.argmax(dim=-1).cpu().numpy()  # (B, L)
        for i, w in enumerate(all_windows[bstart:bend]):
            idx = bstart + i
            model_correct[idx] = (preds[i] == w["targets"]).astype(np.int8)
            regime_arr[idx] = w["regime"]
            crystal_id_arr[idx] = w["crystal_id"]

    # ── Oracle: per-window, per-position ──
    report_every = max(1, N // 20)
    for idx, w in enumerate(all_windows):
        sym = w["sym"]
        tgt = w["targets"]
        W = w["W"]
        ig = w["internal_gaps"]

        for k in range(window_min, L):
            # Context is sym[0:k+1], predicting sym[k+1] = tgt[k]
            context = sym[:k + 1]
            try:
                probs = next_symbol_probabilities(
                    symbolic_sequence=context,
                    internal_gaps=ig,
                    W=W,
                )
            except ValueError:
                continue  # infeasible — leave valid_mask False

            oracle_pred = int(np.argmax(probs))
            oracle_correct[idx, k] = int(oracle_pred == tgt[k])
            valid_mask[idx, k] = True

        if (idx + 1) % report_every == 0:
            print(f"  oracle eval: {idx + 1}/{N}")

    return dict(
        model_correct=model_correct,
        oracle_correct=oracle_correct,
        valid_mask=valid_mask,
        regime=regime_arr,
        crystal_id=crystal_id_arr,
        window_length=L,
        window_min=window_min,
    )


def aggregate_position_accuracy(per_win, bootstrap_samples=200, seed=0):
    """Aggregate per-window correctness into per-position accuracy + CI."""
    if per_win is None:
        return None
    L = per_win["window_length"]
    mc = per_win["model_correct"]
    oc = per_win["oracle_correct"]
    vm = per_win["valid_mask"]
    regime = per_win["regime"]
    crystal_id = per_win["crystal_id"]

    out = dict(ks=np.arange(1, L + 1))

    for name, arr in (("model", mc), ("oracle", oc)):
        correct_sum = (arr * vm).sum(axis=0)
        total = vm.sum(axis=0)
        acc = np.where(total > 0, correct_sum / np.maximum(total, 1), np.nan)
        lo, hi = wilson_ci(correct_sum, total)
        out[f"{name}_acc"] = acc
        out[f"{name}_n"] = total
        out[f"{name}_wilson_lo"] = lo
        out[f"{name}_wilson_hi"] = hi

    for r, v in ((2, 1.0 / 2.0), (3, 1.0 / 3.0)):
        out[f"uniform_r{r}"] = v

    # Bootstrap over unique crystals
    unique_cid, inverse = np.unique(crystal_id, return_inverse=True)
    n_unique = unique_cid.size
    rng = np.random.default_rng(seed)
    boot = np.zeros((bootstrap_samples, L), dtype=np.float64)
    for b in range(bootstrap_samples):
        draws = rng.integers(0, n_unique, size=n_unique)
        counts = np.bincount(draws, minlength=n_unique)
        row_weights = counts[inverse]
        w = row_weights[:, None] * vm
        num = (mc * w).sum(axis=0)
        den = w.sum(axis=0)
        boot[b] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    out["model_boot_lo"] = np.nanquantile(boot, 0.025, axis=0)
    out["model_boot_hi"] = np.nanquantile(boot, 0.975, axis=0)

    for r in (2, 3):
        r_rows = (regime == r)
        if r_rows.any():
            cs = (mc[r_rows] * vm[r_rows]).sum(axis=0)
            t = vm[r_rows].sum(axis=0)
            out[f"model_acc_r{r}"] = np.where(t > 0, cs / np.maximum(t, 1), np.nan)
            out[f"model_n_r{r}"] = t
            cs_o = (oc[r_rows] * vm[r_rows]).sum(axis=0)
            out[f"oracle_acc_r{r}"] = np.where(t > 0, cs_o / np.maximum(t, 1), np.nan)

    return out


def plot_position_accuracy(agg, window_min, out_dir):
    ks = agg["ks"]

    plt.figure(figsize=(8, 4.5))
    plt.fill_between(ks, agg["model_boot_lo"], agg["model_boot_hi"],
                     alpha=0.2, label="model 95% bootstrap")
    plt.fill_between(ks, agg["model_wilson_lo"], agg["model_wilson_hi"],
                     alpha=0.25, label="model 95% Wilson")
    plt.plot(ks, agg["model_acc"], label="model", linewidth=1.5)
    plt.plot(ks, agg["oracle_acc"], "--", label="oracle (known α, W)", linewidth=1.5)
    plt.axhline(1 / 3, color="gray", ls=":", alpha=0.6, label="uniform (3-gap: 1/3)")
    plt.axvline(window_min, color="black", ls=":", alpha=0.3, label=f"window_min={window_min}")
    plt.xlabel("context length k")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "per_position_accuracy.png", dpi=200)
    plt.close()

    # Per-regime split
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, r in zip(axes, (2, 3)):
        key = f"model_acc_r{r}"
        if key not in agg:
            ax.set_title(f"{r}-gap (no data)")
            continue
        ax.plot(ks, agg[key], label="model", linewidth=1.5)
        ax.plot(ks, agg[f"oracle_acc_r{r}"], "--", label="oracle", linewidth=1.5)
        ax.axhline(1 / r, color="gray", ls=":", alpha=0.6, label=f"uniform (1/{r})")
        ax.axvline(window_min, color="black", ls=":", alpha=0.3)
        ax.set_xlabel("context length k")
        ax.set_title(f"{r}-gap")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_dir / "per_position_accuracy_by_regime.png", dpi=200)
    plt.close()


def save_position_accuracy_tables(per_win, agg, out_dir):
    np.savez_compressed(
        out_dir / "per_position_raw.npz",
        model_correct=per_win["model_correct"],
        oracle_correct=per_win["oracle_correct"],
        valid_mask=per_win["valid_mask"],
        regime=per_win["regime"],
        crystal_id=per_win["crystal_id"],
        window_length=per_win["window_length"],
        window_min=per_win["window_min"],
    )
    ks = agg["ks"]
    cols = [
        ("k", ks.astype(np.int64)),
        ("model_acc", agg["model_acc"]),
        ("model_wilson_lo", agg["model_wilson_lo"]),
        ("model_wilson_hi", agg["model_wilson_hi"]),
        ("model_boot_lo", agg["model_boot_lo"]),
        ("model_boot_hi", agg["model_boot_hi"]),
        ("model_n", agg["model_n"]),
        ("oracle_acc", agg["oracle_acc"]),
        ("oracle_wilson_lo", agg["oracle_wilson_lo"]),
        ("oracle_wilson_hi", agg["oracle_wilson_hi"]),
        ("oracle_n", agg["oracle_n"]),
    ]
    for r in (2, 3):
        if f"model_acc_r{r}" in agg:
            cols.append((f"model_acc_r{r}", agg[f"model_acc_r{r}"]))
            cols.append((f"oracle_acc_r{r}", agg[f"oracle_acc_r{r}"]))
            cols.append((f"n_r{r}", agg[f"model_n_r{r}"]))
    header = "\t".join(n for n, _ in cols)
    lines = [header]
    for i in range(ks.size):
        parts = []
        for name, arr in cols:
            v = arr[i]
            if isinstance(v, (np.integer, int)):
                parts.append(str(int(v)))
            elif np.isnan(v):
                parts.append("nan")
            else:
                parts.append(f"{float(v):.6f}")
        lines.append("\t".join(parts))
    (out_dir / "per_position_accuracy.tsv").write_text("\n".join(lines) + "\n")



@torch.no_grad()
def rollout_plausibility(model, crystal, seed_length, K, device):
    """
    Seed the model with the first `seed_length` symbols, then greedily
    extend for K steps.  At each step, check `is_valid_sequence` on the
    full accumulated context.  Once infeasible, it stays infeasible.
    """
    from utils.math_utils import is_valid_sequence

    sym = np.asarray(crystal["sym_ids"], dtype=np.int64)
    W = float(crystal["acceptance_window"])
    internal_gaps = np.asarray(crystal["internal_gaps"], dtype=np.float64)

    ctx = list(sym[:seed_length])
    plausibility = np.zeros(K, dtype=np.int8)
    alive = True

    for k in range(K):
        x_t = torch.from_numpy(np.array(ctx, dtype=np.int64)[None, :]).to(device)
        if hasattr(model, "max_len"):
            x_t = x_t[:, -model.max_len:]
        logits = model(x_t)[0, -1, :]
        nxt = int(logits.argmax().item())
        ctx.append(nxt)

        if not alive:
            continue

        if is_valid_sequence(np.array(ctx, dtype=np.int64), internal_gaps, W):
            plausibility[k] = 1
        else:
            alive = False

    return plausibility


@torch.no_grad()
def rollout_plausibility_uniform(crystal, seed_length, K, rng):
    """Same but picks each next symbol uniformly at random."""
    from utils.math_utils import is_valid_sequence

    sym = np.asarray(crystal["sym_ids"], dtype=np.int64)
    W = float(crystal["acceptance_window"])
    internal_gaps = np.asarray(crystal["internal_gaps"], dtype=np.float64)
    n_sym = int(internal_gaps.size)

    ctx = list(sym[:seed_length])
    plausibility = np.zeros(K, dtype=np.int8)
    alive = True

    for k in range(K):
        nxt = int(rng.integers(0, n_sym))
        ctx.append(nxt)
        if not alive:
            continue
        if is_valid_sequence(np.array(ctx, dtype=np.int64), internal_gaps, W):
            plausibility[k] = 1
        else:
            alive = False

    return plausibility


def plausibility_eval(model, crystals_3gap, seed_lengths, K, device,
                      include_uniform=True, seed=0):
    rng = np.random.default_rng(seed)
    results = {}
    for L_seed in seed_lengths:
        n_c = len(crystals_3gap)
        plaus_model = np.zeros((n_c, K), dtype=np.int8)
        plaus_uni = np.zeros((n_c, K), dtype=np.int8) if include_uniform else None
        for i, c in enumerate(crystals_3gap):
            if c["sym_ids"].size < L_seed + 1:
                continue
            plaus_model[i] = rollout_plausibility(model, c, L_seed, K, device)
            if include_uniform:
                plaus_uni[i] = rollout_plausibility_uniform(c, L_seed, K, rng)
            if (i + 1) % max(1, n_c // 10) == 0:
                print(f"  [seed_len={L_seed}] {i + 1}/{n_c}")
        results[L_seed] = dict(plaus_model=plaus_model, plaus_uniform=plaus_uni)
    return results


def plot_plausibility(plaus_results, K, out_dir):
    seed_lengths = sorted(plaus_results.keys())
    ks = np.arange(1, K + 1)
    cmap = plt.get_cmap("viridis")

    plt.figure(figsize=(8, 5))
    for i, L_seed in enumerate(seed_lengths):
        surv = plaus_results[L_seed]["plaus_model"].mean(axis=0)
        color = cmap(i / max(1, len(seed_lengths) - 1))
        plt.plot(ks, surv, label=f"model (seed={L_seed})", color=color, linewidth=1.5)

    if plaus_results[seed_lengths[0]]["plaus_uniform"] is not None:
        for L_seed in (seed_lengths[0], seed_lengths[-1]):
            surv_u = plaus_results[L_seed]["plaus_uniform"].mean(axis=0)
            plt.plot(ks, surv_u, ":", label=f"uniform (seed={L_seed})",
                     alpha=0.5, linewidth=1)

    plt.xlabel("rollout step k")
    plt.ylabel("fraction still plausible")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right", fontsize=8)
    plt.title("Plausibility survival (3-gap crystals)")
    plt.tight_layout()
    plt.savefig(out_dir / "plausibility_survival.png", dpi=200)
    plt.close()


def save_plausibility_tables(plaus_results, K, out_dir):
    save_kwargs = {}
    for L_seed, v in plaus_results.items():
        save_kwargs[f"plaus_model_seed{L_seed}"] = v["plaus_model"]
        if v["plaus_uniform"] is not None:
            save_kwargs[f"plaus_uniform_seed{L_seed}"] = v["plaus_uniform"]
    np.savez_compressed(out_dir / "plausibility_raw.npz", **save_kwargs)

    ks = np.arange(1, K + 1)
    cols = [("k", ks.astype(np.int64))]
    for L_seed in sorted(plaus_results.keys()):
        surv_m = plaus_results[L_seed]["plaus_model"].mean(axis=0)
        cols.append((f"model_seed{L_seed}", surv_m))
        if plaus_results[L_seed]["plaus_uniform"] is not None:
            surv_u = plaus_results[L_seed]["plaus_uniform"].mean(axis=0)
            cols.append((f"uniform_seed{L_seed}", surv_u))
    header = "\t".join(n for n, _ in cols)
    lines = [header]
    for i in range(K):
        parts = []
        for name, arr in cols:
            v = arr[i]
            parts.append(str(int(v)) if isinstance(v, (np.integer, int)) else f"{float(v):.6f}")
        lines.append("\t".join(parts))
    (out_dir / "plausibility_survival.tsv").write_text("\n".join(lines) + "\n")


# ─────────────────────────────────────── main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="results/prediction_1d/gru/prediction_1d_gru_thesis")
    parser.add_argument("--model_kind", type=str, default="gru",
                        choices=["gru", "transformer"])
    parser.add_argument("--eval_num_crystals", type=int, default=1)
    parser.add_argument("--eval_seed", type=int, default=1044)
    parser.add_argument("--rollout_steps", type=int, default=150)
    parser.add_argument("--max_windows_per_crystal", type=int, default=8)
    parser.add_argument("--bootstrap_samples", type=int, default=200)
    parser.add_argument("--seed_lengths", type=str, default="32,64,128,256,512")
    parser.add_argument("--include_uniform_baseline", type=int, default=1, choices=[0, 1])
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)

    root = find_root("cut-and-project-ML")
    source_dir = root / "source"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    if str(Path(__file__).parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).parent))

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Run: {run_dir}")
    print(f"[INFO] Out: {out_dir}")

    model, params = load_model(run_dir, args.model_kind, device)
    ds_cfg = params["dataset_cfg"]
    window_length = int(ds_cfg["window_length"])
    window_min = int(ds_cfg.get("window_min", 10))
    slope_range = ds_cfg["slope_range"]
    W_range = ds_cfg["W_range"]

    # ── Build eval crystals (using the same builder as training) ──
    print("[INFO] Building evaluation crystals...")
    if args.model_kind == "gru":
        from scripts.training.train_1d_predict_gru import build_crystals
    elif args.model_kind == "transformer":
        from scripts.training.train_1d_predict_transformer import build_crystals

    max_seed = max(int(x) for x in args.seed_lengths.split(","))
    needed_len = max(ds_cfg["len_crystal"], max_seed + args.rollout_steps + 64)

    eval_crystals = build_crystals(
        num_crystals=args.eval_num_crystals,
        len_crystal=needed_len,
        slope_lower=slope_range[0], slope_upper=slope_range[1],
        acceptance_window_lower=W_range[0], acceptance_window_upper=W_range[1],
        seed=args.eval_seed,
        attempt_multiplier=50,
        min_tiles=int(ds_cfg.get("min_tiles", 2)),
        max_tiles=int(ds_cfg.get("max_tiles", 3)),
    )
    n_r2 = sum(1 for c in eval_crystals if c["regime"] == 2)
    n_r3 = sum(1 for c in eval_crystals if c["regime"] == 3)
    print(f"[INFO] Built {len(eval_crystals)} crystals (2-gap: {n_r2}, 3-gap: {n_r3})")

    # ── Per-position accuracy ──
    print("[INFO] Per-position accuracy eval...")
    per_win = per_position_accuracy(
        model, eval_crystals, window_length, window_min, device,
        max_windows_per_crystal=args.max_windows_per_crystal,
    )
    if per_win is not None:
        agg = aggregate_position_accuracy(
            per_win, bootstrap_samples=args.bootstrap_samples, seed=args.eval_seed,
        )
        plot_position_accuracy(agg, window_min, out_dir)
        save_position_accuracy_tables(per_win, agg, out_dir)
        print(f"[OK] Per-position accuracy → {out_dir}")

    # ── Plausibility survival (3-gap only) ──
    seed_lengths = sorted(int(x) for x in args.seed_lengths.split(","))
    crystals_3gap = [c for c in eval_crystals if c["regime"] == 3]
    print(f"[INFO] Plausibility survival on {len(crystals_3gap)} 3-gap crystals, "
          f"seed lengths {seed_lengths}, K={args.rollout_steps}")
    plaus_results = plausibility_eval(
        model, crystals_3gap, seed_lengths, args.rollout_steps, device,
        include_uniform=bool(args.include_uniform_baseline), seed=args.eval_seed,
    )
    plot_plausibility(plaus_results, args.rollout_steps, out_dir)
    save_plausibility_tables(plaus_results, args.rollout_steps, out_dir)

    # ── Summary ──
    summary_path = out_dir / "eval_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"run_dir: {run_dir}\n")
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
        f.write(f"model_kind: {args.model_kind}\n")
        f.write(f"eval_seed: {args.eval_seed}\n")
        f.write(f"window_length: {window_length}, window_min: {window_min}\n")
        f.write(f"n_eval_crystals: {len(eval_crystals)} "
                f"(2-gap: {n_r2}, 3-gap: {n_r3})\n")
        f.write(f"rollout_steps: {args.rollout_steps}\n")
        f.write(f"seed_lengths: {seed_lengths}\n\n")

        if per_win is not None:
            f.write("== Per-position accuracy (selected k) ==\n")
            for k_rep in (10, 16, 32, 64, 128, 256, window_length - 1):
                if 1 <= k_rep <= agg["ks"].size:
                    i = k_rep - 1
                    f.write(
                        f"  k={k_rep:4d}: "
                        f"model={agg['model_acc'][i]:.4f} "
                        f"(Wilson {agg['model_wilson_lo'][i]:.4f}-"
                        f"{agg['model_wilson_hi'][i]:.4f}, "
                        f"boot {agg['model_boot_lo'][i]:.4f}-"
                        f"{agg['model_boot_hi'][i]:.4f})  "
                        f"oracle={agg['oracle_acc'][i]:.4f}  "
                        f"n={int(agg['model_n'][i])}\n"
                    )
            f.write("\n")

        f.write("== Plausibility survival (3-gap) ==\n")
        for L_seed in seed_lengths:
            surv = plaus_results[L_seed]["plaus_model"].mean(axis=0)
            for k_rep in (10, 25, 50, 100, args.rollout_steps):
                if 1 <= k_rep <= args.rollout_steps:
                    f.write(f"  seed_len={L_seed:3d} k={k_rep:3d}: "
                            f"survival={surv[k_rep - 1]:.4f}\n")
            f.write("\n")

    print(f"[OK] Summary: {summary_path}")
    print("[OK] Done.")


if __name__ == "__main__":
    main()