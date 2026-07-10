import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys


def find_root(root: str = "cut-and-project-ML") -> Path:
    
    cwd = Path.cwd()
    for parent in (cwd, *cwd.parents):
        if parent.name == root: 
            return parent

    raise RuntimeError(f"Specified root '{root}' not found")


root = find_root("cut-and-project-ML")
source_dir = root / "source"

if str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))

from simulations.quasi_crystal import quasi_crystal
from utils.complexity_utils import wordify
from utils.math_utils import truth_frequencies, lookup_table


class Conditioner(nn.Module):
    def __init__(self, number_parameters, token_length, number_tokens=1, hidden=128):
        super().__init__()
        self.tokens = nn.Sequential(
            nn.Linear(number_parameters, hidden),
            nn.ReLU(),
            nn.Linear(hidden, token_length * number_tokens),
        )
        self.number_tokens = number_tokens
        self.token_length = token_length

    def forward(self, x):
        B = x.size(0)
        return self.tokens(x).view(B, self.number_tokens, self.token_length)


class Transformer(nn.Module):
    def __init__(
        self,
        seq_len=16,
        num_classes=3,
        cond_dim=5,
        d_model=64,
        prefix_tokens=1,
        nhead=4,
        num_layers=3,
        dim_ff=256,
        dropout=0.1,
        input="discrete",          
        cont_dim=1,               
    ):
        super().__init__()
        self.prefix_tokens = prefix_tokens
        assert d_model % nhead == 0
        assert input in ("discrete", "continuous")

        self.input = input
        self.seq_len = seq_len
        self.num_classes = num_classes

        if self.input == "discrete":
            self.sym_emb = nn.Embedding(num_classes, d_model)
        else:
            self.sym_emb = nn.Sequential(
                nn.Linear(cont_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )

        self.cond_pfx = Conditioner(cond_dim, d_model, number_tokens=prefix_tokens, hidden=128)

        self.pos = nn.Parameter(torch.randn(1, prefix_tokens + seq_len, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x_ids, p_vec, *, source_key_padding_mask=None):
        """
        x_ids:
        - discrete: LongTensor (B, seq_len) with class indices
        - continuous: FloatTensor (B, seq_len) or (B, seq_len, cont_dim)
        p_vec: FloatTensor (B, cond_dim)

        source_key_padding_mask:
        - optional BoolTensor (B, seq_len) where True = PAD positions in x_ids
        - applies ONLY to the x part (not the prefix tokens)
        """
        if self.input == "discrete":
            x_tok = self.sym_emb(x_ids)  # (B, L, D)
        else:
            x = x_ids
            if x.dim() == 2:
                x = x.unsqueeze(-1)       # (B, L, 1)
            x = x.float()
            x_tok = self.sym_emb(x)       # (B, L, D)

        pfx = self.cond_pfx(p_vec)        # (B, P, D)
        z = torch.cat([pfx, x_tok], dim=1)  # (B, P+L, D)

        z = z + self.pos[:, : z.size(1), :]

        # Build full padding mask for (prefix + x)
        full_pad_mask = None
        if source_key_padding_mask is not None:
            if source_key_padding_mask.dtype != torch.bool:
                source_key_padding_mask = source_key_padding_mask.bool()
            if source_key_padding_mask.dim() != 2 or source_key_padding_mask.size(1) != self.seq_len:
                raise ValueError(
                    f"source_key_padding_mask must be (B, {self.seq_len}) bool, got {tuple(source_key_padding_mask.shape)}"
                )

            B = source_key_padding_mask.size(0)
            # prefix tokens are never padding
            pfx_pad = torch.zeros((B, self.prefix_tokens), dtype=torch.bool, device=source_key_padding_mask.device)
            full_pad_mask = torch.cat([pfx_pad, source_key_padding_mask], dim=1)  # (B, P+L)

        # Encoder attends ignoring PAD positions (True = PAD)
        z = self.encoder(z, source_key_padding_mask=full_pad_mask)  # (B, P+L, D)

        # Pool from last *valid* token position.
        if source_key_padding_mask is None:
            # original behavior: last position
            pooled = z[:, -1, :]
        else:
            # valid lengths in x (count of non-pad)
            valid_len = (~source_key_padding_mask).sum(dim=1)  # (B,)
            # clamp so we never index before the x starts
            valid_len = torch.clamp(valid_len, min=1)
            # index into z: prefix_tokens + (valid_len - 1)
            idx = (self.prefix_tokens + valid_len - 1).to(torch.long)  # (B,)

            # gather pooled states
            pooled = z[torch.arange(z.size(0), device=z.device), idx, :]  # (B, D)

        logits = self.head(pooled)  # (B, num_classes)
        return logits



class QCDataset(Dataset):

    def __init__(self,
                 number_of_crystals,
                 windows_per_crystal,
                 window_length,
                 mode="sequential",
                 truth_type="frequency",
                 input_type="discrete",   
                 gaussian_ratio=0.0,
                 poisson_ratio=0.0,
                 dropout=0.0,
                 points_per_crystal=4096,
                 frequency_check_length=2**20,
                 slope_upper=5,
                 slope_lower=1,
                 acceptance_window_upper=5,
                 acceptance_window_lower=0.5,
                 attempt_multiplier=20,
                 seed=123,
                 ):
        super().__init__()

        assert mode in ("random", "sequential")
        assert truth_type in ("frequency", "next")
        assert input_type in ("discrete", "continuous")

        random_number_generator = np.random.default_rng(seed=seed)

        def parse_range(value, name: str):
            """
            Accepts:
              - single float/int (treated as fixed value),
              - (lower, upper) tuple/list,
              - [lower, upper] list.
            Returns (lower, upper) floats with lower <= upper and both >= 0.
            """
            if isinstance(value, (int, float, np.integer, np.floating)):
                lower_value = upper_value = float(value)
            elif isinstance(value, (tuple, list)) and len(value) == 2:
                lower_value, upper_value = float(value[0]), float(value[1])
                if upper_value < lower_value:
                    lower_value, upper_value = upper_value, lower_value
            else:
                raise TypeError(f"{name} must be a float or a (lower, upper) tuple/list, got {type(value)}")

            if lower_value < 0.0 or upper_value < 0.0:
                raise ValueError(f"{name} range must be >= 0, got ({lower_value}, {upper_value})")

            return lower_value, upper_value

        gaussian_ratio_lower, gaussian_ratio_upper = parse_range(gaussian_ratio, "gaussian_ratio")
        poisson_ratio_lower, poisson_ratio_upper = parse_range(poisson_ratio, "poisson_ratio")
        dropout_lower, dropout_upper = parse_range(dropout, "dropout")

        self.seq_len = int(window_length)
        self.crystals = []

        parameter_vectors, inputs, targets = [], [], []

        attempts = 0
        valid = 0
        maximum_attempts = attempt_multiplier * number_of_crystals

        def fix_length(array_values, required_length):
            array_values = np.asarray(array_values, dtype=np.float32)
            if array_values.size == 0:
                return np.zeros((required_length,), dtype=np.float32)
            if array_values.size < required_length:
                padding = np.full((required_length - array_values.size,), array_values[-1], dtype=np.float32)
                return np.concatenate([array_values, padding], axis=0)
            return array_values[:required_length]

        while valid < number_of_crystals and attempts < maximum_attempts:
            attempts += 1
            crystal_seed = int(random_number_generator.integers(0, 2**27 - 1))

            gaussian_ratio_value = (
                random_number_generator.uniform(gaussian_ratio_lower, gaussian_ratio_upper)
                if gaussian_ratio_upper > gaussian_ratio_lower else gaussian_ratio_lower
            )
            poisson_ratio_value = (
                random_number_generator.uniform(poisson_ratio_lower, poisson_ratio_upper)
                if poisson_ratio_upper > poisson_ratio_lower else poisson_ratio_lower
            )
            dropout_value = (
                random_number_generator.uniform(dropout_lower, dropout_upper)
                if dropout_upper > dropout_lower else dropout_lower
            )

            spacings_clean, distances, indices, slope, acceptance_window, poisson_rate, standard_deviation, number_tiles, tiles = quasi_crystal(
                slope_upper=slope_upper,
                slope_lower=slope_lower,
                acceptance_window_lower=acceptance_window_lower,
                acceptance_window_upper=acceptance_window_upper,
                number_of_points=points_per_crystal,
                gaussian_ratio=0.0,
                poisson_ratio=0.0,
                dropout=0.0,
                seed=crystal_seed
            )

            if number_tiles != 3:
                continue

            symbol_sequence, *_ = wordify(spacings_clean, alphabet=(0, 1, 2))
            symbol_sequence = np.asarray(symbol_sequence, dtype=np.int64)

            maximum_start_index = len(symbol_sequence) - (window_length + 1)
            if maximum_start_index <= 0 or windows_per_crystal > maximum_start_index:
                continue

            if mode == "random":
                start_indices = random_number_generator.choice(
                    maximum_start_index, size=windows_per_crystal, replace=False
                ).astype(np.int64)
            else:
                start_indices = np.arange(0, windows_per_crystal, dtype=np.int64)

            frequency_map = None
            if truth_type == "frequency":
                spacings_extended_length, *_ = quasi_crystal(slope=slope,
                                                            acceptance_window=acceptance_window,
                                                            number_of_points=frequency_check_length,
                                                            seed=crystal_seed + 1,
                                                            )

                extended_symbol_sequence, *_ = wordify(spacings_extended_length, alphabet=(0, 1, 2))
                extended_symbol_sequence = np.asarray(extended_symbol_sequence, dtype=np.int64)

                table = lookup_table(extended_symbol_sequence, window_length, alphabet=(0, 1, 2))
                frequency_map = {
                    row[0]: np.array([row[1], row[2], row[3]], dtype=np.float32)
                    for row in table
                }

            parameter_vector = np.array(
                [float(slope), float(acceptance_window),
                 float(tiles[0]), float(tiles[1]), float(tiles[2])],
                dtype=np.float32,
            )

            self.crystals.append(dict(
                sym_ids=symbol_sequence.copy(),
                pvec=parameter_vector.copy(),
                gaussian_ratio=float(gaussian_ratio_value),
                poisson_ratio=float(poisson_ratio_value),
                dropout=float(dropout_value),
            ))

            for start in start_indices:
                window_symbol_sequence = symbol_sequence[start:start + window_length]

                if truth_type == "frequency":
                    key = tuple(window_symbol_sequence.tolist())
                    target = frequency_map.get(key)
                    if target is None:
                        target = np.array([1/3, 1/3, 1/3], dtype=np.float32)
                else:
                    target = np.int64(symbol_sequence[start + window_length])

                if input_type == "discrete":
                    input_window = window_symbol_sequence.astype(np.int64)

                else:
                    input_window = spacings_clean[start:start + window_length].astype(np.float32).copy()

                    if dropout_value != 0.0:

                        positions = np.cumsum(input_window)
                        keep_mask = random_number_generator.uniform(0, 1, size=positions.size) > dropout_value
                        positions = positions[keep_mask]
                        if positions.size == 0:
                            positions = np.cumsum(input_window)
                        input_window = np.diff(np.concatenate([[0.0], positions])).astype(np.float32)

                    if poisson_ratio_value != 0.0:

                        positions = np.cumsum(input_window)
                        end_value = float(positions[-1]) if positions.size else 0.0

                        if end_value > 0.0:

                            poisson_rate_for_window = (window_length * float(poisson_ratio_value)) / end_value
                            
                            if poisson_rate_for_window > 0.0:

                                current_value = 0.0
                                additional_positions = []
                                
                                while current_value < end_value:
                                    gap = random_number_generator.exponential(scale=1.0 / poisson_rate_for_window)
                                    current_value += gap
                                   
                                    if current_value < end_value:
                                        additional_positions.append(current_value)

                                if len(additional_positions) > 0:
                                    positions = np.sort(
                                        np.concatenate([positions, np.asarray(additional_positions, dtype=np.float32)])
                                    )

                            input_window = np.diff(np.concatenate([[0.0], positions])).astype(np.float32)

                    if gaussian_ratio_value != 0.0:

                        mean_spacing = float(np.mean(input_window)) if input_window.size else 0.0
                        noise_standard_deviation = mean_spacing * float(gaussian_ratio_value)

                        if noise_standard_deviation > 0.0:

                            noise = random_number_generator.normal(
                                loc=0.0, scale=noise_standard_deviation, size=input_window.size
                            ).astype(np.float32)
                            input_window = input_window + noise

                    input_window = fix_length(input_window, window_length)

                parameter_vectors.append(parameter_vector)
                inputs.append(input_window)
                targets.append(target)

            valid += 1

            if attempts % 100 == 0:
                print(f"[attempts={attempts}] valid={valid}")

        if valid < number_of_crystals:
            raise RuntimeError(f"Only collected {valid}/{number_of_crystals} valid crystals")

        self.P = np.stack(parameter_vectors).astype(np.float32)

        if input_type == "continuous":
            self.X = np.stack(inputs).astype(np.float32)
        else:
            self.X = np.stack(inputs).astype(np.int64)

        if truth_type == "frequency":
            self.Y = np.stack(targets).astype(np.float32)
        else:
            self.Y = np.array(targets, dtype=np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.Y.dtype == np.float32:
            return (
                torch.from_numpy(self.P[idx]),
                torch.from_numpy(self.X[idx]),
                torch.from_numpy(self.Y[idx]),
            )
        else:
            return (
                torch.from_numpy(self.P[idx]),
                torch.from_numpy(self.X[idx]),
                torch.tensor(self.Y[idx], dtype=torch.long),
            )


class QCSeq2SeqData(Dataset):
    """
    Seq2Seq dataset for denoising/reconstruction.

    Generates many crystals, then samples random windows from each crystal.
    For each window:
      - target (Y): clean spacings (variable length) padded to max_output_len
      - input  (X): corrupted spacings (dropout/poisson/gaussian) padded to max_input_len

    Returns:
      (P, X, X_len, Y, Y_len)
    where:
      P      : (p_dim,)
      X      : (max_input_len,)  float32
      X_len  : scalar long
      Y      : (max_output_len,) float32
      Y_len  : scalar long
    """

    def __init__(
        self,
        number_of_crystals: int,
        windows_per_crystal: int,
        min_window_length: int,
        max_window_length: int,
        max_input_len: int = 128,
        max_output_len: int = 256,

        gaussian_ratio=0.0,
        poisson_ratio=0.0,
        dropout=0.0,

        points_per_crystal: int = 4096,
        slope_upper: float = 5.0,
        slope_lower: float = 1.0,
        acceptance_window_upper: float = 5.0,
        acceptance_window_lower: float = 0.5,
        attempt_multiplier: int = 20,
        seed: int = 123,

        # If True: normalize spacings by mean spacing of the *clean* window
        normalize_by_clean_mean: bool = True,
        # Padding value for continuous sequences (will be masked via lengths)
        pad_value: float = 0.0,
    ):
        super().__init__()

        assert 1 <= min_window_length <= max_window_length
        assert max_window_length <= max_output_len, (
            "max_window_length must be <= max_output_len (since targets are padded to max_output_len)"
        )
        assert max_input_len >= 1 and max_output_len >= 1

        rng = np.random.default_rng(seed=seed)

        def parse_range(value, name: str):
            if isinstance(value, (int, float, np.integer, np.floating)):
                lo = hi = float(value)
            elif isinstance(value, (tuple, list)) and len(value) == 2:
                lo, hi = float(value[0]), float(value[1])
                if hi < lo:
                    lo, hi = hi, lo
            else:
                raise TypeError(f"{name} must be a float or a (lower, upper) tuple/list, got {type(value)}")
            if lo < 0.0 or hi < 0.0:
                raise ValueError(f"{name} range must be >= 0, got ({lo}, {hi})")
            return lo, hi

        g_lo, g_hi = parse_range(gaussian_ratio, "gaussian_ratio")
        p_lo, p_hi = parse_range(poisson_ratio, "poisson_ratio")
        d_lo, d_hi = parse_range(dropout, "dropout")

        def sample_value(lo, hi):
            return float(rng.uniform(lo, hi)) if hi > lo else float(lo)

        def pad_1d(arr: np.ndarray, L: int, pad: float):
            arr = np.asarray(arr, dtype=np.float32)
            out = np.full((L,), pad, dtype=np.float32)
            n = min(arr.size, L)
            if n > 0:
                out[:n] = arr[:n]
            return out, int(n)

        def corrupt_spacings(spacings: np.ndarray, dropout_value: float, poisson_ratio_value: float, gaussian_ratio_value: float):
            """
            spacings: clean spacings for a window (length m)
            returns: corrupted spacings (variable length <= m + poisson inserts, >=1), float32
            """
            s = np.asarray(spacings, dtype=np.float32).copy()
            if s.size == 0:
                return s

            # Work in position space for dropout/poisson, then return to spacings
            pos = np.cumsum(s).astype(np.float32)

            # Dropout: remove some observed points (positions)
            if dropout_value > 0.0:
                keep = rng.uniform(0.0, 1.0, size=pos.size) > dropout_value
                pos_kept = pos[keep]
                if pos_kept.size == 0:
                    pos_kept = pos  # fallback
                pos = pos_kept

            if poisson_ratio_value > 0.0:
                end_value = float(pos[-1]) if pos.size else 0.0
                if end_value > 0.0:
                    poisson_rate = (len(spacings) * float(poisson_ratio_value)) / end_value
                    if poisson_rate > 0.0:
                        current = 0.0
                        extra = []
                        while current < end_value:
                            gap = float(rng.exponential(scale=1.0 / poisson_rate))
                            current += gap
                            if current < end_value:
                                extra.append(current)
                        if extra:
                            pos = np.sort(np.concatenate([pos, np.asarray(extra, dtype=np.float32)]))

            s = np.diff(np.concatenate([[0.0], pos]).astype(np.float32)).astype(np.float32)

            if gaussian_ratio_value > 0.0:
                mean_s = float(np.mean(s)) if s.size else 0.0
                std = mean_s * float(gaussian_ratio_value)
                if std > 0.0:
                    s = s + rng.normal(0.0, std, size=s.size).astype(np.float32)

            s = np.maximum(s, 1e-6).astype(np.float32)

            return s

        self.max_input_len = int(max_input_len)
        self.max_output_len = int(max_output_len)
        self.pad_value = float(pad_value)
        self.normalize_by_clean_mean = bool(normalize_by_clean_mean)

        P_list, X_list, Xlen_list, Y_list, Ylen_list = [], [], [], [], []

        attempts = 0
        valid = 0
        max_attempts = int(attempt_multiplier) * int(number_of_crystals)

        while valid < number_of_crystals and attempts < max_attempts:
            attempts += 1
            crystal_seed = int(rng.integers(0, 2**27 - 1))

            g_val = sample_value(g_lo, g_hi)
            p_val = sample_value(p_lo, p_hi)
            d_val = sample_value(d_lo, d_hi)

            spacings_clean, distances, indices, slope, acceptance_window, poisson_rate, standard_deviation, number_tiles, tiles = quasi_crystal(
                slope_upper=slope_upper,
                slope_lower=slope_lower,
                acceptance_window_lower=acceptance_window_lower,
                acceptance_window_upper=acceptance_window_upper,
                number_of_points=points_per_crystal,
                gaussian_ratio=0.0,
                poisson_ratio=0.0,
                dropout=0.0,
                seed=crystal_seed,
            )

            if number_tiles != 3:
                continue

            spacings_clean = np.asarray(spacings_clean, dtype=np.float32)
            if spacings_clean.size < (max_window_length + 1):
                continue

            pvec = np.array(
                [float(slope), float(acceptance_window),
                 float(tiles[0]), float(tiles[1]), float(tiles[2])],
                dtype=np.float32,
            )

            max_start = spacings_clean.size - max_window_length
            if max_start <= 0:
                continue

            if windows_per_crystal > max_start:
                continue

            start_indices = rng.choice(max_start, size=windows_per_crystal, replace=False).astype(np.int64)

            for start in start_indices:
                L = int(rng.integers(min_window_length, max_window_length + 1))  # target length in spacings
                clean = spacings_clean[start:start + L].astype(np.float32)

                if self.normalize_by_clean_mean:
                    mean_clean = float(np.mean(clean)) if clean.size else 1.0
                    if mean_clean > 0.0:
                        clean_norm = clean / mean_clean
                    else:
                        clean_norm = clean
                else:
                    clean_norm = clean

                corrupted = corrupt_spacings(clean, d_val, p_val, g_val)
                if self.normalize_by_clean_mean:
                    if mean_clean > 0.0:
                        corrupted_norm = corrupted / mean_clean
                    else:
                        corrupted_norm = corrupted
                else:
                    corrupted_norm = corrupted

                X_pad, X_len = pad_1d(corrupted_norm, self.max_input_len, self.pad_value)
                Y_pad, Y_len = pad_1d(clean_norm, self.max_output_len, self.pad_value)

                P_list.append(pvec)
                X_list.append(X_pad)
                Xlen_list.append(X_len)
                Y_list.append(Y_pad)
                Ylen_list.append(Y_len)

            valid += 1

        if valid < number_of_crystals:
            raise RuntimeError(f"Only collected {valid}/{number_of_crystals} valid crystals (attempts={attempts})")

        self.P = torch.from_numpy(np.stack(P_list).astype(np.float32))                 # (N, p_dim)
        self.X = torch.from_numpy(np.stack(X_list).astype(np.float32))                 # (N, max_input_len)
        self.X_len = torch.tensor(Xlen_list, dtype=torch.long)                         # (N,)
        self.Y = torch.from_numpy(np.stack(Y_list).astype(np.float32))                 # (N, max_output_len)
        self.Y_len = torch.tensor(Ylen_list, dtype=torch.long)                         # (N,)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.P[idx], self.X[idx], self.X_len[idx], self.Y[idx], self.Y_len[idx]


class QCDatasetSequentialNext(Dataset):
    """
    Sequential sliding-window next-symbol dataset.

    Each crystal contributes num_windows windows:
      X[i] = symbols[start : start + window_length]
      y[i] = symbols[start + window_length]
    with start = 0..num_windows-1 (per crystal).
    """

    def __init__(self,
                 num_crystals: int,
                 len_crystal: int,
                 window_length: int,
                 num_windows: int,
                 slope_lower: float = 1.0,
                 slope_upper: float = 5.0,
                 acceptance_window_lower: float = 0.5,
                 acceptance_window_upper: float = 5.0,
                 seed: int = 123,
                 attempt_multiplier: int = 50,
                 require_num_tiles: int = 3,
                 ):
        super().__init__()

        self.num_crystals = int(num_crystals)
        self.len_crystal = int(len_crystal)
        self.window_length = int(window_length)
        self.num_windows = int(num_windows)

        if self.window_length < 1:
            raise ValueError("window_length must be >= 1")
        if self.num_windows < 1:
            raise ValueError("num_windows must be >= 1")

        min_required = self.window_length + self.num_windows
        if self.len_crystal <= min_required:
            raise ValueError(
                f"len_crystal too small. Need len_crystal > window_length + num_windows "
                f"({self.len_crystal} vs {self.window_length}+{self.num_windows}). "
                f"Try increasing len_crystal or decreasing num_windows/window_length."
            )

        rng = np.random.default_rng(seed=seed)

        self.crystals = []   

        attempts = 0
        valid = 0
        max_attempts = attempt_multiplier * self.num_crystals

        while valid < self.num_crystals and attempts < max_attempts:
            attempts += 1
            crystal_seed = int(rng.integers(0, 2**27 - 1))

            spacings, distances, indices, slope, acceptance_window, poisson_rate, std, number_tiles, tiles = quasi_crystal(
                slope=0,
                slope_lower=slope_lower,
                slope_upper=slope_upper,
                acceptance_window=0,
                acceptance_window_lower=acceptance_window_lower,
                acceptance_window_upper=acceptance_window_upper,
                number_of_points=self.len_crystal,
                gaussian_ratio=0.0,
                poisson_ratio=0.0,
                dropout=0.0,
                seed=crystal_seed,
            )

            if require_num_tiles is not None and number_tiles != require_num_tiles:
                continue

            sym, centers = wordify(spacings, alphabet=(0, 1, 2))
            sym = np.asarray(sym, dtype=np.int64)

            if sym.size <= (self.window_length + self.num_windows):
                continue

            self.crystals.append({
                "sym_ids": sym,                       
                "spacings": np.asarray(spacings, dtype=np.float32),
                "slope": float(slope),
                "acceptance_window": float(acceptance_window),
                "tiles": np.asarray(tiles, dtype=np.float32) if tiles is not None else None,
                "seed": crystal_seed,
            })

            valid += 1

            if attempts % 100 == 0:
                print(f"[attempts={attempts}] valid={valid}/{self.num_crystals}")

        if valid < self.num_crystals:
            raise RuntimeError(f"Only collected {valid}/{self.num_crystals} valid crystals (attempts={attempts})")

        self._length = self.num_crystals * self.num_windows

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int):
        c = int(idx // self.num_windows)
        start = int(idx % self.num_windows)

        sym = self.crystals[c]["sym_ids"]

        x = sym[start:start + self.window_length]
        y = sym[start + self.window_length]

        return (
            torch.tensor(x, dtype=torch.long),  # (window_length,)
            torch.tensor(y, dtype=torch.long),  # scalar
        )

    def get_crystal_meta(self, crystal_index: int) -> dict:
        """Convenience: per-crystal parameters for logging/debugging."""
        d = self.crystals[int(crystal_index)]
        return {
            "slope": d["slope"],
            "acceptance_window": d["acceptance_window"],
            "tiles": None if d["tiles"] is None else d["tiles"].copy(),
            "seed": d["seed"],
        }


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalTransformer(nn.Module):
    """
    Decoder-only transformer for next-token prediction.

    Input:  x (B, T) of token ids in [0, vocab_size-1]
    Output: logits (B, T, vocab_size) for next-token at each position
    """

    def __init__(
        self,
        vocab_size: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 512,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
        param_dim: int = 0,
        param_tokens: int = 0,
        param_hidden: int = 128,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.param_dim = int(param_dim)
        self.param_tokens = int(param_tokens)
        self.param_hidden = int(param_hidden)

        self.tok_emb = nn.Embedding(vocab_size, d_model)

        total_len = max_len + max(self.param_tokens, 0)
        self.pos_emb = nn.Embedding(total_len, d_model)

        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,     # (B, T, D)
            norm_first=True,      # Pre-LN is usually more stable
        )
        # Avoid noisy nested-tensor warning with norm_first=True; behavior is unchanged.
        self.blocks = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self.ln_f = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        if self.param_tokens > 0:
            mask = torch.triu(torch.ones(total_len, total_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

        self.param_mlp = None
        if self.param_tokens > 0:
            if self.param_dim <= 0:
                raise ValueError("param_dim must be > 0 when param_tokens > 0")
            self.param_mlp = nn.Sequential(
                nn.Linear(self.param_dim, self.param_hidden),
                nn.ReLU(),
                nn.Linear(self.param_hidden, d_model * self.param_tokens),
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, param_vec: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, T) token ids
        returns logits: (B, T, vocab_size)
        """
        B, T = x.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length T={T} exceeds max_len={self.max_len}")

        tok = self.tok_emb(x)
        if self.param_tokens > 0:
            if param_vec is None:
                raise ValueError("param_vec is required when param_tokens > 0")
            if param_vec.dim() != 2 or param_vec.size(0) != B or param_vec.size(1) != self.param_dim:
                raise ValueError(
                    f"param_vec must be (B, {self.param_dim}); got {tuple(param_vec.shape)}"
                )
            pfx = self.param_mlp(param_vec).view(B, self.param_tokens, self.d_model)
            h = torch.cat([pfx, tok], dim=1)
        else:
            h = tok

        pos = torch.arange(h.size(1), device=x.device).unsqueeze(0)  # (1, T+P)
        h = h + self.pos_emb(pos)
        h = self.drop(h)

        attn_mask = self.causal_mask[: h.size(1), : h.size(1)]                 # (T+P, T+P)
        h = self.blocks(h, mask=attn_mask)                   # (B, T, D)

        h = self.ln_f(h)
        logits = self.lm_head(h)                             # (B, T+P, V)
        if self.param_tokens > 0:
            logits = logits[:, self.param_tokens :, :]
        return logits

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forcing loss: predict next token at each position.
        x: (B, T) tokens
        """
        logits = self.forward(x[:, :-1])                     # predict positions 1..T-1
        target = x[:, 1:]                                    # (B, T-1)
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target.reshape(-1),
        )

    @torch.no_grad()
    def rollout(
        self,
        prefix: torch.Tensor,
        steps: int,
        temperature: float = 1.0,
        greedy: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        prefix: (T0,) or (1, T0) LongTensor of token ids
        steps: how many new tokens to generate
        returns: (T0 + steps,) tokens
        """
        self.eval()

        if prefix.dim() == 1:
            x = prefix.unsqueeze(0)  # (1, T0)
        else:
            x = prefix

        for _ in range(int(steps)):
            x_cond = x[:, -self.max_len:]
            logits = self.forward(x_cond)                    # (1, Tc, V)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            if greedy:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)  # (1,1)
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)          # (1,1)

            x = torch.cat([x, next_id], dim=1)

        return x.squeeze(0)
