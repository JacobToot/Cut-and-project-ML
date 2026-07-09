"""This module contains functions related to computation. """

from __future__ import annotations
import random
import math
import numpy as np


def generate_irrational(upper_limit=10, max_value_integer=10**5, lower_limit=0, rng=None):
    """
    Generate an irrational number in [lower_limit, upper_limit).

    Parameters
    ----------
    upper_limit, lower_limit : float
        Bounds for the output.
    max_value_integer : int
        Upper bound for the random integer n whose sqrt provides the
        irrational fractional part.
    rng : np.random.Generator or None
        If provided, use this RNG for reproducibility.  Falls back to the
        global ``random`` module when *None* (legacy behaviour).
    """
    while True:
        if rng is not None:
            n = int(rng.integers(1, max_value_integer + 1))
        else:
            n = random.randint(1, max_value_integer)
        n_sqrt = math.sqrt(n)
        if not n_sqrt.is_integer():
            frac = n_sqrt % 1.0
            return lower_limit + frac * (upper_limit - lower_limit)


def continued_fraction(factors=[1, 2], k_max=5):
    """Calculates the continued fraction given factors."""
    fraction = 0
    period = factors[1:]
    period_length = len(period)
    start = -(k_max - 1) % period_length + 1
    for n in range(k_max - 1):
        fraction = 1 / (fraction + period[(start + n) % period_length])
    return factors[0] + fraction


def truth_frequencies(full_word, sub_word, alphabet=['A', 'B', 'C']):

    truths = []
    n = len(sub_word)
    truth_array = [0, 0, 0]

    for i in range(n, len(full_word)):
        if (full_word[i - n:i] == sub_word).all():
            truths.append(full_word[i])

    truth_counts = np.unique(truths, return_counts=True)

    for letter, count in zip(*truth_counts):
        idx = alphabet.index(letter)
        truth_array[idx] = count
        
    total_count = np.sum(truth_array)
    return np.array([(count / total_count) for count in truth_array])


def lookup_table(word, window_length, alphabet=(0, 1, 2)):
    word = np.asarray(word, dtype=np.int16)
    length = int(window_length)
    if word.ndim != 1:
        raise ValueError("word must be 1D")
    if len(word) <= length:
        raise ValueError("word too short for window_length")
    if len(alphabet) != 3:
        raise ValueError("alphabet must have length 3 for (fA,fB,fC)")
    idxs = {int(alphabet[i]): i for i in range(3)}
    counts = {}
    for i in range(len(word) - length):
        sub_word = tuple(word[i:i + length])
        next_sym = int(word[i + length])
        j = idxs.get(next_sym, None)
        if j is None:
            continue
        row = counts.get(sub_word)
        if row is None:
            row = np.zeros(3, dtype=np.int64)
            counts[sub_word] = row
        row[j] += 1
    table = np.empty((len(counts), 4), dtype=object)
    for r, (sub_word, row) in enumerate(counts.items()):
        s = int(row.sum())
        if s == 0:
            f = (1 / 3, 1 / 3, 1 / 3)
        else:
            f = (row[0] / s, row[1] / s, row[2] / s)
        table[r, 0] = sub_word
        table[r, 1] = float(f[0])
        table[r, 2] = float(f[1])
        table[r, 3] = float(f[2])
    return table


def _interval_for_gstar(g: float, W: float) -> tuple[float, float]:
    """
    Return [a, b) such that x_k* in [a,b) guarantees both x_k* and
    x_{k+1}* = x_k* + g are in (-W, W).

    a = max(-W, -W - g)
    b = min( W,  W - g)
    This is the fundamental geometric constraint.
    """
    return max(-W, -W - g), min(W, W - g)


def compute_valid_interval(
    symbolic_sequence,
    internal_gaps,
    W: float,
    tol: float = 1e-10,
) -> tuple[float, float]:
    """
    Compute the feasible interval [x0_min, x0_max) for x_0*.

    x0_min > x0_max means infeasible.
    """
    ds = np.asarray(internal_gaps, dtype=np.float64).ravel()
    syms = np.asarray(symbolic_sequence, dtype=np.int64).ravel()
    n = int(syms.size)

    if n == 0:
        return float(-W), float(W)

    if np.any((syms < 0) | (syms >= ds.size)):
        raise ValueError(f"Symbol ids out of range [0, {ds.size}).")

    deltas = ds[syms]
    C = np.empty(n, dtype=np.float64)
    C[0] = 0.0
    if n > 1:
        C[1:] = np.cumsum(deltas[:-1])

    lows  = np.empty(n, dtype=np.float64)
    highs = np.empty(n, dtype=np.float64)
    
    for k in range(n):
        a, b = _interval_for_gstar(float(deltas[k]), W)
        lows[k]  = a - C[k]
        highs[k] = b - C[k]

    return float(lows.max()), float(highs.min())


def is_valid_sequence(
    symbolic_sequence,
    internal_gaps,
    W: float,
    tol: float = 1e-10,
) -> bool:
    """Return True iff the sequence is consistent with a cut-and-project sequence."""
    x1_min, x1_max = compute_valid_interval(symbolic_sequence, internal_gaps, W, tol=tol)
    return x1_min <= x1_max + tol


def next_symbol_probabilities(
    symbolic_sequence,
    internal_gaps,
    W: float,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Probability distribution over the next symbol.

    Assumes x_1* uniform over [x1_min, x1_max).
    P(next=s) = overlap(x_{n+1}* range, [a_s, b_s)) / valid_width.

    Raises ValueError if the sequence is infeasible.
    """
    ds = np.asarray(internal_gaps, dtype=np.float64).ravel()
    n_syms = int(ds.size)

    x1_min, x1_max = compute_valid_interval(symbolic_sequence, ds, W, tol=tol)
    if x1_min > x1_max + tol:
        raise ValueError(
            f"Infeasible: x1_min={x1_min:.9g} > x1_max={x1_max:.9g}."
        )
    x1_max = max(x1_max, x1_min)

    syms = np.asarray(symbolic_sequence, dtype=np.int64).ravel()
    C_next = float(ds[syms].sum()) if syms.size > 0 else 0.0

    xn_min = x1_min + C_next
    xn_max = x1_max + C_next

    probs = np.zeros(n_syms, dtype=np.float64)
    for s in range(n_syms):
        a_s, b_s = _interval_for_gstar(float(ds[s]), W)
        probs[s] = max(0.0, min(xn_max, b_s) - max(xn_min, a_s))

    total = float(probs.sum())
    if total < tol:
        return np.full(n_syms, 1.0 / n_syms, dtype=np.float64)
    return probs / total