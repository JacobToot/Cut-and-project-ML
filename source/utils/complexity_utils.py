import numpy as np
from simulations.quasi_crystal import _find_tiles_robust

def wordify(spacings, centers=None, alphabet=("A","B","C")):
    x = np.asarray(spacings, float)
    x = x[np.isfinite(x)]

    if centers is None:
        centers = _find_tiles_robust(x).astype(np.float64)
    else:
        centers = np.asarray(centers, dtype=np.float64)

    if len(centers) not in (2, 3):
        raise ValueError(f"Found {len(centers)} length clusters; expected 2 or 3. "
                         f"Try different tol or ensure no noise is added.")

    d = np.abs(x[:, None] - centers[None, :])
    idx = np.argmin(d, axis=1)

    used_letters = alphabet[:len(centers)]
    word = np.array([used_letters[i] for i in idx], dtype="U1")

    return word, centers

def complexity_Cn(word, n):
    """
    Exact factor complexity of a finite word: number of distinct length-n factors for given word. 
    word: array/list/string of symbols
    n: integer that determines length of subwords to check for
    returns: integer value representing the number of distinct subwords in given word with length n.
    """
    N = len(word)
    if n <= 0:
        return 1
    if n > N:
        return 0
    return len({tuple(word[i:i+n]) for i in range(N - n + 1)})

def profile(word, n_max):
    return np.array([complexity_Cn(word, n) for n in range(1, n_max + 1)], dtype=int)

def complexity_report(word, n_max=30):
    
    """
    Compute C(n) for n=1..n_max and compare to 2n+1.
    Returns a dict with arrays.
    """
    
    C = profile(word, n_max)
    n = np.arange(1, n_max + 1)
    target = 2*n + 1

    cap = (len(word) - n + 1)
    target_capped = np.minimum(target, cap)

    return {
        "n": n,
        "C": C,
        "target_2n_plus_1": target,
        "target_capped": target_capped,
        "difference": C - target_capped,
    }

def complexity_pipeline(spacings, n_max=500, tol=1e-8):
    word, centers = wordify(spacings, tol=tol)

    N = len(word)
    n = np.arange(1, n_max + 1)

    C = np.array([complexity_Cn(word, k) for k in n])
    target = 2*n + 1
    target_capped = np.minimum(target, N - n + 1)

    return {
        "word": word,
        "centers": centers,
        "n": n,
        "C": C,
        "target_capped": target_capped,
        "N": N
    }