import numpy as np

def permutation_1d(sequence, number_of_pairs=100):
    """Takes in an input sequence and permutes a number of specified pairs by swapping their indices.

    Args:
        sequence (array-like): The input sequence to be permuted.
        number_of_pairs (int, optional): The number of permuted sequences to generate. Defaults to 100.

    Returns:
        list: A list containing the permuted sequences.
    """

    n = len(sequence)
    copy = sequence.copy()
    vals = np.random.choice(n, size=(number_of_pairs // 2, 2), replace=False)

    for k in range(number_of_pairs // 2):
        i, j = vals[k]

        attempts = 0
        while copy[i] == copy[j] and attempts < n:
            attempts += 1
            j = (j + 1) % (n-1)
        
        copy[i], copy[j] = copy[j], copy[i]

    return copy