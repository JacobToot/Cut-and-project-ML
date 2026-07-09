import numpy as np

def random_step_1d(step_size, positive_probability, number_of_points = 100):
    """Generates a 1D step pattern where each step is increased by a fixed step size in either the positive or negative direction with a given probability.

    Args:
        step_size (float): fixed length of each step.
        positive_probability (float): probability of taking a positive step (between 0 and 1).
        number_of_points (int, optional): number of points to generate. Defaults to 100.
    
    Returns:
        sequence (np.ndarray): sorted array of distances of point i to point i + 1."""
    
    sequence = np.zeros(number_of_points + 1)
    step = step_size

    for i in range(number_of_points):
        if np.random.rand() < positive_probability:
            step = step + step_size
        else:
            step = step - step_size

        sequence[i+1] = sequence[i] + step

    sequence = np.diff(sequence)
    
    return sequence