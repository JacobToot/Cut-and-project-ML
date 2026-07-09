import numpy as np

def gaussian_step_1d(step_size = 1, std_dev = 0.1, number_of_points = 100):
    """ Generates a 1D step pattern where the average step length is step_size and the step lengths are normally distributed with standard deviation std_dev. 
    Args:
        step_size (float, optional): average step length. Defaults to 1.
        std_dev (float, optional): standard deviation of step lengths. Defaults to 0.1.
        number_of_points (int, optional): number of points to generate. Defaults to 100.
        
    Returns: 
        sequence (np.ndarray): array of distances of point i to point i + 1."""

    sequence = np.zeros(number_of_points + 1)

    for i in range(number_of_points):
        sequence[i + 1] = sequence[i] + np.random.normal(loc=step_size, scale=std_dev)
    
    # convert to difference in consecutive distances
    sequence = np.diff(sequence)

    return sequence