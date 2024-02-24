import numpy as np


def scale_values_to_range(values, new_min = 0, new_max =1):
    """
    Scale an array of values to a new specified range.

    Parameters:
    - values: numpy array of original values.
    - new_min: The minimum value of the new range.
    - new_max: The maximum value of the new range.

    Returns:
    - scaled_values: numpy array of values scaled to the new range.
    """
    # Ensure input is a numpy array
    values = np.array(values)

    # Calculate the original range
    original_min = values.min()
    original_max = values.max()

    # Scale the values to the new range
    scaled_values = new_min + ((values - original_min) * (new_max - new_min) / (original_max - original_min))

    return scaled_values