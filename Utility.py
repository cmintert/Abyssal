import numpy as np

def scale_values_to_range(values, new_min=0, new_max=1):
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
    scaled_values = new_min + (
        (values - original_min) * (new_max - new_min) / (original_max - original_min)
    )

    return scaled_values


class StarNames:

    def __init__(self):
        # Combined elements for star naming
        self.prefixes = ["Al", "Be", "Si", "Ar", "Ma", "Cy", "De", "Er", "Fi", "Gi"]
        self.middles = [
            "pha",
            "ri",
            "gel",
            "min",
            "con",
            "bel",
            "dra",
            "lon",
            "nar",
            "tel",
        ]
        self.suffixes = ["us", "a", "ae", "ion", "ium", "is", "or", "os", "um", "ix"]
        self.constellations = [
            "And",
            "Aqr",
            "Aql",
            "Ari",
            "Aur",
            "Boo",
            "Cyg",
            "Gem",
            "Her",
            "Leo",
            "Lyr",
            "Ori",
            "Peg",
            "Per",
            "Tau",
            "UMa",
            "Vir",
        ]
        self.designations = [
            "Alpha",
            "Beta",
            "Gamma",
            "Delta",
            "Epsilon",
            "Zeta",
            "Eta",
            "Theta",
            "Iota",
            "Kappa",
        ]
        self.catalogs = ["HD", "HIP"]

    # Combined star name generation function

    def generate_combined_star_name(self):
        name = (
            np.random.choice(self.prefixes)
            + np.random.choice(self.middles)
            + np.random.choice(self.suffixes)
        )
        constellation = np.random.choice(self.constellations)
        designation = np.random.choice(self.designations)
        catalog = np.random.choice(self.catalogs)
        number = np.random.randint(1, 9999)

        # Format options for the combined approach
        format_options = [
            f"{name}",
            f"{name} {constellation}",
            f"{designation} {constellation}",
            f"{catalog} {number}",
            f"{catalog} {number} ({name})",
            f"{catalog} {number} ({designation} {constellation})",
        ]

        # Randomly select a format option
        return np.random.choice(format_options)
