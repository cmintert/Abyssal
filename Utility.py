import numpy as np


def insert_linebreaks(text, max_line_length=50):
    words = text.split()
    current_line_length = 0
    lines = []
    current_line = []

    for word in words:
        if current_line_length + len(word) > max_line_length:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_line_length = len(word) + 1  # plus one for the space
        else:
            current_line.append(word)
            current_line_length += len(word) + 1  # plus one for the space

    lines.append(' '.join(current_line))  # Add the last line

    return '<br>'.join(lines)


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
        return [
            np.random.choice(format_options),
            name,
            constellation,
            designation,
            catalog,
            number,
        ]


class PlanetNames:
    def __init__(self, star, orbit_number=0):
        # Combined elements for planet naming
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
            "Lambda",
            "Mu",
            "Nu",
            "Xi",
            "Omicron",
            "Pi",
            "Rho",
            "Sigma",
        ]
        self.star = star
        self.orbit_number = orbit_number

        # Combined planet name generation function

    def generate_combined_planet_name(self):
        name = (
            np.random.choice(self.prefixes)
            + np.random.choice(self.middles)
            + np.random.choice(self.suffixes)
        )
        star_name = self.star.name
        orbit_number = self.orbit_number

        # transform orbit_number to roman numeral
        roman_numerals = [
            "I",
            "II",
            "III",
            "IV",
            "V",
            "VI",
            "VII",
            "VIII",
            "IX",
            "X",
            "XI",
            "XII",
            "XIII",
            "XIV",
            "XV",
        ]
        orbital_roman_numeral = roman_numerals[orbit_number]

        # Format options for the combined approach
        format_options = [
            f"{name}",
            f"{star_name[1]} {orbital_roman_numeral}",
        ]

        # Randomly select a format option
        name = np.random.choice(format_options)
        return name
