import numpy as np


def insert_linebreaks(text, max_line_length=50):
    words = text.split()
    current_line_length = 0
    lines = []
    current_line = []

    for word in words:
        if current_line_length + len(word) > max_line_length:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_line_length = len(word) + 1  # plus one for the space
        else:
            current_line.append(word)
            current_line_length += len(word) + 1  # plus one for the space

    lines.append(" ".join(current_line))  # Add the last line

    return "<br>".join(lines)


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


class RareMinerals:
    properties = {
        "WaterIce": {
            "rarity": 3,
            "value": 100,
            "description": "Vital for life support and fuel in space.",
        },
        "Rock": {
            "rarity": 3,
            "value": 50,
            "description": "Used in construction and as a raw material.",
        },
        "Platinum": {
            "rarity": 0.05,
            "value": 900,
            "description": "Used in electronics, jewelry, and catalysts.",
        },
        "Palladium": {
            "rarity": 0.04,
            "value": 850,
            "description": "Important for fuel cells and electronics.",
        },
        "Gold": {
            "rarity": 0.06,
            "value": 800,
            "description": "Valuable for electronics, currency, and decoration.",
        },
        "Neodymium": {
            "rarity": 0.1,
            "value": 750,
            "description": "Essential for powerful magnets in motors and generators.",
        },
        "Rhodium": {
            "rarity": 0.02,
            "value": 1000,
            "description": "Used in high-performance alloys and catalytic converters.",
        },
        "Iridium": {
            "rarity": 0.03,
            "value": 950,
            "description": "Used in high-temperature and corrosion-resistant alloys.",
        },
        "Titanium": {
            "rarity": 0.2,
            "value": 500,
            "description": "Strong, lightweight, used in aerospace and medical implants.",
        },
        "Helium3": {
            "rarity": 0.07,
            "value": 1200,
            "description": "Potential fuel for nuclear fusion, rare on Earth.",
        },
        "Lithium": {
            "rarity": 0.15,
            "value": 600,
            "description": "Critical for rechargeable batteries and mood-stabilizing drugs.",
        },
        "Copper": {
            "rarity": 0.5,
            "value": 400,
            "description": "Crucial for electrical wiring and electronics.",
        },
        "Nickel": {
            "rarity": 0.45,
            "value": 350,
            "description": "Used in stainless steel and batteries.",
        },
        "Tantalum": {
            "rarity": 0.08,
            "value": 700,
            "description": "Important for electronic components and mobile phones.",
        },
        "Cobalt": {
            "rarity": 0.09,
            "value": 650,
            "description": "Critical for rechargeable batteries and superalloys.",
        },
        "Xenon": {
            "rarity": 0.02,
            "value": 1200,
            "description": "Used in lighting, electronics, and space propulsion.",
        },
        "Phosphorus": {
            "rarity": 0.3,
            "value": 500,
            "description": "Essential for agriculture and biochemical processes.",
        },
        "Iron": {
            "rarity": 0.6,
            "value": 300,
            "description": "Used in construction, vehicles, and steel.",
        },
        "Silicon": {
            "rarity": 0.4,
            "value": 450,
            "description": "Critical for electronics, solar panels, and glass.",
        },
        # Add more as needed
    }

    def get_minerals(self):
        """Returns a list of all available minerals."""
        return list(self.properties.keys())

    @staticmethod
    def get_mineral_info(mineral_name):
        """Returns the properties of the requested mineral."""
        return RareMinerals.properties.get(mineral_name, None)

    def get_random_mineral(self, number_of_minerals=1):
        """Returns a list of unique random minerals and their properties."""
        all_mineral_names = list(self.properties.keys())
        number_of_minerals = min(
            number_of_minerals, len(all_mineral_names)
        )  # Ensure request does not exceed available minerals

        selected_mineral_names = np.random.choice(
            all_mineral_names, size=number_of_minerals, replace=False
        )
        random_minerals = [
            (mineral_name, self.properties[mineral_name])
            for mineral_name in selected_mineral_names
        ]

        return random_minerals
