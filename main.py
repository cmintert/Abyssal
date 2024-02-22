import random
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math
import plotly.graph_objects as go

SPECTRAL_CLASSES_LUM_MASS_RATIO = {
    "G-Type": 1 / 4,
    "K-Type": 1 / 3.5,
    "M-Type": 1 / 3,
}

class Nation:
    def __init__(self, name, origin=None, expansion_rate=5, space_boundary=500):
        self.name = name
        self.expansion_rate = expansion_rate
        self.current_radius = 0
        self.nation_stars = []
        if origin is None:
            # If no origin is provided, generate a random point within the boundary
            self.origin = {
                "x": np.random.uniform(-space_boundary, space_boundary),
                "y": np.random.uniform(-space_boundary, space_boundary),
                "z": np.random.uniform(-space_boundary, space_boundary)
            }
        else:
            self.origin = origin

    def expand_influence(self):
        """Expand the nation's sphere of influence based on its expansion rate."""
        self.current_radius += self.expansion_rate

    def is_star_within_influence(self, star):
        """Determine if a given star is within the nation's sphere of influence."""
        distance = np.sqrt(
            (star.x - self.origin["x"])**2 +
            (star.y - self.origin["y"])**2 +
            (star.z - self.origin["z"])**2
        )
        return distance <= self.current_radius

    def __str__(self):
        return f"{self.name}: Origin at {self.origin}, Current Radius: {self.current_radius}, Expansion Rate: {self.expansion_rate}"


class StarNames():

    def __init__(self):
        # Combined elements for star naming
        self.prefixes = ['Al', 'Be', 'Si', 'Ar', 'Ma', 'Cy', 'De', 'Er', 'Fi', 'Gi']
        self.middles = ['pha', 'ri', 'gel', 'min', 'con', 'bel', 'dra', 'lon', 'nar', 'tel']
        self.suffixes = ['us', 'a', 'ae', 'ion', 'ium', 'is', 'or', 'os', 'um', 'ix']
        self.constellations = ['And', 'Aqr', 'Aql', 'Ari', 'Aur', 'Boo', 'Cyg', 'Gem', 'Her', 'Leo', 'Lyr', 'Ori', 'Peg', 'Per',
            'Tau', 'UMa', 'Vir']
        self.designations = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa']
        self.catalogs = ['HD', 'HIP']

    # Combined star name generation function

    def generate_combined_star_name(self):
        name = random.choice(self.prefixes) + random.choice(self.middles) + random.choice(self.suffixes)
        constellation = random.choice(self.constellations)
        designation = random.choice(self.designations)
        catalog = random.choice(self.catalogs)
        number = random.randint(1, 9999)

        # Format options for the combined approach
        format_options = [
            f"{name}",
            f"{name} {constellation}",
            f"{designation} {constellation}",
            f"{catalog} {number}",
            f"{catalog} {number} ({name})",
            f"{catalog} {number} ({designation} {constellation})"
        ]

        # Randomly select a format option
        return random.choice(format_options)

class Star:
    def __init__(self, id, name=None, x=None, y=None, z=None, r=None, theta=None, phi=None, spectral_class=None, luminosity=None):
        self.id = id
        self.name = name
        # Cartesian coordinates
        self.x = x
        self.y = y
        self.z = z
        # Spherical coordinates
        self.r = r
        self.theta = theta
        self.phi = phi
        # Class and Luminosity
        self.spectral_class = spectral_class
        self.luminosity = luminosity
        # Mass
        self.mass = self.calculate_mass()
        # If spherical coordinates are provided, convert them to Cartesian
        if r is not None and theta is not None and phi is not None:
            self.convert_to_cartesian()
        # Alternatively, if Cartesian coordinates are provided, convert them to spherical
        elif x is not None and y is not None and z is not None:
            self.convert_to_spherical()

    def convert_to_cartesian(self):
        """Converts spherical coordinates to Cartesian coordinates and updates the star's position."""
        self.x = self.r * math.sin(self.phi) * math.cos(self.theta)
        self.y = self.r * math.sin(self.phi) * math.sin(self.theta)
        self.z = self.r * math.cos(self.phi)

    def convert_to_spherical(self):
        """Converts Cartesian coordinates to spherical coordinates and updates the star's position."""
        self.r = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.theta = math.atan2(self.y, self.x)
        self.phi = math.acos(self.z / self.r)

    def set_spectral_class(self, spectral_class):
        self.spectral_class = spectral_class

    def set_luminosity(self, luminosity):
        self.luminosity = luminosity
        self.mass = self.calculate_mass()
        print(f"Mass of star {self.id} is changed to {self.mass} fitting the new luminosity.")

    def calculate_mass(self):
        """Calculates the mass of the star based on its luminosity and spectral class."""
        if self.luminosity is not None and self.spectral_class in SPECTRAL_CLASSES_LUM_MASS_RATIO:
            exponent = SPECTRAL_CLASSES_LUM_MASS_RATIO[self.spectral_class]
            mass_relative_to_sun = self.luminosity ** exponent
            return mass_relative_to_sun
        raise ValueError("Luminosity or spectral class not set.")

    def __str__(self):
        return f"Star ID: {self.id}, Cartesian: ({self.x}, {self.y}, {self.z}), Spherical: (r={self.r}, theta={self.theta}, phi={self.phi})"

# Generate stars
class Starmap:
    def __init__(self):
        self.stars = []
        self.nations = []
        self.used_star_names = []
        self.spectral_classes = {
            "G-Type": (0.6,1.5),
            "K-Type": (0.08,0.6),
            "M-Type": (0.02,0.08),
        }
        self.star_names = StarNames()

    def generate_star_name(self):
        """Generates a star name that is not already in use."""
        while True:
            name = self.star_names.generate_combined_star_name()
            if name not in self.used_star_names:
                self.used_star_names.append(name)
                return name

    def generate_stars(self, seed=50):
        np.random.seed(seed)
        for i in range(500):
            # Generate random spherical coordinates
            r = 500 * (np.random.uniform(0, 1) ** (1/3))
            theta = 2 * np.random.uniform(0, 1) * np.pi
            phi = np.random.uniform(0, 1) * np.pi
            # Create random spectral class
            spectral_class = np.random.choice(list(self.spectral_classes.keys()), p=[0.1, 0.2, 0.7])
            # Create random luminosity
            luminosity = np.random.uniform(*self.spectral_classes[spectral_class])
            # Create unique name
            name = self.generate_star_name()
            # Set star properties
            current_star = Star(i, name=name,r=r, theta=theta, phi=phi, spectral_class=spectral_class, luminosity=luminosity)
            # add star to map
            self.stars.append(current_star)

    def generate_nations(self, n=5, space_boundary=500):
        for i in range(n):
            name = f"Nation {i + 1}"
            new_nation = Nation(name=name, space_boundary=space_boundary)
            self.nations.append(new_nation)

    def asign_stars_to_nations(self):
        while True:
            for star in self.stars:
                for nation in self.nations:
                    if nation.is_star_within_influence(star):
                        if not any(star in n.nation_stars for n in self.nations):
                            nation.nation_stars.append(star)
                            break
                    nation.expand_influence()
            # Check if all stars are assigned
            if all(any(star in n.nation_stars for n in self.nations) for star in self.stars):
                break

    def get_luminosities(self):
        return np.array([star.luminosity for star in self.stars])

    def get_masses(self):
        return np.array([star.mass for star in self.stars])

    def get_normalized_luminosities(self, weight=1):
        # Extract luminosities into a NumPy array for efficient computation
        luminosities = np.array([star.luminosity for star in self.stars])
        # Normalize luminosities to the range [0, 1]
        normalized_luminosities = ((luminosities - luminosities.min()) / (luminosities.max() - luminosities.min())) * weight
        return np.array(normalized_luminosities)

    def get_normalized_masses(self, weight=1):
        # Extract masses into a NumPy array for efficient computation
        masses = np.array([star.mass for star in self.stars])
        # Normalize masses to the range [0, 1]
        normalized_masses = ((masses - masses.min()) / (masses.max() - masses.min())) * weight
        return np.array(normalized_masses)

    def plot(self):
        masses = self.get_normalized_masses(weight=15)
        luminosities = self.get_normalized_luminosities(weight=2)

        # Create a trace for the stars
        trace = go.Scatter3d(
            x=[star.x for star in self.stars],
            y=[star.y for star in self.stars],
            z=[star.z for star in self.stars],
            mode='markers+text',
            marker=dict(
                size=masses,
                color=luminosities,  # set color to an array/list of desired values
                colorscale='solar',  # choose a colorscale
                opacity=0.8
            ),
            text=[star.name for star in self.stars],
            hoverinfo='text'
        )

        data = [trace]

        # Create layout for the plot
        layout = go.Layout(
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            ),
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
            )
        )

        fig = go.Figure(data=data, layout=layout)
        fig.show()

    def __str__(self):
        # Count spectral classes
        spectral_classes_count = {}
        for star in self.stars:
            if star.spectral_class in spectral_classes_count:
                spectral_classes_count[star.spectral_class] += 1
            else:
                spectral_classes_count[star.spectral_class] = 1

        # Print count of each spectral class
        for spectral_class, count in spectral_classes_count.items():
            print(f"{spectral_class}: {count}")

        # Get masses and luminosities
        masses = self.get_masses()
        luminosities = self.get_luminosities()

        # Print highest and lowest luminosities and masses before normalization
        print("Highest luminosity:", luminosities.max())
        print("Lowest luminosity:", luminosities.min())
        print("Highest mass:", masses.max())
        print("Lowest mass:", masses.min())
        return "-----"

actualmap = Starmap()
actualmap.generate_stars(49)
actualmap.generate_nations(5)
actualmap.asign_stars_to_nations()
print(actualmap)
actualmap.plot()
print(actualmap.used_star_names)
