import numpy as np

import plotly.graph_objects as go
from plotly.offline import plot

from Map_Components import Nation, Star, Planetary_System, SmallBody, Planet
from Utility import scale_values_to_range, StarNames


# Generate stars
class Starmap:
    def __init__(self):
        self.stars = []
        self.nations = []
        self.spectral_classes = {
            "G-Type": (0.6, 1.5),
            "K-Type": (0.08, 0.6),
            "M-Type": (0.02, 0.08),
        }
        self.used_star_names = []

    def generate_stars(self, number_of_stars=500):

        for i in range(number_of_stars):
            # Generate random spherical coordinates
            r = 500 * (np.random.uniform(0, 1) ** (1 / 3))
            theta = 2 * np.random.uniform(0, 1) * np.pi
            phi = np.arccos(2 * np.random.uniform(0, 1) - 1)
            # Create random spectral class
            spectral_class = np.random.choice(
                list(self.spectral_classes.keys()), p=[0.1, 0.2, 0.7]
            )
            # Create random luminosity
            luminosity = np.random.uniform(*self.spectral_classes[spectral_class])
            # Set star properties
            current_star = Star(
                i,
                starmap=self,
                r=r,
                theta=theta,
                phi=phi,
                spectral_class=spectral_class,
                luminosity=luminosity,
            )
            # Instance the planetary system
            current_star.planetary_system = Planetary_System(current_star)
            # Generate orbits
            # generate random integer number between 1 and 10 for number of orbits
            orbits = np.random.randint(1, 11)
            current_star.planetary_system.generate_orbits(
                include_habitable_zone=True, num_orbits=orbits
            )
            # generate planets for each orbit
            current_star.planetary_system.generate_planets_and_asteroid_belts()
            # add star to map
            (self.stars.append(current_star))
        # Add noise to star locations
        self.star_location_noise()
        self.star_location_stretch()

    def star_location_noise(self, noise=10):

        for star in self.stars:
            star.x += np.random.uniform(-noise, noise)
            star.y += np.random.uniform(-noise, noise)
            star.z += np.random.uniform(-noise, noise)

    def star_location_stretch(self, stretch_x=1, stretch_y=1, stretch_z=0.6):
        for star in self.stars:
            star.x *= stretch_x
            star.y *= stretch_y
            star.z *= stretch_z

    def generate_nations(
        self,
        n=5,
        space_boundary=500,
        name_set=None,
        nation_colour_set=None,
        origin_set=None,
        expansion_rate_set=None,
    ):

        if name_set is not None:
            for name in name_set:
                new_nation = Nation(name=name, space_boundary=space_boundary)
                self.nations.append(new_nation)
        else:
            for i in range(n):
                name = f"Nation {i + 1}"
                new_nation = Nation(name=name, space_boundary=space_boundary)
                self.nations.append(new_nation)

        if nation_colour_set is not None:
            for i in range(len(self.nations)):
                self.nations[i].nation_colour = nation_colour_set[i]

        if len(self.nations) > len(nation_colour_set):
            print(
                "Not enough colours for all nations, using random colours for the rest."
            )

        if origin_set is not None:
            for i in range(len(self.nations)):
                self.nations[i].origin = origin_set[i]

        if origin_set is not None:
            if len(self.nations) > len(origin_set):
                print(
                    "Not enough origins for all nations, using random origins for the rest."
                )

        if expansion_rate_set is not None:
            for i in range(len(self.nations)):
                self.nations[i].expansion_rate = expansion_rate_set[i]

        if expansion_rate_set is not None:
            if len(self.nations) > len(expansion_rate_set):
                print(
                    "Not enough expansion rates for all nations, using random expansion rates for the rest."
                )

    def assign_stars_to_nations(self):
        for star in self.stars:
            closest_nation = None
            min_weighted_distance = float("inf")
            for nation in self.nations:
                # Raw Euclidean distance
                raw_distance = np.sqrt(
                    (star.x - nation.origin["x"]) ** 2
                    + (star.y - nation.origin["y"]) ** 2
                    + (star.z - nation.origin["z"]) ** 2
                )
                # Apply weighting using expansion_rate as influence factor
                weighted_distance = (
                    raw_distance / nation.expansion_rate
                )  # Assuming higher expansion rates denote more influence

                if weighted_distance < min_weighted_distance:
                    closest_nation = nation
                    min_weighted_distance = weighted_distance
            closest_nation.nation_stars.append(star)

    def get_luminosities(self):
        return np.array([star.luminosity for star in self.stars])

    def get_masses(self):
        return np.array([star.mass for star in self.stars])

    def get_normalized_luminosities(self):
        # Extract luminosities into a NumPy array for efficient computation
        luminosities = np.array([star.luminosity for star in self.stars])
        # Normalize luminosities to the range [0, 1]
        normalized_luminosities = scale_values_to_range(luminosities, 0, 1)
        return np.array(normalized_luminosities)

    def get_normalized_masses(self):
        # Extract masses into a NumPy array for efficient computation
        masses = np.array([star.mass for star in self.stars])
        # Normalize masses to the range [0, 1]
        normalized_masses = scale_values_to_range(masses, 0, 1)
        return np.array(normalized_masses)

    def plot(self):
        masses = self.get_normalized_masses()
        masses = scale_values_to_range(masses, 8, 12)

        luminosities = self.get_normalized_luminosities()

        # Create a trace for the stars
        trace_stars = go.Scatter3d(
            x=[star.x for star in self.stars],
            y=[star.y for star in self.stars],
            z=[star.z for star in self.stars],
            mode="markers+text",
            marker=dict(
                size=masses,
                color=luminosities,  # set color to an array/list of desired values
                colorscale="ylorrd_r",  # choose a colorscale
                opacity=1,
            ),
            text=[
                f"{star.name[0]}{', ' + star.spectral_class if star.spectral_class is not None else ''}"
                for star in self.stars
            ],
            hoverinfo="text",
            name="Stars",
        )

        # Create trace for the nations
        trace_nations = go.Scatter3d(
            x=[star.x for star in self.stars],
            y=[star.y for star in self.stars],
            z=[star.z for star in self.stars],
            mode="markers",
            marker=dict(
                size=30,
                color=[
                    next(
                        (
                            nation.nation_colour
                            for nation in self.nations
                            if star in nation.nation_stars
                        ),
                        "white",
                    )
                    for star in self.stars
                ],  # set color to the color of the nation the star is assigned to
                opacity=0.2,
            ),
            name="Nations",
            hoverinfo="none",
        )

        # Create trace for the planets

        planet_x = []
        planet_y = []
        planet_z = []
        planet_mass = []
        planet_colors = []
        planet_names = []

        # Offset for placing planetary dots to the right of the star

        base_offset = 1.5
        offset_increment = 1

        for star in self.stars:
            offset = base_offset
            for planet in star.planetary_system.celestial_bodies:
                # Assume each star's planetary system has a method or attribute to check habitability
                is_habitable = (
                    hasattr(planet, "habitable") and planet.habitable
                )  # Placeholder condition
                planet_color = "green" if is_habitable else "black"

                # Add the planet dot position and color
                planet_x.append(star.x + offset)
                planet_y.append(star.y)
                planet_z.append(star.z)

                if planet.body_type == "Planet":
                    planet_mass.append(planet.mass)
                else:
                    planet_mass.append(1)

                planet_colors.append(planet_color)

                planet_names.append(planet.name)
                # Increment the offset for the next planet
                offset += offset_increment

        # Create trace for the planetary dots
        trace_planets = go.Scatter3d(
            x=planet_x,
            y=planet_y,
            z=planet_z,
            mode="markers",
            marker=dict(
                size=scale_values_to_range(planet_mass, 5, 10),  # Adjust size as needed
                color=planet_colors,  # Color based on habitability
            ),
            text=planet_names,
            name="trace_planets",
            hoverinfo="text",
        )
        data = [trace_stars, trace_nations, trace_planets]

        # Create layout for the plot
        layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z"),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5),
                ),
            ),
            legend=dict(x=0, y=1),
        )

        fig = go.Figure(data=data, layout=layout)
        plot(fig, filename='Abyssal_showcase.html', output_type='file')
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


name_set = [
    "Haven",
    "New Frontier Alliance",
    "Sol Protectorate",
    "United Stellar Colonies",
    "Void Confederacy",
]
colour_set = [
    (0.5, 0.5, 0.5),
    (0.2, 0.8, 0.2),
    (0.8, 0.2, 0.2),
    (0.2, 0.2, 0.8),
    (0.8, 0.8, 0.2),
]
origin_set = [
    {"x": -200, "y": 100, "z": -100},
    {"x": -50, "y": 100, "z": 90},
    {"x": 0, "y": 0, "z": 0},
    {"x": 50, "y": 50, "z": 20},
    {"x": 100, "y": 100, "z": -50},
]
expansion_rate_set = [0.7, 0.8, 1, 1, 0.9]

np.random.seed(50)

actualmap = Starmap()
actualmap.generate_stars(number_of_stars=10)
actualmap.generate_nations(
    name_set=name_set,
    nation_colour_set=colour_set,
    origin_set=origin_set,
    expansion_rate_set=expansion_rate_set,
)
actualmap.assign_stars_to_nations()
print(actualmap)
actualmap.plot()

# Print stars and planetary system names
for star in actualmap.stars:
    print(star.name[0])
    print("-----")

    for planet in star.planetary_system.celestial_bodies:
        print(planet.name)
