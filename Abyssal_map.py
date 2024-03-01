import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
from Map_Components import Nation, Star, Planetary_System
from Utility import scale_values_to_range

# Generate stars
class Starmap:
    def __init__(self):
        self.stars = []
        self.nations = []
        self.spectral_classes = {
            "O-Type": (16, 300),
            "B-Type": (2.1, 16),
            "A-Type": (1.4, 2.1),
            "F-Type": (1.04, 1.4),
            "G-Type": (0.6, 1.5),
            "K-Type": (0.08, 0.6),
            "M-Type": (0.02, 0.08),
        }
        self.used_star_names = []

    def generate_stars(self, number_of_stars=500, map_radius=500):

        for id in range(number_of_stars):
            phi, r, theta = self.random_spherical_coordinate(map_radius)
            spectral_class = self.random_spectral_class()
            luminosity = self.random_luminosity(spectral_class)

            current_star = self.instance_star(id, luminosity, phi, r, spectral_class, theta)

            # Instance the planetary system
            current_star.planetary_system = Planetary_System(current_star)

            # Generate orbits for the star
            self.generate_orbits_for_star(current_star, number_of_orbits=10, include_habitable_zone=True)

            # generate planets for each orbit
            current_star.planetary_system.generate_planets_and_asteroid_belts()

            # add star to map
            (self.stars.append(current_star))
        # Add noise to star locations
        self.star_location_noise()
        self.star_location_stretch()

    def generate_orbits_for_star(self, current_star, number_of_orbits=3, include_habitable_zone = True):
        """
        Generates orbits for a given star in the starmap.

        Args:
                current_star (Star): The star for which to generate orbits.
                number_of_orbits (int, optional): The maximum number of orbits to generate. Defaults to 3.
                include_habitable_zone (bool, optional): Whether to include a habitable zone in the generated orbits. Defaults to True.
        """
        orbits_count = np.random.randint(1, number_of_orbits+1)
        current_star.planetary_system.generate_orbits(
            include_habitable_zone, num_orbits=orbits_count
        )

    def instance_star(self, id, luminosity, phi, r, spectral_class, theta):
        # Set star properties
        current_star = Star(
            id,
            starmap=self,
            r=r,
            theta=theta,
            phi=phi,
            spectral_class=spectral_class,
            luminosity=luminosity,
        )
        return current_star

    def random_luminosity(self, spectral_class):
        # Create random luminosity
        luminosity = np.random.uniform(*self.spectral_classes[spectral_class])
        return luminosity

    def random_spectral_class(self, include_habitable_zone = True):
        # Create random spectral class
        if include_habitable_zone:
            # choose only G K or M Type stars
            useable_spectral_classes = ["G-Type", "K-Type", "M-Type"]
            spectral_class = np.random.choice(
                useable_spectral_classes, p=[0.3, 0.3, 0.4] # shifting distribution to G and K type stars, more like SOL
            )
        else:
            spectral_class = np.random.choice(
                list(self.spectral_classes.keys()), p = [0.00003,0.13,0.6,3,7.6,12.1,76.5]
            )
        return spectral_class

    def random_spherical_coordinate(self, map_radius):
        # Generate random spherical coordinates
        r = map_radius * (np.random.uniform(0, 1) ** (1 / 3))
        theta = 2 * np.random.uniform(0, 1) * np.pi
        phi = np.arccos(2 * np.random.uniform(0, 1) - 1)
        return phi, r, theta

    def star_location_noise(self, noise=10):
        """
        Adds noise to the location of each star in the starmap.

        This method iterates over each star in the starmap and adds a random offset to the x, y, and z coordinates of the star.
        The offset is a random float between -noise and noise.

        Args:
            noise (int, optional): The maximum absolute value of the offset to be added to the star's location. Defaults to 10.
        """
        for star in self.stars:
            star.x += np.random.uniform(-noise, noise)
            star.y += np.random.uniform(-noise, noise)
            star.z += np.random.uniform(-noise, noise)

    def star_location_stretch(self, stretch_x=1, stretch_y=1, stretch_z=0.6):
        """
        Stretches the location of each star in the starmap along the x, y, and z axes.

        This method iterates over each star in the starmap and multiplies the x, y, and z coordinates of the star by the corresponding stretch factors.

        Args:
            stretch_x (float, optional): The factor by which to stretch the x-coordinate of each star's location. Defaults to 1.
            stretch_y (float, optional): The factor by which to stretch the y-coordinate of each star's location. Defaults to 1.
            stretch_z (float, optional): The factor by which to stretch the z-coordinate of each star's location. Defaults to 0.6.
        """
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
        trace_stars = self.trace_stars(luminosities, masses)

        # Create trace for the nations
        trace_nations = self.trace_nations()

        # Create trace for the planets

        trace_planets = self.trace_planets()

        # Create trace for the planetary orbits
        trace_planets_orbits = self.trace_planets_orbits()

        # Create trace for the asteroid belts
        trace_asteroid_belts = self.trace_asteroid_belts()

        # Create layout for the plot
        layout = self.define_layout()

        self.create_figure(
            layout, trace_nations, trace_planets, trace_stars, trace_planets_orbits, trace_asteroid_belts, html=False
        )

    def create_figure(
        self, layout, trace_nations, trace_planets, trace_stars, trace_planets_orbits,trace_asteroid_belts, html=True
    ):
        data = [trace_stars, trace_nations, trace_planets,trace_planets_orbits,trace_asteroid_belts]
        fig = go.Figure(data=data, layout=layout)
        if html:
            plot(fig, filename="Abyssal_showcase.html", output_type="file")
        fig.show()

    def define_layout(self):
        """
        Defines the layout for the 3D plot of the starmap.

        This method creates a layout for the 3D plot with the following characteristics:
        - No margins
        - X, Y, and Z axes titled as "X", "Y", and "Z" respectively
        - A camera positioned at (1.5, 1.5, 1.5) with its up direction along the Z-axis
        - A legend positioned at the top left corner of the plot

        Returns:
            layout (plotly.graph_objs.Layout): A Layout object representing the layout of the 3D plot.
        """
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
        return layout

    def trace_planets_orbits(self):
        """
        Creates a trace for the orbits of the planets in the starmap.

        This method iterates over each star in the starmap, and for each star, it iterates over its planets.
        For each planet, it calculates the points of its orbit in the 3D space.
        It then creates a Scatter3d trace with this information.

        Returns:
            trace_planets_orbits (plotly.graph_objs._scatter3d.Scatter3d): A Scatter3d trace representing the orbits of the planets in the starmap.
        """
        orbit_x = []
        orbit_y = []
        orbit_z = []

        # Get all orbits and normalize them
        all_orbits = [planet.orbit for star in self.stars for planet in star.planetary_system.celestial_bodies]
        normalized_orbits = scale_values_to_range(all_orbits, 1, 17)



        orbit_index = 0
        for star in self.stars:
            for planet in star.planetary_system.celestial_bodies:
                # Use the normalized orbit as the offset
                offset = normalized_orbits[orbit_index]
                orbit_index += 1

                # Initialize a temporary list to store the orbit's points
                temp_orbit_x = []
                temp_orbit_y = []
                temp_orbit_z = []
                # Add the orbit circumference points
                for i in range(0, 360, 5):
                    x = star.x + offset * np.cos(np.radians(i))
                    y = star.y + offset * np.sin(np.radians(i))
                    z = star.z
                    temp_orbit_x.append(x)
                    temp_orbit_y.append(y)
                    temp_orbit_z.append(z)
                # Add the first point again to close the orbit
                temp_orbit_x.append(temp_orbit_x[0])
                temp_orbit_y.append(temp_orbit_y[0])
                temp_orbit_z.append(temp_orbit_z[0])
                # Append the orbit points to the main lists
                orbit_x.extend(temp_orbit_x + [None])  # Add None to break the line
                orbit_y.extend(temp_orbit_y + [None])
                orbit_z.extend(temp_orbit_z + [None])

        # Create trace for the orbit circumferences with closed loops
        trace_planets_orbits = go.Scatter3d(
            x=orbit_x,
            y=orbit_y,
            z=orbit_z,
            mode="lines",
            line=dict(color="black", width=1),
            opacity=0.2,
            name="trace_planets_orbits",
            hoverinfo="none",
        )
        return trace_planets_orbits

    def trace_planets(self):
        """
        Creates a trace for the planets in the starmap.

        This method iterates over each star in the starmap, and for each star, it iterates over its planets.
        For each planet, it calculates its position in the 3D space based on the planet's orbit attribute, its mass, and its color based on its habitability.
        It then creates a Scatter3d trace with this information.

        Returns:
            trace_planets (plotly.graph_objs._scatter3d.Scatter3d): A Scatter3d trace representing the planets in the starmap.
        """
        planet_x = []
        planet_y = []
        planet_z = []
        planet_mass = []
        planet_colors = []
        planet_names = []

        # Get all orbits and normalize them
        all_orbits = [planet.orbit for star in self.stars for planet in star.planetary_system.celestial_bodies]
        normalized_orbits = scale_values_to_range(all_orbits, 1, 17)

        orbit_index = 0
        for star in self.stars:
            for planet in star.planetary_system.celestial_bodies:
                # Use the normalized orbit as the offset
                offset = normalized_orbits[orbit_index]
                orbit_index += 1

                if planet.body_type == "Planet":
                    # Assume each star's planetary system has a method or attribute to check habitability
                    is_habitable = (
                            hasattr(planet, "habitable") and planet.habitable
                    )  # Placeholder condition
                    planet_color = "green" if is_habitable else "black"

                    # Add the planet dot position and color
                    angle = np.random.uniform(0, 2 * np.pi)  # Random angle for the position on the orbit
                    planet_x.append(star.x + offset * np.cos(angle))
                    planet_y.append(star.y + offset * np.sin(angle))
                    planet_z.append(star.z)

                    if planet.body_type == "Planet":
                        planet_mass.append(planet.mass)
                    else:
                        planet_mass.append(1)

                    planet_colors.append(planet_color)

                    planet_names.append(planet.name)

        # Create trace for the planetary dots
        trace_planets = go.Scatter3d(
            x=planet_x,
            y=planet_y,
            z=planet_z,
            mode="markers",
            marker=dict(
                size=scale_values_to_range(planet_mass, 7, 12),  # Adjust size as needed
                color=planet_colors,  # Color based on habitability
            ),
            text=planet_names,
            name="trace_planets",
            hoverinfo="text",
        )
        return trace_planets

    def trace_asteroid_belts(self):

        asteroid_belt_x = []
        asteroid_belt_y = []
        asteroid_belt_z = []
        asteroid_belt_names = []

        # Get all orbits and normalize them
        all_orbits = [planet.orbit for star in self.stars for planet in star.planetary_system.celestial_bodies]
        normalized_orbits = scale_values_to_range(all_orbits, 1, 17)

        orbit_index = 0
        for star in self.stars:
            for belt in star.planetary_system.celestial_bodies:
                # Use the normalized orbit as the offset
                offset = normalized_orbits[orbit_index]
                orbit_index += 1
                print("Increased orbit index by +1, Orbit index is now: ", orbit_index)
                print("Belt body type is: ", belt.body_type)
                if belt.body_type == "Asteroid Belt":
                    # Add the planet dot position and color
                    angle = np.random.uniform(0, 2 * np.pi)  # Random angle for the position on the orbit
                    asteroid_belt_x.append(star.x + offset * np.cos(angle))
                    asteroid_belt_y.append(star.y + offset * np.sin(angle))
                    asteroid_belt_z.append(star.z)

                    asteroid_belt_names.append(belt.name)

        # Create trace for the planetary dots
        trace_asteroid_belts = go.Scatter3d(
            x=asteroid_belt_x,
            y=asteroid_belt_y,
            z=asteroid_belt_z,
            mode="markers",
            marker=dict(
                size= 2,  # Adjust size as needed
                color="blue"
            ),
            text=asteroid_belt_names,
            name="trace_asteroid_belts",
            hoverinfo="text",
        )
        return trace_asteroid_belts

    def trace_nations(self):
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
        return trace_nations

    def trace_stars(self, luminosities, masses):
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
        return trace_stars

    def __str__(self):
        """
        Returns a string representation of the Starmap object.

        This method counts the number of stars of each spectral class in the starmap and prints the count.
        It also retrieves the masses and luminosities of the stars, and prints the highest and lowest values of each.
        It ends by printing a line of dashes to separate the starmap information from other output.
        """
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
        print("-----")
        return ""

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

actual_map = Starmap()
actual_map.generate_stars(number_of_stars=10)
actual_map.generate_nations(
    name_set=name_set,
    nation_colour_set=colour_set,
    origin_set=origin_set,
    expansion_rate_set=expansion_rate_set,
)
actual_map.assign_stars_to_nations()
print(actual_map)
actual_map.plot()

