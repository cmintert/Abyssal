import numpy as np
import plotly.graph_objects as go
import json

from plotly.offline import plot
from map_components import Nation, Star, Planetary_System, MineralMap
from Utility import scale_values_to_range, insert_linebreaks, RareMinerals
from random_generator import RandomGenerator

import config

random_generator = RandomGenerator.get_instance()


class StarSystemFilter:
    def __init__(self):
        self.active_filters = {}

    def add_filter(self, filter_name, filter_function):
        """Add a filter with a name and filtering function"""
        self.active_filters[filter_name] = filter_function

    def remove_filter(self, filter_name):
        """Remove a filter by name"""
        if filter_name in self.active_filters:
            del self.active_filters[filter_name]

    def clear_filters(self):
        """Remove all filters"""
        self.active_filters = {}

    def apply_filters(self, stars):
        """Apply all active filters to the star list"""
        filtered_stars = stars
        for filter_name, filter_function in self.active_filters.items():
            filtered_stars = list(filter(filter_function, filtered_stars))
        return filtered_stars


# Generate stars
class Starmap:
    """
    A class to generate and manage a 3D map of star systems with their associated nations and planetary bodies.

    This class serves as the main engine for creating a scientifically plausible star map for the Abyssal game universe.
    It handles the generation of stars, planetary systems, and the political division of space into nations.

    Attributes:
        stars (list): List of Star objects in the map
        nations (list): List of Nation objects controlling various regions
        spectral_classes (dict): Mapping of star types to their luminosity ranges
        used_star_names (list): Tracks used names to prevent duplicates
        plot_generator (PlotGenerator): Handles visualization of the starmap
        mineral_maps (dict): Maps of mineral distributions across space
    """

    def __init__(self):
        """
        Initializes a new Starmap instance with empty collections and default settings.
        """
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
        self.plot_generator = PlotGenerator(self)
        self.mineral_maps = {}

    @staticmethod
    def write_to_json(data, filename):
        """
        Serializes and writes data to a JSON file.

        Args:
            data: Data structure to serialize
            filename (str): Target JSON file path

        Returns:
            None
        """
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
        print(f"Data written to {filename}")

    def get_serialized_stars(self):
        """
        Returns a list of serialized star data.

        Returns:
            list: Serialized star data
        """
        return [star.serialize_star_to_dict() for star in self.stars]

    def get_serialized_nations(self):
        """
        Returns a list of serialized nation data.

        Returns:
            list: Serialized nation data
        """
        return [nation.serialize_nation_to_dict() for nation in self.nations]

    def get_serialized_planetary_systems(self):
        """
        Returns a list of serialized planetary system data.

        Returns:
            list: Serialized planetary system data
        """
        return [
            star.planetary_system.serialize_planetary_system_to_dict()
            for star in self.stars
        ]

    def get_serialized_planets(self):
        """
        Returns a list of serialized planet data.

        Returns:
            list: Serialized planet data
        """
        return [
            planet.serialize_planet_to_dict()
            for star in self.stars
            for planet in star.planetary_system.celestial_bodies
            if planet.body_type == "Planet"
        ]

    def get_serialized_asteroid_belts(self):
        """
        Returns a list of serialized asteroid belt data.

        Returns:
            list: Serialized asteroid belt data
        """
        return [
            belt.serialize_asteroid_belt_to_dict()
            for star in self.stars
            for belt in star.planetary_system.celestial_bodies
            if belt.body_type == "Asteroid Belt"
        ]

    def write_stars_to_json(self):
        """
        Writes serialized star data to a JSON file.

        Returns:
            None
        """
        self.write_to_json(self.get_serialized_stars(), "json_data/star_data.json")

    def write_nations_to_json(self):
        """
        Writes serialized nation data to a JSON file.

        Returns:
            None
        """
        self.write_to_json(self.get_serialized_nations(), "json_data/nation_data.json")

    def write_planetary_systems_to_json(self):
        """
        Writes serialized planetary system data to a JSON file.

        Returns:
            None
        """
        self.write_to_json(
            self.get_serialized_planetary_systems(),
            "json_data/planetary_system_data.json",
        )

    def write_planets_to_json(self):
        """
        Writes serialized planet data to a JSON file.

        Returns:
            None
        """
        self.write_to_json(self.get_serialized_planets(), "json_data/planet_data.json")

    def write_asteroid_belts_to_json(self):
        """
        Writes serialized asteroid belt data to a JSON file.

        Returns:
            None
        """
        self.write_to_json(
            self.get_serialized_asteroid_belts(), "json_data/asteroid_belt_data.json"
        )

    def write_all_to_json(self):
        """
        Writes all data to JSON files.

        Returns:
            None
        """
        self.write_stars_to_json()
        self.write_nations_to_json()
        self.write_planetary_systems_to_json()
        self.write_planets_to_json()
        self.write_asteroid_belts_to_json()

    def generate_mineral_maps(self, area=500, number=6):
        """
        Generates mineral maps for rare minerals.

        Args:
            area (int): Area of the map
            number (int): Number of mineral zones

        Returns:
            None
        """
        rare_minerals = RareMinerals()
        list_of_minerals = rare_minerals.get_minerals()
        for mineral in list_of_minerals:
            mineral_map = MineralMap(mineral)
            mineral_map.zone_points = mineral_map.generate_zone_points(
                mineral, area, number
            )
            self.mineral_maps[mineral] = mineral_map

    def generate_star_systems(self, number_of_stars=500, map_radius=500):
        """
        Generates a complete star system including stars, planets, and asteroid belts.

        Args:
            number_of_stars (int): Number of stars to generate
            map_radius (float): Radius of the spherical game space

        Returns:
            None
        """
        self.generate_mineral_maps(number=10)

        for id in range(number_of_stars):
            phi, r, theta = self.random_spherical_coordinate(map_radius)
            spectral_class = self.random_spectral_class()
            luminosity = self.random_luminosity(spectral_class)

            current_star = self.instance_star(
                id, luminosity, phi, r, spectral_class, theta
            )

            # Instance the planetary system
            current_star.planetary_system = Planetary_System(current_star)

            # Generate orbits for the star
            self.generate_orbits_for_star(
                current_star, number_of_orbits=10, include_habitable_zone=True
            )

            # generate planets for each orbit
            current_star.planetary_system.generate_planets_and_asteroid_belts()
            current_star.planetary_system.generate_description()

            # add star to map
            (self.stars.append(current_star))

        # Add noise and stretch to star locations after all stars are generated
        # This distorts the star locations to create a more realistic distribution
        self.star_location_noise(config.STAR_LOCATION_NOISE)
        self.star_location_stretch(
            config.STAR_LOCATION_STRETCH[0],
            config.STAR_LOCATION_STRETCH[1],
            config.STAR_LOCATION_STRETCH[2],
        )

    @staticmethod
    def generate_orbits_for_star(
        current_star, number_of_orbits=3, include_habitable_zone=True
    ):
        """
        Generates orbits for a given star in the starmap.

        Args:
            current_star (Star): The star for which to generate orbits.
            number_of_orbits (int, optional): The maximum number of orbits to generate. Defaults to 3.
            include_habitable_zone (bool, optional): Whether to include a habitable zone in the generated orbits. Defaults to True.

        Returns:
            None
        """

        orbits_count = random_generator.randint(1, number_of_orbits + 1)

        current_star.planetary_system.generate_orbits(
            include_habitable_zone, num_orbits=orbits_count
        )

    def instance_star(self, id, luminosity, phi, r, spectral_class, theta):
        """
        Creates a new Star instance.

        Args:
            id (int): Unique identifier for the star
            luminosity (float): Luminosity of the star
            phi (float): Spherical coordinate phi
            r (float): Spherical coordinate r
            spectral_class (str): Spectral class of the star
            theta (float): Spherical coordinate theta

        Returns:
            Star: The created Star instance
        """
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
        """
        Generates a random luminosity for a given spectral class.

        Args:
            spectral_class (str): Spectral class of the star

        Returns:
            float: Random luminosity value
        """
        luminosity = random_generator.uniform(*self.spectral_classes[spectral_class])
        return luminosity

    def random_spectral_class(self, include_habitable_zone=True):
        """
        Generates a random spectral class.

        Args:
            include_habitable_zone (bool): Whether to prioritize habitable spectral classes

        Returns:
            str: Random spectral class
        """
        if include_habitable_zone:
            # choose only G K or M Type stars
            useable_spectral_classes = ["G-Type", "K-Type", "M-Type"]

            spectral_class = random_generator.choice(
                useable_spectral_classes,
                p=[
                    0.3,
                    0.3,
                    0.4,
                ],  # shifting distribution to G and K type stars, more like SOL
            )

        else:
            spectral_class = random_generator.choice(
                list(self.spectral_classes.keys()),
                p=[0.00003, 0.13, 0.6, 3, 7.6, 12.1, 76.5],
            )

        return spectral_class

    @staticmethod
    def random_spherical_coordinate(map_radius):
        """
        Generates random spherical coordinates.

        Args:
            map_radius (float): Radius of the spherical space

        Returns:
            tuple: Spherical coordinates (phi, r, theta)
        """

        r = map_radius * (random_generator.uniform(0, 1) ** (1 / 3))
        theta = 2 * random_generator.uniform(0, 1) * np.pi
        phi = np.arccos(2 * random_generator.uniform(0, 1) - 1)

        return phi, r, theta

    def star_location_noise(self, noise=10):
        """
        Adds noise to the location of each star in the starmap.

        Args:
            noise (int, optional): The maximum absolute value of the offset to be added to the star's location. Defaults to 10.

        Returns:
            None
        """

        for star in self.stars:

            position_x = star.get_cartesian_position()[0] + random_generator.uniform(
                -noise, noise
            )
            position_y = star.get_cartesian_position()[1] + random_generator.uniform(
                -noise, noise
            )
            position_z = star.get_cartesian_position()[2] + random_generator.uniform(
                -noise, noise
            )

            star.set_cartesian_position(position_x, position_y, position_z)

    def star_location_stretch(self, stretch_x=1, stretch_y=1, stretch_z=0.6):
        """
        Stretches the location of each star in the starmap along the x, y, and z axes.

        This method iterates over each star in the starmap and multiplies the x, y, and z coordinates of the star by the corresponding stretch factors.

        Args:
            stretch_x (float, optional): The factor by which to stretch the x-coordinate of each star's location. Defaults to 1.
            stretch_y (float, optional): The factor by which to stretch the y-coordinate of each star's location. Defaults to 1.
            stretch_z (float, optional): The factor by which to stretch the z-coordinate of each star's location. Defaults to 0.6.

        Returns:
            None
        """
        for star in self.stars:
            position_x = star.get_cartesian_position()[0] * stretch_x
            position_y = star.get_cartesian_position()[1] * stretch_y
            position_z = star.get_cartesian_position()[2] * stretch_z
            star.set_cartesian_position(position_x, position_y, position_z)

    def generate_nations(
        self,
        n=5,
        space_boundary=500,
        name_set=None,
        nation_colour_set=None,
        origin_set=None,
        expansion_rate_set=None,
    ):
        """
        Creates political entities (nations) that control regions of space.

        Args:
            n (int): Number of nations to generate if name_set is None
            space_boundary (float): Maximum coordinate value for nation origins
            name_set (list, optional): Predefined nation names
            nation_colour_set (list, optional): Colors for nation visualization
            origin_set (list, optional): Starting coordinates for each nation
            expansion_rate_set (list, optional): Growth rates for each nation

        Returns:
            None
        """
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

        if origin_set is not None and len(self.nations) > len(origin_set):
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
        """
        Assigns stars to nations based on proximity and expansion rate based
        on the nation origin.

        Returns:
            None
        """
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
            star.nation = closest_nation

    def get_luminosities(self):
        """
        Retrieves luminosities of all stars.

        Returns:
            np.array: Array of luminosities
        """
        return np.array([star.luminosity for star in self.stars])

    def get_masses(self):
        """
        Retrieves masses of all stars.

        Returns:
            np.array: Array of masses
        """
        return np.array([star.mass for star in self.stars])

    def get_normalized_luminosities(self):
        """
        Retrieves normalized luminosities of all stars.

        Returns:
            np.array: Array of normalized luminosities
        """
        luminosities = np.array([star.luminosity for star in self.stars])
        # Normalize luminosities to the range [0, 1]
        normalized_luminosities = scale_values_to_range(luminosities, 0, 1)
        return np.array(normalized_luminosities)

    def get_normalized_masses(self):
        """
        Retrieves normalized masses of all stars.

        Returns:
            np.array: Array of normalized masses
        """
        masses = np.array([star.mass for star in self.stars])
        # Normalize masses to the range [0, 1]
        normalized_masses = scale_values_to_range(masses, 0, 1)
        return np.array(normalized_masses)

    def plot(self):
        """
        Plots the starmap using Plotly.

        Returns:
            None
        """
        self.plot_generator.plot()

    def __str__(self):
        """
        Returns a string representation of the Starmap object.

        Returns:
            str: String representation of the starmap
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


class PlotGenerator:
    def __init__(self, starmap):
        self.starmap = starmap

    def plot(self, html=True, return_fig=False, star_filter=None):
        """
        Generate the plot with optional filtering

        Parameters:
        - html: Whether to save as HTML
        - return_fig: Whether to return the figure object
        - star_filter: StarSystemFilter object to apply
        """
        # Apply filters if provided
        if star_filter and star_filter.active_filters:
            stars_to_use = star_filter.apply_filters(self.starmap.stars)
        else:
            stars_to_use = self.starmap.stars

        # Check if we have any stars after filtering
        if not stars_to_use:
            # Create an empty figure with a message
            layout = self.define_layout()
            fig = go.Figure(layout=layout)

            # Add an annotation explaining that no stars match the filter
            fig.add_annotation(
                text="No stars match the current filter criteria",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=20, color="white"),
            )

            if return_fig:
                return fig
            if html:
                plot(fig, filename="Abyssal_showcase.html", output_type="file")
            fig.show()
            return fig

        # If we have stars, proceed with normal plotting
        masses = scale_values_to_range(
            [star.mass for star in stars_to_use],
            config.STAR_SIZE_RANGE[0],
            config.STAR_SIZE_RANGE[1],
        )

        luminosities = scale_values_to_range(
            [star.luminosity for star in stars_to_use], 0, 1
        )

        # Create traces with the filtered stars
        trace_stars = self.trace_stars(luminosities, masses, stars_to_use)
        trace_nations = self.trace_nations(stars_to_use)
        trace_planets = self.trace_planets(stars_to_use)
        trace_planets_orbits = self.trace_planets_orbits(stars_to_use)
        trace_asteroid_belts = self.trace_asteroid_belts(stars_to_use)
        trace_planetary_system = self.trace_planetary_system(stars_to_use)

        # Create layout for the plot
        layout = self.define_layout()

        return self.create_figure(
            layout,
            trace_nations,
            trace_planets,
            trace_stars,
            trace_planets_orbits,
            trace_asteroid_belts,
            trace_planetary_system,
            html=html,
            return_fig=return_fig,
        )

    @staticmethod
    def create_figure(
        layout,
        trace_nations,
        trace_planets,
        trace_stars,
        trace_planets_orbits,
        trace_asteroid_belts,
        trace_planetary_system,
        html=True,
        return_fig=False,
    ):
        data = [
            trace_stars,
            trace_nations,
            trace_planets,
            trace_planets_orbits,
            trace_asteroid_belts,
            trace_planetary_system,
        ]
        fig = go.Figure(data=data, layout=layout)
        if return_fig:
            return fig
        if html:
            plot(fig, filename="Abyssal_showcase.html", output_type="file")
        fig.show()

    @staticmethod
    def define_layout():
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
        grid_color = "rgba(20, 89, 104, 0.2)"

        layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            paper_bgcolor="black",
            scene=dict(
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=0.6, y=0.6, z=1),
                ),
                xaxis=dict(
                    backgroundcolor=config.BACKGROUND_COLOR,
                    gridcolor=grid_color,
                    showbackground=True,
                    zerolinecolor="white",
                ),
                yaxis=dict(
                    backgroundcolor=config.BACKGROUND_COLOR,
                    gridcolor=grid_color,
                    showbackground=True,
                    zerolinecolor="white",
                ),
                zaxis=dict(
                    backgroundcolor=config.BACKGROUND_COLOR,
                    gridcolor=grid_color,
                    showbackground=True,
                    zerolinecolor="white",
                ),
            ),
            legend=dict(x=0, y=1),
            title=dict(
                text="ABYSSAL 2675 AD",  # Set your plot title here
                x=0.5,  # Centers the title
                y=0.95,  # Position the title at the top of the plot
                xanchor="center",
                yanchor="top",
                font=dict(  # Customize the global font of the plot
                    family="OCR A, sans-serif",  # Specify the font family here
                    size=50,  # Specify the font size here
                    color="white",  # Specify the font color here
                ),
            ),
        )
        return layout

    def trace_planets_orbits(self, stars_to_use=None):
        stars_to_plot = stars_to_use if stars_to_use is not None else self.starmap.stars

        orbit_x = []
        orbit_y = []
        orbit_z = []

        # Get all orbits and normalize them
        all_planets = [
            planet
            for star in stars_to_plot
            for planet in star.planetary_system.celestial_bodies
        ]

        if not all_planets:
            # Return empty trace if no planets
            return go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                line=dict(color=config.ORBIT_COLOR, width=config.ORBIT_LINE_WIDTH),
                opacity=config.ORBIT_OPACITY,
                name="trace_planets_orbits",
                hoverinfo="text",
            )

        all_orbits = [planet.orbit for planet in all_planets]
        normalized_orbits = scale_values_to_range(all_orbits, 1, 17)

        orbit_index = 0
        for star in stars_to_plot:
            for _ in star.planetary_system.celestial_bodies:
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
            mode="lines, text",
            line=dict(color=config.ORBIT_COLOR, width=config.ORBIT_LINE_WIDTH),
            opacity=config.ORBIT_OPACITY,
            name="trace_planets_orbits",
            hoverinfo="text",
        )
        return trace_planets_orbits

    def trace_planets(self, stars_to_use=None):
        stars_to_plot = stars_to_use if stars_to_use is not None else self.starmap.stars

        planet_x = []
        planet_y = []
        planet_z = []
        planet_mass = []
        planet_colors = []
        planet_names = []
        planet_hover_texts = []

        # Get all orbits and normalize them
        all_planets = [
            planet
            for star in stars_to_plot
            for planet in star.planetary_system.celestial_bodies
        ]

        if not all_planets:
            # Return empty trace if no planets
            return go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                marker=dict(size=[], color=[]),
                text=[],
                name="trace_planets",
                hoverinfo="text",
            )

        all_orbits = [planet.orbit for planet in all_planets]
        normalized_orbits = scale_values_to_range(all_orbits, 1, 17)

        orbit_index = 0
        for star in stars_to_plot:
            for planet in star.planetary_system.celestial_bodies:
                # Use the normalized orbit as the offset
                offset = normalized_orbits[orbit_index]
                orbit_index += 1

                # Create hover text using autogen_description with additional_info if available
                hover_text = planet.autogen_description
                if planet.additional_info:
                    hover_text += (
                        "<br><br>Additional Notes:<br>" + planet.additional_info
                    )

                if hover_text is not None:
                    hover_text = insert_linebreaks(hover_text, max_line_length=50)

                planet_hover_texts.append(hover_text)

                planet_color = "lightgrey"  # Default color for all celestial bodies
                if planet.body_type == "Planet":
                    planet_color = "green" if planet.habitable else "lightgrey"

                # Add the planet dot position and color

                angle = random_generator.uniform(0, 2 * np.pi)

                # Random angle for the position on the orbit
                planet_x.append(star.x + offset * np.cos(angle))
                planet_y.append(star.y + offset * np.sin(angle))
                planet_z.append(star.z)

                if planet.body_type == "Planet":
                    planet_mass.append(planet.mass)
                else:
                    planet_mass.append(0)

                planet_colors.append(planet_color)
                planet_names.append(planet.name)

        # Create trace for the planetary dots
        trace_planets = go.Scatter3d(
            x=planet_x,
            y=planet_y,
            z=planet_z,
            mode="markers",
            marker=dict(
                size=scale_values_to_range(
                    planet_mass,
                    config.PLANET_SIZE_RANGE[0],
                    config.PLANET_SIZE_RANGE[1],
                ),
                color=planet_colors,
            ),
            text=[
                f"{name}: {info}"
                for name, info in zip(planet_names, planet_hover_texts)
            ],
            name="trace_planets",
            hoverinfo="text",
        )
        return trace_planets

    def trace_asteroid_belts(self, stars_to_use=None):
        stars_to_plot = stars_to_use if stars_to_use is not None else self.starmap.stars

        asteroid_belt_x = []
        asteroid_belt_y = []
        asteroid_belt_z = []
        hover_texts = []

        # Get all orbits and normalize them
        all_planets = [
            planet
            for star in stars_to_plot
            for planet in star.planetary_system.celestial_bodies
        ]

        if not all_planets:
            # Return empty trace if no planets
            return go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                marker=dict(size=1, color="grey"),
                text=[],
                name="Asteroid Belts",
                hoverinfo="text",
            )

        all_orbits = [planet.orbit for planet in all_planets]
        normalized_orbits = scale_values_to_range(all_orbits, 1, 17)

        orbit_index = 0
        for star in stars_to_plot:
            for belt in star.planetary_system.celestial_bodies:
                # Use the normalized orbit as the offset
                scatter_number = 0
                offset = normalized_orbits[orbit_index]
                orbit_index += 1

                if belt.body_type == "Asteroid Belt":
                    # Create a number of points to scatter along the orbit
                    if belt.density == "Sparse":
                        scatter_number = 10
                    elif belt.density == "Moderate":
                        scatter_number = 40
                    elif belt.density == "Dense":
                        scatter_number = 80

                    for _ in range(scatter_number + 1):
                        angle = random_generator.uniform(
                            0, 2 * np.pi
                        )  # Random angle for the position on the orbit

                        asteroid_belt_x.append(star.x + offset * np.cos(angle))
                        asteroid_belt_y.append(star.y + offset * np.sin(angle))
                        asteroid_belt_z.append(star.z)

                        # Use autogen_description with additional_info if available
                        hover_text = belt.autogen_description
                        if belt.additional_info:
                            hover_text += (
                                "<br><br>Additional Notes:<br>" + belt.additional_info
                            )

                        if hover_text is not None:
                            hover_text = insert_linebreaks(
                                hover_text, max_line_length=50
                            )

                        hover_texts.append(hover_text)

        # Create trace for the planetary dots
        trace_asteroid_belts = go.Scatter3d(
            x=asteroid_belt_x,
            y=asteroid_belt_y,
            z=asteroid_belt_z,
            mode="markers",
            marker=dict(size=1, color="grey"),  # Adjust size as needed
            text=hover_texts,
            name="Asteroid Belts",
            hoverinfo="text",
        )
        return trace_asteroid_belts

    def trace_nations(self, stars_to_use=None):
        stars_to_plot = stars_to_use if stars_to_use is not None else self.starmap.stars

        # Initially, create a dictionary mapping each star to its nation name
        star_to_nation = {}
        for nation in self.starmap.nations:
            for star in nation.nation_stars:
                star_to_nation[star] = nation.name  # Map star to nation name

        # Generate hover text for each star, defaulting to 'Unknown' if the star isn't in the dictionary
        hovertext = [star_to_nation.get(star, "Unknown") for star in stars_to_plot]

        # Create the Scatter3d trace
        trace_nations = go.Scatter3d(
            x=[star.x for star in stars_to_plot],
            y=[star.y for star in stars_to_plot],
            z=[star.z for star in stars_to_plot],
            mode="markers",
            marker=dict(
                size=30,
                color=[
                    next(
                        (
                            nation.nation_colour
                            for nation in self.starmap.nations
                            if star in nation.nation_stars
                        ),
                        "white",
                    )
                    for star in stars_to_plot
                ],
                opacity=0.2,
            ),
            hovertext=hovertext,
            name="Nations",
            hoverinfo="text",
        )
        return trace_nations

    def trace_stars(self, luminosities, masses, stars_to_use=None):
        stars_to_plot = stars_to_use if stars_to_use is not None else self.starmap.stars

        trace_stars = go.Scatter3d(
            x=[star.x for star in stars_to_plot],
            y=[star.y for star in stars_to_plot],
            z=[star.z for star in stars_to_plot],
            mode="markers+text",
            marker=dict(
                size=masses,
                color=luminosities,
                colorscale="ylorrd_r",
                opacity=1,
            ),
            text=[
                f"{star.name[0]}{', ' + star.spectral_class if star.spectral_class is not None else ''}"
                for star in stars_to_plot
            ],
            hoverinfo="text",
            name="Stars",
            textfont=dict(color="rgba(180, 205, 203, 1)"),
        )
        return trace_stars

    def trace_planetary_system(self, stars_to_use=None):
        stars_to_plot = stars_to_use if stars_to_use is not None else self.starmap.stars

        descriptions = []
        for star in stars_to_plot:
            # Use autogen_description with additional_info if available
            description = star.planetary_system.autogen_description
            if star.planetary_system.additional_info:
                description += (
                    "<br><br>Additional Notes:<br>"
                    + star.planetary_system.additional_info
                )

            description = insert_linebreaks(description, max_line_length=50)
            descriptions.append(description)

        trace_planetary_system = go.Scatter3d(
            x=[star.x for star in stars_to_plot],
            y=[star.y for star in stars_to_plot],
            z=[star.z for star in stars_to_plot],
            mode="markers",
            opacity=0,
            text=descriptions,
            hoverinfo="text",
            name="Planetary System",
            hoverlabel=dict(
                bgcolor="black",
                bordercolor="cyan",
                font=dict(color="cyan", size=12, family="Futura, sans-serif"),
                align="left",
            ),
        )
        return trace_planetary_system
