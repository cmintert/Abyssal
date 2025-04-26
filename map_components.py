import math
import numpy as np
from numpy import random

from Utility import StarNames, PlanetNames, RareMinerals

SPECTRAL_CLASSES_LUM_MASS_RATIO = {
    "G-Type": 1 / 4,
    "K-Type": 1 / 3.5,
    "M-Type": 1 / 3,
}


class Nation:
    def __init__(
        self,
        name,
        origin=None,
        nation_colour=None,
        space_boundary=500,
        expansion_rate=0.5,
    ):
        self.name = name
        self.current_radius = 0
        self.nation_stars = []
        self.additional_info = None
        self.expansion_rate = expansion_rate

        if nation_colour is None:
            # if no colour is provided, generate a random colour
            self.nation_colour = (
                np.random.random(),
                np.random.random(),
                np.random.random(),
            )
        else:
            self.nation_colour = nation_colour

        if origin is None:
            # If no origin is provided, generate a random point within the boundary
            self.origin = {
                "x": np.random.uniform(-space_boundary, space_boundary),
                "y": np.random.uniform(-space_boundary, space_boundary),
                "z": np.random.uniform(-space_boundary, space_boundary),
            }
        else:
            self.origin = origin

    def expand_influence(self):
        """Expand the nation's sphere of influence based on its expansion rate."""
        self.current_radius += self.expansion_rate

    def add_additional_info(self, info):
        """Add additional information to the nation."""
        self.additional_info = info

    def serialize_nation_to_dict(self):

        # stars in nation
        included_stars_id = []
        for star in self.nation_stars:
            included_stars_id.append(star.id)
        return {
            "name": self.name,
            "origin": self.origin,
            "current_radius": self.current_radius,
            "expansion_rate": self.expansion_rate,
            "nation_colour": self.nation_colour,
            "nation_stars": included_stars_id,
            "additional_info": self.additional_info,
        }

    def __str__(self):
        return f"{self.name}: Origin at {self.origin}, Current Radius: {self.current_radius}, Expansion Rate: {self.expansion_rate}"


class SmallBody:
    """
    A class used to represent a SmallBody in a star system.

    Attributes
    ----------
    name : str
        the name of the small body
    body_type : str
        the type of the small body (e.g., "Planet", "Moon", "Asteroid", etc.)
    orbit : float
        the distance of the small body from the star in AU (astronomical units)
    additional_info : str
        additional information about the small body
    star : Star
        the star that the small body orbits

    Methods
    -------
    __str__():
        Returns a string representation of the small body.
    return_orbit_number():
        Returns the orbit number of the small body, starting with 1 for the closest orbit in the system.
    """

    def __init__(self, name, star, body_type=None, orbit=None, additional_info=None):
        self.name = name
        self.body_type = body_type  # Planet, Moon, Asteroid, etc.
        self.orbit = orbit  # Distance from the star in AU
        self.additional_info = additional_info
        self.star = star
        self.hill_sphere_radius = 0

    def __str__(self):
        info = f", {self.additional_info}" if self.additional_info else ""
        return f"{self.name}: {self.body_type} at {self.orbit:.2f} AU{info}"

    def serialize_small_body_to_dict(self):
        return {
            "name": self.name,
            "body_type": self.body_type,
            "orbit": self.orbit,
            "star": self.star.serialize_star_to_dict(),
            "additional_info": self.additional_info,
        }

    def add_additional_info(self, info):
        """Add additional information to the small body."""
        self.additional_info = info

    def return_orbit_number(self):
        # Return the orbit number of the body start with 1 for the closest orbit in the system
        # Access the star object and get the planetary system object to get the orbit list
        # Then return the index of the orbit of the body + 1
        orbit_number = self.star.planetary_system.orbits.index(self.orbit) + 1
        return orbit_number


class Star:
    """
    A class used to represent a Star.

    Attributes
    ----------
    id : int
        a unique identifier for the star
    star_map : Starmap
        the starmap object that the star belongs to
    name : str
        the name of the star
    x : float
        the x-coordinate of the star in Cartesian coordinates
    y : float
        the y-coordinate of the star in Cartesian coordinates
    z : float
        the z-coordinate of the star in Cartesian coordinates
    r : float
        the radial distance of the star in spherical coordinates
    theta : float
        the polar angle of the star in spherical coordinates
    phi : float
        the azimuthal angle of the star in spherical coordinates
    spectral_class : str
        the spectral class of the star (e.g., "G-Type", "K-Type", "M-Type")
    luminosity : float
        the luminosity of the star
    mass : float
        the mass of the star, calculated based on its luminosity and spectral class
    planetary_system : Planetary_System
        the planetary system that the star hosts

    Methods
    -------
    generate_star_name():
        Generates a random unique name for the star.
    convert_to_cartesian():
        Converts the star's position from spherical to Cartesian coordinates.
    convert_to_spherical():
        Converts the star's position from Cartesian to spherical coordinates.
    set_spectral_class(spectral_class):
        Sets the spectral class of the star.
    set_luminosity(luminosity):
        Sets the luminosity of the star and recalculates its mass.
    calculate_mass():
        Calculates the mass of the star based on its luminosity and spectral class.
    goldilocks_zone():
        Calculates the habitable zone of the star.
    """

    def __init__(
        self,
        id,
        starmap,
        name=None,
        nation=None,
        x=None,
        y=None,
        z=None,
        r=None,
        theta=None,
        phi=None,
        spectral_class=None,
        luminosity=None,
        planetary_system=None,
    ):
        self.id = id
        self.star_map = starmap
        if name is None:
            self.name = self.generate_star_name()
        else:
            self.name = name
        self.nation = nation
        self.additional_info = None
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
        # Planets
        self.planetary_system = planetary_system
        # If spherical coordinates are provided, convert them to Cartesian
        if r is not None and theta is not None and phi is not None:
            self.convert_to_cartesian()
        # Alternatively, if Cartesian coordinates are provided, convert them to spherical
        elif x is not None and y is not None and z is not None:
            self.convert_to_spherical()

    def add_additional_info(self, info):
        """Add additional information to the star."""
        self.additional_info = info

    def serialize_star_to_dict(self):
        data = {
            "id": self.id,
            "name": self.name,
            "nation": self.nation.name if self.nation else None,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "r": self.r,
            "theta": self.theta,
            "phi": self.phi,
            "spectral_class": self.spectral_class,
            "luminosity": self.luminosity,
            "mass": self.mass,
            "additional_info": self.additional_info,
        }

        return data

    def generate_star_name(self):
        """Generates a random name for the star. Make name unique for each star."""
        while True:
            name = StarNames().generate_combined_star_name()
            if name not in self.star_map.used_star_names:
                self.star_map.used_star_names.append(name)
                return name

    def convert_to_cartesian(self):
        """Converts spherical coordinates to Cartesian coordinates and updates the star's position."""
        self.x = self.r * math.sin(self.phi) * math.cos(self.theta)
        self.y = self.r * math.sin(self.phi) * math.sin(self.theta)
        self.z = self.r * math.cos(self.phi)

    def convert_to_spherical(self):
        """Converts Cartesian coordinates to spherical coordinates and updates the star's position."""
        self.r = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.theta = math.atan2(self.y, self.x) if self.r != 0 else 0
        self.phi = math.acos(self.z / self.r) if self.r != 0 else 0

    def set_spectral_class(self, spectral_class):
        self.spectral_class = spectral_class

    def set_luminosity(self, luminosity):
        self.luminosity = luminosity
        self.mass = self.calculate_mass()
        print(
            f"Mass of star {self.id} is changed to {self.mass} fitting the new luminosity."
        )

    def calculate_mass(self):
        """Calculates the mass of the star based on its luminosity and spectral class."""
        if (
            self.luminosity is not None
            and self.spectral_class in SPECTRAL_CLASSES_LUM_MASS_RATIO
        ):
            exponent = SPECTRAL_CLASSES_LUM_MASS_RATIO[self.spectral_class]
            mass_relative_to_sun = self.luminosity**exponent
            return mass_relative_to_sun
        raise ValueError("Luminosity or spectral class not set.")

    def goldilocks_zone(self):
        """Calculates the goldilocks zone of the star."""
        # The goldilocks zone is the distance from the star where the temperature is just right for liquid water to exist
        # The formula for the goldilocks zone is sqrt(luminosity) * 0.95
        if self.luminosity is not None:
            return [
                math.sqrt(self.luminosity) * 0.95,
                math.sqrt(self.luminosity) * 1.37,
            ]
        raise ValueError("Luminosity not set.")

    def get_cartesian_position(self):
        return self.x, self.y, self.z

    def set_cartesian_position(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.convert_to_spherical()

    def __str__(self):
        return f"Star ID: {self.id}, Cartesian: ({self.x}, {self.y}, {self.z}), Spherical: (r={self.r}, theta={self.theta}, phi={self.phi})"


class Planet(SmallBody):
    def __init__(
        self,
        star,
        name=None,
        mass=None,
        composition=None,
        density=None,
        orbital_time=None,
        rotation_period=None,
        tilt=None,
        moons="Unkonwn",
        atmosphere=None,
        surface_temperature=None,
        presence_of_water=None,
        radius=None,
        orbit=None,
        gravity=None,
        habitable=False,
    ):
        super().__init__(name, star, "Planet", orbit)

        self.star = star
        self.mass = mass
        self.density = density
        self.radius = radius
        self.composition = composition
        self.orbital_time = orbital_time
        self.rotation_period = rotation_period
        self.tilt = tilt
        self.moons = moons
        self.atmosphere = atmosphere
        self.surface_temperature = surface_temperature
        self.presence_of_water = presence_of_water
        self.radius = radius
        self.habitable = habitable
        self.gravity = gravity

    def __str__(self):
        return f"{self.name}: {self.composition} planet at {self.orbit:.2f} AU, Mass: {self.mass:.2f} Earth masses, Radius: {self.radius:.2f} km, Density: {self.density:.2f} g/cm^3, Surface Temperature: {self.surface_temperature:.2f}°C, Presence of Water: {self.presence_of_water}, Atmosphere: {self.atmosphere}, Axial Tilt: {self.tilt}°, Rotation Period: {self.rotation_period} hours, Habitable: {self.habitable}"

    def create_description(self):
        """Generate a detailed, human-readable description of the planet."""
        description = f"Planet {self.name} is a {self.composition} planet orbiting the star {self.star.name[0]} at a distance of {self.orbit:.2f} AU. "
        description += f"It has a mass of {self.mass:.2f} Earth masses and a radius of {self.radius:.2f} km. "
        description += f"The planet's density is {self.density:.2f} g/cm^3 and it has a surface temperature of {self.surface_temperature:.2f}°C. "
        description += f"The planet's atmosphere is described as: {self.atmosphere}. "
        description += f"The axial tilt of the planet is {self.tilt}° and it has a rotation period of {self.rotation_period} hours. "
        description += f"The planet's gravity is {self.gravity:.2f} m/s^2. "
        description += f"{'The planet is habitable.' if self.habitable else 'The planet is inhabitable'}. "
        description += f"Presence of water: {self.presence_of_water}. "
        return description

    def serialize_planet_to_dict(self):
        data = super().serialize_small_body_to_dict()
        data.update(
            {
                "mass": self.mass,
                "density": self.density,
                "radius": self.radius,
                "composition": self.composition,
                "orbital_time": self.orbital_time,
                "rotation_period": self.rotation_period,
                "tilt": self.tilt,
                "moons": self.moons,
                "atmosphere": self.atmosphere,
                "surface_temperature": self.surface_temperature,
                "presence_of_water": self.presence_of_water,
                "gravity": self.gravity,
                "habitable": self.habitable,
            }
        )
        return data

    def generate_planet(self, orbit, star, habitable=False):
        """
        Generate a planet based on its orbit and the properties of its star.
        - orbit: The semi-major axis of the planet's orbit, in AU.
        - star: The star around which the planet orbits.
        - habitable: Whether the planet is within the habitable zone.
        """
        self.orbit = orbit

        if habitable:
            self.habitable = True
        else:
            self.habitable = False

        # Define the planet's name based on its orbit
        name = PlanetNames(
            star, self.return_orbit_number()
        ).generate_combined_planet_name()
        self.name = name

        # Calculate the mass of the planet
        if habitable:
            self.mass = np.random.uniform(0.5, 2)
        else:
            self.mass = self.generate_planet_mass(self.orbit, star.goldilocks_zone())

        # Calculate the planet's composition
        if habitable:
            self.composition = "Rocky"
        else:
            self.composition = self.generate_composition(
                self.mass, self.orbit, star.luminosity
            )

        # Calculate the planet's density

        if habitable:
            self.density = np.random.uniform(3, 6)
        else:
            self.density = self.generate_density(self.composition)

        # Calculate the planet's radius

        self.radius = self.generate_radius(self.mass, self.density)

        # Calculate the planet's atmosphere
        if habitable:
            self.atmosphere = (
                "Thick atmosphere with nitrogen, oxygen, and possibly carbon dioxide"
            )
        else:
            self.atmosphere = self.generate_atmosphere(
                self.mass, self.orbit, star.luminosity, self.composition
            )

        # Calculate the planet's surface temperature
        if habitable:
            self.surface_temperature = np.random.uniform(-5, 30)  # In degrees Celsius
        else:
            self.surface_temperature = self.estimate_surface_temperature(
                self.orbit, star.luminosity, self.atmosphere
            )

        # Calculate the presence of water
        if habitable:
            self.presence_of_water = "Liquid water"
        else:
            self.presence_of_water = self.estimate_water_presence(
                self.orbit,
                self.mass,
                self.star,
                self.composition,
                self.surface_temperature,
            )

        # Calculate the planet's axial tilt
        if habitable:
            self.tilt = np.random.uniform(20, 30)  # In degrees
        else:
            self.tilt = self.generate_axial_tilt()

        # Calculate the planet's rotation period
        if habitable:
            self.rotation_period = np.random.uniform(20, 30)  # Hours
        else:
            self.rotation_period = self.generate_rotation_period(self)

        # Calculate the planet's magnetic field
        if habitable:
            self.has_magnetic_field = True
        else:
            self.has_magnetic_field = self.generate_magnetic_field(self)

        # Calculate the planet's gravity
        self.gravity = self.generate_gravity()

        # calculate the orbital time
        self.orbital_time = self.generate_orbital_time()

        # Generate a description of the planet
        self.additional_info = self.create_description()

        self.check_attributes()

        return self

    def generate_orbital_time(self):
        """
        Generate the orbital time of the planet based on its orbit and the star's mass.
        """
        # Calculate the orbital time based on the star's mass and the planet's orbit
        # Using the formula: orbital time = 2 * pi * sqrt((orbit distance^3) / (G * star mass))
        # Where G is the gravitational constant
        G = 6.674 * (10**-11)
        orbital_time = (
            2
            * math.pi
            * math.sqrt((self.orbit**3) / (G * (self.star.mass * 1.989 * (10**30))))
        )
        return orbital_time

    def check_attributes(self):
        """Check if all attributes are set and raise an error if not."""
        for attr, value in self.__dict__.items():
            if value is None:
                raise ValueError(f"The attribute '{attr}' is not set.")

    @staticmethod
    def generate_planet_mass(orbit_distance, goldilocks_zone):
        """
        Generate a planet's mass based on the star's mass and its orbit.

        Parameters:
        - orbit_distance: Distance of the planet's orbit from the star in AU (astronomical units).
        - goldilocks_zone: Tuple of (inner edge, outer edge) of the star's habitable zone in AU.

        Returns:
        - A float representing the planet's mass in Earth masses.
        """

        # Define probabilities based on orbit distance
        if orbit_distance < goldilocks_zone[0]:
            # Closer to the star, higher chance for terrestrial planets
            planet_type = np.random.choice(
                ["terrestrial", "gas giant", "ice giant"], p=[0.7, 0.2, 0.1]
            )
        elif orbit_distance > goldilocks_zone[1]:
            # Farther from the star, higher chance for gas/ice giants
            planet_type = np.random.choice(
                ["terrestrial", "gas giant", "ice giant"], p=[0.1, 0.45, 0.45]
            )
        else:
            # Within the habitable zone, favor terrestrial
            planet_type = np.random.choice(
                ["terrestrial", "gas giant", "ice giant"], p=[0.8, 0.1, 0.1]
            )

        # Generate mass based on planet type
        if planet_type == "terrestrial":
            mass = np.random.uniform(0.05, 5)  # Earth masses
        elif planet_type == "gas giant":
            mass = np.random.uniform(10, 300)
        else:  # ice giant
            mass = np.random.uniform(5, 50)

        return mass

    @staticmethod
    def generate_composition(mass, orbit_distance, star_luminosity):
        """
        Determines the composition of a planet based on its mass, orbit distance,
        and the characteristics of its star (mass and luminosity).

        Parameters:
        - mass: Planet's mass in Earth masses.
        - orbit_distance: Distance of the planet's orbit from the star in AU.
        - star: Luminosity of the star in solar luminosities.

        Returns:
        - A string describing the planet's composition.
        """
        # Approximate frost line calculation (simplified)
        frost_line = 4.85 * (star_luminosity**0.5)  # In AU, very rough estimate

        # Determine composition based on mass and orbit distance
        if mass < 0.5:
            composition = "Rocky"
        elif mass <= 5:
            if orbit_distance < frost_line:
                composition = "Rocky"
            else:
                composition = "Ice"
        elif mass <= 10:
            composition = "Ice Giant"
        else:
            composition = "Gas Giant"

        return composition

    def generate_density(self, composition):
        """
        Generate the density of the planet based on its composition and mass.
        """
        # Define density ranges based on composition
        if composition == "Rocky":
            density = np.random.uniform(3, 6)  # g/cm^3
        elif composition == "Ice":
            density = np.random.uniform(1, 3)
        elif composition == "Ice Giant":
            density = np.random.uniform(1, 2)
        elif composition == "Gas Giant":
            density = np.random.uniform(0.5, 1.5)
        else:
            density = np.random.uniform(0.5, 5)

        return density

    def generate_radius(self, mass, density):
        """
        Generate the radius of the planet based on its mass and density. Mass is given in Earth masses. Density is given in g/cm^3.
        """
        # Calculate the radius based on mass and density
        # Using the formula: volume = mass / density
        # Then calculate the radius from the volume of a sphere
        EARTH_MASS = 5.972 * (10**24)  # kg
        volume = (
            mass * EARTH_MASS / (density * 1000)
        )  # Mass is provided in Earth masses, density in g/cm^3
        radius = (3 * volume / (4 * math.pi)) ** (1 / 3) / 1000  # Convert to km
        return radius

    @staticmethod
    def generate_atmosphere(mass, orbit_distance, star_luminosity, composition):
        """
        Generates a simplified description of a planet's atmosphere based on its mass,
        orbit distance, star luminosity, and composition.

        Parameters:
        - mass: Planet's mass in Earth masses.
        - orbit_distance: Planet's orbit distance from the star in AU.
        - star: Star's luminosity in solar luminosities.
        - composition: Planet's composition (e.g., "Rocky", "Ice", "Ice Giant", "Gas Giant").

        Returns:
        - A string describing the planet's atmosphere.
        """
        # Thresholds for no atmosphere or thin atmosphere based on mass and orbit
        if mass < 0.1 or (orbit_distance < 0.5 and star_luminosity > 1):
            return "No atmosphere or very thin atmosphere due to low gravity or solar radiation"

        # Composition-specific considerations
        if composition == "Rocky":
            if mass > 1 and orbit_distance > 0.7 and orbit_distance < 1.5:
                return "Thick atmosphere with nitrogen, oxygen, and possibly carbon dioxide"
            else:
                return "Thin atmosphere, primarily carbon dioxide"
        elif composition == "Ice":
            return "Thin atmosphere with methane and ammonia"
        elif composition == "Ice Giant":
            return "Thick atmosphere of hydrogen, helium, and methane"
        elif composition == "Gas Giant":
            return "Very thick atmosphere of hydrogen and helium"
        else:
            return "Unknown atmospheric composition"

    def generate_breathable_amtmosphere_composition(self):
        """
        Generate a breathable atmosphere composition for the planet.

        """
        gases = {
            "Oxygen (O₂)": random.uniform(19.5, 23.5),
            "Nitrogen (N₂)": random.uniform(75, 78.5),
            "Carbon Dioxide (CO₂)": random.uniform(0.04, 0.1),
            "Argon (Ar)": random.uniform(0.01, 0.1),
            "Trace Gases": random.uniform(0.0001, 0.01),
        }
        return gases

    @staticmethod
    def estimate_surface_temperature(
        orbit_distance, star_luminosity, atmosphere_description
    ):
        """
        Estimates a planet's surface temperature based on its orbit distance,
        the luminosity of its star, and a simplified description of its atmosphere.

        Parameters:
        - orbit_distance: Planet's orbit distance from the star in AU.
        - star: Star's luminosity in solar luminosities.
        - atmosphere_description: A string describing the planet's atmosphere.

        Returns:
        - Estimated surface temperature in degrees Celsius.
        """
        # Baseline temperature estimation without atmosphere
        # Using a simplified version of the inverse square law for solar radiation
        # and assuming Earth's albedo and greenhouse effect as a base
        baseline_temp = (
            278 * (star_luminosity**0.25) / (orbit_distance**0.5) - 273.15
        )  # Convert to Celsius

        # Adjust temperature based on atmospheric description
        if "thick atmosphere" in atmosphere_description:
            temperature_adjustment = 100  # Simplified adjustment for greenhouse effect
        elif "thin atmosphere" in atmosphere_description:
            temperature_adjustment = 20
        else:  # No atmosphere or unknown
            temperature_adjustment = 0

        estimated_temp = baseline_temp + temperature_adjustment

        return estimated_temp

    @staticmethod
    def estimate_water_presence(
        orbit_distance, mass, star, composition, surface_temperature
    ):
        """
        Estimates the likelihood of the presence of water on a planet based on its orbit distance, mass,
        the luminosity of its star, its composition, and surface temperature.

        Parameters:
        - orbit_distance: Planet's orbit distance from the star in AU.
        - mass: Planet's mass in Earth masses.
        - star: Star's luminosity in solar luminosities.
        - composition: Planet's composition (e.g., "Rocky", "Gas Giant").
        - surface_temperature: Estimated surface temperature in degrees Celsius.

        Returns:
        - A string describing the likelihood of water presence.
        """
        # Define the habitable zone based on star luminosity (simplified model)
        gz_zone = star.goldilocks_zone()

        # Check for presence in the habitable zone
        if gz_zone[0] <= orbit_distance <= gz_zone[1]:
            habitable_zone = True
        else:
            habitable_zone = False

        # Water presence logic
        if (
            composition == "Rocky"
            and mass > 0.5
            and habitable_zone
            and 0 <= surface_temperature <= 100
        ):
            water_presence = "Liquid water"
        elif composition == "Rocky" and habitable_zone:
            water_presence = "Moderate likelihood of liquid water"
        else:
            water_presence = "Low likelihood of liquid water"

        return water_presence

    @staticmethod
    def generate_axial_tilt():
        """
        Generate the axial tilt of the planet based on its mass and distance from the star.
        """
        # Generate a tilt with a bias towards moderate values
        tilt = np.random.normal(
            23, 10
        )  # Using Earth's average tilt as a base, with a standard deviation to allow variability
        tilt = np.clip(
            tilt, 0, 90
        )  # Clamp the values to ensure they're within realistic bounds
        return tilt

    @staticmethod
    def generate_rotation_period(self):
        """
        Generates a rotation period for the planet, with different ranges based on planet composition.
        This method simplifies the diversity of factors that influence rotation periods in reality.
        """
        # Base ranges on composition, assuming terrestrial planets rotate faster due to historical collisions and gas giants rotate slower due to their size
        if self.composition == "Gas Giant":
            rotation_period = np.random.uniform(
                10, 24
            )  # Hours, gas giants tend to have faster rotation periods
        elif self.composition == "Ice Giant":
            rotation_period = np.random.uniform(
                16, 22
            )  # Hours, similar to Uranus and Neptune
        else:  # Terrestrial
            rotation_period = np.random.uniform(
                20, 40
            )  # Hours, Earth rotates once every ~24 hours

        return rotation_period

    @staticmethod
    def generate_magnetic_field(self):
        """
        Generates a boolean value indicating the presence of a significant magnetic field.
        This method simplifies the complex processes behind magnetic field generation,
        using the planet's mass and rotation period as proxies.
        """
        # Simplified logic: larger, faster-rotating planets are more likely to have a magnetic field
        has_magnetic_field = False
        if self.mass and self.rotation_period:
            # Assuming mass is in Earth masses and rotation period in Earth days
            mass_threshold = 0.5  # Minimum mass to potentially have a magnetic field, in Earth masses
            rotation_threshold = 1.5  # Maximum rotation period to likely maintain a magnetic field, in Earth days

            if (
                self.mass >= mass_threshold
                and self.rotation_period <= rotation_threshold
            ):
                has_magnetic_field = np.random.choice(
                    [True, False], p=[0.8, 0.2]
                )  # 80% chance if conditions are met
            else:
                has_magnetic_field = np.random.choice(
                    [True, False], p=[0.3, 0.7]
                )  # 30% chance otherwise

        return has_magnetic_field

    def generate_gravity(self):
        """
        Generate the gravity of the planet based on its mass and radius.
        """
        # Calculate the gravity based on mass and radius
        # Using the formula: gravity = (G * mass) / radius^2
        # Where G is the gravitational constant
        G = 6.674 * (10**-11)
        gravity = (G * self.mass * (5.972 * (10**24))) / ((self.radius * 1000) ** 2)

        return gravity

    def adjust_orbit(self, orbit_adjustment):
        """Adjust the orbit of the planet"""
        new_orbit = self.orbit + orbit_adjustment
        return new_orbit


class AsteroidBelt(SmallBody):
    def __init__(self, star, name=None, **kwargs):
        super().__init__(name, star, **kwargs)
        self.density = None
        self.minerals = None

    def __str__(self):
        return f"{self.star.name[0]}:  {self.name}: {self.body_type} at {self.orbit:.2f} AU, Density: {self.density}, Minerals: {self.minerals}"

    def serialize_asteroid_belt_to_dict(self):
        data = super().serialize_small_body_to_dict()
        data.update({"density": self.density})
        data.update({"minerals": self.minerals})
        return data

    def generate_asteroid_belt(self, orbit, star):
        """
        Generate an asteroid belt based on its orbit and the properties of its star.
        - orbit: The semi-major axis of the asteroid belt's orbit, in AU.
        - star: The star around which the asteroid belt orbits.
        """
        self.star = star
        self.orbit = orbit
        self.name = f"Asteroid Belt {self.return_orbit_number()}"
        self.body_type = "Asteroid Belt"
        self.density = self.generate_density()
        self.minerals = self.generate_minerals()
        # Calculate the density of the asteroid belt
        # self.additional_info = self.generate_density()

        return self

    def generate_density(self):
        """
        Generate the density of the asteroid belt based on its orbit.
        """
        # Define density ranges based on orbit distance

        if self.orbit < 0.8:
            density = "Sparse"
        elif self.orbit < 2:
            density = "Moderate"
        else:
            density = "Dense"

        return density

    def generate_minerals(self):
        # generate a list of available minerals in the asteroid belt with their abundance in percentage
        # using the RareMineral class from Utility.py to generate the minerals and the MineralMap Class to generate the abundance

        rare_minerals = RareMinerals()
        belt_minerals = []
        list_of_minerals_to_assign = rare_minerals.get_minerals()
        stellar_position_of_belt = self.star.get_cartesian_position()

        # Zones for all the minerals
        scarcity_map = self.star.star_map.mineral_maps

        # generate the abundance of each mineral in the asteroid belt
        for mineral in list_of_minerals_to_assign:
            actual_mineral_map = scarcity_map[mineral]
            abundance = RareMinerals.properties[mineral]["rarity"]

            if self.orbit > 1.2 and mineral == "Rock":
                abundance = abundance * 0.2
            if self.orbit <= 1.2 and mineral == "Water":
                abundance = abundance * 0.2

            # I have to access the nearest zone point in the map for the mineral
            # So I have to chose the correct mineral map from the scarcity_map
            # Then I have to find the nearest point to the belt position

            mineral_scarcity_zones = actual_mineral_map.find_nearest_zone_point(
                stellar_position_of_belt
            )

            # Out of scarcity zone I have to access the scarcity value, the last in the list
            scarcity = mineral_scarcity_zones[-1]
            final_abundance = abundance * scarcity * np.random.uniform(0.8, 1.2)
            belt_minerals.append({mineral: final_abundance})

        # sum up all the abundances

        sum_of_abundances = 0
        for mineral in belt_minerals:
            sum_of_abundances += sum(mineral.values())

        correction_factor = 100 / sum_of_abundances
        for mineral in belt_minerals:
            for key in mineral:
                mineral[key] = round(mineral[key] * correction_factor, 4)

        # sum up all the abundances again

        sum_of_abundances = 0
        for mineral in belt_minerals:
            sum_of_abundances += sum(mineral.values())

        return belt_minerals


class Planetary_System:
    def __init__(self, star, orbits=[], celestial_bodies=None):
        self.star = star
        self.orbits = orbits
        self.celestial_bodies = celestial_bodies if celestial_bodies is not None else []
        self.description = None

    def __str__(self):
        return f"Star: {self.star.name}, Planets: {', '.join([planet.name for planet in self.planets])}"

    def serialize_planetary_system_to_dict(self):
        data = {
            "star": self.star.serialize_star_to_dict(),
            "orbits": self.orbits,
            "celestial_bodies": [
                body.serialize_small_body_to_dict() for body in self.celestial_bodies
            ],
            "description": self.description,
        }
        return data

    def generate_orbits(self, include_habitable_zone=True, num_orbits=1):
        """
        Generate orbits for the planetary system, incorporating advanced considerations.
        - num_orbits: Total number of orbits to generate, including habitable zone if specified.
        - include_habitable_zone: Whether to ensure one orbit is within the habitable zone.
        """
        # Define initial orbit parameters based on protoplanetary disk properties
        initial_orbit = 0.1  # Starting close to the star, in AU
        self.orbits = []  # Ensure the orbits list is empty before starting

        # Calculate the habitable zone if needed
        if include_habitable_zone and self.star.luminosity:

            goldilocks_span = self.star.goldilocks_zone()
            # Ensure one orbit is within the habitable zone
            habitable_zone_orbit = np.random.uniform(
                goldilocks_span[0], goldilocks_span[1]
            )
            self.orbits.append(habitable_zone_orbit)
            num_orbits -= 1  # Adjust the number of additional orbits to generate

        # Generate additional orbits, ensuring the total number does not exceed num_orbits
        while len(self.orbits) < num_orbits + (1 if include_habitable_zone else 0):
            orbit_separation_factor = np.random.lognormal(mean=0.5, sigma=0.2)
            new_orbit = initial_orbit * orbit_separation_factor

            if self.orbits:  # If there are already orbits defined
                min_separation = 0.15  # Minimum separation in AU, placeholder for Hill sphere calculation
                while any(
                    abs(existing_orbit - new_orbit) < min_separation
                    for existing_orbit in self.orbits
                ):
                    new_orbit *= 1.1  # Increase orbit slightly to ensure stability

            self.orbits.append(new_orbit)
            initial_orbit = new_orbit  # Update for next iteration

        self.orbits.sort()  # Sort orbits for easier readability/processing

    def generate_planets_and_asteroid_belts(self):
        self.celestial_bodies = []
        goldilocks_zone = self.star.goldilocks_zone()
        for orbit in self.orbits:
            # Check if the orbit is within the Goldilocks zone
            is_in_goldilocks_zone = goldilocks_zone[0] <= orbit <= goldilocks_zone[1]

            # Decide between generating a planet or an asteroid belt
            # Ensure asteroid belts are not generated within the Goldilocks zone
            if (
                np.random.rand() > 0.75 and not is_in_goldilocks_zone
            ):  # 25% chance to generate an asteroid belt outside Goldilocks zone

                body = AsteroidBelt(star=self.star)
                body = body.generate_asteroid_belt(orbit, self.star)

            else:
                body = Planet(star=self.star)
                if is_in_goldilocks_zone:
                    body = body.generate_planet(orbit, self.star, habitable=True)

                else:
                    body = body.generate_planet(orbit, self.star, habitable=False)

            self.celestial_bodies.append(body)

    def generate_description(self):
        """Generate a description of the planetary system based on its properties."""
        # Handling singular/plural for celestial bodies
        celestial_body_count = len(self.celestial_bodies)
        celestial_body_word = "body" if celestial_body_count == 1 else "bodies"

        description = f"The planetary system of {self.star.name[0]} consists of {celestial_body_count} {celestial_body_word}. "
        if self.star.luminosity:
            description += f"The star has a luminosity of {self.star.luminosity:.2f} solar luminosities. "
        if self.orbits:
            orbit_count = len(self.orbits)
            orbit_word = "orbit" if orbit_count == 1 else "orbits"
            description += f"The system has {orbit_count} distinct {orbit_word}, ranging from {min(self.orbits):.2f} AU to {max(self.orbits):.2f} AU. "
        if celestial_body_count > 0:
            planet_count = sum(
                1 for body in self.celestial_bodies if body.body_type == "Planet"
            )
            asteroid_belt_count = sum(
                1 for body in self.celestial_bodies if body.body_type == "Asteroid Belt"
            )

            # Handling singular/plural for planets
            planet_word = "planet" if planet_count == 1 else "planets"
            # Handling singular/plural for asteroid belts
            asteroid_belt_word = (
                "asteroid belt" if asteroid_belt_count == 1 else "asteroid belts"
            )

            description += f"The system contains {planet_count} {planet_word} and {asteroid_belt_count} {asteroid_belt_word}. "
            # Add a list of all celestial bodies and theit orbits
            description += "The celestial bodies in the system are: "
            for body in self.celestial_bodies:
                description += f"{body.name} at {body.orbit:.2f} AU, "

        self.description = description
        return description

    def assembel_gpt_data(self):
        data = {
            "star": self.star.name,
            "orbits": self.orbits,
            "celestial_bodies": [
                body.serialize_small_body_to_dict() for body in self.celestial_bodies
            ],
        }
        return data


class MineralMap:

    def __init__(self, mineral=None, zone_points=None):

        self.mineral = mineral
        self.zone_points = zone_points

    def __str__(self):

        text = f"{self.mineral} map with {len(self.zone_points)} zone points."
        text += f" The zone points are: {self.zone_points}"

        return text

    def find_nearest_zone_point(self, space_point):
        x, y, z = space_point  # Unpack the coordinates of the given point
        nearest_zone = None
        min_distance = float("inf")
        for point in self.zone_points:
            zone_x, zone_y, zone_z, weight, scarcity = (
                point  # Ignore weight in this comparison
            )

            if weight <= 0:
                continue

            distance = (
                np.sqrt((zone_x - x) ** 2 + (zone_y - y) ** 2 + (zone_z - z) ** 2)
                / weight
            )
            if distance < min_distance:
                min_distance = distance
                nearest_zone = point
        return nearest_zone

    def generate_zone_points(self, mineral, area=500, number=6):
        """
        Generate a random number of zone points for the mineral map.
        """
        zone_points = []
        for i in range(number):
            x = np.random.randint(-area, area)
            y = np.random.randint(-area, area)
            z = np.random.randint(-area, area)
            weight = np.random.uniform(0.1, 1)

            if mineral == "Rock" or mineral == "WaterIce":
                mineral_scarcity = np.random.uniform(0.8, 1.2)

            else:
                mineral_scarcity = np.random.uniform(0, 1.3)

            zone_points.append((x, y, z, weight, mineral_scarcity))

        return zone_points

    def create_maps_for_all_minerals(self, area=500, number=6):
        """
        Create mineral maps for a list of minerals and store them in a dictionary.
        """
        minerals = RareMinerals().get_minerals()
        mineral_maps = {}  # Use a dictionary to store mineral maps

        for mineral in minerals:
            zone_points = self.generate_zone_points(mineral, area, number)
            mineral_map = MineralMap(mineral, zone_points)
            mineral_maps[mineral] = (
                mineral_map  # Store the map with the mineral name as the key
            )

        return mineral_maps
