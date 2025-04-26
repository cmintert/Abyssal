# persistence.py
import os
import json
from map_components import Nation, Star, Planetary_System, Planet, AsteroidBelt, \
    MineralMap
from abyssal_map import Starmap


class StarmapReader:
    """Handles reading starmap data from JSON files and constructing objects"""

    def __init__(self):
        pass

    def read_from_json(self, filename):
        """Read data from a JSON file"""
        try:
            with open(filename, "r") as file:
                data = json.load(file)
            print(f"Data loaded from {filename}")
            return data
        except Exception as e:
            print(f"Error loading data from {filename}: {e}")
            return None

    def load_starmap(self):
        """Loads and reconstructs a complete starmap from JSON files"""
        starmap = Starmap()

        # Load stars
        star_data = self.read_from_json("json_data/star_data.json")
        if not star_data:
            return None

        # Load nations
        nation_data = self.read_from_json("json_data/nation_data.json")
        if not nation_data:
            return None

        # Load planetary systems
        planetary_system_data = self.read_from_json(
            "json_data/planetary_system_data.json")
        if not planetary_system_data:
            return None

        # Load planets
        planet_data = self.read_from_json("json_data/planet_data.json")
        if not planet_data:
            return None

        # Load asteroid belts
        asteroid_belt_data = self.read_from_json(
            "json_data/asteroid_belt_data.json")
        if not asteroid_belt_data:
            return None

        # Create mineral maps first
        starmap.generate_mineral_maps()

        # Create nation objects first (stars reference nations)
        for nation_info in nation_data:
            nation = Nation(
                name=nation_info["name"],
                origin=nation_info["origin"],
                nation_colour=nation_info["nation_colour"]
            )
            nation.current_radius = nation_info["current_radius"]
            nation.expansion_rate = nation_info["expansion_rate"]
            nation.additional_info = nation_info["additional_info"]
            starmap.nations.append(nation)

        # Create star objects with placeholder planetary systems
        for star_info in star_data:
            # Find the corresponding nation
            nation = None
            if star_info["nation"]:
                nation = next((n for n in starmap.nations if
                               n.name == star_info["nation"]), None)

            star = Star(
                id=star_info["id"],
                starmap=starmap,
                name=star_info["name"],
                nation=nation,
                x=star_info["x"],
                y=star_info["y"],
                z=star_info["z"],
                r=star_info["r"],
                theta=star_info["theta"],
                phi=star_info["phi"],
                spectral_class=star_info["spectral_class"],
                luminosity=star_info["luminosity"]
            )
            star.additional_info = star_info["additional_info"]

            # Create an empty planetary system for now
            star.planetary_system = Planetary_System(star)

            starmap.stars.append(star)

            # Add star to nation's stars if applicable
            if nation:
                nation.nation_stars.append(star)

        # Process planetary systems data
        for system_info in planetary_system_data:
            # Find the corresponding star
            star_id = system_info["star"]["id"]
            star = next((s for s in starmap.stars if s.id == star_id), None)

            if star:
                # Set orbital data
                star.planetary_system.orbits = system_info["orbits"]
                star.planetary_system.description = system_info["description"]

        # Process planets data and add to their stars
        for planet_info in planet_data:
            # Find the corresponding star
            star_id = planet_info["star"]["id"]
            star = next((s for s in starmap.stars if s.id == star_id), None)

            if star:
                # Create planet
                planet = Planet(star=star)

                # Set basic SmallBody properties
                planet.name = planet_info["name"]
                planet.body_type = planet_info["body_type"]
                planet.orbit = planet_info["orbit"]
                planet.additional_info = planet_info["additional_info"]

                # Set Planet-specific properties
                planet.mass = planet_info["mass"]
                planet.density = planet_info["density"]
                planet.radius = planet_info["radius"]
                planet.composition = planet_info["composition"]
                planet.orbital_time = planet_info["orbital_time"]
                planet.rotation_period = planet_info["rotation_period"]
                planet.tilt = planet_info["tilt"]
                planet.moons = planet_info["moons"]
                planet.atmosphere = planet_info["atmosphere"]
                planet.surface_temperature = planet_info["surface_temperature"]
                planet.presence_of_water = planet_info["presence_of_water"]
                planet.gravity = planet_info["gravity"]
                planet.habitable = planet_info["habitable"]

                # Add planet to the star's planetary system
                star.planetary_system.celestial_bodies.append(planet)

        # Process asteroid belts data and add to their stars
        for belt_info in asteroid_belt_data:
            # Find the corresponding star
            star_id = belt_info["star"]["id"]
            star = next((s for s in starmap.stars if s.id == star_id), None)

            if star:
                # Create asteroid belt
                belt = AsteroidBelt(star=star)

                # Set basic SmallBody properties
                belt.name = belt_info["name"]
                belt.body_type = belt_info["body_type"]
                belt.orbit = belt_info["orbit"]
                belt.additional_info = belt_info["additional_info"]

                # Set AsteroidBelt-specific properties
                belt.density = belt_info["density"]
                belt.minerals = belt_info["minerals"]

                # Add belt to the star's planetary system
                star.planetary_system.celestial_bodies.append(belt)

        # Sort celestial bodies by orbit for each star
        for star in starmap.stars:
            star.planetary_system.celestial_bodies.sort(
                key=lambda body: body.orbit)

        # Populate starmap.used_star_names to prevent duplicates if new stars are added
        starmap.used_star_names = [star.name for star in starmap.stars]

        return starmap

    def check_json_files_exist(self):
        """Check if required JSON files exist"""
        return (
                os.path.exists("json_data/star_data.json") and
                os.path.exists("json_data/nation_data.json") and
                os.path.exists("json_data/planetary_system_data.json") and
                os.path.exists("json_data/planet_data.json") and
                os.path.exists("json_data/asteroid_belt_data.json")
        )


class StarmapWriter:
    """Handles serializing and writing starmap data to JSON files"""

    def __init__(self):
        self.ensure_json_directory()

    def ensure_json_directory(self):
        """Make sure the json_data directory exists"""
        if not os.path.exists("json_data"):
            os.makedirs("json_data")

    def write_to_json(self, data, filename):
        """Write data to a JSON file"""
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
        print(f"Data written to {filename}")

    def save_starmap(self, starmap):
        """Serializes and saves a complete starmap to JSON files"""
        # Save stars
        self.write_to_json(
            [star.serialize_star_to_dict() for star in starmap.stars],
            "json_data/star_data.json"
        )

        # Save nations
        self.write_to_json(
            [nation.serialize_nation_to_dict() for nation in starmap.nations],
            "json_data/nation_data.json"
        )

        # Save planetary systems
        self.write_to_json(
            [star.planetary_system.serialize_planetary_system_to_dict()
             for star in starmap.stars if hasattr(star, 'planetary_system')],
            "json_data/planetary_system_data.json"
        )

        # Save planets
        self.write_to_json(
            [planet.serialize_planet_to_dict()
             for star in starmap.stars
             for planet in star.planetary_system.celestial_bodies
             if planet.body_type == "Planet"],
            "json_data/planet_data.json"
        )

        # Save asteroid belts
        self.write_to_json(
            [belt.serialize_asteroid_belt_to_dict()
             for star in starmap.stars
             for belt in star.planetary_system.celestial_bodies
             if belt.body_type == "Asteroid Belt"],
            "json_data/asteroid_belt_data.json"
        )