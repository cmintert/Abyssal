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
        # Create mineral maps and then load the rest of the data
        starmap = Starmap()
        starmap.generate_mineral_maps()

        # Load data from JSON files
        star_data = self.read_from_json("json_data/star_data.json")
        nation_data = self.read_from_json("json_data/nation_data.json")
        planetary_system_data = self.read_from_json(
            "json_data/planetary_system_data.json")
        planet_data = self.read_from_json("json_data/planet_data.json")
        asteroid_belt_data = self.read_from_json(
            "json_data/asteroid_belt_data.json")

        if not all([star_data, nation_data, planetary_system_data, planet_data,
                    asteroid_belt_data]):
            return None

        # Create nation objects first (stars reference nations)
        for nation_info in nation_data:
            nation = Nation(
                name=nation_info["name"],
                origin=nation_info["origin"],
                nation_colour=nation_info["nation_colour"]
            )
            nation.current_radius = nation_info["current_radius"]
            nation.expansion_rate = nation_info["expansion_rate"]

            # Handle the new fields with backward compatibility
            if "autogen_description" in nation_info:
                nation.autogen_description = nation_info["autogen_description"]
            else:
                # Generate a new description if not present in the data
                nation.generate_description()

            nation.additional_info = nation_info.get("additional_info", None)
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

            # Handle the new fields with backward compatibility
            if "autogen_description" in star_info:
                star.autogen_description = star_info["autogen_description"]
            else:
                # Generate a new description if not present in the data
                star.generate_description()

            star.additional_info = star_info.get("additional_info", None)

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

                # Handle the new fields with backward compatibility
                if "autogen_description" in system_info:
                    star.planetary_system.autogen_description = system_info[
                        "autogen_description"]
                else:
                    # Use the description field or generate a new one
                    if "description" in system_info:
                        star.planetary_system.autogen_description = system_info[
                            "description"]
                    else:
                        # Generate a new description if not present in the data
                        star.planetary_system.generate_description()

                star.planetary_system.additional_info = system_info.get(
                    "additional_info", None)

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

                # Handle the new fields with backward compatibility
                if "autogen_description" in planet_info:
                    planet.autogen_description = planet_info[
                        "autogen_description"]
                else:
                    # If no autogen_description, use the additional_info field as the autogen_description
                    # and set additional_info to None (to avoid duplicating the description)
                    planet.autogen_description = planet_info.get(
                        "additional_info", None)
                    # Only clear additional_info if it matches the auto-generated content
                    if planet.autogen_description and planet.autogen_description.startswith(
                            f"Planet {planet.name} is a"):
                        # This appears to be an auto-generated description that was stored in additional_info
                        planet_info["additional_info"] = None

                    # If still no description, generate a new one
                    if not planet.autogen_description:
                        # We need to set all the properties first
                        planet.mass = planet_info["mass"]
                        planet.density = planet_info["density"]
                        planet.radius = planet_info["radius"]
                        planet.composition = planet_info["composition"]
                        planet.orbital_time = planet_info["orbital_time"]
                        planet.rotation_period = planet_info["rotation_period"]
                        planet.tilt = planet_info["tilt"]
                        planet.moons = planet_info["moons"]
                        planet.atmosphere = planet_info["atmosphere"]
                        planet.surface_temperature = planet_info[
                            "surface_temperature"]
                        planet.presence_of_water = planet_info[
                            "presence_of_water"]
                        planet.gravity = planet_info["gravity"]
                        planet.habitable = planet_info["habitable"]

                        # Now we can generate the description
                        planet.autogen_description = planet.create_description()

                planet.additional_info = planet_info.get("additional_info",
                                                         None)

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

                # Handle the new fields with backward compatibility
                if "autogen_description" in belt_info:
                    belt.autogen_description = belt_info["autogen_description"]
                else:
                    # If no description, create one
                    belt_description = f"Asteroid Belt {belt.name} orbiting {star.name[0]} at {belt.orbit:.2f} AU. "

                    # Set AsteroidBelt-specific properties so we can generate a description
                    belt.density = belt_info["density"]
                    belt.minerals = belt_info["minerals"]

                    belt_description += f"The belt density is '{belt.density}'. "

                    mineral_composition = "Mineral Composition: "
                    for mineral in belt.minerals:
                        for key, value in mineral.items():
                            mineral_composition += f"{key}: {value}%, "

                    belt_description += mineral_composition
                    belt.autogen_description = belt_description

                belt.additional_info = belt_info.get("additional_info", None)

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