import numpy as np
import math
from datetime import datetime
from random_generator import RandomGenerator

random_generator = RandomGenerator.get_instance()


class ColonyPopulation:
    """
    Represents the human population on a habitable planet.

    Attributes:
        planet (Planet): The planet object this population resides on
        founding_year (int): The year the colony was established
        initial_population (int): The starting population of the colony
        current_population (int): The current population size
        growth_rate (float): Annual growth rate as a decimal
        habitability_score (float): 0-1 score representing planet habitability
        classification (str): Classification of the colony (eg. "Primary Hub")
    """

    def __init__(self, planet, founding_year=None, initial_population=None,
                 growth_rate=None, habitability_score=None,
                 classification=None):
        """
        Initialize a colony population.

        Args:
            planet (Planet): The planet this population is on
            founding_year (int, optional): The founding year (defaults to random)
            initial_population (int, optional): The initial population (defaults to calculated)
            growth_rate (float, optional): Annual growth rate (defaults to calculated)
            habitability_score (float, optional): 0-1 score (defaults to calculated)
            classification (str, optional): Colony classification (defaults to calculated)
        """
        self.planet = planet
        self.founding_year = founding_year if founding_year is not None else self._calculate_founding_year()
        self.habitability_score = habitability_score if habitability_score is not None else self._calculate_habitability()
        self.classification = classification if classification is not None else "Unclassified"
        self.initial_population = initial_population if initial_population is not None else self._calculate_initial_population()
        self.growth_rate = growth_rate if growth_rate is not None else self._calculate_growth_rate()
        self.current_population = self._calculate_current_population()

    def _calculate_founding_year(self):
        """Calculate a plausible founding year based on accessibility."""
        # Access transport network data from the planet's star
        star = self.planet.star
        has_projector = False
        accessibility_score = 100  # Default high (poor) accessibility

        # If the star has transport network data
        if hasattr(star, 'transport_data') and star.transport_data:
            has_projector = star.transport_data.get('has_projector', False)
            accessibility_score = star.transport_data.get('accessibility_score',
                                                          100)

        # Base parameters
        earliest_colonization = 2200  # First wave of colonization
        latest_colonization = 2650  # Present day is 2675, with some recent colonies

        # Accessibility affects founding date
        # More accessible systems are colonized earlier
        normalized_accessibility = min(1.0, accessibility_score / 100)
        year_range = latest_colonization - earliest_colonization

        # Stars with projectors are colonized first
        if has_projector:
            founding_year = earliest_colonization + int(
                normalized_accessibility * year_range * 0.5)
        else:
            # Later colonization for non-projector systems
            min_year = earliest_colonization + int(
                year_range * 0.3)  # At least 30% into colonization period
            founding_year = min_year + int(
                normalized_accessibility * (latest_colonization - min_year))

        # Add some randomness
        founding_year += random_generator.randint(-20, 20)

        # Ensure within reasonable bounds
        return max(earliest_colonization,
                   min(latest_colonization, founding_year))

    def _calculate_habitability(self):
        """Calculate a 0-1 habitability score based on planet properties."""
        habitability_score = 0.0

        # Temperature factor (Earth-like is ideal: 15°C)
        temp = self.planet.surface_temperature
        temp_factor = max(0, 1 - abs(
            temp - 15) / 40)  # Penalty for deviation from 15°C

        # Gravity factor (Earth-like is ideal: 9.8 m/s²)
        gravity = self.planet.gravity
        gravity_factor = max(0, 1 - abs(
            gravity - 9.8) / 10)  # Penalty for deviation from Earth gravity

        # Water factor
        water_factor = 0.0
        if self.planet.presence_of_water == "Liquid water":
            water_factor = 1.0
        elif self.planet.presence_of_water == "Moderate likelihood of liquid water":
            water_factor = 0.6
        elif self.planet.presence_of_water == "Low likelihood of liquid water":
            water_factor = 0.3

        # Atmosphere factor
        atmosphere_factor = 0.0
        if "oxygen" in self.planet.atmosphere.lower():
            atmosphere_factor = 1.0
        elif "thick atmosphere" in self.planet.atmosphere.lower():
            atmosphere_factor = 0.7
        elif "thin atmosphere" in self.planet.atmosphere.lower():
            atmosphere_factor = 0.4
        else:
            atmosphere_factor = 0.2

        # Combine factors with different weights
        habitability_score = (
                temp_factor * 0.3 +
                gravity_factor * 0.2 +
                water_factor * 0.3 +
                atmosphere_factor * 0.2
        )

        # Normalize to 0-1 range
        return max(0.1, min(1.0, habitability_score))

    def _calculate_initial_population(self):
        """Calculate initial colonization population based on accessibility and habitability."""
        # Default initial populations based on route type
        primary_network_base = 500000  # 500k people for primary network colonies
        secondary_network_base = 50000  # 50k people for secondary network colonies

        # Access transport network data from the planet's star
        star = self.planet.star
        has_projector = False
        accessibility_score = 100  # Default high (poor) accessibility

        # If the star has transport network data
        if hasattr(star, 'transport_data') and star.transport_data:
            has_projector = star.transport_data.get('has_projector', False)
            accessibility_score = star.transport_data.get('accessibility_score',
                                                          100)

        # Base population depends on network type
        if has_projector:
            base_population = primary_network_base
        else:
            base_population = secondary_network_base

        # Modify based on habitability
        habitability_modifier = 0.5 + (
                    self.habitability_score * 1.5)  # 0.5x to 2.0x

        # Modify based on accessibility (more accessible = more initial settlers)
        accessibility_modifier = 2.0 - min(1.0,
                                           accessibility_score / 100)  # 1.0x to 2.0x

        # Calculate initial population
        initial_pop = int(
            base_population * habitability_modifier * accessibility_modifier)

        # Add randomness (±15%)
        random_factor = random_generator.uniform(0.85, 1.15)
        initial_pop = int(initial_pop * random_factor)

        # Classification based on initial population
        if initial_pop >= 1000000:
            self.classification = "Primary Hub"
        elif initial_pop >= 500000:
            self.classification = "Primary Colony"
        elif initial_pop >= 100000:
            self.classification = "Secondary Hub"
        else:
            self.classification = "Outpost"

        return initial_pop

    def _calculate_growth_rate(self):
        """Calculate annual population growth rate based on colony characteristics."""
        # Base growth rates
        primary_network_growth = 0.023  # 2.3% for primary network
        secondary_network_growth = 0.015  # 1.5% for secondary network

        # Access transport network data from the planet's star
        star = self.planet.star
        has_projector = False
        hub_score = 0

        # If the star has transport network data
        if hasattr(star, 'transport_data') and star.transport_data:
            has_projector = star.transport_data.get('has_projector', False)
            hub_score = star.transport_data.get('hub_score', 0)

        # Base growth rate depends on network type
        if has_projector:
            base_growth = primary_network_growth
        else:
            base_growth = secondary_network_growth

        # Modify based on habitability
        # (better planets have higher growth from both higher birth rates and immigration)
        habitability_modifier = 0.7 + (
                    self.habitability_score * 0.6)  # 0.7x to 1.3x

        # Hub systems get increased growth rate from immigration
        hub_modifier = 1.0 + min(0.5,
                                 hub_score / 20)  # Up to 1.5x for important hubs

        # Age factor - newer colonies grow faster
        colony_age = 2675 - self.founding_year  # Current year is 2675
        age_factor = max(0.8, min(1.3, 1.5 - (
                    colony_age / 300)))  # Higher growth for newer colonies

        # Calculate growth rate
        growth_rate = base_growth * habitability_modifier * hub_modifier * age_factor

        # Add randomness (±0.3%)
        random_factor = random_generator.uniform(-0.003, 0.003)
        return growth_rate + random_factor

    def _calculate_current_population(self):
        """Calculate the current population based on initial population, growth rate, and time."""
        current_year = 2675  # Current year in the Abyssal universe
        years_since_founding = max(0, current_year - self.founding_year)

        # Logistic growth model with carrying capacity based on habitability
        # Carrying capacity is in billions and scales with habitability
        max_habitability_capacity = 15  # 15 billion for perfect planets
        carrying_capacity = max_habitability_capacity * self.habitability_score * 1e9

        # Special case for new colonies
        if years_since_founding < 5:
            # Linear growth in the first few years
            return self.initial_population * (
                        1 + (years_since_founding * self.growth_rate))

        # Logistic growth model: P(t) = K / (1 + ((K-P0)/P0) * e^(-rt))
        # Where:
        # - K is carrying capacity
        # - P0 is initial population
        # - r is growth rate
        # - t is time in years
        population = carrying_capacity / (
                1 + ((
                                 carrying_capacity - self.initial_population) / self.initial_population) *
                math.exp(-self.growth_rate * years_since_founding)
        )

        # Add some randomness (±5%)
        random_factor = random_generator.uniform(0.95, 1.05)
        return int(population * random_factor)

    def update_population(self, current_year=2675):
        """Update the population for a given year."""
        years_since_founding = max(0, current_year - self.founding_year)

        # Skip calculation if colony not yet founded
        if years_since_founding <= 0:
            self.current_population = 0
            return 0

        # Recalculate using the same model as _calculate_current_population
        max_habitability_capacity = 15  # 15 billion for perfect planets
        carrying_capacity = max_habitability_capacity * self.habitability_score * 1e9

        # Special case for new colonies
        if years_since_founding < 5:
            # Linear growth in the first few years
            self.current_population = int(self.initial_population * (
                        1 + (years_since_founding * self.growth_rate)))
            return self.current_population

        # Logistic growth model
        population = carrying_capacity / (
                1 + ((
                                 carrying_capacity - self.initial_population) / self.initial_population) *
                math.exp(-self.growth_rate * years_since_founding)
        )

        self.current_population = int(population)
        return self.current_population

    def get_formatted_population(self):
        """Returns the population formatted for display."""
        if self.current_population >= 1e9:
            return f"{self.current_population / 1e9:.2f} billion"
        elif self.current_population >= 1e6:
            return f"{self.current_population / 1e6:.2f} million"
        elif self.current_population >= 1e3:
            return f"{self.current_population / 1e3:.1f} thousand"
        else:
            return f"{self.current_population}"

    def get_colony_summary(self):
        """Returns a summary of the colony for display."""
        years_established = 2675 - self.founding_year
        population_str = self.get_formatted_population()

        return (
            f"{self.classification}: Founded in {self.founding_year} ({years_established} years ago)\n"
            f"Population: {population_str}\n"
            f"Growth rate: {self.growth_rate * 100:.1f}% per year\n"
            f"Habitability score: {self.habitability_score:.2f} (scale 0-1)")

    def serialize_to_dict(self):
        """Serialize the colony data to a dictionary."""
        return {
            'planet_name': self.planet.name,
            'star_id': self.planet.star.id,
            'founding_year': self.founding_year,
            'initial_population': self.initial_population,
            'current_population': self.current_population,
            'growth_rate': self.growth_rate,
            'habitability_score': self.habitability_score,
            'classification': self.classification
        }


class PopulationModel:
    """
    Manages population across all habitable planets in the Abyssal universe.

    This class handles the generation and tracking of human colonies on habitable
    planets, taking into account the transport network, planet habitability,
    strategic importance, and time-based growth.

    Attributes:
        starmap (Starmap): Reference to the master starmap
        transport_network (TransportNetwork): Reference to the transport network
        colonies (dict): Dictionary mapping planet objects to ColonyPopulation objects
        current_year (int): Current year in the Abyssal universe
    """

    def __init__(self, starmap, transport_network, current_year=2675):
        """
        Initialize the population model.

        Args:
            starmap (Starmap): Reference to the master starmap
            transport_network (TransportNetwork): Reference to the transport network
            current_year (int): Current year in the simulation (default 2675)
        """
        self.starmap = starmap
        self.transport_network = transport_network
        self.colonies = {}
        self.current_year = current_year

    def generate_populations(self):
        """
        Generate colonial populations for all habitable planets.
        """
        # First, ensure all stars have transport data
        self._attach_transport_data_to_stars()

        # Then iterate through all planets
        for star in self.starmap.stars:
            for body in star.planetary_system.celestial_bodies:
                # Only create colonies on habitable planets
                if body.body_type == "Planet" and body.habitable:
                    # Create a colony population
                    colony = ColonyPopulation(body)
                    self.colonies[body] = colony

                    # Attach the colony data to the planet
                    body.colony = colony

                    # Update planet description to include population info
                    self._update_planet_description(body)

    def _attach_transport_data_to_stars(self):
        """Attach transport network data to each star."""
        for star in self.starmap.stars:
            transport_data = self.transport_network.get_star_accessibility(
                star.id)
            if transport_data:
                star.transport_data = transport_data
            else:
                # Default values if no transport data available
                star.transport_data = {
                    'has_projector': False,
                    'primary_connected': False,
                    'secondary_connected': False,
                    'shortest_path_from_sol': float('inf'),
                    'accessibility_score': 100,
                    'hub_score': 0
                }

    def _update_planet_description(self, planet):
        """Update a planet's description to include population information."""
        colony = self.colonies.get(planet)
        if not colony:
            return

        # Create population information
        pop_info = (
            f"\n\nHuman Population: {colony.get_formatted_population()}\n"
            f"Colony Class: {colony.classification}\n"
            f"Established: {colony.founding_year} (Age: {self.current_year - colony.founding_year} years)\n"
            f"Growth Rate: {colony.growth_rate * 100:.1f}% per year"
        )

        # Append to existing description if it exists
        if planet.additional_info:
            if "Human Population:" not in planet.additional_info:
                planet.additional_info += pop_info
        else:
            planet.additional_info = pop_info

    def update_populations(self, new_year=None):
        """
        Update all colony populations to a new year.

        Args:
            new_year (int, optional): Year to update to (default: current_year)
        """
        if new_year:
            self.current_year = new_year

        for planet, colony in self.colonies.items():
            colony.update_population(self.current_year)
            self._update_planet_description(planet)

    def get_total_population(self):
        """Get the total human population across all colonies."""
        return sum(
            colony.current_population for colony in self.colonies.values())

    def get_population_summary(self):
        """Get a summary of population statistics."""
        total_population = self.get_total_population()
        num_colonies = len(self.colonies)

        # Count colonies by classification
        classification_counts = {}
        for colony in self.colonies.values():
            if colony.classification in classification_counts:
                classification_counts[colony.classification] += 1
            else:
                classification_counts[colony.classification] = 1

        # Format total population
        if total_population >= 1e12:
            pop_str = f"{total_population / 1e12:.2f} trillion"
        elif total_population >= 1e9:
            pop_str = f"{total_population / 1e9:.2f} billion"
        else:
            pop_str = f"{total_population / 1e6:.2f} million"

        summary = [
            f"Total Human Population: {pop_str}",
            f"Number of Colonies: {num_colonies}",
            "Colony Classifications:"
        ]

        for classification, count in classification_counts.items():
            summary.append(f"  - {classification}: {count}")

        # Add oldest and newest colonies
        if self.colonies:
            oldest = min(self.colonies.values(), key=lambda c: c.founding_year)
            newest = max(self.colonies.values(), key=lambda c: c.founding_year)

            summary.extend([
                f"Oldest Colony: {oldest.planet.name} (Founded {oldest.founding_year})",
                f"Newest Colony: {newest.planet.name} (Founded {newest.founding_year})"
            ])

        return "\n".join(summary)

    def get_largest_colonies(self, n=5):
        """Get the n largest colonies by population."""
        sorted_colonies = sorted(
            self.colonies.values(),
            key=lambda c: c.current_population,
            reverse=True
        )

        return sorted_colonies[:n]

    def time_advance_simulation(self, years=10):
        """
        Advance the simulation by a specified number of years.

        Args:
            years (int): Number of years to advance
        """
        new_year = self.current_year + years
        self.update_populations(new_year)

        return f"Advanced simulation to year {new_year}. New total population: {self.get_formatted_total_population()}"

    def get_formatted_total_population(self):
        """Returns the total population formatted for display."""
        total = self.get_total_population()

        if total >= 1e12:
            return f"{total / 1e12:.2f} trillion"
        elif total >= 1e9:
            return f"{total / 1e9:.2f} billion"
        else:
            return f"{total / 1e6:.2f} million"

    def serialize_to_dict(self):
        """Serialize all colony data to a dictionary."""
        return {
            'current_year': self.current_year,
            'total_population': self.get_total_population(),
            'colonies': [colony.serialize_to_dict() for colony in
                         self.colonies.values()]
        }

    def save_population_data(self, filename="json_data/population_data.json"):
        """Save population data to a JSON file."""
        import json
        import os

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w') as f:
            json.dump(self.serialize_to_dict(), f, indent=4)