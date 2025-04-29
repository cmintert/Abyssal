import numpy as np
import networkx as nx

import config
from random_generator import RandomGenerator
from stell_network_generator import StellarNetworkGenerator

random_generator = RandomGenerator.get_instance()


class TransportNetwork:
    """
    Models the interstellar transportation infrastructure of the Abyssal universe.

    This class handles the generation and analysis of the dual transportation system:
    1. Primary Network: Stellar Projector stations enabling 20 LY jumps
    2. Secondary Network: Ship-based projectors enabling shorter 1 LY jumps

    Attributes:
        starmap (Starmap): Reference to the master starmap
        primary_network (nx.Graph): Network graph of stellar projector connections
        secondary_network (nx.Graph): Network graph of ship-based projector routes
        sol_coordinates (tuple): Coordinates of Sol (origin point)
        stellar_projector_range (float): Maximum jump distance for stellar projectors in LY
        ship_projector_range (float): Maximum jump distance for ship projectors in LY
        stellar_projector_density (float): Probability of a star having a stellar projector
    """

    def __init__(self, starmap,
                 stellar_projector_range=config.STELLAR_PROJECTOR_RANGE,
                 ship_projector_range=config.SHIP_PROJECTOR_RANGE,
                 stellar_projector_density=config.STELLAR_PROJECTOR_DENSITY):
        """
        Initialize the transport network.

        Args:
            starmap (Starmap): Reference to the master starmap
            stellar_projector_range (float): Maximum jump distance for stellar projectors in LY
            ship_projector_range (float): Maximum jump distance for ship projectors in LY
            stellar_projector_density (float): Approximate fraction of stars with stellar projectors
        """
        self.starmap = starmap
        self.primary_network = nx.Graph()
        self.secondary_network = nx.Graph()
        self.sol_coordinates = (0, 0, 0)  # Origin point
        self.stellar_projector_range = stellar_projector_range
        self.ship_projector_range = ship_projector_range
        self.stellar_projector_density = stellar_projector_density

        # Mapping from star ID to star object for quick lookups
        self.star_map = {star.id: star for star in self.starmap.stars}

    def generate_networks(self):
        """
        Generate both the primary and secondary transportation networks using the advanced generator.
        """
        # Create an instance of the advanced network generator
        network_generator = StellarNetworkGenerator(
            starmap=self.starmap,
            stellar_projector_range=self.stellar_projector_range,
            ship_projector_range=self.ship_projector_range,
            stellar_projector_density=self.stellar_projector_density,
            max_connections_per_system=4
            # You might want to make this configurable
        )

        # Generate the networks
        self.primary_network, self.secondary_network = network_generator.generate_networks()

        # Update the star_map reference in case it was modified
        self.star_map = {star.id: star for star in self.starmap.stars}

        # Add accessibility data to stars directly
        for star in self.starmap.stars:
            star.transport_data = self._get_star_accessibility_from_network(
                star.id)

        # Log network statistics
        stats = self._calculate_network_stats()
        print(f"Network Generation Complete:")
        print(
            f"  - Projector Systems: {stats['primary_node_count']} ({stats['primary_node_count'] / len(self.starmap.stars) * 100:.1f}%)")
        print(f"  - Primary Network Connections: {stats['primary_edge_count']}")
        print(
            f"  - Secondary Network Connections: {stats['secondary_edge_count']}")

        return self.primary_network, self.secondary_network

    def _get_star_accessibility_from_network(self, star_id):
        """
        Extract accessibility metrics for a star from the network.
        """
        if star_id in self.primary_network.nodes():
            return {
                'has_projector': self.primary_network.nodes[star_id].get(
                    'has_projector', False),
                'primary_connected': self.primary_network.nodes[star_id].get(
                    'primary_connected', False),
                'secondary_connected': self.primary_network.nodes[star_id].get(
                    'secondary_connected', False),
                'shortest_path_from_sol': self.primary_network.nodes[
                    star_id].get('shortest_path_from_sol', float('inf')),
                'accessibility_score': self.primary_network.nodes[star_id].get(
                    'accessibility_score', 100),
                'hub_score': self.primary_network.nodes[star_id].get(
                    'hub_score', 0)
            }
        return None

    def _calculate_network_stats(self):
        """
        Calculate basic network statistics.
        """
        primary_node_count = sum(
            1 for _, data in self.primary_network.nodes(data=True)
            if data.get('has_projector', False))

        return {
            'primary_node_count': primary_node_count,
            'primary_edge_count': self.primary_network.number_of_edges(),
            'secondary_edge_count': self.secondary_network.number_of_edges(),
            'average_node_degree_primary': sum(
                dict(self.primary_network.degree()).values()) /
                                           self.primary_network.number_of_nodes() if self.primary_network.number_of_nodes() > 0 else 0,
            'average_node_degree_secondary': sum(
                dict(self.secondary_network.degree()).values()) /
                                             self.secondary_network.number_of_nodes() if self.secondary_network.number_of_nodes() > 0 else 0
        }

    def _find_nation_capital(self, nation_stars):
        """Find the capital for a nation (closest to nation origin)."""
        if not nation_stars:
            return None

        # Assume nation is the same for all stars in the list
        nation = nation_stars[0].nation
        if not nation:
            return None

        # Find star closest to nation origin
        closest_star = None
        min_distance = float('inf')

        for star in nation_stars:
            distance = np.sqrt(
                (star.x - nation.origin['x']) ** 2 +
                (star.y - nation.origin['y']) ** 2 +
                (star.z - nation.origin['z']) ** 2
            )

            if distance < min_distance:
                min_distance = distance
                closest_star = star

        return closest_star

    def _is_nation_capital(self, star, nation_stars):
        """Check if a star is the capital of its nation."""
        capital = self._find_nation_capital(nation_stars)
        return capital and capital.id == star.id

    def _find_sol_node(self):
        """Find the star closest to Sol (origin point)."""
        sol_node = None
        min_dist = float('inf')
        for node, data in self.primary_network.nodes(data=True):
            coords = data['coords']
            dist = np.sqrt(sum(c ** 2 for c in coords))
            if dist < min_dist:
                min_dist = dist
                sol_node = node
        return sol_node

    def _calculate_system_importance(self, star):
        """Calculate a star system's strategic importance."""
        importance = 0

        # Nation capitals get priority
        if star.nation and star.nation.origin['x'] == star.x and \
                star.nation.origin['y'] == star.y and star.nation.origin[
            'z'] == star.z:
            importance += 10

        # Systems closer to Sol get priority
        dist_to_sol = np.sqrt(star.x ** 2 + star.y ** 2 + star.z ** 2)
        importance += max(0, 500 - dist_to_sol) / 50

        # G-Type stars (like Sol) get priority for stellar projectors
        if star.spectral_class == "G-Type":
            importance += 3
        elif star.spectral_class == "K-Type":
            importance += 2
        elif star.spectral_class == "M-Type":
            importance += 1

        # Systems with more habitable planets get higher importance
        habitable_count = sum(
            1 for body in star.planetary_system.celestial_bodies
            if body.body_type == "Planet" and body.habitable
        )
        importance += habitable_count * 2

        return importance





    def get_star_accessibility(self, star_id):
        """
        Get the accessibility metrics for a specific star.

        Args:
            star_id (int): The ID of the star

        Returns:
            dict: Dictionary of accessibility metrics
        """
        if star_id in self.primary_network.nodes():
            return {
                'has_projector': self.primary_network.nodes[star_id][
                    'has_projector'],
                'primary_connected': self.primary_network.nodes[star_id][
                    'primary_connected'],
                'secondary_connected': self.primary_network.nodes[star_id][
                    'secondary_connected'],
                'shortest_path_from_sol': self.primary_network.nodes[star_id][
                    'shortest_path_from_sol'],
                'accessibility_score': self.primary_network.nodes[star_id][
                    'accessibility_score'],
                'hub_score': self.primary_network.nodes[star_id]['hub_score']
            }
        return None

    def get_network_stats(self):
        """
        Get statistics about the network.

        Returns:
            dict: Dictionary of network statistics
        """
        primary_projector_count = sum(
            1 for _, data in self.primary_network.nodes(data=True)
            if data['has_projector'])

        return {
            'primary_node_count': primary_projector_count,
            'primary_edge_count': self.primary_network.number_of_edges(),
            'secondary_edge_count': self.secondary_network.number_of_edges(),
            'average_node_degree_primary': sum(dict(
                self.primary_network.degree()).values()) / self.primary_network.number_of_nodes(),
            'average_node_degree_secondary': sum(dict(
                self.secondary_network.degree()).values()) / self.secondary_network.number_of_nodes()
        }