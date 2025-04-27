import numpy as np
import networkx as nx
from random_generator import RandomGenerator

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
                 stellar_projector_range=20.0,
                 ship_projector_range=1.0,
                 stellar_projector_density=0.2):
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
        Generate both the primary and secondary transportation networks.
        """
        self._generate_primary_network()
        self._generate_secondary_network()
        self._calculate_accessibility_metrics()

    def _generate_primary_network(self):
        """
        Generate the primary network of stellar projector stations.

        This creates a network where:
        1. Sol (0,0,0) always has a stellar projector
        2. ~20% of habitable planets get stellar projectors
        3. Important systems (nation capitals, etc.) are prioritized
        4. Stellar projectors can connect if within 20 LY of each other
        """
        # Add all stars as nodes to the primary network
        for star in self.starmap.stars:
            self.primary_network.add_node(star.id,
                                          coords=(star.x, star.y, star.z),
                                          star=star,
                                          has_projector=False)

        # Identify habitable systems
        habitable_systems = []
        for star in self.starmap.stars:
            has_habitable_planet = any(
                body.body_type == "Planet" and body.habitable
                for body in star.planetary_system.celestial_bodies
            )

            if has_habitable_planet:
                # Sort habitable systems by importance
                importance = 0

                # Nation capitals get priority
                if star.nation and star.nation.origin['x'] == star.x and \
                        star.nation.origin['y'] == star.y and \
                        star.nation.origin['z'] == star.z:
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

                habitable_systems.append((star, importance))

        # Sort by importance (highest first)
        habitable_systems.sort(key=lambda x: x[1], reverse=True)

        # Find the star closest to Sol and ensure it has a projector
        sol_star = None
        min_dist = float('inf')
        for star in self.starmap.stars:
            dist = np.sqrt(star.x ** 2 + star.y ** 2 + star.z ** 2)
            if dist < min_dist:
                min_dist = dist
                sol_star = star

        # Always place a projector at Sol
        if sol_star:
            self.primary_network.nodes[sol_star.id]['has_projector'] = True

        # Assign stellar projectors based on importance
        num_projectors = int(
            len(self.starmap.stars) * self.stellar_projector_density)
        projector_systems = [sol_star.id] if sol_star else []

        for star, _ in habitable_systems:
            if len(projector_systems) >= num_projectors:
                break

            # Only add if not already included
            if star.id not in projector_systems:
                self.primary_network.nodes[star.id]['has_projector'] = True
                projector_systems.append(star.id)

        # Connect stellar projectors if within range
        projector_nodes = [node for node, data in
                           self.primary_network.nodes(data=True)
                           if data['has_projector']]

        for i, node1 in enumerate(projector_nodes):
            coords1 = self.primary_network.nodes[node1]['coords']
            for node2 in projector_nodes[i + 1:]:
                coords2 = self.primary_network.nodes[node2]['coords']

                # Calculate distance in light years (simple estimate: 1 unit = 1 LY)
                distance = np.sqrt(
                    sum((c1 - c2) ** 2 for c1, c2 in zip(coords1, coords2)))

                # Connect if within stellar projector range
                if distance <= self.stellar_projector_range:
                    self.primary_network.add_edge(node1, node2,
                                                  distance=distance,
                                                  type='stellar_projector')

    def _generate_secondary_network(self):
        """
        Generate the secondary network of ship-based projector routes.

        This network connects:
        1. All stars within ship projector range of each other
        2. Creates a more dense but slower transportation network
        """
        # Start with a copy of all nodes from primary network
        self.secondary_network.add_nodes_from(
            self.primary_network.nodes(data=True))

        # Connect all stars within ship projector range
        star_ids = list(self.secondary_network.nodes())

        for i, node1 in enumerate(star_ids):
            coords1 = self.secondary_network.nodes[node1]['coords']
            for node2 in star_ids[i + 1:]:
                coords2 = self.secondary_network.nodes[node2]['coords']

                # Calculate distance
                distance = np.sqrt(
                    sum((c1 - c2) ** 2 for c1, c2 in zip(coords1, coords2)))

                # Connect if within ship projector range
                if distance <= self.ship_projector_range:
                    self.secondary_network.add_edge(node1, node2,
                                                    distance=distance,
                                                    type='ship_projector')

        # Include all primary network connections in the secondary network
        for node1, node2, data in self.primary_network.edges(data=True):
            if not self.secondary_network.has_edge(node1, node2):
                self.secondary_network.add_edge(node1, node2, **data)

    def _calculate_accessibility_metrics(self):
        """
        Calculate accessibility metrics for each star system.

        This adds several metrics to each node:
        1. primary_connected: Whether the system has a direct primary network connection
        2. secondary_connected: Whether the system has a direct secondary network connection
        3. primary_shortest_path: Shortest path (in jumps) to a primary network node
        4. accessibility_score: Overall accessibility score (lower is more accessible)
        """
        # Calculate which systems are connected to each network
        for node in self.primary_network.nodes():
            # Check if connected to the primary network
            primary_connected = self.primary_network.nodes[node][
                'has_projector']
            self.primary_network.nodes[node][
                'primary_connected'] = primary_connected
            self.secondary_network.nodes[node][
                'primary_connected'] = primary_connected

            # Check if connected to the secondary network
            secondary_connected = len(
                list(self.secondary_network.neighbors(node))) > 0
            self.primary_network.nodes[node][
                'secondary_connected'] = secondary_connected
            self.secondary_network.nodes[node][
                'secondary_connected'] = secondary_connected

        # Find Sol-like node (closest to origin)
        sol_node = None
        min_dist = float('inf')
        for node, data in self.primary_network.nodes(data=True):
            coords = data['coords']
            dist = np.sqrt(sum(c ** 2 for c in coords))
            if dist < min_dist:
                min_dist = dist
                sol_node = node

        # Calculate shortest paths and accessibility scores
        if sol_node:
            # Use combined network for path calculations
            combined_network = self.secondary_network.copy()

            # Calculate shortest path lengths from Sol
            try:
                path_lengths = nx.shortest_path_length(combined_network,
                                                       source=sol_node)

                for node in self.primary_network.nodes():
                    # Get shortest path length (defaults to infinity if unreachable)
                    path_length = path_lengths.get(node, float('inf'))

                    # Store shortest path length
                    self.primary_network.nodes[node][
                        'shortest_path_from_sol'] = path_length
                    self.secondary_network.nodes[node][
                        'shortest_path_from_sol'] = path_length

                    # Calculate accessibility score:
                    # - Primary network nodes are most accessible (0-10)
                    # - Nodes close to primary network have medium accessibility (10-50)
                    # - Isolated nodes have poor accessibility (50+)
                    accessibility = path_length * 10

                    # Primary network nodes get a bonus
                    if self.primary_network.nodes[node]['has_projector']:
                        accessibility *= 0.5

                    # Store accessibility score
                    self.primary_network.nodes[node][
                        'accessibility_score'] = accessibility
                    self.secondary_network.nodes[node][
                        'accessibility_score'] = accessibility

            except nx.NetworkXNoPath:
                # Handle disconnected components
                for node in self.primary_network.nodes():
                    self.primary_network.nodes[node][
                        'shortest_path_from_sol'] = float('inf')
                    self.secondary_network.nodes[node][
                        'shortest_path_from_sol'] = float('inf')
                    self.primary_network.nodes[node][
                        'accessibility_score'] = 100
                    self.secondary_network.nodes[node][
                        'accessibility_score'] = 100

        # Calculate hub score (betweenness centrality)
        try:
            betweenness = nx.betweenness_centrality(self.primary_network)
            for node, score in betweenness.items():
                # Scale to 0-10 range
                hub_score = score * 100
                self.primary_network.nodes[node]['hub_score'] = hub_score
                self.secondary_network.nodes[node]['hub_score'] = hub_score
        except:
            # Default hub score if calculation fails
            for node in self.primary_network.nodes():
                self.primary_network.nodes[node]['hub_score'] = 0
                self.secondary_network.nodes[node]['hub_score'] = 0

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