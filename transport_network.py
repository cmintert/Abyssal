import numpy as np
import networkx as nx

import config
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
        Generate both the primary and secondary transportation networks.
        """
        self._generate_primary_network()
        self._generate_secondary_network()
        self._calculate_accessibility_metrics()

    def _generate_primary_network(self):
        """
        Generate the primary network of stellar projector stations ensuring:
        1. Nations have internally connected networks
        2. All systems eventually connect back to Sol
        3. Limited connections per system (1-3)
        """
        # Add all stars as nodes to the primary network
        for star in self.starmap.stars:
            self.primary_network.add_node(star.id,
                                          coords=(star.x, star.y, star.z),
                                          star=star,
                                          has_projector=False)

        # Find the star closest to Sol (origin)
        sol_node = self._find_sol_node()

        # Organize stars by nation
        nation_stars = {}
        nationless_stars = []

        for star in self.starmap.stars:
            if star.nation:
                nation_name = star.nation.name
                if nation_name not in nation_stars:
                    nation_stars[nation_name] = []
                nation_stars[nation_name].append(star)
            else:
                nationless_stars.append(star)

        # Calculate importance for all stars
        star_importance = {}
        for star in self.starmap.stars:
            importance = self._calculate_system_importance(star)
            star_importance[star.id] = importance

        # Determine which systems get projectors
        num_projectors = int(
            len(self.starmap.stars) * self.stellar_projector_density)
        projector_systems = [sol_node]  # Sol always gets a projector
        self.primary_network.nodes[sol_node]['has_projector'] = True

        # Always give nation capitals projectors first
        for nation_name, stars in nation_stars.items():
            # Find the nation capital (closest to nation origin)
            capital = self._find_nation_capital(stars)
            if capital and capital.id not in projector_systems:
                self.primary_network.nodes[capital.id]['has_projector'] = True
                projector_systems.append(capital.id)

        # Assign remaining projectors to most important systems
        all_stars = []
        for nation_stars_list in nation_stars.values():
            all_stars.extend(nation_stars_list)
        all_stars.extend(nationless_stars)

        # Sort stars by importance
        all_stars.sort(key=lambda star: star_importance.get(star.id, 0),
                       reverse=True)

        # Assign projectors until we hit our limit
        for star in all_stars:
            if len(projector_systems) >= num_projectors:
                break
            if star.id not in projector_systems:
                self.primary_network.nodes[star.id]['has_projector'] = True
                projector_systems.append(star.id)

        # Track connections per node
        connections_count = {node: 0 for node in projector_systems}

        # Set max connections based on importance
        max_connections = {}
        for node in projector_systems:
            importance = star_importance.get(node, 0)
            # Nation capitals and Sol get 3 connections, others get 1
            if node == sol_node:
                max_connections[node] = 3  # Sol gets 3 connections
            elif any(self._is_nation_capital(self.star_map[node],
                                             nation_stars_list)
                     for nation_name, nation_stars_list in
                     nation_stars.items()):
                max_connections[node] = 3  # Nation capitals get 3 connections
            elif importance > 5:
                max_connections[
                    node] = 3  # Very important systems get 3 connections
            else:
                max_connections[node] = 1  # Regular systems get 1 connection

        # PHASE 1: Connect each nation's systems internally
        for nation_name, stars in nation_stars.items():
            nation_projector_systems = [star.id for star in stars
                                        if star.id in projector_systems]

            if not nation_projector_systems:
                continue  # Skip nations with no projector systems

            # Find the nation capital
            capital = self._find_nation_capital(stars)

            if not capital or capital.id not in projector_systems:
                # If no capital with projector, use the most important system
                nation_projector_systems.sort(
                    key=lambda node: star_importance.get(node, 0), reverse=True)
                nation_capital_id = nation_projector_systems[0]
            else:
                nation_capital_id = capital.id

            # Connect other nation systems to the capital where possible
            connected_systems = {nation_capital_id}
            remaining = [node for node in nation_projector_systems
                         if node != nation_capital_id]

            # Build a minimal spanning tree for the nation
            while remaining and any(
                    connections_count[node] < max_connections[node]
                    for node in connected_systems):
                # Find best connection between a connected and unconnected system
                best_connection = None
                shortest_distance = float('inf')

                for connected_node in connected_systems:
                    # Skip if this node already has max connections
                    if connections_count[connected_node] >= max_connections[
                        connected_node]:
                        continue

                    for unconnected_node in remaining:
                        # Calculate distance
                        coords1 = self.primary_network.nodes[connected_node][
                            'coords']
                        coords2 = self.primary_network.nodes[unconnected_node][
                            'coords']
                        distance = np.sqrt(sum(
                            (c1 - c2) ** 2 for c1, c2 in zip(coords1, coords2)))

                        # If within standard range and better than current best
                        if distance <= self.stellar_projector_range and distance < shortest_distance:
                            shortest_distance = distance
                            best_connection = (connected_node, unconnected_node,
                                               distance)

                # If no standard connection possible, try mega projector within the nation
                if best_connection is None and any(
                        connections_count[node] < max_connections[node]
                        for node in connected_systems):
                    for connected_node in connected_systems:
                        # Skip if this node already has max connections
                        if connections_count[connected_node] >= max_connections[
                            connected_node]:
                            continue

                        for unconnected_node in remaining:
                            coords1 = \
                            self.primary_network.nodes[connected_node]['coords']
                            coords2 = \
                            self.primary_network.nodes[unconnected_node][
                                'coords']
                            distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in
                                                   zip(coords1, coords2)))

                            # If within mega projector range and better than current best
                            if distance <= 300.0 and distance < shortest_distance:
                                shortest_distance = distance
                                best_connection = (connected_node,
                                                   unconnected_node, distance)
                                # Mark this as a mega projector connection
                                connection_type = 'mega_projector'

                # If we found a connection, add it
                if best_connection:
                    node1, node2, distance = best_connection
                    connection_type = 'stellar_projector' if distance <= self.stellar_projector_range else 'mega_projector'

                    self.primary_network.add_edge(node1, node2,
                                                  distance=distance,
                                                  type=connection_type)
                    connections_count[node1] += 1
                    connections_count[node2] += 1

                    # Mark as connected
                    connected_systems.add(node2)
                    remaining.remove(node2)
                else:
                    # If no more connections possible within this nation, break
                    break

        # PHASE 2: Connect nation networks back to Sol
        # Group all connected components
        components = list(nx.connected_components(self.primary_network))

        # Find the component containing Sol
        sol_component = None
        for component in components:
            if sol_node in component:
                sol_component = component
                break

        # If Sol isn't connected to anything yet, we need to fix that
        if not sol_component:
            sol_component = {sol_node}

        # Connect other components to the Sol component
        for component in components:
            if component == sol_component:
                continue

            # Find best nodes to connect between components
            best_connection = None
            shortest_distance = float('inf')
            connection_type = 'mega_projector'  # Assume mega projector for inter-component

            for sol_comp_node in sol_component:
                # Skip if this node already has max connections
                if connections_count.get(sol_comp_node,
                                         0) >= max_connections.get(
                        sol_comp_node, 0):
                    continue

                for other_comp_node in component:
                    # Skip if this node already has max connections
                    if connections_count.get(other_comp_node,
                                             0) >= max_connections.get(
                            other_comp_node, 0):
                        continue

                    coords1 = self.primary_network.nodes[sol_comp_node][
                        'coords']
                    coords2 = self.primary_network.nodes[other_comp_node][
                        'coords']
                    distance = np.sqrt(
                        sum((c1 - c2) ** 2 for c1, c2 in zip(coords1, coords2)))

                    # Check for standard projector range first
                    if distance <= self.stellar_projector_range:
                        connection_type = 'stellar_projector'
                    # Then check for mega projector range
                    elif distance <= 300.0:
                        connection_type = 'mega_projector'
                    else:
                        continue  # Skip if beyond mega projector range

                    # If this is the best connection so far, save it
                    if distance < shortest_distance:
                        shortest_distance = distance
                        best_connection = (sol_comp_node, other_comp_node,
                                           distance, connection_type)

            # Add the best connection if found
            if best_connection:
                node1, node2, distance, conn_type = best_connection
                self.primary_network.add_edge(node1, node2,
                                              distance=distance,
                                              type=conn_type)
                connections_count[node1] = connections_count.get(node1, 0) + 1
                connections_count[node2] = connections_count.get(node2, 0) + 1

                # Merge this component into the Sol component
                sol_component.update(component)
            else:
                print(
                    f"Warning: Unable to connect a component ({len(component)} nodes) back to Sol network")

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