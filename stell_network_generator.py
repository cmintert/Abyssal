import numpy as np
import networkx as nx
import logging
import math
from collections import defaultdict

import config

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StellarNetwork")


class StellarNetworkGenerator:
    """
    Advanced generator for stellar transportation networks based on modern transportation planning principles.
    Creates realistic projector networks that connect habitable planets, prioritize nation internal connections,
    and apply appropriate connection limits per system.
    """

    def __init__(self, starmap,
                 stellar_projector_range=config.STELLAR_PROJECTOR_RANGE,
                 ship_projector_range=config.SHIP_PROJECTOR_RANGE,
                 stellar_projector_density=config.STELLAR_PROJECTOR_DENSITY,
                 max_connections_per_system=config.MAX_CONNECTIONS_PER_SYSTEM,
                 log_level=logging.INFO):
        """
        Initialize the stellar network generator.

        Args:
            starmap: The starmap containing stars and their properties
            stellar_projector_range: Maximum range for stellar projectors in LY
            ship_projector_range: Maximum range for ship projectors in LY
            stellar_projector_density: Desired fraction of stars with projectors
            max_connections_per_system: Maximum number of connections per system
            log_level: Logging level
        """
        self.starmap = starmap
        self.stellar_projector_range = stellar_projector_range
        self.ship_projector_range = ship_projector_range
        self.stellar_projector_density = stellar_projector_density
        self.max_connections_per_system = max_connections_per_system

        # Configure logging
        logger.setLevel(log_level)

        # Initialize network graphs
        self.primary_network = nx.Graph()
        self.secondary_network = nx.Graph()

        # Create star ID to star object mapping
        self.star_map = {star.id: star for star in self.starmap.stars}

        # Setup data structures
        self.sol_node = None
        self.nation_capitals = {}
        self.star_importance = {}
        self.connection_counts = defaultdict(int)
        self.max_connections = {}
        self.habitable_systems = set()

        logger.info(
            f"Initialized StellarNetworkGenerator with {len(self.starmap.stars)} stars")

    def generate_networks(self):
        """
        Generate both primary and secondary networks according to a phased approach.
        """
        logger.info("Starting network generation process")

        # Phase 1: Pre-analysis
        self._pre_analysis_phase()

        # Phase 2: Projector allocation
        self._projector_allocation_phase()

        # Phase 3: Nation internal connection
        self._nation_internal_connection_phase()

        # Phase 4: Inter-nation connection
        self._inter_nation_connection_phase()

        # Phase 5: Network optimization
        self._network_optimization_phase()

        # Phase 6: Validation and metrics calculation
        self._validation_and_metrics_phase()

        # Generate secondary network
        self._generate_secondary_network()

        logger.info("Network generation completed successfully")
        return self.primary_network, self.secondary_network

    def _pre_analysis_phase(self):
        """
        Analyze the star distribution to determine if network parameters are feasible.
        Based on nearest-neighbor analysis rather than all-to-all distances.
        """
        logger.info("Starting pre-analysis phase")

        # Add all stars as nodes to the primary network
        for star in self.starmap.stars:
            self.primary_network.add_node(star.id,
                                          coords=(star.x, star.y, star.z),
                                          star=star,
                                          has_projector=False)

        # Find Sol node (closest to origin)
        self.sol_node = self._find_sol_node()
        logger.info(f"Identified Sol node: Star ID {self.sol_node}")

        # Calculate nearest neighbor distances
        # For each star, find the distance to its K nearest neighbors
        k_nearest = 8  # Number of nearest neighbors to analyze
        all_nearest_distances = []

        # Calculate all pairwise distances once for efficiency
        star_coords = [(star.id, (star.x, star.y, star.z)) for star in
                       self.starmap.stars]

        for i, (star_id, coords) in enumerate(star_coords):
            # Calculate distances to all other stars
            distances = []
            for j, (other_id, other_coords) in enumerate(star_coords):
                if i != j:  # Don't compare a star to itself
                    dist = math.sqrt(sum(
                        (c1 - c2) ** 2 for c1, c2 in zip(coords, other_coords)))
                    distances.append((other_id, dist))

            # Sort by distance and take k nearest
            distances.sort(key=lambda x: x[1])
            nearest_k = distances[:k_nearest]

            # Extract just the distances
            nearest_distances = [d for _, d in nearest_k]
            all_nearest_distances.extend(nearest_distances)

        # Calculate statistics on nearest neighbor distances
        avg_distance = np.mean(all_nearest_distances)
        median_distance = np.median(all_nearest_distances)
        min_distance = np.min(all_nearest_distances)
        max_distance = np.max(all_nearest_distances)

        logger.info(f"Nearest {k_nearest} neighbors distance statistics: "
                    f"min={min_distance:.2f} LY, max={max_distance:.2f} LY, "
                    f"avg={avg_distance:.2f} LY, median={median_distance:.2f} LY")

        # Check if our projector range is realistic
        if median_distance > self.stellar_projector_range:
            logger.warning(
                f"Stellar projector range ({self.stellar_projector_range} LY) is less than "
                f"median nearest neighbor distance ({median_distance:.2f} LY). Network may be disconnected.")

            suggested_range = math.ceil(median_distance * 1.5)
            logger.info(
                f"Suggested stellar projector range: {suggested_range} LY")

        # Estimate minimum required projector density
        # Based on percolation theory principles for random networks
        # This now uses nearest neighbor connectivity
        connected_pairs = sum(1 for d in all_nearest_distances if
                              d <= self.stellar_projector_range)
        total_pairs = len(all_nearest_distances)
        conn_prob = connected_pairs / total_pairs if total_pairs > 0 else 0

        # If connection probability is too low, we need more projectors
        if conn_prob < 0.2:
            # Rough estimate based on network theory - need to connect at least 20% of nearest neighbors
            estimated_min_density = 0.2 / conn_prob if conn_prob > 0 else 1.0
            estimated_min_density = min(estimated_min_density,
                                        0.9)  # Cap at 90% to avoid unreasonable values
        else:
            estimated_min_density = self.stellar_projector_density

        logger.info(
            f"Connection probability for nearest neighbors: {conn_prob:.4f}")
        logger.info(
            f"Estimated minimum projector density: {estimated_min_density:.4f}")

        if self.stellar_projector_density < estimated_min_density:
            logger.warning(
                f"Current projector density ({self.stellar_projector_density}) may be too low for a connected network")
            logger.info(
                f"Suggested minimum projector density: {estimated_min_density:.4f}")

        # Identify habitable systems
        for star in self.starmap.stars:
            for body in star.planetary_system.celestial_bodies:
                if body.body_type == "Planet" and hasattr(body,
                                                          'habitable') and body.habitable:
                    self.habitable_systems.add(star.id)
                    break

        logger.info(
            f"Identified {len(self.habitable_systems)} systems with habitable planets")

        # Organize stars by nation
        self.nations = defaultdict(list)
        self.nationless_stars = []

        for star in self.starmap.stars:
            if star.nation:
                nation_name = star.nation.name
                self.nations[nation_name].append(star)
            else:
                self.nationless_stars.append(star)

        logger.info(
            f"Identified {len(self.nations)} nations and {len(self.nationless_stars)} nationless stars")

        # Identify nation capitals
        for nation_name, stars in self.nations.items():
            capital = self._find_nation_capital(stars)
            if capital:
                self.nation_capitals[nation_name] = capital
                logger.info(
                    f"Identified capital for {nation_name}: Star ID {capital.id}")
            else:
                logger.warning(f"Could not identify capital for {nation_name}")

    def _projector_allocation_phase(self):
        """
        Allocate projectors to stars based on strategic importance.
        """
        logger.info("Starting projector allocation phase")

        # Calculate importance for all stars
        for star in self.starmap.stars:
            self.star_importance[star.id] = self._calculate_system_importance(
                star)

        # Determine total number of projectors to allocate
        num_projectors = int(
            len(self.starmap.stars) * self.stellar_projector_density)
        logger.info(
            f"Planning to allocate {num_projectors} projectors ({self.stellar_projector_density * 100:.1f}% of systems)")

        # Create a list to track which stars have projectors
        projector_systems = []

        # Sol always gets a projector
        self.primary_network.nodes[self.sol_node]['has_projector'] = True
        projector_systems.append(self.sol_node)
        logger.info(f"Allocated projector to Sol (Star ID {self.sol_node})")

        # Nation capitals always get projectors
        for nation_name, capital in self.nation_capitals.items():
            if capital.id not in projector_systems:
                self.primary_network.nodes[capital.id]['has_projector'] = True
                projector_systems.append(capital.id)
                logger.info(
                    f"Allocated projector to {nation_name} capital (Star ID {capital.id})")

        # Habitable systems get high priority for projectors
        habitable_candidates = sorted(
            [(star_id, self.star_importance[star_id]) for star_id in
             self.habitable_systems
             if star_id not in projector_systems],
            key=lambda x: x[1], reverse=True
        )

        habitable_allocation = min(len(habitable_candidates),
                                   int(num_projectors * 0.5))
        logger.info(
            f"Allocating {habitable_allocation} projectors to habitable systems")

        for i in range(habitable_allocation):
            if i < len(habitable_candidates):
                star_id = habitable_candidates[i][0]
                self.primary_network.nodes[star_id]['has_projector'] = True
                projector_systems.append(star_id)
                logger.info(
                    f"Allocated projector to habitable system (Star ID {star_id}, "
                    f"Importance: {self.star_importance[star_id]:.2f})")

        # Remaining projectors allocated by importance
        remaining_candidates = sorted(
            [(star.id, self.star_importance[star.id]) for star in
             self.starmap.stars
             if star.id not in projector_systems],
            key=lambda x: x[1], reverse=True
        )

        remaining_allocation = num_projectors - len(projector_systems)
        logger.info(
            f"Allocating {remaining_allocation} remaining projectors based on importance")

        for i in range(remaining_allocation):
            if i < len(remaining_candidates):
                star_id = remaining_candidates[i][0]
                self.primary_network.nodes[star_id]['has_projector'] = True
                projector_systems.append(star_id)
                logger.info(
                    f"Allocated projector to system (Star ID {star_id}, "
                    f"Importance: {self.star_importance[star_id]:.2f})")

        # Set maximum connections per system based on importance
        for node in projector_systems:
            importance = self.star_importance.get(node, 0)

            if node == self.sol_node:
                self.max_connections[
                    node] = self.max_connections_per_system  # Sol gets maximum
            elif node in [capital.id for capital in
                          self.nation_capitals.values()]:
                self.max_connections[
                    node] = self.max_connections_per_system - 1  # Capitals get high capacity
            elif importance > 8:
                self.max_connections[
                    node] = self.max_connections_per_system - 1  # Very important systems
            elif importance > 5:
                self.max_connections[
                    node] = self.max_connections_per_system - 2  # Important systems
            else:
                self.max_connections[node] = 2  # Regular systems

        logger.info(f"Allocated a total of {len(projector_systems)} projectors")

    def _nation_internal_connection_phase(self):
        """
        Create connections within each nation, prioritizing habitable worlds.
        """
        logger.info("Starting nation internal connection phase")

        # Process each nation separately
        for nation_name, stars in self.nations.items():
            logger.info(f"Processing internal connections for {nation_name}")

            # Find systems with projectors in this nation
            nation_projector_systems = [star.id for star in stars
                                        if self.primary_network.nodes[star.id][
                                            'has_projector']]

            if not nation_projector_systems:
                logger.warning(
                    f"No projector systems found in {nation_name}, skipping")
                continue

            capital_id = self.nation_capitals.get(nation_name, None)
            if capital_id is None or capital_id.id not in nation_projector_systems:
                logger.warning(
                    f"Capital not found among projector systems for {nation_name}")
                # Use the most important system as a substitute capital
                nation_projector_systems.sort(
                    key=lambda x: self.star_importance.get(x, 0), reverse=True)
                capital_id = nation_projector_systems[0]
                logger.info(
                    f"Using Star ID {capital_id} as substitute capital for {nation_name}")
            else:
                capital_id = capital_id.id

            # First pass: connect habitable systems within the nation
            habitable_projectors = [node for node in nation_projector_systems
                                    if node in self.habitable_systems]

            logger.info(
                f"First pass: connecting {len(habitable_projectors)} habitable projector systems in {nation_name}")
            self._connect_systems_mst(
                nodes=habitable_projectors,
                start_node=capital_id if capital_id in habitable_projectors else
                habitable_projectors[0] if habitable_projectors else None,
                tag=f"habitable-{nation_name}"
            )

            # Second pass: connect remaining projector systems to the network
            remaining_projectors = [node for node in nation_projector_systems
                                    if node not in habitable_projectors]

            if habitable_projectors:
                start_nodes = habitable_projectors
            else:
                start_nodes = [capital_id]

            logger.info(
                f"Second pass: connecting {len(remaining_projectors)} remaining projector systems in {nation_name}")
            self._connect_systems_to_network(
                nodes=remaining_projectors,
                existing_network=start_nodes,
                tag=f"internal-{nation_name}"
            )

    def _inter_nation_connection_phase(self):
        """
        Create connections between nations, focusing on connecting to Sol.
        """
        logger.info("Starting inter-nation connection phase")

        # First, identify all connected components in the current network
        components = list(nx.connected_components(self.primary_network))
        logger.info(
            f"Current network has {len(components)} disconnected components")

        # Find the component containing Sol
        sol_component = None
        for component in components:
            if self.sol_node in component:
                sol_component = component
                break

        if not sol_component:
            logger.warning(
                "Sol component not found in network, creating singleton component")
            sol_component = {self.sol_node}

        logger.info(f"Sol component contains {len(sol_component)} systems")

        # For each nation not connected to Sol, create strategic connections
        nations_connected = set()
        for nation_name, stars in self.nations.items():
            # Check if any system in this nation is in the Sol component
            nation_in_sol = any(star.id in sol_component for star in stars)

            if nation_in_sol:
                logger.info(
                    f"{nation_name} is already connected to Sol network")
                nations_connected.add(nation_name)
                continue

            # Find nation systems with projectors
            nation_projectors = [star.id for star in stars
                                 if self.primary_network.nodes[star.id][
                                     'has_projector']]

            if not nation_projectors:
                logger.warning(
                    f"No projector systems found in {nation_name}, cannot connect to Sol")
                continue

            # Find the best connection to Sol component
            best_connection = self._find_best_inter_component_connection(
                component1=nation_projectors,
                component2=sol_component,
                mega_projector_range=100.0
                # Allow longer connections between nations
            )

            if best_connection:
                node1, node2, distance, conn_type = best_connection
                self.primary_network.add_edge(
                    node1, node2,
                    distance=distance,
                    type=conn_type,
                    description=f"Inter-nation link: {nation_name} to Sol network"
                )

                # Update connection counts
                self.connection_counts[node1] += 1
                self.connection_counts[node2] += 1

                logger.info(
                    f"Connected {nation_name} to Sol network via {conn_type} link "
                    f"(Star {node1} to Star {node2}, distance {distance:.2f} LY)")

                nations_connected.add(nation_name)
            else:
                logger.warning(
                    f"Could not connect {nation_name} to Sol network within range limits")

        logger.info(
            f"Connected {len(nations_connected)} nations to Sol network")

        # After connecting nations to Sol, connect any remaining isolated components
        components = list(nx.connected_components(self.primary_network))
        logger.info(
            f"Network now has {len(components)} disconnected components")

        # Find the main component (should contain Sol)
        main_component = max(components, key=len)
        logger.info(
            f"Main network component contains {len(main_component)} systems")

        # Connect other components to the main component
        components_connected = 0
        for component in components:
            if component == main_component:
                continue

            best_connection = self._find_best_inter_component_connection(
                component1=component,
                component2=main_component,
                mega_projector_range=100.0
            )

            if best_connection:
                node1, node2, distance, conn_type = best_connection
                self.primary_network.add_edge(
                    node1, node2,
                    distance=distance,
                    type=conn_type,
                    description=f"Inter-component link: isolated cluster to main network"
                )

                # Update connection counts
                self.connection_counts[node1] += 1
                self.connection_counts[node2] += 1

                logger.info(
                    f"Connected isolated component ({len(component)} systems) to main network via {conn_type} link "
                    f"(Star {node1} to Star {node2}, distance {distance:.2f} LY)")

                components_connected += 1
            else:
                logger.warning(
                    f"Could not connect isolated component ({len(component)} systems) "
                    f"to main network within range limits")

        logger.info(
            f"Connected {components_connected} additional isolated components to main network")

    def _network_optimization_phase(self):
        """
        Optimize the network by adding strategic connections to reduce travel times.
        """
        logger.info("Starting network optimization phase")

        # Calculate diameter (maximum shortest path) of the largest component
        components = list(nx.connected_components(self.primary_network))
        if not components:
            logger.warning("No connected components found in network")
            return

        main_component = max(components, key=len)
        main_component_subgraph = self.primary_network.subgraph(main_component)

        try:
            diameter = nx.diameter(main_component_subgraph)
            logger.info(f"Current network diameter: {diameter} jumps")

            # Calculate the eccentricity for each node (maximum distance to any other node)
            eccentricity = nx.eccentricity(main_component_subgraph)

            # Find systems with high eccentricity (far from other systems)
            high_eccentricity = {node: ecc for node, ecc in eccentricity.items()
                                 if ecc > diameter * 0.8 and
                                 self.connection_counts[
                                     node] < self.max_connections.get(node,
                                                                      self.max_connections_per_system)}

            logger.info(
                f"Identified {len(high_eccentricity)} systems with high eccentricity")

            # Find strategic bypass connections to reduce diameter
            bypass_connections = 0
            for node1, ecc1 in high_eccentricity.items():
                for node2, ecc2 in high_eccentricity.items():
                    if node1 >= node2:  # Avoid duplicates
                        continue

                    # Only connect systems that are far apart in the network but physically close
                    try:
                        network_distance = nx.shortest_path_length(
                            main_component_subgraph, node1, node2)
                    except:
                        continue

                    if network_distance < 4:  # Already reasonably close in the network
                        continue

                    # Calculate physical distance
                    coords1 = self.primary_network.nodes[node1]['coords']
                    coords2 = self.primary_network.nodes[node2]['coords']
                    physical_distance = math.sqrt(
                        sum((c1 - c2) ** 2 for c1, c2 in zip(coords1, coords2)))

                    # If physical distance is within range but network distance is long, add a bypass
                    if physical_distance <= self.stellar_projector_range:
                        conn_type = 'stellar_projector'
                        eligible = True
                    elif physical_distance <= 100.0:  # Mega projector range
                        conn_type = 'mega_projector'
                        # Only use mega projectors for very important systems
                        eligible = (self.star_importance.get(node1, 0) > 7 and
                                    self.star_importance.get(node2, 0) > 7)
                    else:
                        eligible = False

                    if eligible and self.connection_counts[
                        node1] < self.max_connections.get(node1,
                                                          self.max_connections_per_system) and \
                            self.connection_counts[
                                node2] < self.max_connections.get(node2,
                                                                  self.max_connections_per_system):
                        self.primary_network.add_edge(
                            node1, node2,
                            distance=physical_distance,
                            type=conn_type,
                            description=f"Strategic bypass: reduces network diameter"
                        )

                        # Update connection counts
                        self.connection_counts[node1] += 1
                        self.connection_counts[node2] += 1

                        logger.info(
                            f"Added strategic bypass connection between Star {node1} and Star {node2} "
                            f"({conn_type}, {physical_distance:.2f} LY)")

                        bypass_connections += 1

                        # Limit the number of bypasses to add
                        if bypass_connections >= 5:
                            break

                if bypass_connections >= 5:
                    break

            logger.info(
                f"Added {bypass_connections} strategic bypass connections")

        except nx.NetworkXError as e:
            logger.warning(f"Could not calculate network diameter: {e}")

    def _validation_and_metrics_phase(self):
        """
        Validate the network and calculate accessibility metrics.
        """
        logger.info("Starting validation and metrics phase")

        # Check connectivity of the network
        components = list(nx.connected_components(self.primary_network))
        logger.info(f"Final network has {len(components)} components")

        main_component = max(components, key=len)
        logger.info(f"Main component contains {len(main_component)} systems "
                    f"({len(main_component) / len(self.primary_network.nodes()) * 100:.1f}% of total)")

        # Calculate which systems are connected to the primary network
        projector_count = 0
        for node in self.primary_network.nodes():
            # Check if connected to the primary network
            has_projector = self.primary_network.nodes[node].get(
                'has_projector', False)
            if has_projector:
                projector_count += 1

            primary_connected = has_projector
            self.primary_network.nodes[node][
                'primary_connected'] = primary_connected

        logger.info(f"Network has {projector_count} systems with projectors "
                    f"({projector_count / len(self.primary_network.nodes()) * 100:.1f}% of total)")

        # Calculate path statistics from Sol
        try:
            # Calculate shortest path lengths from Sol
            path_lengths = nx.shortest_path_length(self.primary_network,
                                                   source=self.sol_node)

            max_path = max(path_lengths.values())
            avg_path = sum(path_lengths.values()) / len(path_lengths)

            logger.info(
                f"From Sol: max path length = {max_path} jumps, avg path length = {avg_path:.2f} jumps")

            # Calculate accessibility scores for all nodes
            for node in self.primary_network.nodes():
                # Get shortest path length (defaults to infinity if unreachable)
                path_length = path_lengths.get(node, float('inf'))

                # Store shortest path length
                self.primary_network.nodes[node][
                    'shortest_path_from_sol'] = path_length

                # Calculate accessibility score:
                # - Primary network nodes are most accessible (0-10)
                # - Nodes close to primary network have medium accessibility (10-50)
                # - Isolated nodes have poor accessibility (50+)
                accessibility = path_length * 10

                # Primary network nodes get a bonus
                if self.primary_network.nodes[node].get('has_projector', False):
                    accessibility *= 0.5

                # Store accessibility score
                self.primary_network.nodes[node][
                    'accessibility_score'] = accessibility

        except nx.NetworkXNoPath:
            logger.warning(
                "No path exists from Sol to some nodes, network is disconnected")

            # Set default values for disconnected nodes
            for node in self.primary_network.nodes():
                if node not in path_lengths:
                    self.primary_network.nodes[node][
                        'shortest_path_from_sol'] = float('inf')
                    self.primary_network.nodes[node][
                        'accessibility_score'] = 100

        # Calculate hub score (betweenness centrality) for major nodes
        try:
            projector_nodes = [node for node, data in
                               self.primary_network.nodes(data=True)
                               if data.get('has_projector', False)]

            projector_subgraph = self.primary_network.subgraph(projector_nodes)
            betweenness = nx.betweenness_centrality(projector_subgraph)

            # Scale the values to 0-10 range
            if betweenness:
                max_betweenness = max(
                    betweenness.values()) if betweenness.values() else 1
                for node, score in betweenness.items():
                    # Scale to 0-10 range
                    hub_score = (score / max_betweenness) * 10
                    self.primary_network.nodes[node]['hub_score'] = hub_score

            # Set default hub score for non-projector nodes
            for node in self.primary_network.nodes():
                if 'hub_score' not in self.primary_network.nodes[node]:
                    self.primary_network.nodes[node]['hub_score'] = 0

        except Exception as e:
            logger.warning(f"Failed to calculate hub scores: {e}")
            # Set default hub score for all nodes
            for node in self.primary_network.nodes():
                self.primary_network.nodes[node]['hub_score'] = 0

        # Report connection statistics
        edge_count = self.primary_network.number_of_edges()
        projector_count = sum(
            1 for _, data in self.primary_network.nodes(data=True)
            if data.get('has_projector', False))

        avg_connections = 2 * edge_count / projector_count if projector_count > 0 else 0

        logger.info(
            f"Network has {edge_count} connections with an average of {avg_connections:.2f} connections per projector system")

        # Count connection types
        connection_types = {}
        for _, _, data in self.primary_network.edges(data=True):
            conn_type = data.get('type', 'unknown')
            connection_types[conn_type] = connection_types.get(conn_type, 0) + 1

        for conn_type, count in connection_types.items():
            logger.info(f"Connection type '{conn_type}': {count} connections")

    def _generate_secondary_network(self):
        """
        Generate the secondary network of ship-based projector routes.
        """
        logger.info("Starting secondary network generation")

        # Start with a copy of all nodes from primary network
        self.secondary_network.add_nodes_from(
            self.primary_network.nodes(data=True))

        # Connect all stars within ship projector range
        star_ids = list(self.secondary_network.nodes())
        connections_added = 0

        for i, node1 in enumerate(star_ids):
            coords1 = self.secondary_network.nodes[node1]['coords']
            for node2 in star_ids[i + 1:]:
                coords2 = self.secondary_network.nodes[node2]['coords']

                # Calculate distance
                distance = math.sqrt(
                    sum((c1 - c2) ** 2 for c1, c2 in zip(coords1, coords2)))

                # Connect if within ship projector range
                if distance <= self.ship_projector_range:
                    self.secondary_network.add_edge(node1, node2,
                                                    distance=distance,
                                                    type='ship_projector')
                    connections_added += 1

        logger.info(
            f"Added {connections_added} ship projector connections to secondary network")

        # Include all primary network connections in the secondary network
        primary_connections_added = 0
        for node1, node2, data in self.primary_network.edges(data=True):
            if not self.secondary_network.has_edge(node1, node2):
                self.secondary_network.add_edge(node1, node2, **data)
                primary_connections_added += 1

        logger.info(
            f"Added {primary_connections_added} primary network connections to secondary network")

        # Copy node attributes from primary network
        for node, data in self.primary_network.nodes(data=True):
            for key, value in data.items():
                self.secondary_network.nodes[node][key] = value

        logger.info(
            f"Secondary network complete with {self.secondary_network.number_of_edges()} total connections")

    def _find_sol_node(self):
        """Find the star closest to Sol (origin point)."""
        sol_node = None
        min_dist = float('inf')
        for node, data in self.primary_network.nodes(data=True):
            coords = data['coords']
            dist = math.sqrt(sum(c ** 2 for c in coords))
            if dist < min_dist:
                min_dist = dist
                sol_node = node
        return sol_node

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
            distance = math.sqrt(
                (star.x - nation.origin['x']) ** 2 +
                (star.y - nation.origin['y']) ** 2 +
                (star.z - nation.origin['z']) ** 2
            )

            if distance < min_distance:
                min_distance = distance
                closest_star = star

        return closest_star

    def _calculate_system_importance(self, star):
        """Calculate a star system's strategic importance."""
        importance = 0

        # Nation capitals get priority
        if star.nation and self._is_capital(star, star.nation):
            importance += 10
            logger.debug(f"Star {star.id} importance +10 (nation capital)")

        # Systems closer to Sol get priority
        dist_to_sol = math.sqrt(star.x ** 2 + star.y ** 2 + star.z ** 2)
        sol_factor = max(0, 500 - dist_to_sol) / 50
        importance += sol_factor
        logger.debug(
            f"Star {star.id} importance +{sol_factor:.2f} (proximity to Sol)")

        # G-Type stars (like Sol) get priority for stellar projectors
        if star.spectral_class == "G-Type":
            importance += 3
            logger.debug(f"Star {star.id} importance +3 (G-Type star)")
        elif star.spectral_class == "K-Type":
            importance += 2
            logger.debug(f"Star {star.id} importance +2 (K-Type star)")
        elif star.spectral_class == "M-Type":
            importance += 1
            logger.debug(f"Star {star.id} importance +1 (M-Type star)")

        # Systems with habitable planets get higher importance
        habitable_count = sum(
            1 for body in star.planetary_system.celestial_bodies
            if body.body_type == "Planet" and hasattr(body,
                                                      'habitable') and body.habitable
        )
        hab_factor = habitable_count * 3
        importance += hab_factor
        if hab_factor > 0:
            logger.debug(
                f"Star {star.id} importance +{hab_factor} ({habitable_count} habitable planets)")

        # Systems with more planets and asteroid belts get additional importance
        planet_count = sum(
            1 for body in star.planetary_system.celestial_bodies
            if body.body_type == "Planet" and not (
                        hasattr(body, 'habitable') and body.habitable)
        )
        belt_count = sum(
            1 for body in star.planetary_system.celestial_bodies
            if body.body_type == "Asteroid Belt"
        )

        resource_factor = planet_count * 0.5 + belt_count * 0.75
        importance += resource_factor
        if resource_factor > 0:
            logger.debug(
                f"Star {star.id} importance +{resource_factor:.2f} (resources: {planet_count} planets, {belt_count} belts)")

        return importance

    def _is_capital(self, star, nation):
        """Check if a star is the capital of its nation."""
        if not nation:
            return False

        # Calculate distance to nation origin
        distance = math.sqrt(
            (star.x - nation.origin['x']) ** 2 +
            (star.y - nation.origin['y']) ** 2 +
            (star.z - nation.origin['z']) ** 2
        )

        # Consider it the capital if it's very close to the origin (within 5 LY)
        return distance < 5

    def _connect_systems_mst(self, nodes, start_node=None, tag=""):
        """
        Connect a set of systems using a Minimum Spanning Tree approach.

        Args:
            nodes: List of node IDs to connect
            start_node: Optional starting node for the MST
            tag: Tag for logging purposes
        """
        if not nodes:
            logger.warning(f"No nodes provided for MST connection ({tag})")
            return

        if len(nodes) == 1:
            logger.info(f"Only one node provided for MST connection ({tag})")
            return

        # Create a complete graph of all possible connections
        complete_graph = nx.Graph()

        # Add all nodes
        for node in nodes:
            complete_graph.add_node(node)

        # Add all possible edges with weights based on distance
        for i, node1 in enumerate(nodes):
            coords1 = self.primary_network.nodes[node1]['coords']
            for node2 in nodes[i + 1:]:
                coords2 = self.primary_network.nodes[node2]['coords']

                # Calculate distance
                distance = math.sqrt(
                    sum((c1 - c2) ** 2 for c1, c2 in zip(coords1, coords2)))

                # Only add edge if within range
                if distance <= self.stellar_projector_range:
                    complete_graph.add_edge(node1, node2, weight=distance)
                elif distance <= 100.0:  # Mega projector range
                    # Mega projectors have a penalty factor to discourage their use except when necessary
                    complete_graph.add_edge(node1, node2, weight=distance * 3)

        # Find MST edges
        try:
            if start_node and start_node in complete_graph:
                # For a specific start node, use Prim's algorithm (via networkx's minimum_spanning_edges)
                mst_edges = list(
                    nx.minimum_spanning_edges(complete_graph, weight='weight',
                                              algorithm='prim', data=True))
            else:
                # Otherwise use Kruskal's algorithm
                mst_edges = list(
                    nx.minimum_spanning_edges(complete_graph, weight='weight',
                                              algorithm='kruskal', data=True))

            edges_added = 0
            for u, v, data in mst_edges:
                # Check connection limits
                if (self.connection_counts[u] >= self.max_connections.get(u,
                                                                          self.max_connections_per_system) or
                        self.connection_counts[v] >= self.max_connections.get(v,
                                                                              self.max_connections_per_system)):
                    logger.debug(
                        f"Skipping connection {u}-{v} due to connection limits")
                    continue

                # Determine connection type based on distance
                distance = data['weight']
                if distance <= self.stellar_projector_range:
                    conn_type = 'stellar_projector'
                else:
                    conn_type = 'mega_projector'

                    # Only use mega projectors if both nodes are important enough
                    if (self.star_importance.get(u, 0) < 7 or
                            self.star_importance.get(v, 0) < 7):
                        logger.debug(
                            f"Skipping mega projector {u}-{v} due to insufficient importance")
                        continue

                # Add the edge to the primary network
                self.primary_network.add_edge(
                    u, v,
                    distance=distance / (
                        3 if conn_type == 'mega_projector' else 1),
                    # Remove the penalty factor
                    type=conn_type,
                    description=f"MST connection ({tag})"
                )

                # Update connection counts
                self.connection_counts[u] += 1
                self.connection_counts[v] += 1

                edges_added += 1
                logger.debug(
                    f"Added MST edge {u}-{v} ({conn_type}, {distance:.2f} LY)")

            logger.info(f"Added {edges_added} connections for MST ({tag})")

        except Exception as e:
            logger.warning(f"Error creating MST ({tag}): {e}")

    def _connect_systems_to_network(self, nodes, existing_network, tag=""):
        """
        Connect systems to an existing network by finding shortest distances.

        Args:
            nodes: List of node IDs to connect to the network
            existing_network: List of node IDs already in the network
            tag: Tag for logging purposes
        """
        if not nodes or not existing_network:
            logger.warning(
                f"Missing nodes or existing network for connection ({tag})")
            return

        connections_made = 0
        for node in nodes:
            # Find closest connection to existing network
            best_distance = float('inf')
            best_connection = None

            for existing_node in existing_network:
                # Skip self-connection
                if node == existing_node:
                    continue

                # Check connection limits
                if (self.connection_counts[node] >= self.max_connections.get(
                        node, self.max_connections_per_system) or
                        self.connection_counts[
                            existing_node] >= self.max_connections.get(
                            existing_node, self.max_connections_per_system)):
                    continue

                # Calculate distance
                coords1 = self.primary_network.nodes[node]['coords']
                coords2 = self.primary_network.nodes[existing_node]['coords']
                distance = math.sqrt(
                    sum((c1 - c2) ** 2 for c1, c2 in zip(coords1, coords2)))

                # Check if within stellar projector range
                if distance <= self.stellar_projector_range and distance < best_distance:
                    best_distance = distance
                    best_connection = (node, existing_node, distance,
                                       'stellar_projector')
                # Check if within mega projector range, but only for important systems
                elif (distance <= 100.0 and distance < best_distance and
                      self.star_importance.get(node, 0) > 7 and
                      self.star_importance.get(existing_node, 0) > 7):
                    best_distance = distance
                    best_connection = (node, existing_node, distance,
                                       'mega_projector')

            # If we found a connection, add it
            if best_connection:
                u, v, distance, conn_type = best_connection

                # Add the edge to the primary network
                self.primary_network.add_edge(
                    u, v,
                    distance=distance,
                    type=conn_type,
                    description=f"Network connection ({tag})"
                )

                # Update connection counts
                self.connection_counts[u] += 1
                self.connection_counts[v] += 1

                # Add this node to the existing network for future connections
                existing_network.append(node)

                connections_made += 1
                logger.debug(
                    f"Connected {node} to existing network via {v} ({conn_type}, {distance:.2f} LY)")

        logger.info(
            f"Made {connections_made} connections to existing network ({tag})")

    def _find_best_inter_component_connection(self, component1, component2,
                                              mega_projector_range=100.0):
        """
        Find the best connection between two components.

        Args:
            component1: List of node IDs in the first component
            component2: List of node IDs in the second component
            mega_projector_range: Maximum distance for mega projectors

        Returns:
            Tuple (node1, node2, distance, connection_type) or None if no connection possible
        """
        best_connection = None
        best_distance = float('inf')

        for node1 in component1:
            # Skip if this node already has max connections
            if self.connection_counts[node1] >= self.max_connections.get(node1,
                                                                         self.max_connections_per_system):
                continue

            # Get node1 coordinates
            coords1 = self.primary_network.nodes[node1]['coords']
            importance1 = self.star_importance.get(node1, 0)

            for node2 in component2:
                # Skip if this node already has max connections
                if self.connection_counts[node2] >= self.max_connections.get(
                        node2, self.max_connections_per_system):
                    continue

                # Get node2 coordinates
                coords2 = self.primary_network.nodes[node2]['coords']
                importance2 = self.star_importance.get(node2, 0)

                # Calculate distance
                distance = math.sqrt(
                    sum((c1 - c2) ** 2 for c1, c2 in zip(coords1, coords2)))

                # Check if within stellar projector range
                if distance <= self.stellar_projector_range:
                    conn_type = 'stellar_projector'
                    weighted_distance = distance  # No penalty for standard projectors
                    eligible = True
                # Check if within mega projector range, but only for important systems
                elif distance <= mega_projector_range:
                    conn_type = 'mega_projector'
                    # Importance requirement for mega projectors
                    eligible = (importance1 > 6 and importance2 > 6)
                    # Penalty factor for mega projectors to discourage use unless necessary
                    weighted_distance = distance * 1.5
                else:
                    eligible = False
                    weighted_distance = float('inf')

                # If eligible and better than current best, update best connection
                if eligible and weighted_distance < best_distance:
                    best_distance = weighted_distance
                    best_connection = (node1, node2, distance, conn_type)

        return best_connection

    def get_star_accessibility(self, star_id):
        """
        Get the accessibility metrics for a specific star.

        Args:
            star_id: The ID of the star

        Returns:
            dict: Dictionary of accessibility metrics
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

    def get_network_stats(self):
        """
        Get statistics about the network.

        Returns:
            dict: Dictionary of network statistics
        """
        primary_projector_count = sum(
            1 for _, data in self.primary_network.nodes(data=True)
            if data.get('has_projector', False))

        component_sizes = [len(c) for c in
                           nx.connected_components(self.primary_network)]

        return {
            'total_stars': len(self.primary_network.nodes()),
            'projector_count': primary_projector_count,
            'projector_density': primary_projector_count / len(
                self.primary_network.nodes()),
            'primary_edge_count': self.primary_network.number_of_edges(),
            'secondary_edge_count': self.secondary_network.number_of_edges(),
            'average_node_degree_primary': sum(
                dict(self.primary_network.degree()).values()) / len(
                self.primary_network.nodes()),
            'average_node_degree_secondary': sum(
                dict(self.secondary_network.degree()).values()) / len(
                self.secondary_network.nodes()),
            'components': len(component_sizes),
            'largest_component_size': max(
                component_sizes) if component_sizes else 0,
            'connectivity': max(component_sizes) / len(
                self.primary_network.nodes()) if component_sizes else 0
        }