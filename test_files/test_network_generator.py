# test_network_generator.py

import logging
from starmap import Starmap
from persistence import StarmapReader, StarmapWriter
import config

# Set up logging
logging.basicConfig(level=logging.INFO)


def test_network_generation():
    # Load an existing starmap or create a new one
    reader = StarmapReader()
    if reader.check_json_files_exist():
        print("Loading existing starmap...")
        starmap = reader.load_starmap()
    else:
        print("Creating new starmap...")
        starmap = Starmap()
        starmap.generate_mineral_maps()
        starmap.generate_star_systems(
            number_of_stars=100)  # Smaller for testing
        starmap.generate_nations(
            name_set=config.DEFAULT_NATIONS,
            nation_colour_set=config.DEFAULT_NATION_COLORS,
            origin_set=config.DEFAULT_NATION_ORIGINS,
            expansion_rate_set=config.DEFAULT_EXPANSION_RATES,
        )
        starmap.assign_stars_to_nations()

    # Generate the transport network using the new generator
    print("Generating transport network with new generator...")
    transport_network = starmap.generate_transport_network(
        stellar_projector_range=config.STELLAR_PROJECTOR_RANGE,
        ship_projector_range=config.SHIP_PROJECTOR_RANGE,
        stellar_projector_density=config.STELLAR_PROJECTOR_DENSITY
    )

    # Print some network statistics
    print("\nNetwork Statistics:")
    projector_count = sum(
        1 for _, data in transport_network.primary_network.nodes(data=True)
        if data.get('has_projector', False))
    print(f"Total stars: {len(starmap.stars)}")
    print(
        f"Systems with projectors: {projector_count} ({projector_count / len(starmap.stars) * 100:.1f}%)")
    print(
        f"Primary network connections: {transport_network.primary_network.number_of_edges()}")
    print(
        f"Secondary network connections: {transport_network.secondary_network.number_of_edges()}")

    # Save the updated starmap
    print("\nSaving starmap with new network...")
    writer = StarmapWriter()
    writer.save_starmap(starmap)

    print("Test completed successfully")


if __name__ == "__main__":
    test_network_generation()