# test_persistence.py
from persistence import StarmapReader, StarmapWriter
from abyssal_map import Starmap
import config


def test_persistence():
    """Test saving and loading a starmap"""

    # Set up the writer
    writer = StarmapWriter()

    # Create a new starmap
    print("Generating new starmap...")
    starmap = Starmap()
    starmap.generate_mineral_maps()
    starmap.generate_star_systems(
        number_of_stars=10)  # Use a small number for testing
    starmap.generate_nations(
        name_set=config.DEFAULT_NATIONS[:2],  # Just use the first two nations
        nation_colour_set=config.DEFAULT_NATION_COLORS[:2],
        origin_set=config.DEFAULT_NATION_ORIGINS[:2],
        expansion_rate_set=config.DEFAULT_EXPANSION_RATES[:2],
    )
    starmap.assign_stars_to_nations()

    # Save the starmap
    print("Saving starmap to JSON...")
    writer.save_starmap(starmap)

    # Set up the reader
    reader = StarmapReader()

    # Load the starmap
    print("Loading starmap from JSON...")
    loaded_starmap = reader.load_starmap()

    # Verify that the loaded starmap has the same structure
    if loaded_starmap:
        print(
            f"Original starmap: {len(starmap.stars)} stars, {len(starmap.nations)} nations")
        print(
            f"Loaded starmap: {len(loaded_starmap.stars)} stars, {len(loaded_starmap.nations)} nations")

        # Check a few values
        original_star = starmap.stars[0]
        loaded_star = loaded_starmap.stars[0]

        print(
            f"Original first star: {original_star.name}, ID: {original_star.id}")
        print(f"Loaded first star: {loaded_star.name}, ID: {loaded_star.id}")

        if original_star.id == loaded_star.id and original_star.name == loaded_star.name:
            print("Basic star properties match!")
        else:
            print("Star properties don't match.")

        print("Persistence test completed.")
    else:
        print("Failed to load starmap from JSON.")


if __name__ == "__main__":
    test_persistence()