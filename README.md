# Abyssal StellarMap Documentation

## Overview
Abyssal StellarMap is a sophisticated space simulation and visualization system that generates realistic 3D star maps complete with planetary systems, political territories, and resource distributions. It combines procedural generation with astronomical principles to create an immersive fictional universe for exploration.

## Installation

### Prerequisites
- Python 3.x
- Required packages: numpy, plotly, dash, openai, pytest

### Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py` for the web interface

## System Architecture

### Core Modules

#### abyssal_map.py
The primary module containing the `Starmap`, `PlotGenerator`, and `StarSystemFilter` classes. This module orchestrates the generation and visualization of the entire star system.

- **Starmap Class**
  - Manages stars, nations, and mineral distributions
  - Handles the generation of all space objects
  - Exports data to JSON files for persistence
  - Methods for managing star positions and properties

- **PlotGenerator Class**
  - Creates 3D visualizations using Plotly
  - Generates traces for stars, planets, asteroid belts, and nations
  - Configures the layout and appearance of the visualization

- **StarSystemFilter Class**
  - Provides filtering functionality for the star map
  - Allows filtering by nation, star type, and planet habitability

#### map_components.py
Contains the fundamental celestial object classes and their behaviors.

- **Nation**
  - Represents political entities in space
  - Properties include name, origin point, expansion rate, and color
  - Controls territorial influence over stars

- **Star**
  - Properties include position, spectral class, luminosity, and mass
  - Hosts a planetary system
  - Methods for coordinate conversion and habitable zone calculation

- **SmallBody**
  - Base class for planets and asteroid belts
  - Handles common properties like orbit and name

- **Planet**
  - Simulates planets with realistic properties
  - Includes atmosphere, composition, surface temperature, and habitability
  - Generates physical characteristics based on orbit and star properties

- **AsteroidBelt**
  - Represents belts of asteroids around stars
  - Contains distributions of minerals and resources
  - Varies in density and composition

- **Planetary_System**
  - Manages orbits and celestial bodies around a star
  - Generates plausible orbital arrangements
  - Places planets and asteroid belts in orbits

- **MineralMap**
  - Manages the spatial distribution of minerals across the universe
  - Creates "zones" where certain minerals are more abundant
  - Influences resource availability in asteroid belts

#### Utility.py
Helper classes and functions for procedural generation and data processing.

- **StarNames**
  - Generates plausible and varied star names
  - Combines prefixes, middles, and suffixes with constellations and designations

- **PlanetNames**
  - Creates appropriate names for planets based on their star and orbital position
  - Uses Roman numerals for orbital designations

- **RareMinerals**
  - Defines minerals with properties like rarity and value
  - Provides descriptions and composition information

- **Helper Functions**
  - insert_linebreaks: Formats text for display with appropriate line breaks
  - scale_values_to_range: Normalizes values to a specified range

#### app.py
A Dash web application providing an interactive interface for the star map.

- Creates a responsive web interface
- Allows filtering of stars by nation, star type, and habitability
- Provides interactive 3D visualization with camera controls
- Supports reset functionality to clear filters

#### config.py
Contains configuration settings for the Abyssal StellarMap:

- Seed for reproducibility
- Default values for star systems, nations, and visual settings
- Configuration for colors, sizes, and appearances

## Using the System

### Web Application
The included Dash application provides an interactive web interface:

1. Run `python app.py`
2. Open a web browser to `http://127.0.0.1:8050/`
3. Use the filters to view specific nations, star types, or habitable planets
4. Explore the 3D star map interactively
5. Reset filters with the "Reset Filters" button

### Generating a New Universe
To generate a new star map programmatically:

```python
from abyssal_map import Starmap

# Create a new starmap
my_map = Starmap()

# Generate star systems
my_map.generate_star_systems(number_of_stars=500)

# Generate nations
my_map.generate_nations(
  name_set=["Empire", "Republic", "Federation"],
  nation_colour_set=[(0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.2, 0.8)],
  origin_set=[{"x": 0, "y": 0, "z": 0},
              {"x": 100, "y": 100, "z": 100},
              {"x": -100, "y": -100, "z": -100}],
  expansion_rate_set=[1.0, 0.8, 0.9]
)

# Assign stars to nations
my_map.assign_stars_to_nations()

# Visualize the map
my_map.plot()

# Export data to JSON files
my_map.write_all_to_json()
```

## Simulation Details

### Star Generation
Stars are generated with realistic properties:
- Position in 3D space (spherical coordinates converted to Cartesian)
- Spectral class (G, K, M types prioritized for habitable zones)
- Luminosity based on spectral class
- Mass calculated from luminosity and spectral class
- Location noise and stretching for realistic spatial distribution

### Planetary System Generation
For each star:
- A set of stable orbits is created
- One orbit may be placed in the habitable zone
- Planets and asteroid belts are placed in orbits
- Planet properties are based on orbit distance and star characteristics

### Planet Properties
Planets have detailed simulated properties:
- Mass and radius
- Composition (rocky, icy, gas giant)
- Surface temperature
- Atmosphere
- Presence of water
- Rotation period and axial tilt
- Density and gravity
- Habitability assessment

### Asteroid Belts
Asteroid belts include:
- Varying density (sparse, moderate, dense)
- Mineral compositions based on location in space
- Resource distributions influenced by mineral maps

### Nation Territories
Political entities with:
- Origin points in space
- Expansion rates determining influence
- Colors for visualization
- Stars assigned based on weighted distance

## Data Persistence
The system exports all data to JSON files for persistence and external analysis:
- star_data.json
- nation_data.json
- planetary_system_data.json
- planet_data.json
- asteroid_belt_data.json

## Visualization
The 3D visualization uses Plotly to create an interactive experience:
- Stars colored by luminosity and sized by mass
- Planets with orbits and properties
- Asteroid belts shown as particle scatters
- Nations highlighted by color
- Hover information for all objects
- Camera position tracking

## Expanding the System

### Adding New Minerals
To add new minerals, modify the `RareMinerals.properties` dictionary in Utility.py:

```python
"NewMineral": {
    "rarity": 0.1,  # 0-3 scale
    "value": 800,   # 0-1200 scale
    "description": "Description of the new mineral"
}
```

### Creating Custom Nations
Customize nations when initializing the starmap:

```python
my_map.generate_nations(
    name_set=["Your Custom Nation Names"],
    nation_colour_set=[(r, g, b) values],
    origin_set=[{"x": x, "y": y, "z": z} coordinates],
    expansion_rate_set=[rate values]
)
```

### Adjusting Generation Parameters
Various parameters can be adjusted in config.py:
- SEED for reproducibility
- DEFAULT_NUM_STARS for star count
- SPACE_BOUNDARY for size of universe
- DEFAULT_NATIONS, DEFAULT_NATION_COLORS, etc. for political entities
- Visual settings for star and planet sizes

## Technical Notes

- The simulation uses NumPy for efficient numerical operations
- 3D visualization is handled by Plotly's Scatter3d
- Coordinates are managed in both spherical and Cartesian formats
- Star names use a combination of prefixes, middles, and suffixes to create unique identifiers
- Planet physical properties are calculated using simplified approximations of real astronomical principles
- Testing is implemented with pytest