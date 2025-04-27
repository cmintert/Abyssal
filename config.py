# config.py
"""Configuration settings for the Abyssal StellarMap."""

# General settings
SEED = 50

# Star system generation
DEFAULT_NUM_STARS = 521 #521
MAP_RADIUS = 500
STAR_LOCATION_NOISE = 100
STAR_LOCATION_STRETCH = (1, 1,.6)  # x, y, z

# Nation settings
DEFAULT_NATIONS = [
    "Haven",
    "New Frontier Alliance",
    "Sol Protectorate",
    "United Stellar Colonies",
    "Void Confederacy",
]
DEFAULT_NATION_COLORS = [
    (0.5, 0.5, 0.5),
    (0.2, 0.8, 0.2),
    (0.8, 0.2, 0.2),
    (0.2, 0.2, 0.8),
    (0.8, 0.8, 0.2),
]

NATION_OPACITY= 0.05
NATION_DESATURATION = 0.5

DEFAULT_NATION_ORIGINS = [
    {"x": -200, "y": 100, "z": -100},
    {"x": -50, "y": 100, "z": 90},
    {"x": 0, "y": 0, "z": 0},
    {"x": 50, "y": 50, "z": 20},
    {"x": 100, "y": 100, "z": -50},
]
DEFAULT_EXPANSION_RATES = [0.7, 0.8, 1, 1, 0.9]

# Visual settings
STAR_SIZE_RANGE = (8, 12)
PLANET_SIZE_RANGE = (7, 12)
ORBIT_LINE_WIDTH = 1
ORBIT_OPACITY = 0.7
ORBIT_COLOR = "grey"

# Plot settings
BACKGROUND_COLOR = "black"

# Transportation settings
STELLAR_PROJECTOR_RANGE = 100
SHIP_PROJECTOR_RANGE = 1
STELLAR_PROJECTOR_DENSITY = 0.2