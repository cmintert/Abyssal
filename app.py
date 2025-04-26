import dash
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output, State

import config
from abyssal_map import Starmap, PlotGenerator, StarSystemFilter

"""
A Dash web application for visualizing an interactive 3D star map of the Abyssal universe.

This application allows users to:
- View a 3D visualization of star systems and their controlling nations
- Filter stars by nation
- Reset the view to default
- Interact with the map through zooming, rotating and panning

The starmap is generated once at startup to ensure consistency during the session.
"""

# Initialize the Dash app
app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

# Set the seed for reproducibility
np.random.seed(config.SEED)

# Nation data
NATIONS = config.DEFAULT_NATIONS
NATION_COLORS = config.DEFAULT_NATION_COLORS

# Create the starmap (do this once at startup)
starmap = Starmap()
starmap.generate_star_systems(
    number_of_stars=config.DEFAULT_NUM_STARS,
)
starmap.generate_nations(
    name_set=config.DEFAULT_NATIONS,  # Skip "All Nations"
    nation_colour_set=config.DEFAULT_NATION_COLORS,  # Skip None
    origin_set=config.DEFAULT_NATION_ORIGINS,
    expansion_rate_set=config.DEFAULT_EXPANSION_RATES,
)
starmap.assign_stars_to_nations()

starmap.write_all_to_json()

plot_generator = PlotGenerator(starmap)

# Define the app layout
app.layout = html.Div(
    [
        html.H1("Abyssal Star Map",
                style={"textAlign": "center", "color": "white"}),
        html.Div(
            [
                html.Div(
                    [
                        # Nations filter section
                        html.Div(
                            [
                                html.Label(
                                    "Filter by Nation:",
                                    style={"color": "white",
                                           "fontWeight": "bold",
                                           "marginBottom": "5px"},
                                ),
                                dcc.Checklist(
                                    id="nation-filter",
                                    options=[
                                        {"label": nation, "value": nation}
                                        for nation in NATIONS
                                    ],
                                    value=[],
                                    # Default to no nations selected (show all)
                                    style={"color": "white"},
                                    labelStyle={"display": "block",
                                                "marginBottom": "3px"},
                                ),
                            ],
                            style={"marginRight": "20px", "minWidth": "200px"},
                        ),

                        # Star Type filter section
                        html.Div(
                            [
                                html.Label(
                                    "Filter by Star Type:",
                                    style={"color": "white",
                                           "fontWeight": "bold",
                                           "marginBottom": "5px"},
                                ),
                                dcc.Checklist(
                                    id="star-type-filter",
                                    options=[
                                        {"label": "G-Type", "value": "G-Type"},
                                        {"label": "K-Type", "value": "K-Type"},
                                        {"label": "M-Type", "value": "M-Type"},
                                    ],
                                    value=[],
                                    # Default to no types selected (show all)
                                    style={"color": "white"},
                                    labelStyle={"display": "block",
                                                "marginBottom": "3px"},
                                ),
                            ],
                            style={"marginRight": "20px", "minWidth": "150px"},
                        ),

                        # Habitable planet filter
                        html.Div(
                            [
                                html.Label(
                                    "Other Filters:",
                                    style={"color": "white",
                                           "fontWeight": "bold",
                                           "marginBottom": "5px"},
                                ),
                                dcc.Checklist(
                                    id="habitable-filter",
                                    options=[{"label": "Habitable Planets Only",
                                              "value": "yes"}],
                                    value=[],
                                    style={"color": "white"},
                                    labelStyle={"display": "block",
                                                "marginBottom": "3px"},
                                ),
                            ],
                            style={"minWidth": "200px"},
                        ),

                        # Reset button
                        html.Button(
                            "Reset Filters",
                            id="reset-button",
                            style={"marginLeft": "20px", "height": "40px"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "flex-start",
                        "justifyContent": "center",
                        "padding": "15px",
                        "flexWrap": "wrap",
                        "backgroundColor": "rgba(30, 30, 30, 0.7)",
                        "borderRadius": "10px",
                        "margin": "10px",
                    },
                ),
                dcc.Graph(
                    id="starmap-3d",
                    figure=plot_generator.plot(html=False, return_fig=True),
                    style={"height": "80vh"},
                ),
            ]
        ),
        html.Div(
            [
                html.P(
                    "Interactive Star Map for Abyssal Universe",
                    style={"textAlign": "center", "color": "white"},
                ),
                html.Div(
                    id='camera-position',
                    style={"textAlign": "center", "color": "white",
                           "fontSize": "12px"}
                )
            ]
        ),
    ],
    style={"backgroundColor": "black", "color": "white", "height": "100vh"},
)


@app.callback(
    Output("starmap-3d", "figure"),
    [
        Input("nation-filter", "value"),
        Input("star-type-filter", "value"),
        Input("habitable-filter", "value"),
        Input("reset-button", "n_clicks"),
    ],
)
def update_figure(selected_nations, selected_star_types, habitable_only,
                  n_clicks):
    """
    Updates the 3D starmap visualization based on filter selections.

    Args:
        selected_nations (list): List of selected nation names
        selected_star_types (list): List of selected star types
        habitable_only (list): List containing 'yes' if the habitable filter is checked
        n_clicks (int): Number of times the reset button has been clicked

    Returns:
        dict: Updated figure configuration
    """
    # Create a filter object
    star_filter = StarSystemFilter()

    # Apply nation filter if any nations are selected
    if selected_nations:
        star_filter.add_filter(
            "nation",
            lambda star: star.nation and star.nation.name in selected_nations
        )

    # Apply star type filter if any types are selected
    if selected_star_types:
        star_filter.add_filter(
            "star_type",
            lambda star: star.spectral_class in selected_star_types
        )

    # Apply habitable planet filter if selected
    if "yes" in habitable_only:
        star_filter.add_filter(
            "habitable",
            lambda star: any(
                hasattr(planet, 'habitable') and planet.habitable
                for planet in star.planetary_system.celestial_bodies
                if planet.body_type == "Planet"
            )
        )

    # Generate the plot with applied filters
    return plot_generator.plot(html=False, return_fig=True,
                               star_filter=star_filter)


@app.callback(
    Output("camera-position", "children"),
    Input("starmap-3d", "relayoutData"),
    prevent_initial_call=True
)
def update_camera_position(relayoutData):
    """
    Updates the display of the current camera position.

    Args:
        relayoutData (dict): Contains camera position data from the 3D graph

    Returns:
        str: Formatted camera position information
    """
    if relayoutData is None or 'scene.camera' not in relayoutData:
        return "Camera: waiting for movement..."

    camera = relayoutData['scene.camera']
    return f"Camera - Center: ({camera['center']['x']:.2f}, {camera['center']['y']:.2f}, {camera['center']['z']:.2f}) | Eye: ({camera['eye']['x']:.2f}, {camera['eye']['y']:.2f}, {camera['eye']['z']:.2f})"

@app.callback(
    [
        Output("nation-filter", "value"),
        Output("star-type-filter", "value"),
        Output("habitable-filter", "value")
    ],
    Input("reset-button", "n_clicks"),
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    """Reset all filters to their default values"""
    return [], [], []


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
