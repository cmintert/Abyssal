import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

import config
from Abyssal_map import Starmap, PlotGenerator

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
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

# Nation data
ALL_NATIONS_ENTRY = "All Nations"
NATIONS = [ALL_NATIONS_ENTRY] + config.DEFAULT_NATIONS

NATION_COLORS = [None] + config.DEFAULT_NATION_COLORS

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
plot_generator = PlotGenerator(starmap)

# Define the app layout
app.layout = html.Div(
    [
        html.H1("Abyssal Star Map", style={"textAlign": "center", "color": "white"}),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Filter by Nation:",
                            style={"color": "white", "marginRight": "10px"},
                        ),
                        dcc.Dropdown(
                            id="nation-filter",
                            options=[
                                {"label": nation, "value": i}
                                for i, nation in enumerate(NATIONS)
                            ],
                            value=0,  # Default to "All Nations"
                            style={
                                "width": "250px",
                                "backgroundColor": "black",
                                "color": "white",
                            },
                        ),
                        html.Button(
                            "Reset View",
                            id="reset-button",
                            style={"marginLeft": "10px"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "padding": "10px",
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
                )
            ]
        ),
    ],
    style={"backgroundColor": "black", "color": "white", "height": "100vh"},
)


@app.callback(
    Output("starmap-3d", "figure"),
    [Input("nation-filter", "value"), Input("reset-button", "n_clicks")],
    [State("starmap-3d", "figure")],
)
def update_figure(selected_nation_idx, n_clicks, current_fig):
    """
    Updates the 3D starmap visualization based on user input.

    This callback function filters the displayed stars based on the selected nation
    and handles view reset requests. It's crucial for providing an interactive
    experience while maintaining performance by updating only necessary elements.

    Args:
        selected_nation_idx (int): Index of the selected nation in the NATIONS list
        n_clicks (int): Number of times the reset button has been clicked
        current_fig (dict): Current figure state of the 3D starmap

    Returns:
        dict: Updated figure configuration for the 3D starmap

    Why:
        We need this callback to provide real-time filtering of stars by nation,
        which helps users focus on specific areas of interest in the map while
        maintaining the overall context of the universe.
    """
    # Create a new figure based on the selected nation
    if selected_nation_idx == 0:  # "All Nations"
        return plot_generator.plot(html=False, return_fig=True)

    # Filter stars by nation
    filtered_stars = [
        star
        for star in starmap.stars
        if star.nation.name == NATIONS[selected_nation_idx]
    ]

    # Create a custom figure with only the selected nation's stars
    fig = current_fig

    # Update the stars trace
    for trace in fig["data"]:
        if trace["name"] == "Stars":
            # Filter x, y, z coordinates for the selected nation
            x_coords = [star.x for star in filtered_stars]
            y_coords = [star.y for star in filtered_stars]
            z_coords = [star.z for star in filtered_stars]

            trace["x"] = x_coords
            trace["y"] = y_coords
            trace["z"] = z_coords

    return fig


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
