import dash
import os
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output, State
import json
from dash import no_update

import config
from starmap import Starmap, PlotGenerator, StarSystemFilter
from persistence import StarmapReader, StarmapWriter


def initialize_starmap():
    """Initialize the starmap, either from JSON or by generating a new one"""
    reader = StarmapReader()
    writer = StarmapWriter()

    if reader.check_json_files_exist():
        # Try to load existing data
        starmap = reader.load_starmap()
        if starmap:
            print("Successfully loaded existing universe from JSON")

            # Generate transport network and population model
            try:
                print("Generating transport network...")
                starmap.generate_transport_network()

                print("Generating population model...")
                starmap.generate_colony_populations()

                print("Population summary:", starmap.get_population_summary())
            except Exception as e:
                print(f"Error generating transport network or population: {e}")

            return starmap
        else:
            print("Failed to load universe from JSON, generating new universe")
    else:
        print("JSON files not found, generating new universe")

    # Generate new universe
    starmap = Starmap()
    starmap.generate_mineral_maps()
    starmap.generate_star_systems(
        number_of_stars=config.DEFAULT_NUM_STARS,
        map_radius=config.MAP_RADIUS,
    )
    starmap.generate_nations(
        name_set=config.DEFAULT_NATIONS,
        nation_colour_set=config.DEFAULT_NATION_COLORS,
        origin_set=config.DEFAULT_NATION_ORIGINS,
        expansion_rate_set=config.DEFAULT_EXPANSION_RATES,
    )
    starmap.assign_stars_to_nations()

    # Generate transport network and population model
    try:
        print("Generating transport network...")
        starmap.generate_transport_network()

        print("Generating population model...")
        starmap.generate_colony_populations()

        print("Population summary:", starmap.get_population_summary())
    except Exception as e:
        print(f"Error generating transport network or population: {e}")

    # Save the newly generated universe
    writer.save_starmap(starmap)
    starmap.write_population_data_to_json()

    return starmap


"""
A Dash web application for visualizing an interactive 3D star map of the Abyssal universe.

This application allows users to:
- View a 3D visualization of star systems and their controlling nations
- Filter stars by nation
- Reset the view to default
- Interact with the map through zooming, rotating and panning
- Click on objects to edit their additional information
- Save changes to JSON files

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

starmap = initialize_starmap()

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
                                        {"label": "O-Type", "value": "O-Type"},
                                        {"label": "B-Type", "value": "B-Type"},
                                        {"label": "A-Type", "value": "A-Type"},
                                        {"label": "F-Type", "value": "F-Type"},
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

                        html.Div(
                            [
                                html.Label(
                                    "Population Filters:",
                                    style={"color": "white", "fontWeight": "bold", "marginBottom": "5px"},
                                ),
                                dcc.Checklist(
                                    id="population-filter",
                                    options=[
                                        {"label": "Primary Hubs Only", "value": "primary_hub"},
                                        {"label": "Million+ Colonies Only", "value": "large_colonies"}
                                    ],
                                    value=[],
                                    style={"color": "white"},
                                    labelStyle={"display": "block", "marginBottom": "3px"},
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
                        html.Div(
                            [
                                html.Label(
                                    "Time Advancement:",
                                    style={"color": "white", "fontWeight": "bold", "marginBottom": "5px"},
                                ),
                                dcc.Input(
                                    id="time-advance-input",
                                    type="number",
                                    min=1,
                                    max=100,
                                    value=10,
                                    style={"width": "60px", "marginRight": "10px"}
                                ),
                                html.Button(
                                    "Advance Years",
                                    id="time-advance-button",
                                    style={"height": "40px"},
                                ),
                                html.Div(
                                    id="time-advance-info",
                                    style={"color": "white", "marginTop": "5px", "fontSize": "14px"}
                                )
                            ],
                            style={"marginLeft": "20px", "display": "flex", "alignItems": "center", "flexWrap": "wrap"},
                        ),

                        # Add a population statistics box to display population summary:
                        html.Div(
                            [
                                html.H3("Population Statistics", style={"color": "white", "textAlign": "center"}),
                                html.Pre(
                                    id="population-stats",
                                    style={
                                        "color": "white",
                                        "backgroundColor": "rgba(30, 30, 30, 0.7)",
                                        "padding": "15px",
                                        "borderRadius": "10px",
                                        "whiteSpace": "pre-wrap",
                                        "fontFamily": "monospace"
                                    }
                                )
                            ],
                            style={"margin": "20px 10px"}
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
                    config={"scrollZoom": True, "displayModeBar": True}
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
                ),
                # Debug output
                html.Pre(id='click-data-debug',
                         style={"color": "white", "fontSize": "10px",
                                "maxHeight": "150px",
                                "overflow": "auto",
                                "backgroundColor": "rgba(0,0,0,0.5)",
                                "padding": "10px", "display": "none"})
            ]
        ),

        # Modal for editing additional info
        html.Div(
            id='edit-modal',
            className='modal',
            style={
                'display': 'none',
                'position': 'fixed',
                'z-index': '9999',  # Increased z-index
                'left': '0',
                'top': '0',
                'width': '100%',
                'height': '100%',
                'overflow': 'auto',
                'backgroundColor': 'rgba(0,0,0,0.8)',  # Darker background
            },
            children=[
                html.Div(
                    className='modal-content',
                    style={
                        'backgroundColor': '#333',
                        'margin': '10% auto',
                        'padding': '20px',
                        'border': '1px solid #888',
                        'width': '80%',
                        'maxWidth': '600px',
                        'color': 'white',
                        'borderRadius': '10px',
                        'boxShadow': '0 4px 8px rgba(0,0,0,0.5)',
                    },
                    children=[
                        html.Div(id='modal-header', children=[
                            html.H3(id='modal-title', style={'color': 'white'}),
                            html.Span(
                                '×',
                                id='close-modal',
                                style={
                                    'color': 'white',
                                    'float': 'right',
                                    'fontSize': '28px',
                                    'fontWeight': 'bold',
                                    'cursor': 'pointer',
                                },
                            ),
                        ]),
                        html.Div(id='modal-body', children=[
                            html.P('Edit additional information:'),
                            dcc.Textarea(
                                id='info-textarea',
                                style={
                                    'width': '100%',
                                    'height': '200px',
                                    'padding': '12px',
                                    'backgroundColor': '#222',
                                    'color': 'white',
                                    'border': '1px solid #666',
                                    'borderRadius': '5px',
                                },
                            ),
                            html.P(id='object-type-display',
                                   style={'marginTop': '10px'}),
                            # Store the object ID and type
                            html.Div(id='object-id-store',
                                     style={'display': 'none'}),
                            html.Div(id='object-type-store',
                                     style={'display': 'none'}),
                        ]),
                        html.Div(id='modal-footer', children=[
                            html.Button(
                                'Save Changes',
                                id='save-changes-button',
                                style={
                                    'backgroundColor': '#4CAF50',
                                    'color': 'white',
                                    'padding': '10px 20px',
                                    'margin': '10px 0',
                                    'border': 'none',
                                    'borderRadius': '5px',
                                    'cursor': 'pointer',
                                },
                            ),
                            html.Div(id='save-status',
                                     style={'color': '#4CAF50',
                                            'marginTop': '10px'})
                        ]),
                    ],
                ),
            ],
        ),
    ],
    style={"backgroundColor": "black", "color": "white", "minHeight": "100vh"},
)


@app.callback(
    Output("starmap-3d", "figure"),
    [
        Input("nation-filter", "value"),
        Input("star-type-filter", "value"),
        Input("habitable-filter", "value"),
        Input("population-filter", "value"),
        Input("reset-button", "n_clicks"),
    ],
)
def update_figure(selected_nations, selected_star_types, habitable_only,
                  population_filter, n_clicks):
    """
    Updates the 3D starmap visualization based on filter selections.
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

    # Apply population filters if selected
    if population_filter:
        if "primary_hub" in population_filter:
            star_filter.add_filter(
                "primary_hub",
                lambda star: any(
                    hasattr(planet,
                            'colony') and planet.colony.classification == "Primary Hub"
                    for planet in star.planetary_system.celestial_bodies
                    if
                    planet.body_type == "Planet" and hasattr(planet, 'colony')
                )
            )

        if "large_colonies" in population_filter:
            star_filter.add_filter(
                "large_colonies",
                lambda star: any(
                    hasattr(planet,
                            'colony') and planet.colony.current_population >= 1000000
                    for planet in star.planetary_system.celestial_bodies
                    if
                    planet.body_type == "Planet" and hasattr(planet, 'colony')
                )
            )

    # Generate the plot with applied filters
    fig = plot_generator.plot(html=False, return_fig=True,
                              star_filter=star_filter)

    # Update hover behaviors to make clicking more reliable
    for i in range(len(fig.data)):
        fig.data[i].hoverinfo = "text"
        if hasattr(fig.data[i], "marker"):
            fig.data[
                i].marker.opacity = 1.0  # Full opacity for better clickability

    # Change the layout to improve clickability
    fig.update_layout(
        clickmode="event+select",
        hoverdistance=10,
        hovermode="closest"
    )

    return fig


@app.callback(
    Output("population-stats", "children"),
    [Input("starmap-3d", "figure")],
    # This will trigger when the figure updates
    prevent_initial_call=True
)
def update_population_stats(_):
    """Update the population statistics display"""
    if hasattr(starmap, 'population_model'):
        return starmap.get_population_summary()
    else:
        return "No population data available."


# Add a callback for time advancement:
@app.callback(
    [Output("time-advance-info", "children"),
     Output("population-stats", "children", allow_duplicate=True)],
    [Input("time-advance-button", "n_clicks")],
    [State("time-advance-input", "value")],
    prevent_initial_call=True
)
def advance_time(n_clicks, years):
    """Advance the simulation time"""
    if not n_clicks or not years:
        return no_update, no_update

    result = starmap.advance_time(years)

    # Update the population stats
    if hasattr(starmap, 'population_model'):
        stats = starmap.get_population_summary()
    else:
        stats = "No population data available."

    return result, stats


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
        Output("habitable-filter", "value"),
        Output("population-filter", "value")
    ],
    Input("reset-button", "n_clicks"),
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    """Reset all filters to their default values"""
    return [], [], []


# Add debug callback to see what click data we're getting
@app.callback(
    Output('click-data-debug', 'children'),
    Input('starmap-3d', 'clickData'),
    prevent_initial_call=True
)
def display_click_data(clickData):
    if clickData:
        return json.dumps(clickData, indent=2)
    return "No click data"


# Simplified modal trigger - show the modal whenever the map is clicked
@app.callback(
    Output('edit-modal', 'style'),
    Input('starmap-3d', 'clickData'),
    State('edit-modal', 'style'),
    prevent_initial_call=True
)
def show_modal_on_click(clickData, current_style):
    """Simple handler to show the modal on any click"""
    if clickData is None:
        return no_update

    point_data = clickData['points'][0]

    # Check if this is a meaningful click (contains text or is a known curve number)
    if 'text' in point_data or point_data.get('curveNumber') in [0, 2, 4, 5]:
        print("Click detected!")  # Debug print
        new_style = current_style.copy()
        new_style['display'] = 'block'
        return new_style

    return no_update


@app.callback(
    [
        Output('modal-title', 'children'),
        Output('info-textarea', 'value'),
        Output('object-id-store', 'children'),
        Output('object-type-store', 'children'),
        Output('object-type-display', 'children'),
    ],
    Input('starmap-3d', 'clickData'),
    prevent_initial_call=True
)
def update_modal_content(clickData):
    """
    Update the modal content based on the clicked object
    This version uses name-based identification from the hover text
    """
    if clickData is None:
        return no_update, no_update, no_update, no_update, no_update

    # Get the clicked point data
    point_data = clickData['points'][0]
    curve_number = point_data.get('curveNumber')

    # Default values
    title = "Edit Object"
    info_text = ""
    object_id = "unknown"
    object_type = "unknown"
    object_type_display = "Click on a star, planet, or asteroid belt to edit its information"

    print(f"Clicked: {json.dumps(point_data)}")

    # Get the object name and text if available
    hover_text = point_data.get('text', '')
    if hover_text:
        print(f"Hover text: {hover_text}")

    # Try to identify the object based on the available information
    try:
        # If it has hover text, we can try to parse it
        if hover_text:
            # Extract the object name (usually before the first colon)
            if ":" in hover_text:
                object_name = hover_text.split(':', 1)[0].strip()

                # Identify object type
                if "Planet" in hover_text:
                    # It's a planet
                    for star in starmap.stars:
                        found = False
                        for body in star.planetary_system.celestial_bodies:
                            if body.body_type == "Planet" and body.name == object_name:
                                star_name = star.name[0] if isinstance(
                                    star.name, list) else star.name
                                title = f"Edit Planet: {body.name}"
                                # Use additional_info for the editable text
                                info_text = body.additional_info if body.additional_info else ""
                                object_id = f"{star.id}_{body.name}"
                                object_type = "planet"
                                object_type_display = f"Object Type: Planet | Name: {body.name} | Star: {star_name}"
                                print(f"Found planet {body.name} orbiting {star_name}")
                                found = True
                                break
                        if found:
                            break

                elif "Asteroid Belt" in hover_text or "asteroid belt" in hover_text:
                    # It's an asteroid belt
                    for star in starmap.stars:
                        found = False
                        for body in star.planetary_system.celestial_bodies:
                            if body.body_type == "Asteroid Belt" and body.name == object_name:
                                title = f"Edit Asteroid Belt: {body.name}"
                                info_text = body.additional_info if body.additional_info else ""
                                object_id = f"{star.id}_{body.name}"
                                object_type = "asteroid_belt"
                                object_type_display = f"Object Type: Asteroid Belt | Name: {body.name} | Star: {star.name[0]}"
                                print(f"Found asteroid belt {body.name} around {star.name[0]}")
                                found = True
                                break
                        if found:
                            break

            # Check if this is a star (based on text format)
            elif "," in hover_text and not hover_text.startswith(
                    "The planetary system"):
                # It's likely a star (e.g., "Detelus, M-Type")
                object_name = hover_text.split(',')[0].strip()

                # Find the star in the starmap
                for star in starmap.stars:
                    star_name = star.name[0] if isinstance(star.name,
                                                        list) else star.name
                    if star_name == object_name:
                        title = f"Edit Star: {star_name}"
                        info_text = star.additional_info if star.additional_info else ""
                        object_id = star.id
                        object_type = "star"
                        object_type_display = f"Object Type: Star | ID: {star.id} | Name: {star_name}"
                        print(f"Found star {star_name}")
                        break

            # Check if this is a planetary system description
            elif hover_text.startswith("The planetary system"):
                # Extract the star name
                try:
                    system_of_text = hover_text.split("The planetary system of ")[1].split(
                        "consists")[0].strip()
                    object_name = system_of_text

                    # Find the star with this name
                    for star in starmap.stars:
                        star_name = star.name[0] if isinstance(star.name,
                                                            list) else star.name
                        # Check if the name is in the system text
                        if star_name in object_name:
                            title = f"Edit Star System: {star_name}"
                            info_text = star.planetary_system.additional_info if star.planetary_system.additional_info else ""
                            object_id = f"system_{star.id}"
                            object_type = "system"
                            object_type_display = f"Object Type: Star System | Star: {star_name}"
                            print(f"Found star system for {star_name}")
                            break
                except Exception as e:
                    print(f"Error parsing system description: {e}")

        # If we couldn't identify the object by text, try by curve number
        if object_type == "unknown" and curve_number is not None:
            if curve_number == 0:
                # It's likely a star
                x, y, z = point_data.get('x'), point_data.get(
                    'y'), point_data.get('z')
                if x is not None and y is not None and z is not None:
                    # Find the closest star to these coordinates
                    closest_star = None
                    min_dist = float('inf')
                    for star in starmap.stars:
                        dist = ((star.x - x) ** 2 + (star.y - y) ** 2 + (
                                    star.z - z) ** 2) ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            closest_star = star

                    if closest_star and min_dist < 10:  # Reasonable threshold
                        star_name = closest_star.name[0] if isinstance(
                            closest_star.name, list) else closest_star.name
                        title = f"Edit Star: {star_name}"
                        info_text = closest_star.additional_info if closest_star.additional_info else ""
                        object_id = closest_star.id
                        object_type = "star"
                        object_type_display = f"Object Type: Star | ID: {closest_star.id} | Name: {star_name}"
                        print(f"Found star {star_name} by coordinates")

    except Exception as e:
        print(f"Error in update_modal_content: {e}")

    return title, info_text, object_id, object_type, object_type_display


@app.callback(
    Output('edit-modal', 'style', allow_duplicate=True),
    Input('close-modal', 'n_clicks'),
    State('edit-modal', 'style'),
    prevent_initial_call=True
)
def close_modal(n_clicks, current_style):
    """
    Close the modal when the close button is clicked
    """
    if n_clicks:
        new_style = current_style.copy()
        new_style['display'] = 'none'
        return new_style
    return no_update


@app.callback(
    [Output('save-status', 'children'),
     Output('save-status', 'style')],
    Input('save-changes-button', 'n_clicks'),
    [State('info-textarea', 'value'),
     State('object-id-store', 'children'),
     State('object-type-store', 'children')],
    prevent_initial_call=True
)
def save_changes(n_clicks, new_info, object_id, object_type):
    """
    Save changes to the object's additional_info and persist to JSON
    """
    if not n_clicks:
        return no_update, no_update

    if object_id is None or object_id == "unknown":
        return "Error: No object selected", {"color": "#FF5733",
                                             "marginTop": "10px"}

    if object_type is None or object_type == "unknown":
        return "Error: Unknown object type", {"color": "#FF5733",
                                              "marginTop": "10px"}

    try:
        print(f"Saving changes for {object_type} with ID {object_id}")
        found_object = False

        # Update the object based on its type
        if object_type == "star":
            try:
                # Find the star by id
                star_id = int(object_id)
                for star in starmap.stars:
                    if star.id == star_id:
                        star_name = star.name[0] if isinstance(star.name,
                                                               list) else star.name
                        star.additional_info = new_info
                        print(
                            f"Updated star {star_name} (ID: {star_id}) additional_info")
                        found_object = True
                        break
            except ValueError:
                print(f"Invalid star ID: {object_id}")
                return f"Error: Invalid star ID {object_id}", {
                    "color": "#FF5733", "marginTop": "10px"}

        elif object_type == "planet" or object_type == "asteroid_belt":
            # Parse the combined ID to get star ID and object name
            parts = object_id.split('_', 1)
            if len(parts) == 2:
                star_id, obj_name = parts
                try:
                    star_id = int(star_id)

                    # Find the star by id
                    for star in starmap.stars:
                        if star.id == star_id:
                            # Find the object in the star's planetary system
                            for body in star.planetary_system.celestial_bodies:
                                if body.name == obj_name:
                                    body.additional_info = new_info
                                    print(
                                        f"Updated {body.body_type} {body.name} of star {star.id} additional_info")
                                    found_object = True
                                    break
                            if found_object:
                                break
                except ValueError:
                    print(f"Invalid star ID: {star_id}")
                    return f"Error: Invalid star ID {star_id}", {
                        "color": "#FF5733", "marginTop": "10px"}
            else:
                print(f"Invalid object ID format: {object_id}")
                return f"Error: Invalid object ID format {object_id}", {
                    "color": "#FF5733", "marginTop": "10px"}

        elif object_type == "system":
            # Parse the system ID
            if object_id.startswith("system_"):
                try:
                    star_id = int(object_id.split('_')[1])

                    # Find the star by id
                    for star in starmap.stars:
                        if star.id == star_id:
                            star.planetary_system.additional_info = new_info
                            print(
                                f"Updated star system for star {star.id} additional_info")
                            found_object = True
                            break
                except ValueError:
                    print(f"Invalid star ID in system ID: {object_id}")
                    return f"Error: Invalid star ID in system ID {object_id}", {
                        "color": "#FF5733", "marginTop": "10px"}
            else:
                print(f"Invalid system ID format: {object_id}")
                return f"Error: Invalid system ID format {object_id}", {
                    "color": "#FF5733", "marginTop": "10px"}

        if not found_object:
            print(
                f"Object not found with ID {object_id} and type {object_type}")
            return f"Error: Object not found with ID {object_id}", {
                "color": "#FF5733", "marginTop": "10px"}

        # Save the updated data to JSON files
        writer = StarmapWriter()
        writer.save_starmap(starmap)
        print("Data saved to JSON files")

        return "Changes saved successfully!", {"color": "#4CAF50",
                                               "marginTop": "10px"}

    except Exception as e:
        print(f"Error saving changes: {str(e)}")
        return f"Error saving changes: {str(e)}", {"color": "#FF5733",
                                                   "marginTop": "10px"}


# Add custom CSS for the modal
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Abyssal Star Map</title>
        {%favicon%}
        {%css%}
        <style>
            /* Modal Animation */
            @keyframes fadeIn {
                from {opacity: 0}
                to {opacity: 1}
            }
            .modal {
                animation: fadeIn 0.3s;
            }
            /* Modal Content Animation */
            @keyframes slideIn {
                from {transform: translateY(-50px); opacity: 0;}
                to {transform: translateY(0); opacity: 1;}
            }
            .modal-content {
                animation: slideIn 0.4s;
            }
            /* Save button hover effect */
            #save-changes-button:hover {
                background-color: #45a049;
            }
            /* Close button hover effect */
            #close-modal:hover {
                color: #ccc;
            }
            /* Ensure modal is on top of everything */
            .modal {
                z-index: 9999 !important;
            }
            body {
                margin: 0;
                padding: 0;
                overflow-x: hidden;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Run the app
if __name__ == "__main__":
    app.run(debug=True)