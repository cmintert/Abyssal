import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

from Abyssal_map import Starmap, PlotGenerator

# Initialize the Dash app
app = dash.Dash(__name__, 
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server

# Nation data
NATIONS = [
    "All Nations",
    "Haven",
    "New Frontier Alliance",
    "Sol Protectorate",
    "United Stellar Colonies",
    "Void Confederacy",
]

NATION_COLORS = [
    None,  # All nations
    (0.5, 0.5, 0.5),
    (0.2, 0.8, 0.2),
    (0.8, 0.2, 0.2),
    (0.2, 0.2, 0.8),
    (0.8, 0.8, 0.2),
]

# Create the starmap (do this once at startup)
starmap = Starmap()
starmap.generate_star_systems(number_of_stars=521)
starmap.generate_nations(
    name_set=NATIONS[1:],  # Skip "All Nations"
    nation_colour_set=NATION_COLORS[1:],  # Skip None
    origin_set=[
        {"x": -200, "y": 100, "z": -100},
        {"x": -50, "y": 100, "z": 90},
        {"x": 0, "y": 0, "z": 0},
        {"x": 50, "y": 50, "z": 20},
        {"x": 100, "y": 100, "z": -50},
    ],
    expansion_rate_set=[0.7, 0.8, 1, 1, 0.9]
)
starmap.assign_stars_to_nations()
plot_generator = PlotGenerator(starmap)

# Define the app layout
app.layout = html.Div([
    html.H1("Abyssal Star Map", style={'textAlign': 'center', 'color': 'white'}),
    html.Div([
        html.Div([
            html.Label("Filter by Nation:", style={'color': 'white', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='nation-filter',
                options=[{'label': nation, 'value': i} for i, nation in enumerate(NATIONS)],
                value=0,  # Default to "All Nations"
                style={'width': '250px', 'backgroundColor': 'black', 'color': 'white'}
            ),
            html.Button('Reset View', id='reset-button', style={'marginLeft': '10px'})
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'padding': '10px'}),
        dcc.Graph(
            id='starmap-3d',
            figure=plot_generator.plot(html=False, return_fig=True),
            style={'height': '80vh'}
        )
    ]),
    html.Div([
        html.P("Interactive Star Map for Abyssal Universe", 
               style={'textAlign': 'center', 'color': 'white'})
    ])
], style={'backgroundColor': 'black', 'color': 'white', 'height': '100vh'})

@app.callback(
    Output('starmap-3d', 'figure'),
    [Input('nation-filter', 'value'),
     Input('reset-button', 'n_clicks')],
    [State('starmap-3d', 'figure')]
)
def update_figure(selected_nation_idx, n_clicks, current_fig):
    # Create a new figure based on the selected nation
    if selected_nation_idx == 0:  # "All Nations"
        return plot_generator.plot(html=False, return_fig=True)

    # Filter stars by nation
    filtered_stars = [star for star in starmap.stars if star.nation.name == NATIONS[selected_nation_idx]]

    # Create a custom figure with only the selected nation's stars
    fig = current_fig

    # Update the stars trace
    for trace in fig['data']:
        if trace['name'] == 'Stars':
            # Filter x, y, z coordinates for the selected nation
            x_coords = [star.x for star in filtered_stars]
            y_coords = [star.y for star in filtered_stars]
            z_coords = [star.z for star in filtered_stars]

            trace['x'] = x_coords
            trace['y'] = y_coords
            trace['z'] = z_coords

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
