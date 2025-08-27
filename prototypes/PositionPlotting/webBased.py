import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

# --- Fleet tracks ---
alpha_track = [(0, 0), (20, 0), (40, 0), (60, 0), (80, 0), (100, 0)]
bravo_track = [(0, 200), (20, 180), (40, 160), (60, 140), (80, 120), (100, 100)]
charlie_track = [(200, 0), (180, 20), (160, 40), (140, 60), (120, 80), (100, 100)]

ships = {
    "Alpha": {"track": alpha_track, "color": "green"},
    "Bravo": {"track": bravo_track, "color": "blue"},
    "Charlie": {"track": charlie_track, "color": "red"},
}

max_frames = max(len(s["track"]) for s in ships.values())

# --- Dash app ---
app = dash.Dash(__name__)
app.title = "Fleet Simulation"

app.layout = html.Div([
    html.H3("Fleet Simulation (Browser Version)"),
    
    dcc.Graph(id='fleet-graph'),
    
    html.Div([
        html.Button('Previous Step', id='prev-button', n_clicks=0),
        html.Button('Next Step', id='next-button', n_clicks=0),
        html.Span(id='step-counter', style={'margin-left': '20px', 'font-weight': 'bold'})
    ])
])

# Store current step in a hidden div
app.layout.children.append(html.Div(id='current-step', children='0', style={'display':'none'}))

# --- Update callback ---
@app.callback(
    Output('fleet-graph', 'figure'),
    Output('current-step', 'children'),
    Output('step-counter', 'children'),
    Input('prev-button', 'n_clicks'),
    Input('next-button', 'n_clicks'),
    State('current-step', 'children')
)
def update_graph(prev_clicks, next_clicks, current_step):
    frame = int(current_step)
    
    # Determine which button was pressed
    ctx = dash.callback_context
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'next-button':
            frame = (frame + 1) % max_frames
        elif button_id == 'prev-button':
            frame = (frame - 1) % max_frames

    # Create figure
    fig = go.Figure()
    
    # Add ships
    for name, data in ships.items():
        x, y = data["track"][frame] if frame < len(data["track"]) else data["track"][-1]
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(symbol='triangle-up', size=15, color=data["color"]),
            text=[name],
            textposition="top center",
            name=name
        ))
    
    # Add salvo line at frame 2 (third step)
    if frame == 2:
        bx, by = ships["Bravo"]["track"][frame]
        cx, cy = ships["Charlie"]["track"][frame]
        fig.add_trace(go.Scatter(
            x=[bx, cx], y=[by, cy],
            mode='lines',
            line=dict(color='red', width=3, dash='dash'),
            name='Bravo fires on Charlie',
            showlegend=False
        ))
    
    fig.update_layout(
        xaxis=dict(range=[-50, 250], title='km (east)'),
        yaxis=dict(range=[-50, 250], title='km (north)'),
        height=600
    )
    
    return fig, str(frame), f"Step: {frame}"

# --- Run server ---
if __name__ == '__main__':
    app.run(debug=False)
