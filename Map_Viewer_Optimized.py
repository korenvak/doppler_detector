# Optimized Map Viewer with DataLoader caching
import os
import glob
import base64
import tkinter as tk
from tkinter import filedialog

import pandas as pd
import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import dash_leaflet as dl

# ----------------------------
# 1) Optimized Data Loading
# ----------------------------
class DataLoader:
    """Centralized data loading with caching"""

    def __init__(self):
        self.df_sum = None
        self.df_fiber = None
        self.trace_dir = None
        self.graphs_dir = None
        self.summary_csv = None
        self.pixel_data_cache = {}
        self.flight_data_cache = {}
        self.image_cache = {}

    def load_all_data(self):
        """Load all data at once with progress feedback"""
        root = tk.Tk()
        root.withdraw()

        print("🚀 Starting optimized data loading...")

        # Summary CSV
        self.summary_csv = filedialog.askopenfilename(
            title="Select shifted summary CSV",
            filetypes=[("CSV files", "*.csv")]
        )
        if not self.summary_csv:
            raise SystemExit("No summary CSV selected.")

        print("📊 Loading summary data...")
        self.df_sum = pd.read_csv(self.summary_csv)

        if 'Sensor Lat' in self.df_sum.columns and 'Sensor Lon' in self.df_sum.columns:
            print("✅ Summary data loaded successfully")
        else:
            raise ValueError("Missing required sensor location columns")

        # Fiber config
        fiber_csv = filedialog.askopenfilename(
            title="Select fiber config CSV",
            filetypes=[("CSV files", "*.csv")]
        )
        if not fiber_csv:
            raise SystemExit("No fiber config CSV selected.")

        self.df_fiber = pd.read_csv(fiber_csv).sort_values('Location (pxl)')

        # Flight logs directory
        self.trace_dir = filedialog.askdirectory(
            title="Select folder with Flight_*_logs.csv"
        )
        if not self.trace_dir:
            raise SystemExit("Flight logs folder required.")

        # Graphs directory
        self.graphs_dir = filedialog.askdirectory(
            title="Select 'Graphs and statistics' folder (or Cancel to skip)"
        )
        if not self.graphs_dir:
            output_dir = os.path.dirname(self.summary_csv)
            self.graphs_dir = os.path.join(output_dir, "Graphs and statistics")
            if not os.path.exists(self.graphs_dir):
                print("⚠️ Warning: No graphs directory found")
                self.graphs_dir = None

        # Pre-load all pixel data
        self._preload_pixel_data()

        return self
    def _preload_pixel_data(self):
        """Pre-load all pixel CSV files into memory"""
        print("📂 Pre-loading pixel data...")
        summary_dir = os.path.dirname(self.summary_csv)

        unique_pixels = self.df_sum['Pixel'].unique()
        for px in unique_pixels:
            path = os.path.join(summary_dir, f"points_pixel_{px}.csv")
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    if 'lat' in df.columns and 'GPS Lat' not in df.columns:
                        df['GPS Lat'] = df['lat']
                    if 'lon' in df.columns and 'GPS Lon' not in df.columns:
                        df['GPS Lon'] = df['lon']
                    if 'alt' in df.columns and 'GPS Alt' not in df.columns:
                        df['GPS Alt'] = df['alt']
                    self.pixel_data_cache[px] = df
                except Exception as e:
                    print(f"⚠️ Error loading pixel {px}: {e}")

        print(f"✅ Loaded {len(self.pixel_data_cache)} pixel files")

    def get_pixel_data(self, pixel):
        """Get cached pixel data"""
        return self.pixel_data_cache.get(pixel)

    def get_flight_data(self, flight_number):
        """Get cached flight data"""
        if flight_number not in self.flight_data_cache:
            path = os.path.join(self.trace_dir, f"Flight_{flight_number}_logs.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['Flight number'] = flight_number
                self.flight_data_cache[flight_number] = df
            else:
                return None
        return self.flight_data_cache[flight_number]

    def get_image(self, filename):
        """Get cached image data"""
        if filename not in self.image_cache:
            self.image_cache[filename] = self._load_image(filename)
        return self.image_cache[filename]

    def _load_image(self, filename):
        """Load image with simplified path resolution"""
        if not filename or pd.isna(filename) or not self.graphs_dir:
            return None

        paths_to_try = [
            os.path.join(self.graphs_dir, filename),
            os.path.join(self.graphs_dir, os.path.basename(filename)),
            os.path.join(os.path.dirname(self.summary_csv), filename)
        ]

        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        raw = f.read()
                    b64 = base64.b64encode(raw).decode()
                    ext = path.lower().split('.')[-1]
                    mime = 'jpeg' if ext in ['jpg', 'jpeg'] else 'png'
                    return f"data:image/{mime};base64,{b64}"
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
        return None

# ----------------------------
# 2) Optimized Data Processing
# ----------------------------
def process_flight_data_optimized(data_loader):
    """Process flight data using pre-loaded cache"""
    dict_pixel = {}

    for _, ev in data_loader.df_sum.iterrows():
        fl = int(ev['Flight number'])
        px = int(ev['Pixel'])
        start = ev['Start time']
        end = ev['End time']

        dfx = data_loader.get_pixel_data(px)
        if dfx is None:
            continue

        window = dfx[
            (dfx['Flight number'] == fl) &
            (dfx['Time'] >= start) &
            (dfx['Time'] <= end)
        ]

        if window.empty:
            continue

        if 'lat' in window.columns and 'lon' in window.columns:
            coords = window[['lat', 'lon']].values.tolist()
        elif 'GPS Lat' in window.columns and 'GPS Lon' in window.columns:
            coords = window[['GPS Lat', 'GPS Lon']].values.tolist()
        else:
            continue

        meta = window.to_dict('records')
        dict_pixel.setdefault((fl, px), []).append((coords, meta))

    return dict_pixel

# ----------------------------
# 3) Optimized Coverage Loading
# ----------------------------
def load_coverage_data_optimized(graphs_dir):
    """Load coverage data with better error handling"""
    coverage_per_pixel = None
    coverage_per_type = None

    if graphs_dir:
        try:
            pixel_coverage_path = os.path.join(graphs_dir, "coverage_per_pixel.csv")
            if os.path.exists(pixel_coverage_path):
                coverage_per_pixel = pd.read_csv(pixel_coverage_path, index_col='Pixel')
                coverage_per_pixel.columns = [
                    f"Flight_{col}" if not str(col).startswith('Flight_') else col
                    for col in coverage_per_pixel.columns
                ]
                print(f"✅ Loaded coverage per pixel: {coverage_per_pixel.shape}")

            type_coverage_path = os.path.join(graphs_dir, "coverage_per_type.csv")
            if os.path.exists(type_coverage_path):
                coverage_per_type = pd.read_csv(type_coverage_path, index_col='Type')
                coverage_per_type.columns = [
                    f"Flight_{col}" if not str(col).startswith('Flight_') else col
                    for col in coverage_per_type.columns
                ]
                print(f"✅ Loaded coverage per type: {coverage_per_type.shape}")
        except Exception as e:
            print(f"⚠️ Error loading coverage data: {e}")

    return coverage_per_pixel, coverage_per_type

# ----------------------------
# 4) Initialize Data
# ----------------------------
print("🔄 Initializing optimized Map Viewer...")
data_loader = DataLoader().load_all_data()

dict_pixel = process_flight_data_optimized(data_loader)
coverage_per_pixel, coverage_per_type = load_coverage_data_optimized(data_loader.graphs_dir)

# Extract key information
df_sum = data_loader.df_sum
df_fiber = data_loader.df_fiber
fiber_coords = df_fiber[['Latitude', 'Longitude']].values.tolist()

flight_files = glob.glob(os.path.join(data_loader.trace_dir, "Flight_*_logs.csv"))
flight_numbers = sorted(int(os.path.basename(p).split("_")[1]) for p in flight_files)

sensor_positions = {}
for px in df_sum['Pixel'].unique():
    row = df_sum[df_sum['Pixel'] == px].iloc[0]
    sensor_positions[int(px)] = (
        row['Sensor Lat'],
        row['Sensor Lon'],
        row['Sensor Type'],
        row.get('min_dist3D', 0),
        row.get('max_dist3D', 0)
    )

if 'Event Lat' in df_sum and 'Event Lon' in df_sum:
    center = [df_sum['Event Lat'].mean(), df_sum['Event Lon'].mean()]
else:
    center = [df_fiber['Latitude'].mean(), df_fiber['Longitude'].mean()]


def generate_color_schemes():
    pixel_colors_list = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
        "#F7DC6F", "#BB8FCE", "#85C1E9", "#F8C471", "#82E0AA"
    ]
    type_colors_list = [
        "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#8E44AD"
    ]

    pixel_colors = {}
    type_colors = {}

    for i, px in enumerate(sorted(df_sum['Pixel'].unique())):
        pixel_colors[px] = pixel_colors_list[i % len(pixel_colors_list)]

    for i, stype in enumerate(sorted(df_sum['Sensor Type'].unique())):
        type_colors[stype] = type_colors_list[i % len(type_colors_list)]

    return pixel_colors, type_colors


pixel_colors, type_colors = generate_color_schemes()

MOVEMENT_COLORS = {
    'approaching': '#2ecc71',
    'departing': '#e74c3c',
    'hovering': '#9b59b6',
    'accelerating': '#f39c12',
    'decelerating': '#3498db',
    'turning': '#8e44ad',
    'cruising': '#95a5a6'
}


def build_popup(px, meta):
    """Create HTML popup content for a point"""
    return html.Div([
        html.H6(f"Pixel {px}"),
        html.P([
            f"Time: {meta.get('Time', 'N/A')}", html.Br(),
            f"Distance: {meta.get('distance_to_sensor', meta.get('dist3D', 'N/A')):.1f}m", html.Br(),
            f"Movement: {meta.get('pixel_movement_type', 'N/A')}"
        ], className='small')
    ])

# ----------------------------
# 5) Dash App with Optimized Callbacks
# ----------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🗺️ Flight Sensor Analysis Dashboard", className="text-primary mb-3"),
            html.P("Interactive visualization of flight paths and sensor data", className="text-muted")
        ])
    ]),
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Flight Selection", className="fw-bold"),
                    dcc.Dropdown(id='flight-dd', options=[{'label': f'Flight {f}', 'value': f} for f in flight_numbers], value=flight_numbers[0] if flight_numbers else None)
                ], md=3),
                dbc.Col([
                    html.Label("View Mode", className="fw-bold"),
                    dbc.RadioItems(id='view-mode', options=[{'label': 'Individual Pixels', 'value': 'individual'}, {'label': 'Clustered by Type', 'value': 'clustered'}], value='individual', inline=True)
                ], md=3),
                dbc.Col([
                    html.Label("Filter Selection", className="fw-bold"),
                    dcc.Dropdown(id='selection-dd', multi=True, value=[])
                ], md=4),
                dbc.Col([
                    html.Label("Display Options", className="fw-bold"),
                    dbc.Checklist(id='display-options', options=[{'label': 'Flight Path', 'value': 'flight_path'}, {'label': 'Fiber Cable', 'value': 'fiber'}, {'label': 'Movement Colors', 'value': 'movement_colors'}], value=['flight_path', 'fiber'])
                ], md=2)
            ])
        ])
    ], className="mb-3"),
    dl.Map(id='map', center=center, zoom=13, children=[dl.TileLayer(url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png")], style={'width': '100%', 'height': '70vh'}),
    html.Div(id='coverage-display', className="mt-3")
], fluid=True)

@app.callback([
    Output('selection-dd', 'options'),
    Output('selection-dd', 'value')],
    Input('view-mode', 'value')
)
def update_dropdown_options(view_mode):
    if view_mode == 'clustered':
        options = [{'label': f'{stype}', 'value': stype} for stype in sorted(df_sum['Sensor Type'].unique())]
        return options, []
    options = [{'label': f'Pixel {px}', 'value': px} for px in sorted(sensor_positions.keys())]
    return options, []


@app.callback(
    Output('map', 'children'),
    [Input('flight-dd', 'value'), Input('view-mode', 'value'), Input('selection-dd', 'value'), Input('display-options', 'value')]
)
def update_map_optimized(flight, view_mode, selection, display_options):
    if flight is None:
        return [dl.TileLayer(url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png")]

    layers = [dl.TileLayer(url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png")]

    if 'fiber' in (display_options or []):
        layers.append(dl.Polyline(positions=fiber_coords, color='#FFD700', weight=3, opacity=0.8))

    if 'flight_path' in (display_options or []):
        flight_data = data_loader.get_flight_data(flight)
        if flight_data is not None:
            coords = flight_data[['GPS Lat', 'GPS Lon']].values.tolist()
            layers.append(dl.Polyline(positions=coords, color='#00BFFF', weight=2, opacity=0.7))

    if view_mode == 'individual':
        pixels = selection or []
        for px in pixels:
            key = (flight, px)
            windows = dict_pixel.get(key, [])
            col = pixel_colors.get(px, '#0066CC')
            for coords, meta in windows:
                layers.append(dl.Polyline(positions=coords, color=col, weight=4, opacity=0.8))
                step = max(1, len(coords) // 20)
                for i in range(0, len(coords), step):
                    lat, lon = coords[i]
                    m = meta[i]
                    if 'movement_colors' in (display_options or []):
                        mv_type = m.get('pixel_movement_type', 'cruising')
                        if isinstance(mv_type, str):
                            mv_key = mv_type.split(',')[0].strip().lower()
                            marker_color = MOVEMENT_COLORS.get(mv_key, col)
                        else:
                            marker_color = col
                    else:
                        marker_color = col
                    layers.append(
                        dl.CircleMarker(
                            center=[lat, lon],
                            radius=4,
                            color=marker_color,
                            fill=True,
                            fillOpacity=0.8,
                            children=[dl.Popup(build_popup(px, m))]
                        )
                    )
        for px, (lat, lon, stype, _, _) in sensor_positions.items():
            is_selected = px in pixels
            col = pixel_colors.get(px, '#666666')
            layers.append(dl.CircleMarker(center=[lat, lon], radius=10 if is_selected else 6, color=col, fill=True, fillOpacity=1.0 if is_selected else 0.5, children=[dl.Tooltip(f"Pixel {px} ({stype})")]))
    else:
        selected_types = selection or []
        for stype in selected_types:
            col = type_colors[stype]
            pixels_of_type = [px for px, (_, _, ptype, _, _) in sensor_positions.items() if ptype == stype]
            for px in pixels_of_type:
                key = (flight, px)
                windows = dict_pixel.get(key, [])
                for coords, _ in windows:
                    layers.append(dl.Polyline(positions=coords, color=col, weight=4, opacity=0.8))

    return layers


@app.callback(
    Output('coverage-display', 'children'),
    [Input('flight-dd', 'value'), Input('view-mode', 'value'), Input('selection-dd', 'value')]
)
def update_coverage_display(flight, view_mode, selection):
    if not flight or not selection:
        return ""

    coverage_cards = []

    if view_mode == 'individual' and coverage_per_pixel is not None:
        flight_col = f"Flight_{flight}"
        for px in selection:
            if px in coverage_per_pixel.index and flight_col in coverage_per_pixel.columns:
                coverage = coverage_per_pixel.loc[px, flight_col]
                if data_loader.graphs_dir:
                    img_filename = f"donut_px_{px}_fl_{flight}.png"
                    img_data = data_loader.get_image(img_filename)
                    if img_data:
                        coverage_cards.append(dbc.Col([dbc.Card([dbc.CardBody([html.H6(f"Pixel {px}"), html.Img(src=img_data, style={'width': '100%', 'maxWidth': '200px'})])])], md=3))
                    else:
                        coverage_cards.append(dbc.Col([dbc.Card([dbc.CardBody([html.H6(f"Pixel {px}"), html.H2(f"{coverage:.1f}%", className="text-primary"), html.P("Coverage", className="text-muted")])])], md=3))

    return dbc.Row(coverage_cards) if coverage_cards else ""

if __name__ == '__main__':
    print("🚀 Starting Optimized Flight Dashboard...")
    print(f"📊 Loaded {len(df_sum)} events from {len(flight_numbers)} flights")
    print(f"🎯 {len(sensor_positions)} unique pixels")
    print(f"💾 Pre-loaded {len(data_loader.pixel_data_cache)} pixel data files")
    print("📡 Dashboard available at: http://localhost:8050")

    app.run_server(debug=True, port=8050)
