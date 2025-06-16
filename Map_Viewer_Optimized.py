import os
import glob
import base64
import tkinter as tk
from tkinter import filedialog

import pandas as pd
from functools import lru_cache

import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import dash_leaflet as dl

# ----------------------------
# 1) Optimized DataLoader
# ----------------------------
class DataLoader:
    """Centralized data loading with caching and dynamic column detection."""

    def __init__(self):
        self.df_sum = None
        self.df_fiber = None
        self.trace_dir = None
        self.graphs_dir = None
        self.summary_csv = None
        self.pixel_data_cache = {}
        self.image_raw_cache = {}

    def load_all_data(self):
        """Show file dialogs, load summary & fiber CSVs, and preload pixel data."""
        root = tk.Tk()
        root.withdraw()
        print("🚀 Starting optimized data loading...")

        # --- Summary CSV ---
        self.summary_csv = filedialog.askopenfilename(
            title="Select shifted summary CSV",
            filetypes=[("CSV files", "*.csv")]
        )
        if not self.summary_csv:
            raise SystemExit("No summary CSV selected.")

        print("📊 Loading summary data...")
        # Only load columns we actually use, with compact dtypes
        desired = [
            'Flight number','Pixel','Start time','End time',
            'Sensor Lat','Sensor Lon','Sensor Type','max_dist3D',
            'Snapshot','Event Lat','Event Lon'
        ]
        dtypes = {
            'Flight number': 'uint16',
            'Pixel':          'uint16',
            'Sensor Lat':     'float32',
            'Sensor Lon':     'float32',
            'Sensor Type':    'category',
            'max_dist3D':     'float32'
        }
        # Peek header to avoid KeyError
        hdr = pd.read_csv(self.summary_csv, nrows=0)
        usecols = [c for c in desired if c in hdr.columns]
        self.df_sum = pd.read_csv(
            self.summary_csv,
            usecols=usecols,
            dtype={k: v for k, v in dtypes.items() if k in usecols}
        )
        if 'Sensor Lat' not in self.df_sum.columns or 'Sensor Lon' not in self.df_sum.columns:
            raise ValueError("Missing required sensor latitude/longitude columns.")
        print("✅ Summary data loaded successfully.")

        # --- Fiber config CSV ---
        fiber_csv = filedialog.askopenfilename(
            title="Select fiber config CSV",
            filetypes=[("CSV files", "*.csv")]
        )
        if not fiber_csv:
            raise SystemExit("No fiber config CSV selected.")
        print("📊 Loading fiber config...")

        fiber_desired = ['Location (pxl)', 'Latitude', 'Longitude']
        hdr2 = pd.read_csv(fiber_csv, nrows=0)
        fiber_usecols = [c for c in fiber_desired if c in hdr2.columns]
        df_fiber = pd.read_csv(fiber_csv, usecols=fiber_usecols)
        # Downcast numeric columns safely
        if 'Location (pxl)' in df_fiber:
            df_fiber['Location (pxl)'] = pd.to_numeric(
                df_fiber['Location (pxl)'],
                errors='ignore',
                downcast='unsigned'
            )
        for coord in ('Latitude', 'Longitude'):
            if coord in df_fiber:
                df_fiber[coord] = pd.to_numeric(
                    df_fiber[coord],
                    errors='ignore',
                    downcast='float'
                )
        self.df_fiber = df_fiber.sort_values('Location (pxl)')

        # --- Flight logs directory ---
        self.trace_dir = filedialog.askdirectory(
            title="Select folder with Flight_*_logs.csv"
        )
        if not self.trace_dir:
            raise SystemExit("Flight logs folder required.")

        # --- Graphs directory (optional) ---
        self.graphs_dir = filedialog.askdirectory(
            title="Select 'Graphs and statistics' folder (or Cancel to skip)"
        )
        if not self.graphs_dir:
            fallback = os.path.join(
                os.path.dirname(self.summary_csv),
                "Graphs and statistics"
            )
            if os.path.exists(fallback):
                self.graphs_dir = fallback
            else:
                print("⚠️ No graphs directory found; continuing without it.")
                self.graphs_dir = None

        # Preload pixel CSV files
        self._preload_pixel_data()
        root.destroy()
        return self

    def _preload_pixel_data(self):
        """Load per-pixel CSVs into memory once."""
        print("📂 Pre-loading pixel data...")
        base = os.path.dirname(self.summary_csv)
        for px in self.df_sum['Pixel'].unique():
            path = os.path.join(base, f"points_pixel_{px}.csv")
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path)
                # Normalize column names
                if 'lat' in df.columns and 'GPS Lat' not in df.columns:
                    df['GPS Lat'] = df['lat']
                if 'lon' in df.columns and 'GPS Lon' not in df.columns:
                    df['GPS Lon'] = df['lon']
                if 'alt' in df.columns and 'GPS Alt' not in df.columns:
                    df['GPS Alt'] = df['alt']
                self.pixel_data_cache[px] = df
            except Exception as e:
                print(f"⚠️ Error loading pixel {px}: {e}")
        print(f"✅ Loaded {len(self.pixel_data_cache)} pixel files.")

    def get_pixel_data(self, pixel):
        return self.pixel_data_cache.get(pixel)

    @lru_cache(maxsize=3)
    def _load_flight_df(self, flight_number: int) -> pd.DataFrame:
        """
        Dynamically detect Time/Lat/Lon columns, load only those,
        then convert Time to datetime manually.
        """
        path = os.path.join(self.trace_dir, f"Flight_{flight_number}_logs.csv")

        # 1) Read only header to detect column names
        hdr = pd.read_csv(path, nrows=0)
        cols = list(hdr.columns)

        # 2) Detect time column
        time_col = next((c for c in cols if c.lower() == 'time'), None)
        if time_col is None:
            time_col = next((c for c in cols if 'time' in c.lower()), None)
        if time_col is None:
            raise ValueError("No time column found in flight log.")

        # 3) Detect latitude & longitude columns
        lat_col = 'GPS Lat' if 'GPS Lat' in cols else ('lat' if 'lat' in cols else None)
        lon_col = 'GPS Lon' if 'GPS Lon' in cols else ('lon' if 'lon' in cols else None)
        if lat_col is None or lon_col is None:
            raise ValueError("No latitude/longitude columns found in flight log.")

        usecols = [time_col, lat_col, lon_col]

        # 4) Read only the three columns (without parse_dates)
        df = pd.read_csv(
            path,
            usecols=usecols,
            dtype={lat_col: 'float32', lon_col: 'float32'}
        )

        # 5) Rename & convert Time → datetime
        df.rename(columns={time_col: 'Time', lat_col: 'GPS Lat', lon_col: 'GPS Lon'}, inplace=True)
        try:
            df['Time'] = pd.to_datetime(df['Time'])
        except Exception:
            # If you know the exact format you can pass format='...' here
            df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

        df['Flight number'] = flight_number
        return df

    def get_flight_data(self, flight_number: int):
        """Public accessor for the LRU-cached flight DataFrame."""
        path = os.path.join(self.trace_dir, f"Flight_{flight_number}_logs.csv")
        if not os.path.exists(path):
            return None
        return self._load_flight_df(flight_number)

    def _load_image_bytes(self, filename):
        """Load raw image bytes and return (bytes, mime)."""
        if not filename or pd.isna(filename) or not self.graphs_dir:
            return None

        candidates = [
            os.path.join(self.graphs_dir, filename),
            os.path.join(self.graphs_dir, os.path.basename(filename)),
            os.path.join(os.path.dirname(self.summary_csv), filename)
        ]
        for p in candidates:
            if os.path.exists(p):
                try:
                    raw = open(p, 'rb').read()
                    ext = p.lower().split('.')[-1]
                    mime = 'jpeg' if ext in ('jpg', 'jpeg') else 'png'
                    return raw, mime
                except Exception as e:
                    print(f"⚠️ Error loading image {p}: {e}")
        return None

    def get_image(self, filename):
        """Get and cache raw image bytes + mime."""
        if filename not in self.image_raw_cache:
            img = self._load_image_bytes(filename)
            if img:
                if len(self.image_raw_cache) >= 50:
                    # evict oldest entry
                    self.image_raw_cache.pop(next(iter(self.image_raw_cache)))
                self.image_raw_cache[filename] = img
        return self.image_raw_cache.get(filename)


# ----------------------------
# 2) Data Processing Helpers
# ----------------------------
def process_flight_data_optimized(data_loader: DataLoader):
    """
    Build dict_pixel[(flight, pixel)] = list of (coords, meta_records, snapshot).
    """
    dict_pixel = {}
    for _, ev in data_loader.df_sum.iterrows():
        fl = int(ev['Flight number'])
        px = int(ev['Pixel'])
        start, end = ev['Start time'], ev['End time']
        snapshot = ev.get('Snapshot', '')
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
            coords = window[['lat','lon']].values.tolist()
        else:
            coords = window[['GPS Lat','GPS Lon']].values.tolist()

        meta = window.to_dict('records')
        dict_pixel.setdefault((fl, px), []).append((coords, meta, snapshot))

    return dict_pixel


def load_coverage_data_optimized(graphs_dir):
    """
    Load coverage_per_pixel.csv and coverage_per_type.csv if present.
    """
    coverage_per_pixel = None
    coverage_per_type = None
    if graphs_dir:
        try:
            p1 = os.path.join(graphs_dir, "coverage_per_pixel.csv")
            if os.path.exists(p1):
                coverage_per_pixel = pd.read_csv(p1, index_col='Pixel')
                coverage_per_pixel.columns = [
                    f"Flight_{c}" if not str(c).startswith("Flight_") else c
                    for c in coverage_per_pixel.columns
                ]
                print(f"✅ Loaded coverage per pixel: {coverage_per_pixel.shape}")

            p2 = os.path.join(graphs_dir, "coverage_per_type.csv")
            if os.path.exists(p2):
                coverage_per_type = pd.read_csv(p2, index_col='Type')
                coverage_per_type.columns = [
                    f"Flight_{c}" if not str(c).startswith("Flight_") else c
                    for c in coverage_per_type.columns
                ]
                print(f"✅ Loaded coverage per type: {coverage_per_type.shape}")
        except Exception as e:
            print(f"⚠️ Error loading coverage data: {e}")

    return coverage_per_pixel, coverage_per_type


# ----------------------------
# 3) Initialize & Global Variables
# ----------------------------
print("🔄 Initializing optimized Map Viewer...")
data_loader = DataLoader().load_all_data()
dict_pixel = process_flight_data_optimized(data_loader)
coverage_per_pixel, coverage_per_type = load_coverage_data_optimized(data_loader.graphs_dir)

df_sum = data_loader.df_sum
df_fiber = data_loader.df_fiber
fiber_coords = df_fiber[['Latitude','Longitude']].values.tolist()

flight_files = glob.glob(os.path.join(data_loader.trace_dir, "Flight_*_logs.csv"))
flight_numbers = sorted(int(os.path.basename(p).split("_")[1]) for p in flight_files)

# Sensor static info
sensor_positions = {
    int(px): (row['Sensor Lat'], row['Sensor Lon'], row['Sensor Type'])
    for px, row in df_sum
    .drop_duplicates('Pixel')
    .set_index('Pixel')[['Sensor Lat','Sensor Lon','Sensor Type']]
    .iterrows()
}

max_dist_by_pixel_flight = df_sum.groupby(
    ['Flight number','Pixel']
)['max_dist3D'].max().to_dict()

if 'Event Lat' in df_sum.columns and 'Event Lon' in df_sum.columns:
    center = [df_sum['Event Lat'].mean(), df_sum['Event Lon'].mean()]
else:
    center = [df_fiber['Latitude'].mean(), df_fiber['Longitude'].mean()]


def get_pixel_coverage(pixel: int, flight: int):
    if coverage_per_pixel is not None:
        col = f"Flight_{flight}"
        if pixel in coverage_per_pixel.index and col in coverage_per_pixel.columns:
            return float(coverage_per_pixel.loc[pixel, col])
    return None


def generate_color_schemes():
    pixel_colors_list = [
        "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
        "#e6ab02", "#a6761d", "#666666", "#f7b7a3", "#6f5b4b"
    ]
    type_colors_list = [
        "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"
    ]
    pixel_colors = {
        px: pixel_colors_list[i % len(pixel_colors_list)]
        for i, px in enumerate(sorted(df_sum['Pixel'].unique()))
    }
    type_colors = {
        st: type_colors_list[i % len(type_colors_list)]
        for i, st in enumerate(sorted(df_sum['Sensor Type'].unique()))
    }
    return pixel_colors, type_colors


pixel_colors, type_colors = generate_color_schemes()

MOVEMENT_COLORS = {
    'approaching': '#2ecc71',
    'departing':   '#e74c3c',
    'hovering':    '#9b59b6',
    'accelerating':'#f39c12',
    'decelerating':'#3498db',
    'turning':     '#8e44ad',
    'cruising':    '#95a5a6'
}

OSM_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
SAT_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


def build_popup(px, meta, snapshot=None):
    """Construct HTML popup for a single trace marker."""
    items = [
        html.H6(f"Pixel {px}"),
        html.P([
            f"Time: {meta.get('Time','N/A')}", html.Br(),
            f"Distance: {meta.get('distance_to_sensor', meta.get('dist3D','N/A')):.1f} m", html.Br(),
            f"Movement: {meta.get('pixel_movement_type','N/A')}"
        ], className='small')
    ]
    if snapshot:
        raw = data_loader.get_image(snapshot)
        if raw:
            raw_bytes, mime = raw
            b64 = base64.b64encode(raw_bytes).decode()
            src = f"data:image/{mime};base64,{b64}"
            items.append(html.Img(src=src, style={'width':'200px'}))
    return html.Div(items)


def build_sensor_popup(px, stype, flight, coverage):
    """Popup for sensor anchor marker including max distance and coverage."""
    rows = [html.Div([
        html.I(className="fas fa-map-marker-alt me-1"),
        html.Strong(f"Pixel {px} ({stype})")
    ])]
    max_d = max_dist_by_pixel_flight.get((flight, px))
    if max_d is not None:
        rows.append(html.P(f"Max distance: {max_d:.1f} m", className='mb-0'))
    if coverage is not None:
        rows.append(html.P(f"Coverage: {coverage:.1f}%", className='mb-0'))
    return html.Div(rows, className='small')


# ----------------------------
# 4) Dash App & Callbacks
# ----------------------------
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)

map_tab = html.Div([
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
                    dcc.Dropdown(
                        id='flight-dd',
                        options=[{'label': f'Flight {f}', 'value': f} for f in flight_numbers],
                        value=flight_numbers[0] if flight_numbers else None
                    )
                ], md=3),
                dbc.Col([
                    html.Label("View Mode", className="fw-bold"),
                    dbc.RadioItems(
                        id='view-mode',
                        options=[
                            {'label': 'Individual Pixels', 'value': 'individual'},
                            {'label': 'Clustered by Type', 'value': 'clustered'}
                        ],
                        value='individual',
                        inline=True
                    )
                ], md=3),
                dbc.Col([
                    html.Label("Filter Selection", className="fw-bold"),
                    dcc.Dropdown(id='selection-dd', multi=True, value=[])
                ], md=3),
                dbc.Col([
                    html.Label("Display Options", className="fw-bold"),
                    dbc.Checklist(
                        id='display-options',
                        options=[
                            {'label': 'Flight Path', 'value': 'flight_path'},
                            {'label': 'Fiber Cable', 'value': 'fiber'},
                            {'label': 'Movement Colors', 'value': 'movement_colors'}
                        ],
                        value=['flight_path', 'fiber']
                    )
                ], md=3)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Map Style", className="fw-bold"),
                    dbc.RadioItems(
                        id='map-style',
                        options=[
                            {'label': 'Satellite', 'value': 'satellite'},
                            {'label': 'OpenStreetMap', 'value': 'osm'}
                        ],
                        value='satellite',
                        inline=True
                    )
                ], md=4)
            ], className="mt-2")
        ])
    ], className="mb-3"),
    dl.Map(
        id='map',
        center=center,
        zoom=13,
        children=[dl.TileLayer(url=OSM_URL)],
        style={'width': '100%', 'height': '70vh'}
    )
])

analysis_tab = html.Div([
    html.H4("Analysis Graphs", className="mt-3"),
    html.Div(id='analysis-images')
])

app.layout = dbc.Container([
    dcc.Tabs([
        dcc.Tab(label='Map', children=[map_tab]),
        dcc.Tab(label='Analysis', children=[analysis_tab])
    ])
], fluid=True)


@app.callback(
    [Output('selection-dd', 'options'), Output('selection-dd', 'value')],
    Input('view-mode', 'value')
)
def update_dropdown_options(view_mode):
    if view_mode == 'clustered':
        opts = [{'label': st, 'value': st} for st in sorted(df_sum['Sensor Type'].unique())]
        return opts, []
    opts = [{'label': f'Pixel {px}', 'value': px} for px in sorted(sensor_positions.keys())]
    return opts, []


@app.callback(
    Output('map', 'children'),
    [
        Input('flight-dd', 'value'),
        Input('view-mode', 'value'),
        Input('selection-dd', 'value'),
        Input('display-options', 'value'),
        Input('map-style', 'value')
    ]
)
def update_map(flight, view_mode, selection, display_options, map_style):
    # disable snapshots if selecting >1 pixel
    disable_snap = (view_mode == 'individual' and len(selection or []) > 1)

    tile_url = SAT_URL if map_style == 'satellite' else OSM_URL
    layers = [dl.TileLayer(url=tile_url, id='base')]

    if not flight:
        return layers

    # overlay fiber cable
    if 'fiber' in (display_options or []):
        layers.append(dl.Polyline(positions=fiber_coords, color='#FFD700', weight=3, opacity=0.8))

    # overlay flight path
    if 'flight_path' in (display_options or []):
        fd = data_loader.get_flight_data(flight)
        if fd is not None:
            coords = fd[['GPS Lat','GPS Lon']].values.tolist()
            layers.append(dl.Polyline(positions=coords, color='#00BFFF', weight=2, opacity=0.7))

    # render traces & markers
    if view_mode == 'individual':
        for px in selection or []:
            for w_idx, (coords, meta, snap) in enumerate(dict_pixel.get((flight, px), [])):
                col = pixel_colors.get(px, '#0066CC')
                layers.append(dl.Polyline(positions=coords, color=col, weight=4, opacity=0.8))
                step = max(1, len(coords) // 20)
                for i in range(0, len(coords), step):
                    lat, lon = coords[i]
                    m = meta[i]
                    marker_color = col
                    if 'movement_colors' in (display_options or []):
                        mv = m.get('pixel_movement_type','cruising').split(',')[0].strip().lower()
                        marker_color = MOVEMENT_COLORS.get(mv, col)
                    popup = build_popup(px, m, None if disable_snap else snap)
                    layers.append(dl.CircleMarker(
                        center=[lat,lon], radius=4, color=marker_color,
                        fill=True, fillOpacity=0.8,
                        children=[dl.Popup(popup)]
                    ))
        # sensor anchor markers
        for px in selection or []:
            lat, lon, stype = sensor_positions.get(px,(None,None,None))
            if lat is None: continue
            cov = get_pixel_coverage(px, flight)
            popup = build_sensor_popup(px, stype, flight, cov)
            layers.append(dl.CircleMarker(
                center=[lat,lon], radius=10,
                color=pixel_colors.get(px,'#666666'),
                fill=True, fillOpacity=1.0,
                children=[dl.Tooltip(f"Pixel {px} ({stype})"), dl.Popup(popup)]
            ))
    else:
        # clustered by sensor type
        for stype in selection or []:
            col = type_colors.get(stype,'#333333')
            for px,(_,_,t) in sensor_positions.items():
                if t != stype: continue
                for coords, _, _ in dict_pixel.get((flight,px), []):
                    layers.append(dl.Polyline(positions=coords, color=col, weight=4, opacity=0.8))
                lat, lon, _ = sensor_positions[px]
                cov = get_pixel_coverage(px, flight)
                popup = build_sensor_popup(px, stype, flight, cov)
                layers.append(dl.CircleMarker(
                    center=[lat,lon], radius=6,
                    color=col, fill=True, fillOpacity=0.6,
                    children=[dl.Tooltip(f"Pixel {px} ({stype})"), dl.Popup(popup)]
                ))

    return layers


@app.callback(
    Output('analysis-images', 'children'),
    [
        Input('flight-dd', 'value'),
        Input('view-mode', 'value'),
        Input('selection-dd', 'value')
    ]
)
def update_analysis_images(flight, view_mode, selection):
    if not data_loader.graphs_dir or not flight:
        return ""
    imgs = []
    def add_img(filename):
        raw = data_loader.get_image(filename)
        if raw:
            b, mime = raw
            src = f"data:image/{mime};base64," + base64.b64encode(b).decode()
            imgs.append(html.Img(src=src, style={'maxWidth':'400px','width':'100%'}))
    add_img('avg_coverage_per_pixel.png')
    add_img(f'union_all_fl_{flight}.png')
    if view_mode == 'individual':
        for px in selection or []:
            add_img(f'donut_px_{px}_fl_{flight}.png')
            add_img(f'hist_pixel_{px}_flight_{flight}.png')
    else:
        for stype in selection or []:
            add_img(f'coverage_type_{stype}_fl_{flight}.png')
    if not imgs:
        return ""
    return dbc.Row([dbc.Col(img, md=4) for img in imgs])


if __name__ == '__main__':
    print("🚀 Starting Optimized Flight Dashboard…")
    print(f"📊 Loaded {len(df_sum)} events from {len(flight_numbers)} flights")
    print(f"🎯 {len(sensor_positions)} unique pixels")
    print(f"💾 Pre-loaded {len(data_loader.pixel_data_cache)} pixel data files")
    app.run(debug=True, port=8050, use_reloader=False)
