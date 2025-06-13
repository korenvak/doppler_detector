import os
import glob
import itertools
import base64
import json
from datetime import datetime

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from functools import lru_cache
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist

import dash
from dash import html, dcc, Input, Output, State, ALL, callback_context, dash_table
import dash_leaflet as dl
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ----------------------------
# 1) Enhanced File Loading with Progress
# ----------------------------
def load_data_with_progress():
    """Enhanced data loading with better error handling and progress feedback"""
    root = tk.Tk()
    root.withdraw()

    try:
        print("🚀 Starting data loading process...")

        # Summary CSV
        summary_csv = filedialog.askopenfilename(
            title="Select shifted summary CSV",
            filetypes=[("CSV files", "*.csv")]
        )
        if not summary_csv:
            raise SystemExit("No summary CSV selected.")

        print("📊 Loading summary data...")
        df_sum = pd.read_csv(summary_csv)

        # Validate required columns
        required_cols = ['Flight number', 'Pixel', 'Sensor Lat', 'Sensor Lon', 'Sensor Type']
        missing_cols = [col for col in required_cols if col not in df_sum.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Fiber config
        fiber_csv = filedialog.askopenfilename(
            title="Select fiber config CSV",
            filetypes=[("CSV files", "*.csv")]
        )
        if not fiber_csv:
            raise SystemExit("No fiber config CSV selected.")

        print("🔗 Loading fiber configuration...")
        df_fiber = pd.read_csv(fiber_csv).sort_values('Location (pxl)')

        # Flight logs
        trace_dir = filedialog.askdirectory(
            title="Select folder with Flight_*_logs.csv"
        )
        if not trace_dir:
            raise SystemExit("Flight logs folder required.")

        # Graphs directory
        graphs_dir = filedialog.askdirectory(
            title="Select 'Graphs and statistics' folder"
        )
        if not graphs_dir:
            output_dir = os.path.dirname(summary_csv)
            graphs_dir = os.path.join(output_dir, "Graphs and statistics")
            if not os.path.exists(graphs_dir):
                print("⚠️ Warning: No graphs directory found. Coverage data will not be available.")
                graphs_dir = None

        print("✅ Data loading completed successfully!")
        return df_sum, df_fiber, trace_dir, graphs_dir, summary_csv

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise


# Load data
df_sum, df_fiber, trace_dir, graphs_dir, summary_csv = load_data_with_progress()


# ----------------------------
# 2) Enhanced Coverage and Analytics
# ----------------------------
def load_coverage_data(graphs_dir):
    """Load coverage data with error handling"""
    coverage_per_pixel = None
    coverage_per_type = None

    if graphs_dir:
        try:
            pixel_coverage_path = os.path.join(graphs_dir, "coverage_per_pixel.csv")
            if os.path.exists(pixel_coverage_path):
                coverage_per_pixel = pd.read_csv(pixel_coverage_path, index_col='Pixel')
                print(f"✅ Loaded coverage per pixel data: {coverage_per_pixel.shape}")

            type_coverage_path = os.path.join(graphs_dir, "coverage_per_type.csv")
            if os.path.exists(type_coverage_path):
                coverage_per_type = pd.read_csv(type_coverage_path, index_col='Type')
                print(f"✅ Loaded coverage per type data: {coverage_per_type.shape}")

        except Exception as e:
            print(f"⚠️ Warning: Error loading coverage data: {e}")

    return coverage_per_pixel, coverage_per_type


coverage_per_pixel, coverage_per_type = load_coverage_data(graphs_dir)


# ----------------------------
# 3) Enhanced Analytics Functions
# ----------------------------
def calculate_flight_analytics(df_sum, flight_numbers):
    """Calculate comprehensive flight analytics"""
    analytics = {}

    for flight in flight_numbers:
        flight_data = df_sum[df_sum['Flight number'] == flight]

        if not flight_data.empty:
            analytics[flight] = {
                'total_pixels': len(flight_data['Pixel'].unique()),
                'total_events': len(flight_data),
                'sensor_types': list(flight_data['Sensor Type'].unique()),
                'avg_min_distance': flight_data['min_dist3D'].mean() if 'min_dist3D' in flight_data.columns else 0,
                'avg_max_distance': flight_data['max_dist3D'].mean() if 'max_dist3D' in flight_data.columns else 0,
                'distance_range': (flight_data['max_dist3D'].max() - flight_data[
                    'min_dist3D'].min()) if 'max_dist3D' in flight_data.columns and 'min_dist3D' in flight_data.columns else 0,
                'doppler_types': list(
                    flight_data['Doppler Type'].unique()) if 'Doppler Type' in flight_data.columns else []
            }

    return analytics


def calculate_data_quality_metrics(df_sum, coverage_per_pixel, coverage_per_type):
    """Calculate data quality and completeness metrics"""
    total_events = len(df_sum)
    events_with_snapshots = len(
        df_sum[df_sum['Snapshot'].notna() & (df_sum['Snapshot'] != '')]) if 'Snapshot' in df_sum.columns else 0

    quality_metrics = {
        'total_events': total_events,
        'snapshot_completeness': (events_with_snapshots / total_events * 100) if total_events > 0 else 0,
        'unique_pixels': len(df_sum['Pixel'].unique()),
        'unique_flights': len(df_sum['Flight number'].unique()),
        'coverage_data_available': coverage_per_pixel is not None or coverage_per_type is not None
    }

    return quality_metrics


def calculate_flight_dynamics(flight_df, sensor_lat=None, sensor_lon=None):
    """Calculate flight dynamics including speed, acceleration, turning angles,
    and relative motion to an optional sensor position."""
    if len(flight_df) < 3:
        return flight_df

    flight_df = flight_df.copy()

    # Calculate time differences
    if 'Time' in flight_df.columns:
        flight_df['parsed_time'] = pd.to_datetime(
            flight_df['Time'], errors='coerce', infer_datetime_format=True
        )
        if flight_df['parsed_time'].isna().sum() > len(flight_df) * 0.5:
            flight_df['parsed_time'] = pd.date_range(start='2020-01-01', periods=len(flight_df), freq='S')
        flight_df['time_diff'] = flight_df['parsed_time'].diff().dt.total_seconds()
    else:
        flight_df['parsed_time'] = pd.date_range(start='2020-01-01', periods=len(flight_df), freq='S')
        flight_df['time_diff'] = 1.0

    # Calculate distances between consecutive points (3D)
    coord_cols = ['GPS Lat', 'GPS Lon']
    if 'GPS Alt' in flight_df.columns:
        coord_cols.append('GPS Alt')
    coords = flight_df[coord_cols].values
    distances = [0]
    from geopy.distance import geodesic
    for i in range(1, len(coords)):
        lat1, lon1 = coords[i-1][:2]
        lat2, lon2 = coords[i][:2]
        alt1 = coords[i-1][2] if len(coords[i-1]) > 2 else 0
        alt2 = coords[i][2] if len(coords[i]) > 2 else 0
        dist_2d = geodesic((lat1, lon1), (lat2, lon2)).meters
        dist_3d = np.sqrt(dist_2d**2 + (alt2 - alt1)**2)
        distances.append(dist_3d)

    flight_df['distance'] = distances

    # Calculate speed (m/s)
    flight_df['speed'] = flight_df['distance'] / flight_df['time_diff'].replace(0, 1)
    flight_df['speed'] = flight_df['speed'].fillna(0)

    # Smooth speed using Savitzky-Golay filter if enough points
    if len(flight_df) > 10:
        window_length = min(11, len(flight_df))
        if window_length % 2 == 0:
            window_length -= 1
        window_length = max(3, window_length)
        flight_df['speed_smooth'] = savgol_filter(
            flight_df['speed'].fillna(0),
            window_length=window_length,
            polyorder=min(3, window_length - 1)
        )
    else:
        flight_df['speed_smooth'] = flight_df['speed']

    # Calculate acceleration
    flight_df['acceleration'] = flight_df['speed_smooth'].diff() / flight_df['time_diff'].fillna(1)

    # Relative movement to sensor if provided
    if sensor_lat is not None and sensor_lon is not None:
        from geopy.distance import geodesic
        sensor_coord = (sensor_lat, sensor_lon)
        dist_sensor = []
        for row in coords:
            lat, lon = row[:2]
            alt = row[2] if len(row) > 2 else 0
            dist_2d = geodesic(sensor_coord, (lat, lon)).meters
            dist_sensor.append(np.sqrt(dist_2d**2 + alt**2))
        flight_df['dist_to_sensor'] = dist_sensor
        flight_df['dist_change'] = flight_df['dist_to_sensor'].diff().fillna(0)
        flight_df['relative_movement'] = flight_df['dist_change'].apply(
            lambda d: 'approaching' if d < -1 else (
                'departing' if d > 1 else 'steady')
        )
    else:
        flight_df['relative_movement'] = 'unknown'

    # Calculate heading changes (turning angles)
    headings = []
    for i in range(1, len(coords)):
        lat1, lon1 = coords[i - 1][:2]
        lat2, lon2 = coords[i][:2]
        # Calculate bearing
        dlon = np.radians(lon2 - lon1)
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        y = np.sin(dlon) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
        bearing = np.degrees(np.arctan2(y, x))
        headings.append(bearing)

    headings = [0] + headings
    flight_df['heading'] = headings

    # Calculate heading change (turning rate)
    flight_df['heading_change'] = flight_df['heading'].diff()
    # Normalize heading changes to [-180, 180]
    flight_df['heading_change'] = flight_df['heading_change'].apply(
        lambda x: x - 360 if x > 180 else (x + 360 if x < -180 else x)
    )

    # Classify movement patterns
    flight_df['movement_type'] = 'cruising'

    # Define thresholds
    ACCEL_THRESHOLD = 2.0  # m/s²
    DECEL_THRESHOLD = -2.0  # m/s²
    TURN_THRESHOLD = 15.0  # degrees per second
    SPEED_THRESHOLD = 5.0  # m/s

    # Classify movements
    flight_df.loc[flight_df['acceleration'] > ACCEL_THRESHOLD, 'movement_type'] = 'accelerating'
    flight_df.loc[flight_df['acceleration'] < DECEL_THRESHOLD, 'movement_type'] = 'decelerating'
    flight_df.loc[abs(
        flight_df['heading_change'] / flight_df['time_diff'].fillna(1)) > TURN_THRESHOLD, 'movement_type'] = 'turning'
    flight_df.loc[flight_df['speed_smooth'] < SPEED_THRESHOLD, 'movement_type'] = 'hovering'

    # Add Flight number if not present
    if 'Flight number' not in flight_df.columns:
        flight_df['Flight number'] = flight_df.index[0] if len(flight_df) > 0 else 0

    return flight_df


def calculate_relative_movement_to_pixel(df_flight, sensor_lat, sensor_lon, start_time, end_time):
    """Classify movement relative to a pixel's sensor."""
    df = df_flight.copy()
    df['parsed_time'] = pd.to_datetime(
        df.get('Time'), errors='coerce', infer_datetime_format=True
    )
    start = pd.to_datetime(start_time)
    end = pd.to_datetime(end_time)
    df = df[(df['parsed_time'] >= start) & (df['parsed_time'] <= end)].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=['time','lat','lon','alt','distance_to_sensor','delta_distance','speed','delta_speed','heading','delta_heading','pixel_movement_type'])

    df['time_diff'] = df['parsed_time'].diff().dt.total_seconds().fillna(1)

    coord_cols = ['GPS Lat', 'GPS Lon']
    if 'GPS Alt' in df.columns:
        coord_cols.append('GPS Alt')
    coords = df[coord_cols].values
    from geopy.distance import geodesic

    # distance to sensor
    sensor_coord = (sensor_lat, sensor_lon)
    dist_to_sensor = []
    for row in coords:
        lat, lon = row[:2]
        alt = row[2] if len(row) > 2 else 0
        d2d = geodesic((lat, lon), sensor_coord).meters
        dist_to_sensor.append(np.sqrt(d2d**2 + alt**2))
    df['distance_to_sensor'] = dist_to_sensor
    df['delta_distance'] = df['distance_to_sensor'].diff().fillna(0)

    # path distance and speed
    dist_path = [0]
    for i in range(1, len(coords)):
        lat1, lon1 = coords[i-1][:2]
        lat2, lon2 = coords[i][:2]
        alt1 = coords[i-1][2] if len(coords[i-1]) > 2 else 0
        alt2 = coords[i][2] if len(coords[i]) > 2 else 0
        d2d = geodesic((lat1, lon1), (lat2, lon2)).meters
        d3d = np.sqrt(d2d**2 + (alt2 - alt1)**2)
        dist_path.append(d3d)
    df['dist3d'] = dist_path
    df['speed'] = df['dist3d'] / df['time_diff'].replace(0, 1)
    df['delta_speed'] = df['speed'].diff().fillna(0)

    # heading
    headings = [0]
    for i in range(1, len(coords)):
        lat1, lon1 = coords[i-1][:2]
        lat2, lon2 = coords[i][:2]
        dlon = np.radians(lon2 - lon1)
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        y = np.sin(dlon) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
        headings.append(np.degrees(np.arctan2(y, x)))
    df['heading'] = headings
    df['delta_heading'] = df['heading'].diff().fillna(0).apply(lambda x: x-360 if x>180 else (x+360 if x<-180 else x))

    def classify(r):
        if r['delta_distance'] < -1:
            return 'approaching'
        if r['delta_distance'] > 1:
            return 'departing'
        if r['speed'] < 1:
            return 'hovering'
        if r['delta_speed'] > 2:
            return 'accelerating'
        if r['delta_speed'] < -2:
            return 'decelerating'
        if abs(r['delta_heading']) > 15:
            return 'turning'
        return 'cruising'

    df['pixel_movement_type'] = df.apply(classify, axis=1)

    df = df.rename(columns={'parsed_time':'time','GPS Lat':'lat','GPS Lon':'lon','GPS Alt':'alt'})
    base_cols = ['time','lat','lon','alt','distance_to_sensor','delta_distance','speed','delta_speed','heading','delta_heading','pixel_movement_type']
    extras = [c for c in ['Sensor Type','Doppler Type','Time'] if c in df.columns]
    return df[extras + base_cols]


def analyze_sensor_detection_by_movement(pixel_dict, flight):
    """Analyze movement types relative to pixel detections for a flight."""
    detection_analysis = {
        'movement_type': [],
        'total_points': [],
        'detected_points': [],
        'detection_rate': [],
        'avg_distance': []
    }

    counts = {}
    total_points = 0

    for (fl, px), windows in pixel_dict.items():
        if fl != flight:
            continue
        for coords, meta, _ in windows:
            for m in meta:
                mv = m.get('pixel_movement_type', 'unknown')
                dist = m.get('distance_to_sensor', 0)
                if pd.isna(dist):
                    dist = 0
                if mv not in counts:
                    counts[mv] = {'count': 0, 'dist_sum': 0.0}
                counts[mv]['count'] += 1
                counts[mv]['dist_sum'] += dist
                total_points += 1

    if total_points == 0:
        return pd.DataFrame(detection_analysis)

    for mv, data in counts.items():
        detection_analysis['movement_type'].append(mv)
        detection_analysis['total_points'].append(total_points)
        detection_analysis['detected_points'].append(data['count'])
        detection_analysis['detection_rate'].append(data['count'] / total_points * 100)
        avg_dist = data['dist_sum'] / data['count'] if data['count'] > 0 else 0
        detection_analysis['avg_distance'].append(avg_dist)

    return pd.DataFrame(detection_analysis)


# ----------------------------
# 4) Enhanced Caching with Analytics
# ----------------------------
@lru_cache(maxsize=None)
def load_flight_path_with_analytics(flight_number):
    """Load flight path with additional analytics"""
    try:
        path = os.path.join(trace_dir, f"Flight_{flight_number}_logs.csv")
        df = pd.read_csv(path)

        coords = df[['GPS Lat', 'GPS Lon']].values.tolist()

        # Calculate flight statistics
        total_distance = 0
        if len(coords) > 1:
            from geopy.distance import geodesic
            for i in range(1, len(coords)):
                total_distance += geodesic(coords[i - 1], coords[i]).meters

        stats = {
            'total_distance_km': total_distance / 1000,
            'total_points': len(coords),
            'altitude_range': (df['GPS Alt'].min(), df['GPS Alt'].max()) if 'GPS Alt' in df.columns else (0, 0),
            'flight_bounds': {
                'min_lat': df['GPS Lat'].min(),
                'max_lat': df['GPS Lat'].max(),
                'min_lon': df['GPS Lon'].min(),
                'max_lon': df['GPS Lon'].max()
            }
        }

        return coords, stats
    except Exception as e:
        print(f"Error loading flight {flight_number}: {e}")
        return [], {}


# Keep existing caching functions but enhance them
@lru_cache(maxsize=None)
def load_flight_path(flight_number):
    coords, _ = load_flight_path_with_analytics(flight_number)
    return coords


@lru_cache(maxsize=None)
def load_pixel_csv(px):
    fpath = os.path.join(os.path.dirname(summary_csv), f"points_pixel_{px}.csv")
    return pd.read_csv(fpath) if os.path.exists(fpath) else None


@lru_cache(maxsize=None)
def get_image_data(path):
    """Enhanced image loading with better error handling"""
    if not path or pd.isna(path) or str(path).lower() == 'nan':
        return None

    path = str(path).strip()
    possible_paths = []

    if os.path.isabs(path) and os.path.exists(path):
        possible_paths.append(path)
    else:
        if graphs_dir:
            possible_paths.extend([
                os.path.join(graphs_dir, path),
                os.path.join(graphs_dir, os.path.basename(path))
            ])

        summary_dir = os.path.dirname(summary_csv)
        possible_paths.extend([
            os.path.join(summary_dir, path),
            os.path.join(summary_dir, os.path.basename(path))
        ])

    for try_path in possible_paths:
        if os.path.exists(try_path):
            try:
                with open(try_path, 'rb') as f:
                    raw = f.read()
                b64 = base64.b64encode(raw).decode()

                ext = try_path.lower().split('.')[-1]
                if ext in ['jpg', 'jpeg']:
                    return f"data:image/jpeg;base64,{b64}"
                elif ext == 'png':
                    return f"data:image/png;base64,{b64}"
                else:
                    return f"data:image/png;base64,{b64}"

            except Exception as e:
                print(f"Error loading image {try_path}: {e}")
                continue

    return None


def get_coverage_data(pixel, flight):
    """Get coverage percentage for a pixel and flight."""
    if coverage_per_pixel is not None and pixel in coverage_per_pixel.index:
        flight_str = str(flight)
        flight_col = f"Flight_{flight}"
        if flight_str in coverage_per_pixel.columns:
            return coverage_per_pixel.loc[pixel, flight_str]
        elif flight_col in coverage_per_pixel.columns:
            return coverage_per_pixel.loc[pixel, flight_col]
    return None


def get_type_coverage_data(sensor_type, flight):
    """Get coverage percentage for a sensor type and flight."""
    if coverage_per_type is not None and sensor_type in coverage_per_type.index:
        flight_str = str(flight)
        flight_col = f"Flight_{flight}"
        if flight_str in coverage_per_type.columns:
            return coverage_per_type.loc[sensor_type, flight_str]
        elif flight_col in coverage_per_type.columns:
            return coverage_per_type.loc[sensor_type, flight_col]
    return None


# ----------------------------
# 5) Enhanced Data Processing
# ----------------------------
def process_flight_data():
    """Process and enrich flight data with analytics"""
    dict_pixel = {}

    for _, ev in df_sum.iterrows():
        fl = int(ev['Flight number'])
        px = int(ev['Pixel'])
        start = ev['Start time']
        end = ev['End time']
        snapshot = ev.get('Snapshot', '')

        dfx = load_pixel_csv(px)
        if dfx is None:
            print(f"\u26A0\uFE0F Missing pixel CSV for pixel {px}")
            continue

        window = dfx[
            (dfx['Flight number'] == fl) &
            (dfx['Time'] >= start) &
            (dfx['Time'] <= end)
            ]
        if window.empty:
            continue

        # Calculate movement relative to this pixel's sensor
        rel_window = calculate_relative_movement_to_pixel(
            dfx[dfx['Flight number'] == fl],
            ev['Sensor Lat'],
            ev['Sensor Lon'],
            start,
            end
        )
        if rel_window.empty:
            continue

        coords = rel_window[['lat', 'lon']].values.tolist()
        meta = rel_window.to_dict('records')
        dict_pixel.setdefault((fl, px), []).append((coords, meta, snapshot))

    return dict_pixel


dict_pixel = process_flight_data()


# ----------------------------
# 6) Enhanced Color Management
# ----------------------------
def generate_color_schemes():
    """Generate enhanced color schemes with better contrast"""
    pixel_colors_list = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
        "#F7DC6F", "#BB8FCE", "#85C1E9", "#F8C471", "#82E0AA",
        "#F1948A", "#85C1E9", "#F9E79F", "#D7BDE2", "#A9DFBF"
    ]

    type_colors_list = [
        "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#8E44AD",
        "#16A085", "#E67E22", "#34495E", "#27AE60", "#E74C3C"
    ]

    pixel_colors = {}
    type_colors = {}

    for i, px in enumerate(sorted(df_sum['Pixel'].unique())):
        pixel_colors[px] = pixel_colors_list[i % len(pixel_colors_list)]

    for i, stype in enumerate(sorted(df_sum['Sensor Type'].unique())):
        type_colors[stype] = type_colors_list[i % len(type_colors_list)]

    return pixel_colors, type_colors


pixel_colors, type_colors = generate_color_schemes()

# ----------------------------
# 7) Prepare data for dashboard
# ----------------------------
fiber_coords = df_fiber[['Latitude', 'Longitude']].values.tolist()
flight_files = glob.glob(os.path.join(trace_dir, "Flight_*_logs.csv"))
flight_numbers = sorted(int(os.path.basename(p).split("_")[1]) for p in flight_files)

# Calculate analytics
flight_analytics = calculate_flight_analytics(df_sum, flight_numbers)
quality_metrics = calculate_data_quality_metrics(df_sum, coverage_per_pixel, coverage_per_type)

# Sensor positions
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

# Map center
if 'Event Lat' in df_sum and 'Event Lon' in df_sum:
    center = [df_sum['Event Lat'].mean(), df_sum['Event Lon'].mean()]
else:
    center = [df_fiber['Latitude'].mean(), df_fiber['Longitude'].mean()]

# ----------------------------
# 8) Enhanced App Layout with Bootstrap
# ----------------------------
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
                suppress_callback_exceptions=True)

# Add custom CSS for better tab visibility
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Make tabs more visible */
            .nav-tabs {
                border-bottom: 2px solid #dee2e6;
                margin-bottom: 1rem;
                background-color: #f8f9fa;
                padding: 10px 10px 0 10px;
            }
            .nav-tabs .nav-link {
                border: 1px solid #dee2e6;
                margin-right: 5px;
                padding: 10px 15px;
                font-weight: 500;
                background-color: white;
                color: #495057;
            }
            .nav-tabs .nav-link:hover {
                background-color: #e9ecef;
                border-color: #dee2e6;
            }
            .nav-tabs .nav-link.active {
                background-color: #007bff !important;
                color: white !important;
                border-color: #007bff !important;
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

# Enhanced layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1([
                    html.I(className="fas fa-plane me-3", style={"color": "#3498db"}),
                    "Flight Sensor Analysis Dashboard"
                ], className="text-primary mb-2"),
                html.P("Interactive visualization and analysis of flight paths, sensor data, and coverage metrics",
                       className="text-muted mb-4 lead")
            ])
        ])
    ], className="mb-4"),

    # Quick Stats Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{quality_metrics['total_events']}", className="text-primary mb-1"),
                    html.P("Total Events", className="text-muted mb-0 small")
                ])
            ], className="text-center border-start border-primary border-4")
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{quality_metrics['unique_flights']}", className="text-success mb-1"),
                    html.P("Flights", className="text-muted mb-0 small")
                ])
            ], className="text-center border-start border-success border-4")
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{quality_metrics['unique_pixels']}", className="text-info mb-1"),
                    html.P("Pixels", className="text-muted mb-0 small")
                ])
            ], className="text-center border-start border-info border-4")
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{quality_metrics['snapshot_completeness']:.1f}%", className="text-warning mb-1"),
                    html.P("Data Complete", className="text-muted mb-0 small")
                ])
            ], className="text-center border-start border-warning border-4")
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("✓" if quality_metrics['coverage_data_available'] else "✗",
                            className="text-success mb-1" if quality_metrics[
                                'coverage_data_available'] else "text-danger mb-1"),
                    html.P("Coverage Data", className="text-muted mb-0 small")
                ])
            ], className="text-center border-start border-secondary border-4")
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{len(df_sum['Sensor Type'].unique())}", className="text-dark mb-1"),
                    html.P("Sensor Types", className="text-muted mb-0 small")
                ])
            ], className="text-center border-start border-dark border-4")
        ], md=2)
    ], className="mb-4"),

    # Control Panel
    dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-sliders-h me-2", style={"color": "#3498db"}),
                "Control Panel"
            ], className="mb-0 text-primary")
        ]),
        dbc.CardBody([
            dbc.Row([
                # Flight Selection
                dbc.Col([
                    html.Label([
                        html.I(className="fas fa-plane me-1"),
                        "Flight Selection"
                    ], className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id='flight-dd',
                        options=[{'label': f'Flight {f}', 'value': f} for f in flight_numbers],
                        value=flight_numbers[0] if flight_numbers else None,
                        multi=False,
                        placeholder="Select a flight...",
                        className="mb-2"
                    ),
                    html.Div(id='flight-info', className="small text-muted")
                ], md=3),

                # View Mode
                dbc.Col([
                    html.Label([
                        html.I(className="fas fa-eye me-1"),
                        "View Mode"
                    ], className="fw-bold mb-2"),
                    dbc.RadioItems(
                        id='view-mode',
                        options=[
                            {'label': ' Individual Pixels', 'value': 'individual'},
                            {'label': ' Clustered by Type', 'value': 'clustered'}
                        ],
                        value='individual',
                        inline=True
                    )
                ], md=3),

                # Selection
                dbc.Col([
                    html.Label([
                        html.I(className="fas fa-filter me-1"),
                        "Filter Selection"
                    ], className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id='selection-dd',
                        multi=True,
                        value=[],
                        placeholder="Select items to display..."
                    )
                ], md=3),

                # Display Options - REMOVED HEATMAP
                dbc.Col([
                    html.Label([
                        html.I(className="fas fa-cog me-1"),
                        "Display Options"
                    ], className="fw-bold mb-2"),
                    dbc.Checklist(
                        id='display-options',
                        options=[
                            {'label': ' Flight Path', 'value': 'flight_path'},
                            {'label': ' Fiber Cable', 'value': 'fiber'},
                            {'label': ' Cluster Markers', 'value': 'cluster'}
                        ],
                        value=['flight_path', 'fiber'],
                        className="small"
                    )
                ], md=3)
            ])
        ])
    ], className="mb-4"),

    # Main Content Area with Better Navigation
    dbc.Card([
        dbc.CardBody([
            # Custom Navigation Buttons
            html.Div([
                dbc.ButtonGroup([
                    dbc.Button("🗺️ Interactive Map", id="btn-map", color="primary", className="me-1", active=True),
                    dbc.Button("🎮 3D View", id="btn-3d", color="secondary", className="me-1"),
                    dbc.Button("🏃 Movement Analysis", id="btn-movement", color="secondary", className="me-1"),
                    dbc.Button("📊 Analytics", id="btn-analytics", color="secondary", className="me-1"),
                    dbc.Button("📋 Data Table", id="btn-data", color="secondary", className="me-1"),
                    dbc.Button("⚙️ Settings", id="btn-settings", color="secondary", className="me-1")
                ], size="sm")
            ], className="mb-3"),

            # Content area
            html.Div(id="tab-content", style={"minHeight": "500px"})
        ])
    ], className="shadow")
], fluid=True, style={"backgroundColor": "#f8f9fa"})


# ----------------------------
# 9) Enhanced Callbacks
# ----------------------------

# Tab content callback

@app.callback(
    [Output('tab-content', 'children'),
    Output('btn-map', 'active'),
    Output('btn-3d', 'active'),
    Output('btn-movement', 'active'),
    Output('btn-analytics', 'active'),
    Output('btn-data', 'active'),
    Output('btn-settings', 'active'),
    Output('btn-map', 'color'),
    Output('btn-3d', 'color'),
    Output('btn-movement', 'color'),
    Output('btn-analytics', 'color'),
    Output('btn-data', 'color'),
    Output('btn-settings', 'color')],
    [Input('btn-map', 'n_clicks'),
     Input('btn-3d', 'n_clicks'),
     Input('btn-movement', 'n_clicks'),
     Input('btn-analytics', 'n_clicks'),
     Input('btn-data', 'n_clicks'),
     Input('btn-settings', 'n_clicks')]
)
def render_content_from_buttons(btn_map, btn_3d, btn_movement, btn_analytics, btn_data, btn_settings):
    # Determine which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        active_tab = "map"
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        active_tab = button_id.split('-')[1]

    # Set active states
    active_states = [False] * 6
    colors = ['secondary'] * 6

    # Content to display
    content = None

    if active_tab == "map":
        active_states[0] = True
        colors[0] = 'primary'
        content = html.Div([
            html.Div(id="map-controls", className="mb-3"),
            dl.Map(
                id='map',
                center=center,
                zoom=13,
                children=[
                    dl.TileLayer(
                        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                        attribution="Esri"
                    )
                ],
                style={'width': '100%', 'height': '70vh', 'border-radius': '8px', 'border': '1px solid #dee2e6'}
            )
        ])

    elif active_tab == "3d":
        active_states[1] = True
        colors[1] = 'primary'
        content = html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Select Flight for 3D View:", className="fw-bold"),
                    dcc.Dropdown(
                        id='3d-flight-select',
                        options=[{'label': f'Flight {f}', 'value': f} for f in flight_numbers],
                        value=flight_numbers[0] if flight_numbers else None
                    )
                ], md=4),
                dbc.Col([
                    dbc.Checklist(
                        id='3d-options',
                        options=[
                            {'label': ' Show Sensor Detections', 'value': 'detections'},
                            {'label': ' Color by Speed', 'value': 'speed'}
                        ],
                        value=['detections'],
                        inline=True
                    )
                ], md=8)
            ], className="mb-3"),
            dcc.Graph(id='3d-flight-view', style={'height': '70vh'})
        ])

    elif active_tab == "movement":
        active_states[2] = True
        colors[2] = 'primary'
        content = html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Select Flight for Movement Analysis:", className="fw-bold"),
                    dcc.Dropdown(
                        id='movement-flight-select',
                        options=[{'label': f'Flight {f}', 'value': f} for f in flight_numbers],
                        value=flight_numbers[0] if flight_numbers else None
                    )
                ], md=6)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='movement-timeline', style={'height': '400px'})
                ], md=12)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='detection-by-movement', style={'height': '400px'})
                ], md=6),
                dbc.Col([
                    dcc.Graph(id='movement-statistics', style={'height': '400px'})
                ], md=6)
            ])
        ])

    elif active_tab == "analytics":
        active_states[3] = True
        colors[3] = 'primary'
        content = html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='coverage-chart', style={'height': '400px'})
                ], md=6),
                dbc.Col([
                    dcc.Graph(id='distance-analysis', style={'height': '400px'})
                ], md=6)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='sensor-distribution', style={'height': '400px'})
                ], md=6),
                dbc.Col([
                    dcc.Graph(id='flight-comparison', style={'height': '400px'})
                ], md=6)
            ])
        ])

    elif active_tab == "data":
        active_states[4] = True
        colors[4] = 'primary'
        content = html.Div([
            html.Div(id="data-table-container")
        ])

    elif active_tab == "settings":
        active_states[5] = True
        colors[5] = 'primary'
        content = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H5("🎨 Map Styling"),
                    dbc.Card([
                        dbc.CardBody([
                            html.Label("Map Theme:"),
                            dcc.Dropdown(
                                id='map-theme',
                                options=[
                                    {'label': 'Satellite', 'value': 'satellite'},
                                    {'label': 'OpenStreetMap', 'value': 'osm'},
                                    {'label': 'Terrain', 'value': 'terrain'}
                                ],
                                value='satellite'
                            ),
                            html.Hr(),
                            html.Label("Marker Size:"),
                            dcc.Slider(id='marker-size', min=1, max=10, value=5, step=1),
                            html.Hr(),
                            html.Label("Path Opacity:"),
                            dcc.Slider(id='path-opacity', min=0.1, max=1.0, value=0.8, step=0.1)
                        ])
                    ])
                ], md=6),
                dbc.Col([
                    html.H5("📊 Analysis Options"),
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Checklist(
                                options=[
                                    {'label': 'Show Statistics Panel', 'value': 'stats'},
                                    {'label': 'Auto-refresh Data', 'value': 'refresh'},
                                    {'label': 'Export Enabled', 'value': 'export'}
                                ],
                                value=['stats']
                            )
                        ])
                    ])
                ], md=6)
            ])
        ])

    return [content] + active_states + colors

# Update dropdown options
@app.callback(
    [Output('selection-dd', 'options'),
     Output('selection-dd', 'value')],
    Input('view-mode', 'value')
)
def update_dropdown_options(view_mode):
    if view_mode == 'clustered':
        options = [{'label': f'📱 {stype}', 'value': stype} for stype in sorted(df_sum['Sensor Type'].unique())]
        return options, []
    else:
        options = [{'label': f'🎯 Pixel {px}', 'value': px} for px in sorted(sensor_positions.keys())]
        return options, []


# Update flight info
@app.callback(
    Output('flight-info', 'children'),
    Input('flight-dd', 'value')
)
def update_flight_info(flight):
    if not flight or flight not in flight_analytics:
        return html.Small("No flight data available", className="text-muted")

    stats = flight_analytics[flight]
    coords, flight_stats = load_flight_path_with_analytics(flight)

    return html.Div([
        html.Small([
            html.I(className="fas fa-route me-1"),
            f"Distance: {flight_stats.get('total_distance_km', 0):.1f} km"
        ], className="d-block"),
        html.Small([
            html.I(className="fas fa-crosshairs me-1"),
            f"Pixels: {stats['total_pixels']}"
        ], className="d-block"),
        html.Small([
            html.I(className="fas fa-stopwatch me-1"),
            f"Events: {stats['total_events']}"
        ], className="d-block")
    ])


# Map controls
@app.callback(
    Output('map-controls', 'children'),
    [Input('flight-dd', 'value'),
     Input('selection-dd', 'value')]
)
def update_map_controls(flight, selection):
    if not flight:
        return dbc.Alert("Please select a flight to view map controls", color="info")

    selection_count = len(selection) if selection else 0

    return dbc.Alert([
        html.Span([
            html.I(className="fas fa-info-circle me-2"),
            f"Flight {flight} • {selection_count} items selected"
        ])
    ], color="primary", className="mb-0")


# Enhanced main map callback
@app.callback(
    Output('map', 'children'),
    [Input('flight-dd', 'value'),
     Input('view-mode', 'value'),
     Input('selection-dd', 'value'),
     Input('display-options', 'value')]
)
def update_map(flight, view_mode, selection, display_options):
    if flight is None:
        return [dl.TileLayer(
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attribution="Esri"
        )]

    layers = [dl.TileLayer(
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attribution="Esri"
    )]

    # Enhanced fiber path
    if 'fiber' in (display_options or []):
        layers.append(dl.Polyline(
            positions=fiber_coords,
            color='#FFD700',
            weight=4,
            opacity=0.9,
            dashArray="5, 5"
        ))

    # Enhanced flight track
    if 'flight_path' in (display_options or []):
        try:
            fp = load_flight_path(flight)
            layers.append(dl.Polyline(
                positions=fp,
                color='#00BFFF',
                weight=3,
                opacity=0.7
            ))
        except Exception as e:
            print(f"Error loading flight path: {e}")

    # Main data visualization logic (keeping your existing logic but with enhancements)
    if view_mode == 'individual':
        pixels = selection or []

        # Draw traces with enhanced styling
        for px in pixels:
            key = (flight, px)
            windows = dict_pixel.get(key, [])
            col = pixel_colors.get(px, '#0066CC')

            for window_idx, (coords, meta, snapshot) in enumerate(windows):
                img_uri = get_image_data(snapshot)
                trace_id = f"trace-{px}-{flight}-{window_idx}"
                layers.append(dl.Polyline(
                    id=trace_id,
                    positions=coords,
                    color=col,
                    weight=5,
                    opacity=0.8
                ))

                # Enhanced markers with clustering option
                cluster_markers = 'cluster' in (display_options or [])
                marker_step = 3 if cluster_markers else 10
                marker_size = 2 if cluster_markers else 4

                for i in range(0, len(coords), marker_step):
                    lat, lon = coords[i]
                    m = meta[i]

                    coverage = get_coverage_data(px, flight)
                    coverage_text = f"\n📊 Coverage: {coverage:.1f}%" if coverage is not None else ""

                    # Enhanced popup with better formatting
                    popup_content = html.Div([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6(f"🎯 Pixel {px}", className="text-primary mb-0")
                            ]),
                            dbc.CardBody([
                                html.P([
                                    html.Strong("✈️ Flight: "), f"{flight}", html.Br(),
                                    html.Strong("📱 Type: "), f"{m['Sensor Type']}", html.Br(),
                                    html.Strong("📡 Doppler: "), f"{m['Doppler Type']}", html.Br(),
                                    html.Strong("⏰ Time: "), f"{m['Time']}", html.Br(),
                                    html.Strong("📏 Distance: "), f"{m['distance_to_sensor']:.2f}m",
                                    html.Br(),
                                    html.Strong("🚦 Movement: "), f"{m.get('pixel_movement_type', 'n/a')}",
                                    coverage_text
                                ], className="mb-2 small"),

                                # Image display
                                html.Div(id=f"image-{px}-{window_idx}-{i}")
                            ])
                        ], style={"maxWidth": "350px"})
                    ])

                    if img_uri:
                        popup_content.children[0].children[1].children.append(
                            html.Img(
                                src=img_uri,
                                style={
                                    'width': '100%',
                                    'max-width': '300px',
                                    'height': '200px',
                                    'object-fit': 'cover',
                                    'border-radius': '4px',
                                    'margin-top': '10px'
                                }
                            )
                        )

                    marker_id = f"marker-{px}-{flight}-{window_idx}-{i}"
                    layers.append(dl.CircleMarker(
                        id=marker_id,
                        center=[lat, lon],
                        radius=marker_size,
                        color=col,
                        fill=True,
                        fillColor=col,
                        fillOpacity=0.8,
                        weight=2,
                        children=[dl.Popup(popup_content, maxWidth=400)]
                    ))

        # Enhanced sensor positions
        for px, (lat, lon, stype, min_dist, max_dist) in sensor_positions.items():
            is_selected = px in pixels
            col = pixel_colors.get(px, '#666666')
            radius = 15 if is_selected else 10
            opacity = 1.0 if is_selected else 0.5

            coverage = get_coverage_data(px, flight)
            coverage_text = f"\n📊 Coverage: {coverage:.1f}%" if coverage is not None else ""

            sensor_popup = html.Div([
                dbc.Card([
                    dbc.CardHeader([
                        html.H6(f"🎯 Pixel {px}", className="text-primary mb-0")
                    ]),
                    dbc.CardBody([
                        html.P([
                            html.Strong("📱 Type: "), f"{stype}", html.Br(),
                            html.Strong("📏 Min Distance: "), f"{min_dist:.2f}m", html.Br(),
                            html.Strong("📏 Max Distance: "), f"{max_dist:.2f}m",
                            coverage_text
                        ], className="small mb-0")
                    ])
                ], style={"maxWidth": "300px"})
            ])

            sensor_id = f"sensor-{px}"
            layers.append(dl.CircleMarker(
                id=sensor_id,
                center=[lat, lon],
                radius=radius,
                color=col,
                fill=True,
                fillColor=col,
                fillOpacity=opacity,
                weight=3,
                children=[
                    dl.Tooltip(f"🎯 Pixel {px} ({stype})"),
                    dl.Popup(sensor_popup, maxWidth=350)
                ]
            ))

    else:
        # Clustered by type mode with similar enhancements
        selected_types = selection or []

        for stype in selected_types:
            col = type_colors[stype]
            pixels_of_type = [px for px, (_, _, ptype, _, _) in sensor_positions.items() if ptype == stype]

            # Draw traces for this type
            for px in pixels_of_type:
                key = (flight, px)
                windows = dict_pixel.get(key, [])

                for window_idx, (coords, meta, snapshot) in enumerate(windows):
                    trace_id = f"type-trace-{stype}-{px}-{window_idx}"
                    layers.append(dl.Polyline(
                        id=trace_id,
                        positions=coords,
                        color=col,
                        weight=5,
                        opacity=0.8
                    ))

                    # Enhanced markers for clustered view
                    marker_step = 8 if 'cluster' in (display_options or []) else 15
                    for i in range(0, len(coords), marker_step):
                        lat, lon = coords[i]
                        m = meta[i]

                        type_coverage = get_type_coverage_data(stype, flight)
                        coverage_text = f"\n📊 Type Coverage: {type_coverage:.1f}%" if type_coverage is not None else ""

                        popup_content = html.Div([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H6(f"📱 {stype} - Pixel {px}", className="text-primary mb-0")
                                ]),
                                dbc.CardBody([
                                    html.P([
                                        html.Strong("✈️ Flight: "), f"{flight}", html.Br(),
                                        html.Strong("📡 Doppler: "), f"{m['Doppler Type']}", html.Br(),
                                        html.Strong("⏰ Time: "), f"{m['Time']}", html.Br(),
                                        html.Strong("📏 Distance: "), f"{m['distance_to_sensor']:.2f}m",
                                        html.Br(),
                                        html.Strong("🚦 Movement: "), f"{m.get('pixel_movement_type', 'n/a')}",
                                        coverage_text
                                    ], className="small mb-0")
                                ])
                            ], style={"maxWidth": "350px"})
                        ])

                        marker_id = f"type-marker-{stype}-{px}-{window_idx}-{i}"
                        layers.append(dl.CircleMarker(
                            id=marker_id,
                            center=[lat, lon],
                            radius=4,
                            color=col,
                            fill=True,
                            fillColor=col,
                            fillOpacity=0.8,
                            children=[dl.Popup(popup_content, maxWidth=400)]
                        ))

        # Enhanced sensor positions for clustered view
        for px, (lat, lon, stype, min_dist, max_dist) in sensor_positions.items():
            is_selected_type = stype in selected_types
            col = type_colors[stype] if is_selected_type else '#666666'
            radius = 15 if is_selected_type else 10
            opacity = 1.0 if is_selected_type else 0.4

            type_coverage = get_type_coverage_data(stype, flight)
            coverage_text = f"\n📊 Type Coverage: {type_coverage:.1f}%" if type_coverage is not None else ""

            sensor_popup = html.Div([
                dbc.Card([
                    dbc.CardHeader([
                        html.H6(f"📱 {stype} - Pixel {px}", className="text-primary mb-0")
                    ]),
                    dbc.CardBody([
                        html.P([
                            html.Strong("📏 Min Distance: "), f"{min_dist:.2f}m", html.Br(),
                            html.Strong("📏 Max Distance: "), f"{max_dist:.2f}m",
                            coverage_text
                        ], className="small mb-0")
                    ])
                ], style={"maxWidth": "300px"})
            ])

            sensor_id = f"type-sensor-{px}"
            layers.append(dl.CircleMarker(
                id=sensor_id,
                center=[lat, lon],
                radius=radius,
                color=col,
                fill=True,
                fillColor=col,
                fillOpacity=opacity,
                weight=3,
                children=[
                    dl.Tooltip(f"📱 {stype} - Pixel {px}"),
                    dl.Popup(sensor_popup, maxWidth=350)
                ]
            ))

    return layers


# NEW CALLBACKS FOR 3D AND ANIMATION

@app.callback(
    Output('3d-flight-view', 'figure'),
    [Input('3d-flight-select', 'value'),
     Input('3d-options', 'value')]
)
def update_3d_view(flight, options):
    if not flight:
        fig = go.Figure()
        fig.add_annotation(
            text="Select a flight to view 3D visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(height=700)
        return fig

    try:
        flight_path = os.path.join(trace_dir, f"Flight_{flight}_logs.csv")
        flight_df = pd.read_csv(flight_path)

        flight_df['Flight number'] = flight

        flight_df = calculate_flight_dynamics(flight_df)

        fig = go.Figure()

        if 'speed' in (options or []) and 'speed_smooth' in flight_df.columns:
            fig.add_trace(go.Scattermapbox(
                lon=flight_df['GPS Lon'],
                lat=flight_df['GPS Lat'],
                mode='lines+markers',
                name='Flight Path',
                line=dict(color=flight_df['speed_smooth'], colorscale='Viridis', width=4),
                marker=dict(size=4, color=flight_df['speed_smooth'], colorscale='Viridis', colorbar=dict(title="Speed (m/s)"))
            ))
        else:
            fig.add_trace(go.Scattermapbox(
                lon=flight_df['GPS Lon'],
                lat=flight_df['GPS Lat'],
                mode='lines+markers',
                name='Flight Path',
                line=dict(color='cyan', width=4),
                marker=dict(size=4, color='cyan')
            ))

        if 'detections' in (options or []):
            flight_events = df_sum[df_sum['Flight number'] == flight]
            if not flight_events.empty:
                for stype in flight_events['Sensor Type'].unique():
                    type_events = flight_events[flight_events['Sensor Type'] == stype]
                    fig.add_trace(go.Scattermapbox(
                        lon=type_events['Sensor Lon'],
                        lat=type_events['Sensor Lat'],
                        mode='markers',
                        name=f'Sensor {stype}',
                        marker=dict(size=8, color=type_colors.get(stype, '#FF0000'), symbol='diamond')
                    ))

        center_lat = flight_df['GPS Lat'].mean()
        center_lon = flight_df['GPS Lon'].mean()

        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=12,
                pitch=60,
            ),
            height=700,
            title=f"3D Map - Flight {flight}",
            margin=dict(l=0, r=0, t=40, b=0)
        )

        return fig

    except Exception as e:
        print(f"Error in 3D view: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading 3D view: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(height=700)
        return fig



@app.callback(
    [Output('movement-timeline', 'figure'),
     Output('detection-by-movement', 'figure'),
     Output('movement-statistics', 'figure')],
    Input('movement-flight-select', 'value')
)
def update_movement_analysis(flight):
    if not flight:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Select a flight for movement analysis", xref="paper", yref="paper", x=0.5, y=0.5,
                                 font=dict(size=20))
        empty_fig.update_layout(height=400)
        return empty_fig, empty_fig, empty_fig

    try:
        # Load and analyze flight data
        flight_path = os.path.join(trace_dir, f"Flight_{flight}_logs.csv")
        flight_df = pd.read_csv(flight_path)

        # Add flight number
        flight_df['Flight number'] = flight

        flight_df = calculate_flight_dynamics(flight_df)

        # Movement timeline
        timeline_fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Speed Profile', 'Acceleration', 'Turning Rate'],
            shared_xaxes=True,
            vertical_spacing=0.1
        )

        if 'Time' in flight_df.columns:
            x_axis = pd.to_datetime(flight_df['Time'], errors='coerce')
            if x_axis.isna().all():
                x_axis = pd.date_range(start='2020-01-01', periods=len(flight_df), freq='S')
        else:
            x_axis = pd.date_range(start='2020-01-01', periods=len(flight_df), freq='S')

        # Speed
        if 'speed_smooth' in flight_df.columns:
            timeline_fig.add_trace(
                go.Scatter(x=x_axis, y=flight_df['speed_smooth'], name='Speed', line=dict(color='blue')),
                row=1, col=1
            )

        # Acceleration
        if 'acceleration' in flight_df.columns:
            timeline_fig.add_trace(
                go.Scatter(x=x_axis, y=flight_df['acceleration'], name='Acceleration', line=dict(color='green')),
                row=2, col=1
            )

        # Turning rate
        if 'heading_change' in flight_df.columns:
            timeline_fig.add_trace(
                go.Scatter(x=x_axis, y=flight_df['heading_change'], name='Heading Change', line=dict(color='orange')),
                row=3, col=1
            )

        timeline_fig.update_yaxes(title_text="m/s", row=1, col=1)
        timeline_fig.update_yaxes(title_text="m/s²", row=2, col=1)
        timeline_fig.update_yaxes(title_text="degrees", row=3, col=1)
        timeline_fig.update_layout(height=600, showlegend=False, title=f"Flight Dynamics - Flight {flight}")

        # Detection by movement type
        detection_analysis = analyze_sensor_detection_by_movement(dict_pixel, flight)

        if not detection_analysis.empty:
            detection_fig = go.Figure(data=[
                go.Bar(
                    x=detection_analysis['movement_type'],
                    y=detection_analysis['detection_rate'],
                    text=[f"{rate:.1f}%" for rate in detection_analysis['detection_rate']],
                    textposition='auto',
                    marker_color=['blue', 'green', 'red', 'orange', 'purple'][:len(detection_analysis)]
                )
            ])
            detection_fig.update_layout(
                title="Detection Rate by Movement Type",
                xaxis_title="Movement Type",
                yaxis_title="Detection Rate (%)",
                showlegend=False,
                height=400
            )
        else:
            detection_fig = go.Figure()
            detection_fig.add_annotation(text="No detection data available", xref="paper", yref="paper", x=0.5, y=0.5)
            detection_fig.update_layout(height=400)

        # Movement statistics
        if 'movement_type' in flight_df.columns:
            movement_counts = flight_df['movement_type'].value_counts()

            stats_fig = go.Figure(data=[
                go.Pie(
                    labels=movement_counts.index,
                    values=movement_counts.values,
                    hole=0.3
                )
            ])
            stats_fig.update_layout(
                title="Movement Type Distribution",
                showlegend=True,
                height=400
            )
        else:
            stats_fig = go.Figure()
            stats_fig.add_annotation(text="Movement analysis not available", xref="paper", yref="paper", x=0.5, y=0.5)
            stats_fig.update_layout(height=400)

        return timeline_fig, detection_fig, stats_fig

    except Exception as e:
        print(f"Error in movement analysis: {e}")
        empty_fig = go.Figure()
        empty_fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5,
                                 font=dict(color="red"))
        empty_fig.update_layout(height=400)
        return empty_fig, empty_fig, empty_fig


# Coverage chart callback
@app.callback(
    Output('coverage-chart', 'figure'),
    [Input('flight-dd', 'value'),
     Input('view-mode', 'value'),
     Input('selection-dd', 'value')]
)
def update_coverage_chart(flight, view_mode, selection):
    if not flight or not selection:
        fig = go.Figure()
        fig.add_annotation(
            text="Select items to view coverage analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Coverage Analysis",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white',
            height=400
        )
        return fig

    if view_mode == 'individual' and coverage_per_pixel is not None:
        fig = go.Figure()

        coverage_data = []
        colors_list = []
        labels_list = []

        for px in selection:
            if px in coverage_per_pixel.index:
                flight_col = f"Flight_{flight}"
                if flight_col in coverage_per_pixel.columns:
                    cov = coverage_per_pixel.loc[px, flight_col]
                    if not pd.isna(cov):
                        coverage_data.append(cov)
                        colors_list.append(pixel_colors.get(px, '#0066CC'))
                        labels_list.append(f'Pixel {px}')

        if coverage_data:
            fig.add_trace(go.Bar(
                x=labels_list,
                y=coverage_data,
                marker_color=colors_list,
                name="Coverage"
            ))

            fig.update_layout(
                title=f"Coverage Analysis - Flight {flight}",
                xaxis_title="Pixels",
                yaxis_title="Coverage (%)",
                showlegend=False,
                plot_bgcolor='white',
                height=400
            )

        return fig

    elif view_mode == 'clustered' and coverage_per_type is not None:
        fig = go.Figure()

        coverage_data = []
        colors_list = []
        labels_list = []

        for stype in selection:
            if stype in coverage_per_type.index:
                flight_col = f"Flight_{flight}"
                if flight_col in coverage_per_type.columns:
                    cov = coverage_per_type.loc[stype, flight_col]
                    if not pd.isna(cov):
                        coverage_data.append(cov)
                        colors_list.append(type_colors.get(stype, '#0066CC'))
                        labels_list.append(stype)

        if coverage_data:
            fig.add_trace(go.Bar(
                x=labels_list,
                y=coverage_data,
                marker_color=colors_list,
                name="Coverage"
            ))

            fig.update_layout(
                title=f"Coverage by Sensor Type - Flight {flight}",
                xaxis_title="Sensor Types",
                yaxis_title="Coverage (%)",
                showlegend=False,
                plot_bgcolor='white',
                height=400
            )

        return fig

    # Default empty chart
    fig = go.Figure()
    fig.add_annotation(
        text="Coverage data not available",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        title="Coverage Analysis",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='white',
        height=400
    )
    return fig


@app.callback(
    Output('distance-analysis', 'figure'),
    [Input('flight-dd', 'value'),
     Input('selection-dd', 'value')]
)
def update_distance_analysis(flight, selection):
    if not flight:
        fig = go.Figure()
        fig.add_annotation(
            text="Select a flight to view distance analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Distance Analysis",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white',
            height=400
        )
        return fig

    flight_data = df_sum[df_sum['Flight number'] == flight]

    if selection:
        flight_data = flight_data[flight_data['Pixel'].isin(selection)]

    if not flight_data.empty and 'min_dist3D' in flight_data.columns and 'max_dist3D' in flight_data.columns:
        fig = go.Figure()

        # Add min and max distance traces
        fig.add_trace(go.Scatter(
            x=flight_data['Pixel'],
            y=flight_data['min_dist3D'],
            mode='markers+lines',
            name='Min Distance',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        ))

        fig.add_trace(go.Scatter(
            x=flight_data['Pixel'],
            y=flight_data['max_dist3D'],
            mode='markers+lines',
            name='Max Distance',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ))

        fig.update_layout(
            title=f"Distance Analysis - Flight {flight}",
            xaxis_title="Pixel",
            yaxis_title="Distance (m)",
            plot_bgcolor='white',
            hovermode='x unified',
            height=400
        )

        return fig

    # Empty chart
    fig = go.Figure()
    fig.add_annotation(
        text="No distance data available",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        title="Distance Analysis",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='white',
        height=400
    )
    return fig


@app.callback(
    Output('sensor-distribution', 'figure'),
    Input('flight-dd', 'value')
)
def update_sensor_distribution(flight):
    if not flight:
        fig = go.Figure()
        fig.add_annotation(
            text="Select a flight to view sensor distribution",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Sensor Type Distribution",
            plot_bgcolor='white',
            height=400
        )
        return fig

    flight_data = df_sum[df_sum['Flight number'] == flight]
    sensor_counts = flight_data['Sensor Type'].value_counts()

    colors = [type_colors.get(stype, '#0066CC') for stype in sensor_counts.index]

    fig = go.Figure(data=[
        go.Pie(
            labels=sensor_counts.index,
            values=sensor_counts.values,
            marker_colors=colors,
            textposition='inside',
            textinfo='percent+label'
        )
    ])

    fig.update_layout(
        title=f"Sensor Distribution - Flight {flight}",
        plot_bgcolor='white',
        height=400
    )

    return fig


@app.callback(
    Output('flight-comparison', 'figure'),
    Input('view-mode', 'value')
)
def update_flight_comparison(view_mode):
    if view_mode == 'individual':
        # Compare flights by pixel count
        flight_pixel_counts = df_sum.groupby('Flight number')['Pixel'].nunique().sort_index()

        fig = go.Figure(data=[
            go.Bar(
                x=flight_pixel_counts.index,
                y=flight_pixel_counts.values,
                marker_color='#3498db'
            )
        ])

        fig.update_layout(
            title="Pixels per Flight",
            xaxis_title="Flight Number",
            yaxis_title="Number of Pixels",
            plot_bgcolor='white',
            height=400
        )

    else:
        # Compare flights by sensor type coverage
        flight_type_counts = df_sum.groupby(['Flight number', 'Sensor Type']).size().unstack(fill_value=0)

        fig = go.Figure()
        for stype in flight_type_counts.columns:
            fig.add_trace(go.Bar(
                x=flight_type_counts.index,
                y=flight_type_counts[stype],
                name=stype,
                marker_color=type_colors.get(stype, '#0066CC')
            ))

        fig.update_layout(
            title="Sensor Types per Flight",
            xaxis_title="Flight Number",
            yaxis_title="Number of Events",
            barmode='stack',
            plot_bgcolor='white',
            height=400
        )

    return fig


@app.callback(
    Output('data-table-container', 'children'),
    [Input('flight-dd', 'value'),
     Input('selection-dd', 'value')]
)
def update_data_table(flight, selection):
    if not flight:
        return dbc.Alert("Select a flight to view data table", color="info")

    flight_data = df_sum[df_sum['Flight number'] == flight].copy()

    if selection:
        flight_data = flight_data[flight_data['Pixel'].isin(selection)]

    if flight_data.empty:
        return dbc.Alert("No data available for selected criteria", color="warning")

    # Prepare data for display
    display_columns = ['Pixel', 'Sensor Type']
    if 'Doppler Type' in flight_data.columns:
        display_columns.append('Doppler Type')
    if 'Start time' in flight_data.columns:
        display_columns.append('Start time')
    if 'End time' in flight_data.columns:
        display_columns.append('End time')
    if 'min_dist3D' in flight_data.columns:
        display_columns.append('min_dist3D')
    if 'max_dist3D' in flight_data.columns:
        display_columns.append('max_dist3D')
    if 'Snapshot' in flight_data.columns:
        display_columns.append('Snapshot')

    # Filter to only existing columns
    display_columns = [col for col in display_columns if col in flight_data.columns]

    table_data = flight_data[display_columns].round(2)

    return html.Div([
        html.H5(f"Data Table - Flight {flight}"),
        dash_table.DataTable(
            data=table_data.to_dict('records'),
            columns=[{"name": col, "id": col} for col in display_columns],
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={
                'backgroundColor': '#3498db',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f8f9fa'
                }
            ],
            sort_action="native",
            filter_action="native",
            page_size=20,
            export_format="csv"
        )
    ])


# ----------------------------
# 10) Run Server
# ----------------------------
if __name__ == '__main__':
    print("🚀 Starting Enhanced Flight Dashboard with 3D view...")
    print(f"📊 Loaded {len(df_sum)} events from {len(flight_numbers)} flights")
    print(f"🎯 {len(sensor_positions)} unique pixels")
    print(f"📱 {len(df_sum['Sensor Type'].unique())} sensor types")
    print("✨ New features: 3D visualization and movement analysis")
    print("📡 Dashboard will be available at: http://localhost:8050")
    print("\n⚠️ Make sure you have all required packages installed:")
    print("   pip install scipy geopy")
    print("\n🔍 Look for the tabs above the map to access new features!")
    # Use localhost instead of 0.0.0.0 for local development
    app.run(debug=True, host='127.0.0.1', port=8050)
