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
    SPEED_THRESHOLD = 5.0  # m/s

    # Classify movements
    flight_df.loc[flight_df['acceleration'] > ACCEL_THRESHOLD, 'movement_type'] = 'accelerating'
    flight_df.loc[flight_df['acceleration'] < DECEL_THRESHOLD, 'movement_type'] = 'decelerating'
    flight_df.loc[flight_df['speed_smooth'] < SPEED_THRESHOLD, 'movement_type'] = 'hovering'

    # Add Flight number if not present
    if 'Flight number' not in flight_df.columns:
        flight_df['Flight number'] = flight_df.index[0] if len(flight_df) > 0 else 0

    return flight_df


def calculate_relative_movement_to_pixel(df_flight, sensor_lat, sensor_lon, start_time, end_time):
    """Analyze aircraft motion relative to a pixel within a time window."""
    df = df_flight.copy()
    df['parsed_time'] = pd.to_datetime(df.get('Time'), errors='coerce', infer_datetime_format=True)

    start = pd.to_datetime(start_time)
    end = pd.to_datetime(end_time)
    df = df[(df['parsed_time'] >= start) & (df['parsed_time'] <= end)].reset_index(drop=True)
    if len(df) < 2:
        cols = [
            'time', 'lat', 'lon', 'alt', 'dt', 'dist3d', 'speed', 'delta_speed',
            'heading', 'delta_heading', 'distance_to_sensor', 'delta_distance', 'pixel_movement_type'
        ]
        return pd.DataFrame(columns=cols)

    sensor_alt = df['GPS Alt'].min() if 'GPS Alt' in df.columns else 0

    lat = df['GPS Lat'].to_numpy()
    lon = df['GPS Lon'].to_numpy()
    alt = df['GPS Alt'].fillna(0).to_numpy() if 'GPS Alt' in df.columns else np.zeros(len(df))

    dt = df['parsed_time'].diff().dt.total_seconds().fillna(1).to_numpy()

    dist3d = [0.0]
    heading = [0.0]
    from geopy.distance import geodesic

    for i in range(1, len(df)):
        d2d = geodesic((lat[i-1], lon[i-1]), (lat[i], lon[i])).meters
        dz = alt[i] - alt[i-1]
        dist3d.append(np.sqrt(d2d**2 + dz**2))

        dlon = np.radians(lon[i] - lon[i-1])
        lat1_rad = np.radians(lat[i-1])
        lat2_rad = np.radians(lat[i])
        y = np.sin(dlon) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
        heading.append(np.degrees(np.arctan2(y, x)))

    speed = np.array(dist3d) / np.where(dt == 0, 1, dt)
    delta_speed = np.insert(np.diff(speed), 0, 0)

    heading = np.array(heading)
    delta_heading = np.insert(np.diff(heading), 0, 0)
    delta_heading = (delta_heading + 180) % 360 - 180

    dist_to_sensor = []
    for la, lo, al in zip(lat, lon, alt):
        d2d = geodesic((la, lo), (sensor_lat, sensor_lon)).meters
        dist_to_sensor.append(np.sqrt(d2d**2 + (al - sensor_alt)**2))
    dist_to_sensor = np.array(dist_to_sensor)
    delta_dist = np.insert(np.diff(dist_to_sensor), 0, 0)

    movement = []
    for i in range(len(df)):
        mv = []
        if abs(delta_heading[i]) > 15:
            mv.append('turning')
        if delta_dist[i] < -1:
            mv.append('approaching')
        elif delta_dist[i] > 1:
            mv.append('departing')
        if delta_speed[i] > 2:
            mv.append('accelerating')
        elif delta_speed[i] < -2:
            mv.append('decelerating')
        if speed[i] < 1:
            mv.append('hovering')
        if not mv:
            mv = ['cruising']
        movement.append(', '.join(mv))

    result = pd.DataFrame({
        'time': df['parsed_time'],
        'lat': lat,
        'lon': lon,
        'alt': alt,
        'dt': dt,
        'dist3d': dist3d,
        'speed': speed,
        'delta_speed': delta_speed,
        'heading': heading,
        'delta_heading': delta_heading,
        'distance_to_sensor': dist_to_sensor,
        'delta_distance': delta_dist,
        'pixel_movement_type': movement
    })

    extra_cols = [c for c in ['Sensor Type', 'Doppler Type', 'Time'] if c in df.columns]
    return pd.concat([df[extra_cols].reset_index(drop=True), result], axis=1)


def analyze_sensor_detection_by_movement(pixel_dict, flight, pixels=None, sensor_types=None):
    """Analyze movement types relative to pixel detections for a flight.
    Optionally filter by specific pixels or sensor types."""
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
        if pixels is not None and px not in pixels:
            continue
        for coords, meta, _ in windows:
            if sensor_types is not None and meta:
                sensor_type = meta[0].get('Sensor Type')
                if sensor_type not in sensor_types:
                    continue
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


PIXEL_DATA = {}

def preload_pixel_csvs():
    """Preload all pixel CSV files for faster access."""
    summary_dir = os.path.dirname(summary_csv)
    for px in sorted(df_sum['Pixel'].unique()):
        fpath = os.path.join(summary_dir, f"points_pixel_{px}.csv")
        if os.path.exists(fpath):
            PIXEL_DATA[px] = pd.read_csv(fpath)
        else:
            PIXEL_DATA[px] = None
            print(f"\u26A0\uFE0F Missing pixel CSV for pixel {px}")

preload_pixel_csvs()

def load_pixel_csv(px):
    return PIXEL_DATA.get(px)


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


def load_graph_image(pixel_id, flight_num, graphs_dir):
    """Return base64 image for a pixel's donut graph if available."""
    if not graphs_dir:
        return None
    fname = f"donut_px_{pixel_id}_fl_{flight_num}.png"
    paths = [
        os.path.join(graphs_dir, fname),
        os.path.join(graphs_dir, 'GRAPHS', fname)
    ]
    for p in paths:
        if os.path.exists(p):
            return get_image_data(p)
    return None


def load_type_coverage_image(detection_type, flight_num, graphs_dir):
    """Return base64 image for a sensor type coverage graph if available."""
    if not graphs_dir:
        return None
    fname = f"coverage_type_{detection_type}_fl_{flight_num}.png"
    paths = [
        os.path.join(graphs_dir, fname),
        os.path.join(graphs_dir, 'GRAPHS', fname)
    ]
    for p in paths:
        if os.path.exists(p):
            return get_image_data(p)
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

# Color map for movement classifications
MOVEMENT_COLORS = {
    'approaching': '#2ecc71',  # green
    'departing': '#e74c3c',   # red
    'hovering': '#9b59b6',    # purple
    'accelerating': '#f39c12',  # orange
    'decelerating': '#3498db',  # blue
    'turning': '#8e44ad',       # violet
    'cruising': '#95a5a6'      # gray
}

def generate_movement_legend():
    """Return an HTML legend explaining movement colors."""
    items = []
    for name, col in MOVEMENT_COLORS.items():
        items.append(
            html.Span(
                [html.Span(style={'display': 'inline-block',
                                 'width': '12px',
                                 'height': '12px',
                                 'backgroundColor': col,
                                 'marginRight': '6px'}),
                 name.capitalize()],
                className="me-3"
            )
        )
    return html.Div(items, className="small")

movement_legend = generate_movement_legend()

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
                    ),
                    dbc.Button(
                        "Clear All",
                        id="clear-selection",
                        color="secondary",
                        size="sm",
                        className="mt-2"
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
                            {'label': ' Cluster Markers', 'value': 'cluster'},
                            {'label': ' Movement Colors', 'value': 'movement_colors'}
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
    Output('btn-data', 'active'),
    Output('btn-settings', 'active'),
    Output('btn-map', 'color'),
    Output('btn-3d', 'color'),
    Output('btn-movement', 'color'),
    Output('btn-data', 'color'),
    Output('btn-settings', 'color')],
    [Input('btn-map', 'n_clicks'),
     Input('btn-3d', 'n_clicks'),
     Input('btn-movement', 'n_clicks'),
     Input('btn-data', 'n_clicks'),
     Input('btn-settings', 'n_clicks')]
)
def render_content_from_buttons(btn_map, btn_3d, btn_movement, btn_data, btn_settings):
    # Determine which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        active_tab = "map"
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        active_tab = button_id.split('-')[1]

    # Set active states
    active_states = [False] * 5
    colors = ['secondary'] * 5

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
            ),
            html.Div(id='movement-legend', className="mt-2")
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
                            {'label': ' Color by Speed', 'value': 'speed'},
                            {'label': ' Show Pixel Traces', 'value': 'pixel_traces'}
                        ],
                        value=['detections'],
                        inline=True
                    )
                ], md=5),
                dbc.Col([
                    html.Label("Select Pixels:", className="fw-bold"),
                    dcc.Dropdown(id="3d-pixel-select", multi=True)
                ], md=3)
            ], className="mb-3"),
            dcc.Graph(id='3d-flight-view', style={'height': '70vh'})
        ])

    elif active_tab == "movement":
        active_states[2] = True
        colors[2] = 'primary'

        content = html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Select Flight:", className="fw-bold"),
                    dcc.Dropdown(
                        id='movement-flight-select',
                        options=[{'label': f'Flight {f}', 'value': f} for f in flight_numbers],
                        value=flight_numbers[0] if flight_numbers else None
                    )
                ], md=4),
                dbc.Col([
                    html.Label("View Mode:", className="fw-bold"),
                    dcc.Dropdown(
                        id='movement-view-mode',
                        options=[{'label': 'By Pixel', 'value': 'individual'}, {'label': 'By Type', 'value': 'clustered'}],
                        value='individual'
                    )
                ], md=3),
                dbc.Col([
                    html.Label("Select Item:", className="fw-bold"),
                    dcc.Dropdown(id='movement-selection', multi=True)
                ], md=5)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([dcc.Graph(id='movement-coverage', style={'height': '400px'})], md=6),
                dbc.Col([dcc.Graph(id='detection-by-movement', style={'height': '400px'})], md=6)
            ]),
            html.Div(id='movement-images', className='mt-3')
        ])
    elif active_tab == "data":
        active_states[3] = True
        colors[3] = 'primary'
        content = html.Div([
            html.Div(id="data-table-container")
        ])

    elif active_tab == "settings":
        active_states[4] = True
        colors[4] = 'primary'
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


# Clear selection button
@app.callback(
    Output('selection-dd', 'value', allow_duplicate=True),
    Input('clear-selection', 'n_clicks'),
    prevent_initial_call=True
)
def clear_selection(n):
    return []


# Dropdown options for movement analysis
@app.callback([
    Output('movement-selection', 'options'),
    Output('movement-selection', 'value')],
    Input('movement-view-mode', 'value')
)
def update_movement_dropdown(view_mode):
    if view_mode == 'clustered':
        opts = [{'label': stype, 'value': stype} for stype in sorted(df_sum['Sensor Type'].unique())]
        return opts, []
    else:
        opts = [{'label': f'Pixel {px}', 'value': px} for px in sorted(sensor_positions.keys())]
        return opts, []


# Pixel selection for 3D view
@app.callback([
    Output('3d-pixel-select', 'options'),
    Output('3d-pixel-select', 'value')],
    Input('3d-flight-select', 'value'))
def update_3d_pixel_dropdown(flight):
    if not flight:
        return [], []
    pixels = sorted(df_sum[df_sum['Flight number'] == flight]['Pixel'].unique())
    opts = [{'label': f'Pixel {p}', 'value': p} for p in pixels]
    return opts, pixels


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
                    if 'movement_colors' in (display_options or []):
                        mv_raw = m.get('pixel_movement_type', 'cruising')
                        mv_key = str(mv_raw).split(',')[0].lower()
                        mv_col = MOVEMENT_COLORS.get(mv_key, col)
                    else:
                        mv_col = col
                    layers.append(dl.CircleMarker(
                        id=marker_id,
                        center=[lat, lon],
                        radius=marker_size,
                        color=mv_col,
                        fill=True,
                        fillColor=mv_col,
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
                        if 'movement_colors' in (display_options or []):
                            mv_raw = m.get('pixel_movement_type', 'cruising')
                            mv_key = str(mv_raw).split(',')[0].lower()
                            mv_col = MOVEMENT_COLORS.get(mv_key, col)
                        else:
                            mv_col = col
                        layers.append(dl.CircleMarker(
                            id=marker_id,
                            center=[lat, lon],
                            radius=4,
                            color=mv_col,
                            fill=True,
                            fillColor=mv_col,
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


# Toggle movement legend based on display options
@app.callback(
    Output('movement-legend', 'children'),
    Input('display-options', 'value')
)
def update_movement_legend(display_options):
    if display_options and 'movement_colors' in display_options:
        return movement_legend
    return ""

# NEW CALLBACKS FOR 3D AND ANIMATION

@app.callback(
    Output('3d-flight-view', 'figure'),
    [Input('3d-flight-select', 'value'),
     Input('3d-options', 'value'),
     Input('3d-pixel-select', 'value')]
)
def update_3d_view(flight, options, pixel_selection):
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
            marker = dict(size=3, color=flight_df['speed_smooth'], colorscale='Viridis', colorbar=dict(title='Speed (m/s)'))
            line = dict(color='gray', width=2)
        else:
            marker = dict(size=3, color='cyan')
            line = dict(color='cyan', width=2)

        fig.add_trace(go.Scatter3d(
            x=flight_df['GPS Lon'],
            y=flight_df['GPS Lat'],
            z=flight_df['GPS Alt'],
            mode='lines+markers',
            name='Flight Path',
            line=line,
            marker=marker
        ))

        min_alt = flight_df['GPS Alt'].min() if 'GPS Alt' in flight_df.columns else 0

        if 'detections' in (options or []):
            flight_events = df_sum[df_sum['Flight number'] == flight]
            if not flight_events.empty:
                for stype in flight_events['Sensor Type'].unique():
                    type_events = flight_events[flight_events['Sensor Type'] == stype]
                    fig.add_trace(go.Scatter3d(
                        x=type_events['Sensor Lon'],
                        y=type_events['Sensor Lat'],
                        z=[min_alt] * len(type_events),
                        mode='markers',
                        name=f'Sensor {stype}',
                        marker=dict(size=8, color=type_colors.get(stype, '#FF0000'), symbol='diamond')
                    ))

        if 'pixel_traces' in (options or []):
            if pixel_selection:
                pixels_to_show = pixel_selection
            else:
                pixels_to_show = sorted(df_sum[df_sum['Flight number'] == flight]['Pixel'].unique())
            for px in pixels_to_show:
                windows = dict_pixel.get((flight, px), [])
                col = pixel_colors.get(px, '#0066CC')
                for window_idx, (coords, meta, _snap) in enumerate(windows):
                    lons = [c[1] for c in coords]
                    lats = [c[0] for c in coords]
                    alts = [m['alt'] for m in meta]
                    fig.add_trace(go.Scatter3d(
                        x=lons,
                        y=lats,
                        z=alts,
                        mode='lines',
                        line=dict(color=col, width=4),
                        name=f'Pixel {px} Trace'
                    ))
                    colors = [MOVEMENT_COLORS.get(str(m.get('pixel_movement_type','cruising')).split(',')[0].lower(), col) for m in meta]
                    fig.add_trace(go.Scatter3d(
                        x=lons,
                        y=lats,
                        z=alts,
                        mode='markers',
                        marker=dict(size=3, color=colors),
                        name=f'Pixel {px} Movement',
                        showlegend=False
                    ))

        fig.update_layout(
            title=f"3D Flight Path - Flight {flight}",
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Altitude (m)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=700
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
    [Output('movement-coverage', 'figure'),
     Output('detection-by-movement', 'figure')],
    [Input('movement-flight-select', 'value'),
     Input('movement-view-mode', 'value'),
     Input('movement-selection', 'value')]
)
def update_movement_analysis(flight, view_mode, selection):
    if not flight or not selection:
        fig = go.Figure()
        fig.add_annotation(text="Select items for analysis", xref="paper", yref="paper", x=0.5, y=0.5)
        fig.update_layout(height=400)
        return fig, fig

    flight_col = f"Flight_{flight}"

    coverage_vals = []
    labels = []
    colors_list = []

    if view_mode == 'individual' and coverage_per_pixel is not None:
        for px in selection:
            if px in coverage_per_pixel.index and flight_col in coverage_per_pixel.columns:
                cov = coverage_per_pixel.loc[px, flight_col]
                if not pd.isna(cov):
                    coverage_vals.append(cov)
                    labels.append(f'Pixel {px}')
                    colors_list.append(pixel_colors.get(px, '#0066CC'))
        detection_df = analyze_sensor_detection_by_movement(dict_pixel, flight, pixels=set(selection))
    elif view_mode == 'clustered' and coverage_per_type is not None:
        for stype in selection:
            if stype in coverage_per_type.index and flight_col in coverage_per_type.columns:
                cov = coverage_per_type.loc[stype, flight_col]
                if not pd.isna(cov):
                    coverage_vals.append(cov)
                    labels.append(stype)
                    colors_list.append(type_colors.get(stype, '#0066CC'))
        detection_df = analyze_sensor_detection_by_movement(dict_pixel, flight, sensor_types=set(selection))
    else:
        detection_df = pd.DataFrame()

    cov_fig = go.Figure()
    if coverage_vals:
        cov_fig.add_trace(go.Bar(x=labels, y=coverage_vals, marker_color=colors_list))
        cov_fig.update_layout(title="Coverage", yaxis_title="Coverage (%)", showlegend=False, height=400)
    else:
        cov_fig.add_annotation(text="Coverage data not available", xref="paper", yref="paper", x=0.5, y=0.5)
        cov_fig.update_layout(height=400)

    det_fig = go.Figure()
    if not detection_df.empty:
        det_fig.add_trace(go.Bar(x=detection_df['movement_type'], y=detection_df['detected_points'], marker_color=[MOVEMENT_COLORS.get(mv, '#888') for mv in detection_df['movement_type']]))
        det_fig.update_layout(title="Detections by Movement", xaxis_title="Movement Type", yaxis_title="Count", showlegend=False, height=400)
    else:
        det_fig.add_annotation(text="No detection data", xref="paper", yref="paper", x=0.5, y=0.5)
        det_fig.update_layout(height=400)

    return cov_fig, det_fig


@app.callback(
    Output('movement-images', 'children'),
    [Input('movement-flight-select', 'value'),
     Input('movement-view-mode', 'value'),
     Input('movement-selection', 'value')]
)
def update_movement_images(flight, view_mode, selection):
    if not flight or not selection or not graphs_dir:
        return ""

    imgs = []
    if view_mode == 'individual':
        for px in selection:
            uri = load_graph_image(px, flight, graphs_dir)
            if uri:
                imgs.append(html.Img(src=uri, style={'maxWidth': '100%', 'height': 'auto'}, className='mb-3'))
    else:
        for stype in selection:
            uri = load_type_coverage_image(stype, flight, graphs_dir)
            if uri:
                imgs.append(html.Img(src=uri, style={'maxWidth': '100%', 'height': 'auto'}, className='mb-3'))

    if not imgs:
        return html.Small("No graphs available", className="text-muted")
    return imgs


# Coverage chart callback


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
