# -*- coding: utf-8 -*-
"""
combined_event_analysis.py — Full pipeline: 1) shifted-event summary + per-pixel extraction,
2) coverage statistics & plots (including per-Type union charts).
"""
from __future__ import annotations
import os
import math
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from geopy.distance import geodesic
import tkinter as tk
from tkinter import filedialog

# ---------------------- STYLE ------------------------------------------
LIGHT_BLUE = "#73C2FB"
LIGHT_GREY = "#D3D3D3"
TEXT_COL = "#262626"
FONT_FAM = "Century Gothic"
mpl.rcParams.update({
    "font.family": FONT_FAM,
    "text.color": TEXT_COL,
    "axes.labelcolor": TEXT_COL,
    "xtick.color": TEXT_COL,
    "ytick.color": TEXT_COL,
    "axes.edgecolor": LIGHT_GREY,
})

# ---------------------- HELPERS ----------------------------------------
def _safe_save(fig: plt.Figure, path: Path):
    try:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = path.with_name(f"{path.stem}_{ts}{path.suffix}")
        fig.savefig(alt, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)

def haversine(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in meters."""
    R = 6_371_000
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def compute_coverage_by_track(coords: pd.Series, total_track: float, max_gap_m: float = 50.0) -> float:
    if coords.empty or total_track <= 0:
        return 0.0
    gaps = coords.diff() > max_gap_m
    clusters = coords.groupby(gaps.cumsum()).agg(start="min", end="max")
    covered = (clusters.end - clusters.start).sum()
    return 100.0 * covered / total_track

def build_track_dist(df_trace: pd.DataFrame) -> dict[int, pd.DataFrame]:
    tracks: dict[int, pd.DataFrame] = {}
    for fl, grp in df_trace.groupby('Flight number'):
        df = grp.dropna(subset=['real time']).sort_values('real time').reset_index(drop=True)
        lats = df['GPS Lat'].to_numpy(); lons = df['GPS Lon'].to_numpy()
        segs = [0.0] + [
            haversine(lats[i-1], lons[i-1], lats[i], lons[i])
            for i in range(1, len(df))
        ]
        df['track_dist_m'] = pd.Series(segs, index=df.index).cumsum()
        tracks[fl] = df[['real time','track_dist_m']].rename(columns={'real time':'dt'})
    return tracks

# ----------------------------------------------------------------------
# Additional analytics for each extracted window
def _movement_metrics(df: pd.DataFrame, sensor_lat: float, sensor_lon: float) -> pd.DataFrame:
    """Compute relative movement metrics for a flight window."""
    df = df.copy()
    df['parsed_time'] = pd.to_datetime(df['real time'], errors='coerce')
    if len(df) < 2:
        df['time'] = df['parsed_time']
        df['lat'] = df['GPS Lat']
        df['lon'] = df['GPS Lon']
        df['alt'] = df.get('GPS Alt', 0)
        for col in ['dt','dist3d','speed','delta_speed','heading','delta_heading','distance_to_sensor','delta_distance','pixel_movement_type']:
            df[col] = 0.0 if 'pixel_movement_type' not in col else 'cruising'
        return df

    lat = df['GPS Lat'].to_numpy()
    lon = df['GPS Lon'].to_numpy()
    alt = df.get('GPS Alt', pd.Series(0, index=df.index)).fillna(0).to_numpy()

    dt = df['parsed_time'].diff().dt.total_seconds().fillna(1).to_numpy()

    lat1 = lat[:-1]; lon1 = lon[:-1]
    lat2 = lat[1:]; lon2 = lon[1:]
    d2d = haversine(lat1, lon1, lat2, lon2)
    dz = alt[1:] - alt[:-1]
    dist3d = np.concatenate(([0.0], np.sqrt(d2d**2 + dz**2)))

    dlon = np.radians(lon2 - lon1)
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    y = np.sin(dlon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    heading = np.concatenate(([0.0], np.degrees(np.arctan2(y, x))))

    speed = dist3d / np.where(dt == 0, 1, dt)
    delta_speed = np.insert(np.diff(speed), 0, 0)

    delta_heading = np.insert(np.diff(heading), 0, 0)
    delta_heading = (delta_heading + 180) % 360 - 180

    sensor_alt = df['GPS Alt'].min() if 'GPS Alt' in df.columns else 0
    d2_sensor = haversine(lat, lon, sensor_lat, sensor_lon)
    dist_to_sensor = np.sqrt(d2_sensor**2 + (alt - sensor_alt) ** 2)
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

    df['time'] = df['parsed_time']
    df['lat'] = lat
    df['lon'] = lon
    df['alt'] = alt
    df['dt'] = dt
    df['dist3d'] = dist3d
    df['speed'] = speed
    df['delta_speed'] = delta_speed
    df['heading'] = heading
    df['delta_heading'] = delta_heading
    df['distance_to_sensor'] = dist_to_sensor
    df['delta_distance'] = delta_dist
    df['pixel_movement_type'] = movement

    return df

# ---------------------- MAIN -------------------------------------------
def main():
    tk.Tk().withdraw()

    # 1) Fiber config
    fiber_csv = filedialog.askopenfilename(
        title="Select Liman Configuration CSV",
        filetypes=[("CSV files","*.csv")]
    )
    df_fiber = pd.read_csv(fiber_csv)
    df_fiber.rename(columns={'Location (pxl)':'Pixel'}, inplace=True)

    # 2) Events file
    ev_file = filedialog.askopenfilename(
        title="Select Event Queries file",
        filetypes=[("CSV","*.csv"),("Excel","*.xls;*.xlsx")]
    )
    ext = Path(ev_file).suffix.lower()
    df_events = pd.read_csv(ev_file) if ext=='.csv' else pd.read_excel(ev_file)
    df_events.rename(
        columns={'Type':'Doppler Type','Start':'Start time','End':'End time'},
        inplace=True
    )
    for c in ['Start time','End time']:
        df_events[c] = pd.to_datetime(
            df_events[c].astype(str)
                        .str.replace(r"(\d{2}:\d{2}:\d{2}):(\d+)$", r"\1.\2", regex=True),
            errors='coerce'
        ).dt.time
    df_events['Dof'] = df_events['Pixel'] * 2.042739

    # 3) Flight logs & output dir for summary
    trace_dir = filedialog.askdirectory(title="Select Flight logs folder")
    out_dir = filedialog.askdirectory(title="Select output directory for summary & per-pixel CSVs")
    os.makedirs(out_dir, exist_ok=True)

    # --- SHIFTED SUMMARY & PER-PIXEL CSVs ---
    logs: list[pd.DataFrame] = []
    for fn in sorted(os.listdir(trace_dir)):
        if fn.lower().startswith("flight_") and fn.lower().endswith("_logs.csv"):
            fl = int(fn.split("_")[1])
            df = pd.read_csv(os.path.join(trace_dir, fn))
            df['real time'] = (
                pd.to_datetime(df['Israel Time'], errors='coerce')
                  .dt.tz_convert('Asia/Jerusalem').dt.tz_localize(None)
            )
            df.dropna(subset=['real time'], inplace=True)
            df['GPS Lat'] = df['GPS[0].Lat'] / 1e7
            df['GPS Lon'] = df['GPS[0].Lng'] / 1e7
            df['GPS Alt'] = df['GPS[0].Alt']
            df['Flight number'] = fl
            logs.append(df)
            df.to_csv(os.path.join(trace_dir, fn), index=False)
    df_trace = pd.concat(logs, ignore_index=True)
    df_trace.dropna(subset=['real time'], inplace=True)
    df_trace.sort_values(['Flight number','real time'], inplace=True)
    df_trace.reset_index(drop=True, inplace=True)
    base_date = df_trace['real time'].dt.date.iloc[0]

    # Event summary
    summary: list[dict] = []
    times = df_trace['real time']
    SPEED = 343
    for _, ev in df_events.iterrows():
        st = datetime.combine(base_date, ev['Start time'])
        et = datetime.combine(base_date, ev['End time'])
        px = ev['Pixel']; dop = ev['Doppler Type']; dof = ev['Dof']; snap = ev.get('Snapshot','')
        si = (df_fiber['Pixel'] - px).abs().idxmin()
        r = df_fiber.loc[si]
        sensor_lat, sensor_lon = r['Latitude'], r['Longitude']
        sensor_type = r['Type']
        mid = st + (et - st)/2
        sid = (times - mid).abs().idxmin(); samp = df_trace.loc[sid]
        elat, elon, ealt = samp['GPS Lat'], samp['GPS Lon'], samp['GPS Alt']
        d2 = geodesic((elat,elon),(sensor_lat,sensor_lon)).km * 1000
        d3 = math.hypot(d2, ealt)
        delay = timedelta(seconds=d3/SPEED)
        cs, ce = st - delay, et - delay
        window = df_trace[(times>=cs)&(times<=ce)]
        if window.empty: continue
        dists = window.apply(
            lambda r: math.hypot(
                geodesic((r['GPS Lat'],r['GPS Lon']), (sensor_lat,sensor_lon)).km*1000,
                r['GPS Alt']
            ), axis=1
        )
        summary.append({
            'Flight number': samp['Flight number'],
            'Pixel': px,
            'Doppler Type': dop,
            'Dof': dof,
            'Sensor Type': sensor_type,
            'Sensor Lat': sensor_lat,
            'Sensor Lon': sensor_lon,
            'Event Lat': elat,
            'Event Lon': elon,
            'Event Alt': ealt,
            'Start time': st.strftime("%H:%M:%S"),
            'End time': et.strftime("%H:%M:%S"),
            'min_dist3D': round(dists.min(),2),
            'max_dist3D': round(dists.max(),2),
            'Snapshot': snap
        })
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(os.path.join(out_dir, "event_summary_shifted_3D.csv"), index=False)

    # Per-pixel extraction
    for px in df_summary['Pixel'].unique():
        rows: list[pd.DataFrame] = []
        for _, r in df_summary[df_summary['Pixel']==px].iterrows():
            st = datetime.strptime(r['Start time'], "%H:%M:%S").replace(
                year=base_date.year, month=base_date.month, day=base_date.day)
            et = datetime.strptime(r['End time'], "%H:%M:%S").replace(
                year=base_date.year, month=base_date.month, day=base_date.day)
            w = df_trace[(times>=st)&(times<=et)].copy()
            if w.empty: continue
            for fld in ['Flight number','Doppler Type','Dof','Sensor Type','Pixel','Snapshot']:
                w[fld] = r[fld]
            w['Time'] = w['real time'].dt.strftime("%H:%M:%S")
            w['Sensor Lat'] = r['Sensor Lat']
            w['Sensor Lon'] = r['Sensor Lon']
            w['dist3D'] = w.apply(lambda row: math.hypot(
                geodesic((row['GPS Lat'],row['GPS Lon']), (row['Sensor Lat'],row['Sensor Lon'])).km*1000,
                row['GPS Alt']
            ), axis=1)
            w = _movement_metrics(w, r['Sensor Lat'], r['Sensor Lon'])
            rows.append(w)
        if rows:
            out = pd.concat(rows, ignore_index=True)
            out.to_csv(os.path.join(out_dir, f"points_pixel_{px}.csv"), index=False)
    print("✓ Shifted summary & per-pixel CSVs written to", out_dir)

    # ------------------ STATISTICS & PLOTS -----------------------------
    graphs_dir = Path(out_dir)/"Graphs and statistics"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Map pixel → Sensor Type from summary
    pixel_to_type = df_summary.set_index('Pixel')['Sensor Type'].to_dict()

    # Discover per-pixel CSVs
    pixel_files = list(Path(out_dir).glob("points_pixel_*.csv"))
    pixels = sorted(int(p.stem.split("_")[-1]) for p in pixel_files)

    # Build track distances
    flight_logs = build_track_dist(df_trace)

    # Prepare records and unions
    records: list[dict] = []
    type_union: dict[tuple[str,int],list[float]] = {}
    all_union: dict[int,list[float]] = {}

    for pfile in pixel_files:
        df_px = pd.read_csv(pfile)
        px = int(Path(pfile).stem.split("_")[-1])
        df_px['dt'] = pd.to_datetime(df_px['real time'], errors='coerce')
        for fl, grp in df_px.groupby('Flight number'):
            log = flight_logs.get(fl)
            if log is None: continue
            total = float(log['track_dist_m'].max())
            merged = pd.merge_asof(
                grp.sort_values('dt'), log.sort_values('dt'),
                on='dt', direction='nearest', tolerance=pd.Timedelta(seconds=1)
            ).dropna(subset=['track_dist_m'])
            coords = merged['track_dist_m'].tolist()

            # Deduplicate for true union coverage
            coords_ser = pd.Series(coords, dtype=float).drop_duplicates().sort_values()
            pct = compute_coverage_by_track(coords_ser, total)
            records.append({'Pixel':px,'Flight':fl,'Coverage':pct})

            t = pixel_to_type.get(px,'Unknown')
            type_union.setdefault((t,fl),[]).extend(coords)
            all_union.setdefault(fl,[]).extend(coords)

            # Per-pixel donut
            fig,ax = plt.subplots(figsize=(3,3))
            ax.pie([pct,100-pct], colors=[LIGHT_BLUE,LIGHT_GREY],
                   startangle=90,counterclock=False,
                   wedgeprops=dict(width=0.4,edgecolor=LIGHT_GREY))
            ax.text(0,0,f"{pct:.1f}%",ha='center',va='center',fontsize=16)
            ax.set_title(f"Pixel {px} – Flight {int(fl)} Coverage")
            ax.axis('equal')
            _safe_save(fig,graphs_dir/f"donut_px_{px}_fl_{fl}.png")

            # Distance histogram
            dist_vals = grp['dist3D'].dropna()
            if not dist_vals.empty:
                max_d = dist_vals.max()
                bins = list(range(0, int(max_d)+201, 100))
                fig,ax = plt.subplots(figsize=(4,2.7))
                ax.hist(dist_vals, bins=bins, color=LIGHT_BLUE, edgecolor=LIGHT_GREY)
                ax.set_xlim(0, max_d+200)
                ax.set_title(f"Pixel {px} – Flight {int(fl)} Distance Distribution")
                ax.set_xlabel("Distance [m]"); ax.set_ylabel("Count")
                _safe_save(fig,graphs_dir/f"hist_pixel_{px}_flight_{fl}.png")

    cov_df = pd.DataFrame(records)

    # Per-pixel coverage CSV
    cov_df.pivot(index='Pixel',columns='Flight',values='Coverage')\
          .fillna(0.0)\
          .to_csv(graphs_dir/"coverage_per_pixel.csv",index_label="Pixel")

    # Per-Type coverage CSV
    rows = []
    for (t,fl),coords in type_union.items():
        total = float(flight_logs[fl]['track_dist_m'].max())
        coords_ser = pd.Series(coords,dtype=float).drop_duplicates().sort_values()
        pct = compute_coverage_by_track(coords_ser,total)
        rows.append({'Type':t,'Flight':int(fl),'Coverage (%)':pct})
    pd.DataFrame(rows)\
      .pivot(index='Type',columns='Flight',values='Coverage (%)')\
      .fillna(0.0)\
      .to_csv(graphs_dir/"coverage_per_type.csv",index_label="Type")

    # All-Pixels union coverage CSV
    rows = []
    for fl,coords in all_union.items():
        total = float(flight_logs[fl]['track_dist_m'].max())
        coords_ser = pd.Series(coords,dtype=float).drop_duplicates().sort_values()
        pct = compute_coverage_by_track(coords_ser,total)
        rows.append({'Group':'AllPixels','Flight':int(fl),'Coverage (%)':pct})
    pd.DataFrame(rows)\
      .pivot(index='Group',columns='Flight',values='Coverage (%)')\
      .to_csv(graphs_dir/"coverage_union_allpixels.csv",index_label="Group")

    # Average coverage per pixel bar
    avg = cov_df.groupby('Pixel')['Coverage'].mean().reindex(pixels)
    fig,ax = plt.subplots(figsize=(8,3))
    bars = ax.bar(avg.index.astype(str), avg.values, color=LIGHT_BLUE, edgecolor=LIGHT_GREY)
    ax.set_title("Average Coverage per Pixel")
    ax.set_xlabel("Pixel"); ax.set_ylabel("Coverage %")
    ax.set_ylim(0, (avg.max()*1.25) if not avg.empty else 1)
    for b in bars:
        h=b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h+1, f"{h:.1f}%", ha='center', va='bottom')
    _safe_save(fig, graphs_dir/"avg_coverage_per_pixel.png")

    # Per-Type union vs individuals per flight
    seen = set()
    for (t, fl), coords in type_union.items():
        if (t, fl) in seen:
            continue
        seen.add((t, fl))

        total = float(flight_logs[fl]['track_dist_m'].max())
        coords_ser = pd.Series(coords, dtype=float).drop_duplicates().sort_values()
        pct_u = compute_coverage_by_track(coords_ser, total)

        # Individual per-pixel coverage for this type & flight
        indiv = cov_df[(cov_df['Flight'] == fl) &
                       (cov_df['Pixel'].map(pixel_to_type) == t)]

        labels = ['Union'] + indiv['Pixel'].astype(str).tolist()
        vals   = [pct_u] + indiv['Coverage'].tolist()

        fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(labels)), 3))
        bars = ax.bar(labels, vals, color=LIGHT_BLUE, edgecolor=LIGHT_GREY)

        ax.set_title(f"Type {t} – Flight {fl} Coverage")
        ax.set_ylabel("Coverage %")
        ax.set_ylim(0, max(vals) * 1.25)

        # Centered percentage labels inside each bar
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h / 2,
                f"{h:.1f}%",
                ha='center', va='center',
                color=TEXT_COL
            )

        _safe_save(fig, graphs_dir / f"coverage_type_{t}_fl_{fl}.png")

    # All-Pixels union vs individuals per flight
    for fl,coords in all_union.items():
        total = float(flight_logs[fl]['track_dist_m'].max())
        coords_ser = pd.Series(coords,dtype=float).drop_duplicates().sort_values()
        pct_u = compute_coverage_by_track(coords_ser,total)
        indiv = cov_df[cov_df['Flight']==fl]
        labels=['Union']+indiv['Pixel'].astype(str).tolist()
        vals=[pct_u]+indiv['Coverage'].tolist()
        fig,ax = plt.subplots(figsize=(max(5,0.5*len(labels)),3))
        bars=ax.bar(labels,vals,color=LIGHT_BLUE,edgecolor=LIGHT_GREY)
        ax.set_title(f"All Pixels – Flight {int(fl)}")
        ax.set_ylabel("Coverage %")
        ax.set_ylim(0,max(vals)*1.25)
        for b in bars:
            h=b.get_height()
            ax.text(b.get_x()+b.get_width()/2, h+1, f"{h:.1f}%", ha='center', va='bottom')
        _safe_save(fig,graphs_dir/f"union_all_fl_{fl}.png")

    print("✓ All steps complete. Check outputs in", graphs_dir)

if __name__ == '__main__':
    main()
