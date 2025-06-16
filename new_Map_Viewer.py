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
# 1) Basemap URLs
# ----------------------------
OSM_URL  = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
SAT_URL  = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}")
DARK_URL = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"

# ----------------------------
# 2) DataLoader
# ----------------------------
class DataLoader:
    def __init__(self):
        self.df_sum = None
        self.df_fiber = None
        self.trace_dir = None
        self.graphs_dir = None
        self.summary_csv = None
        self.pixel_data_cache = {}
        self.image_raw_cache = {}

    def load_all_data(self):
        root = tk.Tk(); root.withdraw()

        # Summary CSV
        self.summary_csv = filedialog.askopenfilename(
            title="Select shifted summary CSV",
            filetypes=[("CSV files","*.csv")]
        )
        if not self.summary_csv:
            raise SystemExit("No summary CSV selected.")
        hdr = pd.read_csv(self.summary_csv, nrows=0)
        desired = ['Flight number','Pixel','Start time','End time',
                   'Sensor Lat','Sensor Lon','Sensor Type','max_dist3D',
                   'Snapshot','Event Lat','Event Lon']
        dtypes = {'Flight number':'uint16','Pixel':'uint16',
                  'Sensor Lat':'float32','Sensor Lon':'float32',
                  'Sensor Type':'category','max_dist3D':'float32'}
        usecols = [c for c in desired if c in hdr.columns]
        self.df_sum = pd.read_csv(
            self.summary_csv,
            usecols=usecols,
            dtype={k:v for k,v in dtypes.items() if k in usecols}
        )
        if 'Sensor Lat' not in self.df_sum or 'Sensor Lon' not in self.df_sum:
            raise ValueError("Missing sensor lat/lon columns")

        # Fiber config CSV
        fiber_csv = filedialog.askopenfilename(
            title="Select fiber config CSV",
            filetypes=[("CSV files","*.csv")]
        )
        hdr2 = pd.read_csv(fiber_csv, nrows=0)
        fiber_desired = ['Location (pxl)','Latitude','Longitude']
        fiber_usecols = [c for c in fiber_desired if c in hdr2.columns]
        df_f = pd.read_csv(fiber_csv, usecols=fiber_usecols)
        if 'Location (pxl)' in df_f:
            df_f['Location (pxl)'] = pd.to_numeric(
                df_f['Location (pxl)'], errors='ignore', downcast='unsigned'
            )
        for col in ('Latitude','Longitude'):
            if col in df_f:
                df_f[col] = pd.to_numeric(
                    df_f[col], errors='ignore', downcast='float'
                )
        self.df_fiber = df_f.sort_values('Location (pxl)')

        # Flight logs dir
        self.trace_dir = filedialog.askdirectory(
            title="Select folder with Flight_*_logs.csv"
        )

        # Optional graphs dir
        self.graphs_dir = filedialog.askdirectory(
            title="Select 'Graphs and statistics' folder (or Cancel to skip)"
        )
        if not self.graphs_dir:
            fallback = os.path.join(
                os.path.dirname(self.summary_csv),
                "Graphs and statistics"
            )
            self.graphs_dir = fallback if os.path.exists(fallback) else None

        # Preload per-pixel CSVs
        self._preload_pixel_data()
        root.destroy()
        return self

    def _preload_pixel_data(self):
        base = os.path.dirname(self.summary_csv)
        for px in self.df_sum['Pixel'].unique():
            path = os.path.join(base, f"points_pixel_{px}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            if 'lat' in df and 'GPS Lat' not in df:
                df['GPS Lat'] = df['lat']
            if 'lon' in df and 'GPS Lon' not in df:
                df['GPS Lon'] = df['lon']
            if 'alt' in df and 'GPS Alt' not in df:
                df['GPS Alt'] = df['alt']
            self.pixel_data_cache[px] = df

    def get_pixel_data(self, pixel):
        return self.pixel_data_cache.get(pixel)

    @lru_cache(maxsize=3)
    def _load_flight_df(self, flight_number:int) -> pd.DataFrame:
        path = os.path.join(self.trace_dir, f"Flight_{flight_number}_logs.csv")
        hdr = pd.read_csv(path, nrows=0)
        cols = list(hdr.columns)
        # detect time
        time_col = next((c for c in cols if c.lower()=='time'),None)
        if not time_col:
            time_col = next((c for c in cols if 'time' in c.lower()),None)
        # detect lat/lon
        lat_col = 'GPS Lat' if 'GPS Lat' in cols else ('lat' if 'lat' in cols else None)
        lon_col = 'GPS Lon' if 'GPS Lon' in cols else ('lon' if 'lon' in cols else None)
        df = pd.read_csv(
            path,
            usecols=[time_col, lat_col, lon_col],
            dtype={lat_col:'float32', lon_col:'float32'}
        )
        df.rename(columns={time_col:'Time', lat_col:'GPS Lat', lon_col:'GPS Lon'},
                  inplace=True)
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        df['Flight number'] = flight_number
        return df

    def get_flight_data(self, flight_number:int):
        path = os.path.join(self.trace_dir, f"Flight_{flight_number}_logs.csv")
        if not os.path.exists(path):
            return None
        return self._load_flight_df(flight_number)

    def _load_image_bytes(self, filename):
        if not filename or pd.isna(filename) or not self.graphs_dir:
            return None
        for p in (
            os.path.join(self.graphs_dir, filename),
            os.path.join(self.graphs_dir, os.path.basename(filename)),
            os.path.join(os.path.dirname(self.summary_csv), filename)
        ):
            if os.path.exists(p):
                raw = open(p,'rb').read()
                ext = p.lower().split('.')[-1]
                mime = 'jpeg' if ext in ('jpg','jpeg') else 'png'
                return raw, mime
        return None

    def get_image(self, filename):
        img = self.image_raw_cache.get(filename)
        if not img:
            img = self._load_image_bytes(filename)
            if img and len(self.image_raw_cache)>=50:
                self.image_raw_cache.pop(next(iter(self.image_raw_cache)))
            if img:
                self.image_raw_cache[filename] = img
        return img

# ----------------------------
# 3) Processing Helpers
# ----------------------------
def process_flight_data(loader:DataLoader):
    out = {}
    for _,ev in loader.df_sum.iterrows():
        fl,px = int(ev['Flight number']), int(ev['Pixel'])
        start,end = ev['Start time'], ev['End time']
        snap = ev.get('Snapshot','')
        dfpx = loader.get_pixel_data(px)
        if dfpx is None: continue
        win = dfpx[(dfpx['Flight number']==fl)&
                   (dfpx['Time']>=start)&
                   (dfpx['Time']<=end)]
        if win.empty: continue
        coords = (win[['lat','lon']].values.tolist()
                  if 'lat' in win.columns
                  else win[['GPS Lat','GPS Lon']].values.tolist())
        meta = win.to_dict('records')
        out.setdefault((fl,px),[]).append((coords,meta,snap))
    return out

def load_coverage(graphs_dir):
    cov_px=cov_ty=None
    if graphs_dir:
        p1=os.path.join(graphs_dir,"coverage_per_pixel.csv")
        if os.path.exists(p1):
            cov_px=pd.read_csv(p1,index_col='Pixel')
            cov_px.columns=[f"Flight_{c}" if not str(c).startswith("Flight_") else c
                            for c in cov_px.columns]
        p2=os.path.join(graphs_dir,"coverage_per_type.csv")
        if os.path.exists(p2):
            cov_ty=pd.read_csv(p2,index_col='Type')
            cov_ty.columns=[f"Flight_{c}" if not str(c).startswith("Flight_") else c
                            for c in cov_ty.columns]
    return cov_px, cov_ty

# ----------------------------
# 4) Initialize & Globals
# ----------------------------
print("🔄 Initializing Map Viewer...")
loader = DataLoader().load_all_data()
dict_px       = process_flight_data(loader)
coverage_px, coverage_ty = load_coverage(loader.graphs_dir)

df_sum        = loader.df_sum
df_fiber      = loader.df_fiber
fiber_coords  = df_fiber[['Latitude','Longitude']].values.tolist()
flight_files  = glob.glob(os.path.join(loader.trace_dir,"Flight_*_logs.csv"))
flight_numbers= sorted(int(os.path.basename(p).split("_")[1]) for p in flight_files)

sensor_positions = {
    int(px):(r['Sensor Lat'],r['Sensor Lon'],r['Sensor Type'])
    for px,r in df_sum.drop_duplicates('Pixel')\
                    .set_index('Pixel')[['Sensor Lat','Sensor Lon','Sensor Type']]\
                    .iterrows()
}
max_dist = df_sum.groupby(['Flight number','Pixel'])['max_dist3D'].max().to_dict()
center   = ([df_sum['Event Lat'].mean(),df_sum['Event Lon'].mean()]
            if 'Event Lat' in df_sum and 'Event Lon' in df_sum
            else [df_fiber['Latitude'].mean(),df_fiber['Longitude'].mean()])

# Color schemes
pixel_colors = {px:c for px,c in zip(
    sorted(df_sum['Pixel'].unique()),
    ["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e",
     "#e6ab02","#a6761d","#666666","#f7b7a3","#6f5b4b"]
)}
type_colors = {st:c for st,c in zip(
    sorted(df_sum['Sensor Type'].unique()),
    ["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e"]
)}
MOVEMENT_COLORS = {
    'approaching':'#2ecc71','departing':'#e74c3c','hovering':'#9b59b6',
    'accelerating':'#f39c12','decelerating':'#3498db','turning':'#8e44ad',
    'cruising':'#95a5a6'
}

def get_pixel_coverage(px,fl):
    if coverage_px is not None:
        col=f"Flight_{fl}"
        if px in coverage_px.index and col in coverage_px.columns:
            return float(coverage_px.loc[px,col])
    return None

def build_popup(px,meta,snap=None):
    items=[
        html.H6(f"Pixel {px}"),
        html.P([
            f"Time: {meta.get('Time','N/A')}", html.Br(),
            f"Distance: {meta.get('distance_to_sensor', meta.get('dist3D','N/A')):.1f} m", html.Br(),
            f"Movement: {meta.get('pixel_movement_type','N/A')}"
        ], className='small')
    ]
    if snap:
        raw = loader.get_image(snap)
        if raw:
            b,m = raw; b64=base64.b64encode(b).decode()
            items.append(html.Img(src=f"data:image/{m};base64,{b64}",
                                  style={'width':'200px'}))
    return html.Div(items)

def build_sensor_popup(px,stype,fl,cov):
    rows=[html.Div(html.Strong(f"Pixel {px} ({stype})"))]
    md=max_dist.get((fl,px))
    if md is not None:
        rows.append(html.P(f"Max dist: {md:.1f} m",className='mb-0'))
    if cov is not None:
        rows.append(html.P(f"Coverage: {cov:.1f}%",className='mb-0'))
    return html.Div(rows,className='small')

# Static legend below map
# -----------------------
# Build a flat list of children, then wrap in one html.Div
legend_children = []

# 1. Pixel Colors header
legend_children.append(
    html.Div("Pixel Colors", style={'fontWeight': 'bold'})
)
# 2. Pixel color entries
for px, col in pixel_colors.items():
    legend_children.append(
        html.Div([
            html.Span(style={
                'display':'inline-block','width':'12px','height':'12px',
                'backgroundColor': col, 'marginRight':'6px'
            }),
            html.Span(f"Pixel {px}", style={'marginRight':'12px'})
        ])
    )

# 3. Separator
legend_children.append(html.Hr())

# 4. Movement Colors header
legend_children.append(
    html.Div("Movement Colors", style={'fontWeight': 'bold'})
)
# 5. Movement color entries
for mv, col in MOVEMENT_COLORS.items():
    legend_children.append(
        html.Div([
            html.Span(style={
                'display':'inline-block','width':'12px','height':'12px',
                'backgroundColor': col, 'marginRight':'6px'
            }),
            html.Span(mv.capitalize(), style={'marginRight':'12px'})
        ])
    )

legend_div = html.Div(
    legend_children,
    style={
        'padding':'10px',
        'background':'white',
        'borderRadius':'4px',
        'boxShadow':'0 0 6px rgba(0,0,0,0.2)',
        'marginBottom':'20px'
    }
)
# ----------------------------
# 5) Dash App & Layout
# ----------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)

map_controls = dbc.Card([
    dbc.CardBody([
        dbc.Row([
            dbc.Col([html.Label("Flight"), dcc.Dropdown(
                id='flight-dd',
                options=[{'label':f'Flight {f}','value':f} for f in flight_numbers],
                value=flight_numbers[0] if flight_numbers else None
            )], md=3),
            dbc.Col([html.Label("Mode"), dcc.RadioItems(
                id='view-mode',
                options=[{'label':'Individual','value':'individual'},
                         {'label':'Clustered','value':'clustered'}],
                value='individual', inline=True
            )], md=2),
            dbc.Col([html.Label("Filter"), dcc.Dropdown(
                id='selection-dd', multi=True, value=[]
            )], md=4),
            dbc.Col([html.Label("Display"), dbc.Checklist(
                id='display-options',
                options=[
                    {'label':'Flight Path','value':'flight_path'},
                    {'label':'Fiber Cable','value':'fiber'},
                    {'label':'Movement Colors','value':'movement_colors'}
                ],
                value=['flight_path','fiber']
            )], md=3)
        ], className="g-3"),
        dbc.Row([dbc.Col([html.Label("Basemap"), dcc.RadioItems(
            id='map-style',
            options=[
                {'label':'Satellite','value':'satellite'},
                {'label':'OSM','value':'osm'},
                {'label':'Dark','value':'dark'}
            ], value='satellite', inline=True
        )], md=4)], className="mt-3")
    ])
], className="mb-3", body=True)

map_tab = html.Div([
    html.H1("🗺️ Flight Sensor Analysis", className="mb-2"),
    map_controls,
    dl.Map(
        id='map', center=center, zoom=13,
        children=[dl.TileLayer(url=OSM_URL)],
        style={'width':'100%','height':'65vh'}
    ),
    legend_div
])

analysis_tab = html.Div([
    html.H4("Analysis Graphs", className="mt-4"),
    html.Div(id='analysis-images')
])

app.layout = dbc.Container([
    dcc.Tabs([
        dcc.Tab(label='Map', children=[map_tab]),
        dcc.Tab(label='Analysis', children=[analysis_tab])
    ])
], fluid=True, style={'padding':'20px','backgroundColor':'#f8f9fa'})

# ----------------------------
# 6) Callbacks
# ----------------------------
@app.callback(
    [Output('selection-dd','options'), Output('selection-dd','value')],
    Input('view-mode','value')
)
def update_selection(view_mode):
    if view_mode=='clustered':
        opts = [{'label': st,'value':st} for st in sorted(df_sum['Sensor Type'].unique())]
        return opts, []
    opts = [{'label':f'Pixel {px}','value':px} for px in sorted(sensor_positions)]
    return opts, []

@app.callback(
    Output('map','children'),
    Input('flight-dd','value'),
    Input('view-mode','value'),
    Input('selection-dd','value'),
    Input('display-options','value'),
    Input('map-style','value')
)
def update_map(flight, view_mode, sel, disp_opts, mstyle):
    disable_snap = (view_mode=='individual' and len(sel or [])>1)
    tile = {'satellite':SAT_URL,'dark':DARK_URL,'osm':OSM_URL}[mstyle]
    layers = [dl.TileLayer(url=tile)]

    if not flight:
        return layers

    # fiber cable
    if 'fiber' in disp_opts:
        layers.append(dl.Polyline(positions=fiber_coords,
                                  color='#FFD700',weight=3,opacity=0.8))
    # flight path downsampled
    if 'flight_path' in disp_opts:
        fd = loader.get_flight_data(flight)
        if fd is not None:
            pts = fd[['GPS Lat','GPS Lon']].values.tolist()
            step = max(1,len(pts)//200)
            layers.append(dl.Polyline(positions=pts[::step],
                                      color='#00BFFF',weight=2,opacity=0.7))

    # traces & markers
    if view_mode=='individual':
        for px in sel or []:
            for coords,meta,snap in dict_px.get((flight,px),[]):
                col = pixel_colors.get(px,'#0066CC')
                step_t = max(1,len(coords)//50)
                layers.append(dl.Polyline(positions=coords[::step_t],
                                          color=col,weight=4,opacity=0.8))
                step_m = max(1,len(coords)//20)
                for i in range(0,len(coords),step_m):
                    lat,lon = coords[i]; m = meta[i]
                    mc = col
                    if 'movement_colors' in disp_opts:
                        key = str(m.get('pixel_movement_type','cruising')).split(',')[0].strip().lower()
                        mc = MOVEMENT_COLORS.get(key,col)
                    popup = build_popup(px,m,None if disable_snap else snap)
                    layers.append(dl.CircleMarker(
                        center=[lat,lon],radius=4,color=mc,fill=True,fillOpacity=0.8,
                        children=[dl.Popup(popup)]
                    ))
        for px in sel or []:
            lat,lon,stype = sensor_positions.get(px,(None,None,None))
            if lat is None: continue
            cov = get_pixel_coverage(px,flight)
            popup = build_sensor_popup(px,stype,flight,cov)
            layers.append(dl.CircleMarker(
                center=[lat,lon],radius=10,
                color=pixel_colors.get(px,'#666666'),fill=True,fillOpacity=1.0,
                children=[dl.Tooltip(f"Pixel {px} ({stype})"),dl.Popup(popup)]
            ))
    else:
        for stype in sel or []:
            col = type_colors.get(stype,'#333333')
            for px,(_,_,t) in sensor_positions.items():
                if t!=stype: continue
                for coords,_,_ in dict_px.get((flight,px),[]):
                    step_t = max(1,len(coords)//50)
                    layers.append(dl.Polyline(positions=coords[::step_t],
                                              color=col,weight=4,opacity=0.8))
                lat,lon,_ = sensor_positions[px]
                cov = get_pixel_coverage(px,flight)
                popup = build_sensor_popup(px,stype,flight,cov)
                layers.append(dl.CircleMarker(
                    center=[lat,lon],radius=6,
                    color=col,fill=True,fillOpacity=0.6,
                    children=[dl.Tooltip(f"Pixel {px} ({stype})"),dl.Popup(popup)]
                ))

    return layers

@app.callback(
    Output('analysis-images','children'),
    Input('flight-dd','value'),
    Input('view-mode','value'),
    Input('selection-dd','value')
)
def update_analysis(flight, view_mode, sel):
    if not loader.graphs_dir or not flight:
        return ""
    imgs=[]
    def add(fn):
        raw=loader.get_image(fn)
        if raw:
            b,m=raw
            src="data:image/"+m+";base64,"+base64.b64encode(b).decode()
            imgs.append(html.Img(src=src,
                                 style={'maxWidth':'400px','width':'100%','marginBottom':'10px'}))
    add('avg_coverage_per_pixel.png')
    add(f'union_all_fl_{flight}.png')
    if view_mode=='individual':
        for px in sel or []:
            add(f'donut_px_{px}_fl_{flight}.png')
            add(f'hist_pixel_{px}_flight_{flight}.png')
    else:
        for st in sel or []:
            add(f'coverage_type_{st}_fl_{flight}.png')
    return dbc.Row([dbc.Col(img,md=4) for img in imgs]) if imgs else ""

if __name__=='__main__':
    print("🚀 Starting Dashboard…")
    app.run(debug=True, port=8050, use_reloader=False)
