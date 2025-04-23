import streamlit as st
import pandas as pd
import altair as alt
import json
import plotly.express as px
from PIL import Image
import json
from shapely.geometry import Polygon, shape,mapping, Point
from shapely.ops import unary_union
from collections import defaultdict

import geopandas as gpd

import folium
from folium.features import GeoJsonTooltip
from folium.plugins import Fullscreen
from pyproj import Transformer
from folium.raster_layers import ImageOverlay

from branca.colormap import LinearColormap, StepColormap

import numpy as np
import xarray as xr
import rasterio
import base64
import matplotlib.colors as mcolors
import io

#######################
# dfm.columns = [x.replace('_', ' ') for x in dfm.columns]
logo_wide = r'data/logo_wide.png'
logo_small = r'data/logo_small.png'

IPA_description = {
    "beneficial fraction": ":blue[Beneficial fraction (BF)] is the ratio of the water that is consumed as transpiration\
         compared to overall field water consumption (ETa). ${\\footnotesize BF = T_a/ET_a}$. \
         It is a measure of the efficiency of on farm water and agronomic practices in use of water for crop growth.",
    "crop water deficit": ":blue[crop water deficit (CWD)] is measure of adequacy and calculated as the ration of seasonal\
        evapotranspiration to potential or reference evapotranspiration ${\\footnotesize CWD= ET_a/ET_p}$",
    "relative water deficit": ":blue[relative water deficit (RWD)] is also a measure of adequacy which is 1 minus crop water\
          deficit ${\\footnotesize RWD= 1-ET_a/ET_p}$",
    "total seasonal biomass production": ":blue[total seasonal biomass production (TBP)] is total biomass produced in tons. \
        ${\\footnotesize TBP = (NPP * 22.222) / 1000}$",
    "seasonal yield": ":blue[seasonal yield] is the yield in a season which is crop specific and calculated using \
        the TBP and yield factors such as moisture content, harvest index, light use efficiency correction \
            factor and above ground over total biomass production ratio (AOT) \
                ${\\footnotesize Yiled = TBP*HI*AOT*f_c/(1-MC)}$",
    "crop water productivity": ":blue[crop water productivity (CWP)] is the seasonal yield per the amount of water \
        consumed in ${kg/m^3}$"
}

stat_dict = {'Standard deviation':'std', 'Minimum': 'min', 'Maximum':'max', 'Average':'mean', 'Median':'meadian'}

units = {'beneficial fraction':'-', 'crop water deficit': '-',
    'relative water deficit': '-', 'total seasonal biomass production': 'ton',
    'seasonal yield': 'ton/ha', 'crop water productivity': 'kg/m³'}

crop_calendar = {'wheats': 'November to March', 'sorgums':'June to December', 'cottons':'June to March'}

# @st.cache_data(ttl=300)
def load_image(image_name: str) -> Image:
    """Displays an image.

    Parameters
    ----------
    image_name : str
        Local path of the image.

    Returns
    -------
    Image
        Image to be displayed.
    """
    return Image.open(image_name)


def logos():
    img_small = load_image(logo_small)
    img_wide = load_image(logo_wide)

    return  img_small, img_wide


#######################
BACKGROUND_COLOR = 'white'
COLOR = 'black'

def set_page_container_style(
        max_width: int = 1100, max_width_100_percent: bool = False,
        padding_top: int = 1, padding_right: int = 1, padding_left: int = 1, padding_bottom: int = 10,
        color: str = COLOR, background_color: str = BACKGROUND_COLOR,
    ):
        # Page configuration
        st.set_page_config(
            page_title="Gezira Irrigation Scheme Irrigation Performance Indicators by Sections Dashboard",
            page_icon="📈🌿",
            layout="wide",
            initial_sidebar_state="expanded")

        alt.themes.enable("dark")

        st.markdown("""
            <style>
            header.stAppHeader {
                background-color: transparent;
            }
            section.stMain .block-container {
                padding-top: 0rem;
                z-index: 1;
            }
            </style>""", unsafe_allow_html=True)

        hide_github_icon = """
            <style>
                .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK{ display: none; } #MainMenu{ visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
            </style>
        """
        st.markdown(hide_github_icon, unsafe_allow_html=True)
        #######################
        
# Load data
@st.cache_data(ttl=300)
def read_df_and_geo(selected_crop):
    dfm = pd.read_csv(fr'data/Gezira_IPA_statistic_{selected_crop}.csv') 
    with open(r'data/Gezira_IR.json') as response:
        geo = json.load(response)

    return dfm, geo

@st.cache_data(ttl=300)
def read_crop_area_df():
    df = pd.read_csv(r'data/crop_type_precent.csv') 
    return df

def indicator_title(indicator, stat_dict):
    # stat_dict = {'std':'Standard deviation', 'min':'Minimum', 'max':'Maximum', 'mean':'Average', 'median':'Median'}
    lst = indicator.split('_')
    t1 = ' '.join(lst[1:])
    t2 = f"{[k for k, v in stat_dict.items() if v == lst[0]][0]} {t1}" 
    return t1,t2

# merge sections to divisionss
def merge_sections_to_divisions(geo, df_section):
    new_features = []
    for i, name in enumerate(df_section.division):
        polygons = []
        to_combine = [f for f in geo["features"] if f["properties"]['division']==name]

        for feat in to_combine:
            geom = shape(feat["geometry"])
            polygons.append(geom)    
            
        # new_geometry = mapping(unary_union(polygons)) # This line merges the polygones
        new_geometry = mapping(unary_union(polygons))
# 
        # new_geometry = polygons
        new_feature = dict(type='Feature', id=i, properties=dict(division=name),
                        geometry=dict(type=new_geometry['type'], 
                                        coordinates=new_geometry['coordinates']))
        new_features.append(new_feature)
            
    divisions = dict(type='FeatureCollection', 
                    crs= dict(type='name', properties=dict(name='urn:ogc:def:crs:OGC:1.3:CRS84')), 
                    features=new_features)
    return divisions




def make_folium_choropleth(geo, indicator, df, col_name):
    df = df.round(2)
    ylable, text = indicator_title(indicator, stat_dict)
    
    # Convert DataFrame to dictionary for mapping
    data_df = df.set_index(col_name)[indicator]

    # Convert DataFrame to dictionary for fast lookup
    data_dict = data_df.to_dict()

      # Add ESRI aerial imagery tile layer
    esri = folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Aerial Imagery",
        overlay=False,
        control=True
    )#.add_to(m)
 
    m = folium.Map(location=[14.429, 33.01], 
                   zoom_start=12, height=300, width= '100%',
                    tiles=esri,  # Add ESRI arial imagery as default tile layer 
  
    )

    # Add OSM map
    folium.TileLayer("OpenStreetMap", name="OSM", control=True).add_to(m)
    # # bounds = get_bounds(geo)
    minv = df[indicator].min()
    maxv = df[indicator].max()

    # Define a custom color scale
    colormap = StepColormap(
        ["#ff0000", "#ff4500", "#ff7f50", "#ffb347", "#ffdd44", 
        "#ccff33", "#99ff33", "#66ff33", "#33cc33", "#009933"], 
        vmin=minv, vmax=maxv, caption="colormap"
    )
    # Update geojson for tooltip
    for feature in geo["features"]:
        region_id = feature["properties"].get(col_name)  # Get region ID from GeoJSON
        if region_id in data_dict:  # Check if ID exists in CSV
            feature["properties"][indicator] = data_dict[region_id]
        else:
            feature["properties"][indicator] = None  # Assign None if not found

    if 'section' in df.columns:
        fields = ['division', col_name, indicator]
        aliases = ["Division:", "Section:", f"{ylable}:"]
        geo["ch_name"] = "Gezira Sections"	
    else:
        fields = [col_name, indicator]
        aliases = ["Division:", f"{ylable}:"]
        geo["ch_name"] = "Gezira Divisions"
    
    # Add Choropleth layer
    tooltip=folium.GeoJsonTooltip(
            fields=fields,#[col_name, indicator],
            aliases=aliases,#["Section:", f"{ylable}:"],
            localize=True,
            sticky=False,
            labels=True,
            style="""
                background-color: #F0EFEF;
                border: 1px solid black;
                border-radius: 3px;
                box-shadow: 3px;
                font-size: 12px;
                font-weight: normal;
                # max-width: 400px; 
                # white-space: normal; 
            """,
            # max_width=200,
            html=True  # Enables HTML in the tooltip
        )
    
    choropleth = folium.GeoJson(
        geo,
        name=geo["ch_name"],
        tooltip=tooltip,
        style_function=lambda feature: {
            "fillColor": colormap(data_dict[feature["properties"].get(col_name)]),
            "color": "black",
            "weight": 0.5,
            "fillOpacity": 0.7
        },
    ).add_to(m)

    # Add Click event
    click_marker = folium.Marker(
        location=[0, 0],  # Default position (hidden initially)
        popup="Click on the map",
        icon=folium.Icon(color="red")
    )
    m.add_child(click_marker)
           
    folium.LayerControl().add_to(m)
    folium.plugins.Fullscreen().add_to(m)   
    bounds = choropleth.get_bounds()  # Automatically calculates min/max lat/lon

    # Fit map to bounds
    m.fit_bounds(bounds)
    return m


# histogram plot
def make_alt_chart(df,indicator):
    ylable, text = indicator_title(indicator, stat_dict)
    title = alt.TitleParams(f'Yearly {text} by section', anchor='middle')
    barchart = alt.Chart(df, title=title).mark_bar().encode(
        x=alt.X('division:N', axis=None),
        y=alt.Y(f'{indicator}:Q', title=ylable),
        color='division:N',
        column='season:N'
    ).properties(width=80, height=120).configure_legend(
        orient='bottom'
    )
    return barchart

def format_number(num):
    return f"{num:.2f}"

# Calculation season-over-season difference in metrix
def calculate_indicator_difference(input_df, indicator, input_season):
  selected_season_data = input_df[input_df['season'] == input_season].reset_index()
  previous_season_data = input_df[input_df['season'] == input_season - 1].reset_index()
  selected_season_data['indicator_difference'] = selected_season_data[indicator].sub(previous_season_data[indicator], fill_value=0)
  return pd.concat([selected_season_data['division'], selected_season_data[indicator], selected_season_data.indicator_difference], axis=1).sort_values(by="indicator_difference", ascending=False)


def history_df(df1, df2, idx_col, selected_indicator):
    d2 = df1.pivot(index=idx_col, columns='season', values=selected_indicator).reset_index()
    d3 = df2.groupby(idx_col).agg({selected_indicator:'mean'}).reset_index()
    d4 = d3.merge(d2, on=idx_col, how = 'inner')
    d4[d4.columns[2:]]= d4[d4.columns[2:]].round(2)
    d4['history'] = d4[d4.columns[2:]].values.tolist()
    d4 = d4.drop(columns = d4.columns[2:-1])
    return d4.round(2)

select = alt.selection_point(name="select", on="click")
highlight = alt.selection_point(name="highlight", on="pointerover", empty=False)

stroke_width = (
    alt.when(select).then(alt.value(2, empty=False))
    .when(highlight).then(alt.value(1))
    .otherwise(alt.value(0))
)

def alt_bar_chart(df, indicator, col_name, season):
    xlable, text = indicator_title(indicator, stat_dict)
    df = df.round(2)
    indicator_name = indicator.replace("_"," ")
    plot_title = f'{indicator_name.title()} - {str(season)}'
    # x_title = f'{xlable.title()} [{units[xlable]}]'
    # title = f'{indicator_name.title()} - {str(season)}'
    
    if len(col_name.split("_"))>1:
        area_id = col_name.split("_")[0]
    else:
        area_id = col_name
    
    
    row_count = len(df)
    pixel_size = 60 - 2* (row_count-5)
    height = row_count * pixel_size  # 30 pixels per row


    select = alt.selection_point(name="select", on="click")
    highlight = alt.selection_point(name="highlight", on="pointerover", empty=False)

    stroke_width = (
        alt.when(select).then(alt.value(2, empty=False))
        .when(highlight).then(alt.value(1))
        .otherwise(alt.value(0))
    )

    chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y(f'{col_name}:N', sort=alt.EncodingSortField(field="indicator", op="count", order='descending'),title=area_id),  # Rename Y-axis
        x=alt.X(f'{indicator}:Q', title=indicator_name),  # Rename X-axis
        color=alt.Color(f'{indicator}:N', legend=None),  # Remove the legend
        fillOpacity=alt.when(select).then(alt.value(1)).otherwise(alt.value(0.3)),
        strokeWidth=stroke_width,

        tooltip=[
            alt.Tooltip(f'{col_name}:N', title=area_id),  
            alt.Tooltip(f'{indicator}:Q', title=indicator_name, format='.2f'),  # Format Value as decimal with 2 digits
        ],
        

    ).properties(
        height=height, #title=plot_title
        ).configure_view(
            continuousWidth=600,  # Default width to avoid shrinking
            continuousHeight=300
        ).configure(
            autosize="fit"  # Ensures it resizes correctly
        ).add_params(select, highlight)

    return chart, plot_title


def move_rows_to_top(df, column, value):
    """Reorder DataFrame rows to bring specified column value to the top."""
    df_top = df[df[column] == value]  # Rows with the specified value
    df_rest = df[df[column] != value]  # All other rows
    return pd.concat([df_top, df_rest], ignore_index=True)  # Merge and reset index


def make_alt_linechart(df, indicator, col_name, season, selected_section):
    ylable, text = indicator_title(indicator, stat_dict)

    df['year'] = df['season'].str.split('-').str[1]
    df=df.assign(year= pd.to_datetime(df['year'], format='%Y')).round(2)
    indicator_name = indicator.replace("_"," ")
    
    min_value = df[indicator].min()
    max_value = df[indicator].max()
    
    if len(col_name.split("_"))>1:
        area_id = col_name.split("_")[0]
    else:
        area_id = col_name

    plot_title = f'{text.title()} per {area_id} for the past seasons'
    y_title = f'{ylable.title()} [{units[ylable]}]' 

    # title = f'{indicator_name.title()} per {area_id}s for the past seasons'

    if selected_section is not None:
        # df['order'] = ['first' if x== selected_section else 'last' for x in df['block']]
        df_sorted = move_rows_to_top(df, 'section', selected_section).iloc[::-1]
               
        chart = alt.Chart(df_sorted).mark_line(size=3).encode(
            x=alt.X('year:T',title='Season', axis=alt.Axis(tickCount="year")), 
            y=alt.Y(f'{indicator}:Q',title=y_title, scale=alt.Scale(domain=[min_value, max_value])),
            color=alt.Color(f'{col_name}:N', title = area_id,  legend=alt.Legend(orient="top")),
            opacity=alt.condition(
                alt.datum.section == selected_section,  # Highlight selected category
                alt.value(1),  # Full opacity for selected
                alt.value(0.2)  # Lower opacity for others
            ),

            tooltip=[
                alt.Tooltip(f'{col_name}:N', title=area_id),
                alt.Tooltip('year:T', title='season',format='%Y'),    
                alt.Tooltip(f'{indicator}:Q', title=indicator, format='.2f'),  # Format Value as decimal with 2 digits
            ]
        ).properties(
        #    title=plot_title,
           height=300,
           bounds="flush",  # Ensures title does not affect chart size
        ).configure_view(
            continuousWidth=600,  # Default width to avoid shrinking
            continuousHeight=300
        ).configure(
            autosize="fit",  # Ensures it resizes correctly
        )
        
        return chart, plot_title
    else:
        chart = alt.Chart(df).mark_line().encode(
            x=alt.X('year:T',title='Season', axis=alt.Axis(tickCount="year",
                            labelExpr='(parseInt(timeFormat(datum.value, \'%Y\')) - 1) + "–" + timeFormat(datum.value, \'%Y\')')),  
            y=alt.Y(f'{indicator}:Q',title=y_title, scale=alt.Scale(domain=[min_value, max_value])),
            color=alt.Color(f'{col_name}:N', title = area_id,  legend=alt.Legend(orient="top")),

            tooltip=[
                alt.Tooltip(f'{col_name}:N', title=area_id),
                alt.Tooltip('year:T', title='season',format='%Y'),    
                alt.Tooltip(f'{indicator}:Q', title=indicator, format='.2f'),  # Format Value as decimal with 2 digits
            ]
        ).properties(
            # title=plot_title,
            height=300, 
            bounds="flush",  # Ensures title does not affect chart size
        ).configure_view(
            continuousWidth=600,  # Default width to avoid shrinking
            continuousHeight=300
        ).configure(
            autosize="fit",  # Ensures it resizes correctly
        )

        return chart, plot_title


#=================================
# Load xarray dataset
@st.cache_data(ttl=300)
def create_base_map(center_lat, center_lon, zoom):
    """Create base map with satellite/aerial imagery as default and fullscreen control"""    
    # Add Satellite/Aerial imagery as default base layer
       # Add ESRI aerial imagery tile layer
    esri = folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Aerial Imagery",
        overlay=False,
        control=True
    )#.add_to(m))
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles=esri,
        control_scale=True
    )
    # Add OpenStreetMap as an alternative base layer
    folium.TileLayer("OpenStreetMap", name="OSM", control=True).add_to(m)
    
    # Add Fullscreen control
    Fullscreen(
        position='topleft',
        title='Fullscreen mode',
        title_cancel='Exit fullscreen mode',
        force_separate_button=True
    ).add_to(m)
    
    return m

def get_value_at_point(da, lat, lon, variable):
    ts = da.sel(lat=lat, lon=lon, method="nearest")
    ts = ts.to_dataframe().loc[:,variable] 
    return ts


def extraxt_ts(da, locations):
    # Extract time series for each location
    time_series = {}
    for idx, (lat, lon) in enumerate(locations):
        ts = da.sel(lat=lat, lon=lon, method="nearest")  # Nearest neighbor selection
        time_series[f'point_{idx+1}'] = ts.to_pandas()  # Convert to Pandas Series for easy manipulation

    # Convert to a DataFrame for better analysis
    df = pd.DataFrame(time_series).reset_index()
    return df

def alt_line_chart(df, indicator):
    # df2=df.assign(time= pd.to_datetime(df['time']).dt.season).dropna(axis=1, how='all').round(2)
    
    df2=df.assign(season = pd.to_datetime(df['season'],format='%Y')).dropna(axis=1, how='all').round(2)
    indicator_name = indicator.replace("_"," ")
    plot_title = f'{indicator_name.title()} for the pixels over the seasons'
    y_title = f'{indicator_name.title()} [{units[indicator_name]}]' 
    data = df2.melt('season')
    minv = data['value'].min()
    maxv = data['value'].max()
    chart = alt.Chart(data).mark_line().encode(
            x=alt.X('season:T',title='Year', axis=alt.Axis(tickCount="year")),  
            y=alt.Y(f'value:Q', title=y_title, scale=alt.Scale(domain=[minv*0.9, maxv*1.1])),
            color=alt.Color(f'variable:N',  title='Point', legend=alt.Legend(orient="right")),

            tooltip=[
                # alt.Tooltip(f'{col_name}:N', title=area_id),
                alt.Tooltip('season:T', title='season',format='%Y'),    
                alt.Tooltip(f'{indicator}:Q', title=indicator, format='.2f'),  # Format Value as decimal with 2 digits
            ]
        ).properties(
            # title=plot_title,
            height=300, 
            bounds="flush",  # Ensures title does not affect chart size
        ).configure_view(
            continuousWidth=600,  # Default width to avoid shrinking
            continuousHeight=300
        ).configure(
            autosize="fit",  # Ensures it resizes correctly
        )

    return chart, plot_title


def get_image_from_ds(data, minv, maxv, nodata, colors):
    try:
        # Apply scale factor
        data = np.nan_to_num(data, nan=-9999)
        data = np.flip(data,0)
        data = data.astype(float)
        
        # Create a custom colormap
        n_bins = 10
        cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
        
        # Normalize data
        vmin, vmax = minv, maxv
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Apply colormap to data
        colored_data = cmap(norm(data))
        
        # Set alpha channel to 0 for no-data values
        colored_data[..., 3] = np.where(data == nodata, 0, 0.9)
        
        # Convert to PIL Image
        img = Image.fromarray((colored_data * 255).astype(np.uint8))
    
        # Save image to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Encode image to base64
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode()

        return img_base64
    
    except Exception as e:
        st.error(f"An error occurred while processing the data: {str(e)}")
        st.write("Debug information:")
        st.write(f"Data shape: {data.shape}")

@st.cache_data
def read_dataset(ds_path):
    # chunks = {'season': 1, 'latitude': 2000, 'longitude': 2000}
    # with xr.open_dataset(ds_path, chunks=chunks) as dataset:  
    with xr.open_dataset(ds_path) as dataset:  
        # data = dataset.beneficial_fraction[0].values
        dataset = dataset.transpose('season', 'lat', 'lon')  # change axis order
        transform = dataset.rio.transform()
        crs = dataset.rio.crs
        nodata = -9999 #dataset.nodata
        bd = dataset.rio.bounds()
        bounds = rasterio.coords.BoundingBox(bd[0], bd[1], bd[2], bd[3])
    
    return dataset, transform, crs, nodata, bounds
    

@st.cache_data
def get_stats(_data):
      # Compute spatial statistics
    _data = _data.where(_data>0, np.nan)
    stats = {
        'Minimum': _data.min(dim=['lat', 'lon']),
        'Maximum': _data.max(dim=['lat', 'lon']),
        'Mean': _data.mean(dim=['lat', 'lon']),
        'Median': _data.median(dim=['lat', 'lon']),
        'St. deviation': _data.std(dim=['lat', 'lon']),
        "25% quantile": _data.quantile(0.25, dim=['lat', 'lon'], method='linear')
                        .drop_vars('quantile'),
        "75% quantile": _data.quantile(0.75, dim=['lat', 'lon'], method='linear')
                        .drop_vars('quantile'),
    }

    # pd.DataFrame.from_dict(d)
    df_stat = pd.DataFrame.from_dict({k: v.values.item() for k, v in stats.items()}, 
                                    orient='index', columns = ['Values']).round(2)
    df_stat.index.names = ['Stats']
    return df_stat


# Efficient function to get image data for overlay
def get_image_from_ds(data, minv, maxv, nodata, colors):

    try:
        data = np.nan_to_num(data, nan=nodata)
        data = np.flip(data, 0).astype(float)
        
        # Normalize and apply colormap
        norm = mcolors.Normalize(vmin=minv, vmax=maxv)
        cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=100)
        colored_data = cmap(norm(data))
        
        # Set alpha channel for no-data values
        colored_data[..., 3] = np.where(data == nodata, 0, 0.9)
        
        # Convert to PIL image and then to base64
        img = Image.fromarray((colored_data * 255).astype(np.uint8))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return base64.b64encode(img_bytes.getvalue()).decode()
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")

    
# Function to extract time series in a vectorized way
def extract_time_series(da, locations):
    lats, lons = zip(*locations)
    ts = da.sel(lat=list(lats), lon=list(lons), method="nearest")
    return ts.to_dataframe().reset_index()

# Efficient Folium Map Initialization

def transform_bounds(bounds, crs):
    """Transform bounds to EPSG:4326."""
    left, bottom, right, top = bounds
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return transformer.transform(left, bottom), transformer.transform(right, top)

def create_colormap(data, colors, variable, folium_map):
    """Create and add a colormap legend to the map."""
    colormap = LinearColormap(colors=colors, vmin=data.min(), vmax=data.max())
    colormap.caption = f"{variable} Values"
    colormap.add_to(folium_map)
    return colormap

def add_image_overlay(data, bounds, colors, variable, folium_map):
    """Generate an image overlay for the map."""
    img_base64 = get_image_from_ds(data, data.min(), data.max(), -9999, colors)
    ImageOverlay(
        name=f"{variable.replace('_', ' ')}".title(),
        image=f"data:image/png;base64,{img_base64}",
        bounds=bounds,
        opacity=0.9,
    ).add_to(folium_map)

def add_geojson_layer(geo, folium_map):
    """Add a GeoJSON layer with tooltips to the map."""
    geo_layer = folium.GeoJson(
        geo,
        name="irrigation divisions",
        style_function=lambda _: {
            'fillColor': '#00000000', 
            'color': 'black',
            "weight": 0.5,
        },
    ).add_to(folium_map)

    tooltip = GeoJsonTooltip(
        fields=['division', 'section'],
        aliases=["Division: ", "Section: "],
        localize=True,
        sticky=False,
        labels=True,
        smooth_factor=0,
        style="""
            background-color: #F0EFEF;
            border: 1px solid black;
            border-radius: 3px;
            box-shadow: 3px;
            font-size: 12px;
            font-weight: normal;
        """,
        max_width=750,
    )
    geo_layer.add_child(tooltip)
    return geo_layer

def add_click_markers(folium_map, clicked_locations):
    """Add markers for clicked locations."""
    for idx, (lat, lon) in enumerate(clicked_locations):
        folium.Marker(
            [lat, lon], 
            popup=f"Point {idx + 1}: lat: {lat:.4f}, lon: {lon:.4f}", 
            icon=folium.Icon(color="blue")
        ).add_to(folium_map)

        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                icon_size=(50, 50),
                icon_anchor=(3, 17),
                html=f'<div style="font-size: 24ptpx;font-weight: bold; color: white;">{idx + 1}</div>'
            ),
            zIndexOffset=1000
        ).add_to(folium_map)

def create_folium_map(data, geo, bounds, crs, variable):
    """Main function to create the folium map."""
    (left, bottom), (right, top) = transform_bounds(bounds, crs)
    
    folium_map = create_base_map((bottom + top) / 2, (left + right) / 2, 12)
    
    colors = ['red', 'yellow', 'green']
    create_colormap(data, colors, variable, folium_map)
    add_image_overlay(data, [[bottom, left], [top, right]], colors, variable, folium_map)
    
    geo_layer = add_geojson_layer(geo, folium_map)
    folium_map.fit_bounds(geo_layer.get_bounds())

    click_marker = folium.Marker(
        location=[0, 0],
        popup="Click on the map",
        icon=folium.Icon(color="red")
    )
    folium_map.add_child(click_marker)
    folium_map.add_child(folium.LatLngPopup())

    add_click_markers(folium_map, st.session_state.clicked_locations)

    folium.LayerControl().add_to(folium_map)

    return folium_map


@st.cache_data
def get_gdf_from_json(geo):
     return gpd.GeoDataFrame.from_features(geo['features'])

def filter_points_within_polygon(points, polygon):
    """Filter points that lie within the given polygon."""
    # Create GeoSeries for points
    points = [Point(lon, lat) for lat, lon in points]
    points_in = [x for x in points if polygon.contains(x).any()]

    return [(lambda point: (point.y, point.x))(point) for point in points_in]


def plotly_pie_chart(dfca, name, year):

    df = dfca.melt(
        value_vars=[col for col in dfca.columns if '_pct' in col],
        var_name='landuse_type',
        value_name='percentage'
    )

    # Clean up the landuse_type column for better labels
    df['landuse_type'] = df['landuse_type'].str.replace('_pct', '')

    fig = px.pie(df, values='percentage', names='landuse_type')
    # Set a general hoverlabel style (one for all slices)
    fig.update_traces(
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.7)",  # or pick a neutral color like gray or black
            font=dict(color="white")
        )
    )
    
    fig.update_traces(hole=.3, textposition='inside', textinfo='percent+label', textfont_size=16)
    fig.update_layout(showlegend=False)

    title = f'Area covered by each landuse class for {name} - {year}'
    return fig, title

