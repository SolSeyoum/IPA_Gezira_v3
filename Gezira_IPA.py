#######################
import streamlit as st
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from streamlit_folium import st_folium
#######################
from util import common2 as cm


cm.set_page_container_style(
        max_width = 1100, max_width_100_percent = True,
        padding_top = 0, padding_right = 0, padding_left = 0, padding_bottom = 0
)


logo_small, logo_wide = cm.logos()

col = st.columns((5.5, 2.5), gap='small')
# Add a Reset Button as a Folium Marker
def reset_map():
    """Resets the selected polygon state."""
    st.session_state.selected_division = None
    st.rerun()

# Sidebar
with st.sidebar:

    st.logo(logo_wide, size="large", link='https://www.un-ihe.org/', icon_image=logo_small)
    st.title('Mwea Irrigation Performance Indicators')

    crop_lst = ['wheats', 'sorgums', 'cottons']
    selected_crop = st.selectbox('Select crop type', crop_lst)
    dfm, geo = cm.read_df_and_geo(selected_crop)
    dfc = cm.read_crop_area_df()
    # dfm = pd.read_csv(fr'data/Gezira_IPA_statistic_{selected_crop}.csv')

    season_list = list(dfm.season.unique())[::-1]
    ll = list(dfm.columns.unique())[3:][::-1]
    indicator_lst = [' '.join(l.split('_')[1:]) for l in ll]
    indicator_lst = list(set(indicator_lst))
    indicator_lst = [x for x in indicator_lst if x != ""]
    
    selected_season = st.selectbox('Select a season', season_list, index = 0, 
                                 help="Choose the Year/Season to visualize")
    st.write(f'The :blue[season] for :blue[{selected_crop}] runs from months of :blue[{cm.crop_calendar[selected_crop]}].')

    indicator = st.selectbox('Select an indicator', indicator_lst, index = 0,
                             help="Choose the IPA indicator type to visualize")
    selected_stat = st.selectbox('Select a statistics', cm.stat_dict.keys(), index = 3, 
                                 help="Choose the statistics to visualize")
    selected_stat_abbr = cm.stat_dict[selected_stat]
    selected_indicator = f'{selected_stat_abbr}_{indicator.replace(' ', '_')}'
       
    st.write(f'{cm.IPA_description[indicator]}')
    stat_description = cm.stat_dict[selected_stat]

    df_selected = dfm[dfm.season == selected_season][['division', selected_indicator]]
    df_selected_sorted = df_selected.sort_values(by=selected_indicator, ascending=False)

    #aggregate by divisions
    df_division=df_selected_sorted.groupby('division').agg({selected_indicator:'mean'})#.rename(columns=d)
    df_division = df_division.sort_values(by=selected_indicator, ascending=False).reset_index()

    st.markdown("---")
    with st.expander("ℹ️ About the Indicator Map"):
        st.markdown("""
        This Indicator Map provides view of the Irrigation Performance Indicators (IPA) for Mwea Irrigation Scheme.
        - IPAs are calculated using data from: [FAO WaPOR data](https://www.fao.org/in-action/remote-sensing-for-water-productivity/wapor-data/en).
        - :orange[**Indicator Map**]: Shows the irrigation schemes section or blocks values for the selected indicator and selected statistics.      
        - Year/Season, and indicator type and statistic type can be selected to view the indicator selected by year/season and by statistics type.
        - 📊 :orange[**Bar Chart**]: on the right side shows the indicator for the selected year for the section or the block depending on which view is on the indicator map sorted by the selected indicator.                             
        - 📈 :orange[**Line Chart**]: or the timeseries plot below the map shows the trend over the years (seasons) for the selected indicator.

        """)

#######################
# Dashboard Main Panel

df_map = pd.DataFrame()
df_chart = pd.DataFrame()
col_name = ''
units = cm.units

# geopandas dataframe of the geo (AIO)
gdf = gpd.GeoDataFrame.from_features(geo['features'])

# Initialize session state
if "selected_division" not in st.session_state:
    st.session_state.selected_division = None
if "selected_section" not in st.session_state:
    st.session_state.selected_section = None

selected_year = int(selected_season.split('-')[1])
# Filter data based on selection
if st.session_state.selected_division is not None:
    # Add a reset button to Streamlit UI (works better than Folium custom HTML)
    selected_poly = st.session_state.selected_division
    filtered = [sgeo for sgeo in geo["features"] if sgeo['properties']['division'] in selected_poly]
    
    filtered_geojson = {
        "type": "FeatureCollection",
        'name': 'test',#geo['name'],
        'crs': geo['crs'],
        "features": filtered
    }
    
    df_section = dfm[dfm.season == selected_season][['division', 'section', selected_indicator]]
    df_section = df_section.sort_values(by=selected_indicator, ascending=False)
    df_section_division = df_section.loc[df_section['division']==selected_poly]
    col_name = df_section_division.columns[1]
    geo2plot = filtered_geojson
    df_map = df_section_division
    dfm_var = dfm[['season','division', 'section',selected_indicator]]
    df_chart = dfm_var.loc[dfm_var['division']==selected_poly]

    selected_sections_ids = [feature["properties"]['id'] for feature in filtered_geojson["features"]]
    dfca = dfc[(dfc['season'] == selected_year) & 
               (dfc['polygon_id'].isin(selected_sections_ids))].mean(numeric_only=True).to_frame().T.round(1)
else:
    col_name = df_division.columns[0]
    geo2plot = cm.merge_sections_to_divisions(geo, df_division)
    df_map = df_division

    dfm_var = dfm[['season','division', selected_indicator]].groupby(['season','division'])
    df_chart = dfm_var.agg({selected_indicator:'mean'}).reset_index()

    dfca = dfc[dfc['season'] == selected_year].mean(numeric_only=True).to_frame().T.round(1)


choropleth = cm.make_folium_choropleth(geo2plot, selected_indicator, df_map, 
                                       col_name)
line_chart, title = cm.make_alt_linechart(df_chart, selected_indicator, col_name, 
                                   selected_season, st.session_state.selected_section)
title = f'<p style="font:Courier; color:gray; font-size: 20px;">{title}</p>'

selected_name = 'Gezira'
if st.session_state.selected_division is not None:
    selected_name = st.session_state.selected_division

    
# piechart = cm.crop_area_piechart(dfca)
piechart, titlepie = cm.plotly_pie_chart(dfca, selected_name, selected_year)
titlepie = title = f'<p style="font:Courier; color:gray; font-size: 20px;">{titlepie}</p>'

with col[0]:
    # st.markdown('#####        Indicator Map')
    # st.markdown("<h4 style='text-align: center; color: white;'>Indicator Map</h4>", unsafe_allow_html=True)

    left, right = st.columns([0.7, 0.3])
    # left, right = st.columns((6, 2), gap='medium')
    with left:
        st.markdown("### Indicator Map")

    with right:
        # st.markdown("<br><br>", unsafe_allow_html=True) 
        st.markdown("<div style='margin-top: 12px;'>", unsafe_allow_html=True)
        if st.session_state.selected_polygon is not None:
            if st.button("🔄 Reset Map"):
                st.session_state.selected_block = None
                reset_map()
   
    map_data = st_folium(choropleth,  height=450, use_container_width=True)

    # st.write(map_data)
    st.write("")  
    st.markdown(title, unsafe_allow_html=True)  
    st.altair_chart(line_chart, use_container_width=True)
    
    if map_data and "last_clicked" in map_data and map_data["last_clicked"] != None :
        # Find the clicked polygon
        clicked_point = Point(map_data["last_clicked"]["lng"], map_data["last_clicked"]["lat"])
        matching_polygon = gdf[gdf.contains(clicked_point)]
        if not matching_polygon.empty:
            clicked_division = matching_polygon.iloc[0]['division']
            clicked_section = matching_polygon.iloc[0]['section']
            if (st.session_state.selected_division != clicked_division) or (st.session_state.selected_section != clicked_section):
                    st.session_state.selected_division = clicked_division
                    st.session_state.selected_section = clicked_section
                    st.rerun()           
    

with col[1]:
    # st.markdown('###### Bar chart of the selected indicator')
    # st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(titlepie, unsafe_allow_html=True)
    st.plotly_chart(piechart, use_container_width=True)
    chart, title  = cm.alt_bar_chart(df_map, selected_indicator, col_name, selected_season)
    title = f'<p style="font:Courier; color:gray; font-size: 20px;">{title}</p>'
    st.markdown(title, unsafe_allow_html=True)

    st.altair_chart(chart, use_container_width=True)

    
