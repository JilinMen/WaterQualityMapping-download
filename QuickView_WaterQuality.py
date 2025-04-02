import ee
import geemap
from IPython.display import display
from ipyleaflet import WidgetControl, DrawControl, TileLayer
from geemap.foliumap import Map

import streamlit as st
import warnings
import sys, os
sys.path.append(os.path.dirname(__file__))
import waterquality_functions as wqf
from typing import Any, Optional, Dict
import fiona
import geopandas as gpd
import numpy as np
import datetime

st.set_page_config(layout="wide")
warnings.filterwarnings("ignore")

# @st.cache_data
# def ee_authenticate(token_name="EARTHENGINE_TOKEN"):
#     geemap.ee_initialize(token_name=token_name)
def uploaded_file_to_gdf(data):
    import tempfile
    import os
    import uuid

    _, file_extension = os.path.splitext(data.name)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(tempfile.gettempdir(), f"{file_id}{file_extension}")

    with open(file_path, "wb") as file:
        file.write(data.getbuffer())

    if file_path.lower().endswith(".kml"):
        fiona.drvsupport.supported_drivers["KML"] = "rw"
        gdf = gpd.read_file(file_path, driver="KML")
    else:
        gdf = gpd.read_file(file_path)

    return gdf

# st.sidebar.info(
#     """
#     - put something
#     - GitHub repository: <https://github.com/giswqs/streamlit-geospatial>
#     """
# )
#
# st.sidebar.title("Contact")
# st.sidebar.info(
#     """
#     Jilin Men at jmen@ua.edu
#     """
# )
st.title('Quick View of Water Quality')
st.markdown(
"""
Quickly mapping chlorophyll-a, CDOM, turbidity, and total suspended sediment for inland waters
"""
)

# ee_authenticate(token_name="EARTHENGINE_TOKEN")
# 读取 GEE 账号和密钥
service_account = st.secrets["GEE_SERVICE_ACCOUNT"]
private_key = st.secrets["GEE_PRIVATE_KEY"]

# 解析密钥
credentials = ee.ServiceAccountCredentials(service_account, key_data=private_key)
ee.Initialize(credentials)

with st.expander(
    "Please draw a rectangle on the map -> Export it as a GeoJSON -> Upload it back to the app -> Click the Submit button. Expand this tab to see a demo👇"
):
    video = st.empty()
    video.video("https://youtu.be/n8qGTBnSBYE")
data = st.file_uploader(
    "Upload a GeoJSON file to use as an ROI. Customize parameters and then click the Submit button👇",
    type=["geojson", "kml", "zip"],
    )

col1, col2, col3 = st.columns([6,1,1])

# 初始化 session_state 变量
default_values = {
    "min_lon": 0,
    "max_lon": 0,
    "min_lat": 0,
    "max_lat": 0,
    "sensor": "L8_OLI",
    "atmospheric_correction": "ACOLITE",
    "bios": ["Chl-a"],
    "chl_low": 0,
    "chl_up": 1,
    "tss_low": 0,
    "tss_up": 1,
    "cdom_low":0,
    "cdom_up":1,
    "turbidity_low":0,
    "turbidity_up":1,
}

# 检查并设置缺失的 session_state 变量
for key, default_value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

st.session_state['m'] = Map(center=(35, -95), zoom=4, Draw_export=True)

if data:
    gdf = uploaded_file_to_gdf(data)
    st.session_state["roi"] = geemap.gdf_to_ee(gdf, geodesic=False)
    st.session_state['m'].add_gdf(gdf, "ROI")
    bounds = np.array(gdf.bounds)

    # 更新 session_state 中的值（这里要用 `st.session_state.update` 避免冲突）
    st.session_state.update({
        "min_lon": bounds[0][0],
        "min_lat": bounds[0][1],
        "max_lon": bounds[0][2],
        "max_lat": bounds[0][3]
    })

with col2:
    st.write('Date range ##########')
    start_date = st.date_input("start_date:",value=datetime.date.today() - datetime.timedelta(days=30))

    st.write('Coordinates #########')
    # 创建输入框，并绑定到 session_state，同时使用 on_change 回调
    st.number_input("min_lon:", value=st.session_state["min_lon"], key="min_lon")
    st.number_input("min_lat:", value=st.session_state["min_lat"], key="min_lat")
    st.selectbox('Sensor:',["L8_OLI","L9_OLI","S2A_MSI","S2B_MSI"],index=0,key="sensor")
    st.multiselect("Bio-optical:",["Chl-a","TSS","CDOM","Turbidity"],default=st.session_state['bios'],key='bios')
    button_run = st.button("Submit")

with col3:
    st.write("####################")
    end_date = st.date_input("end_date:", value=datetime.date.today())
    st.write("####################")
    st.number_input("max_lon:", value=st.session_state["max_lon"], key="max_lon")
    st.number_input("max_lat:", value=st.session_state["max_lat"], key="max_lat")
    st.selectbox("Atmospheric correction:", ["ACOLITE", "SR"], index=0, key="atmospheric_correction")
    st.write("####################")
    button_download = st.button("Download")
    button_clear = st.button("Reset")

if button_clear:

    st.success("All parameters have been reset!")

if button_run:
    images, imColl = wqf.match_scenes(
        start_date.isoformat(), end_date.isoformat(), day_range=1,
        surface_reflectance=True,
        limit=[st.session_state["min_lat"], st.session_state["min_lon"], st.session_state["max_lat"], st.session_state["max_lon"]],
        st_lat=None, st_lon=None, filter_tiles=None,
        sensors=st.session_state['sensor']
    )
    st.session_state['filename'] = imColl.aggregate_array('system:index').getInfo()
    st.write("Total images:", len(images))
    # st.write("Image list: ",imColl.aggregate_array('system:index').getInfo())
    # st.write("Cloud cover: ",imColl.aggregate_array('CLOUD_COVER').getInfo())

    if len(images)==0:
    
        st.warning('No image founded! Please change the parameters and resubmit')

    else:
        if st.session_state['atmospheric_correction'] == 'SR':
            collection = imColl

            # transfer to surface reflectance
            if st.session_state['sensor'] in ['S2A_MSI', 'S2B_MSI']:
                print('Input S2')
                collection_scaled = collection.map(wqf.scale_reflectance_sentinel)
            elif st.session_state['sensor'] in ['L4_TM', 'L5_TM', 'L7_ETM', 'L8_OLI', 'L9_OLI']:
                print('Input Landsat')
                collection_scaled = collection.map(wqf.scale_reflectance_landsat)
            else:
                print("Unsupported sensor for reflectance conversion.",st.session_state['sensor'])
                collection_scaled = collection

            # mosaic images on the same day
            # print("Band names before mosaic: ",collection_scaled.first().bandNames().getInfo())
            collection_day = wqf.merge_by_day(collection_scaled)
            # print("Band names after mosaic: ",collection_day.first().bandNames().getInfo())
            print("Total images after mosaic:", collection_day.size().getInfo())
            # print(collection_day.first().bandNames().getInfo())
            # mask clouds and land
            water_extracted_collection = collection_day.map(wqf.mask_water)

            st.session_state['collection'] = collection_day

            print("Property names: ",water_extracted_collection.first().propertyNames().getInfo())
            print("Mosaic image list: ",water_extracted_collection.aggregate_array('custom_id').getInfo())
            # print("water_extracted_collection size: ",water_extracted_collection.size().getInfo())

            print("Band names after masking: ",water_extracted_collection.first().bandNames().getInfo())
            # RGB preview
            print('start to map RGB image!')
            wqf.preview_rgb_image(collection_day)
            print('start to map water quality parameters!')
            bios_results = wqf.show_wq(water_extracted_collection)

            print("Processing complete!")
        elif st.session_state['atmospheric_correction'] == 'ACOLITE':
            # st.write("Applying ACOLITE Atmospheric Correction...")
            collection = wqf.ACOLITE_run(
                        [st.session_state["min_lat"], st.session_state["min_lon"], st.session_state["max_lat"], st.session_state["max_lon"]],
                        start_date.isoformat(), end_date.isoformat(),
                        st.session_state['sensor']
                        )
            print("Atmospheric correction complete!")
            print("collection after acolite: ", collection.first().bandNames().getInfo())
            # print("crs of imcoll: ",collection.first().propertyNames().getInfo())

            def transfer_properties(image):
                # inherit properties from collection
                time_start = image.get("time_start")
                matched_image = imColl.filter(ee.Filter.eq("system:time_start", time_start)).first()  # 在 B 中查找匹配影像

                # 如果找到匹配影像，则继承其几何边界，并复制 system:index
                def apply_changes(matched_image):
                    return (image
                            .clip(matched_image.geometry())
                            .copyProperties(matched_image, matched_image.propertyNames())
                            .set("system:index", matched_image.get("system:index")))

                return ee.Image(ee.Algorithms.If(matched_image, apply_changes(matched_image), image))

            collection = collection.map(transfer_properties)
            print("before: ", imColl.aggregate_array("system:index").getInfo())
            print("after: ",collection.aggregate_array("system:index").getInfo())

            print("properties of inherit: ", collection.first().getInfo())

            # Ensure collection and imColl have the same start_time by merging metadata
            def merge_scl_or_qa_pixel(image, reference_image):
                if 'S2A_MSI' in st.session_state['sensor'] or 'S2B_MSI' in st.session_state['sensor']:
                    flag_band = 'SCL'
                elif "L8_OLI" in st.session_state['sensor'] or "L9_OLI" in st.session_state['sensor']:
                    flag_band = 'QA_PIXEL'
                else:
                    print("Sensor can't be identified: ",st.session_state['sensor'])
                # Merge the SCL or QA_PIXEL from imColl to ACOLITE collection
                scl_or_qa_pixel = reference_image.select(flag_band).rename(flag_band)  # Or use QA_PIXEL if needed
                return image.addBands(scl_or_qa_pixel)

            # Apply the merging function to ensure that both collections have the same SCL/QA_PIXEL
            collection = collection.map(lambda image: merge_scl_or_qa_pixel(image,imColl.filterDate(image.get('time_start')).first()))

            collection_day = wqf.merge_by_day(collection)
            # print('collection_day: ', collection_day.aggregate_array('system:id'))

            # mask clouds and land
            water_extracted_collection = collection_day.map(wqf.mask_water)

            st.session_state['collection'] = collection_day

            print("collection_day size: ", collection_day.size().getInfo())
            print("Band names after masking: ",water_extracted_collection.first().bandNames().getInfo())

            # RGB preview
            print('start to map RGB image!')
            wqf.preview_rgb_image(collection_day)
            print('start to map water quality parameters!')
            bios_results = wqf.show_wq(water_extracted_collection)
            print("Processing complete!")
        else:
            print("Unsupported atmospheric correction method.")

        if "Chl-a" in st.session_state['bios']:
            st.session_state['vis_chl'] = {
                "width": 2.5,
                "height": 0.3,
                "vmin": st.session_state['chl_low'],  # 颜色条的最小值
                "vmax": st.session_state['chl_up'],  # 颜色条的最大值
                "orientation": "horizontal",
                "label": "Chl-a (mg/L)",
                # "cmap": "jet",
                "palette": ["#7400b8", "#5e60ce", "#56cfe1", "#80ffdb", "#38b000", "#006400", "#ffb627", "#f85e00",
                            "#800f2f"],  # 颜色渐变
            }
            st.session_state['m'].add_colormap(position=(73, 4), **st.session_state['vis_chl'])

        if "TSS" in st.session_state['bios']:
            st.session_state['vis_tss'] = {  # 可视化参数
                "width": 2.5,
                "height": 0.3,
                "vmin": st.session_state['tss_low'],  # 颜色条的最小值
                "vmax": st.session_state['tss_up'],  # 颜色条的最大值
                "orientation": "horizontal",
                "label": "TSS (g/L)",
                # "cmap": "winter",
                "palette": ["#7400b8", "#5e60ce", "#56cfe1", "#80ffdb", "#38b000", "#006400", "#ffb627", "#f85e00",
                            "#800f2f"],  # 颜色渐变
            }
            st.session_state['m'].add_colormap(position=(73, 18), **st.session_state['vis_tss'])
        if "CDOM" in st.session_state['bios']:
            st.session_state['vis_cdom'] = {
                "width": 2.5,
                "height": 0.3,
                "vmin": st.session_state['cdom_low'],  # 颜色条的最小值
                "vmax": st.session_state['cdom_low'],  # 颜色条的最大值
                "orientation": "horizontal",
                "label": "CDOM (m-1)",
                # "cmap": "rainbow",
                "palette": ["#7400b8", "#5e60ce", "#56cfe1", "#80ffdb", "#38b000", "#006400", "#ffb627", "#f85e00",
                            "#800f2f"],  # 颜色渐变
            }

            st.session_state['m'].add_colormap(position=(73, 32), **st.session_state['vis_cdom'])

        if "Turbidity" in st.session_state['bios']:
            st.session_state['vis_turbidity'] = {
                "width": 2.5,
                "height": 0.3,
                "vmin": st.session_state['turbidity_low'],
                "vmax": st.session_state['turbidity_up'],
                "orientation": "horizontal",
                "label": "Turbidity (NTU)",
                # "cmap": "rainbow",
                "palette": ["#7400b8", "#5e60ce", "#56cfe1", "#80ffdb", "#38b000", "#006400", "#ffb627", "#f85e00",
                            "#800f2f"],
            }
            st.session_state['m'].add_colormap(position=(73, 46), **st.session_state['vis_turbidity'])

if button_download:
    import traceback
    try:
        type_data = type(st.session_state['collection'])
        out_dir = "./download"

        if not os.path.exists(out_dir): os.mkdir(out_dir)
        print("data type: ", type_data)

        if type_data == ee.ImageCollection:
            geemap.download_ee_image_collection(st.session_state['collection'],out_dir,filenames=st.session_state['filename'],scale=100)
        elif type_data == ee.Image:
            geemap.download_ee_image(st.session_state['collection'], "landsat-test.tif", scale=100)
        else:
            print("data type error")
    except Exception as e:
        st.warning(f"Downloading failed! Error: {e}")
        traceback.print_exc()

with col1:

    component = st.session_state['m'].to_streamlit(height=600)





