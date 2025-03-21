import ee
import geemap
from IPython.display import display
from ipyleaflet import WidgetControl, DrawControl, TileLayer
from geemap import Map

import streamlit as st
import warnings
import sys
sys.path.append('/mount/src/waterqualitymapping')
import waterquality_functions as wqf


st.set_page_config(layout="wide")
warnings.filterwarnings("ignore")

@st.cache_data
def ee_authenticate(token_name="EARTHENGINE_TOKEN"):
    geemap.ee_initialize(token_name=token_name)


st.sidebar.info(
    """
    - put something
    - GitHub repository: <https://github.com/giswqs/streamlit-geospatial>
    """
)

st.sidebar.title("Contact")
st.sidebar.info(
    """
    Jilin Men at jmen@ua.edu
    """
)


st.title('Quick View of Water Quality')
st.markdown(
"""
Quickly mapping chlorophyll-a, CDOM, turbidity for inland waters
"""
)
col1, col2 = st.columns([3,1])

height = 600

with col1:
    st.write("map here")
    m = Map(center=(35, -95), zoom=4, draw_control=False)
    
    # 定义计算最大最小经纬度的函数
    min_lat, min_lon, max_lat, max_lon = m.on_draw(wqf.on_draw)

    # 启用绘图控件
    m.add_draw_control()

with col2:
    st.write('Date range:')
    start_date = st.date_input("start_date")
    end_date = st.date_input("end_date")
    st.write('Coordinates:')
    min_lon = st.number_input("min lon")
    max_lon = st.number_input("max lon")
    min_lat = st.number_input("min lat")
    max_lat = st.number_input("max lat")

    sensor = st.multiselect('Sensor:',["L8_OLI","L9_OLI",'S2A_MSI','S2B_MSI'],default=['L8_OLI'])
    atmospheric_correction = st.selectbox("Atmospheric correction:",["SR","ACOLITE"],index=0)
    bios = st.multiselect("Bio-optical:",["Chl-a","TSS","CDOM"],default=['Chl-a'])

    button_run = st.button("Run")
    button_clear = st.button("Clear")

if button_run:
    global collection
    # global water_extracted_collection
    global collection_day

    print('Retrieving images!')

    images, imColl = wqf.match_scenes(
        start_date.value.isoformat(), end_date.value.isoformat(), day_range=1,
        surface_reflectance=True,
        limit=[min_lat.value, min_lon.value, max_lat.value, max_lon.value],
        st_lat=None, st_lon=None, filter_tiles=None,
        sensors=", ".join(sensor.value)
    )

    print("Total images:", len(images))
    print("Image list: ",imColl.aggregate_array('system:index').getInfo())
    print("Cloud cover: ",imColl.aggregate_array('CLOUD_COVER').getInfo())

    if len(images)==0:
    
        print('No image founded!')

    elif atmospheric_correction.value == 'SR':
        collection = imColl

        # transfer to surface reflectance
        if sensor.value[0] in ['S2A_MSI', 'S2B_MSI']:
            print('Input S2')
            collection_scaled = collection.map(wqf.scale_reflectance_sentinel)
        elif sensor.value[0] in ['L4_TM', 'L5_TM', 'L7_ETM', 'L8_OLI', 'L9_OLI']:
            print('Input Landsat')
            collection_scaled = collection.map(wqf.scale_reflectance_landsat)
        else:
            print("Unsupported sensor for reflectance conversion.",sensor.value)
            collection_scaled = collection

        # mosaic images on the same day
        # print("Band names before mosaic: ",collection_scaled.first().bandNames().getInfo())
        collection_day = wqf.merge_by_day(collection_scaled)
        # print("Band names after mosaic: ",collection_day.first().bandNames().getInfo())
        print("Total images after mosaic:", collection_day.size().getInfo())
        # print(collection_day.first().bandNames().getInfo())
        # mask clouds and land
        water_extracted_collection = collection_day.map(wqf.mask_water)
        print("Property names: ",water_extracted_collection.first().propertyNames().getInfo())
        print("Mosaic image list: ",water_extracted_collection.aggregate_array('custom_id').getInfo())
        # print("water_extracted_collection size: ",water_extracted_collection.size().getInfo())

        print("Band names after masking: ",water_extracted_collection.first().bandNames().getInfo())
        # RGB preview
        print('start to map RGB image!')
        wqf.preview_rgb_image(collection_day)
        print('start to map water quality parameters!')
        wqf.show_wq(water_extracted_collection)
        print("Processing complete!")
    elif atmospheric_correction.value == 'ACOLITE':
        # with status_output:
        print("Applying ACOLITE Atmospheric Correction...")
        collection = wqf.ACOLITE_run(
                    [min_lat.value, min_lon.value, max_lat.value, max_lon.value],
                    start_date.value.isoformat(), end_date.value.isoformat(),
                    ", ".join(sensor.value)
                    )
        # Ensure collection and imColl have the same start_time by merging metadata
        def merge_scl_or_qa_pixel(image, reference_image):
            if sensor.value == 'S2A_MSI' or sensor.value == 'S2B_MSI':
                flag_band = 'SCL'
            else:
                flag_band = 'QA_PIXEL'
            # Merge the SCL or QA_PIXEL from imColl to ACOLITE collection
            scl_or_qa_pixel = reference_image.select(flag_band).rename(flag_band)  # Or use QA_PIXEL if needed
            return image.addBands(scl_or_qa_pixel)

        # Apply the merging function to ensure that both collections have the same SCL/QA_PIXEL
        collection = collection.map(lambda image: merge_scl_or_qa_pixel(image,imColl.filterDate(image.get('time_start')).first()))
        
        print("Atmospheric correction complete!")
        
        collection_day = wqf.merge_by_day(collection)

        # mask clouds and land
        water_extracted_collection = collection_day.map(wqf.mask_water)
        print("Band names after masking: ",water_extracted_collection.first().bandNames().getInfo())

        # RGB preview
        print('start to map RGB image!')
        wqf.preview_rgb_image(collection_day)
        print('start to map water quality parameters!')
        wqf.show_wq(water_extracted_collection)
        print("Processing complete!")
    else:
        print("Unsupported atmospheric correction method.")


if button_clear:
    print("Clear successfully!")


