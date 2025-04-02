import streamlit as st
import folium
from streamlit_folium import st_folium
import geemap.foliumap as geemap
import ee
import folium.plugins
import json
from datetime import datetime
from branca.colormap import linear

# 读取 GEE 账号和密钥
service_account = st.secrets["GEE_SERVICE_ACCOUNT"]
private_key = st.secrets["GEE_PRIVATE_KEY"]

# 解析密钥
credentials = ee.ServiceAccountCredentials(service_account, key_data=private_key)
ee.Initialize(credentials)

# 选择 ROI
st.sidebar.header("选择参数")
lat = st.sidebar.number_input("纬度", value=34.0)
lon = st.sidebar.number_input("经度", value=-118.0)
radius = st.sidebar.slider("缓冲区半径 (km)", 1, 50, 10)

# 选择时间范围
start_date = st.sidebar.date_input("开始日期", datetime(2024, 1, 1))
end_date = st.sidebar.date_input("结束日期", datetime(2024, 3, 31))

# 构建 ROI
pos = ee.Geometry.Point([lon, lat])
roi = pos.buffer(radius * 1000)

# 选择影像集 (最新的 Landsat 8 SR 数据)
dataset = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
    .filterBounds(roi) \
    .filterDate(str(start_date), str(end_date)) \
    .median()

# 设定可视化参数
vis_params = {
    "bands": ["SR_B4", "SR_B3", "SR_B2"],  # RGB
    "min": 5000,
    "max": 30000,
    "gamma": 1.4
}
# 计算叶绿素指数 (Chlorophyll-a Proxy)
chl = dataset.expression(
    'B5 / B4',
    {
        'B5': dataset.select('SR_B5'),  # 近红外
        'B4': dataset.select('SR_B4')   # 红光
    }
)

# 设定可视化参数
chl_vis = {
    "min": 0.5,
    "max": 2,
    "palette": ["blue", "green", "yellow", "red"]
}


# 显示地图
Map = geemap.Map(center=[lat, lon], zoom=8)
# 显示地图
Map.addLayer(chl, chl_vis, "Chlorophyll-a Index")
Map.addLayer(dataset, vis_params, "Landsat SR")
Map.addLayer(roi, {}, "ROI")
Map.to_streamlit(height=500)