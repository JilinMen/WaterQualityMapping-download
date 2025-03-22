import os
import sys
import datetime
from pathlib import Path
import streamlit as st
st.write("file path: ",os.path.dirname(__file__))
sys.path.append(r'/mount/src/waterqualitymapping/acolite')

from acolite import gee

## written by Quinten Vanhellemont, RBINS
def match_scenes(isodate_start, isodate_end=None, day_range=1,
                surface_reflectance=False,
                limit=None, st_lat=None, st_lon=None, filter_tiles=None,
                sensors=['L4_TM', 'L5_TM', 'L7_ETM', 'L8_OLI', 'L9_OLI', 'S2A_MSI', 'S2B_MSI']):
    import ee
    #ee.Authenticate() ## assume ee use is authenticated in current environment
    #ee.Initialize()

    import dateutil.parser, datetime

    if filter_tiles is not None:
        if type(filter_tiles) is not list:
            filter_tiles = [filter_tiles]

    ## check isodate
    if isodate_start is None:
        print('Please provide start date.')
        return()
    else:
        dstart = dateutil.parser.parse(isodate_start)
        isodate_start = dstart.isoformat()[0:10]

    ## get date range
    if isodate_start == isodate_end: isodate_end = None
    if isodate_end is None:
        dend = dstart + datetime.timedelta(days=0)
    else:
        if isodate_end in ['now', 'today']:
            dend = datetime.datetime.now()
        else:
            dend = dateutil.parser.parse(isodate_end)
    dend += datetime.timedelta(days=1) ## add one day so end date is included
    isodate_end = dend.isoformat()[0:10]

    print('Date range {} {}'.format(isodate_start, isodate_end))

    ## identify collections
    collections = []
    landsats = []
    ## MultiSpectral Scanners
    if 'L1_MSS' in sensors: landsats.append('LM01')
    if 'L2_MSS' in sensors: landsats.append('LM02')
    if 'L3_MSS' in sensors: landsats.append('LM03')
    if 'L4_MSS' in sensors: landsats.append('LM04')
    if 'L5_MSS' in sensors: landsats.append('LM05')

    ## newer sensors
    if 'L4_TM' in sensors: landsats.append('LT04')
    if 'L5_TM' in sensors: landsats.append('LT05')
    if 'L7_ETM' in sensors: landsats.append('LE07')
    if 'L8_OLI' in sensors: landsats.append('LC08')
    if 'L9_OLI' in sensors: landsats.append('LC09')
    landsat_tiers = ['T1', 'T2']
    landsat_collections = ['C02']

    for landsat in landsats:
        for tier in landsat_tiers:
            for coll in landsat_collections:
                if surface_reflectance:
                    if landsat[1] == 'M':
                        print('No SR for MSS.')
                    else:
                        collections.append('{}/{}/{}/{}_L2'.format('LANDSAT', landsat, coll, tier))
                else:
                    if landsat[1] == 'M':
                        collections.append('{}/{}/{}/{}'.format('LANDSAT', landsat, coll, tier))
                    else:
                        collections.append('{}/{}/{}/{}_TOA'.format('LANDSAT', landsat, coll, tier))

    if ('S2A_MSI' in sensors) or ('S2B_MSI' in sensors):
        ## harmonized has scenes from new processing shifted to old processing
        ## we take the offset into account in agh for >= PB4 data
        if surface_reflectance:
            #collections += ['COPERNICUS/S2_SR'] # COPERNICUS/S2_SR_HARMONIZED
            collections += ['COPERNICUS/S2_SR_HARMONIZED'] # COPERNICUS/S2_SR superseded by COPERNICUS/S2_SR_HARMONIZED in Jun 2024
        else:
            #collections.append('COPERNICUS/S2') # 'COPERNICUS/S2_HARMONIZED'
            collections.append('COPERNICUS/S2_HARMONIZED') # COPERNICUS/S2 superseded by COPERNICUS/S2_HARMONIZED in Jun 2024

    print('Checking collections {}'.format(' '.join(collections)))
    print(limit)
    ## set up region
    if limit is not None:
        region = ee.Geometry.BBox(limit[1], limit[0], limit[3], limit[2])
    elif (st_lon is not None) & (st_lat is not None):
        region = ee.Geometry.Point([st_lon, st_lat])
    else:
        print('Warning! No limit or st_lat, st_lon combination specified. Function may return too many images.')
        region = None
    ## set up ee date
    sdate=ee.Date(isodate_start)
    edate=ee.Date(isodate_end)
    ## search ee collections
    imColl = None

    for coll in collections:
        if 'LANDSAT' in coll:
            cloud_name = 'CLOUD_COVER'
        elif 'COPERNICUS' in coll:
            cloud_name = 'CLOUDY_PIXEL_PERCENTAGE'

        imC = ee.ImageCollection(coll).filterDate(sdate, edate).filter(ee.Filter.lt(cloud_name, 50))
        if region is not None: imC = imC.filterBounds(region)

        if imColl is None:
            imColl = imC
        else:
            imColl = imColl.merge(imC)
    iml = imColl.getInfo()
    nimages = len(iml['features'])
    images = []
    if nimages > 0:
        limages = imColl.toList(nimages).getInfo()
        for im in limages:
            if 'PRODUCT_ID' in im['properties']: ## Sentinel-2 image
                fkey = 'PRODUCT_ID'
                pid = im['properties'][fkey]
            elif 'LANDSAT_PRODUCT_ID' in im['properties']: ## Landsat image
                fkey = 'LANDSAT_PRODUCT_ID'
                pid = im['properties'][fkey]
            else: continue

            skip = False
            if filter_tiles is not None:
                skip = True
                for tile in filter_tiles:
                    if tile in pid: skip = False
            if skip: continue
            images.append((fkey,pid))
    return(images, imColl)

# Atmospheric correction: update gee_settings.txt
def update_settings(limit, isodate_start, isodate_end, sensor, output, output_scale,target_scale,glint_correction,
                    store_rhot,store_rhos,store_geom,store_sr,store_st,store_sp,
                    store_output_google_drive,
                    store_output_locally,
                    output_format,
                    old_agh=False,tile_size=606606):

    params = {}
    params["limit="] = ','.join(map(str,limit))
    params["isodate_start="] = isodate_start
    params["isodate_end="] = isodate_end
    params["sensors="] = sensor
    params["output="] = output
    params["convert_output="] = False
    params["output_scale="] = output_scale
    params["target_scale="] = target_scale
    params["glint_correction="] = glint_correction
    params["surface_reflectance="] = False
    params["store_rhot="] = store_rhot
    params["store_rhos="] = store_rhos
    params["store_geom="] = store_geom
    params["store_sr="] = store_sr
    params["store_st="] = store_st
    params["store_sp="] = store_sp
    params["store_output_google_drive="] = store_output_google_drive
    params["store_output_locally="] = store_output_locally
    params["output_format="] = output_format
    params["st_crop="] = False
    # write these parameters to the acolite/gee_settings.txt
    gee_settings = os.path.join('/content/acolite',"config/gee_settings.txt")

    try:
        with open(gee_settings,'r') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            for key,value in params.items():
                if line.startswith(key):
                    lines[i] = f"{key}{value}\n"
                    break
        with open(gee_settings, 'w') as file:
                file.writelines(lines)
        print('setting updated!')
    except Exception as e:
        import traceback
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]
        pymsg = ("PYTHON ERRORS:\nTraceback info:\n" + tbinfo +
                "\nError Info:\n" + str(sys.exc_info() [1]))
        print(pymsg)
        return

def ACOLITE_run(limit, isodate_start, isodate_end, sensor,
                output="/conetent/drive/MyDrive/ACOLITE/", output_scale=None,target_scale=None,glint_correction=True,
                store_rhot=False,store_rhos=True,store_geom=False,store_sr=False,store_st=False,store_sp=False,
                store_output_google_drive=False,
                store_output_locally=False,
                output_format=None
                ):
    update_settings(limit,
            isodate_start, isodate_end,
            sensor,
            output,
            output_scale, target_scale,
            glint_correction,
            store_rhot,store_rhos,store_geom,store_sr,store_st,store_sp,
            store_output_google_drive,
            store_output_locally,
            output_format
            )
    out_acolite = gee.agh_run(old_agh=False)
    return out_acolite

# RGB preview
def preview_rgb_image(collection,num_images = 10):
    if collection is None:
        print("No images found. Please search for images first.")
        return

    # Limit the collection to the first 'num_images' if necessary
    if collection.size().getInfo() > num_images:
        collection = collection.limit(num_images)

    # Get the list of images from the collection
    images = collection.toList(collection.size())

    # Get the collection size
    count = collection.size().getInfo()

    # # get the first image
    # first_image = ee.Image(collection.first())
    # image_date = ee.Date(first_image.get('system:time_start')).format('YYYY-MM-dd')

    # select RGB bands
    if atmospheric_correction.value == 'SR':
        if 'L8_OLI' in sensor.value[0] or 'L9_OLI' in sensor.value[0]:
            rgb_bands = ['SR_B4', 'SR_B3', 'SR_B2']
        elif 'S2A_MSI' in sensor.value[0] or 'S2B_MSI' in sensor.value[0]:
            rgb_bands = ['B4', 'B3', 'B2']
    else:
        rgb_bands = ['B4', 'B3', 'B2']

    # vislization parameters
    vis_params = {
        'bands': rgb_bands,
        'min': 0,
        'max': 0.3,  # reflectance range of  0-0.3
        'gamma': 1.4
    }

    for i in range(count):
        image = ee.Image(images.get(i))
        # image exists or not
        if image is None:
            print(f"Image at index {i} is null. Skipping.")
            continue
        image_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        print(f"Processing image {i + 1}/{count}: {image_date}")
        # add to map
        m.addLayer(image, vis_params, f"RGB_{image_date}")

def show_map(collect,algorithm,label='Chl mg/L',vis_params=None,num_images = 10):
    '''
    collect: ee.ImageCollection
    algorithm: water quality function
    vis_params: visualization parameters (optional)
    '''
    if label == "WaterClass":
        test_lambda = np.array([412,443,490,510,555,667,680])
    else:
        # Apply the algorithm to the image collection
        algo_collection = collect.map(algorithm)

    # Limit the collection to the first 'num_images' if necessary
    # print(algo_collection.size().getInfo())
    if algo_collection.size().getInfo() > num_images:
        algo_collection = algo_collection.limit(num_images)

    # Get the list of images from the collection
    images = algo_collection.toList(algo_collection.size())


    # Get the collection size
    count = algo_collection.size().getInfo()

    # Set default visualization parameters if not provided
    if vis_params is None:
        vis_params = {
            "min": 0,
            "max": 30,
            "palette": ["blue", "cyan", "green", "yellow", "red"]
        }

    # Iterate through the images and add them to the map
    for i in range(count):
        image = ee.Image(images.get(i))
        # image exists or not
        if image is None:
            print(f"Image at index {i} is null. Skipping.")
            continue

        image_date = ee.Date(image.get('system:time_start')).getInfo()
        if image_date is None:
            print("system:time_start is None, get time_start")
            image_date = ee.Date(image.get('time_start')).format('YYYY-MM-dd').getInfo()
        else:
            image_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()

        print(f"Processing image {i + 1}/{count}: {image_date}")

        # Add the image to the map
        try:
            print("Add water quality map to layer!")
            m.addLayer(image, vis_params, f"{label}_{image_date}")
        except Exception as e:
            print(f"Error adding image to the map: {e}")

    # Ensure colorbar is added only once per label
    if not hasattr(m, "added_labels") or not isinstance(m.added_labels, set):
        m.added_labels = set()

    if label not in m.added_labels:
        # Ensure 'colorbars' is a list to avoid AttributeError
        if hasattr(m, 'colorbars'):
            if isinstance(m.colorbars, set):
                m.colorbars = list(m.colorbars)
        else:
            m.colorbars = []

        m.add_colorbar(
            vis_params,
            label=label,
            orientation='horizontal',
            transparent_bg=True
        )
        m.added_labels.add(label)

    return algo_collection

def Chl_algorithm(image):
    '''
    John E. O'Reilly.RSE.Chlorophyll algorithms for ocean color sensors - OC4, OC5 & OC6. 2019
    '''
    print("Calculating Chlorophyll-a concentration...")
    try:
        if atmospheric_correction.value == 'SR':
            if 'S2A_MSI' in sensor.value[0] or 'S2B_MSI' in sensor.value[0]:
                blue1 = 'B1'
                blue2 = 'B2'
                green = 'B3'
            elif 'L8_OLI' in sensor.value[0] or 'L9_OLI' in sensor.value[0]:
                blue1 = 'SR_B1'
                blue2 = 'SR_B2'
                green = 'SR_B3'
            else:
                print("Unsupported sensor for chl calculation.")
                return None
        else:
            blue1 = 'B1'
            blue2 = 'B2'
            green = 'B3'

        B1 = image.select(blue1)
        B2 = image.select(blue2)
        G = image.select(green)
        X = (B1.max(B2)).divide(G).log10()

        # float to ee.Image.constant
        c0 = ee.Image.constant(0.30963)
        c1 = ee.Image.constant(-2.40052)
        c2 = ee.Image.constant(1.28932)
        c3 = ee.Image.constant(0.52802)
        c4 = ee.Image.constant(-1.33825)

        # model
        chl = ee.Image(10).pow(
              c0.add(X.multiply(c1))
              .add(X.pow(2).multiply(c2))
              .add(X.pow(3).multiply(c3))
              .add(X.pow(4).multiply(c4))
        )
        # Get the start_time and assign it to chl
        is_date_valid = image.propertyNames().contains('system:time_start')
        start_time = ee.Algorithms.If(is_date_valid, image.get("system:time_start"), image.get("time_start"))
        chl = chl.set("system:time_start", start_time)

        return chl.rename('Chl-a')
    except Exception as e:
        print(f"Error calculating Chl-a: {e}")
        return None
def TSS_algorithm(image):
    print("Calculating total suspended solid...")
    try:
        # band select
        if atmospheric_correction.value == 'SR':
            if 'S2A_MSI' in sensor.value[0] or 'S2B_MSI' in sensor.value[0]:
                green = 'B3'
                red = 'B4'
            elif 'L8_OLI' in sensor.value[0] or 'L9_OLI' in sensor.value[0]:
                green = 'SR_B3'
                red = 'SR_B4'
            else:
                print("Unsupported sensor for TSS calculation.")
                return None
        else:
            green = 'B3'
            red = 'B4'

        # bands
        G = image.select(green)
        R = image.select(red)

        # log transform
        log_G = G.log10()
        log_R = R.log10()

        # empeirical coefficients
        a = ee.Image.constant(1.5)
        b = ee.Image.constant(-1.2)
        c = ee.Image.constant(0.7)

        # TSS model
        TSS = ee.Image(10).pow(
            a.multiply(log_G)  # a * log10(G)
            .add(b.multiply(log_R))  # + b * log10(R)
            .add(c)  # + c
        )

        is_date_valid = image.propertyNames().contains('system:time_start')
        start_time = ee.Algorithms.If(is_date_valid, image.get("system:time_start"), image.get("time_start"))
        TSS = TSS.set("system:time_start", start_time)

        return TSS.rename('TSS')
    except Exception as e:
        print(f"Error calculating TSS: {e}")
        return None
def CDOM_algorithm(image):
    print("Calculating colored dissolved organic matter (CDOM)...")
    try:
        if atmospheric_correction.value == 'SR':
            if 'S2A_MSI' in sensor.value[0] or 'S2B_MSI' in sensor.value[0]:
                blue = 'B2'
                green = 'B3'
            elif 'L8_OLI' in sensor.value[0] or 'L9_OLI' in sensor.value[0]:
                blue = 'SR_B2'
                green = 'SR_B3'
            else:
                print("Unsupported sensor for CDOM calculation.")
                return None
        else:
            blue = 'B2'
            green = 'B3'

        B = image.select(blue)
        G = image.select(green)

        log_B = B.log10()
        log_G = G.log10()

        a = ee.Image.constant(1.2)
        b = ee.Image.constant(-0.8)
        c = ee.Image.constant(0.5)

        # CDOM model
        CDOM = ee.Image(10).pow(
            a.multiply(log_B)  # a * log10(B)
            .add(b.multiply(log_G))  # + b * log10(G)
            .add(c)  # + c
        )

        is_date_valid = image.propertyNames().contains('system:time_start')
        start_time = ee.Algorithms.If(is_date_valid, image.get("system:time_start"), image.get("time_start"))
        CDOM = CDOM.set("system:time_start", start_time)

        return CDOM.rename('CDOM')
    except Exception as e:
        print(f"Error calculating CDOM: {e}")
        return None
def extract_water_landsat(image):
    """
    extract water bodies using Landsat imagery
    """
    # # image ID
    # system_id = ee.String(image.get('system:id'))

    # # check Landsat 8/9
    # is_landsat89 = system_id.match('LANDSAT_8|LANDSAT_9').length().gt(0)

    # QA band
    qa_band = image.select('QA_PIXEL').toInt()

    water_bit = ee.Number(7) #7 is water in QA_PIXEL

    water_mask = qa_band.bitwiseAnd(ee.Number(1).leftShift(water_bit)).neq(0)
    # print(qa_band.getInfo())
    # print("water_mask: ",water_mask.propertyNames().getInfo())
    return water_mask

def extract_water_sentinel(image):
    """
    extract water bodies using Sentinel-2 imagery
    """
    # SCL
    scl = image.select('SCL')

    # extract water areas
    water_mask = scl.eq(6)
    # print("water_mask: ",water_mask.propertyNames().getInfo())
    return water_mask

def apply_cloud_mask_sentinel(image):
    """
    mask clouds and shadows with Sentinel-2
    """
    # SCL
    scl = image.select('SCL')

    # 3: cloud shadow, 8: cloud medium probability, 9: cloud high probability
    # If any of these conditions are true (cloud or shadow present), mask should be invalid
    invalid_mask = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9))
    # Invert to get clear pixels (1 for clear, 0 for cloudy/shadow)
    clear_mask = invalid_mask.Not()

    # print("water_mask: ",clear_mask.propertyNames().getInfo())
    return clear_mask

def apply_cloud_mask_landsat(image):
    """
    mask clouds and shadows with Landsat
    """
    # # Image ID
    # system_id = ee.String(image.get('system:id'))

    # # check Landsat 8/9
    # is_landsat89 = system_id.match('LANDSAT_8|LANDSAT_9').length().gt(0)

    # QA
    qa_band = image.select('QA_PIXEL').toInt()
    # print("cloud",qa_band.getInfo())
    # cloud and shadow bit
    cloud_bit = ee.Number(3)
    shadow_bit = ee.Number(4)

    # mask clouds
    cloud_mask = qa_band.bitwiseAnd(ee.Number(1).leftShift(cloud_bit)).eq(0)
    # mask shadows
    shadow_mask = qa_band.bitwiseAnd(ee.Number(1).leftShift(shadow_bit)).eq(0)

    # combine clouds and shadows
    mask = cloud_mask.Or(shadow_mask)
    # print("water_mask: ",mask.propertyNames().getInfo())
    return mask

def mask_water(image):
    """
    extract waters
    """
    if not image:
        raise ValueError("Input image is required")
    # try system:id then custom_id
    is_valid = image.propertyNames().contains('system:id')
    system_id = ee.String(ee.Algorithms.If(
        is_valid,
        image.get('system:id'),
        image.get('custom_id')
    ))

    # Landsat or Sentinel
    is_landsat = system_id.match('LANDSAT').length().gt(0)
    is_sentinel = system_id.match('COPERNICUS').length().gt(0)
    # print("is_landsat: ",is_landsat.getInfo())
    # print("is_sentinel: ",is_sentinel.getInfo())

    # water areas
    water_mask = ee.Algorithms.If(
        is_landsat,
        extract_water_landsat(image),
        ee.Algorithms.If(
            is_sentinel,
            extract_water_sentinel(image),
            image.updateMask(ee.Image.constant(0))
        )
    )

    # mask cloud and land
    cloud_mask = ee.Algorithms.If(
        is_landsat,
        apply_cloud_mask_landsat(image),
        ee.Algorithms.If(
            is_sentinel,
            apply_cloud_mask_sentinel(image),
            image.updateMask(ee.Image.constant(0))
        )
    )

    # combine clouds and land
    final_mask = ee.Image(water_mask).And(ee.Image(cloud_mask))
    # print("final_mask: ",final_mask.propertyNames().getInfo())
    # apply mask
    masked_image = image.updateMask(final_mask)
    # print("masked_image: ", masked_image.bandNames().getInfo())
    return masked_image

def scale_reflectance_landsat(image):
    """
    Notes:
        - Landsat 8/9 scale:0.0000275 offset:-0.2
        - Sentinel-2 scale:1/10000
    """
    landsat_bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
    scaled_image = (image
        .select(landsat_bands)
        .multiply(0.0000275)
        .add(-0.2)
        .copyProperties(image, image.propertyNames()))
    return image.addBands(scaled_image, landsat_bands, True)
def scale_reflectance_sentinel(image):
    # bands define
    sentinel_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
                     'B8A', 'B9', 'B11', 'B12']

    # transfer and copy properties
    scaled_image = (image
        .select(sentinel_bands)
        .multiply(0.0001)
        .copyProperties(image, image.propertyNames()))

    # add all other bands
    return image.addBands(scaled_image, sentinel_bands, True)

# @title merge by day
def merge_by_day(collection):
    """
    merge satellite images by day.

    collection should include properties at least:
        - system:time_start
        - system:id

    Returns:
        ee.ImageCollection: Mosaiced images
    """
    # obtain valid time_start
    is_date_valid = collection.first().propertyNames().contains('system:time_start')
    prop_date = ee.String(ee.Algorithms.If(
        is_date_valid,
        'system:time_start',
        'time_start'
    ))
    # get unique dates
    dates = collection.aggregate_array(prop_date) \
        .removeAll([None]) \
        .map(lambda time: ee.Date(time).format('YYYY-MM-dd')) \
        .distinct()

    # merge for the same day
    def fuse_images_by_date(date):
        date_obj = ee.Date(date)
        start_date = date_obj.millis()
        end_date = date_obj.advance(1, 'day').millis()

        # get images accroding to date
        # daily_images = collection.filterDate(start_date, end_date) #use default date system:time_start
        daily_images = collection.filter(ee.Filter.gte(prop_date, start_date)) \
                    .filter(ee.Filter.lt(prop_date, end_date))
        image_count = daily_images.size()

        # obtain valid id
        is_id_valid = daily_images.first().propertyNames().contains('system:id')
        prop_id = ee.String(ee.Algorithms.If(
            is_id_valid,
            'system:id',
            'custom_id'
        ))

        # get image ID
        image_ids = daily_images.aggregate_array(prop_id)

        # get bandNames
        band_names = ee.List(ee.Algorithms.If(
            image_count.gt(0),
            ee.Image(daily_images.first()).bandNames(),
            ee.List(["default_band"])  # avoid `None` error
        ))

        # image_count = 0
        no_images = ee.Image.constant(0) \
            .rename(band_names) \
            .set('system:time_start', date_obj.millis()) \
            .set('date', date) \
            .set('image_count', 0)\
            .set('custom_id',ee.List([]))

        # image_count = 1
        single_image = ee.Image(daily_images.first()) \
            .set('system:time_start', date_obj.millis()) \
            .set('date', date) \
            .set('image_count', 1)\
            .set('custom_id',image_ids.get(0))

        # image_count = 0 > 1
        fused_image = daily_images.reduce(ee.Reducer.mean()) \
            .rename(band_names) \
            .set('system:time_start', date_obj.millis()) \
            .set('date', date) \
            .set('image_count', image_count)\
            .set('custom_id',image_ids.get(0))  #use the first ID for fused image

        return ee.Algorithms.If(
            image_count.eq(0), no_images,
            ee.Algorithms.If(image_count.eq(1), single_image, fused_image)
        )

    # map
    fused_collection = ee.ImageCollection.fromImages(dates.map(fuse_images_by_date))

    # exclude image with image_count of 0
    return fused_collection.filter(ee.Filter.gt('image_count', 0))

# @title show water quality as layers
def show_wq(collection):
    """
    show water quality
    """
    if 'Chl-a' in bios.value:
        vis_params = {"min": 0,"max": 30,"palette": ["blue", "cyan", "green", "yellow", "red"]}
        label = "Chl-a"
        show_map(collection,Chl_algorithm,label,vis_params)
    if 'TSS' in bios.value:
        vis_params = {"min": 0,"max": 10,"palette": ["blue", "cyan", "green", "yellow", "red"]}
        label = "TSS"
        show_map(collection,TSS_algorithm,label,vis_params)
    if 'CDOM' in bios.value:
        vis_params = {"min": 0,"max": 2,"palette": ["blue", "cyan", "green", "yellow", "red"]}
        label = "CDOM"
        show_map(collection,CDOM_algorithm,label,vis_params)

def get_bounding_box(coordinates):
    # 获取最小和最大经纬度
    lats, lons = zip(*coordinates)
    min_lat = min(lats)
    max_lat = max(lats)
    min_lon = min(lons)
    max_lon = max(lons)
    
    return min_lat, min_lon, max_lat, max_lon

# 定义绘制事件处理函数
def on_draw(event):
    geom = event['geometry']  # 获取绘制的几何形状
    geom_type = geom['type']  # 获取绘制的图形类型（点、线、多边形等）
    
    if geom_type == 'Point':
        # 点形状：获取点的经纬度
        coordinates = geom['coordinates']
        
    elif geom_type == 'Polygon':
        # 多边形：获取外环的坐标（第一个坐标数组）
        coordinates = geom['coordinates'][0]
        min_lat, min_lon, max_lat, max_lon = get_bounding_box(coordinates)
        
    elif geom_type == 'Rectangle':
        # 矩形：矩形的四个角的坐标
        coordinates = geom['coordinates'][0]
        min_lat, min_lon, max_lat, max_lon = get_bounding_box(coordinates)
        
    return min_lat, min_lon, max_lat, max_lon
