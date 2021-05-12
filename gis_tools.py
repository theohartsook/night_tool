import geopandas as gpd
import math
import os
import pandas as pd
import subprocess

from shapely.geometry import Polygon, box

def detectionsFromPoints(input_csv, output_shp, buffer_col='r', epsg_code='EPSG:3310'):
    """This is a convenience function for converting my Hough detections to
    polygons based off radius.

    :param input_csv: Filepath to an input .csv created by Hough tools
    :type input_csv: str
    :param ouput_shp: Filepath to desired output .shp
    :type output_shp: str
    :param buffer_col: Used to show what variable should be used to determine the buffer, defaults to 'r' (radius)
    :type buffer_col: str
    :param epsg_code: The EPSG code of the input coordinate system, defaults to 'EPSG:3310'
    :type epsg_code: str

    :return: Returns 0 if file successfully created, otherwise returns 1.
    :rtype: int
    """

    df = pd.read_csv(input_csv)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    gdf.crs = epsg_code
    gdf['geometry'] = gdf.buffer(distance=df[buffer_col])
    gdf.to_file(output_shp, driver='ESRI Shapefile')
    if os.path.exists(output_shp):
        return 0
    else:
        return 1

def detectionsOnDir(input_dir, output_dir, csv_name='_real_coords.csv', buffer_col='r', epsg_code='EPSG:3310'):
    """This is a convenience function for bulk conversion. Needs some refactoring tbh.

    :param input_dir: Filepath to an input directory with .csv's created by Hough tools
    :type input_dir: str
    :param ouput_dir: Filepath to directory where output .shp's will be saved.
    :type output_dir: str
    :param csv_name: The ending target for .csv's, defaults to '_real_coords.csv' 
    :type csv_name: str
    :param buffer_col: Used to show what variable should be used to determine the buffer, defaults to 'r' (radius)
    :type buffer_col: str
    :param epsg_code: The EPSG code of the input coordinate system, defaults to 'EPSG:3310'
    :type epsg_code: str

    :return: Returns 0 if file successfully created, otherwise returns 1.
    :rtype: int
    """

    for i in sorted(os.listdir(input_dir)):
        if not i.endswith(csv_name):
            continue
        input_csv = input_dir + '/' + i
        print(i)
        output_shp = output_dir + '/' + i[:-27] + '.shp'
        print(input_csv)
        detectionsFromPoints(input_csv, output_shp, buffer_col=buffer_col, epsg_code=epsg_code)
        if not os.path.exists(output_shp):
            return 1
    return 0

def getCoordinates(input_raster):
    """This function uses GDAL to get the coordinates and pixel resolution from a georeferenced image.

    :param input_raster: Filepath to a georeferenced raster.
    :type input_raster: str

    :return: Returns a 3 element list containing the latitude, longitude, and pixel resolution.
    :rtype: list[lat, lon, pixel_res]
    """

    p = subprocess.Popen(['gdalinfo', input_raster], stdout=subprocess.PIPE)
    info = str(p.communicate())
    parsed_info = info.split('\\n')
    for i in parsed_info:
        if 'Pixel Size' in i:
            pixel_res = i[14:31]
        if 'Upper Left' in i:
            lat = i[14:25]
            lon = i[28:38]
    return_info = [lat, lon, pixel_res]
    return return_info

def pixelToReal(lat, lon, pixel_res, pix_x, pix_y, rad):
    """This function transforms pixel coordinates (and radius) into realworld coordinates.

    :param lat: Latitude of upper left pixel
    :type lat: float
    :param lon: Longitude of upper left pixel
    :type lon: float
    :param pixel_res: Pixel resolution of real world coordinates
    :type pixel_res: float
    :param pix_x: x position of pixel coordinate
    :type pix_x: int
    :param pix_y: y position of puxel coordinate
    :type pix_y: int

    :return: Returns a 3 element list containing the latitude, longitude, and radius in real world coordinates.
    :rtype: list[lat, lon, pixel_res]
    """

    new_lat = float(lat) - (pix_y * float(pixel_res))
    new_lon = float(lon) + (pix_x * float(pixel_res)) 
    new_rad = rad * float(pixel_res)
    new_line = [new_lon, new_lat, new_rad]
    return new_line

def applyPixelToReal(input_raster, input_csv, output_csv):
    """This is a convenience function to transform all detections in pixel coordinates to real world coordinates.

    :param input_raster: Filepath to a georeferenced raster.
    :type input_raster: str
    :param input_csv: Filepath to an input .csv with Hough detections.
    :type input_csv: str
    :param output_csv: Filepath to output .csv.
    :type output_csv: str

    :return: Returns 0 if file successfully created, otherwise returns 1.
    :rtype: int
    """

    lon, lat, pixel_res = getCoordinates(input_raster)
    output_df = pd.DataFrame(columns = ["plot", "x", "y", "r", "weight", "inner", "outer", "core"])
    df = pd.read_csv(input_csv)
    for index, row in df.iterrows():
        real_coords = pixelToReal(lat, lon, pixel_res, row["x"], row["y"], row["r"])
        output_df = output_df.append({"plot" : row["plot"], "x" : real_coords[0], "y" : real_coords[1], "r": real_coords[2], "weight" : row["weight"], "inner" : row["inner"], "outer" : row["outer"], "core" : row["core"]}, ignore_index=True)
    output_df.to_csv(output_csv, index=False)
    if os.path.exists(output_csv):
        return 0
    else:
        return 1

def findSketchyCircles(input_gdf, weight_thresh=0.4, size_quant=0.9):
    """This is a convenience function to select circles that need more review.

    :param input_gdf: A set of geometries with weights (output from Hough step 1).
    :type input_csv: geopandas GeoDataFrame
    :param weight_thresh: Any polygon with a weight >= to the threshold will be included.
    :type weight_thresh: float
    :param size_quant: The upper quantile of circle sizes to be included.
    :type size_quant: float

    :return: Returns a GeoDataFrame containing the circles needed for more review.
    :rtype: geopandas GeoDataFrame
    """

    high_weights_gdf = input_gdf[input_gdf.weight >= weight_thresh]

    quants = high_weights_gdf.r.quantile(size_quant)

    all_geom = high_weights_gdf.geometry.unary_union

    min_area = (quants*math.pi)**2

    merged_geom = gpd.GeoDataFrame(geometry=[all_geom])

    merged_geom = merged_geom.explode().reset_index(drop=True)

    target_df = merged_geom[merged_geom.geometry.area > min_area]

    return(target_df)

def polysToBoundingBoxes(input_gdf):
    """This is a convenience function to get the bounding boxes of a bunch of
       polygons. 

    :param input_gdf: A set of geometries with weights (output from Hough step 1).
    :type input_csv: geopandas GeoDataFrame

    :return: Returns a GeoDataFrame of the bounding boxes from the input.
    :rtype: geopandas GeoDataFrame
    """
    boxin = input_gdf.bounds

    coords = []

    for index, row in boxin.iterrows():
        ul_x = row['minx']
        ul_y = row['miny']
        lr_x = row['maxx']
        lr_y = row['maxy']

        bb = box(ul_x, ul_y, lr_x, lr_y)
        coords.append(bb)

    # make a new gpd to fill in box polygons

    boxout = gpd.GeoSeries()

    for i in coords:
        poly = gpd.GeoSeries(i)
        boxout = boxout.append(poly)

    output_df = gpd.GeoDataFrame(geometry=boxout)

    return(output_df)

def polyMasks(input_shp, img_path, output_dir):
    shp = gpd.read_file(input_shp)
    masks = shp.geometry

    with rasterio.open(img_path) as data:
        for index, row in masks.iterrows():
            poly = row.geometry
            masked_img, aff_tran = rasterio.mask.mask(data, [poly], crop=True)
            # save some stuff here

def pointsToGeoDataFrame(input_df, buffer_col=None, x='x', y='y', epsg_code='EPSG:3310'):
    gdf = gpd.GeoDataFrame(input_df, geometry=gpd.points_from_xy(input_df.x, input_df.y))
    gdf.crs = epsg_code
    if buffer_col is not None:
        gdf['geometry'] = gdf.buffer(distance=gdf[buffer_col])

    return gdf