from gis_tools import findSketchyCircles, polysToBoundingBoxes
from hough_tools import loopApplyHough
from scipy_vision_tools import saveImage, getDistanceRaster, ridgeDetection, connectedComponentsImage, exportBlobs, edgeDetection, morphologicalCloseImage, skeletonizeImage, getImageThreshold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
from scipy import ndimage

'''
input_csv = '/Users/theo/data/vision_pipeline_mp_test/metrics_coords/TLS_0001_20170531_01_v003_clipped_40_classified_1.345_1.395_real_world_coords.csv'
epsg_code = 'EPSG:3310'
buffer_col = 'r'
group_col = 'plot'
output_shp = '/Users/theo/data/vision_pipeline_mp_test/more_tests_box.shp'

df = pd.read_csv(input_csv)

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
gdf.crs = epsg_code
gdf['geometry'] = gdf.buffer(distance=df[buffer_col])

sketch_gdf = findSketchyCircles(gdf, weight_thresh=0.4, size_quant=0.9)

boxes_gdf = polysToBoundingBoxes(sketch_gdf)

boxes_gdf.plot(cmap='cividis', alpha=0.7, edgecolor='black')
plt.show()
'''

boxes = '/Users/theo/data/vision_pipeline_mp_test/more_tests_box.shp'
img = '/Users/theo/data/registered_rasters/1_37/TLS_0001_20170531_01_v003_clipped_40_classified_1.345_1.395.tif'
out_root = '/Users/theo/data/vision_pipeline_mp_test/forever/'

PARAMS = [[7,10,18],[11,15,18],[16,20,12],[21,25,12],[26,30,8],[31,35,8],[36,40,8],[41,45,6],[46,50,6],[51,55,4],[56,60,4],[61,65,4],[66,70,4],[71,75,4],[76,80,4],[81,85,4],[86,90,4],
[91,95,4],[96,100,4],[101,250,1],[251,500,1],[501,1000,1]]

boxes = gpd.read_file(boxes)
box_in = boxes.geometry

with rasterio.open(img) as data:
    i = 0
    for index, row in boxes.iterrows():
        out_csv = out_root + str(i) + '.csv'
        mask_tif = out_root + str(i) + '_mask.tif'
        dist_tif = out_root + str(i) + '_dist.tif'
        ridge_tif = out_root + str(i) + '_ridge.tif'
        edge_tif = out_root + str(i) + '_edge.tif'
        close_tif = out_root + str(i) + '_closed.tif'
        skel_tif = out_root + str(i) + '_skel.tif'
        thresh_tif = out_root + str(i) + '_thresh.tif'
        median_tif = out_root + str(i) + '_median.tif'
        unif_tif = out_root + str(i) + '_unif.tif'
        combo_tif = out_root + str(i) + '_combo.tif'


        poly = row.geometry
        sub, dub = rasterio.mask.mask(data, [poly], crop=True)
        sub = np.squeeze(sub)
        saveImage(sub, mask_tif)
        med_fil_2 = ndimage.median_filter(sub, size=2)
        saveImage(med_fil_2,  out_root + str(i) + '_med_2.tif')
        max_fil = ndimage.maximum_filter(sub, size=5)
        saveImage(max_fil,  out_root + str(i) + '_max_5.tif')
        combo = max_fil * med_fil_2
        saveImage(combo, combo_tif)
        redux = ndimage.median_filter(combo, size=2)
        saveImage(redux, out_root + str(i) + '_redux.tif')
        closed = morphologicalCloseImage(redux, 'np', stel=5)
        saveImage(closed, close_tif)
        edges = edgeDetection(closed, 'np')
        saveImage(edges, edge_tif)
        skel = skeletonizeImage(edges, 'np')
        saveImage(skel, skel_tif)  
        np_blobs = connectedComponentsImage(skel, 'np')
        exportBlobs(redux, np_blobs, 'np', out_root + str(i))
        plt.imsave(out_root + str(i) + '_blobs_cmap.png', np_blobs, cmap='nipy_spectral')
        
        '''
        med_fil_2 = ndimage.median_filter(sub, size=2)
        saveImage(med_fil_2,  out_root + str(i) + '_med_2.tif')
        max_fil = ndimage.maximum_filter(sub, size=5)
        saveImage(max_fil,  out_root + str(i) + '_max_5.tif')
        combo = max_fil * med_fil_2
        saveImage(combo, combo_tif)
        closed = morphologicalCloseImage(combo, 'np', stel=5)
        saveImage(closed, close_tif)
        edges = edgeDetection(closed, 'np')
        saveImage(edges, edge_tif)
        skel = skeletonizeImage(edges, 'np')
        saveImage(skel, skel_tif)  
        np_blobs = connectedComponentsImage(skel, 'np')
        plt.imsave(out_root + str(i) + '_blobs_cmap.png', np_blobs, cmap='nipy_spectral')
        '''

        #loopApplyHough(skel, out_csv, PARAMS)
        i += 1
    '''
    sub, dub = mask(data, boxes.geometry)
    print(type(sub))
    sub = np.squeeze(sub)
    plt.imshow(sub)
    plt.show()
    '''