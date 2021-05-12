import pandas as pd
import geopandas as gpd

from gis_tools import pointsToGeoDataFrame
from shapely.geometry import Polygon

input_csv = '/Users/theo/data/hough_dataset/predicted_train.csv'

output_csv = '/Users/theo/data/hough_dataset/predicted_train_cleaned.csv'

epsg_code='EPSG:3310'

#output_shp = '/Users/theo/data/test_'

input_df = pd.read_csv(input_csv)

plots = input_df['plot'].unique()

output = gpd.GeoDataFrame()

for i in plots:
    df = input_df[(input_df['plot'] == i) & (input_df['y_pred'] == 1)]
    points = pointsToGeoDataFrame(df)
    points.crs = epsg_code
    gdf = pointsToGeoDataFrame(df, buffer_col='r')
    gdf.crs = epsg_code

    

    merged = gdf.geometry.unary_union
    merged = merged.buffer(distance=0.01)
    merged_geom = gpd.GeoDataFrame(geometry=[merged])

    merged_geom = merged_geom.explode().reset_index(drop=True)
    merged_geom.crs = epsg_code

    for index, row in merged_geom.iterrows():
        container = gpd.GeoSeries(row['geometry'])
        container = gpd.GeoDataFrame(geometry=container)
        container.crs = epsg_code
        poly = container.geometry[0]

        overlaps = points[points.within(poly)]


        if overlaps.shape[0] == 0:
            print('missed something')
            output = output.append(row)
            continue
        elif overlaps.shape[0] == 1:
            output = output.append(overlaps)
        else:
            best_weight = 0
            best_index = 0
            best = 0
            for index2, row2 in overlaps.iterrows():
                if row2['weight'] > best_weight:
                    best_weight = row2['weight']
                    best = row2
            output = output.append(best)
        
    #output.to_file(output_shp + str(i) + '.shp', driver='ESRI Shapefile')
final = pd.DataFrame(output.drop(columns='geometry'))
final.to_csv(output_csv, index=False)

