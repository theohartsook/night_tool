import os
import pandas as pd
import geopandas as gpd

from utility_tools import bulkAssignIDs, assignIDs
from accuracy_tools import matchAtDifferentHeights, detectionAccuracyRF, findBestPredictionsRF

#input_csv = '/Users/theo/data/detections/3_00/TLS_0001_20170531_01_v003_clipped_40_classified_2.975_3.025/real_coords.csv'
#output_csv = '/Users/theo/data/razzle_dazzle.csv'


#input_dir = '/Users/theo/data/hough_dataset/augmented_pixel_coords_4'
#output_dir = '/Users/theo/data/hough_dataset/whatever'

base_csv = '/Users/theo/data/hough_dataset/predicted/TLS_0001_20170531_01_v003_clipped_40_classified_1.345_1.395_real_world_coords.csv'
three_csv = '/Users/theo/data/hough_dataset/augmented_pixel_coords_3/TLS_0001_20170531_01_v003_clipped_40_classified_2.975_3.025_real_world_coords.csv'
four_csv = '/Users/theo/data/hough_dataset/augmented_pixel_coords_4/TLS_0001_20170531_01_v003_clipped_40_classified_3.975_4.025_real_world_coords.csv'

'''
base_df = pd.read_csv(base_csv)
comp1_df = pd.read_csv(three_csv)
comp2_df = pd.read_csv(four_csv)

out1_df = matchAtDifferentHeights(base_df, comp1_df, 3)
out1_df.to_csv('/Users/theo/data/razzle.csv', index=False)

out2_df = matchAtDifferentHeights(out1_df, comp2_df, 4)
out2_df.to_csv('/Users/theo/data/dazzle.csv', index=False)
'''
pred_df = pd.read_csv('/Users/theo/data/mega_razzle.csv')
obs_df = gpd.read_file('/Users/theo/data/hough_dataset/observed/TLS_0001_20170531_01_v003_clipped_40_classified_1.345_1.395_dbh_species.shp')

#detectionAccuracyRF(obs_df, pred_df, '/Users/theo/data/mega_razzle.csv')
findBestPredictionsRF(obs_df, pred_df, '/Users/theo/data/dazzle_vmax.csv')