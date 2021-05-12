from accuracy_tools import findGoodEnoughPredictions

import pandas as pd
import geopandas as gpd

pred_csv = '/Users/theo/data/hough_dataset/219_treeLS_comp.csv'
output_csv = '/Users/theo/data/hough_dataset/219_treeLS_comp_eval.csv'

pred = pd.read_csv(pred_csv)

findGoodEnoughPredictions(pred, output_csv)
