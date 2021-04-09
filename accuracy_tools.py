import pandas as pd
import geopandas as gpd
import math

def accuracyOnDir(pred_dir, obs_dir, output_dir, pred_target='.csv', obs_target='.shp', overwrite=False):
    for pred in sorted(os.listdir(pred_dir)):
        if not pred.endswith(pred_target):
            continue
        pred_fp = pred_dir + '/' + pred
        obs_fp = matchTLS(pred, obs_dir, target=obs_target)
        if obs_fp == 'no match found':
            continue
        tls_id = extractTLSID(pred)
        acc_csv = output_dir + '/' + pred[:-4] + '_accuracy.csv'
        best_csv = output_dir + '/' + pred[:-4] + '_best.csv'
        if os.path.exists(acc_csv):
            print('Skipping', tls_id, '...')
            continue
        pred_df = pd.read_csv(pred_fp)
        obs_df = gpd.read_file(obs_fp)
        detectionAccuracy(obs_df, pred_df, acc_csv)
        acc_df = pd.read_csv(acc_csv)
        findBestPredictions(obs_df, acc_df, best_csv)

def mergeMetrics(acc_dir, output_csv, target='accuracy.csv'):
    detections = []
    for i in sorted(os.listdir(acc_dir)):
        if not i.endswith(target):
            continue
        df = pd.read_csv(acc_dir + '/' + i)
        plot = extractTLSID(i)
        for index, row in df.iterrows():
            detection = [plot, row['x'], row['y'], row['r'], row['weight'], row['inner'], row['outer'], row['core'], row['treeID'], row['xy_dist'], row['r_diff']]
            detections.append(detection)


    header = ['plot', 'x', 'y', 'r', 'weight', 'inner', 'outer', 'core', 'treeID', 'xy_dist', 'r_diff']
    output_df = pd.DataFrame(list(detections), columns=header)
    output_df.to_csv(output_csv, index=False)    

def mergeGroundTruth(obs_dir, output_csv, target='.shp'):
    observed = []
    for i in sorted(os.listdir(obs_dir)):
        if not i.endswith(target):
            continue
        df = gpd.read_file(obs_dir + '/' + i)
        plot = extractTLSID(i)
        for index, row, in df.iterrows():
            observation = [plot, row['Tree_Id'], row['X_Location'], row['Y_Location'], row['DBH_m']]
            observed.append(observation)
    header = ['plot', 'Tree_Id', 'X_Location', 'Y_Location', 'DBH_m']
    output_df = pd.DataFrame(list(observed), columns=header)
    output_df.to_csv(output_csv, index=False)   