import pandas as pd
import geopandas as gpd
import math

from utility_tools import euclidDist, normalizeObservedData

def matchAtDifferentHeights(base_df, comp_df, comp_height, max_dist=0.5, max_diff=0.5):
    header = list(base_df)
    uid = str(comp_height) + '_uid'
    delta_xy = str(comp_height) + '_xy_dist'
    delta_r = str(comp_height) + '_r_diff'
    new_cols = [uid, delta_xy, delta_r]
    header.extend(new_cols)
    match_df = base_df.reindex(columns=header)
    for index1, row1 in match_df.iterrows():
        x1 = row1['x']
        y1 = row1['y']
        r1 = row1['r']
        best_match = 0
        best_sim = 100
        shortest_dist = 0
        smallest_diff = 0

        sub_df = comp_df[(comp_df['x'] <= x1+max_dist) & (comp_df['x'] >= x1-max_dist)]
        sub_df = sub_df[(sub_df['y'] <= y1+max_dist) & (sub_df['y'] >= y1-max_dist)]

        for index2, row2 in sub_df.iterrows():
            x2 = row2['x']
            y2 = row2['y']
            r2 = row2['r']
            dist = euclidDist(x2, y2, x1, y1)
            diff = abs(r2 - r1)
            sim = (dist + diff)
            if sim < best_sim:
                best_sim = sim
                best_match = row2['uid']
                shortest_dist = dist  
                smallest_diff = diff   
            else:
                continue

        if best_match != 0:
            match_df.loc[index1, uid] = best_match
            match_df.loc[index1, delta_xy] = shortest_dist
            match_df.loc[index1, delta_r] = smallest_diff
    return match_df     

def detectionAccuracyRF(obs_df, pred_df, out_csv):
    header = list(pred_df)
    uid = 'gt_uid'
    delta_xy = 'gt_xy_dist'
    delta_r = 'gt_r_diff'
    new_cols = [uid, delta_xy, delta_r]
    header.extend(new_cols)
    output_df = pred_df.reindex(columns=header)
    for index1, row1 in output_df.iterrows():
        xp = row1['x']
        yp = row1['y']
        rp = row1['r']
        best_match = 0
        best_sim = 100
        shortest_dist = 0
        smallest_diff = 0
        for index2, row2 in obs_df.iterrows():
            treeID, xo, yo, ro = normalizeObservedData(row2, 'Tree_Id', 'X_Location', 'Y_Location', 'DBH_m')
            if ro < 0.075:
                continue
            dist = euclidDist(xo, yo, xp, yp)
            diff = abs(ro - rp)
            sim = (dist + diff)
            if sim < best_sim:
                best_sim = sim
                best_match = treeID
                shortest_dist = dist  
                smallest_diff = diff
            else:
                continue
        if best_match != 0:
            output_df.loc[index1, uid] = best_match
            output_df.loc[index1, delta_xy] = shortest_dist
            output_df.loc[index1, delta_r] = smallest_diff
    output_df.to_csv(out_csv, index=False)    

def detectionAccuracy(obs_df, pred_df, out_csv):
    """ Matches each detection to the tree it best represents.

    :param obs_df: The ground truth data
    :type obs_df: pandas dataframe
    :param pred_df: Hough detections from calculatePixelMetrics
    :type pred_df: pandas dataframe
    :param output_csv: Filepath to save the best matches
    :type output_csv: str

    """

    matches = []
    for index1, row1 in pred_df.iterrows():
        xp = row1['x']
        yp = row1['y']
        rp = row1['r']
        best_match = 0
        best_sim = 100
        shortest_dist = 0
        smallest_diff = 0
        for index2, row2 in obs_df.iterrows():
            treeID, xo, yo, ro = normalizeObservedData(row2, 'Tree_Id', 'X_Location', 'Y_Location', 'DBH_m')
            if ro < 0.075:
                continue
            dist = euclidDist(xo, yo, xp, yp)
            diff = abs(ro - rp)
            sim = (dist + diff)
            if sim < best_sim:
                best_sim = sim
                best_match = treeID
                shortest_dist = dist  
                smallest_diff = diff
            else:
                continue
        best = [xp, yp, rp, row1['weight'], row1['inner'], row1['outer'], row1['core'], best_match, shortest_dist, smallest_diff]
        matches.append(best)
    
    header = ['x', 'y', 'r', 'weight', 'inner', 'outer', 'core', 'treeID', 'xy_dist', 'r_diff']
    output_df = pd.DataFrame(list(matches), columns=header)
    output_df.to_csv(out_csv, index=False)

def findBestPredictionsRF(obs_df, pred_df, output_csv):

    trees = obs_df.Tree_Id.unique()
    for i in sorted(trees):
        #detections = pred_df[(pred_df['gt_uid'] == i)]
        best_index = -1
        best_sim = 100
        print(i)
        for index, row in pred_df.iterrows():
            if row['gt_uid'] != i:
                continue
            sim = row['gt_xy_dist'] + row['gt_r_diff']
            if sim < best_sim:
                print('old sim: ', best_sim, '\nnew_sim: ', sim, '\n')
                best_sim = sim
                best_index = index   
            else: continue
        for index, row in pred_df.iterrows():
            if row['gt_uid'] != i:
                continue            
            elif index == best_index:
                pred_df.loc[index, 'gt_uid'] = best_index
            else:
                pred_df.loc[index, 'gt_uid'] = 0
    pred_df.to_csv(output_csv, index=False)   

def findBestPredictions(obs_df, pred_df, output_csv):
    """ Finds the detections that most closely match ground truth.

    :param obs_df: The ground truth data
    :type obs_df: pandas dataframe
    :param pred_df: Hough detections evaluated by detectionAccuracy
    :type pred_df: pandas dataframe
    :param output_csv: Filepath to save the best matches
    :type output_csv: str

    """

    matches = []
    trees = obs_df.Tree_Id.unique()
    for i in sorted(trees):
        detections = pred_df[pred_df['treeID'] == i]
        best_index = -1
        best_sim = 100
        for index, row in detections.iterrows():
            sim = row['xy_dist'] + row['r_diff']
            if sim < best_sim:
                best_sim = sim
                best_index = index   
            else: continue
        if best_index < 0:
            continue
        else:
            matches.append(pred_df.loc[best_index])
    header = ['x', 'y', 'r', 'weight', 'inner', 'outer', 'core', 'treeID', 'xy_dist', 'r_diff']
    output_df = pd.DataFrame(list(matches), columns=header)
    output_df.to_csv(output_csv, index=False)

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