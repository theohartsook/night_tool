import pandas as pd
import geopandas as gpd
import math
import os

from utility_tools import euclidDist, normalizeObservedData, matchTLS, extractTLSID
from multiprocessing import Pool, Process, Queue



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

def findBestPredictionsRF(acc_df, output_csv):

    trees = acc_df.gt_uid.unique()
    for tree in sorted(trees):
        best_index = -1
        best_sim = 100
        detections = acc_df.loc[acc_df['gt_uid'] == tree]
        for index, row in detections.iterrows():
            uid = row['uid']
            sim = row['gt_xy_dist'] + row['gt_r_diff']
            if sim < best_sim:
                if best_index < 0:
                    print(best_index)
                    best_index = acc_df.loc[acc_df['uid'] == uid].index.item()
                    print(best_index)
                    acc_df.at[best_index, 'gt_uid'] = tree
                else:
                    acc_df.at[best_index, 'gt_uid'] = 0
                    best_index = acc_df.loc[acc_df['uid'] == uid].index.item()
                    acc_df.at[best_index, 'gt_uid'] = tree
                best_sim = sim
            else:
                bad_index = acc_df.loc[acc_df['uid'] == uid].index.item()
                acc_df.at[bad_index, 'gt_uid'] = 0



    acc_df.to_csv(output_csv, index=False)   

def findGoodEnoughPredictions(acc_df, output_csv):

    for index, row in acc_df.iterrows():
        xy_dist = row['gt_xy_dist']
        r_diff = row['gt_r_diff']
        if xy_dist <= 0.04:
            if r_diff <= 0.02:
                continue
        else:
            acc_df.at[index, 'gt_uid'] = 0
    acc_df.to_csv(output_csv, index=False)

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

def accuracyOnDirRF(pred_dir, obs_dir, output_dir, pred_target='.csv', obs_target='.shp', overwrite=False):
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
        detectionAccuracyRF(obs_df, pred_df, acc_csv)
        acc_df = pd.read_csv(acc_csv)
        findBestPredictionsRF(acc_df, best_csv)

def bestOnDirMP(acc_dir, output_dir, acc_target='.csv', overwrite=False, num_workers=8):
    q = Queue()
    for acc in sorted(os.listdir(acc_dir)):
        if not acc.endswith(acc_target):
            continue
        best_csv = output_dir + '/' + acc[:-4] + '_best.csv'
        if os.path.exists(best_csv) and overwrite==False:
            print('Skipping', acc, '...')
            continue
        acc_fp = acc_dir + '/' + acc
        inputs = [acc_fp, best_csv]
        q.put(inputs)
    workers = Pool(num_workers, bestQueue,(q,))
    workers.close()
    workers.join()

def goodEnoughOnDirMP(acc_dir, output_dir, target='.csv', overwrite=False, num_workers=8):
    q = Queue()
    for acc in sorted(os.listdir(acc_dir)):
        if not acc.endswith(target):
            continue
        good_csv = output_dir + '/' + acc[:-4] + '_good.csv'
        if os.path.exists(good_csv) and overwrite==False:
            print('Skipping', acc, '...')
            continue
        input_csv = acc_dir + '/' + acc
        inputs = [input_csv, good_csv]
        q.put(inputs)
    workers = Pool(num_workers, goodQueue,(q,))
    workers.close()
    workers.join()

def bestQueue(q):
    while not q.empty():
        try:
            inputs = q.get()
            acc_fp = inputs[0]
            best_csv = inputs[1]
            acc_df = pd.read_csv(acc_fp)
            findBestPredictionsRF(acc_df, best_csv)
        except ValueError as val_error:
            print(val_error)
        except Exception as error:
            print(error)

def goodQueue(q):
    while not q.empty():
        try:
            inputs = q.get()
            acc_fp = inputs[0]
            best_csv = inputs[1]
            acc_df = pd.read_csv(acc_fp)
            findGoodEnoughPredictions(acc_df, best_csv)
        except ValueError as val_error:
            print(val_error)
        except Exception as error:
            print(error)

def accuracyOnDirMP(pred_dir, obs_dir, output_dir, pred_target='.csv', obs_target='.shp', overwrite=False, num_workers=8):
    q = Queue()
    for pred in sorted(os.listdir(pred_dir)):
        if not pred.endswith(pred_target):
            continue
        pred_fp = pred_dir + '/' + pred
        obs_fp = matchTLS(pred, obs_dir, target=obs_target)
        if obs_fp == 'no match found':
            continue
        tls_id = extractTLSID(pred)
        acc_csv = output_dir + '/' + pred[:-4] + '_accuracy.csv'
        if os.path.exists(acc_csv):
            print('Skipping', tls_id, '...')
            continue
        inputs = [pred_fp, obs_fp, acc_csv]
        q.put(inputs)
    workers = Pool(num_workers, accuracyQueue,(q,))
    workers.close()
    workers.join()

def accuracyQueue(q):
    while not q.empty():
        try:
            inputs = q.get()
            pred_fp = inputs[0]
            obs_fp = inputs[1]
            acc_csv = inputs[2]
            obs_df = gpd.read_file(obs_fp)
            pred_df = pd.read_csv(pred_fp)
            detectionAccuracyRF(obs_df, pred_df, acc_csv)
        except ValueError as val_error:
            print(val_error)
        except Exception as error:
            print(error)

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