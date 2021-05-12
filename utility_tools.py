import math
import os
import re
import shutil

import pandas as pd

from sklearn import model_selection
from multiprocessing import Pool, Process, Queue

def heightMatchOnDir(base_dir, comp_dir, output_dir, comp_height, max_dist=0.5, max_diff=0.5):
    for i in sorted(os.listdir(base_dir)):
        input_csv = base_dir + '/' + i
        comp_csv = matchTLS(i, comp_dir, target='.csv')
        if comp_csv == 'no match found':
            continue
        input_df = pd.read_csv(input_csv)
        comp_df = pd.read_csv(comp_csv)
        print(input_csv, comp_csv)
        output_csv = output_dir + '/' + i
        output_df = matchAtDifferentHeights(input_df, comp_df, comp_height, max_dist=max_dist, max_diff=max_diff)
        output_df.to_csv(output_csv, index=False)

def heightMatchQueue(q, comp_height, max_dist, max_diff):
    while not q.empty():
        try:
            inputs = q.get()
            input_csv = inputs[0]
            comp_csv = inputs[1]
            output_csv = inputs[2]
            input_df = pd.read_csv(input_csv)
            comp_df = pd.read_csv(comp_csv)
            output_df = matchAtDifferentHeights(input_df, comp_df, comp_height, max_dist=max_dist, max_diff=max_diff)
            output_df.to_csv(output_csv, index=False)  
        except ValueError as val_error:
            print(val_error)
        except Exception as error:
            print(error)

def heightMatchOnDirMP(base_dir, comp_dir, output_dir, comp_height, max_dist=0.5, max_diff=0.5, num_workers=8):
    q = Queue()
    for i in sorted(os.listdir(base_dir)):
        input_csv = base_dir + '/' + i
        comp_csv = matchTLS(i, comp_dir, target='.csv')
        if comp_csv == 'no match found':
            continue
        output_csv = output_dir + '/' + i
        inputs = [input_csv, comp_csv, output_csv]
        q.put(inputs)
    workers = Pool(num_workers, heightMatchQueue,(q, comp_height, max_dist, max_diff))
    workers.close()
    workers.join()

def matchAtDifferentHeights(base_df, comp_df, comp_height, max_dist=0.5, max_diff=0.5):
    header = list(base_df)
    uid = str(comp_height) + '_uid'
    height_weight = str(comp_height) + '_weight'
    height_inner = str(comp_height) + '_inner'
    height_outer = str(comp_height) + '_outer'
    height_core = str(comp_height) + '_core'
    delta_xy = str(comp_height) + '_xy_dist'
    delta_r = str(comp_height) + '_r_diff'
    new_cols = [uid, height_weight, height_inner, height_outer, height_core, delta_xy, delta_r]
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
                weight = row2['weight']
                inner = row2['inner']
                outer = row2['outer']
                core = row2['core']
            else:
                continue

        if best_match != 0:
            match_df.loc[index1, uid] = best_match
            match_df.loc[index1, delta_xy] = shortest_dist
            match_df.loc[index1, delta_r] = smallest_diff
            match_df.loc[index1, height_weight] = weight
            match_df.loc[index1, height_inner] = inner
            match_df.loc[index1, height_outer] = outer
            match_df.loc[index1, height_core] = core
    return match_df     

def matchTLS(input_file, target_dir, target='.tif'):
    """ This is a convenience function to match two files linked by TLS ID.

    :param input_file: Filename with the TLS ID to be matched.
    :type input_file: str
    :param target_dir: Filepath to the directory to look for matches.
    :type: target_dir: str
    :param target: The file ending for valid inputs, defaults to '.tif'
    :type target: str

    :return: Returns the full filepath to the matching file, or 'no match found.'
    :rtype: str
    """
    
    match = extractTLSID(input_file)
    for i in sorted(os.listdir(target_dir)):
        if not i.endswith(target):
            continue
        tls_id = extractTLSID(i)
        if tls_id == match:
            return(target_dir + '/' + i)
    return ('no match found')

def extractTLSID(input_str):
    """ Convenience function to extract TLS ID. 

    :param input_str: String containing TLS ID.
    :type input_str: str    

    :return: Returns the extracted TLS ID.
    :rtype: str
    """
    tls_id = re.match("TLS_\d{4}_\d{8}_\d{2}", input_str)
    if tls_id:
        return(tls_id.group())

def extractHeight(input_str):
    height = re.search("\d\.\d{3}_\d\.\d{3}", input_str)
    if height:
        height = height.group()
        height = height.split('_')
        min_h = float(height[0])
        max_h = float(height[1])
        avg_height = int((min_h + max_h)/2)
        return str(avg_height)

def normalizeObservedData(line, id_target, x_target, y_target, r_target, dbh=True):
    """ This is a convenience function to normalize the names of inputs for the
        accuracy assessment. 

    :param line: a row from pandas dataframe.itterow()
    :type line: pandas series
    :param id_target: name of treeID column. Usually 'Tree_Id'
    :type id_target: str
    :param x_target: name of x column. Usually 'X_Location'
    :type x_target: str
    :param y_target: name of y column. Usually 'Y_Location'
    :type y_target: str
    :param r_target: name of radius or DBH column. Usually 'DBH_m'
    :type r_target: str    
    :param dbh: Divides size by 2 when set to True, dfaults to True.

    :return: Returns the normalized information from input.
    :rtype: str, float, float, float

    """

    treeID = line[id_target]
    x = line[x_target]
    y = line[y_target]
    if dbh == True:
        r = line[r_target]/2.
    else:
        r = line[r_target]
    
    return treeID, x, y, r

def euclidDist(x1, y1, x2, y2):
    """ Convenience function to get Euclidean distance between two 2D points.
    :param x1: x coordinate of first point.
    :type x1: float 
    :param y1: y coordinate of first point.
    :type y1: float 
    :param x2: x coordinate of second point.
    :type x2: float 
    :param y2: x coordinate of second point.
    :type y2: float 

    :return: Euclidean distance between two points.
    :rtype: float

    """
    c = math.sqrt(((x2-x1)**2) + ((y2-y1)**2))

    return c

def trainValTest(input_list, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """ Convenience function for train, validation, and test split. 
    
    :param input_list: List of data to split.
    :type input_list: list
    :param train_ratio: Proportion of training data, defaults to 0.7
    :type train_ratio: float
    :param val_ratio: Proportion of validation data, defaults to 0.2
    :type val_ratio: float
    :param test_ratio: Proportion of test data, defaults to 0.1
    :type test_ratio: float

    :return: Returns 3 lists: training data, validation data, and testing data
    :rtype: list, list, list
    """

    num_train = int(len(input_list)*train_ratio)
    num_not_train = len(input_list) - num_train
    num_val = int(len(input_list)*val_ratio)
    num_test = num_not_train - num_val

    train, not_train = model_selection.train_test_split(input_list, test_size=num_not_train, train_size=num_train)
    val, test = model_selection.train_test_split(not_train, test_size=num_test, train_size=num_val)

    return train, val, test

def arrangeData(input_dir, output_root, target='.shp', train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, overwrite=False):
    """ Convenience function to divide up all files in an input directory and
        save them into training, validation, or testing directories.


    :param input_dir: Filepath to directory with data to split.
    :type input_dir: str
    :param output_root: Filepath to directory where split data will be stored.
    :type output_root: str
    :param target: File ending to search for IDs with, defaults to '.shp'
    :type target: str
    :param train_ratio: Proportion of training data, defaults to 0.7
    :type train_ratio: float
    :param val_ratio: Proportion of validation data, defaults to 0.2
    :type val_ratio: float
    :param test_ratio: Proportion of test data, defaults to 0.1
    :type test_ratio: float
    :param overwrite: Flag to overwrite existing outputs, defaults to False.
    :type overwrite: bool

    :return: Returns 3 lists: training data, validation data, and testing data
    :rtype: list, list, list
    """

    plots = []
    train_dir = output_root + '/train'
    val_dir = output_root + '/val'
    test_dir = output_root + '/test'

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for i in sorted(os.listdir(input_dir)):
        if not i.endswith('.shp'):
            continue
        tls_id = extractTLSID(i)
        plots.append(tls_id)
    train, val, test = trainValTest(plots, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    for i in sorted(os.listdir(input_dir)):
        input_file = input_dir + '/' + i
        tls_id = extractTLSID(i)
        if tls_id in test:
            output_file = test_dir + '/' + i
        elif tls_id in val:
            output_file = val_dir + '/' + i
        else:
            output_file = train_dir + '/' + i
        if os.path.exists(output_file) and overwrite==False:
            continue
        shutil.copy(input_file, output_file)

def newID(plot, height, x):
    """ This is a convenience function to generate unique IDs. """

    new_id = str(plot) + '_' + str(height) + '_' + str(x)

    return new_id

def assignIDs(input_df, plot, height):
    """ This is a convenience function to give every detection in a dataframe a
        unique ID. """

    uids = []
    for index, row in input_df.iterrows():
        uid = newID(plot, height, index)
        uids.append(uid)
    output_df = input_df.assign(uid=uids)

    return output_df

def bulkAssignIDs(input_dir, target='.csv'):
    for i in sorted(os.listdir(input_dir)):
        if i.endswith(target):
            input_csv = input_dir + '/' + i
            input_df = pd.read_csv(input_csv)
            plot = extractTLSID(i)
            height = extractHeight(i)
            output_df = assignIDs(input_df, plot, height)
            output_df.to_csv(input_csv, index=False)


def parseID(uid):
    """ This is a convenience function to extract the relevant info from UIDs. """

    info = uid.split('_')
    if len(info) != 3:
        print('invalid ID')
        return(1)
    plot = info[0]
    height = info[1]
    tree_id = info[2]

    return plot, height, tree_id

def mergeCSVs(input_dir, output_csv):
    target_csvs = os.listdir(input_dir)
    df_list = []
    for i in target_csvs:
        input_csv = input_dir + '/' + i
        df_list.append(pd.read_csv(input_csv))
    merged_df = pd.concat(df_list)
    merged_df.to_csv(output_csv, index=False)