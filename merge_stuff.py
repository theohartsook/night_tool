import os
import pandas as pd

from accuracy_tools import accuracyOnDirMP, bestOnDirMP, findGoodEnoughPredictions, goodEnoughOnDirMP
from utility_tools import mergeCSVs, bulkAssignIDs

pred_dir = '/Users/theo/data/hough_dataset/oof_5'
obs_dir = '/Users/theo/data/hough_dataset/observed'
acc_dir = '/Users/theo/data/hough_dataset/oof_acc'
best_dir = '/Users/theo/data/hough_dataset/oof_best'
good_dir = '/Users/theo/data/hough_dataset/oof_good'

base_dir = '/Users/theo/data/hough_dataset/oof_good'

test_dir = '/Users/theo/data/hough_dataset/test'
val_dir = '/Users/theo/data/hough_dataset/val'
train_dir = '/Users/theo/data/hough_dataset/train'
output_root = '/Users/theo/data/hough_dataset/good_enough_'
target = 'good.csv'

#if __name__ ==  '__main__':
#    goodEnoughOnDirMP(acc_dir, good_dir)

df_list = []
for i in sorted(os.listdir(test_dir)):
    if not i.endswith('.shp'):
        continue
    for j in sorted(os.listdir(base_dir)):
        if not j.endswith(target):
            continue
        if j[0:15] != i[0:15]:
            continue
        else:
            df_list.append(pd.read_csv(base_dir + '/' + j))

merged_df = pd.concat(df_list)
merged_df.to_csv(output_root + 'test.csv', index=False)

df_list = []
for i in sorted(os.listdir(val_dir)):
    if not i.endswith('.shp'):
        continue
    for j in sorted(os.listdir(base_dir)):
        if not j.endswith(target):
            continue
        if j[0:15] != i[0:15]:
            continue
        else:
            df_list.append(pd.read_csv(base_dir + '/' + j))
merged_df_2 = pd.concat(df_list)
merged_df_2.to_csv(output_root + 'val.csv', index=False)

df_list = []
for i in sorted(os.listdir(train_dir)):
    if not i.endswith('.shp'):
        continue
    for j in sorted(os.listdir(base_dir)):
        if not j.endswith(target):
            continue
        if j[0:15] != i[0:15]:
            continue
        else:
            df_list.append(pd.read_csv(base_dir + '/' + j))
merged_df_3 = pd.concat(df_list)
merged_df_3.to_csv(output_root + 'train.csv', index=False)