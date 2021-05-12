import os
import subprocess

input_dir = '/Users/theo/data/segmentation_test_las'
output_root = '/Users/theo/data/treels_pipeline_test'
r_path = '/Users/theo/Documents/GitHub/night_tool/treels_funcs.R'
target = '.las'
viz = 'FALSE'

for i in sorted(os.listdir(input_dir)):
    if not i.endswith(target):
        continue
    input_las = input_dir + '/' + i
    output_dir = output_root + '/' + i[:-4]
    xy_map = output_dir + '_xy.csv'
    seg_map = output_dir + '_seg.csv'
    treeLS_call = ['Rscript', r_path, input_las, xy_map, seg_map, viz]
    subprocess.run(treeLS_call)