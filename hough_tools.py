import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os
import shutil
import subprocess

from multiprocessing import Pool, Process, Queue, Manager
from PIL import Image
from scipy import ndimage
from skimage import feature, filters, measure, morphology, data, color
from skimage.transform import hough_circle, hough_circle_peaks

from scipy_vision_tools import loadImage, circleMask, calculatePixelMetricsMP
from gis_tools import detectionsOnDir
from utility_tools import matchTLS

# I got these from Rodney and so far they have worked well.
PARAMS = [[7,10,18],[11,15,18],[16,20,12],[21,25,12],[26,30,8],[31,35,8],[36,40,8],[41,45,6],[46,50,6],[51,55,4],[56,60,4],[61,65,4],[66,70,4],[71,75,4],[76,80,4],[81,85,4],[86,90,4],
[91,95,4],[96,100,4],[101,250,1]]

# high level
def houghStep1(input_root, temp_root, pixel_dir, original_img_dir, num_workers, target='.tif', save_intermediate=False, overwrite=False):
    """ Parallel Hough detections to pixel coordinates. 
    
    :param input_root: Filepath to the directory containing clustered subimages.
    :type input_root: str
    :param temp_root: Filepath to the directory where intermediate outputs will be created.
    :type temp_root: str
    :param pixel_dir: The filepath to the directory where pixel detection outputs will be stored.
    :type pixel_dir: str
    :param original_img_dir: Filepath with the original images used to create subimages.
    :type original_img_dir: str
    :param num_workers: Number of workers to assign.
    :type num_workers: int
    :param target: The file ending for valid inputs, defaults to '.tif'
    :type target: str
    :param save_intermediate: Flag to save intermediate outputs to disk, defaults to False.
    :type save_intermediate: bool
    :param overwrite: Flag to overwrite existing outputs, defaults to False.
    :type overwrite: bool     

    """

    houghAllDirsMP(input_root, temp_root, pixel_dir, original_img_dir, num_workers, target, save_intermediate, overwrite)

def houghStep2(original_img_dir, pixel_dir, output_dir, num_workers, target='.tif'):
    """Parallel pixel metrics. 

    :param original_img_dir: Filepath with the original images used to create subimages.
    :type original_img_dir: str
    :param pixel_dir: The filepath to the directory where pixel detection outputs will be stored.
    :type pixel_dir: str
    :param num_workers: Number of workers to assign.
    :type num_workers: int
    :param target: The file ending for valid inputs, defaults to '.tif'
    :type target: str

    """

    for i in sorted(os.listdir(pixel_dir)):
        input_df = pd.read_csv(pixel_dir + '/' + i)
        input_img = matchTLS(i, original_img_dir, target)
        if input_img == 'no match found':
            continue
        output_df = calculatePixelMetricsMP(input_img, input_df, num_workers)
        output_csv = output_dir + '/' + i[:-17] + '_augmented_pixel_coords.csv'
        print(output_csv)
        output_df.to_csv(output_csv, index=False)

def houghStep3(input_dir, output_dir):
    """ Apply CRS and transform to buffered .shp. 

    :param input_dir: Filepath to an input directory with .csv's created by Hough tools
    :type input_dir: str
    :param ouput_dir: Filepath to directory where output .shp's will be saved.
    :type output_dir: str

    """

    detectionsOnDir(input_dir, output_dir)

# midlevel
def houghOnDir(input_dir, temp_dir, output_csv, base_image, target='.tif', cleanup=True):
    """ Applies Hough transforms to all images in a directory and merges the results.

    :param input_dir: Filepath to the directory with subdivided images.
    :type input_dir: str
    :param temp_dir: The filepath to the directory where intermediate outputs will be stored.
    :type temp_dir: str
    :param output_csv: Filepath to the desired output.
    :type output_csv: str
    :param base_image: Filepath to the original image.
    :type base_img: str
    :param target: The file ending for valid inputs, defaults to '.tif'
    :type target: str
    :param cleanup: Controls whether or not intermediate files are removed upon completion, defaults to True.
    :type cleanup: bool

    """

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    chunkedRastersHough(input_dir, temp_dir, PARAMS)
    info = base_image.split('/')
    info = info[-1]
    info = info.split('_')
    plot = info[1]
    img = loadImage(base_image)
    y,x = img.shape
    print(base_image, output_csv)
    mergeChunkedDetections(plot, temp_dir, x, y, output_csv)

def chunkedRastersHough(input_dir, output_dir, params, target='.tif'):
    """ Applies Hough transforms to each image in a directory.

    :param input_dir: Filepath to the directory with subdivided images.
    :type input_dir: str
    :param output_dir: Filepath to the directory where detections will be saved.
    :type output_dir: str
    :param params: See params section for more informaiton.
    :type params: list[[int, int, int]]
    :param target: The file ending for valid inputs, defaults to '.tif'
    :type target: str

    """

    for i in sorted(os.listdir(input_dir)):
        if not i.endswith(target):
            continue
        input_raster = input_dir + '/' + i
        print(input_raster)
        output_csv = output_dir + '/' + i[:-4] + '.csv'
        loopApplyHough(input_raster, output_csv, params)

def mergeChunkedDetections(plot, input_dir, x_len, y_len, output_csv, target='.csv'):
    """ This is being rewritten for the new clusters. 
    
    :param plot: plot ID for image, used to label detections.
    :type plot: int
    :param input_dir: Filepath to the directory with detections (temp dir from chunkedRastersHough)
    :type input_dir: str  
    :param x_len: Width of the input image
    :type x_len: int
    :param y_len: Length of the input image
    :type y_len: int
    :param output_csv: Filepath to save the merged output .csv
    :type output_csv: str
    :param target: The file ending for valid inputs, defaults to '.csv'
    :type target: str

    """
    df = pd.DataFrame(columns = ["plot", "x", "y", "r", "weight"])
    for i in sorted(os.listdir(input_dir)):
        if not i.endswith(target):
            continue
        input_csv = input_dir + '/' + i
        info = i.split('_')
        ul_x = int(info[0])
        ul_y = int(info[1])
        df2 = pd.read_csv(input_csv)
        for index, row in df2.iterrows():
            x = ul_x + row["x"]
            y = ul_y + row["y"]
            if x > x_len or y > y_len:
                print('x:', ul_x, '+', row["x"], '=', x)
                print('y:', ul_y, '+', row["y"], '=', y)
            df = df.append({"plot" : plot, "x" : x, "y" : y, "r": row["r"], "weight" : row["weight"]}, ignore_index=True)
    df.to_csv(output_csv, index=False)

# lowlevel
def applyHough(input_raster, radii, num_peaks):
    """ Convenience function for loopApplyHough. Needs string of filepaths to 
        input raster and a list of radii made with np.arange().
        Returns the weight, pixel location, and radius of each detected circle.

    :param input_raster: Filepath to the input raster
    :type input_raster: str        
    :param radii: a list of radii made with np.arange()
    :type radii: list(int)
    :param num_peaks: maximum number of detections per search.
    :type num_peaks: int

    :return: Coordinates of the circle detections and their weights
    :rtype: int, int, int, float.

    """
    np_img = loadImage(input_raster)
    hough_res = hough_circle(np_img, radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, radii,
                                               min_xdistance = radii[0],
                                               min_ydistance = radii[0],
                                               normalize=True,
                                               total_num_peaks=num_peaks)
    return (cx, cy, radii, accums)

def loopApplyHough(input_raster, output_csv, params):
    """ Needs strings of filepaths to input raster and output csv, and a list
        of parameters in the format [[min_rad, max_rad, num_peaks], ...].
        Searches an image for all sets of those parameters and saves them in
        the output .csv.

    :param input_raster: Filepath to the input raster
    :type input_raster: str
    :param output_csv: Filepath to save the merged output .csv
    :type output_csv: str
    :param params: See params section for more informaiton.
    :type params: list[[int, int, int]]

    """    
    df = pd.DataFrame(columns = ["x", "y", "r", "weight"])
    img = loadImage(input_raster)
    b,a = img.shape
    for i in params:
        if a < b:
            c = a/2
        else:
            c = b/2
        if i[0] > c:
            continue
        radii = np.arange(i[0], i[1], 1)
        peaks_5_by_5 = i[2]
        num_peaks = math.ceil((peaks_5_by_5/25.)*a*b)
        x, y, r, weight = applyHough(input_raster, radii, num_peaks)
        for j in range(0, len(x)):
            df = df.append({"x" : x[j], "y" : y[j], "r": r[j],
                            "weight" : weight[j]}, ignore_index=True)
    df.to_csv(output_csv, index=False)

# multiprocessing

def houghCircleQueue(q, input_root, temp_root, pixel_dir, original_img_dir, target='.tif', save_intermediate=False, overwrite=False):
    """ This is a queue for multiprocessing.

    :param q: the queue generated by houghAllDrisMP
    :type q: Multiprocessing.Queue object
    :param input_root: Filepath to the directory containing clustered subimages.
    :type input_root: str
    :param temp_root: Filepath to the directory where intermediate outputs will be created.
    :type temp_root: str
    :param pixel_dir: The filepath to the directory where pixel detection outputs will be stored.
    :type pixel_dir: str
    :param original_img_dir: Filepath with the original images used to create subimages.
    :type original_img_dir: str
    :param num_workers: Number of workers to assign.
    :type num_workers: int
    :param target: The file ending for valid inputs, defaults to '.tif'
    :type target: str
    :param save_intermediate: Flag to save intermediate outputs to disk, defaults to False.
    :type save_intermediate: bool
    :param overwrite: Flag to overwrite existing outputs, defaults to False.
    :type overwrite: bool     
    
    """
    while not q.empty():
        try:
            current_dir = q.get()
            input_dir = input_root + '/' + current_dir + '/base'
            print("input dir: ", input_dir)
            base_img = original_img_dir + '/' + current_dir + target
            temp_dir = temp_root + '/' + current_dir
            pixel_csv = pixel_dir + '/' + current_dir + '_pixel_coords.csv'
            houghOnDir(input_dir, temp_dir, pixel_csv, base_img, target='.tif', cleanup=False)
        except ValueError as val_error:
            print(val_error)
        except Exception as error:
            print(error)

def houghAllDirsMP(input_root, temp_root, output_root, original_img_dir, num_workers, target='.tif', save_intermediate=False, overwrite=False):
    """ Multiprocessing implementation of houghOnDir.

    :param input_root: Filepath to the directory containing clustered subimages.
    :type input_root: str
    :param temp_root: Filepath to the directory where intermediate outputs will be created.
    :type temp_root: str
    :param pixel_dir: The filepath to the directory where pixel detection outputs will be stored.
    :type pixel_dir: str
    :param original_img_dir: Filepath with the original images used to create subimages.
    :type original_img_dir: str
    :param num_workers: Number of workers to assign.
    :type num_workers: int
    :param target: The file ending for valid inputs, defaults to '.tif'
    :type target: str
    :param save_intermediate: Flag to save intermediate outputs to disk, defaults to False.
    :type save_intermediate: bool
    :param overwrite: Flag to overwrite existing outputs, defaults to False.
    :type overwrite: bool     

    """

    q = Queue()        
    for i in sorted(os.listdir(input_root)):
        if os.path.isdir(input_root + '/' + i + '/base'):
            q.put(i)
        else:
            continue
    workers = Pool(num_workers, houghCircleQueue,(q, input_root, temp_root, output_root, original_img_dir))
    workers.close()
    workers.join()
