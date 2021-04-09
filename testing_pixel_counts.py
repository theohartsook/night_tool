import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shutil
import subprocess

from multiprocessing import Pool, Process, Queue, Manager
from PIL import Image
from scipy import ndimage
from skimage import feature, filters, measure, morphology, data, color
from skimage.transform import hough_circle, hough_circle_peaks

from scipy_vision_tools import loadImage, circleMask, saveImage, trim2DArray

def countPixels(input_img, input_mode, x, y, r):
    """ Counts pixels within a radius around x, y.

    :param input_img: Either the filepath to the input img (input_mode='fp') or a numpy array (input_mode='np').
    :type input_img: str or numpy array
    :param input_mode: 'fp' if input_img is a filepath or 'np' if input_img is a numpy array.
    :type input_mode: str
    :param x: The x coordinate of the center of the circle.
    :type x: int
    :param y: The y coordinate of the center of the circle.
    :type y: int 
    :param r: The radius around the center of the circle
    :type r: int

    :return: Returns the number of pixels.
    :rtype: int
    """

    if input_mode == 'fp':
        np_img = loadImage(input_img)
        np_img = color.rgb2gray(np_img)
    elif input_mode == 'np':
        np_img = input_img
    else:
        return (input_mode, " is not a supported mode. Supported modes are 'np' or 'fp'.")

    base_img = circleMask(np_img, x, y, r, 'exterior')
    base_count = np.count_nonzero(np_img*base_img)

    if r < 1:
        b = 1
    else:
        b = r

    core_img = circleMask(np_img, x, y, b*0.2, 'exterior')
    core_count = np.count_nonzero(base_img*core_img)

    inner_img = circleMask(np_img, x, y, b*0.8, 'exterior')
    inner_ring = base_img - inner_img
    inner_count = np.count_nonzero(inner_ring)

    outer_img = circleMask(np_img, x, y, b*1.2, 'exterior')
    outer_ring = outer_img - base_img
    outer_count = np.count_nonzero(outer_ring)

    name = '/Users/theo/data/pixel_metrics_testing/img/' + str(x) + '_' + str(y) + '_' + str(r) + '_'



    if base_count > 0:
        #trim = trim2DArray(base_img)
        #saveImage(trim, name + 'base.tif')
        saveImage(base_img, name + 'base.tif')

    if core_count > 0:
        #trim = trim2DArray(core_img)
        #saveImage(trim, name + 'core.tif')
        saveImage(core_img, name + 'core.tif')

    if inner_count > 0:
        #trim = trim2DArray(inner_img)
        #saveImage(trim, name + 'inner.tif')
        saveImage(inner_ring, name + 'inner.tif')

    if outer_count > 0:
        #trim = trim2DArray(outer_img)
        #saveImage(trim, name + 'outer.tif')
        saveImage(outer_ring, name + 'outer.tif')


    return (core_count, inner_count, outer_count)

#input_img = '/Users/theo/data/registered_rasters/1_37/TLS_0001_20170531_01_v003_clipped_40_classified_1.345_1.395.tif'
#input_csv = '/Users/theo/data/pixel_metrics_testing/short_pixel_coords.csv'
input_img = '/Users/theo/data/pixel_metrics_testing/fake_raster.tif'
input_csv = '/Users/theo/data/pixel_metrics_testing/fake_pixel_coords.csv'

df = pd.read_csv(input_csv)

for index, row in df.iterrows():
    x = row[1]
    y = row[2]
    r = row[3]
    core, inner, outer = countPixels(input_img, 'fp', x, y, r)
    print('core:', core)
    print('inner:', inner)
    print('outer:', outer)