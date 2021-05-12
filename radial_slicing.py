from scipy_vision_tools import loadImage, circleMask, trim2DArray, edgeDetection, skeletonizeImage, connectedComponentsImage, morphologicalCloseImage
from gis_tools import getCoordinates
from math import ceil, floor
import matplotlib.pyplot as plt

import numpy as np

from skimage.transform import rotate, rescale
from skimage.color import rgb2gray
from skimage.morphology import skeletonize
from skimage.draw import circle_perimeter
from skimage import measure

def pizzaSlice(np_img, theta):
    nrows, ncols = np_img.shape
    out = np_img.copy()
    half = ceil(nrows/2)
    for i in range(0, half):
        out[:,i] = 0
    out = rotate(out, theta)
    for i in range(0, half):
        out[:,-1*(i+1)] = 0
    out = rotate(out, -1*theta)

    return out

def radialSlices(np_img, theta):
    num_slices = ceil(360/theta)
    slice_out = []
    for i in range(0, num_slices):
        img = rotate(np_img, i*theta)
        out = pizzaSlice(img, theta)
        slice_out.append(out)
    return slice_out

def realToPixel(lat, lon, pixel_res, real_x, real_y, real_rad):
    """This function transforms pixel coordinates (and radius) into realworld coordinates.

    :param lat: Latitude of upper left pixel
    :type lat: float
    :param lon: Longitude of upper left pixel
    :type lon: float
    :param pixel_res: Pixel resolution of real world coordinates
    :type pixel_res: float
    :param pix_x: x position of pixel coordinate
    :type pix_x: int
    :param pix_y: y position of puxel coordinate
    :type pix_y: int

    :return: Returns a 3 element list containing the latitude, longitude, and radius in real world coordinates.
    :rtype: list[lat, lon, pixel_res]
    """

    pix_y = (float(real_y) - lat)/pixel_res*-1
    pix_x = (float(real_x) - lon)/pixel_res
    pix_r = real_rad / float(pixel_res)
    new_line = [int(pix_x), int(pix_y), int(pix_r)]
    return new_line

x = -56267.569613144100003
y = 196175.289770580420736
r = 0.628401164250018/2



test_img = '/Users/theo/data/registered_rasters/1_37/TLS_0001_20170531_01_v003_clipped_40_classified_1.345_1.395.tif'

img = loadImage(test_img)

lon, lat, pixel_res = getCoordinates(test_img)

pix_info = realToPixel(float(lat), float(lon), float(pixel_res), x, y, r)
print(pix_info)

x = pix_info[0]
y = pix_info[1]
r = pix_info[2]



img = circleMask(img, x, y, r, 'exterior')
img = circleMask(img, x, y, r-5, 'interior')

img = trim2DArray(img)

img = np.pad(img, ceil(r/3))

cir_y, cir_x = img.shape

cir_y = int(cir_y/2)
cir_x = int(cir_x/2)

theta = 90

slices = radialSlices(img, theta)
counter = 0

for i in slices:
    counter += 1
    #print(counter)
    edge = edgeDetection(i, 'np', sigma=2)
    close = morphologicalCloseImage(edge, 'np', stel=3)
    obs = np.empty_like(close)
    obs[:,:] = close
    pred = np.zeros(close.shape)

    sub_y, sub_x = circle_perimeter(cir_y, cir_x, r)
    pred[sub_y, sub_x] = (255)


    out = pred*obs
    obs_count = np.count_nonzero(out)
    pred_slice = pizzaSlice(pred, theta)
    pred_slice = pred*pred_slice
    pred_count = np.count_nonzero(pred_slice)
    ratio = obs_count/pred_count
    plt.imshow(out)
    plt.show()
    print(ratio)