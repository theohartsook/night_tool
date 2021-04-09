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

# high level
def applySegmentationSteps(input_img, input_mode, output_root, save_intermediate=False, overwrite=False):
    """ This is a convenience function to apply all my vision steps in one go.

    :param input_img: Either the filepath to the input img (input_mode='fp') or a numpy array (input_mode='np').
    :type input_img: str or numpy array
    :param input_mode: 'fp' if input_img is a filepath or 'np' if input_img is a numpy array.
    :type input_mode: str
    :param output_root: The filepath to the directory where outputs will be created.
    :type output_root: str
    :param save_intermediate: Flag to save intermediate outputs to disk, defaults to False.
    :type save_intermediate: bool
    :param overwrite: Flag to overwrite existing outputs, defaults to False.
    :type overwrite: bool

    :return: Returns 0 if output successfully created, otherwise returns 1.
    :rtype: int
    """

    if save_intermediate == False:
        np_dist = getDistanceRaster(input_img, input_mode=input_mode)
        np_ridge = ridgeDetection(np_dist, 'np', method='meijering', black_ridges=False)
        np_blobs = connectedComponentsImage(np_ridge, 'np', output_path=output_root + '_blobs.tif')
        exportBlobs(input_img, np_blobs, 'np', output_root)
        plt.imsave(output_root + 'blobs_cmap.png', np_blobs, cmap='nipy_spectral')
    else:
        np_dist = getDistanceRaster(input_img, input_mode=input_mode, output_path=output_root + '_distance.tif')
        np_ridge = ridgeDetection(np_dist, 'np', method='meijering', black_ridges=False, output_path=output_root + '_ridge.tif')
        np_blobs = connectedComponentsImage(np_ridge, 'np', output_path=output_root + '_blobs.tif')
        exportBlobs(input_img, np_blobs, 'np', output_root)
        plt.imsave(output_root + 'blobs_cmap.png', np_blobs, cmap='nipy_spectral')

    if os.path.exists(output_root + 'blobs_tif'):
        return 0
    else:
        return 1

def calculatePixelMetrics(input_img, input_df):
    """ Calculates pixel metrics for every detection in a dataframe. 

    :param input_img: Filepath to the input img
    :type input_img: str
    :param input_df: Dataframe with detections
    :type input_df: pandas dataframe

    :return: Returns the dataframe augmented with pixel metrics.
    :rtype: pandas dataframe
    """
    
    new_cir = []
    for index, row in input_df.iterrows():
        plot = row['plot']
        x = row['x']
        y = row['y']
        r = row['r']
        weight = row['weight']
        core, inner, outer = countPixels(input_img, 'np', x, y, r)
        print(core, inner, outer)
        circle = ([plot, x, y, r, weight, core, inner, outer])
        new_cir.append(circle)
    print('circles added:', len(new_cir))
    header = ['plot', 'x', 'y', 'r', 'weight', 'outer', 'inner', 'core']
    output_df = pd.DataFrame(new_cir, columns=header)
    return output_df

# mid level

def getDistanceRaster(input_img, input_mode, output_path=None):
    """ Takes a string containing the file path to the input image and an
    optional string with the desired output filepath. Returns a np array 
    with the distance transform applied. If output path is applied, the
    distance raster will be saved. 
    Note: the logical not is necessary for the scipy function.
    https://stackoverflow.com/questions/44770396/how-does-the-scipy-distance-transform-edt-function-work

    :param input_img: Either the filepath to the input img (input_mode='fp') or a numpy array (input_mode='np').
    :type input_img: str or numpy array
    :param input_mode: 'fp' if input_img is a filepath or 'np' if input_img is a numpy array.
    :type input_mode: str
    :param output_path: The filepath of the image to be saved.
    :type output_path: str, optional

    :return: Returns numpy array of transformed image
    :rtype: numpy array
    """

    if input_mode == 'fp':
        np_img = loadImage(input_img)
    elif input_mode == 'np':
        np_img = input_img
    else:
        return (input_mode, " is not a supported mode. Supported modes are 'np' or 'fp'.")
    np_dist = ndimage.distance_transform_edt(np.logical_not(np_img))
    if output_path is not None:
        saveImage(np_dist, output_path, mode='dist_norm')
    return(np_dist)

def getImageThreshold(input_img, input_mode, threshold, output_path=None):
    """ Takes either a string to an image or a np array as input and a float
        for thresholding. Supported input modes are: "np" (numpy array) or "fp"
        (filepath). Returns a thresholded image. If output path is applied, the
        thresholded raster will be saved. 

    :param input_img: Either the filepath to the input img (input_mode='fp') or a numpy array (input_mode='np').
    :type input_img: str or numpy array
    :param input_mode: 'fp' if input_img is a filepath or 'np' if input_img is a numpy array.
    :type input_mode: str
    :param threshold: The comparison value for thresholding.
    :type threshold: int or float
    :param output_path: The filepath of the image to be saved.
    :type output_path: str, optional

    :return: Returns numpy array of transformed image
    :rtype: numpy array
    """

    if input_mode == 'fp':
        np_img = loadImage(input_img)
    elif input_mode == 'np':
        np_img = input_img
    else:
        return (input_mode, " is not a supported mode. Supported modes are 'np' or 'fp'.")
    np_thresh = np.zeros_like(np_img)
    dimensions = np_img.shape
    for i in range(0, dimensions[0]):
        for j in range(0, dimensions[1]):
            pixel = np_img[i][j]
            if pixel < threshold:
                np_thresh[i][j] = 1
            else:
                np_thresh[i][j] = 0
    if output_path is not None:
        saveImage(np_thresh, output_path)
    return(np_thresh)

def edgeDetection(input_img, input_mode, sigma=1, output_path=None):
    """ Takes either a string to an image or a np array as input and an integer
        for sigma. Default settings are for thresholded images.Supported input
        modes are: "np" (numpy array) or "fp" (filepath). Returns an image of 
        edge detections. If output path is applied, the edge raster will be saved.

    :param input_img: Either the filepath to the input img (input_mode='fp') or a numpy array (input_mode='np').
    :type input_img: str or numpy array
    :param input_mode: 'fp' if input_img is a filepath or 'np' if input_img is a numpy array.
    :type input_mode: str
    :param sigma: Controls the scale of edge detection, defaults to 1.
    :type sigma: int
    :param output_path: The filepath of the image to be saved.
    :type output_path: str, optional

    :return: Returns numpy array of transformed image
    :rtype: numpy array
    """

    if input_mode == 'fp':
        np_img = loadImage(input_img)
    elif input_mode == 'np':
        np_img = input_img
    else:
        return (input_mode, " is not a supported mode. Supported modes are 'np' or 'fp'.")
    edges = feature.canny(np_img, sigma=sigma)
    if output_path is not None:
        saveImage(edges, output_path)
    return(edges)

def morphologicalCloseImage(input_img, input_mode, stel=50, output_path=None):
    """ Takes either a string to an image or a np array as input and an integer
        for structuring element. Applies a circular structuring element. Supported input
        modes are: "np" (numpy array) or "fp" (filepath). Returns an image with
        a closing appplied. If output path is applied, the closed raster will be saved.

    :param input_img: Either the filepath to the input img (input_mode='fp') or a numpy array (input_mode='np').
    :type input_img: str or numpy array
    :param input_mode: 'fp' if input_img is a filepath or 'np' if input_img is a numpy array.
    :type input_mode: str
    :param stel: Controls the size of structuring element, defaults to 50.
    :type stel: int
    :param output_path: The filepath of the image to be saved.
    :type output_path: str, optional

    :return: Returns numpy array of transformed image
    :rtype: numpy array
    """

    if input_mode == 'fp':
        np_img = loadImage(input_img)
    elif input_mode == 'np':
        np_img = input_img
    else:
        return (input_mode, " is not a supported mode. Supported modes are 'np' or 'fp'.")
    closed_img = morphology.binary_closing(np_img, morphology.disk(stel))
    if output_path is not None:
        saveImage(closed_img, output_path)
    return(closed_img)

def skeletonizeImage(input_img, input_mode, output_path=None):
    """ Takes either a string to an image or a np array as input. Supported
        input modes are: "np" (numpy array) or "fp" (filepath). Returns a
        skeletonized image. If output path is applied, the closed raster will
        be saved.

    :param input_img: Either the filepath to the input img (input_mode='fp') or a numpy array (input_mode='np').
    :type input_img: str or numpy array
    :param input_mode: 'fp' if input_img is a filepath or 'np' if input_img is a numpy array.
    :type input_mode: str
    :param output_path: The filepath of the image to be saved.
    :type output_path: str, optional

    :return: Returns numpy array of transformed image
    :rtype: numpy array
    """

    if input_mode == 'fp':
        np_img = loadImage(input_img)
    elif input_mode == 'np':
        np_img = input_img
    else:
        return (input_mode, " is not a supported mode. Supported modes are 'np' or 'fp'.")
    skeleton = morphology.skeletonize(np_img)
    if output_path is not None:
        saveImage(skeleton, output_path)
    return(skeleton)

def connectedComponentsImage(input_img, input_mode, connectivity=1, output_path=None):
    """ Applies connected components labelling to an input image.

    :param input_img: Either the filepath to the input img (input_mode='fp') or a numpy array (input_mode='np').
    :type input_img: str or numpy array
    :param input_mode: 'fp' if input_img is a filepath or 'np' if input_img is a numpy array.
    :type input_mode: str
    :param connectivity: Controls the distance to connect components, defaults to 1.
    :type connectivity: int
    :param output_path: The filepath of the image to be saved.
    :type output_path: str, optional

    :return: Returns numpy array with the connected components.
    :rtype: numpy array
    """

    if input_mode == 'fp':
        np_img = loadImage(input_img)
    elif input_mode == 'np':
        np_img = input_img
    else:
        return (input_mode, " is not a supported mode. Supported modes are 'np' or 'fp'.")
    dimensions = np_img.shape
    blobs = np_img.copy()
    for i in range(0, dimensions[0]):
        for j in range(0, dimensions[1]):
            pixel = np_img[i][j]
            if pixel < 0.5:
                blobs[i][j] = 1
            else:
                blobs[i][j] = 0
    blobs_labels = measure.label(blobs, connectivity=connectivity)
    if output_path is not None:
        saveImage(blobs_labels, output_path, mode='blob_labels')
    return(blobs_labels)

def exportBlobs(input_img, input_blobs, input_mode, output_root):
    """ Input_img and input_blobs will be opened with identical modes. Output is a
    directory containing base and blobs subdirectories. Base has the mask for each blob
    applied to the image.

    :param input_img: Either the filepath to the input img (input_mode='fp') or a numpy array (input_mode='np').
    :type input_img: str or numpy array
    :param input_img: Either the filepath to the input blobs (input_mode='fp') or a numpy array (input_mode='np').
    :type input_img: str or numpy array
    :param input_mode: 'fp' if input_img is a filepath or 'np' if input_img is a numpy array.
    :type input_mode: str
    :param output_root: The filepath to the directory where clusters will be saved.
    :type output_root: str

    :return: Returns 0 upon completion.
    :rtype: numpy array
    """

    if input_mode == 'fp':
        np_img = loadImage(input_img)
        np_blobs = loadImage(input_blobs)
    elif input_mode == 'np':
        np_img = input_img
        np_blobs = input_blobs
    else:
        return (input_mode, " is not a supported mode. Supported modes are 'np' or 'fp'.")
    clusters = np.unique(np_blobs)

    output_base = output_root + '/base'
    output_blobs = output_root + '/blobs'

    if not os.path.exists(output_base):
        os.makedirs(output_base)
    if not os.path.exists(output_blobs):
        os.makedirs(output_blobs)

    group_counter = 0
    print(len(clusters))
    for i in sorted(clusters):
        print("i", i)
        print("group", group_counter)
        single_blob = np.where(np_blobs == i, True, False)
        base_out = single_blob*np_img
        blob_out = single_blob*np_blobs
        trimmed_blob_out = trim2DArray(blob_out)
        ul_x, ul_y = calculateTrimOffsetForward(blob_out)
        lr_x, lr_y = calculateTrimOffsetBackward(blob_out)
        trimmed_base_out = base_out[ul_y:lr_y,ul_x:lr_x]
        print('base', trimmed_base_out.shape)
        print('blob', trimmed_blob_out.shape)
        if testVerySmallClusters(trimmed_blob_out, 'np') or np.count_nonzero(base_out) < 5: # 5 is a magic constant but there's some logic here. how many points are needed to identify a circle? at least 3 needed in theory, more needed depending on the scanner
            print('skipping cluster')
            continue
        sub_base = output_base + '/' + str(ul_x) + '_' + str(ul_y) + '_' +str(group_counter) + '.tif'
        sub_blob = output_blobs + '/' + str(ul_x) + '_' + str(ul_y) + '_' + str(group_counter) + '.tif'
        saveImage(trimmed_base_out, sub_base)
        saveImage(trimmed_blob_out, sub_blob, mode='blob_labels')
        group_counter += 1
    return 0

def ridgeDetection(input_img, input_mode, method='meijering', black_ridges=False, output_path=None):
    """ Applies ridge detection to an input image.

    :param input_img: Either the filepath to the input img (input_mode='fp') or a numpy array (input_mode='np').
    :type input_img: str or numpy array
    :param input_mode: 'fp' if input_img is a filepath or 'np' if input_img is a numpy array.
    :type input_mode: str
    :param method: Ridge detection algorithm, defaults to meijering.
    :type method: str
    :param black_ridges: Controls whether black or white pixels are the ridges, defaults to False (white pixels = ridge).
    :type black_ridges: bool
    :param output_path: The filepath of the image to be saved.
    :type output_path: str, optional

    :return: Returns numpy array with the connected components.
    :rtype: numpy array
    """

    if input_mode == 'fp':
        np_img = loadImage(input_img)
    elif input_mode == 'np':
        np_img = input_img
    else:
        return (input_mode, " is not a supported mode. Supported modes are 'np' or 'fp'.")

    if method == 'meijering':
        np_ridges = filters.meijering(np_img, black_ridges=black_ridges)

    if output_path is not None:
        saveImage(np_ridges, output_path)
    return(np_ridges)    

# low level

def loadImage(img_path):
    """ Returns the image from img path as a np array using PIL. 

    :param img_path: Filepath to the input img
    :type img_path: str 

    :return: Returns the image loaded as a numpy array.
    :rtype: numpy array
    """

    img = Image.open(img_path)
    np_img = np.array(img)
    return (np_img)


def saveImage(np_img, output_path, mode=None, scale=255.0):
    """ Takes a np array as input and saves it to output path using PIL.
    Current supported modes are:
    dist_norm - distance normalization I use for distance transforms
    blob_labels - 255 scaling I use for connected components

    :param np_img: The numpy array to be saved
    :type np_img: numpy array
    :param output_path: The filepath of the image to be saved.
    :type output_path: str
    :param mode: There are two convenience modes: 'dist_norm' (distance transform normalization) and 'blob_labels' (blob labelling), defaults to None.
    :type mode: str
    :param scale: Value used for scaling in distance normalization, defaults to 255.0
    :type scale: int or float


    :return: Returns the image loaded as a numpy array.
    :rtype: numpy array
    """

    if mode is not None:
        if mode == 'dist_norm':
            np_img *= scale/np_img.max()
            np_out = Image.fromarray((np_img).astype('uint8'))
        elif mode == 'blob_labels':
            np_out = Image.fromarray((np_img*255).astype(np.uint8))
        else:
            return (mode, " is not a supported mode.")
    else:
        np_out = Image.fromarray(np_img)
    np_out.save(output_path)

    if os.path.exists(output_path):
        return 0
    else:
        return 1

def trim2DArray(input_arr, threshold=0):
    """ Convenience function to trim a 2D array. Removes any rows/columns
        with values equal to or less than the threshold. Source: 
        https://stackoverflow.com/questions/11188364/remove-zero-lines-2-d-numpy-array?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

    :param input_arr: A 2D numpy array to be trimmed.
    :type input_arr: numpy array
    :param threshold: Any exterior row or column with a total less than or equal to threshold will be removed, defaults to 0.
    :type threshold: int or float

    :return: Returns a subset of the input array.
    :rtype: numpy array
    """    

    print(input_arr.shape)
    ul_x, ul_y = calculateTrimOffsetForward(input_arr, threshold)
    lr_x, lr_y = calculateTrimOffsetBackward(input_arr, threshold)

    output_arr = input_arr[ul_y:lr_y, ul_x:lr_x]

    print(output_arr.shape)

    return (output_arr)

def calculateTrimOffsetForward(input_arr, threshold=0):
    """ This function is used to calculate the offset of the upper left corner
        after trimming. The threshold should be the same as whatever is used in
        trim2DArray. 
        
    :param input_arr: A 2D numpy array to be trimmed.
    :type input_arr: numpy array
    :param threshold: Any exterior row or column with a total less than or equal to threshold will be removed, defaults to 0.
    :type threshold: int or float

    :return: Returns the x and y of the upper left corner.
    :rtype: int, int
    """    

    ul_x, ul_y = 0, 0

    row_sum = np.sum(input_arr, axis=1)
    col_sum = np.sum(input_arr, axis=0)

    for i in range(0, len(col_sum)):
        if col_sum[i] > threshold:
            ul_x = i
            break
    for j in range(0, len(row_sum)):     
        if row_sum[j] > threshold:
            ul_y = j
            break
    return (ul_x, ul_y)

def calculateTrimOffsetBackward(input_arr, threshold=0):
    """ This function is used to calculate the offset of the lower right corner
        after trimming. The threshold should be the same as whatever is used in
        trim2DArray. 
        Returns the x and y of the lower right corner.

   :param input_arr: A 2D numpy array to be trimmed.
    :type input_arr: numpy array
    :param threshold: Any exterior row or column with a total less than or equal to threshold will be removed, defaults to 0.
    :type threshold: int or float

    :return: Returns the x and y of the lower right corner.
    :rtype: int, int
    """   

    y,x = input_arr.shape
    lr_x, lr_y = 0, 0

    row_sum = np.sum(np.flip(input_arr), axis=1)
    col_sum = np.sum(np.flip(input_arr), axis=0)

    for i in range(0, len(col_sum)):
        if col_sum[i] > threshold:
            lr_x = i
            break
    for j in range(0, len(row_sum)):     
        if row_sum[j] > threshold:
            lr_y = j
            break
    return (x-lr_x, y-lr_y)

def testVerySmallClusters(input_img, input_mode, min_rad=7.5):
    """ This functions takes a cluster as input and tests if its biggest
        circle would be too small. 
        Returns True if it's smaller and False if it's bigger.
        
    :param input_img: Either the filepath to the input img (input_mode='fp') or a numpy array (input_mode='np').
    :type input_img: str or numpy array
    :param input_mode: 'fp' if input_img is a filepath or 'np' if input_img is a numpy array.
    :type input_mode: str
    :param min_rad: the minimum radius for a valid cluster, defaults to 7.5
    :type min_rad: int or float


    :return: Returns True if the image is too small to be a valid cluster, otherwise False.
    :rtype: bool
    """

    if input_mode == 'fp':
        np_img = loadImage(input_img)
    elif input_mode == 'np':
        np_img = input_img
    else:
        return (input_mode, " is not a supported mode. Supported modes are 'np' or 'fp'.")
    dim = np_img.shape
    if dim[0] <= dim[1]:
        length = dim[0]
    else:
        length = dim[1]
    if length/2. < min_rad:
        return True
    else:
        return False

def circleMask(img, cir_x, cir_y, r, mode, filter=0):
    """ Takes an img (loaded as a np array), the x and y coordinates to center
    the mask, the radius of the mask, and the mode. Mode must be interior
    or exterior. The default filter value is 0. 
    https://towardsdatascience.com/the-little-known-ogrid-function-in-numpy-19ead3bdae40

    :param input_img: An image loaded as a numpy array.
    :type input_img: numpy array
    :param cir_x: The x coordinate of the center of the circle.
    :type cir_x: int
    :param cir_y: The y coordinate of the center of the circle.
    :type cir_y: int
    :param r: The radius of the circle mask
    :param mode: Determines what is inside and outside the mask. Must be interior or exterior
    :type mode: str
    :type r: int
    :param filter: The value for pixels outside the mask, defaults to 0.
    :type filter: int or float

    :return: Returns True if the image is too small to be a valid cluster, otherwise False.
    :rtype: bool
    """

    if not mode == 'interior' and not mode == 'exterior':
        print(mode, "is not a supported mode. Please enter interior or exterior")
        return 1

    #get the dimensions of the image
    n,m = img.shape

    #create an open grid for our image
    y,x = np.ogrid[0:n, 0:m]
    #operate on a copy of the image
    copyImg = img.copy()

    #get the x and y center points of our image
    center_x = cir_x
    center_y = cir_y

    #create a circle mask
    if mode == 'interior':
        circle_mask = (x-center_x)**2 + (y-center_y)**2 <= r**2
    elif mode == 'exterior':
        circle_mask = (x-center_x)**2 + (y-center_y)**2 >= r**2

    #black out anywhere within the circle mask
    copyImg[circle_mask] = [filter]
    copyImg[copyImg != filter] = [255-filter]

    return copyImg

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
    elif input_mode == 'np':
        np_img = input_img
    else:
        return (input_mode, " is not a supported mode. Supported modes are 'np' or 'fp'.")

    base_img = circleMask(np_img, x, y, r, 'exterior')

    core_img = circleMask(np_img, x, y, r*0.8, 'exterior')
    core_count = np.count_nonzero(base_img*core_img)

    inner_img = circleMask(np_img, x, y, r*0.8, 'exterior')
    inner_ring = base_img - inner_img
    inner_count = np.count_nonzero(inner_ring)

    outer_img = circleMask(np_img, x, y, r*1.2, 'exterior')
    outer_ring = outer_img - base_img
    outer_count = np.count_nonzero(outer_ring)



    return (core_count, inner_count, outer_count)

# multiprocessing

def segmentationVisionQueue(q, output_dir, save_intermediate=False, overwrite=False):
    """ This is a queue for multiprocessing. 

    :param q: the queue generated by segment Directory
    :type q: Multiprocessing.Queue object
    :param output_dir: the filepath to the directory where outputs will be saved.
    :type output_dir: str
    :param save_intermediate: Flag to save intermediate outputs to disk, defaults to False.
    :type save_intermediate: bool
    :param overwrite: Flag to overwrite existing outputs, defaults to False.
    :type overwrite: bool
    """

    while not q.empty():
        try:
            input_img = q.get()
            print("input img: ", input_img)
            # this is hard coded to run on my registered rasters only
            img_info = input_img.split('/')
            img_id = img_info[-1]
            img_id = img_id[:-4]
            output_root = output_dir + '/' + img_id
            applySegmentationSteps(input_img, 'fp', output_root, save_intermediate=save_intermediate, overwrite=overwrite)
        except ValueError as val_error:
            print(val_error)
        except Exception as error:
            print(error)

def segmentDirectoryMP(input_dir, output_root, num_workers, target='.tif', save_intermediate=False, overwrite=False):
    """ A multiprocessing segmenter.
    :param input_dir: Filepath to the input directory.
    :type input_dir: str
    :param output_root: The filepath to the directory where outputs will be created.
    :type output_root: str
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
    for i in sorted(os.listdir(input_dir)):
        if i.endswith(target):
            input_img = input_dir + '/' + i
            q.put(input_img)
        else:
            continue
    workers = Pool(num_workers, segmentationVisionQueue,(q, output_root, save_intermediate, overwrite))
    workers.close()
    workers.join()

def calculatePixelMetricsMP(input_img, input_df, num_workers=8):
    """ A multiproessing pixel metrics implementation.

    :param input_img: Filepath to the input img
    :type input_img: str
    :param input_df: Dataframe with detections
    :type input_df: pandas dataframe
    :param num_workers: Number of workers to assign, defaults to 8.
    :type num_workers: int

    :return: Returns the dataframe augmented with pixel metrics.
    :rtype: pandas dataframe
    """

    manager = Manager()
    new_cir = manager.list()
    q = Queue()
    for index, row in input_df.iterrows():
        plot = row['plot']
        x = row['x']
        y = row['y']
        r = row['r']
        weight = row['weight']
        info = [plot, x, y, r, weight]
        q.put(info)
    workers = Pool(num_workers, calculatePixelMetricsQueue,(q, input_img, input_df, new_cir))
    workers.close()
    workers.join()
        
    header = ['plot', 'x', 'y', 'r', 'weight', 'core', 'inner', 'outer']
    print(len(new_cir))
    output_df = pd.DataFrame(list(new_cir), columns=header)
    return output_df


def calculatePixelMetricsQueue(q, input_img, input_df, output_list):
    """ This is a queue for multiprocessing. 

    :param q: the queue generated by calculatePixelMetricsMP
    :type q: Multiprocessing.Queue object
    :param input_img: Filepath to the input img
    :type input_img: str
    :param input_df: Dataframe with detections
    :type input_df: pandas dataframe
    :param output_list: a Manager generated by calculatePixelMetricsMP
    :type output_list: Multiprocessing.Manager object
    
    """

    counter = 0
    while not q.empty():
        try:
            if counter % 10000 == 0 and counter > 0:
                print(counter)
            info = q.get()
            plot = info[0]
            x = info[1]
            y = info[2]
            r = info[3]
            weight = info[4]
            core, inner, outer = countPixels(input_img, 'fp', x, y, r)
            circle = ([plot, x, y, r, weight, core, inner, outer])
            output_list.append(circle)
            counter += 1
            
        except ValueError as val_error:
            print(val_error)
        except Exception as error:
            print(error)
