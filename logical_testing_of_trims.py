from scipy_vision_tools import trim2DArray, loadImage
from skimage import color

import matplotlib.pyplot as plt

img1 = '/Users/theo/data/array_trimming_test_rasters/1.tif'
img2 = '/Users/theo/data/array_trimming_test_rasters/2.tif'
img3 = '/Users/theo/data/array_trimming_test_rasters/3.tif'

np = loadImage(img1)

np = color.rgb2gray(np)

trimmed = trim2DArray(np)

plt.imshow(trimmed)
plt.show()