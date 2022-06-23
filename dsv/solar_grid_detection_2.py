import sys
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.cm as mtpltcm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import color
from skimage import io
from PIL import Image, ImageEnhance
import random
from skimage.util import img_as_ubyte
from mpl_toolkits import mplot3d
from skimage.transform import radon
from numpy.fft import rfft

from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.transform import rescale
from scipy.signal import convolve2d

def rgb_convolve2d(image, kernel):
    red = convolve2d(image[:,:,0], kernel, 'valid')
    green = convolve2d(image[:,:,1], kernel, 'valid')
    blue = convolve2d(image[:,:,2], kernel, 'valid')
    return np.stack([red, green, blue], axis=2)

def gray_convolve2d(image, kernel):
    return convolve2d(image, kernel, 'valid')

#####################################################################
## https://medium.com/swlh/image-processing-with-python-convolutional-filters-and-kernels-b9884d91a8fd

img_file = '20210107_105906_R.jpg'

# img_file = '20210107_105912_R.jpg'
# img_file = 'crop1.jpg'

# img_file = '20210107_102311_R.jpg'


img_path = '/home/david/code/davidsvaughn/ai_utils/dsv/solar/'
fn = img_path + img_file

img0 = cv2.imread(fn)
# img0 = img0[:, :, ::-1].transpose(2, 0, 1)
# plt.imshow(img0); plt.show()


img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
# plt.imshow(img, cmap='Greys_r'); plt.show()

# sys.exit()

# identity = np.array([[0, 0, 0],
#                      [0, 1, 0],
#                      [0, 0, 0]])
# conv_im1 = gray_convolve2d(img, identity)
# fig, ax = plt.subplots(1,2, figsize=(12,5))
# ax[0].imshow(identity, cmap='gray')
# ax[1].imshow(abs(conv_im1), cmap='gray');


# Edge Detection1
kernel1 = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]])
# Edge Detection2
kernel2 = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]])
# Bottom Sobel Filter
kernel3 = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])
# Top Sobel Filter
kernel4 = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])
# Left Sobel Filter
kernel5 = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])
# Right Sobel Filter
kernel6 = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])


kernels = [kernel1, kernel2, kernel3, kernel4, kernel5, kernel6]
kernel_name = ['Edge Detection #1', 'Edge Detection #2', 
               'Bottom Sobel', 'Top Sobel', 
               'Left Sobel', 'Right Sobel']

figsize = (12,6)
figsize = (20,10)


# figure, axis = plt.subplots(2,3, figsize=figsize)
# for kernel, name, ax in zip(kernels, kernel_name, axis.flatten()):
#      conv_im1 = convolve2d(img, 
#                            kernel[::-1, ::-1]).clip(0,1)
#      ax.imshow(abs(conv_im1), cmap='gray')
#      ax.set_title(name)



# sys.exit()
###########################################################################
## sinusoidal grating
## https://thepythoncodingbook.com/2021/08/30/2d-fourier-transform-in-python-and-fourier-synthesis-of-images/

import numpy as np
import matplotlib.pyplot as plt

k = 20
sigma = 6

x = np.arange(-k, k+1, 1)
X, Y = np.meshgrid(x, x)
y = np.sin(2 * np.pi * x / sigma)

# plt.plot(x, y)
# plt.show()

grating = np.sin(2 * np.pi * X / sigma)
plt.set_cmap("gray")

# plt.imshow(grating)
# plt.show()


conv1 = gray_convolve2d(img, grating[::-1, ::-1])

conv1 = conv1/conv1.max()
plt.imshow(conv1)
plt.show()









sys.exit()
#################################################################


# Sharpen
kernel7 = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
# Emboss
kernel8 =  np.array([[-2, -1, 0],
                     [-1,  1, 1],
                     [ 0,  1, 2]])
# Box Blur
kernel9 = (1 / 9.0) * np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]])
# Gaussian Blur 3x3
kernel10 = (1 / 16.0) * np.array([[1, 2, 1],
                                  [2, 4, 2],
                                  [1, 2, 1]])
# Gaussian Blur 5x5
kernel11 = (1 / 256.0) * np.array([[1, 4, 6, 4, 1],
                                   [4, 16, 24, 16, 4],
                                   [6, 24, 36, 24, 6],
                                   [4, 16, 24, 16, 4],
                                   [1, 4, 6, 4, 1]])
# Unsharp masking 5x5
kernel12 = -(1 / 256.0) * np.array([[1, 4, 6, 4, 1],
                                   [4, 16, 24, 16, 4],
                                   [6, 24, -476, 24, 6],
                                   [4, 16, 24, 16, 4],
                                   [1, 4, 6, 4, 1]])
kernels = [kernel7, kernel8, kernel9, kernel10, kernel11, kernel12]
kernel_name = ['Sharpen', 'Emboss', 'Box Blur', 
               '3x3 Gaussian Blur', '5x5 Gaussian Blur', 
               '5x5 Unsharp Masking']

figure, axis = plt.subplots(2,3, figsize=figsize)
for kernel, name, ax in zip(kernels, kernel_name, axis.flatten()):
     conv_im1 = gray_convolve2d(img, 
                                kernel[::-1, ::-1]).clip(0,1)
     ax.imshow(abs(conv_im1), cmap='gray')
     ax.set_title(name)