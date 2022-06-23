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

# from matplotlib.mlab import rms_flat
try:
    # More accurate peak finding from
    # https://gist.github.com/endolith/255291#file-parabolic-py
    from parabolic import parabolic

    def argmax(x):
        return parabolic(x, numpy.argmax(x))[0]
except ImportError:
    from numpy import argmax
    
def rms_flat(a):
    return np.sqrt(np.mean(np.absolute(a)**2))

def np2pil(x):
    return Image.fromarray(np.uint8(x*255))

def pil2np(x):
    return np.array(x)

def pil2cv(x):
    return np.array(x)[:, :, ::-1]

def cv2pil(x):
    return Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def blur(img, k=3, mode=1):
    # global x
    k = 2*(k//2)+1
    img = np.array(img)[:, :, ::-1]
    
    if mode==1:
        x = cv2.blur(img, (k,k))
    else:
        x = cv2.GaussianBlur(img, (k,k), 1)
    
    ###################
    # kernel = np.ones((k,k),np.float32)/(k*k)
    # x = cv2.filter2D(img, -1, kernel)
    # x = cv2.boxFilter(img, 0, ksize=(k,k))
    ## bilateralFilter
    # x = cv2.bilateralFilter(img, 9, 75, 75)
    # x = cv2.bilateralFilter(img, 20, 100, 100)
    # x = cv2.bilateralFilter(img, 30, 100, 100)
    ########
    return Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))

def save_np(im, f, gray=True):
    if 'PIL' not in str(type(im)):
        im = Image.fromarray(im)
    if gray:
        im = im.convert('L')
    im.save(f)
    
def up(x):
    return (x*255).round().astype(np.uint8)

def gray_convolve2d(image, kernel):
    return convolve2d(image, kernel, 'valid')

#####################################################################
   
# img_path = '/home/david/code/davidsvaughn/ai_utils/dsv/solar/dam/' 
# img_file = '20210107_105906_R.jpg'
# img_file = '20210107_105912_R.jpg'
# img_file = 'crop1.jpg'
# img_file = '20210107_102311_R.jpg'

img_path = '/home/david/code/davidsvaughn/ai_utils/dsv/solar/clay/'
img_file = '7131_DJI_0017.JPG'

###########################################

fn = img_path + img_file

img0 = cv2.imread(fn)
# img0 = img0[:, :, ::-1].transpose(2, 0, 1)
# plt.imshow(img0); plt.show()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


## HARRIS CORNER
## https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html

img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
# plt.imshow(img, cmap='Greys_r'); plt.show()

# img = clahe.apply(img.astype(np.uint8))

## invert???
# img = 255-img

gray = np.float32(img)
# dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
# dst = cv2.cornerHarris(gray, 2, 3, 0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
img0[dst>0.01 * dst.max()] = [0,0,255]
plt.imshow(img0); plt.show()


sys.exit()
###########################################
## Fourier
## https://python.plainenglish.io/bizarre-image-fourier-transform-explained-with-python-e37bd1ffe1be

#gray_img = img_as_ubyte(img)

# f = np.fft.fft2(img)
# f_s = np.fft.fftshift(f)
# plt.figure(num=None, figsize=(10, 8), dpi=80)
# plt.imshow(np.log(abs(f_s)), cmap='gray');

## 3D
# yy,xx = np.mgrid[0:img.shape[0],0:img.shape[1]]
# fig   = plt.figure(figsize=(10,8))
# ax    = plt.axes(projection='3d')
# ax.plot_surface(xx,yy,img,cmap='gray',edgecolor='none')
# ax.set_zlim(0,500)


############################################
## Radon
## https://stackoverflow.com/questions/46084476/radon-transformation-in-python

img0 = cv2.imread(fn)
img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
h, w = img.shape

# img = clahe.apply(img.astype(np.uint8))

img = img - np.mean(img)  # Demean; make the brightness extend above and below zero
# Do the radon transform
sinogram = radon(img)

plt.subplot(2, 2, 2)
plt.imshow(sinogram.T, aspect='auto')
plt.gray()
plt.show()

# Find the RMS value of each row and find "busiest" rotation,
# where the transform is lined up perfectly with the alternating dark
# text and white lines
r = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.transpose()])
rotation = np.argmax(r)
print('Rotation: {:.2f} degrees'.format(90 - rotation))

# Rotate and save with the original resolution
M = cv2.getRotationMatrix2D((w/2, h/2), 90 - rotation, 1)
dst = cv2.warpAffine(img0, M, (w, h))
plt.imshow(dst)
plt.show()


#############################################
## rotate then convolve with gratings

img = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

k = 10
sigma = 8

x = np.arange(-k, k+1, 1)
X, Y = np.meshgrid(x, x)
grating = np.sin(2 * np.pi * X / sigma)
plt.set_cmap("gray")

conv1 = gray_convolve2d(img, grating[::-1, ::-1])
conv1 = conv1/conv1.max()
plt.imshow(conv1)
plt.show()

sys.exit()
#############################################
## rotate then convolve

img = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

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
figure, axis = plt.subplots(2,3, figsize=figsize)
for kernel, name, ax in zip(kernels, kernel_name, axis.flatten()):
     conv_im1 = convolve2d(img, 
                           kernel[::-1, ::-1]).clip(0,1)
     ax.imshow(abs(conv_im1), cmap='gray')
     ax.set_title(name)



sys.exit()
############################################
## Radon 2
## https://gist.github.com/endolith/334196bac1cac45a4893#file-rotation_spacing-py

img0 = cv2.imread(fn)
img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# img = 255-img

img = img - np.mean(img)
plt.subplot(2, 2, 1)
plt.imshow(img)

sinogram = radon(img)
plt.subplot(2, 2, 2)
plt.imshow(sinogram.T, aspect='auto')
plt.gray()

# Find the RMS value of each row and find "busiest" rotation,
# where the transform is lined up perfectly with the alternating dark
# text and white lines
r = np.array([rms_flat(line) for line in sinogram.transpose()])
rotation = np.argmax(r)
print('Rotation: {:.2f} degrees'.format(90 - rotation))
plt.axhline(rotation, color='r')

# Plot the busy row
row = sinogram[:, rotation]
N = len(row)
plt.subplot(2, 2, 3)
plt.plot(row)
plt.show()

# Take spectrum of busy row and find line spacing
window = np.blackman(N)
spectrum = rfft(row * window)
plt.plot(row * window)
frequency = np.argmax(abs(spectrum))
line_spacing = N / frequency  # pixels
print('Line spacing: {:.2f} pixels'.format(line_spacing))

plt.subplot(2, 2, 4)
plt.plot(abs(spectrum))
plt.axvline(frequency, color='r')
plt.yscale('log')
plt.show()
