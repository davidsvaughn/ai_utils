import sys,os
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

import scipy.signal as signal
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.transform import rescale
from scipy.signal import convolve2d
from numpy import argmax
import torch

jpg,txt = '.JPG','.txt'

def rms_flat(a):
    return np.sqrt(np.mean(np.absolute(a)**2))

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def gray_convolve2d(image, kernel):
    return convolve2d(image, kernel, 'valid')

def load_labels(lab_file):
    if not os.path.exists(lab_file):
        return None
    with open(lab_file, 'r') as f:
        labels = [x.split() for x in f.read().strip().splitlines()]
    return np.array(labels, dtype=np.float64)

def xywhn2xyxy(x, h, w):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2)  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2)  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2)  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2)  # bottom right y
    return y.round().astype(np.int32).clip([0,0,0,0,],[w-1,h-1,w-1,h-1])

#####################################################################

img_path = '/home/david/code/phawk/data/solar/thermal/component/s3/Clay_nms/'
lab_path = '/home/david/code/phawk/data/solar/thermal/component/models/model3/detect/Clay_nms/labels/'

# fn = '7131_DJI_0016'
fn = '7131_DJI_0081'

conf_thres = 0.001
#######################


img_file = img_path+fn+jpg
lab_file = lab_path+fn+txt

img = cv2.imread(img_file)
h,w = img.shape[:2]

## filter by conf
labs = load_labels(lab_file)[:,1:]
idx = labs[:,-1]>conf_thres
labs = labs[idx]

##
c = labs[:,-1]
# plt.hist(c,50); plt.show()
# c /= c.sum()
X = xywhn2xyxy(labs[:,:4], h, w)

## sum over all bounding box masks * conf
mask = np.zeros((h,w), dtype=np.float64) # initialize mask
m = mask.copy()

wid = 1
for i,xyxy in enumerate(X):
    m *= 0
    # m[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = c[i]
    
    xc = int((xyxy[0]+xyxy[2])/2)
    m[xyxy[1]:xyxy[3], xc-wid:xc+wid] = c[i]
    
    mask += m
    
mask /= mask.max()

plt.imshow(mask, cmap='hot', interpolation='nearest')
plt.show()

plt.imshow(mask[280:320,200:300], cmap='hot', interpolation='nearest')
plt.show()


###########################################

# y = mask[300,:]
# plt.plot(y)
# plt.show()


###########################################
## specgram....
# Fs = 40
# NFFT = 60
# noverlap = 30

# powerSpectrum, freqFound, time, imageAxis = plt.specgram(x, 
#                                                          Fs=Fs, 
#                                                          NFFT=NFFT, 
#                                                          noverlap=noverlap,
#                                                          )
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.show()

###########################################
## lombscargle...

# rng = np.random.default_rng()
# A = 2.
# w0 = 1.  # rad/sec
# nin = 150
# nout = 100000

# x = rng.uniform(0, 10*np.pi, nin)
# y = A * np.cos(w0*x)
# w = np.linspace(0.01, 10, nout)

# pgram = signal.lombscargle(x, y, w, normalize=True)

# fig, (ax_t, ax_w) = plt.subplots(2, 1, constrained_layout=True)
# ax_t.plot(x, y, 'b+')
# ax_t.set_xlabel('Time [s]')

# ax_w.plot(w, pgram)
# ax_w.set_xlabel('Angular frequency [rad/s]')
# ax_w.set_ylabel('Normalized amplitude')
# plt.show()

######################


y = mask[300,:]
# y = y[200:400]
# y = y[200:300]
y = y[100:300]

n = len(y)
x = np.arange(n)

h,bins,_ = plt.hist(X[:,2]-X[:,0],100)
z = bins[h.argmax()]
print(f'width mode: {z:0.2f}')
# Z = (X[:,2]-X[:,0]) * (c/c.sum())

fig, (ax_t, ax_w) = plt.subplots(2, 1, constrained_layout=True)
# ax_t.plot(x, y, 'b+')
ax_t.plot(y)
ax_t.set_xlabel('pixels')


Y = abs(np.fft.fft(y))/n

freq = x
freq = np.fft.fftfreq(n)

freq = freq[1:n//2]
Y = Y[1:n//2]

ax_w.plot(freq, Y)
ax_w.set_xlabel('freq')
ax_w.set_ylabel('amp')
plt.show()

q = sorted(Y)[-4]
p = 1/freq[Y>q]
print(f'peaks: {p}')




sys.exit()
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
