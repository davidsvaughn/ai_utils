import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from PIL import Image


f = 0.75
s = 0.03
n = 6

def dots(s,vx,vy):
    # global p
    h,w = vx.shape
    nx,ny = (w*s), (h*s)
    p = int( max( min(nx,ny)/n , 1) )
    # p = min(p, int(1 + p**((p-1)/p)))
    p = min(p, int(p**((p-1)/p)))
    
    nx,ny = int(nx), int(ny)
    # dx,dy = vx%nx==0, vy%ny==0
    dx,dy = (vx/p-0.5).astype('int')%n==0, (vy/p-0.5).astype('int')%n==0
    return dx*dy*1

def border(f,vx,vy):
    h,w = vx.shape
    if h<w:
        fy = f
        fx = 1-h*(1-f)/w
    else:
        fx = f
        fy = 1-w*(1-f)/h
    mx = 2*abs(vx-(w-1)/2)/w > fx
    my = 2*abs(vy-(h-1)/2)/h > fy
    return (mx+my)*1


path = '/home/david/code/phawk/data/generic/transmission/claire/detect/flashcrack5/'
fn = path + 'test1.jpg'
fn = path + 'test3.jpg'
fn = path + 'test4.jpg'
fn = path + 'test2.jpg'
fn = path + 'test5.jpg'

img = cv2.imread(fn)
h,w = img.shape[:2]


#####

vy,vx = torch.meshgrid([torch.arange(h), torch.arange(w)])
vx,vy = vx.numpy(), vy.numpy()

# # mx = 2*abs(vx-(w-1)/2)/w > f
# # my = 2*abs(vy-(h-1)/2)/h > f
# if h<w:
#     fy = f
#     fx = 1-h*(1-f)/w
# else:
#     fx = f
#     fy = 1-w*(1-f)/h
# mx = 2*abs(vx-(w-1)/2)/w > fx
# my = 2*abs(vy-(h-1)/2)/h > fy
# m = (mx+my)*1
# # plt.imshow(m)
# # sys.exit()

#####

# nx,ny = (w*s), (h*s)
# p = int( max( min(nx,ny)/n , 1) )
# nx,ny = int(nx), int(ny)
# # dx,dy = vx%nx==0, vy%ny==0
# # dx,dy = (vx/p-0.5).astype('int')%nx==0, (vy/p-0.5).astype('int')%ny==0
# dx,dy = (vx/p-0.5).astype('int')%n==0, (vy/p-0.5).astype('int')%n==0
# d = dx*dy*1
# plt.imshow(d)
# # sys.exit()

###
m = border(f, vx, vy)
d = dots(s, vx, vy)
g = m*d

idx = (g==1)
img[idx] = [0,255,255]

cv2.imwrite(fn.replace('.jpg','_a.jpg'), img)
