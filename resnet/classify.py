import os, sys
import random
import glob
import ntpath
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.optim import lr_scheduler
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision import datasets, transforms, models
from sklearn.metrics import average_precision_score
import copy
from shutil import copyfile
import torch.nn.functional as FF

# from FullyConvolutionalResnet18 import FullyConvolutionalResnet18

## clear CUDA cache...
import gc
gc.collect()
torch.cuda.empty_cache()

'''
https://github.com/cfotache/pytorch_imageclassifier/blob/master/PyTorch_Image_Training.ipynb
https://stackabuse.com/image-classification-with-transfer-learning-and-pytorch/
https://github.com/spmallick/learnopencv/blob/master/PyTorch-Fully-Convolutional-Image-Classification/FullyConvolutionalResnet18.py
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def scale_img_1d(img, scale):
    if scale<0:
        scale = -scale
        q = scale/min(img.shape[1:])
    else:
        q = scale/max(img.shape[1:])
    if q<1:
        h,w = int(img.shape[1]*q), int(img.shape[2]*q) 
        return F.interpolate(img.unsqueeze(0), size=[h,w]).squeeze()
    return img

def scale_img_2d(img, scale):
    area = img.shape[1] * img.shape[2]
    q = np.sqrt(scale/area)
    if q<1:
        h,w = int(img.shape[1]*q), int(img.shape[2]*q) 
        return F.interpolate(img.unsqueeze(0), size=[h,w]).squeeze()
    return img

# def scale_img(img):
#     q = 2
#     h,w = int(img.shape[1]*q), int(img.shape[2]*q) 
#     return F.interpolate(img.unsqueeze(0), size=[h,w]).squeeze()

def scale_img(img, scale, pad=10):
    if scale<0:
        scale = -scale
        q = scale/min(img.shape[1:])
    else:
        q = scale/max(img.shape[1:])
    if q<1:
        h,w = int(img.shape[1]*q), int(img.shape[2]*q)
        img = FF.interpolate(img.unsqueeze(0), size=[h,w]).squeeze()
    if pad>0:
        img = FF.pad(img, pad=(pad, pad, pad, pad), mode='constant', value=0)
    return img


#####################################################################################

ITEM, SCALE, FC = 'Deteriorated', 1024, 256
# ITEM, SCALE, FC = 'Vegetation', 1024, 256

##  dump_aep  dump_claire  dump_master  dist
DATA_ROOT = '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/stuff/'
DATA_PATH = DATA_ROOT + 'dist/'


MODEL_ROOT = '/home/david/code/davidsvaughn/ai_utils/resnet/models'
MODEL_FILE = f'{MODEL_ROOT}/{ITEM.lower()}/{ITEM.lower()}.pt'

SAVE_PATH = DATA_ROOT + f'{ITEM}/maybe/'
mkdirs(SAVE_PATH)

F = np.array([path_leaf(f) for f in glob.glob(os.path.join(DATA_PATH,'*.jpg'))])

## load model
model = models.resnet34()
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs, 2), nn.LogSoftmax(dim=1))
if FC>0:
    model.fc = nn.Sequential(nn.Linear(num_ftrs, FC),
                              nn.ReLU(),
                              nn.Dropout(0),
                              nn.Linear(FC, 2),
                              nn.LogSoftmax(dim=1))
model.load_state_dict(torch.load(MODEL_FILE), strict=False)
model.to(device)
model.eval();

BS = 1
P,data = [],[]
for i,f in enumerate(F):
    # if i>100: break
    #################
    fn = os.path.join(DATA_PATH, f)
    img = np.array(Image.open(fn))
    
    img = img[...,::-1].copy() ## BGR => RGB
    img = torch.FloatTensor(img.transpose([2,0,1]) * (1/255)) ## [h,w,3] => [3,h,w]

    data.append(img)
    if i>2 and ((i+1)%BS==0 or i==len(F)-1):
        if i%100==0:
            print(f'{i}/{len(F)}')
        data = [scale_img(img, SCALE) for img in data]
        # szs = np.array([d.shape[1:] for d in data])
        # H,W = szs.max(0)
        # X = []
        # for i,d in enumerate(data):
        #     h,w = d.shape[1:]
        #     h1,w1 = (H-h)//2, (W-w)//2
        #     h2,w2 = H-h-h1, W-w-w1
        #     x = FF.pad(input=d, pad=(w1, w2, h1, h2), mode='constant', value=0)
            # X.append(d.unsqueeze(0))
        # data = torch.cat(X, dim=0).to(device)
        data = [img.to(device) for img in data]
        with torch.no_grad():
            p = torch.softmax(model(data[0].unsqueeze(0)), dim=1).cpu().numpy()[:,1]
        P.extend(p)
        data = []

P = np.array(P)
idx = np.argsort(P)
P,F = P[idx], F[idx]

plt.plot(P)
plt.show()


sys.exit()

## copy candidates
C = 0.8
idx = P>C
idx = P<C
print(f'Copying {idx.sum()} files...')

for f in F[idx]:
    src = DATA_PATH + f
    dst = SAVE_PATH + f
    copyfile(src, dst)
print('done.')