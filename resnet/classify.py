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
# import cv2
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision import datasets, transforms, models
from sklearn.metrics import average_precision_score
import copy
from shutil import copyfile
import torch.nn.functional as FF
from albumentations.pytorch import ToTensorV2

# from FullyConvolutionalResnet18 import FullyConvolutionalResnet18


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
    if scale>10000:
        area = img.shape[1] * img.shape[2]
        q = np.sqrt(scale/area)
    elif scale<0:
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

# ITEM, SCALE, FC = 'Deteriorated', 1024, 256
# # ITEM, SCALE, FC = 'Vegetation', 1024, 256

# ##  dump_aep  dump_claire  dump_master  dist
# DATA_ROOT = '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/stuff/'
# DATA_PATH = DATA_ROOT + 'dist/'
# MODEL_ROOT = '/home/david/code/davidsvaughn/ai_utils/resnet/models'
# MODEL_FILE = f'{MODEL_ROOT}/{ITEM.lower()}/{ITEM.lower()}.pt'

# SAVE_PATH = DATA_ROOT + f'{ITEM}/maybe/'
# mkdirs(SAVE_PATH)

####################

ITEM, SCALE, RES, FC, NUMCLASS = 'Insulator_Type', 480, 18, 64, 3

ALBUM = False
ALBUM = True

# ITEM, SCALE, RES, FC, NUMCLASS = 'Insulator_Material', 480, 18, 64, 3


ROOT  = '/home/david/code/phawk/data/generic/transmission/master/attribs/'
MODEL_PATH = os.path.join(ROOT, 'models', ITEM)
MODEL_FILE  = os.path.join(MODEL_PATH, f'{ITEM}.pt')
MODEL_FILE = os.path.join(MODEL_PATH, f'{ITEM}_chk.pt')
SAVE_PATH = os.path.join(ROOT, 'add', ITEM)
mkdirs(SAVE_PATH)

# DATA_PATH = '/home/david/code/phawk/data/generic/transmission/claire/detect/transmaster3/crops/Insulator/'
DATA_PATH = os.path.join(ROOT, 'test', ITEM)

## sanity check 
# DATA_PATH = '/home/david/code/phawk/data/generic/transmission/master/attribs/Insulator_Material/1/'
# DATA_PATH = '/home/david/code/phawk/data/generic/transmission/master/attribs/check/'

print(f'Loading model from:\n\t{MODEL_FILE}')

# sys.exit()

#####################################################################################

# def classify():
## get input data
F = np.array([path_leaf(f) for f in glob.glob(os.path.join(DATA_PATH,'*.jpg'))])
random.shuffle(F)

## load model
# model = models.resnet18()
if RES==50:
    model = models.resnet50()
elif RES==18:
    model = models.resnet18()
elif RES==101:
    model = models.resnet101()
else: # 34
    model = models.resnet34()

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs, NUMCLASS), nn.LogSoftmax(dim=1))
if FC>0:
    model.fc = nn.Sequential(nn.Linear(num_ftrs, FC),
                            nn.ReLU(),
                            nn.Dropout(0),
                            nn.Linear(FC, NUMCLASS),
                            nn.LogSoftmax(dim=1))
model.load_state_dict(torch.load(MODEL_FILE), strict=False)
model.to(device)
model.eval()

totensorv2 = ToTensorV2()

BS = 1
P,S,data,files = [],[],[],[]
for i,f in enumerate(F):
    if i>1000: break
    #################
    fn = os.path.join(DATA_PATH, f)
    img = np.array(Image.open(fn))
    
    # img = img[...,::-1].copy() ## BGR => RGB
    if not ALBUM:
        img *= (1/255)
    img = torch.FloatTensor(img.transpose([2,0,1]))
    
    # if ALBUM:
    #     # img =  totensorv2(image=img)['image']
    #     img = img.transpose([2,0,1])
    #     img = torch.FloatTensor(img)
    # else:
    #     # img = img[...,::-1].copy() ## BGR => RGB
    #     img = img.transpose([2,0,1]) * (1/255) ## [h,w,3] => [3,h,w]
    #     img = torch.FloatTensor(img)

    data.append(img)
    files.append(f)
    # if i>2 and ((i+1)%BS==0 or i==len(F)-1):
    if ((i+1)%BS==0 or i==len(F)-1):
        if i%100==0:
            print(f'{i}/{len(F)}')
        data = [scale_img(img, SCALE) for img in data]
        #######################################
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
        #######################################
        data = [img.to(device) for img in data]
        with torch.no_grad():
            p = torch.softmax(model(data[0].unsqueeze(0)), dim=1).cpu().numpy()
        P.append(p)
        S.extend(files)
        data,files = [],[]

P = np.vstack(P)
Y = P.argmax(1)

for y,s in zip(Y,S):
    dst_path = os.path.join(SAVE_PATH, f'{y}')
    mkdirs(dst_path)
    src = os.path.join(DATA_PATH, s)
    dst = os.path.join(dst_path, s)
    copyfile(src, dst)


# idx = np.argsort(P)
# P,F = P[idx], F[idx]

# plt.plot(P)
# plt.show()

# ## copy candidates
# C = 0.8
# idx = P>C
# idx = P<C
# print(f'Copying {idx.sum()} files...')

# for f in F[idx]:
#     src = DATA_PATH + f
#     dst = SAVE_PATH + f
#     copyfile(src, dst)
# print('done.')

# if __name__ == "__main__":
#     classify()