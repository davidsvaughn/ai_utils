import sys,os
import json
# install hasty library: pip install hasty
from hasty import Client
import numpy as np
import cv2
import ntpath, random
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

data_root = '/home/david/code/phawk/data/fpl/component/'
directory_images  = data_root + 'images'
directory_labels  = data_root + 'labels'
yolo_classes_file = data_root + 'classes.txt'

fig_root = '/home/david/code/phawk/data/fpl/component/figs/'

jpg,txt = '.jpg','.txt'

def memoize_img_dim(img_file):
    img_dim_file = img_file.replace('.jpg','.npy')
    if os.path.isfile(img_dim_file):
        img_dim = np.load(img_dim_file)
    else:
        img = cv2.imread(img_file)
        img_dim = np.array(img.shape[:2][::-1])
        np.save(img_dim_file, img_dim)
    width,height = img_dim
    return int(height), int(width)

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_filenames(path, ext=jpg):
    pattern = os.path.join(path, f'*{ext}')
    return np.array([path_leaf(f) for f in glob(pattern)])

def get_labels(lab_path):
    if not os.path.isfile(lab_path):
        labs = []
    else:
        with open(lab_path,'r') as f:
            labs = [s.strip() for s in f.readlines()]
    return np.array([lab.split(' ') for lab in labs]).astype(np.float32)
    
def xywhn2xyxy(x, w=640, h=640):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2)  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2)  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2)  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2)  # bottom right y
    return y.round().astype(np.int32)

## filter out degenerate label boxes (h,w == 0)
def check(x,y):
    w = x[:,2]-x[:,0]
    h = x[:,3]-x[:,1]
    idx = np.minimum(w,h)>0
    return x[idx],y[idx]

###############################################################################

max_dim = 3008

## Get yolo class list
with open(yolo_classes_file,'r') as f:
    classes = [s.strip() for s in f.readlines()]
    
dim_dict = { i:[] for i,_ in enumerate(classes) }
    
img_files = get_filenames(directory_images)
img_files.sort()
# random.shuffle(img_files)
file_number = 0
for filename in img_files:
    file_number+=1
    ############################
    # if file_number<815: continue
    # if file_number>1500: break
    ############################
    
    img_path = (os.path.join(directory_images, filename))
    base = os.path.basename(img_path)
    file_name_without_ext = os.path.splitext(base)[0] # name of the file without the extension
    lab_path  = os.path.join(directory_labels, file_name_without_ext+ "." + 'txt')
    img_name = os.path.basename(img_path) # name of the file without the path
    height,width = memoize_img_dim(img_path)
    ## rescale dimensions...
    f =  max_dim/max(height, width)
    if f<1:
        height, width = int(f*height), int(f*width)
    
    labs = get_labels(lab_path)
    if len(labs)==0:
        # print(f'{img_name} has no labels')
        continue
    
    y,x = labs[:,0].astype(np.int32), xywhn2xyxy(labs[:,1:], width, height)
    ## filter out bad label boxes...
    x,y = check(x,y)
    
    if file_number % 25 == 0: print(f'{file_number}/{len(img_files)}\t{img_name}')
    
    dims = x[:,2:]-x[:,:2]
    for yy in np.unique(y):
        idx = (y==yy)
        dim_dict[yy].append(dims[idx])

T = []
for i,name in enumerate(classes):
    # if i!=5: continue
    ####################
    dim_dict[i] = np.vstack(dim_dict[i])
    # X,Y = dim_dict[i][:,0], dim_dict[i][:,1]
    
    ## remove outliers....
    df = pd.DataFrame(dim_dict[i])
    df = df[(np.abs(stats.zscore(df)) < 2.5).all(axis=1)]
    X,Y = df[0].values, df[1].values
    
    ### set bin width manually...
    # binsize = 20
    # xMin,xMax = X.min(), X.max()
    # yMin,yMax = Y.min(), Y.max()
    # bins = [np.arange(xMin, xMax, binsize), np.arange(yMin, yMax, binsize)]
    ###
    
    ## histogram ...
    bins = 20 ## set num bins
    h, xb, yb, _ = plt.hist2d(X, Y, bins=bins)
    
    ## get mode...
    gx, gy = np.argwhere(h==h.max())[-1]
    T.append([int(xb[gx+1]), int(yb[gy+1])])
    # T.append([int(xb[gx:gx+2].mean()), int(yb[gy:gy+2].mean())])
    
    ## plot...
    plt.xlabel('width (pixels)')
    plt.ylabel('height (pixels)')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('counts')
    plt.title(name)
    figfile = fig_root + f"{name.replace(' ','')}.png"
    # plt.savefig(figfile, bbox_inches='tight')
    plt.show()
    
    # break

T = np.array(T)
for i,name in enumerate(classes):
    print(f'[{T[i,0]}\t{T[i,1]}]  \t{name}')
    