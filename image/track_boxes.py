import os,sys
import numpy as np
import ntpath
from glob import glob
import matplotlib.pyplot as plt
import scipy.optimize
import cv2
import random

icp_path = os.path.join(os.path.dirname(__file__), "icp2/")
sys.path.append(icp_path)
from icp import icp

jpg, txt = '.jpg', '.txt'

pad = 100
conf_thres  = 0.1
quant_thres = 0.5
canny_thres = 10000

##############################################################################

class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self
        
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_filenames(path, ext=jpg, noext=False):
    pattern = os.path.join(path, f'*{ext}')
    x = np.array([path_leaf(f) for f in glob(pattern)])
    if noext:
        x = np.array([f.replace(ext,'') for f in x])
    return x

def append_line(fn, s):
    with open(fn, 'a') as f:
        f.write(f'{s}\n')
        
def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def write_lines(fn, lines):
    with open(fn, 'w') as f:
        for line in lines:
            f.write(f'{line}\n')
            
def read_lines(fn):
    with open(fn, 'r') as f:
        lines = [n.strip() for n in f.readlines()]
    return [line for line in lines if len(line)>0]

def get_labels(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
    labs = np.array( [np.array(line.strip().split(' ')).astype(float).round(6) for line in lines])
    if len(labs)==0:
        return labs#.tolist()
    labs[:,1:] = labs[:,1:].clip(0,1)
    # labs = [[int(x[0])] + x[1:] for x in labs.tolist()]
    return labs

## matrix form: xywh2xy1xy2
def xywh2xyxy(xywh, clip=True):
    x,y,w,h = xywh[:,0],xywh[:,1],xywh[:,2],xywh[:,3]
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    A = np.dstack([x1,y1,x2,y2]).squeeze()
    if clip: A = A.clip(0,1)
    if len(A.shape)==2: return A
    return A[None,:]

def xywh2xyxy(xywh, W=None, H=None, clip=True):
    x,y,w,h = xywh[:,0],xywh[:,1],xywh[:,2],xywh[:,3]
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    A = np.dstack([x1,y1,x2,y2]).squeeze()
    if clip: 
        A = A.clip(0,1)
    if len(A.shape)!=2:
        A = A[None,:]
    if W is not None and H is not None:
        q = np.array([W,H,W,H])
        A = (A*q).round().astype(np.int32)
        if clip:
            A = np.clip(A, [0,0,0,0], q)
    return A

## matrix form
def xyxy2xywh(xyxy, clip=True):
    x1,y1,x2,y2 = xyxy[:,0],xyxy[:,1],xyxy[:,2],xyxy[:,3]
    w = x2-x1
    h = y2-y1
    x = x1+w/2
    y = y1+h/2
    A = np.dstack([x,y,w,h]).squeeze()
    if clip: A = A.clip(0,1)
    if len(A.shape)==2: return A
    return A[None,:]
        
def isint(x):
    try:
        int(x)
    except:
        return False
    return True

def quantile(x,q=0.5):
    i = int(q*len(x))
    return sorted(x)[i]



def find_transform(imf1, imf2, lf1, lf2, t1=-1, t2=-1, verbose=0):
    img1 = cv2.imread(imf1)[:,:,::-1]
    img2 = cv2.imread(imf2)[:,:,::-1]
    H,W = img1.shape[:2]  

    ## get labels
    x1 = get_labels(lf1)[:,1:]
    x1,conf = x1[:,:-1],x1[:,-1]
    q = min(quant_thres, quantile(conf, conf_thres))
    idx = conf > q
    x1 = x1[idx]
    boxes1 = xywh2xyxy(x1, W, H)
    
    ## constrain image to box area
    B = np.hstack([boxes1[:,:2].min(0), boxes1[:,2:].max(0)])
    B = B + np.array([-pad,-pad,pad,pad])
    B = np.clip(B, [0,0,0,0], [W,H,W,H])
                
    xmin,ymin,xmax,ymax = B
    img1 = img1[ymin:ymax,xmin:xmax,:]
    img2 = img2[ymin:ymax,xmin:xmax,:]
    h,w = img1.shape[:2]

    if verbose>3:
        plt.imshow(img1);plt.show()
        plt.imshow(img2);plt.show()
        
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    if t1<0 or t2<0:
        t1,t2 = 150,200
        for j in range(30):
            t1,t2 = t1+25,t2+25
            edges1 = cv2.Canny(image=img1_gray, threshold1=t1, threshold2=t2)
            if np.sum(edges1>0) < canny_thres:
                break
        if verbose>1:
            print(f't1={t1} t2={t2}')
    
    edges1 = cv2.Canny(image=img1_gray, threshold1=t1, threshold2=t2)
    edges2 = cv2.Canny(image=img2_gray, threshold1=t1, threshold2=t2)
    
    if verbose>1:
        plt.imshow(edges1, cmap='gray'); plt.title(i); plt.show()
        plt.imshow(edges2, cmap='gray'); plt.title(i); plt.show()
    
    x1,y1 = np.where(edges1>0)
    x2,y2 = np.where(edges2>0)
    P1 = np.column_stack([x1,y1])
    P2 = np.column_stack([x2,y2])
    
    if verbose>0:
        print(P1.shape)
    
    dim = np.array([H,W])
    P1 = P1/dim
    P2 = P2/dim
    
    T, P3, transformation_history = icp(P2, P1, 
                                         verbose=verbose>3, 
                                         distance_threshold=0.3, 
                                         convergence_translation_threshold=2e-4,
                                         convergence_rotation_threshold=5e-5,
                                         )
    
    if verbose<1:
        return T,(t1,t2)
    
    if verbose>4:
        P3 = (P3 * dim).round().astype(np.int32)
        y3,x3 = P3[:,0],P3[:,1]
        edges3 = edges1*0
        x3,y3 = x3.clip(0,w-1), y3.clip(0,h-1)
        edges3[y3,x3] = 255
        plt.imshow(edges3, cmap='gray'); plt.title(i); plt.show()
    
    
    P4 = np.hstack([P1,np.ones([P1.shape[0],1])])
    P4 = (P4@T.T)[:,:2]
    # P4 = P1.copy()
    # for M in transformation_history:
    #     R,T = M[:,:2],M[:,2]
    #     # P4 = np.dot(P4, R.T) + T
    #     P4 = (P4 @ R.T) + T
    
    P4 = (P4 * dim).round().astype(np.int32)
    y4,x4 = P4[:,0],P4[:,1]
    x4,y4 = x4.clip(0,w-1), y4.clip(0,h-1)
    edges4 = edges1*0
    edges4[y4,x4] = 255
    if verbose>1:
        plt.imshow(edges4, cmap='gray'); plt.title(i); plt.show()
        
    return T,(t1,t2)
        
###############################################################################

# root = '/home/david/code/phawk/data/generic/distribution/data/detect/DJI_0001/component/'
# root = '/home/david/code/phawk/data/generic/distribution/data/detect/DJI_0001_1/component/' # m1,m2 = 12,7

# root = '/home/david/code/phawk/data/generic/distribution/data/detect/DJI_0004/component/'

# root = '/home/david/code/phawk/data/generic/distribution/data/detect/DJI_0010/component/'
root = '/home/david/code/phawk/data/generic/distribution/data/detect/DJI_0010_1/component/' # m1,m2 = 12,5 #********

lab_path = root + 'labels/'
img_path = root + 'frames/'

lab_files = np.array(get_filenames(lab_path, txt))
idx = np.array([int(f.replace(txt,'').split('_')[-1]) for f in lab_files])
a = idx.argsort()
lab_files, idx = lab_files[a], idx[a]
img_files = np.array([f.replace(txt,jpg) for f in lab_files])

first = True

verbose = 4
step = 5
pad = 100


start = step
start = 100
# start = 200
for i in range(start, len(img_files)):
    i = 125
    
    print(i)
    
    imf1 = img_path + img_files[i-step]
    imf2 = img_path + img_files[i]
    
    lf1 = lab_path + lab_files[i-step]
    lf2 = lab_path + lab_files[i]
    
    img1 = cv2.imread(imf1)[:,:,::-1]
    img2 = cv2.imread(imf2)[:,:,::-1]
    
    H,W = img1.shape[:2]
    
    ## get labels
    x1 = get_labels(lf1)[:,1:]
    x1,conf = x1[:,:-1],x1[:,-1]
    q = min(quant_thres, quantile(conf, conf_thres))
    idx = conf > q
    x1 = x1[idx]
    boxes1 = xywh2xyxy(x1, W, H)
    
    ## constrain image to box area
    B = np.hstack([boxes1[:,:2].min(0), boxes1[:,2:].max(0)])
    B = B + np.array([-pad,-pad,pad,pad])
    B = np.clip(B, [0,0,0,0], [W,H,W,H])
                
    xmin,ymin,xmax,ymax = B
    img1 = img1[ymin:ymax,xmin:xmax,:]
    img2 = img2[ymin:ymax,xmin:xmax,:]
    
    h,w = img1.shape[:2]
    
    if verbose>3:
        plt.imshow(img1);plt.show()
        plt.imshow(img2);plt.show()
    # sys.exit()
    ###################################
    
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # img1_blur = cv2.GaussianBlur(img1_gray, (3,3), 0)
    
    # plt.imshow(img1_gray, cmap='gray'); plt.show()
    # plt.imshow(img2_gray, cmap='gray'); plt.show()
    
    # edges = cv2.Sobel(src=img1_gray)#, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    # t1,t2 = 50,200
    # t1,t2 = 100,200
    # t1,t2 = 250,250
    # t1,t2 = 250,350
    # t1,t2 = 300,400
    # t1,t2 = 350,400
    # t1,t2 = 600,600
    if first or i%10==0:
        first = False
        t1,t2 = 150,200
        for j in range(30):
            t1,t2 = t1+25,t2+25
            edges1 = cv2.Canny(image=img1_gray, threshold1=t1, threshold2=t2)
            # if np.mean(edges1>0) < 0.01:
            if np.sum(edges1>0) < canny_thres:
                break
        print(f't1={t1} t2={t2}')
    
    edges1 = cv2.Canny(image=img1_gray, threshold1=t1, threshold2=t2)
    edges2 = cv2.Canny(image=img2_gray, threshold1=t1, threshold2=t2)
    
    if verbose>1:
        plt.imshow(edges1, cmap='gray'); plt.title(i); plt.show()
        plt.imshow(edges2, cmap='gray'); plt.title(i); plt.show()
    
    x1,y1 = np.where(edges1>0)
    x2,y2 = np.where(edges2>0)
    P1 = np.column_stack([x1,y1])
    P2 = np.column_stack([x2,y2])
    
    if verbose>0:
        print(P1.shape)
    
    dim = np.array([H,W])
    P1 = P1/dim
    P2 = P2/dim
    
    T, P3, transformation_history = icp(P2, P1, 
                                         verbose=verbose>3, 
                                         distance_threshold=0.3, 
                                         convergence_translation_threshold=2e-4,
                                         convergence_rotation_threshold=5e-5,
                                         )
    if verbose>4:
        P3 = (P3 * dim).round().astype(np.int32)
        y3,x3 = P3[:,0],P3[:,1]
        edges3 = edges1*0
        x3,y3 = x3.clip(0,w-1), y3.clip(0,h-1)
        edges3[y3,x3] = 255
        plt.imshow(edges3, cmap='gray'); plt.title(i); plt.show()
    
    
    P4 = np.hstack([P1,np.ones([P1.shape[0],1])])
    P4 = (P4@T.T)[:,:2]
    
    # P4 = P1.copy()
    # for M in transformation_history:
    #     R,T = M[:,:2],M[:,2]
    #     # P4 = np.dot(P4, R.T) + T
    #     P4 = (P4 @ R.T) + T
    
    P4 = (P4 * dim).round().astype(np.int32)
    y4,x4 = P4[:,0],P4[:,1]
    x4,y4 = x4.clip(0,w-1), y4.clip(0,h-1)
    edges4 = edges1*0
    edges4[y4,x4] = 255
    if verbose>1:
        plt.imshow(edges4, cmap='gray'); plt.title(i); plt.show()
    
    if verbose>2:
        break
            