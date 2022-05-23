import os,sys,ntpath
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.optimize
import cv2
from PIL import Image, ImageFilter
from matplotlib.patches import Rectangle
# from sklearn.metrics.pairwise import euclidean_distances as ed
# import itertools 
# from itertools import product 
import random
from random import shuffle as shuf
# from queue import PriorityQueue
# from functools import total_ordering
from scipy.spatial.distance import cdist
from shutil import copyfile

txt,jpg,JPG = '.txt','.jpg','.JPG'

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

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_filenames(path, ext=jpg, noext=False):
    pattern = os.path.join(path, f'*{ext}')
    x = np.array([path_leaf(f) for f in glob(pattern)])
    if noext:
        x = np.array([f.replace(ext,'') for f in x])
    return x

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

def fixbox(a):
    b = a.copy()
    if b[0]>b[2]:
        b[0],b[2] = b[2],b[0]
    if b[1]>b[3]:
        b[1],b[3] = b[3],b[1]
    return b

def fixboxes(A):
    return np.array([fixbox(a) for a in A])

def box_iou(boxA, boxB):
    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # ^^ corrected.
    # boxA = fixbox(boxA)
    # boxB = fixbox(boxB)
    
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interW = xB - xA
    interH = yB - yA
    
    # Correction: reject non-overlapping boxes
    if interW <=0 or interH <=0 : 
        return 0
    
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def _bbox_iou(boxA, boxB):
    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # ^^ corrected.
    # boxA = fixbox(boxA)
    # boxB = fixbox(boxB)
    
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interW = xB - xA + 1
    interH = yB - yA + 1
    
    # Correction: reject non-overlapping boxes
    if interW <=0 or interH <=0 : 
        return -1.0
    
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

## accepts a single box and a list of boxes...
## returns array of iou values between 'box' and all elements in 'boxes'
def bbox_iou(box, boxes, e=0):  
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(box[0], boxes[0])
    yA = np.maximum(box[1], boxes[1])
    xB = np.minimum(box[2], boxes[2])
    yB = np.minimum(box[3], boxes[3])
    
    interW = xB - xA + e
    interH = yB - yA + e
    
    # Correction: reject non-overlapping boxes
    z = (interW>0) * (interH>0)
    interArea = z * interW * interH
    
    boxAArea = (box[2] - box[0] + e) * (box[3] - box[1] + e)
    boxBArea = (boxes[2] - boxes[0] + e) * (boxes[3] - boxes[1] + e)
    iou = interArea / (boxAArea + boxBArea - interArea) + e*(z-1)
    return iou

def bbox_ious(boxes1, boxes2, e=0):
    M = np.array([bbox_iou(box1.T, boxes2.T, e) for box1 in boxes1])
    return M


def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.1):
    '''
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.

    Parameters
    ----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2]. 
      The number of bboxes, N1 and N2, need not be the same.
    
    Returns
    -------
    (idxs_true, idxs_pred, ious, labels)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    '''
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    MAX_DIST = 1.0
    MIN_IOU = 0.0

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = _bbox_iou(bbox_gt[i,:], bbox_pred[j,:])

    if n_pred > n_true:
        # there are more predictions than ground-truth - add dummy rows
        diff = n_pred - n_true
        iou_matrix = np.concatenate( (iou_matrix, np.full((diff, n_pred), MIN_IOU)), axis=0)

    if n_true > n_pred:
        # more ground-truth than predictions - add dummy columns
        diff = n_true - n_pred
        iou_matrix = np.concatenate( (iou_matrix, np.full((n_true, diff), MIN_IOU)), axis=1)

    # call the Hungarian matching
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    # remove dummy assignments
    sel_pred = idxs_pred<n_pred
    idx_pred_actual = idxs_pred[sel_pred] 
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)
    K1, K2, IOUS = idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid]
    
    ########################################################################
    ious = np.array([box_iou(bbox_gt[k1], bbox_pred[k2]) for k1,k2 in zip(K1, K2)])
    sv = (ious > IOU_THRESH)
    K1, K2, IOUS = K1[sv], K2[sv], ious[sv]
    
    ########################################################################
    return K1, K2, IOUS

## hungarian matching algorithm
def hm(D):
    n1,n2 = D.shape
    eps = 100
    if n2 > n1: # add dummy rows
        D = np.concatenate( (D, np.full((n2-n1, n2), eps)), axis=0)
    if n1 > n2: # add dummy columns
        D = np.concatenate( (D, np.full((n1, n1-n2), eps)), axis=1)
    # call the Hungarian matching
    idx1, idx2 = scipy.optimize.linear_sum_assignment(D)

    # if (not idx1.size) or (not idx2.size):
    #     v = np.array([])
    # else:
    #     v = D[idx1, idx2]

    # remove dummy assignments
    s2 = idx2 < n2
    idx_pred_actual = idx2[s2] 
    idx_gt_actual = idx1[s2]
    ious_actual = D[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual < eps/2)
    label = sel_valid.astype(int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label

def allcorners(box):
    box = box.reshape((-1,2))
    box = np.vstack([box,box])
    box[2:,0] = box[2:,0][::-1]
    return box

def corner4dist(b1, b2):
    b1 = allcorners(b1)
    b2 = allcorners(b2)
    msd = np.sum((b1-b2)**2,1).mean() ## mean square distance
    #######
    # msd = (np.sum((b1-b2)**2,1)**0.5).mean()
    # msd = np.sqrt(msd) ## root mean square distance
    #######
    return msd

def corner4dists(A1, A2):#, norm_type=0): ## 0=standardize, 1=minmax
    # if abs(norm_type)>0:
    #     A1 = normalize(A1.reshape((-1,2)), t=norm_type).reshape((-1,4))
    #     A2 = normalize(A2.reshape((-1,2)), t=norm_type).reshape((-1,4))
    return np.array([corner4dist(a1,a2) for a1,a2 in zip(A1,A2)])

def corner_dist_matrix(A1, A2):#, norm_type=1):
    ## minmax (0,1) normalize
    # if abs(norm_type)>0:
    #     A1 = normalize(A1.reshape((-1,2)), t=norm_type).reshape((-1,4))
    #     A2 = normalize(A2.reshape((-1,2)), t=norm_type).reshape((-1,4))
    n1, n2 = A1.shape[0], A2.shape[0]
    D = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            D[i,j] = corner4dist(A1[i], A2[j])
    return D

def draw2boxes(M1, M2, idx1=None, idx2=None, txt=None, offset=0):
    M1 = M1.reshape((-1,2))
    M2 = M2.reshape((-1,2)) + offset
    
    fig, ax = plt.subplots()
    MM = np.vstack([M1,M2])
    xymin, xymax = MM.min(0), MM.max(0)
    
    ## reverse y coords....
    ymin,ymax = xymin[1],xymax[1]
    M1[:,1] = ymax-M1[:,1]+ymin
    M2[:,1] = ymax-M2[:,1]+ymin
    
    ax.set_xlim([xymin[0]-.1, xymax[0]+.1])
    ax.set_ylim([xymin[1]-.1, xymax[1]+.1])
    for i,x in enumerate(M1.reshape((-1,4))):
        ax.add_patch(Rectangle(x[:2], x[2]-x[0], x[3]-x[1], linewidth=1,edgecolor='b',facecolor='none'))
        j = i if idx1 is None else idx1[i]
        ax.text(*x[:2], j, color='b')
    for i,x in enumerate(M2.reshape((-1,4))):
        ax.add_patch(Rectangle(x[:2], x[2]-x[0], x[3]-x[1], linewidth=1,edgecolor='r',facecolor='none'))
        j = i if idx2 is None else idx2[i]
        ax.text(*x[:2], j, color='r')
    if txt is not None:
        # ax.text(*xymin, txt, color='k')
        plt.title(txt)
    plt.show()
    
################################

p = '/home/david/code/phawk/data/generic/distribution/data/detect/DJI_0001/component/labels/'

lab_files = np.array(get_filenames(p, txt))
idx = np.array([int(f.replace(txt,'').split('_')[-1]) for f in lab_files])
a = idx.argsort()
lab_files, idx = lab_files[a], idx[a]

V = []

f1 = lab_files[0]
x1 = get_labels(p+f1)[:,1:5]
b1 = xywh2xyxy(x1)

X = x1.copy()

for k,f2 in enumerate(lab_files[1:]):
    
    f1 = lab_files[k]
    x1 = get_labels(p+f1)[:,1:5]
    b1 = xywh2xyxy(x1)
    
    f2 = lab_files[k+1]
    x2 = get_labels(p+f2)[:,1:5]
    b2 = xywh2xyxy(x2)
    
    D = corner_dist_matrix(b1,b2)
    i,j,d,q = hm(D)
    V.append(d)
    
    draw2boxes(b1,b2, txt = f'frame {k+1}', offset = 0.001)
    # x1 = x2
