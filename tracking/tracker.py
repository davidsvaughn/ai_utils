import os
import sys
import ntpath
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.optimize
import scipy as sci
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
import networkx as nx
import torch
from sklearn.metrics.pairwise import cosine_similarity

from track_boxes import get_transform, apply_transform

txt, jpg, JPG = '.txt', '.jpg', '.JPG'


class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


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
    return [line for line in lines if len(line) > 0]


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_filenames(path, ext=jpg, noext=False):
    pattern = os.path.join(path, f'*{ext}')
    x = np.array([path_leaf(f) for f in glob(pattern)])
    if noext:
        x = np.array([f.replace(ext, '') for f in x])
    return x


def get_labels(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
    labs = np.array([np.array(line.strip().split(' ')).astype(
        float).round(6) for line in lines])
    if len(labs) == 0:
        return labs  # .tolist()
    labs[:, 1:] = labs[:, 1:].clip(0, 1)
    # labs = [[int(x[0])] + x[1:] for x in labs.tolist()]
    return labs

# matrix form: xywh2xy1xy2


def xywh2xyxy(xywh, clip=True):
    x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    A = np.dstack([x1, y1, x2, y2]).squeeze()
    if clip:
        A = A.clip(0, 1)
    if len(A.shape) == 2:
        return A
    return A[None, :]

# matrix form


def xyxy2xywh(xyxy, clip=True):
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    w = x2-x1
    h = y2-y1
    x = x1+w/2
    y = y1+h/2
    A = np.dstack([x, y, w, h]).squeeze()
    if clip:
        A = A.clip(0, 1)
    if len(A.shape) == 2:
        return A
    return A[None, :]


def minmax_normalize(M):
    return M.min(0), M.ptp(0)
    # return  (M[s:].max(0)-M[s:].min(0))/2, M[s:].ptp(0)


def standardize(M):
    return M.mean(0), M.std(0)

# 1=minmax, 2=standardize


def normalize(A, t=2, e=0.1):
    B = A.copy()
    A = A.reshape(-1, 2)

    if e > 0:
        drawboxes(B)
        xy = (B[:, :2]+B[:, 2:])/2
        idx = np.hstack([xy > e, (1-xy) > e]).all(1)
        B = B[idx]
        drawboxes(B)

    B = B.reshape(-1, 2)
    if t == 0:
        print('NO NORMALIZATION AT ZERO!!!!!!!!')
        return 1/0
    if abs(t) >= 2:
        mu, sig = standardize(B)
        if abs(t) == 3:
            sig = 1
    else:  # abs(t)==1)
        mu, sig = minmax_normalize(B)
    if t < 0:  # retain H/W ratio
        sig = sig.max() + sig*0

    B = (A-mu) / sig
    return B.reshape(-1, 4)


def fixbox(a):
    b = a.copy()
    if b[0] > b[2]:
        b[0], b[2] = b[2], b[0]
    if b[1] > b[3]:
        b[1], b[3] = b[3], b[1]
    return b


def fixboxes(A):
    return np.array([fixbox(a) for a in A])


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


def overlap(boxes1, boxes2):
    box1 = torch.tensor(boxes1)
    box2 = torch.tensor(boxes2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return torch.sum(inter).numpy().item()


def norm_iou(box1, box2):
    x1 = box1 - np.hstack([box1[:2], box1[:2]])
    x2 = box2 - np.hstack([box2[:2], box2[:2]])
    return box_iou(x1, x2)


def norm_ious(boxes1, boxes2):
    n1, n2 = len(boxes1), len(boxes2)
    M = np.zeros([n1, n2])
    for i in range(n1):
        for j in range(n2):
            M[i, j] = norm_iou(boxes1[i], boxes2[j])
    return M


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
    if interW <= 0 or interH <= 0:
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
    if interW <= 0 or interH <= 0:
        return -1.0

    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# accepts a single box and a list of boxes...
# returns array of iou values between 'box' and all elements in 'boxes'


def bbox_iou(box, boxes, e=0):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(box[0], boxes[0])
    yA = np.maximum(box[1], boxes[1])
    xB = np.minimum(box[2], boxes[2])
    yB = np.minimum(box[3], boxes[3])

    interW = xB - xA + e
    interH = yB - yA + e

    # Correction: reject non-overlapping boxes
    z = (interW > 0) * (interH > 0)
    interArea = z * interW * interH

    boxAArea = (box[2] - box[0] + e) * (box[3] - box[1] + e)
    boxBArea = (boxes[2] - boxes[0] + e) * (boxes[3] - boxes[1] + e)
    iou = interArea / (boxAArea + boxBArea - interArea) + e*(z-1)
    return iou


def bbox_ious(boxes1, boxes2, e=0):
    M = np.array([bbox_iou(box1.T, boxes2.T, e) for box1 in boxes1])
    return M


def bbox_iou2(bbox_gt, bbox_pred, IOU_THRESH=0.1):
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = _bbox_iou(bbox_gt[i, :], bbox_pred[j, :])

    return iou_matrix

# hungarian matching algorithm


def hm(D):
    n1, n2 = D.shape
    eps = 100
    if n2 > n1:  # add dummy rows
        D = np.concatenate((D, np.full((n2-n1, n2), eps)), axis=0)
    if n1 > n2:  # add dummy columns
        D = np.concatenate((D, np.full((n1, n1-n2), eps)), axis=1)
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

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], idx_pred_actual[~sel_valid]


def allcorners(box):
    box = box.reshape((-1, 2))
    box = np.vstack([box, box])
    box[2:, 0] = box[2:, 0][::-1]
    return box


def corner4dist(b1, b2):
    b1 = allcorners(b1)
    b2 = allcorners(b2)
    msd = np.sum((b1-b2)**2, 1).mean()  # mean square distance
    #######
    # msd = (np.sum((b1-b2)**2,1)**0.5).mean()
    msd = np.sqrt(msd)  # root mean square distance
    #######
    return msd

# 1=minmax, 2=standardize


def corner4dists(A1, A2):  # , norm_type=0): ## 0=standardize, 1=minmax
    # if abs(norm_type)>0:
    #     A1 = normalize(A1.reshape((-1,2)), t=norm_type).reshape((-1,4))
    #     A2 = normalize(A2.reshape((-1,2)), t=norm_type).reshape((-1,4))
    return np.array([corner4dist(a1, a2) for a1, a2 in zip(A1, A2)])

# 1=minmax, 2=standardize


def corner_dist_matrix(A1, A2):  # , norm_type=1):
    # minmax (0,1) normalize
    # if abs(norm_type)>0:
    #     A1 = normalize(A1.reshape((-1,2)), t=norm_type).reshape((-1,4))
    #     A2 = normalize(A2.reshape((-1,2)), t=norm_type).reshape((-1,4))
    n1, n2 = A1.shape[0], A2.shape[0]
    D = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            D[i, j] = corner4dist(A1[i], A2[j])
    return D


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def rollavg_convolve_edges(a, n=5):
    'scipy.convolve, edge handling'
    # assert n%2==1
    n = 1 + 2*(n//2)
    return sci.convolve(a, np.ones(n, dtype='float'), 'same')/sci.convolve(np.ones(len(a)), np.ones(n), 'same')


def drawboxes(M1, idx1=None, txt=None):
    M1 = M1.copy().reshape((-1, 2))
    fig, ax = plt.subplots()
    xymin, xymax = M1.min(0), M1.max(0)
    # reverse y coords....
    ymin, ymax = xymin[1], xymax[1]
    M1[:, 1] = ymax-M1[:, 1]+ymin

    ax.set_xlim([xymin[0]-.1, xymax[0]+.1])
    ax.set_ylim([xymin[1]-.1, xymax[1]+.1])
    for i, x in enumerate(M1.reshape((-1, 4))):
        ax.add_patch(Rectangle(x[:2], x[2]-x[0], x[3]-x[1],
                     linewidth=1, edgecolor='b', facecolor='none'))
        j = i if idx1 is None else idx1[i]
        ax.text(*x[:2]-0.005, j, color='b')
    if txt is not None:
        # ax.text(*xymin, txt, color='k')
        plt.title(txt)
    plt.show()


def draw2boxes(M1, M2, idx1=None, idx2=None, txt=None, offset=0, c1='b', c2='r'):
    M1 = M1.copy().reshape((-1, 2))
    M2 = M2.copy().reshape((-1, 2)) + offset

    fig, ax = plt.subplots()
    MM = np.vstack([M1, M2])
    xymin, xymax = MM.min(0), MM.max(0)

    # reverse y coords....
    ymin, ymax = xymin[1], xymax[1]
    M1[:, 1] = ymax-M1[:, 1]+ymin
    M2[:, 1] = ymax-M2[:, 1]+ymin

    ax.set_xlim([xymin[0]-.1, xymax[0]+.1])
    ax.set_ylim([xymin[1]-.1, xymax[1]+.1])
    for i, x in enumerate(M1.reshape((-1, 4))):
        ax.add_patch(Rectangle(x[:2], x[2]-x[0], x[3]-x[1],
                     linewidth=1, edgecolor=c1, facecolor='none'))
        j = i if idx1 is None else idx1[i]
        ax.text(*x[:2], j, color=c1)
    for i, x in enumerate(M2.reshape((-1, 4))):
        ax.add_patch(Rectangle(x[:2], x[2]-x[0], x[3]-x[1],
                     linewidth=1, edgecolor=c2, facecolor='none'))
        j = i if idx2 is None else idx2[i]
        ax.text(*x[:2], j, color=c2)
    if txt is not None:
        # ax.text(*xymin, txt, color='k')
        plt.title(txt)
    plt.show()


def distmats(b1, b2, t=0):
    if t > 0:
        b1 = normalize(b1, t=t)
        b2 = normalize(b2, t=t)

    # b1 = fixboxes(b1)
    # b2 = fixboxes(b2)

    U = bbox_ious(b1, b2)
    D = corner_dist_matrix(b1, b2)
    return D, U

# norm_type: 0=NO normalization, 1=minmax, 2=standardize


def align(b1, b2, iou_min=0.4):
    D, U = distmats(b1, b2)
    i, j, d, _ = hm(D)
    u = U[i, j]
    z = u > iou_min
    i, j = i[z], j[z]
    return adict({'i': i, 'j': j})


zmin = 0.5
quint = 0.75


def median(x, q=0.5):
    i = int(q*len(x))
    return sorted(x)[i]

# norm_type: 0=NO normalization, 1=minmax, 2=standardize


def align_twice(a1, a2, iou_min=0.4, norm_type=2):
    # global d,u
    D, U = distmats(a1, a2)  # , t=norm_type)
    i, j, d, _ = hm(D)
    u = U[i, j]
    # if d.ptp()>0.01: sys.exit()

    z = u > (0 if norm_type > 0 else iou_min)
    if z.mean() < zmin:
        z = d < median(d, quint)  # d.mean()# + d.std()

    i, j = i[z], j[z]
    ## run again ?? ############
    if norm_type > 0:
        b1, b2 = a1[i], a2[j]
        D, U = distmats(b1, b2, t=norm_type)
        ii, jj, d, _ = hm(D)
        u = U[ii, jj]
        z = u > iou_min
        if z.mean() < zmin:
            z = d < median(d, quint)  # d < d.mean()# + d.std()
        ii, jj = ii[z], jj[z]
        i, j = i[ii], j[jj]
    ############################
    return adict({'i': i, 'j': j})


def dpairwise(X):
    Xn = np.sum(X**2, axis=1)
    return -2*np.dot(X, X.T) + Xn + Xn[:, None]


def align2(b1, b2, df=1, df_max=0.01, iou_min=0.3):
    global u, v, i, j, dv, z
    N = norm_ious(b1, b2)
    i, j, d, _ = hm(1-N)
    m = median(d, 0.9)
    z = d <= m
    i, j, d = i[z], j[z], d[z]
    c1 = (b1[i, :2]+b1[i, 2:])/2
    c2 = (b2[j, :2]+b2[j, 2:])/2
    v = c1-c2
    ###############################
    dv = np.sum(v**2, axis=1)**0.5
    zmin = max(min(5, len(dv)), 0.33*len(dv))
    for w in range(10):
        z = dv < (df_max*(df+w))
        if z.sum() >= zmin:
            break
    if z.sum() < zmin:
        z = dv < dv.max()
        if z.sum() < 2:
            z = dv < (dv.max()+1)
    i, j, v = i[z], j[z], v[z]
    ###############################
    # s = cosine_similarity(v)
    s = dpairwise(v)
    u = s[np.triu_indices(len(v), k=1)]
    m = median(u, 0.66) if len(u) > 5 else u.max()
    #####
    # g = u[u<=m]
    # m = g.mean()+g.std()
    #####
    M = (s <= m)*1
    G = nx.from_numpy_matrix(M, create_using=nx.Graph)
    Q = list(nx.clique.find_cliques(G))
    lens = [len(q) for q in Q]
    q = np.array(Q[np.argmax(lens)])
    # b2 offset
    h = v[q].mean(0)
    h = np.hstack([h, h])
    # draw2boxes(b1, b2+h, offset=0)
    D, U = distmats(b1, b2+h)
    i, j, d, _ = hm(D)
    #####################
    z = d < d.mean()+2*d.std()
    i, j = i[z], j[z]
    ##### OR ############
    # dh = np.sum(h[:2]**2)**0.5
    # z = d < dh*.75
    # i,j = i[z],j[z]
    ##### OR ############
    # u = U[i,j]
    # z = u>(iou_min)
    # i,j = i[z],j[z]
    #####################
    # ## run again ??
    #####################
    return adict({'i': i, 'j': j})


def align3(b1, b2, lf1, lf2, step=1, iou_min=0.4):

    imf1 = lf1.replace('/labels/', '/frames/').replace(txt, jpg)
    imf2 = lf2.replace('/labels/', '/frames/').replace(txt, jpg)

    T = get_transform(imf1, imf2, step=step)
    b1 = apply_transform(T, b1)

    D, U = distmats(b1, b2)
    i, j, d, _ = hm(D)
    #####################

    z1 = d < d.mean() + 2.5*d.std()
    u = U[i, j]
    z2 = u > (iou_min)
    z = z1+z2

    i, j = i[z], j[z]

    return adict({'i': i, 'j': j})


def getframe(n, ns):
    for i in range(len(ns)):
        if n < ns[i]:
            return i, n
        n = n-ns[i]
    return len(ns)-1, n


################################

# root = '/home/david/code/phawk/data/generic/distribution/data/detect/DJI_0001/component/'
# root = '/home/david/code/phawk/data/generic/distribution/data/detect/DJI_0001_1/component/' # 11,8 *******

# root = '/home/david/code/phawk/data/generic/distribution/data/detect/DJI_0004/component/'

# root = '/home/david/code/phawk/data/generic/distribution/data/detect/DJI_0010/component/'
# root = '/home/david/code/phawk/data/generic/distribution/data/detect/DJI_0010_1/component/' # m1,m2 = 12,7 #********
root = '/home/david/code/phawk/data/generic/transmission/rgb/master/data/exp23/'
# root = '/home/david/code/phawk/data/generic/transmission/rgb/master/data/exp23/toy2/'


lab_path = root + 'labels/'
frame_path = root + 'frames/'

smooth_path = root + 'smooth/'
smooth_lab_path = smooth_path + 'labels/'
mkdirs(smooth_lab_path)

####################################################

lab_files = np.array(get_filenames(lab_path, txt))
idx = np.array([int(f.replace(txt, '').split('_')[-1]) for f in lab_files])
a = idx.argsort()
lab_files, idx = lab_files[a], idx[a]

# iou min
iou_min = 0.3
# smoothing window
win = 11


# step size
step = 1
# max displacement per frame
# df_max = 0.01 #***********
df_max = 0.005

# pool size

# m1,m2 = 10,7 #********DJI_0001_1
# m1,m2 = 11,8 #********DJI_0001_1

# m1,m2 = 12,7 #********DJI_0010_1
# m1,m2 = 12,8

# m1,m2 = 12,8 ## 25
# m1,m2 = 12,6 ## 21
# m1,m2 = 15,8 ## 20
# m1,M2 = 20,0.5 ## 15

m1, M2 = 30, 0.5  # 13

###############################################################
# find prelim stable regions
L = []
for i,lf in enumerate(lab_files):
    labs = get_labels(lab_path+lf)
    if len(labs.shape)<2:
        labs = labs[:,None]
    L.append(labs[:,0].astype(np.int32))
n = 1 + max([x.max() for x in L if len(x)>0])
m = len(L)
X = np.zeros([m, n])
for i,x in enumerate(L):
    y,z = np.unique(x, return_counts=True)
    X[i,y] = z

# S = np.array( [cosine_similarity(X[i].reshape(1,-1), X[i-1].reshape(1,-1)).item() for i in range(1,m)])
D = np.array( [np.dot(X[i].reshape(1,-1), X[i-1].reshape(-1,1)).item() for i in range(1,m)])
S = X.sum(1)

plt.plot(D); plt.show()
Ds = rollavg_convolve_edges(D, n=300)
plt.plot(Ds); plt.show()

plt.plot(S); plt.show()
Ss = rollavg_convolve_edges(S, n=300)
plt.plot(Ss); plt.show()

t = int(Ss.max())-1
while True:
    print(t)
    A,B = np.where(np.diff((Ss>t)*1)>0)[0], np.where(np.diff((Ss>t)*1)<0)[0]
    L = B-A
    if L.max()>1000:
        break
    t += -0.5

idx = L>300
A,B,L = A[idx],B[idx],L[idx]
print(A)
print(B)
print(L)

sys.exit()
#######################################################################


seed = np.random.randint(10000)
# seed = 1234
# print(f'seed = {seed}')
np.random.seed(seed)

max_offset = 0

s = -step
H, P, S = {}, {}, set()
while True:
    s += step
    if s+m1 > len(lab_files):
        break
    # testing...
    # if s>100: break
    ######
    if s % 50 < step:
        print(f'{s}/{len(lab_files)}')

    F = np.arange(s, s+m1)

    ns, bs, fns = [], [], []
    for f in F:
        fn = lab_files[f]
        x = get_labels(lab_path+fn)
        if len(x) > 0:
            x = x[:, 1:5]
            b = xywh2xyxy(x)
        else:
            b = np.empty([0, 4])
        bs.append(b)
        ns.append(len(b))
        fns.append(lab_path + fn)
        ##
        xy_offset = np.array([0., 0.])
        if len(fns) > 1:
            imf1 = fns[-1].replace('/labels/', '/frames/').replace(txt, jpg)
            imf2 = fns[-2].replace('/labels/', '/frames/').replace(txt, jpg)
            T = get_transform(imf1, imf2)
            xy = T[:2, 2]
            xy_offset += xy
            offset = np.sum(xy_offset**2)**0.5
            if offset > 0.06 and len(ns) > 7:
                print(len(ns))
                break
        ##
    m2 = int(len(ns)*M2)
    # offset = np.sum(xy_offset**2)**0.5
    # if offset>max_offset:
    #     max_offset = np.round(offset, 4)
    #     print(f'{s} {max_offset}')

    ns = np.array(ns)
    n = ns.sum()
    M = np.zeros([n, n])

    for a in range(len(bs)):
        for b in range(a+1, len(bs)):

            fa, fb = F[a], F[b]
            ab = (fa, fb)
            if ab in P:
                p = P[ab]
            else:
                if ns[a] < 2 or ns[b] < 2:
                    p = adict({'i': np.array([], dtype=np.int32),
                              'j': np.array([], dtype=np.int32)})
                else:
                    p = align3(bs[a], bs[b], fns[a], fns[b], step=1)
                P[ab] = p

            x = p.i + ns[:a].sum()
            y = p.j + ns[:b].sum()
            M[x, y] = 1
            M[y, x] = 1

    G = nx.from_numpy_matrix(M, create_using=nx.Graph)
    Q = list(nx.clique.find_cliques(G))
    Q = [sorted(q) for q in Q if len(q) >= m2]

    for q in Q:
        T = [getframe(qq, ns) for qq in q]
        t1 = T[0]
        t1 = (F[t1[0]], t1[1])
        S.add(t1)
        for t2 in T[1:]:
            t2 = (F[t2[0]], t2[1])
            S.add(t2)
            if t2 not in H:
                H[t2] = set()
            H[t2].add(t1)
            if t1[0] == t2[0]:
                print(f'1??????? {t1} --> {t2}')
                sys.exit()


def tupsort(L, reverse=False):
    return sorted(L, key=lambda tup: tup[0] + tup[1]/1000, reverse=reverse)


def find(data, i):
    if i != data[i]:
        data[i] = find(data, data[i])
    return data[i]


def union(data, i, j):
    pi, pj = find(data, i), find(data, j)
    if pi != pj:
        t1, t2 = S[pi], S[pj]
        if t1[0] == t2[0]:
            print(f'2??????? {t1} --> {t2}')
            return
        data[pi] = pj


S = tupsort(list(S), reverse=True)
n = len(S)
data = [i for i in range(n)]
Sidx = {t: i for i, t in enumerate(S)}

for k, t1 in enumerate(S):
    if t1 not in H:
        continue
    tt = H[t1]
    if k % 1000 == 0:
        print(f'{k}/{len(H)}')
    for t2 in tt:
        if t1[0] == t2[0]:
            print(f'3??????? {t1} --> {t2}')
            sys.exit()
        i, j = Sidx[t1], Sidx[t2]
        union(data, i, j)

Z = {}
for i in range(n):
    t1 = S[i]
    j = find(data, i)
    t2 = S[j]
    if t2 not in Z:
        Z[t2] = set()
    Z[t2].add(t1)
    # print(f'{t1} --> {t2}')

V = list(Z.values())
Z = {}
for v in V:
    v = tupsort(list(v))
    Z[v[0]] = v

L = []
H = {}
K = tupsort(list(Z.keys()))
for i, k in enumerate(K):
    s = Z[k]
    L.append(s)
    for t in s:
        H[t] = i
    print(f'{i}\t{s[:3]}')


T = {}
for i, lf in enumerate(lab_files):
    labs = get_labels(lab_path+lf)
    labs = [[int(x[0])] + x[1:] for x in labs.tolist()]
    # lines = []
    nc = 0
    for j in range(len(labs)):
        t = (i, j)
        c = H[t] if t in H else -1
        nc += (c < 0)*1
        # labs[j][0] = c
        # lines.append(' '.join([str(x) for x in labs[j]]))
        if c < 0:
            continue
        if c not in T:
            T[c] = {}
        T[c][i] = labs[j][:5]

    if i % 10 == 0:
        print(nc)
    # write "track" label files ???
    # lf2 = pth2 + lf
    # write_lines(lf2, lines)


def make_empty_file(fn):
    with open(fn, 'w') as f:
        f.write('')


## interpolate and smooth
K = np.array(list(T.keys()))
V = {}
for k in K:
    R = T[k]
    F = np.array(list(R.keys()))
    n = F.ptp()+1
    if n < 30:
        continue
    f1 = F[0]
    x = np.empty([n, 5])
    x[:] = np.nan
    for f in F:
        r = R[f]
        x[f-f1] = r[:5]
    # pick most frequent class
    y = x[:, 0]
    nans, z = nan_helper(y)
    c = np.bincount(y[~nans].astype(np.int32)).argmax()
    y[:] = c
    x[:, 0] = y
    # smooth boxes
    for j in range(1, 5):
        y = x[:, j]
        nans, z = nan_helper(y)
        y[nans] = np.interp(z(nans), z(~nans), y[~nans])
        yy = rollavg_convolve_edges(y, n=win)
        x[:, j] = yy
    for i in range(n):
        f = i+f1
        if f not in V:
            V[f] = []
        # labs = f'{k} ' + ' '.join([str(e) for e in x[i].round(6)]) + ' 1'
        labs = f'{int(x[i,0])} ' + ' '.join([str(e) for e in x[i, 1:5].round(6)]) + f' {k}'
        V[f].append(labs)

# save to new label files
for i, lf in enumerate(lab_files):
    lf3 = smooth_lab_path + lf
    if i not in V:
        make_empty_file(lf3)
        continue
    lines = V[i]
    write_lines(lf3, lines)


sys.exit()


# find stable regions
F = np.array(list(V.keys()))
n = F.ptp() + 1
H = np.zeros([n, K.max()+1], np.int32)
for i in range(n):
    if i not in V:
        continue
    h = np.array([int(z.split(' ')[0]) for z in V[i]])
    H[i, h] = 1
