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

INCMAT = {0:[1],
          1:[2,23],
          2:[3,18,19],#[3]
          3:[4],
          4:[5],
          5:[7,22],
          6:[8],
          7:[9,10],
          8:[6,11,12,13,14,20],
          9:[6,11,12,13,14,20],
          10:[6,11,12,13,14,20],
          11:[6,11,12,13,14,20],
          12:[15],
          13:[17],
          14:[3,18,19],#[18],
          15:[3,18,19],#[19],
          16:[6,11,12,13,14,20],
          17:[21],
          18:[7,22],
          19:[2,23],
          20:[9,10],
          21:[3,18,19],#[18],
          }

def get_mask(y1,y2):
    m,n = len(y1),len(y2)
    Q = np.zeros([m,n]).astype(np.bool)
    for i,ya in enumerate(y1):
        for j,yb in enumerate(y2):
            Q[i,j] = yb in INCMAT[ya]
    return ~Q

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_filenames(path, ext=''):
    pattern = os.path.join(path, f'*{ext}')
    return [path_leaf(f) for f in glob(pattern)]

def load_labels(lab_file):
    with open(lab_file, 'r') as f:
        labels = [x.split() for x in f.read().strip().splitlines()]
    labels = np.array(labels, dtype=np.float32).round(4)
    if len(labels)>0 and len(labels.shape)<2:
        labels = labels[None,:]
    return labels

# ## XY1WH2xywh
# def box2yolo(box,W,H):
#     x,y,w,h = box
#     cx = (x+w/2)/W
#     cy = (y+h/2)/H
#     cw = w/W
#     ch = h/H
#     return np.array([cx,cy,cw,ch]).round(8).tolist()

# ## xywh2XY1WH
# def yolo2box(yolo,W,H):
#     x,y,w,h = yolo
#     w = w*W
#     h = h*H
#     x = x*W - w/2
#     y = y*H - h/2
#     return np.array([x,y,w,h]).round().astype(np.int32)

def XY1WH2xywh(box,W,H):
    x,y,w,h = box
    cx = (x+w/2)/W
    cy = (y+h/2)/H
    cw = w/W
    ch = h/H
    return np.array([cx,cy,cw,ch]).round(8).tolist()

def xywh2XY1WH(yolo,W,H):
    x,y,w,h = yolo
    w = w*W
    h = h*H
    x = x*W - w/2
    y = y*H - h/2
    return np.array([x,y,w,h]).round().astype(np.int32)

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

## min/max normalize x,y
def minmax_normalize(M, s=0):
    return  M[s:].min(0), M[s:].ptp(0)
    # return  (M[s:].max(0)-M[s:].min(0))/2, M[s:].ptp(0)

def standardize(M, s=0):
    return M[s:].mean(0), M[s:].std(0)

def normalize(M, s=0, t=2, trans=False):
    if t==0:
        print('NO NORMALIZATION AT ZERO!!!!!!!!')
        return 1/0
    if abs(t)==2:
        mu, sig = standardize(M, s=s)
    else: #abs(t)==1)
        mu, sig = minmax_normalize(M, s=s)
    if t<0: ## retain H/W ratio
        sig = sig.max() + sig*0
        ## center!!!!
        # mu = M[s:].mean(0)
        
    if trans:
        return (M-mu) / sig, (mu, sig)
    return (M-mu) / sig

def norm(M1, M2):
    x1 = normalize(M1.reshape((-1,2)))
    x2 = normalize(M2.reshape((-1,2)))
    s = np.sum((x1-x2)**2,1).mean()
    # s = s / ((len(M1))**Q)
    return s

def rotmat(theta):
    c, s = np.arccos(theta), np.arcsin(theta)
    rot = np.array([[c, -s], [s, c]])
    return rot

def correct(M, s=0, trans=False):
    X = M.copy()
    pca = PCA(n_components=2).fit(X[s:])
    theta = pca.components_[0,0]
    rot = rotmat(theta)
    X, (mu, sig) = normalize(np.matmul(X, rot), s=s, trans=True)
    if trans:
        return X, (mu, sig, theta)
    return X

def apply_transform(X, T):
    if len(T)==3:
        R = rotmat(T[2])
        X = np.matmul(X, R) ## rotation
    X = (X-T[0])/T[1]
    return X

def inverse_transform(X, T):
    X = X*T[1]+T[0]
    if len(T)==3:
        R = rotmat(T[2])
        X = np.matmul(X, R.transpose()) ## opposite rotation
    return X

def apply_transforms(X, T):
    for t in T:
        X = apply_transform(X, t)
    return X

def inverse_transforms(X, T):
    for t in T[::-1]:
        X = inverse_transform(X, t)
    return X

def apply_std(T, X):
    return (X-T[0])/T[1]

def inv_std(T, X):
    return X*T[1]+T[0]

def boxarea(A, norm=True):
    V = abs(A[:,2]-A[:,0])*abs(A[:,3]-A[:,1])
    if norm: V = (V-V.mean())/V.std()
    return V

def boxratio(A, norm=True):
    R = np.arctan(abs(A[:,3]-A[:,1])/abs(A[:,2]-A[:,0]))
    if norm: R = (R - np.arctan(1))
    return R

def boxcenter(A):
    return np.dstack([A[:,2]+A[:,0], A[:,3]+A[:,1]]).squeeze()/2

def sortmat(a, cols=None):
    if cols is None:
        cols = list(range(a.hape[1]))
    a = a[a[:,cols[-1]].argsort()] # First sort doesn't need to be stable.
    for c in cols[:-1][::-1]:
        a = a[a[:,c].argsort(kind='mergesort')]
    return a

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
    boxA = fixbox(boxA)
    boxB = fixbox(boxB)
    
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
    boxA = fixbox(boxA)
    boxB = fixbox(boxB)
    
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
    s2 = idx2<n2
    idx_pred_actual = idx2[s2] 
    idx_gt_actual = idx1[s2]
    ious_actual = D[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual < eps/2)
    label = sel_valid.astype(int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label

def write_labels(L, fn):
    with open(fn, 'w') as f:
        for lab in L:
            class_id, x, y, w, h = lab
            class_id = int(class_id)
            f.write(f'{class_id} {x} {y} {w} {h}\n') 

def np2pil(x):
    q = 255 if im1.max()<1.001 else 1
    return Image.fromarray(np.uint8(x*q))

def np_resize(x, q):
    im = np2pil(x)
    im = im.resize((int(im.width/q), int(im.height/q)), Image.ANTIALIAS)
    return np.array(im)

def np2gray(x):
    im = np2pil(x)
    return np.array(im.convert('L'))

def draw2boxes(M1, M2, idx1=None, idx2=None, txt=None):
    M1 = M1.reshape((-1,2))
    M2 = M2.reshape((-1,2))
    
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
        ax.text(*xymin, txt, color='k')
    plt.show()
    
def draw_img_boxes(img, L, idx=None, color='b'):
    plt.imshow(img, cmap='Greys_r')
    for i,x in enumerate(L):
        plt.gca().add_patch(Rectangle(x[:2], x[2], x[3], linewidth=.5,edgecolor=color,facecolor='none'))
        j = i if idx is None else idx[i]
        plt.gca().text(*x[:2], j, color=color)
    plt.show()
    
def draw_2_img_boxes(img, X1, X2, idx1=None, idx2=None, colors=['m','r']):
    H,W = img.shape[:2]
    idxs = [idx1,idx2]
    plt.imshow(img, cmap='Greys_r')
    for k,X in enumerate([X1,X2]):
        L = np.array([xywh2XY1WH(x,W,H) for x in X])
        idx,color = idxs[k],colors[k]
        for i,x in enumerate(L):
            plt.gca().add_patch(Rectangle(x[:2], x[2], x[3], linewidth=.5, edgecolor=color, facecolor='none'))
            j = i if idx is None else idx[i]
            plt.gca().text(*(x[:2]-[k*20,0]), j, color=color)
    plt.show()
    
## accepts a single box and a list of boxes...
## returns array of iou values between 'box' and all elements in 'boxes'
def bbox_iou(box, boxes):  
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(box[0], boxes[0])
    yA = np.maximum(box[1], boxes[1])
    xB = np.minimum(box[2], boxes[2])
    yB = np.minimum(box[3], boxes[3])
    
    interW = xB - xA
    interH = yB - yA
    
    # Correction: reject non-overlapping boxes
    z = (interW>0) * (interH>0)
    interArea = z * interW * interH
    
    boxAArea = (box[2] - box[0]) * (box[3] - box[1])
    boxBArea = (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

def batch_iou(boxes1, boxes2):
    return np.array([bbox_iou(box1.T, boxes2.T) for box1 in boxes1])

def multi_dist_matrix(A1, A2, norm_type=1):
    V1 = boxarea(A1)
    V2 = boxarea(A2)
    R1 = boxratio(A1)
    R2 = boxratio(A2)
    P1 = np.dstack([V1, R1]).squeeze()
    P2 = np.dstack([V2, R2]).squeeze()
    S1 = normalize(A1.reshape((-1,2)), t=norm_type).reshape((-1,4))
    S2 = normalize(A2.reshape((-1,2)), t=norm_type).reshape((-1,4))
    C1 = boxcenter(S1)
    C2 = boxcenter(S2)
    P1 = np.hstack([P1,C1])
    P2 = np.hstack([P2,C2])
    D = cdist(P1, P2, metric='euclidean')
    return D

def center_dist_matrix(A1, A2, norm_type=1):
    ## minmax (0,1) normalize
    A1 = normalize(A1.reshape((-1,2)), t=norm_type).reshape((-1,4))
    A2 = normalize(A2.reshape((-1,2)), t=norm_type).reshape((-1,4))
    ## get box centers
    C1 = boxcenter(A1)
    C2 = boxcenter(A2)
    ## centers distance matrix
    D = cdist(C1, C2, metric='euclidean')
    return D

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

def corner4dists(A1, A2, norm_type=0): ## 0=standardize, 1=minmax
    if abs(norm_type)>0:
        A1 = normalize(A1.reshape((-1,2)), t=norm_type).reshape((-1,4))
        A2 = normalize(A2.reshape((-1,2)), t=norm_type).reshape((-1,4))
    return np.array([corner4dist(a1,a2) for a1,a2 in zip(A1,A2)])

def corner_dist_matrix(A1, A2, norm_type=1):
    ## minmax (0,1) normalize
    if abs(norm_type)>0:
        A1 = normalize(A1.reshape((-1,2)), t=norm_type).reshape((-1,4))
        A2 = normalize(A2.reshape((-1,2)), t=norm_type).reshape((-1,4))
    n1, n2 = A1.shape[0], A2.shape[0]
    D = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            D[i,j] = corner4dist(A1[i], A2[j])
    return D

def hm_score(A1, A2, dist_func=None, mask=None, conf=None, r=0.33, norm_type=1):
    global D,v,q
    dist_func = corner_dist_matrix if dist_func is None else dist_func
    D = dist_func(A1, A2, norm_type=norm_type)
    if mask is not None:
        v = D.max()*100
        q = mask
        D[mask] = v
    i, j, d, labs = hm(D)
    weights = None
    if conf is not None:
        weights = conf[j]**r
    if len(d)==0:
        return [10000,-1], (i,j)
    ###########################################################################
    ## recalc distances w/normalization after elim worst pair(s)
    A1,A2 = A1[i],A2[j]
    M1 = normalize(A1.reshape((-1,2)), t=norm_type).reshape((-1,4))
    M2 = normalize(A2.reshape((-1,2)), t=norm_type).reshape((-1,4)) 
    ds = [corner4dist(M1[k], M2[k]) for k in range(len(i))]
    ious = [box_iou(M1[k], M2[k]) for k in range(len(i))]
    d4 = np.average(ds, weights=weights)
    iou = np.average(ious, weights=weights)
    ###########################################################################
    return [d4,iou], [i,j]

#################

def _elim_dupes(L, iou_tol=0.9):
    if len(L.shape)<2:
        return L
    if len(L)<2:
        return L
    labs = L[(-L[:,-1]).argsort()]
    X = xywh2xyxy(labs[:,1:5])
    K = batch_iou(X, X)
    np.fill_diagonal(K,0)
    f = K.max(1)>iou_tol
    if f.sum()==0:
        return L
    i = np.where(f)[0]
    j = K.argmax(1)[f]
    idel = [b for a,b in zip(i,j) if a<b]
    f = np.array([i not in idel for i in range(len(f))])
    return labs[f]

def elim_dupes(L, iou_tol=0.9):
    while True:
        L2 = _elim_dupes(L, iou_tol)
        if len(L2)==len(L):
            return L
        L = L2

## boxes in XY1WH format
def box_candidates(box1, box2, wh_thr=3, ar_thr=3, area_thr=0.33, eps=1e-16):  # box1(4,n), box2(4,n)
    w1, h1 = box1[2], box1[3]
    w2, h2 = box2[2], box2[3]
    # aspect ratio
    # ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    ww, hh = np.array([w1, w2]), np.array([h1, h2])
    ar = ww/(hh + eps)
    arr = ar.max(0)/(ar.min(0) + eps) # aspect ratio ratio
    # return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (arr < ar_thr)  # candidates
        
########################################################
if __name__ == "__main__":
    
    jpg, txt = '.jpg', '.txt'
    
    root = '/home/david/code/phawk/data/fpl/thermal/transfer/'
    rgb_path = root + 'rgb/'
    img_path_rgb = rgb_path + 'images/'
    lab_path_rgb = rgb_path + 'labels/'
    
    th_path = root + 'thermal/'
    img_path_th = th_path + 'images/'
    lab_path_th = th_path + 'detect/thermdamage/labels/'
    lab_path_gray = th_path + 'detect/gray/labels/'
    lab_path_th_dst = th_path + 'annotations/'
    
    rgb_labs = get_filenames(lab_path_rgb, txt)
    th_labs_damage = get_filenames(lab_path_th, txt)
    th_labs_gray = get_filenames(lab_path_gray, txt)
    
    DEBUG = False
    
    if DEBUG: random.shuffle(th_labs_gray)
    
    for th_lab_file in th_labs_gray:
        
        ########################################
        # ##### bad....
        # th_lab_file = 'GATOR_108362_Therm_5722.txt'
        # th_lab_file = 'EDGEWATER_101933_Therm_2130.txt'
        # th_lab_file = 'SAN_MATEO_108433_Therm_5510.txt'
        # th_lab_file = 'GOLDEN_GATE_504965_Therm_228.txt'
        # th_lab_file = 'BONITA_SPRINGS_502168_Therm_1236.txt'
        # th_lab_file = 'LABELLE_502463_Therm_11342.txt'
        # th_lab_file = 'ORANGETREE_507364_Therm_742.txt'
        # th_lab_file = 'FAIRMONT_700731_Therm_1444.txt'
        # th_lab_file = 'SAN_MATEO_108433_Therm_296.txt'
        # th_lab_file = 'PLAZA_410164_Therm_150.txt'
        # th_lab_file = 'GOLDEN_GATE_504964_Therm_576.txt'
        # th_lab_file = 'BONITA_SPRINGS_502168_Therm_228.txt'
        # th_lab_file = 'GATOR_108362_Therm_1224.txt'
        # th_lab_file = 'GOLDEN_GATE_504965_Therm_2198.txt' ## out-of-frame!!
        # th_lab_file = ''
        # th_lab_file = ''
        # th_lab_file = ''
        # th_lab_file = ''
        # th_lab_file = ''
        
        # ##### good...
        # th_lab_file = 'MIAMI_SHORES_803439_Therm_6407.txt'
        # th_lab_file = 'SAN_MATEO_108433_Therm_600.txt'
        # th_lab_file = 'SWEAT_409362_Therm_2734.txt'
        # th_lab_file = 'STONEBRIDGE_704761_Therm_1116.txt'
        # th_lab_file = 'OSLO_402936_Therm_357.txt'
        # th_lab_file = 'BABCOCK_204262_Therm_140.txt'
        # th_lab_file = 'HIBISCUS_203541_Therm_1092.txt'
        # th_lab_file = 'BEVERLY_700833_Therm_519.txt'
        # th_lab_file = 'GOLDEN_GATE_504965_Therm_1313.txt'
        # th_lab_file = 'LAWTLEY_300732_Part1_Therm_6064.txt'
        # th_lab_file = 'RAILWAY_800832_Therm_465.txt'
        # th_lab_file = 'SAN_MATEO_108433_Therm_5861.txt'
        # th_lab_file = 'SWEAT_409362_Therm_687.txt'
        # th_lab_file = 'WELLBORN_309331_Therm_5100.txt'
        # th_lab_file = 'STARKE_303162_Part1_Therm_2949.txt'
        # th_lab_file = 'SOUTH_DAYTONA_100934_Therm_1250.txt'
        # th_lab_file = 'ALLIGATOR_503568_Therm_299.txt'
        # th_lab_file = ''
        # th_lab_file = ''
        # th_lab_file = ''
        # th_lab_file = ''
        # th_lab_file = ''
        # th_lab_file = ''
        # th_lab_file = ''
        ########################################
        
        print(th_lab_file)
        th_root = th_lab_file.replace(txt, '')
        rgb_files = [f for f in rgb_labs if f.startswith(th_root + '.')]
        if len(rgb_files)==0:
            continue
        elif len(rgb_files)>1:
            print('MORE THAN ONE MATCH!')
            print(rgb_files)
            continue
        rgb_lab_file = rgb_files[0]
        rgb_img_file = rgb_lab_file.replace(txt, jpg)
        rgb_lab_file = lab_path_rgb + rgb_lab_file
        rgb_img_file = img_path_rgb + rgb_img_file
        labels1 = load_labels(rgb_lab_file)
        labels1 = elim_dupes(labels1)
        
        th_img_file = th_lab_file.replace(txt, jpg)
        th_img_file = img_path_th + th_img_file
        th_lab_file_gray = lab_path_gray + th_lab_file
        th_lab_file_damage = lab_path_th + th_lab_file
        th_lab_file_dst = lab_path_th_dst + th_lab_file
        
        ###########################################################
        ## get gray model labels
        labels2 = load_labels(th_lab_file_gray)
        labels2 = elim_dupes(labels2)
        
        ###########################################################
        ## combine gray and thermdamage labels...
        if os.path.exists(th_lab_file_damage):
            labels2d = load_labels(th_lab_file_damage)
            labels2d = elim_dupes(labels2d)
            Xa = xywh2xyxy(labels2[:,1:5])
            Xb = xywh2xyxy(labels2d[:,1:5])
            
            if DEBUG:
                img = np.array(Image.open(th_img_file))
                draw_2_img_boxes(img, labels2[:,1:5], labels2d[:,1:5])
            
            ## find best match overlaps
            k1, k2, ious = match_bboxes(Xa, Xb, IOU_THRESH=0.66)
            if len(k1)>0:
                labels2[k1] = labels2d[k2].copy()
            f = np.array([i not in k2 for i in range(len(Xb))])
            if f.sum()>0:
                labels2 = np.vstack([labels2, labels2d[f]])
        ###########################################################
        
        if len(labels1)==0: continue
        if len(labels2)==0: continue
    
        y1 = labels1[:,0].astype(np.int32)
        y2 = labels2[:,0].astype(np.int32)
        X1 = labels1[:,1:5]
        X2 = labels2[:,1:5]
        conf = labels2[:,-1].round(4)
        Y2 = np.column_stack([y2, conf]).round(4)
        
        B1 = xywh2xyxy(X1)
        B2 = xywh2xyxy(X2)
        
        ###########################################################
        ## draw original pics with original labels
        if DEBUG:
            
            im1 = np.array(Image.open(rgb_img_file))
            im2 = np.array(Image.open(th_img_file))
            # im1 = cv2.imread(rgb_img_file)
            # im2 = cv2.imread(th_img_file, 0)
            
            H1,W1 = im1.shape[:2]
            H2,W2 = im2.shape[:2]
            
            L1 = np.array([xywh2XY1WH(x,W1,H1) for x in X1])
            L2 = np.array([xywh2XY1WH(x,W2,H2) for x in X2])
            
            draw_img_boxes(im1, L1, color='b')
            draw_img_boxes(im2, L2, color='r')
            L2a = L2.copy()
        
        ###########################################################
        # typ, thresh = 1, 0.015#  0.0025 # 1=minmax
        # typ, thresh = 2, 0.01 # 2=standardize
        
        # PARAMS = [(1, 0.007)]
        PARAMS = [(2, 0.04)]
        PARAMS = [(2, 0.04), (1, 0.007)]
        IOU_THRESH = 0.5
        
        for (typ, thresh) in PARAMS:

            # dist_func, typ = center_dist_matrix, 1
            # dist_func, typd = multi_dist_matrix, 1
            dist_func, typd = corner_dist_matrix, typ
            
            N = 0
            BS, IOU = [],[]
            A1, A2 = B1.copy(), B2.copy()
            y1 = labels1[:,0].astype(np.int32)
            y2 = labels2[:,0].astype(np.int32)
            conf = labels2[:,-1].round(4).copy()
            Y1, Y2 = y1.copy(), np.column_stack([y2, conf]).round(4)
            Q = get_mask(y1,y2)
            
            if DEBUG:
                M1 = normalize(A1.reshape((-1,2)), t=typ).reshape((-1,4))
                M2 = normalize(A2.reshape((-1,2)), t=typ).reshape((-1,4))
                draw2boxes(M1, M2)
            
            best_scores, p = hm_score(A1, A2, mask=Q, conf=conf, norm_type=typd)
            i,j = p
            best_score, best_iou = best_scores
            
            A1,A2 = A1[i],A2[j]
            I,J = i,j
            bs, iou = best_score, best_iou
            BP = None
            if len(I)>1 and (bs<thresh or iou>IOU_THRESH):
                BP = (I,J)
              
            Y1,Y2 = Y1[i],Y2[j]
            y1,y2 = Y1.copy(), Y2[:,0].copy().astype(np.int32)
            conf = Y2[:,1].copy()
            Q = get_mask(y1,y2)
            
            if DEBUG:
                M1 = normalize(A1.reshape((-1,2)), t=typ).reshape((-1,4))
                M2 = normalize(A2.reshape((-1,2)), t=typ).reshape((-1,4))
                BS.append(bs)
                IOU.append(iou)
                txt = '{0}  {1:0.3g} {2:0.3g}'.format(N, bs, iou)
                draw2boxes(M1, M2, I, J, txt=txt)
                print(txt)
                print('{}\n{}'.format('\t'.join([f'{i}({y}) ' for i,y in zip(I, y1)]), '\t'.join([f'{i}({y}) ' for i,y in zip(J, y2)])))
            
            idx1, idx2 = np.arange(A1.shape[0]), np.arange(A2.shape[0])
            while BP is None and A1.shape[0]>2 and A2.shape[0]>2:
                if not DEBUG and BP is not None: break
                best_score, best_iou = 1000, 0
                if DEBUG: print('---------------------------------------------')
                h = np.arange(A1.shape[0])
                for k in range(len(h)):
                    g = np.where(h!=k)[0]
                    # for (i,j) in [(g,g), (g,h), (h,g)]:
                    for (i,j) in [(g,h), (h,g)]:
                        scores, p = hm_score(A1[i], A2[j], mask=Q[i[:,None],j], conf=conf[j], norm_type=typd)
                        score, iou = scores
                        if len(p[0])>0:
                    
                            if score<best_score:
                                # print('{0:0.4f}*******************************************\n\t{1}\n\t{2}'.format(score, I[i], J[j]))
                                best_score, best_iou, bestij, bestp = score, iou, (i.copy(),j.copy()), p
                                # sys.exit()
                                
                if best_score == 1000:
                    break
                A1 = A1[bestij[0]][bestp[0]]
                A2 = A2[bestij[1]][bestp[1]]
                idx1 = idx1[bestij[0]][bestp[0]]
                idx2 = idx2[bestij[1]][bestp[1]]
                bs, iou = best_score, best_iou
                #####
                Y1 = Y1[bestij[0]][bestp[0]]
                Y2 = Y2[bestij[1]][bestp[1]]
                y1,y2 = Y1.copy(), Y2[:,0].copy().astype(np.int32)
                conf = Y2[:,1].copy()
                Q = get_mask(y1,y2)
                #####
                if BP is None and len(idx1)>1 and (bs<thresh or iou>IOU_THRESH):
                    BP = (I[idx1], J[idx2])
                
                if DEBUG:
                    M1 = normalize(A1.reshape((-1,2)), t=typ).reshape((-1,4))
                    M2 = normalize(A2.reshape((-1,2)), t=typ).reshape((-1,4))
                    N+=1
                    BS.append(bs)
                    IOU.append(iou)
                    txt = '{0}  {1:0.3g} {2:0.3g}'.format(N, bs, iou)
                    draw2boxes(M1, M2, I[idx1], J[idx2], txt=txt)
                    print(txt)
                    print('{}\n{}'.format('\t'.join(I[idx1].astype('str')), '\t'.join(J[idx2].astype('str'))))
                    print('{}\n{}'.format('\t'.join([f'{i}({y}) ' for i,y in zip(I[idx1], y1)]), '\t'.join([f'{i}({y}) ' for i,y in zip(J[idx2], y2)])))
            
            if DEBUG:
                BS = np.array(BS)
                IOU = np.array(IOU)
                fig,ax = plt.subplots()
                ax.plot(BS,'b')
                ax.set_ylabel('dist',color='b')
                ax2=ax.twinx()
                ax2.plot(IOU,'m')
                ax2.set_ylabel('iou',color='m')
                plt.show()
                
            ########################################################################
        
            ## DO LABEL TRANSFER!!!
                
            if BP is None:
                print('\tNOTHING\n')
                continue
                
            idx1, idx2 = BP
            _,T1 = normalize(B1[idx1].reshape((-1,2)), t=2, trans=True)
            _,T2 = normalize(B2[idx2].reshape((-1,2)), t=2, trans=True)
            
            y1 = labels1[:,0].astype(np.int32)
            y2 = labels2[:,0].astype(np.int32)
            conf = labels2[:,-1].round(4)
            
            if DEBUG:
                # print('WINNER\n{}\n{}'.format('\t'.join(idx1.astype('str')), '\t'.join(idx2.astype('str'))))
                print('WINNER\n{}\n{}'.format('\t'.join([f'{i}({y}) ' for i,y in zip(idx1, y1[idx1])]), '\t'.join([f'{i}({y}) ' for i,y in zip(idx2, y2[idx2])])))
            
            ## transform ALL rgb labels to thermal space
            Z = B1.reshape((-1,2))
            Z = apply_transform(Z, T1)
            Z = inverse_transform(Z, T2)
            Z = xyxy2xywh(Z.reshape((-1,4)))
            
            ## replace transferred rgb boxes with original thermal boxes IF they were in match set
            Z[idx1] = X2[idx2].copy()
            
            ## expand transferred rgb boxes IF they were NOT in match set
            nidx = np.ones(Z.shape[0]).astype(np.bool)
            nidx[idx1] = False
            Z[nidx, 2:4] = Z[nidx, 2:4] + 0.01
            
            ## rerun IOU matching betweem transferred boxes and original thermal...
            ## - replace with original thermal if IOU>0.4
            Z0 = Z.copy()
            k1, k2, ious = match_bboxes(xywh2xyxy(Z), xywh2xyxy(X2), IOU_THRESH=0.38)
            if len(k1)>0: Z[k1] = X2[k2].copy()
            
            ## clip to image size (delete if mostly out-of-frame!)
            im2 = np.array(Image.open(th_img_file))
            H2,W2 = im2.shape[:2]
            Ba = np.array([xywh2XY1WH(z,W2,H2) for z in Z])
            Z = xyxy2xywh(np.clip(xywh2xyxy(Z, clip=False), [0,0,0,0], [1,1,1,1]), clip=False)
            Bb = np.array([xywh2XY1WH(z,W2,H2) for z in Z])
            
            i = None
            if (abs(Ba-Bb)).sum()>20:
                i = box_candidates(box1=Ba.T, box2=Bb.T) ## filter candidates
            
            ## insert classes
            Z = np.hstack([y1[:,None], Z]).round(6)
            if i is not None: Z = Z[i] ## select in-frame
            
            ## save labels and image file
            if not DEBUG:
                print(f'\t{th_lab_file_dst}\n')
                Z = Z[Z[:,0].argsort()]
                write_labels(Z, th_lab_file_dst)
                # copyfile(therm_file, therm_img_dst.replace(' ',''))
                break
            
            #############################################################################################
            ### DEBUG STUFF.......
            ###
            ## draw RGB labels on BOTH images UNCROPPED
            im1 = np.array(Image.open(rgb_img_file))
            im2 = np.array(Image.open(th_img_file))
            # im1 = cv2.imread(rgb_img_file)
            # im2 = cv2.imread(th_img_file, 0)
            
            H1,W1 = im1.shape[:2]
            H2,W2 = im2.shape[:2]
            
            if i is not None and i.mean()<1:
                print(f'\nremoving {np.where(~i)[0]}')
            
            L0 = Z0.copy()
            if i is not None: L0 = L0[i] ## select in-frame
            L1 = X1.copy()
            L2 = Z[:,1:].copy()
            
            ## xywh2XY1WH
            B1 = np.array([xywh2XY1WH(x,W1,H1) for x in L1])
            B2 = np.array([xywh2XY1WH(x,W2,H2) for x in L2])
            ## before last match_boxes
            B0 = np.array([xywh2XY1WH(x,W2,H2) for x in L0])
            B0 = np.clip(B0, [0,0,0,0], [W2,H2,W2,H2])
            
            draw_img_boxes(im1, B1, color='b')
            draw_img_boxes(im2, L2a, color='m')
            draw_img_boxes(im2, B0, color='r')
            draw_img_boxes(im2, B2, color='r')

            sys.exit()
            # break
        ############
        if DEBUG: sys.exit()
        