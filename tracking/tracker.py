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
import networkx as nx

txt,jpg,JPG = '.txt','.jpg','.JPG'

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

def bbox_iou2(bbox_gt, bbox_pred, IOU_THRESH=0.1):
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = _bbox_iou(bbox_gt[i,:], bbox_pred[j,:])
            
    return iou_matrix

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

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], idx_pred_actual[~sel_valid]

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

def drawboxes(M1, idx1=None, txt=None):
    M1 = M1.reshape((-1,2))
    fig, ax = plt.subplots()
    xymin, xymax = M1.min(0), M1.max(0)
    ## reverse y coords....
    ymin,ymax = xymin[1],xymax[1]
    M1[:,1] = ymax-M1[:,1]+ymin
    
    ax.set_xlim([xymin[0]-.1, xymax[0]+.1])
    ax.set_ylim([xymin[1]-.1, xymax[1]+.1])
    for i,x in enumerate(M1.reshape((-1,4))):
        ax.add_patch(Rectangle(x[:2], x[2]-x[0], x[3]-x[1], linewidth=1,edgecolor='b',facecolor='none'))
        j = i if idx1 is None else idx1[i]
        ax.text(*x[:2]-0.005, j, color='b')
    if txt is not None:
        # ax.text(*xymin, txt, color='k')
        plt.title(txt)
    plt.show()
    
def draw2boxes(M1, M2, idx1=None, idx2=None, txt=None, offset=0, c1='b', c2='r'):
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
        ax.add_patch(Rectangle(x[:2], x[2]-x[0], x[3]-x[1], linewidth=1,edgecolor=c1,facecolor='none'))
        j = i if idx1 is None else idx1[i]
        ax.text(*x[:2], j, color=c1)
    for i,x in enumerate(M2.reshape((-1,4))):
        ax.add_patch(Rectangle(x[:2], x[2]-x[0], x[3]-x[1], linewidth=1,edgecolor=c2,facecolor='none'))
        j = i if idx2 is None else idx2[i]
        ax.text(*x[:2], j, color=c2)
    if txt is not None:
        # ax.text(*xymin, txt, color='k')
        plt.title(txt)
    plt.show()

def distmats(b1,b2):
    D = corner_dist_matrix(b1,b2)
    U = bbox_ious(b1,b2)
    # U2 = bbox_iou2(b1,b2)
    return D,U
    # return U2,U

def distmats(b1,b2):
    D = corner_dist_matrix(b1,b2)
    U = bbox_ious(b1,b2)
    return D,U

def align(b1,b2,iou_min=0.4):
    # D = corner_dist_matrix(b1,b2)
    D,U = distmats(b1,b2)
    i,j,d,_ = hm(D)
    u = U[i,j]
    z = u>iou_min
    i,j = i[z],j[z]
    n1,n2 = len(b1),len(b2)
    # ii,jj = np.arange(len(b1)), np.arange(len(b2))
    # fi, fj = np.in1d(ii,i), np.in1d(jj,j)
    # ni, nj = np.where(~fi)[0], np.where(~fj)[0]
    # return i,j,ni,nj
    # return adict({'i':i,'j':j,'t':[(ii,jj) for ii,jj in zip(i,j)]})
    # M = np.zeros([n1,n2])
    # M[i,j] = 1
    # return adict({'i':i,'j':j,'M':M})
    return adict({'i':i,'j':j})

def getframe(n,ns):
    for i in range(len(ns)):
        if n < ns[i]:
            return i,n
        n = n-ns[i]
    return len(ns)-1,n
    
################################

pth = '/home/david/code/phawk/data/generic/distribution/data/detect/DJI_0001_1/component/labels/'

pth2 = pth.replace('/labels/', '/labels_track/')
mkdirs(pth2)

lab_files = np.array(get_filenames(pth, txt))
idx = np.array([int(f.replace(txt,'').split('_')[-1]) for f in lab_files])
a = idx.argsort()
lab_files, idx = lab_files[a], idx[a]

## sample size, min clique size...
m1,m2 = 10,6
## step size, pool size...
s1 = 3
s2 = m1

# seed = np.random.randint(10000)
seed = 1234
# print(f'seed = {seed}')
np.random.seed(seed)

s = -s1
H,S = {},set()
while True:
    s += s1
    if s+s2 > len(lab_files):
        break
    ###### testing...
    # if s>100: break
    ######
    if s%50<s1:
        print(f'{s}/{len(lab_files)}')
    F = np.random.permutation(np.arange(s,s+s2))[:m1]
    F.sort()
    ns,bs = [],[]
    for f in F:
        fn = lab_files[f]
        x = get_labels(pth+fn)[:,1:5]
        b = xywh2xyxy(x)
        bs.append(b)
        ns.append(len(b))
    ns = np.array(ns)
    n = ns.sum()
    M = np.zeros([n,n])
    
    for a in range(len(bs)):
        for b in range(a+1,len(bs)):
            p = align(bs[a],bs[b])
            x = p.i + ns[:a].sum()
            y = p.j + ns[:b].sum()
            M[x,y] = 1
            M[y,x] = 1
    
    G = nx.from_numpy_matrix(M, create_using=nx.Graph)
    Q = list(nx.clique.find_cliques(G))
    Q = [sorted(q) for q in Q if len(q)>=m2]
    
    for q in Q:
        T = [getframe(qq,ns) for qq in q]
        t1 = T[0]
        t1 = (F[t1[0]],t1[1])
        S.add(t1)
        for t2 in T[1:]:
            t2 = (F[t2[0]],t2[1])
            S.add(t2)
            if t2 not in H:
                H[t2] = set()
            H[t2].add(t1)
            if t1[0]==t2[0]:
                print(f'??????? {t1} --> {t2}')
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
        t1,t2 = S[pi],S[pj]
        if t1[0]==t2[0]:
            print(f'??????? {t1} --> {t2}')
            sys.exit()
        data[pi] = pj

S = tupsort(list(S), reverse=True)
n = len(S)
data = [i for i in range(n)]
Sidx = {t:i for i,t in enumerate(S)}

for k,t1 in enumerate(S):
    if t1 not in H:
        continue
    tt = H[t1]
    if k%1000==0:
        print(f'{k}/{len(H)}')
    for t2 in tt:
        if t1[0]==t2[0]:
            print(f'??????? {t1} --> {t2}')
            sys.exit()
        i,j = Sidx[t1],Sidx[t2]
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
for i,k in enumerate(K):
    s = Z[k]
    L.append(s)
    for t in s:
        H[t] = i
    print(f'{i}\t{s[:3]}')


for i,lf in enumerate(lab_files):
    labs = get_labels(pth+lf)
    labs = [[int(x[0])] + x[1:] for x in labs.tolist()]
    lines = []
    for j in range(len(labs)):
        t = (i,j)
        c = H[t] if t in H else -1
        labs[j][0] = c
        lines.append(' '.join([str(x) for x in labs[j]]))
    lf2 = pth2 + lf
    write_lines(lf2, lines)