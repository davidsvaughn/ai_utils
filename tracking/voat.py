'''
Video Object Alignment and Tracking
'''

import os,sys,ntpath
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import scipy.optimize
import scipy as sci
import cv2
from shutil import copyfile
import networkx as nx
import open3d as o3d
from pathlib import Path
import re
import argparse

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
## voat.py:172: DeprecationWarning: scipy.convolve is deprecated and will be removed in SciPy 2.0.0, 
## use numpy.convolve instead

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

def make_empty_file(fn):
    with open(fn, 'w') as f:
        f.write('')

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

## change directory and extension of filename
def switch(fn, path, ext):
    root = os.path.splitext(path_leaf(fn))[0]
    return os.path.join(path, f'{root}{ext}')

def get_filenames(path, ext=jpg, noext=False):
    pattern = os.path.join(path, f'*{ext}')
    x = np.array([path_leaf(f) for f in glob(pattern)])
    if noext:
        x = np.array([f.replace(ext,'') for f in x])
    return x

def get_labels(fn):
    try:
        with open(fn, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return np.array([])
    labs = np.array( [np.array(line.strip().split(' ')).astype(float).round(6) for line in lines])
    if len(labs)==0:
        return labs#.tolist()
    labs[:,1:] = labs[:,1:].clip(0,1)
    # labs = [[int(x[0])] + x[1:] for x in labs.tolist()]
    return labs

def increment_path(path, exist_ok=False, sep='_', mkdir=True):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

def draw_box(img, x, label=None, color=(255,0,0), tl=1):
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_4)
    if label:
        fontScale = tl/4
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=fontScale, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # c1 = (c1[0]-t_size[0]*yoff, c1[1]+h*xoff)
        # c2 = (c2[0]-t_size[0]*yoff, c2[1]+h*xoff)
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_4)  # filled
        ox, oy = c1[0], c1[1] - 2
        oy = max(12, oy)
        cv2.putText(img, label, (ox, oy), 0, fontScale, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

def draw_boxes(src, dst, labels, classes=None):
    img = cv2.imread(src)
    H,W = img.shape[:2]
    labels = np.array(labels)
    idx = labels[:,0].astype(np.int32)
    boxes = xywh2xyxy(labels[:,1:5], W, H)
    colors = Colors()
    for i,x in zip(idx,boxes):
        label = str(i) if classes is None else classes[i]
        draw_box(img, x, label=label, color=colors(i, True))
    cv2.imwrite(dst, img)
    
## matrix form
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

def quantile(x, q=0.5):
    return sorted(x)[int(q*len(x))]

def trim_top_quantile(x, v=0.95):
    x = np.array(x)
    q = quantile(x, v)
    return x[x<q] 

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def rollavg_convolve_edges(a, n=5):
    'scipy.convolve, edge handling'
    # assert n%2==1
    n = 1 + 2*(n//2)
    return sci.convolve(a, np.ones(n, dtype='float'), 'same')/sci.convolve(np.ones(len(a)), np.ones(n), 'same')


def allcorners(box):
    box = box.reshape((-1,2))
    box = np.vstack([box,box])
    box[2:,0] = box[2:,0][::-1]
    return box

## rms of 4 distances between 4 corner pairs (of 2 boxes)
def corner4dist(b1, b2):
    b1 = allcorners(b1)
    b2 = allcorners(b2)
    msd = np.sum((b1-b2)**2,1).mean() ## mean square distance
    rmsd = np.sqrt(msd) ## root mean square distance
    return rmsd

def corner4dists(A1, A2):
    return np.array([corner4dist(a1,a2) for a1,a2 in zip(A1,A2)])

def corner_dist_matrix(A1, A2):
    n1, n2 = A1.shape[0], A2.shape[0]
    D = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            D[i,j] = corner4dist(A1[i], A2[j])
    return D

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

## matrix of ious
def bbox_ious(boxes1, boxes2, e=0):
    M = np.array([bbox_iou(box1.T, boxes2.T, e) for box1 in boxes1])
    return M

def distmats(b1,b2,t=0):
    U = bbox_ious(b1,b2)
    D = corner_dist_matrix(b1,b2)
    return D,U

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

    # remove dummy assignments
    s2 = idx2 < n2
    idx_pred_actual = idx2[s2] 
    idx_gt_actual = idx1[s2]
    ious_actual = D[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual < eps/2)
    label = sel_valid.astype(int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], idx_pred_actual[~sel_valid]

## apply Homography transform
def apply_transform(T,x):
    dim = x.shape[1]
    if dim==4:
        x = x.reshape((-1,2))[:,::-1]
    x = np.hstack([x,np.ones([x.shape[0],1])])
    x = (x@T.T)[:,:2]
    if dim==4:
        x = x[:,::-1].reshape((-1,4))
    return x

def o3d_icp(P1, P2):
    initial_T = np.identity(4) # Initial transformation for ICP
    # scale_x = np.max(P1[0]) - np.min(P1[0])
    # scale_y = np.max(P1[1]) - np.min(P1[1])
    # distance = max(scale_x, scale_y)
    # The threshold distance used for searching correspondences (closest points between clouds)
    distance = 0.1
    
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(np.hstack([P1, np.zeros([P1.shape[0],1])]))
    pcd2.points = o3d.utility.Vector3dVector(np.hstack([P2, np.zeros([P2.shape[0],1])]))
    
    # Define the type of registration
    reg_type = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    # "False" means rigid transformation, scale = 1
    
    # Define the number of iterations
    iterations = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 100)
    
    # Do the registration
    result = o3d.pipelines.registration.registration_icp(pcd1, pcd2, distance, initial_T, reg_type, iterations)
    return np.vstack([result.transformation[:2, [0,1,3]] , [0,0,1]])

pad = 50
canny_thres = 5000
conf_thres  = 0.3
quant_thres = 0.4
area_thres  = 0.25
    
def find_transform(imf1, imf2, lf1, lf2, verbose=0):
    img1 = cv2.imread(imf1)[:,:,::-1]
    img2 = cv2.imread(imf2)[:,:,::-1]
    H,W = img1.shape[:2]  

    ## get labels
    x1 = get_labels(lf1)
    if len(x1.shape)==2 and len(x1)>2:
        x1 = x1[:,1:]
        x1,conf = x1[:,:-1],x1[:,-1]
        q = min(quant_thres, quantile(conf, conf_thres))
        idx = conf > q
        x1 = x1[idx]
        boxes1 = xywh2xyxy(x1, W, H)
    
        ## constrain image to box area
        B = np.hstack([boxes1[:,:2].min(0), boxes1[:,2:].max(0)])
        B = B + np.array([-pad,-pad,pad,pad])
        B = np.clip(B, [0,0,0,0], [W,H,W,H])
        area = np.diff(B.reshape(2,-1), axis=0).prod()
        if area/(H*W)<area_thres:
            # print('AREA THING'); sys.exit()
            B = [pad,pad,W-pad,H-pad]
    else:
        B = [pad,pad,W-pad,H-pad]
                
    xmin,ymin,xmax,ymax = B
    img1 = img1[ymin:ymax,xmin:xmax,:]
    img2 = img2[ymin:ymax,xmin:xmax,:]
    
    h,w = img1.shape[:2]
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    ## find canny thresholds
    t1,t2 = (-1,-1)
    if t1<0 or t2<0:
        t1,t2 = 0,50
        for j in range(30):
            t1,t2 = t1+25,t2+25
            edges1 = cv2.Canny(image=img1_gray, threshold1=t1, threshold2=t2)
            if np.sum(edges1>0) < canny_thres:
                break
        if verbose>2:
            print(f't1={t1} t2={t2}')
    
    edges1 = cv2.Canny(image=img1_gray, threshold1=t1, threshold2=t2)
    edges2 = cv2.Canny(image=img2_gray, threshold1=t1, threshold2=t2)
    
    if verbose>2:
        plt.imshow(edges1, cmap='gray'); plt.title(1); plt.show()
        plt.imshow(edges2, cmap='gray'); plt.title(2); plt.show()
    
    x1,y1 = np.where(edges1>0)
    x2,y2 = np.where(edges2>0)
    P1 = np.column_stack([x1,y1])
    P2 = np.column_stack([x2,y2])
    
    dim = np.array([H,W])
    P1 = P1/dim
    P2 = P2/dim
    
    try:
        T = o3d_icp(P1, P2)
    except:
        print(f'ERROR2: {imf1} -> {imf2}')
        return np.eye(3)
        
    if verbose<1:
        return T
    
    P4 = np.hstack([P1,np.ones([P1.shape[0],1])])
    P4 = (P4@T.T)[:,:2]
    
    P4 = (P4 * dim).round().astype(np.int32)
    y4,x4 = P4[:,0],P4[:,1]
    x4,y4 = x4.clip(0,w-1), y4.clip(0,h-1)
    edges4 = edges1*0
    edges4[y4,x4] = 255
    if verbose>2:
        plt.imshow(edges4, cmap='gray'); plt.title('T(1)'); plt.show()
        
    return T

def _get_transform(imf1, imf2, lab_path, trans_path, verbose=0):
    lf1 = switch(imf1, lab_path, txt)
    lf2 = switch(imf2, lab_path, txt)
    _, f1 = ntpath.split(imf1)
    _, f2 = ntpath.split(imf2)
    n1 = int(f1.replace(jpg,'').split('_')[-1])
    n2 = int(f2.replace(jpg,'').split('_')[-1])
    t_file = os.path.join(trans_path, f'T_{n1}_{n2}.npy')
    
    if verbose>3:
        print(t_file)
    
    if os.path.exists(t_file):
        T = np.load(t_file)
    else:
        T = find_transform(imf1, imf2, lf1, lf2, verbose=verbose)
        mkdirs(trans_path)
        np.save(t_file, T)
    return T

def get_transform(imf1, imf2, lab_path, trans_path, verbose=0, step=None):
    if step is None:
        return _get_transform(imf1, imf2, lab_path=lab_path, trans_path=trans_path, verbose=verbose)
    n1 = int(imf1.replace(jpg,'').split('_')[-1])
    n2 = int(imf2.replace(jpg,'').split('_')[-1])
    r = list(range(n1,n2,step)) + [n2]
    T = None
    for k,j in enumerate(r[1:]):
        i = r[k]
        if1 = imf1.replace(f'_{n1}.jpg', f'_{i}.jpg')
        if2 = imf2.replace(f'_{n2}.jpg', f'_{j}.jpg')
        t = _get_transform(if1, if2, lab_path=lab_path, trans_path=trans_path, verbose=verbose)
        try:
            T = t if T is None else t @ T
        except:
            pass
    if T is None:
        print(f'ERROR1: {imf1} -> {imf2}')
        return np.eye(3)
    return T

def align_boxes(b1, b2, lf1, lf2, img_path, trans_path, step=1, iou_min=0.5, verbose=0):
    lab_path = ntpath.split(lf1)[0]
    imf1 = switch(lf1, img_path, jpg)
    imf2 = switch(lf2, img_path, jpg)

    T = get_transform(imf1, imf2, 
                      lab_path=lab_path, 
                      trans_path=trans_path, 
                      step=step, 
                      verbose=verbose)
    b1 = apply_transform(T, b1)
    
    ## pairwise distance matrices of aligned boxes
    D,U = distmats(b1, b2)
    
    ## run hungarian matching algorithm to get pairwise matches of boxes
    i,j,d,_ = hm(D)
    
    ## filter pairwise matches
    z1 = d<d.mean() + 2*d.std()
    u = U[i,j]
    z2 = u>(iou_min)
    z = z1+z2
    i,j = i[z],j[z]
    
    return adict({'i':i,'j':j})

def sort_by_frame(files):
    frames = np.array([int(os.path.splitext(f)[0].split('_')[-1]) for f in files])
    a = frames.argsort()
    return files[a], frames[a]

def getframe(n,ns):
    for i in range(len(ns)):
        if n < ns[i]:
            return i,n
        n = n-ns[i]
    return len(ns)-1,n

def tupsort(L, reverse=False):
    return sorted(L, key=lambda tup: tup[0] + tup[1]/1000, reverse=reverse)


def find(data, i):
    if i != data[i]:
        data[i] = find(data, data[i])
    return data[i]


def union(data, i, j, S, verbose=0):
    pi, pj = find(data, i), find(data, j)
    if pi != pj:
        t1, t2 = S[pi], S[pj]
        if t1[0] == t2[0]:
            if verbose>1:
                print(f'ERROR_2: {t1} --> {t2}')
            return 1
        data[pi] = pj
    return 0


def binary_search(func, kwargs, k, v, eps, order=0, depth=0, max_depth=5):
    if v[1]-v[0]<eps or depth>max_depth:
        return kwargs
    kw = kwargs.copy()
    vtype = type(kw[k])
    kw[k] = vtype(np.mean(v))
    ret,_ = func(**kw)
    print(f'{kw}\tret={ret}')
    if ret==0:
        v[abs(0+order)] = kw[k]
    else:
        v[abs(1+order)] = kw[k]
        kwargs[k] = kw[k]
    return binary_search(func, kwargs, k, v, eps, order, depth+1, max_depth)


def dual_binary_search(func, kwargs, k, v, eps, order, depth=0, best_n=0, best_T=None, min_depth=6, max_depth=12):
    ## flip binary state
    q = depth%2
    ## check convergence
    if v[q][1]-v[q][0]<eps[q] or depth>max_depth or (depth>=min_depth and best_n>0):
        return best_T, kwargs
    kw = kwargs.copy()
    vtype = type(kw[k[q]])
    kw[k[q]] = vtype(np.mean(v[q]))
    n,T = func(**kw)
    if kw['verbose']>0:
        print(f'{kw}\t\tn={n}')
    if n==0: ## loops were found --> no solution
        if kw['verbose']>0:
            print('Loops found: no solution.')
        v[q][abs(0+order[q])] = kw[k[q]]
    else: ## solution without loops was found
        if kw['verbose']>0:
            print(f'Solution found: {n} objects tracked.')
        v[q][abs(1+order[q])] = kw[k[q]]
        kwargs[k[q]] = kw[k[q]]
        ## when non-zero, lower score is better (n = # connected/tracked objects)
        best_n = n if best_n==0 else min(n, best_n)
        if n==best_n:
            best_T = T
    return dual_binary_search(func, kwargs, k, v, eps, order, depth+1, best_n, best_T, min_depth, max_depth)


def track_objects(span, conn, 
                  start_frame, 
                  end_frame,
                  frame_path,
                  label_path,
                  trans_path,
                  step=1, 
                  verbose=0):
    
    ## default internal param settings
    loop_max = 0
    # max_offset,min_span = 0.05, 8
    max_offset,min_span = 0.06, 10
    
    label_files = np.array(get_filenames(label_path, txt))
    label_files, frames = sort_by_frame(label_files)
    idx = (start_frame<=frames) & (frames<end_frame)
    label_files = label_files[idx]
    
    H,P,S,s = {},{},set(),-step
    while True:
        s += step
        
        if s+span > len(label_files):
            break
        if s % 100 == 0 and verbose>0:
            print(f'{s}/{len(label_files)}')
    
        F = np.arange(s,s+span)
        ns,bs,fns = [],[],[]
        for f in F:
            fn = label_files[f]
            x = get_labels(label_path + fn)
            if len(x)>0:
                x = x[:,1:5]
                b = xywh2xyxy(x)
            else:
                b = np.empty([0,4])
            bs.append(b)
            ns.append(len(b))
            fns.append(label_path + fn)
            ##
            xy_offset = np.array([0.,0.])
            if len(fns)>1:
                imf1 = switch(fns[-2], frame_path, jpg)
                imf2 = switch(fns[-1], frame_path, jpg)
                T = get_transform(imf1, imf2, 
                                  lab_path=label_path, 
                                  trans_path=trans_path,
                                  verbose=verbose)
                # if verbose>4: sys.exit()
                xy = T[:2,2]
                xy_offset += xy
                offset = np.sum(xy_offset**2)**0.5
                if offset>max_offset and len(ns)>=min_span:
                    if verbose>0: print(f'reducing span to: {len(ns)}')
                    break
            #####
        #####
        m2 = int(len(ns)*conn)
        ns = np.array(ns)
        n = ns.sum()
        M = np.zeros([n,n])
        
        ## align time-adjacent image detections using ICP transform
        for a in range(len(bs)):
            for b in range(a+1,len(bs)):
                
                fa,fb = F[a],F[b]
                ab = (fa,fb)
                if ab in P:
                    p = P[ab]
                else:
                    if ns[a]<2 or ns[b]<2:
                        p = adict({'i':np.array([], dtype=np.int32), 'j':np.array([], dtype=np.int32)})
                    else:
                        p = align_boxes(bs[a], bs[b], 
                                        fns[a], fns[b], 
                                        img_path=frame_path, 
                                        trans_path=trans_path)
                    P[ab] = p
                
                x = p.i + ns[:a].sum()
                y = p.j + ns[:b].sum()
                M[x,y] = 1
                M[y,x] = 1
        
        ## find all large cliques
        G = nx.from_numpy_matrix(M, create_using=nx.Graph)
        Q = list(nx.clique.find_cliques(G))
        Q = [sorted(q) for q in Q if len(q)>=m2]
        
        ## add cliques to tuple graph
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
                if t1[0]==t2[0] and verbose>1:
                    print(f'ERROR_1: {t1} --> {t2}')
    
    ## end while loop
    
    ## use Disjoint-Set (Union-Find) to find connected detections
    S = tupsort(list(S), reverse=True)
    n = len(S)
    data = [i for i in range(n)]
    Sidx = {t: i for i, t in enumerate(S)}
    
    loops = 0
    for k, t1 in enumerate(S):
        if t1 not in H:
            continue
        tt = H[t1]
        if k % 1000==0 and verbose>0:
            print(f'{k}/{len(H)}')
        for t2 in tt:
            if t1[0]==t2[0] and verbose>1:
                print(f'ERROR_3: {t1} --> {t2}')
                sys.exit()
            i, j = Sidx[t1], Sidx[t2]
            loops += union(data, i, j, S, verbose=verbose)
            
    if loops>loop_max: return 0,None
    
    Z = {}
    for i in range(n):
        t1 = S[i]
        j = find(data, i)
        t2 = S[j]
        if t2 not in Z:
            Z[t2] = set()
        Z[t2].add(t1)
    
    ## find earliest detection for each connected set
    V = list(Z.values())
    Z = {}
    for v in V:
        v = tupsort(list(v))
        Z[v[0]] = v
    
    L,H = [],{}
    K = tupsort(list(Z.keys()))
    for i, k in enumerate(K):
        s = Z[k]
        L.append(s)
        for t in s:
            H[t] = i
        if verbose>0:
            print(f'{i}\t{s[:3]}')
    
    T = {}
    for i, lf in enumerate(label_files):
        labs = get_labels(label_path + lf)
        labs = [[int(x[0])] + x[1:] for x in labs.tolist()]
        nc = 0
        for j in range(len(labs)):
            t = (i, j)
            c = H[t] if t in H else -1
            nc += (c < 0)*1
            if c < 0:
                continue
            if c not in T:
                T[c] = {}
            T[c][i] = labs[j][:5]
        if i % 50 == 0 and verbose>1:
            print(nc)
            
    return len(K),T


## interpolate and smooth
def smooth_labels(T, win=11, min_span=60):
    K = np.array(list(T.keys()))
    U,V = {},{}
    for k in K:
        R = T[k]
        F = np.array(list(R.keys()))
        n = F.ptp()+1
        if n < min_span:
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
                U[f] = []
            V[f].append(x[i]) ## label and bounding box
            U[f].append(k) ## tracking ID
    return U,V

def ranked_regions_of_interest(label_path, 
                               start_frame=0,
                               end_frame=-1, 
                               Lmin=200, 
                               Lmax=1000, 
                               verbose=0):

    label_files = np.array(get_filenames(label_path, txt))
    label_files, frames = sort_by_frame(label_files)
    
    ## fill in missing labels...
    u = np.arange(frames.max())
    idx = np.in1d(u, frames)
    u = u[~idx]
    lf_root = os.path.join(label_path, '_'.join(label_files[0].split('_')[:-1]))
    for n in u:
        lf = f'{lf_root}_{n}.txt'
        make_empty_file(lf)
    
    ## trim to start/end frames
    idx = (start_frame<=frames)
    if end_frame>0:
        idx = idx & (frames<end_frame)
    label_files = label_files[idx]
    
    L = []
    for i,lf in enumerate(label_files):
        labs = get_labels(label_path+lf)
        if len(labs.shape)<2:
            labs = labs[:,None]
        L.append(labs[:,0].astype(np.int32))
    n = 1 + max([x.max() for x in L if len(x)>0])
    m = len(L)
    X = np.zeros([m, n])
    for i,x in enumerate(L):
        y,z = np.unique(x, return_counts=True)
        X[i,y] = z
    S = X.sum(1)
    S = rollavg_convolve_edges(S, n=300)
    
    if verbose>1:
        plt.plot(S); plt.show()

    t = int(S.max())-1
    while True:
        if verbose>0: print(t)
        # A,B = np.where(np.diff((S>t)*1)>0)[0], np.where(np.diff((S>t)*1)<0)[0]
        #####
        x = list(np.where(np.diff(S>t))[0])
        if S[0]>t:
            x = [0] + x
        if S[-1]>t:
            x = x + [len(S)-1]
        x = np.array(x)
        A = x[np.arange(0,len(x),2)]
        B = x[np.arange(1,len(x),2)]
        ######
        L = B-A
        if L.max()>Lmax:
            break
        t += -0.5

    idx = L>Lmin
    A,B,L = A[idx],B[idx],L[idx]
    I = np.vstack([A,B])
    # if verbose>0: print(I); print(L)
    AB = np.ravel(np.column_stack((A,B)))
    SS = np.split(S, AB)
    SS = [SS[i] for i in np.arange(1,len(SS),2)]
    ss_max = np.array([s.max() for s in SS])
    idx = ss_max.argsort()[::-1]
    I = I[:,idx] + start_frame
    if verbose>0: print(ss_max[idx]); print(I); print(np.diff(I,axis=0))
    return I.T


def main(frame_path, label_path, 
         save_path=None, 
         class_file=None,
         start_frame=0,
         end_frame=-1, 
         N=5,
         min_depth=2,
         max_depth=10,
         verbose=1, 
         ):
    ###################################################
    global ROI
    ###################################################
    
    root_path = str(Path(frame_path).parent.absolute())
    
    if save_path is None:
        save_path = os.path.join(root_path, 'output')
    save_path = increment_path(save_path)
    
    ## directory for storing homography transforms
    trans_path = os.path.join(root_path, 'trans')
    mkdirs(trans_path)
    
    ## make sure these end with '/'
    frame_path = frame_path.rstrip('/')+'/'
    label_path = label_path.rstrip('/')+'/'
    trans_path = trans_path.rstrip('/')+'/'
    
    # find prelim stable regions
    ROI = ranked_regions_of_interest(label_path, 
                                     start_frame=start_frame,
                                     end_frame=end_frame,
                                     verbose=verbose)
    
    # sys.exit()
    
    ## loop through ROI's, best first
    for R in ROI[:N]:
        start_frame, end_frame = R
        
        ## sort and trim label files by start/stop frames
        label_files = np.array(get_filenames(label_path, txt))
        label_files, frames = sort_by_frame(label_files)
        idx = (start_frame<=frames) & (frames<end_frame)
        label_files = label_files[idx]
        
        ## parameter scan
        span, conn = 30, 0.8
        step = 5
        kw = {'span':span, 'conn':conn, 
              'start_frame': start_frame, 
              'end_frame': end_frame, 
              'step': step,
              'frame_path': frame_path,
              'label_path': label_path,
              'trans_path': trans_path,
              'verbose': verbose,
              }
        
        k = {0:'conn',      1:'span'}
        v = {0:[0.3,0.9],   1:[10,100]}
        eps = {0:0.05,      1:5}
        order = {0:0,       1:-1}
        T, kw = dual_binary_search(track_objects, kw, k, v, eps, order, 
                                   min_depth=min_depth, 
                                   max_depth=max_depth
                                   )
        
        if T is None: ## i.e. no solution found within region
            continue
        
        # smooth and interpolate labels based on tracking ids
        # returns: U=tracking IDs, V=labels and bounding boxes
        U,V = smooth_labels(T)
        
        # pick frames with most detections
        L = { k:len(v) for k,v in V.items() }
        lvals = trim_top_quantile(list(L.values()), 0.98)
        Lmax = max(lvals)
        F = { k:v for k,v in V.items() if len(v)==Lmax }
        
        ## compute total area of all boxes (% of unit square)
        ## TODO: improve using mask approach...
        def area(x):
            a = np.array([np.prod(y[-2:]) for y in x])
            ## so towers don't dominate area computation...
            a = trim_top_quantile(a, 0.95)
            return a.sum()
        
        ## select frame with maximum box area
        A = { k:area(v) for k,v in F.items() }
        Amax = max(A.values())
        G = { k:F[k] for k,v in A.items() if v==Amax }
        idx,labels = list(G.items())[0]
        lab_file = label_files[idx]
        file_root = os.path.splitext(lab_file)[0]
        
        frame = int(file_root.split('_')[-1])
        save_frame_path = os.path.join(save_path, str(frame))
        mkdirs(save_frame_path)
        lab_file = os.path.join(save_frame_path, lab_file)
        
        ## save detections
        labels = np.array(labels)
        lines = [f'{int(x[0])} ' + ' '.join([str(e) for e in x[1:5].round(6)]) for x in labels]
        write_lines(lab_file, lines)
        
        ## copy raw frame
        src_img_file = os.path.join(frame_path, file_root + jpg)
        dst_img_file = os.path.join(save_frame_path, file_root + jpg)
        copyfile(src_img_file, dst_img_file)
        
        ## draw detections on frame
        src = src_img_file
        dst = os.path.join(save_path, file_root + jpg)
        classes = None if class_file is None else read_lines(class_file)
        draw_boxes(src, dst, labels, classes)
        
        
        # save tracking label files
        label_save_path = os.path.join(save_path, 'labels')
        label_save_path = increment_path(label_save_path)
        
        for i, lf in enumerate(label_files):
            lab_file = os.path.join(label_save_path, lf)
            if i not in V:
                make_empty_file(lab_file)
                continue
            lines = [f'{u} ' + ' '.join([str(e) for e in v[1:5].round(6)]) + f' {int(v[0])}' for u,v in zip(U[i],V[i])]
            write_lines(lab_file, lines)
    

if __name__ == '__main__':
    
    # main('/home/david/code/phawk/data/generic/transmission/rgb/master/data/H264/vid1/frames',
    #       '/home/david/code/phawk/data/generic/transmission/rgb/master/data/H264/vid1/labels',
    #       class_file='/home/david/code/phawk/data/generic/transmission/rgb/master/data/exp23/classes.txt',
    #       verbose=2,
    #       min_depth=4,
    #       )
    
    #######################################
    
    # main('/home/david/code/phawk/data/generic/transmission/rgb/master/data/exp23/frames',
    #       '/home/david/code/phawk/data/generic/transmission/rgb/master/data/exp23/labels',
    #       class_file='/home/david/code/phawk/data/generic/transmission/rgb/master/data/exp23/classes.txt',
    #       verbose=2,
    #       min_depth=4,
    #       )
    
    #######################################
    
    # label_path = '/home/david/code/phawk/data/generic/transmission/rgb/master/data/jun6/labels/'
    # label_path = '/home/david/code/phawk/data/generic/transmission/rgb/master/data/jun6/detect/labels/'
    label_path = '/home/david/code/phawk/data/generic/transmission/rgb/master/data/jun6/detect_2/labels/'
    
    frame_path = '/home/david/code/phawk/data/generic/transmission/rgb/master/data/jun6/frames/'
    # start_frame, end_frame = 0,-1
    # start_frame, end_frame = 0, 6000
    # start_frame, end_frame = 8000, 15000
    start_frame, end_frame = 17500, 24000
    verbose = 2
    
    main(frame_path,
          label_path,
          class_file='/home/david/code/phawk/data/generic/transmission/rgb/master/data/jun6/classes.txt',
          start_frame=start_frame,
          end_frame=end_frame,
          verbose=verbose,
          min_depth=4,
          )
    
    sys.exit()
    
    
    ########################################
    
    parser = argparse.ArgumentParser(description='Use video object tracking to pick best frames/detections')
    parser.add_argument('--frames', help='directory containing raw video frames', type=str, required=True)
    parser.add_argument('--labels', help='directory containing yolo detections', type=str, required=True)
    parser.add_argument('--start', help='starting frame', type=int, default=0)
    parser.add_argument('--end', help='ending frame', type=int, default=-1)
    parser.add_argument('--save_path', help='directory to save output', type=str, default=None)
    parser.add_argument('--classes', help='location of classes file', type=str, default=None)
    parser.add_argument('--verbose', help='verbosity level', type=int, default=1)
    parser.add_argument('--N', help='max number of returned frames', type=int, default=5)
    parser.add_argument('--min_depth', help='min depth of parameter binary search', type=int, default=2)
    parser.add_argument('--max_depth', help='max depth of parameter binary search', type=int, default=10)
    args = parser.parse_args()
    
    main(frame_path=args.frames,
         label_path=args.labels, 
         save_path=args.save_path, 
         class_file=args.classes,
         start_frame=args.start,
         end_frame=args.end, 
         verbose=args.verbose, 
         min_depth=args.min_depth,
         max_depth=args.max_depth,
         N=args.N,
         )
