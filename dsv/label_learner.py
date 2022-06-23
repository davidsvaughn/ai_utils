import os,sys
from glob import glob
import matplotlib.pyplot as plt
import ntpath
import numpy as np
from shutil import copyfile
from sklearn import svm
from sklearn.metrics import average_precision_score
import pickle

jpg,txt = '.jpg','.txt'

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_filenames(path, ext=jpg):
    pattern = os.path.join(path, f'*{ext}')
    return np.array([path_leaf(f) for f in glob(pattern)])

def read_lines(fn):
    with open(fn, 'r') as f:
        lines = [n.strip() for n in f.readlines()]
    return [line for line in lines if len(line)>0]

def load_labels(lab_file):
    if not os.path.exists(lab_file):
        return None
    with open(lab_file, 'r') as f:
        labels = [x.split() for x in f.read().strip().splitlines()]
    return np.array(labels, dtype=np.float32).round(4)

def xywh2xyxy(xywh):
    x,y,w,h = xywh[:,0],xywh[:,1],xywh[:,2],xywh[:,3]
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    A = np.dstack([x1,y1,x2,y2]).squeeze().clip(0,1)
    if len(A.shape)==2: return A
    return A[None,:]

## distance to closest edge... X: [n,:4] coords in YOLO FORMAT !!!
def e_dist(X):
    return np.hstack([X[:,:2], 1-X[:,:2]]).min(1)

## max(e_dist) over 4 box corners... X: [n,:4] coords in YOLO FORMAT !!!
def max_corner_dist(X):
    xyxy = xywh2xyxy(X[:,:4])
    d = np.vstack([e_dist(xyxy[:,[i,j]]) for i,j in zip([0,0,2,2],[1,3,1,3])]).max(0)
    return d

def min_corner_dist(X):
    xyxy = xywh2xyxy(X[:,:4])
    d = np.vstack([e_dist(xyxy[:,[i,j]]) for i,j in zip([0,0,2,2],[1,3,1,3])]).min(0)
    return d

tp_path = '/home/david/code/phawk/data/generic/transmission/claire/detect/flashcrack6_train/labels/tp/'
fp_path = '/home/david/code/phawk/data/generic/transmission/claire/detect/flashcrack6_train/labels/fp/'

tp_files = get_filenames(tp_path, txt)
fp_files = get_filenames(fp_path, txt)

P,N = [],[]

# default confidence for TP...
cmu, csig = 0.5, 0.2

## true pos...
for f in tp_files:
    x = load_labels(tp_path+f)
    if x.shape[1]==5:
        c = np.random.normal(cmu, csig, x.shape[0])[:,None]
        x = np.hstack([x, c])
    P.append(x)
P = np.vstack(P)

## false pos...
for f in fp_files:
    x = load_labels(fp_path+f)
    if x.shape[1]==5:
        x = np.hstack([x,np.zeros([x.shape[0],1])])
    N.append(x)
N = np.vstack(N)

n0,n1 = N.shape[0], P.shape[0]
n = n0+n1
X = np.vstack([P,N])
y = np.hstack([np.ones(n1), np.zeros(n0)])

## class (break or flashed)...
na = N[:,0].astype('int')
pa = P[:,0].astype('int')
Xa = X[:,0]
N,P,X = N[:,1:],P[:,1:],X[:,1:]

## confidence...
nc = N[:,-1]
pc = P[:,-1]
Xc = X[:,-1]
# plt.hist(nc, 20)

## plot center points...
px,py = P[:,0],P[:,1]
nx,ny = N[:,0],N[:,1]
# plt.scatter(nx, ny, c='b')
# plt.scatter(px, py, c='y')
# plt.show()

### edge distance function...
box_dist = e_dist       ## center to closest edge (min)
box_dist = max_corner_dist  ## max(e_dist) over 4 corners (max-min)
# box_dist = min_corner_dist  ## min(e_dist) over 4 corners (min-min)

## center distance to edge...
Pd1 = box_dist(P) #np.hstack([P[:,:2], 1-P[:,:2]]).min(1)
Nd1 = box_dist(N) #np.hstack([N[:,:2], 1-N[:,:2]]).min(1)
Xd1 = box_dist(X) #np.hstack([X[:,:2], 1-X[:,:2]]).min(1)
plt.hist(Nd1, 40, color='b')
plt.hist(Pd1, 40, color='y')
plt.title('ALL')
plt.show()


## BREAK : center distance to edge...
i,j = pa==0, na==0
Pd1 = box_dist(P[i]) #np.hstack([P[i,:2], 1-P[i,:2]]).min(1)
Nd1 = box_dist(N[j]) #np.hstack([N[j,:2], 1-N[j,:2]]).min(1)
plt.hist(Nd1, 40, color='b')
plt.hist(Pd1, 40, color='y')
plt.title('BREAK')
plt.show()

## FLASHED : center distance to edge...
i,j = pa==1, na==1
Pd1 = box_dist(P[i]) #np.hstack([P[i,:2], 1-P[i,:2]]).min(1)
Nd1 = box_dist(N[j]) #np.hstack([N[j,:2], 1-N[j,:2]]).min(1)
plt.hist(Pd1, 40, color='y')
plt.hist(Nd1, 40, color='b')
plt.title('FLASHED')
plt.show()

sys.exit()


## plot sigmoid filter....
d = np.linspace(0,.5,1000)
y = 6/(1 + np.exp(-100*(d-0.05)))
plt.scatter(d,y)
plt.show()

sys.exit()

## closest distance to edge...
Pb = xywh2xyxy(P)
Pd2 = np.hstack([Pb[:,:2], 1-Pb[:,2:]]).min(1)
Nb = xywh2xyxy(N)
Nd2 = np.hstack([Nb[:,:2], 1-Nb[:,2:]]).min(1)
Xb = xywh2xyxy(X)
Xd2 = np.hstack([Xb[:,:2], 1-Xb[:,2:]]).min(1)
##
plt.hist(Nd2, 40, color='b')
plt.hist(Pd2, 40, color='r')
plt.show()


sys.exit()

######
seed = np.random.randint(1000000)
seed = 266334

print(f'seed={seed}')
np.random.seed(seed)

split = 0.8
idx = np.random.permutation(n)
i = int(split*n)
idx_train, idx_test = idx[:i], idx[i:]

## compile features
# F = np.column_stack([Xd1, Xc])
F = np.column_stack([Xd1, Xd2])
# F = np.column_stack([Xd1, Xd2, Xc])
# F = np.column_stack([Xd1, Xd2, Xc, Xa])

Xtr, ytr = F[idx_train,:], y[idx_train]
Xte, yte = F[idx_test,:], y[idx_test]

Ctr, Cte = Xc[idx_train], Xc[idx_test]

clf = svm.SVC(C=1)
clf.probability=True
clf.fit(Xtr, ytr)

svm_file = 'svm1.pkl'
with open(svm_file,'wb') as f:
    pickle.dump(clf,f)
clf = None
with open(svm_file, 'rb') as f:
    clf = pickle.load(f)

pte = clf.predict(Xte)
acc = np.mean(pte==yte)
print(f'test acc == {acc:0.4f}')

# proba = clf.predict_proba(Xte)
# pte2 = proba.argmax(1)
# print(np.sum(pte!=pte2))

p = np.array(clf.decision_function(Xte)) # decision is a voting function
k = p.copy()
p = np.column_stack([1-p,1+p])
prob = np.exp(p)/np.sum(np.exp(p),axis=1, keepdims=True) # softmax after the voting
pte3 = prob.argmax(1)
# print(np.sum(pte!=pte3))
prte = prob[:,-1]

# pte = 1 * (prte>0.133)
# acc = np.mean(pte==yte)
# print(f'test acc == {acc:0.4f}')

yp = np.column_stack([yte,prte])
yp = yp[yp[:,1].argsort()[::-1]]

k = prte * Cte

ap = average_precision_score(yte, k)
print(f'test ap == {ap:0.4f}')
