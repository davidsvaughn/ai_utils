import os,sys
import numpy as np
import ntpath
from glob import glob
import matplotlib.pyplot as plt
import scipy.optimize
import cv2
import random

jpg, txt = '.jpg', '.txt'

img_path =      '/home/product/dvaughn/data/fpl/damage/rgb/images/'
datasets_path = '/home/product/dvaughn/data/fpl/damage/rgb/resnet/datasets/'
label_path =    '/home/product/dvaughn/data/fpl/damage/rgb/resnet/data/labels/'

paths = []
paths.append(('Concrete_Pole/0/',0))
paths.append(('Concrete_Pole/1/',1))
paths.append(('Fuse_Switch_Polymer/0/',2))
paths.append(('Fuse_Switch_Polymer/1/',3))
paths.append(('Fuse_Switch_Porcelain/0/',4))
paths.append(('Fuse_Switch_Porcelain/1/',5))
paths.append(('Mushroom_Insulator/0/',6))
paths.append(('Mushroom_Insulator/1/',7))
paths.append(('Porcelain_Dead-end_Insulator/0/',8))
paths.append(('Porcelain_Dead-end_Insulator/1/',9))
paths.append(('Porcelain_Insulator/0/',10))
paths.append(('Porcelain_Insulator/1/',11))
paths.append(('Surge_Arrester/0/',12))
paths.append(('Surge_Arrester/1/',13))
paths.append(('Transformer/0/',14))
paths.append(('Transformer/1/',15))
paths.append(('Wood_Crossarm/0/',16))
paths.append(('Wood_Crossarm/1/',17))
paths.append(('Wood_Pole/0/',18))
paths.append(('Wood_Pole/1/',19))

##############################################################################

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_filenames(path, ext=jpg):
    pattern = os.path.join(path, f'*{ext}')
    return np.array([path_leaf(f) for f in glob(pattern)])

def append_line(fn, s):
    with open(fn, 'a') as f:
        f.write(f'{s}\n')
        
def isint(x):
    try:
        int(x)
    except:
        return False
    return True

def recover_img_file(s):
    toks = s.split('_')
    idx = 1
    while True:
        if idx==len(toks) or not isint(toks[idx]):
            break
        idx += 1
    return '_'.join(toks[idx:])

CUTOFF = 15
NUM = 100000

def best_coords(img, point, cutoff=CUTOFF):
    v = abs(img-point).sum(axis=2)
    return np.argwhere(v<cutoff)

def XY1WH2xywh(x1,y1,w,h,W,H):
    x = x1+w/2
    y = y1+h/2
    return np.array([x/W,y/H,w/W,h/H])

def find_patch(pf, verbose=False):
    # global crop,patch,bestv
    try:
        if verbose: print(pf)
        ####
        img_file = recover_img_file(pf)
        img_file = img_path + img_file
        pf = patch_path+pf
        
        img0 = cv2.imread(img_file)[:,:,::-1]
        patch0 = cv2.imread(pf)[:,:,::-1]
        img = img0.astype(np.int32)
        patch = patch0.astype(np.int32)
        
        H,W = img.shape[:2]
        h,w = patch.shape[:2]
        
        bestlen=10000000
        for idx in range(20):
            i0,j0 = np.random.randint(h),np.random.randint(w)
            pij = patch[i0,j0,:]
            cij = best_coords(img[i0:H-h+i0,j0:W-w+j0,:], pij)
            if verbose: print(len(cij))
            if len(cij)<bestlen:
                bestlen = len(cij)
                bestcij = cij
            if len(bestcij)<NUM:
                break
        cij = bestcij + np.array([[i0,j0]])
        
        if verbose: print('')
        while True:
            coords = []
            for c in cij:
                i,j = c
                ii,jj = np.random.randint(h),np.random.randint(w)
                point = patch[ii,jj,:]
                pnt = img[i+ii-i0,j+jj-j0,:]
                v = abs(pnt-point).sum()
                if v<CUTOFF:
                    coords.append(c)
            if verbose: print(len(coords))
            if len(coords) < 10:
                break
            cij = coords
        
        coords = np.array(coords)
        coords = coords - np.array([[i0,j0]])
        if verbose: print(coords)
        
        bestv = 100000
        for c in coords:
            i,j = c
            crop = img[i:i+h,j:j+w,:]
            v = np.linalg.norm(crop-patch)/np.prod(crop.shape)
            if v<bestv:
                bestv=np.round(v, 6)
                bestc=c
                
        i,j = bestc
        box = XY1WH2xywh(j,i,w,h,W,H)

        if verbose:
            crop0 = img0[i:i+h,j:j+w,:]
            plt.imshow(patch0)
            plt.show()
            plt.imshow(crop0)
            plt.show()
            print('')
            print(bestv)
            print(bestc)
        
        return box, bestv
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except:
        return None,None
    
def find_patch_helper(pf):
    for i in range(5):
        box,v = find_patch(pf)
        if box is None or v is None:
            continue
        if v > 0.05:
            continue
        return box.round(6)
    return None


if __name__ == "__main__":
    img_files = get_filenames(img_path)
    z=0
    for path in paths:
        patch_path, label = path
        patch_path = datasets_path + patch_path
        patch_files = get_filenames(patch_path)
        # random.shuffle(patch_files)
        prob_file = label_path + f'XXX_{label}.txt'
        try:
            for pf in patch_files:
                # pf = '87_2_4ae4dfb1-9548-3bf9-b993-4a468db2cf9f.jpg'
                z+=1

                if z<1260: continue
                
                box = find_patch_helper(pf)
                if box is None:
                    append_line(prob_file, pf)
                    print(f'PROBLEM --> {pf}')
                else:
                    s = f"{label} {' '.join(box.astype('str'))}"
                    txt_file = recover_img_file(pf).replace(jpg, txt)
                    lab_file = label_path + txt_file
                    append_line(lab_file, s)
                    print(f'{z}\t:\t{s} --> {txt_file}')
                # sys.exit()

        except KeyboardInterrupt:
            print('Interrupted!')
            sys.exit()
            