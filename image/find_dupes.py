import os,sys
import numpy as np
import ntpath
from glob import glob
import matplotlib.pyplot as plt
import scipy.optimize
import cv2
import random

jpg, txt = '.jpg', '.txt'

##############################################################################

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
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
    if ',' in s:
        toks = s.split(',')
        toks = toks[0].split('_')
        return '_'.join(toks[:-1]) + '.jpg'
    toks = s.split('_')
    idx = 1
    while True:
        if idx==len(toks) or not isint(toks[idx]):
            break
        idx += 1
    return '_'.join(toks[idx:])

CUTOFF = 15
NUM = 100000

Q1, Q2, Q3 = 1.5, 3, 0.025
# Q2 = 1 ## Faster!

def best_coords(img, point, cutoff=CUTOFF):
    v = abs(img-point).sum(axis=2)
    return np.argwhere(v<cutoff)

def XY1WH2xywh(x1,y1,w,h,W,H):
    x = x1+w/2
    y = y1+h/2
    return np.array([x/W,y/H,w/W,h/H])

def find_patch(fn1, fn2, verbose=False):
    global img0,patch0
    try:
        if verbose: print(f'\n{fn1}\n{fn2}')
        ####
        
        ## reinforce file size similarity
        sz1 = os.path.getsize(fn1)
        sz2 = os.path.getsize(fn2)
        if abs(sz1-sz2)>2000:
            return None,None
        
        img0 = cv2.imread(fn1)[:,:,::-1]
        patch0 = cv2.imread(fn2)[:,:,::-1]
        
        ## reinforce image shape similarity
        if sum(abs(np.array(img0.shape)-np.array(patch0.shape)))>100:
            return None,None
        
        ######
        q = Q1
        h,w = patch0.shape[:2]
        hh,ww = int(h/q), int(w/q)
        i,j = np.random.randint(h-hh-1),np.random.randint(w-ww-1)
        patch0 = patch0[i:i+hh, j:j+ww, :]
        ######
        
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

        if bestv is not None and bestv<=Q3 and verbose: 
            crop0 = img0[i:i+h,j:j+w,:]
            plt.imshow(patch0)
            plt.show()
            plt.imshow(crop0)
            plt.show()
            print('')
            print(bestv)
            # print(bestc)
        
        return box, bestv
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except:
        return None,None
    
def find_patch_helper(fn1, fn2, verbose=False):
    for i in range(Q2):
        box,v = find_patch(fn1, fn2, verbose=verbose)
        if box is None or v is None:
            if verbose: print('miss1')
            continue
        if v > Q3:
            if verbose: print(f'miss2 : {v}')
            continue
        if verbose: print('HIT')
        #######
        # print(v)
        #######
        return box.round(6)
    if verbose: print('FAIL')
    return None

# def elim_dupes_OLD(p1, p2=None, verbose=True):
#     if p2 is None:
#         p2 = p1
    
#     print(f'{p1}\n{p2}')
    
#     files1 = np.array(get_filenames(p1))
#     files2 = np.array(get_filenames(p2))
    
#     roots1 = np.array([recover_img_file(f) for f in files1])
#     roots2 = np.array([recover_img_file(f) for f in files2])
    
#     n = 0
#     try:
#         for i,f1 in enumerate(files1):
#             idx = (roots2==roots1[i])
#             if p1==p2:
#                 idx[i] = False
#             for f2 in files2[idx]:
#                 fn1 = p1+f1
#                 fn2 = p2+f2
#                 box = find_patch_helper(fn1, fn2, verbose=verbose)
#                 if box is not None:
#                     n+=1
#                     if verbose:
#                         print(f'MATCH:\n\t{fn1}\n\t{fn2}\n')
#                     ## delete file!?!?!?!?
#                     # os.remove(fn2)
                    
#         print(f'\t{n} MATCHES\n')
                    
#     except KeyboardInterrupt:
#         print('Interrupted!')
#         sys.exit()
        
def elim_dupes(p1, p2=None, verbose=True):
    # global files1, files2
    ####
    same=False
    if p2 is None:
        p2 = p1
        same=True
        
    files1 = np.array(get_filenames(p1))
    files2 = np.array(get_filenames(p2))
    files1.sort()
    files2.sort()
    n = 0
    try:
        for i,f1 in enumerate(files1):
            
            # print(f'{i}/{len(files1)}')
            if i%10==0: print(f'{i}/{len(files1)}')
            
            j = i+1 if same else 0
            for f2 in files2[j:]:
                fn1 = p1+f1
                fn2 = p2+f2
                # print(fn1)
                # print(fn2)
                box = find_patch_helper(fn1, fn2, verbose=False)
                if box is not None:
                    n+=1
                    if verbose:
                        print(f'MATCH:\n\t{fn1}\n\t{fn2}\n\tDELETING {fn2}')
                    ## delete 2nd file
                    os.remove(fn2)
                    
                    ## DELETE LABEL ???
                    # lab = fn2.replace('/images/','/labels/').replace(jpg, txt)
                    # os.remove(lab)
                    # print(f'DELETING:\n\t{fn2}\n\t{lab}\n')
                    
        print(f'\t{n} MATCHES\n')
                    
    except KeyboardInterrupt:
        print('Interrupted!')
        sys.exit()

def elim_dupes_helper(p1, p2):
    if os.path.isdir(p1): 
        elim_dupes(p1)
    if os.path.isdir(p2):
        elim_dupes(p2)
    if os.path.isdir(p1) and os.path.isdir(p2):
        elim_dupes(p1, p2)
    
###############################################################################


# path = '/home/david/code/phawk/data/generic/transmission/damage/insulator_damage/images/'

p1 = '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/tags/Deteriorated/1/'
p2 = '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/tags/Deteriorated/0/'

if __name__ == "__main__":
    
    # elim_dupes(insulator_path)
    # fn1 = insulator_path + 'a1.jpg'
    # fn2 = insulator_path + 'a2.jpg'
    # box = find_patch_helper(fn1, fn2, verbose=True)
    
    elim_dupes(p1, p2, verbose=True)
    

        
    