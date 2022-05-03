import os, sys
import random
from glob import glob
import ntpath
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import lr_scheduler
import cv2
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import average_precision_score, precision_recall_curve
import copy
import gc
import pathlib
import time, datetime
from shutil import copyfile
# from functools import lru_cache

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print('NO CUDA?!')
    sys.exit()

## example with tee:
##      python -u train.py | tee log.txt   
###############################################################################
## TRAINING PARAMETERS

## K-fold CV 
K = 4
## do all K rounds (True), or just a single round (False)
cv_complete = True

## random seed
SEED = np.random.randint(0, 1000000)
# SEED = 594194

FPFN = True
PLOT = True
SAVE = True

MINDIM = 0
RAND_SCALE = None

# MINDIM = 75
Q = 0       ## sideline ALL fp/fn
# Q = 0.5     ## sideline MOST extreme Q
# Q = -0.5    ## sideline LEAST extreme Q

## defaults ##
epochs = 5
freeze = 7
pad = 10
fc = 0
rot = 0
res = 34 # resnet version
print_every = 64 ## num samples

batch_size = 16

## scaling
scale = None
## 1D scaling...    scale=N ==>  MAX dimension <= N
scale = 1024
# scale = -128

## fully connected (dense) layer size
fc = 512

## dropout [conv, dense]
drops = [0.66, 0.66]
# drops = [0.25, 0.25]

TEST_FILE = None
FPFN_TEST = None
###############################################################################
## DATA PARAMETERS

## old.######
# DATA_ROOT = '/home/product/dvaughn/data/fpl/damage/rgb/resnet/datasets/'
# DATA_ROOT = '/home/product/dvaughn/data/fpl/damage/rgb/resnet/data/old/patches2/'
# DATA_ROOT = '/home/product/dvaughn/data/fpl/damage/rgb/resnet/datasets2/'
#############

ROOT = '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/'
DATA_ROOT   = ROOT + 'tags/'
MODEL_ROOT  = ROOT + 'models/'

###############################################################################
RAND_SCALE = 0.5

# ITEM, scale, fc, drops, batch_size, print_every, res, SEED  = 'POLY_PORC', 256, 256, [0.8,0.8], 32, 512, 34, None
# ## ITEM, scale, fc, drops, batch_size, print_every, SEED  = 'Concrete_Pole', -128, 512, [0.1,0.25], 16, 128, None
# ## ITEM, scale, fc, drops, batch_size, print_every, SEED  = 'Concrete_Pole', -128, 256, [0.5,0.25], 16, 128, 547663
# ## ITEM, scale, fc, drops, batch_size, print_every, SEED  = 'Concrete_Pole', -256, 256, [0.2,0.5], 16, 128, 547663
# ITEM, scale, fc, drops, batch_size, print_every, res, SEED  = 'Concrete_Pole', 1536, 256, [0.5,0.25], 16, 128, 50, None
# ITEM, scale, fc, drops, batch_size, print_every, res, SEED  = 'Fuse_Switch_Polymer', 256, 256, [0.9,0.8], 16, 256, 34, None
# ITEM, scale, fc, drops, batch_size, print_every, res, SEED  = 'Fuse_Switch_Porcelain', 256, 256, [0.8,0.8], 32, 1024, 50, None
# ##ITEM, scale, fc, drops, batch_size, print_every, res, SEED  = 'Mushroom_Insulator', 256, 0, [0.66,0.66], 8, 256, 50, None
# ITEM, scale, fc, drops, batch_size, print_every, res, SEED  = 'Mushroom_Insulator', 256, 256, [0.66,0.66], 16, 256, 50, 472974
# ITEM, scale, fc, drops, batch_size, print_every, SEED  = 'Porcelain_Dead-end_Insulator', 256, 512, [0.66,0.66], 16, 128, None
# ITEM, scale, fc, drops, batch_size, print_every, res, SEED  = 'Porcelain_Insulator', 256, 256, [0.66,0.66], 16, 256, 50, None
# ITEM, scale, fc, drops, batch_size, print_every, res, SEED  = 'Surge_Arrester', 256, 256, [0.66,0.66], 16, 512, 50, None
# ITEM, scale, fc, drops, batch_size, print_every, SEED  = 'Transformer', 256, 64, [0.8,0.8], 32, 512, None
# ITEM, scale, fc, drops, batch_size, print_every, res, SEED  = 'Wood_Crossarm', 512, 256, [0.33,0.33], 8, 256, 50, 876167
# ITEM, scale, fc, drops, batch_size, print_every, res, SEED  = 'Wood_Pole', 1536, 256, [0.25,0.25], 16, 512, 50, 14699

####################
cv_complete = True
K, epochs, alpha = 5, 10, 0.5


## Deteriorated
K, epochs, alpha = 5, 16, 0.5
#ITEM, scale, fc, drops, batch_size, print_every, SEED  = 'Deteriorated', 1536, 64, [0.5,0.5], 32, 128, None
ITEM, scale, fc, drops, batch_size, print_every, rot, SEED  = 'Deteriorated', 1024, 256, [0.66,0.33], 32, 1024, 0.25, 223665
res = 50

## Vegetation
# K, epochs, alpha = 5, 8, 0.25
# ITEM, scale, fc, drops, batch_size, print_every, rot, SEED  = 'Vegetation', 1024, 256, [0.5,0.5], 64, 128, 0.25, None


################

MODEL_PATH = MODEL_ROOT + f'{ITEM}/'
LAST_TEST = MODEL_PATH + 'last_test.txt'
# TEST_FILE = '/home/product/dvaughn/data/fpl/damage/rgb/resnet/data/test.txt'

# TEST_FILE = LAST_TEST
# FPFN_TEST = TEST_FILE

if SEED is None:
    SEED = np.random.randint(0, 1000000)

if TEST_FILE is not None:
    cv_complete = False

###############################################################################

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_files(path, ext=''):
    return np.array([path_leaf(f) for f in glob('{}/*{}'.format(path, ext))])

def read_lines(fn):
    with open(fn, 'r') as f:
        lines = [n.strip() for n in f.readlines()]
    return [line for line in lines if len(line)>0]

def write_lines(lines, fn):
    with open(fn, 'w') as f:
        for line in lines:
            f.write("%s\n" % line)

def move_files(files, src_path, dst_path):
    n = 0
    for fn in files:
        src = os.path.join(src_path, fn)
        if not os.path.exists(src):
            print(f'FILE DOES NOT EXIST: {src}')
            continue
        dst = os.path.join(dst_path, fn)
        os.rename(src, dst)
        n+=1
    return n

def copy_files(files, dst_path):
    for src in files:
        if not os.path.exists(src):
            print(f'FILE DOES NOT EXIST: {src}')
            continue
        name = path_leaf(src)
        dst = os.path.join(dst_path, name)
        copyfile(src, dst)
        
def scale_img_1d(img, scale):
    if scale<0:
        scale = -scale
        q = scale/min(img.shape[1:])
    else:
        q = scale/max(img.shape[1:])
    if q<1:
    # if q!=1:
        h,w = int(img.shape[1]*q), int(img.shape[2]*q) 
        return F.interpolate(img.unsqueeze(0), size=[h,w]).squeeze()
    return img

def scale_img_2d(img, scale):
    area = img.shape[1] * img.shape[2]
    q = np.sqrt(scale/area)
    if q<1:
    # if q!=1:
        h,w = int(img.shape[1]*q), int(img.shape[2]*q) 
        return F.interpolate(img.unsqueeze(0), size=[h,w]).squeeze()
    return img

# @lru_cache(maxsize=8192)
def scale_img(img, scale=scale, rand=None):
    if rand is not None:
        s2 = int(rand * scale)
        scale = random.randint(min(s2,scale),max(s2,scale))
    if scale>10000:
        return scale_img_2d(img, scale)
    return scale_img_1d(img, scale)
    
def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    global im
    im = img
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

## used for TEST SET
def simple_collate(batch):
    # data = [torch.FloatTensor(np.array(item[0]).transpose([2,0,1]) * (1/255)) for item in batch]
    data = [item[0] for item in batch]
    labels = torch.LongTensor([item[1] for item in batch])
    paths = [item[2] for item in batch]
    
    ## resize...?
    if scale is not None:
        data = [scale_img(img) for img in data]
    
    ## pad test data too...?
    if pad>0:
        d = F.pad(input=data[0], pad=(pad, pad, pad, pad), mode='constant', value=0)
        data = [d]
    
    data = torch.cat(data, dim=0)
    if len(data.shape)==3: data = torch.unsqueeze(data, 0)
    return [data, labels, paths]

def random_rot90(x, p):
    if random.random()<p:
        k = random.randint(0,1)*2-1
        x = torch.rot90(x, k, [1,2])
    return x
    
## used for TRAIN SET...
def my_collate(batch):
    # global data
    # data = [torch.FloatTensor(np.array(item[0]).transpose([2,0,1]) * (1/255)) for item in batch]
    data = [item[0] for item in batch]
    labels = torch.LongTensor([item[1] for item in batch])
    paths = [item[2] for item in batch]
    
    ## resize...?
    if scale is not None:
        data = [scale_img(img, rand=RAND_SCALE) for img in data]
     
    ## randomly rotate +-90 deg...?
    if rot>0:
        data = [random_rot90(img, rot) for img in data]
    
    ## pad images
    szs = np.array([d.shape[1:] for d in data])
    H,W = szs.max(0)
    X = []
    for i,d in enumerate(data):
        h,w = d.shape[1:]
        h1,w1 = (H-h)//2, (W-w)//2
        h2,w2 = H-h-h1, W-w-w1
        x = F.pad(input=d, pad=(w1, w2, h1, h2), mode='constant', value=0)
        X.append(x.unsqueeze(0))

    data = torch.cat(X, dim=0)
    return [data, labels, paths]

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
    
def load_split_train_test(datadir, k=5, test_fold=0, test_file=None):
    # global train_data
    global SEED
    train_transforms = transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        # transforms.RandomRotation(degrees=(90, -90)),
                                        # transforms.RandomResizedCrop(512, scale=(0.1, 0.5)),
                                        transforms.ToTensor(),
                                        # transforms.ColorJitter(*cj),
                                        # transforms.Normalize(mean=[0.436, 0.45 , 0.413], std=[0.212, 0.208, 0.221]),
                                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                       ])

    test_transforms = transforms.Compose([
                                        transforms.ToTensor(),
                                       # transforms.ColorJitter(*cj),
                                        # transforms.Normalize(mean=[0.436, 0.45 , 0.413], std=[0.212, 0.208, 0.221]),
                                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                      ])

    train_data = ImageFolderWithPaths(datadir, transform=train_transforms)
    test_data = ImageFolderWithPaths(datadir, transform=test_transforms)

    ## train/test split
    if test_file is None:
        num_train = len(train_data)
        idx = np.arange(num_train)
        np.random.seed(SEED)
        np.random.shuffle(idx)
        n = int(np.ceil(num_train/k))
        test_idx = idx[n*test_fold:n*(test_fold+1)]
        train_idx = np.setdiff1d(idx, test_idx)
    else:
        test_imgs = [s.replace('.jpg','') for s in read_lines(test_file)]
        files,labs = zip(*train_data.imgs)
        files = np.array(files)
        labs = np.array(labs)
        idx0 = np.where(labs==0)[0]
        idx1 = np.where(labs==1)[0]
        files0 = files[idx0]
        files1 = files[idx1]
        idx1_test = np.array([np.any([s in fn for s in test_imgs]) for fn in files1])
        idx1_train = ~idx1_test
        n = int(idx1_test.mean()*len(idx0))
        np.random.seed(SEED)
        np.random.shuffle(idx0)
        idx0_test = idx0[:n]
        idx0_train = idx0[n:]
        idx1_test = idx1[idx1_test]
        idx1_train = idx1[idx1_train]
        test_idx = np.hstack([idx0_test, idx1_test])
        train_idx = np.hstack([idx0_train, idx1_train])
        test_idx.sort()
        train_idx.sort()
    
    ## samplers
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data, 
                                              sampler=train_sampler,
                                              batch_size=batch_size,
                                              collate_fn = simple_collate if batch_size==1 else my_collate#crop_collate,#my_collate,
                                              )
    testloader = torch.utils.data.DataLoader(test_data, 
                                             sampler=test_sampler,
                                             collate_fn = simple_collate,
                                             batch_size=1)
    return trainloader, testloader

###########################################################

## set paths
DATA_PATH = DATA_ROOT + ITEM
PATH_0 = f'{DATA_PATH}/0/'
PATH_1 = f'{DATA_PATH}/1/'
tmp_dir = 'fpfn'
PATH_FP = f'{DATA_ROOT}{tmp_dir}/{ITEM}/fp/'
PATH_FN = f'{DATA_ROOT}{tmp_dir}/{ITEM}/fn/'
PATH_DEL = f'{DATA_ROOT}{tmp_dir}/{ITEM}/del/'
FILE_FP = f'{DATA_ROOT}{tmp_dir}/{ITEM}/fp.txt'
FILE_FN = f'{DATA_ROOT}{tmp_dir}/{ITEM}/fn.txt'
make_dirs(PATH_FP)
make_dirs(PATH_FN)
make_dirs(MODEL_PATH)

## restore any sidelined data (from last run... see below)
if FPFN:
    ## 0->1
    if os.path.exists(FILE_FP):
        lines = np.array(read_lines(FILE_FP))
        lines = np.unique(lines) ## in case of repeats...
        files = get_files(PATH_FP, '.jpg')
        idx = ~np.in1d(lines, files)
        lines = lines[idx]
        n01 = move_files(lines, PATH_0, PATH_1)
        ## clear
        os.remove(FILE_FP)
        for f in files:
            os.remove(os.path.join(PATH_FP, f))
        if n01>0:
            print(f'\nMOVED {n01} FALSE POSITIVES from:\n\t{PATH_0} to {PATH_1}')
    
    ## 1->0
    if os.path.exists(FILE_FN):
        lines = np.array(read_lines(FILE_FN))
        lines = np.unique(lines) ## in case of repeats...
        files = get_files(PATH_FN, '.jpg')
        idx = ~np.in1d(lines, files)
        lines = lines[idx]
        n10 = move_files(lines, PATH_1, PATH_0)
        ## clear
        os.remove(FILE_FN)
        for f in files:
            os.remove(os.path.join(PATH_FN, f))
        if n10>0:
            print(f'MOVED {n10} FALSE NEGATIVES from:\n\t{PATH_1} to {PATH_0}\n')
    
    ## remove files...
    del_files = get_files(PATH_DEL, '.jpg')
    ## fp
    files = get_files(PATH_0, '.jpg')
    idx = np.in1d(files, del_files)
    files = files[idx]
    for f in files:
        os.remove(os.path.join(PATH_0, f))
    ## fn
    files = get_files(PATH_1, '.jpg')
    idx = np.in1d(files, del_files)
    files = files[idx]
    for f in files:
        os.remove(os.path.join(PATH_1, f))

## set random seed
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

print(f'SEED = {SEED}')
print(f'batch_size = {batch_size}')
print(f'scale = {scale}')
print(f'fc = {fc}')
print(f'drops = {drops}')
print(f'res = {res}\n')

## clear CUDA cache...
gc.collect()
torch.cuda.empty_cache()

###########################################################
## K-fold CV

PS,RS = np.zeros([K,1000]),np.zeros([K,1000])
FNs,FPs,Ts,Ys,Ss = [],[],[],[],[]
for test_fold in range(K):
    
    ## data loaders
    trainloader, testloader = load_split_train_test(DATA_PATH, K, test_fold, TEST_FILE)
    if test_fold==0:
        print(f'CLASSES: {trainloader.dataset.classes}\n')
    
    ## load RESNET
    if res==50:
        model, N = models.resnet50(pretrained=True), freeze
    elif res==18:
        model, N = models.resnet18(pretrained=True), freeze
    else: # 34
        model, N = models.resnet34(pretrained=True), freeze
    
    ## freeze layers 1-N in the total 10 layers of Resnet
    ct = 0
    drop1, drop2 = drops
    if test_fold==0:
        print('MODEL:')
    for name, child in model.named_children():
        ct += 1
        if ct < N+1:
            for param in child.parameters():
                param.requires_grad = False
            if test_fold==0:
                print(f'{ct} {name}\tFROZEN')
        else:
            if drop1>0:
                child.register_forward_hook(lambda m, inp, out: F.dropout(out, p=drop1, training=m.training))
            if test_fold==0:
                print(f'{ct} {name}')
    
    print(f'\nCV FOLD {test_fold+1}/{K}')
    
    ## customize output layer..
    num_ftrs = model.fc.in_features
    
    ## design 1
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 2), nn.LogSoftmax(dim=1))
    
    ## design 2
    if fc>0:
        model.fc = nn.Sequential(nn.Linear(num_ftrs, fc),
                                  nn.ReLU(),
                                  nn.Dropout(drop2),
                                  nn.Linear(fc, 2),
                                  nn.LogSoftmax(dim=1))
    
    LR = 0.001
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, amsgrad=True)
    
    lr_sched = None
    # alpha = 0.5 ## final percentage of initial start rate
    # alpha = 0.25
    gamma = np.exp(np.log(alpha)/epochs)
    lr_sched = lr_scheduler.ExponentialLR(optimizer, gamma=gamma) # 0.98
    # lr_sched = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    ###########################################################
    ## TRAIN LOOP....
    
    steps, frames = 0,0
    running_loss = 0
    best_acc, best_f1, best_ap = 0, 0, 0
    train_losses, test_losses = [], []
    pe = print_every
    
    t0 = time.time()
    for epoch in range(epochs):
        
        ## test metrics more frequently in later epochs....
        if epoch>0 and epoch%3==0 and pe/batch_size>4:
            pe /= 2
        
        for inputs, labels, _ in trainloader:
            steps += batch_size
            frames += batch_size
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if frames >= pe:
                tdiff = time.time()-t0
                t0 = time.time()
                imgs_sec = (frames/tdiff)
                frames = 0
                
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    t,y,p = [],[],[]
                    for inputs, labels, paths in testloader:
                        if min(inputs.shape[-2:])<MINDIM:
                            continue

                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                        
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        t.extend(labels.cpu().numpy())
                        y.extend(ps[:,1].cpu().numpy())
                        p.extend(paths)
                        
                ##################################
                update_model = False
                # accuracy = accuracy/len(testloader)
                t,y,p = np.array(t), np.array(y), np.array(p)
                idx = np.argsort(-y)
                t,y,p = t[idx],y[idx],p[idx]
                ## save test image filenames
                if epoch==0 and TEST_FILE!=LAST_TEST:
                    write_lines([path_leaf(s) for s in p], LAST_TEST)
                
                ## ap
                ap = average_precision_score(t, y)
                pp,rr,tt = precision_recall_curve(t, y)
                ff1 = 2*pp*rr/(pp+rr+1e-16)
                ff1m = ff1.max()
                cc = tt[ff1.argmax()]
                imid = abs(tt-0.5).argmin()
                f1mid,pmid,rmid = ff1[imid],pp[imid],rr[imid]
                    
                exp = ' \t'
                if f1mid > best_f1: # F1m
                    best_f1 = f1mid # F1m
                    exp = '*\t'
                    # update_model = True
                  
                xxx = ' \t'
                if ap > best_ap:
                    best_ap = ap
                    xxx = '*\t'
                    update_model = True
                    
                if update_model:
                    best_model = copy.deepcopy(model)
                    if SAVE:
                        torch.save(best_model.state_dict(), MODEL_PATH + f'{ITEM.lower()}.pt')

                    if PLOT:
                        plt.plot(rr,pp,'g')
                        plt.plot(rr,ff1,'c')
                        plt.title('PR curve')
                        plt.show()
                    
                    ## FPs/FNs...
                    T,Y,S,CC = t,y,p,cc
    
                train_losses.append(running_loss/len(trainloader))
                test_losses.append(test_loss/len(testloader))
                scut = f"cut={cc:0.2g}"# if update_model else ""               
                print(f"Epoch {epoch+1}/{epochs}...\t"
                      f"LOSS: {running_loss/pe:.3f} "
                      f"/ {test_loss/len(testloader):.3f} \t"
                      # f"acc: {acc:.3f}{star} \t"
                      f"f1: {f1mid:.3f} ({pmid:.3f}/{rmid:.3f}){exp}\t"
                      f"AP: {ap:.3f}{xxx}"
                      f"\tf1max: {ff1m:.3f} {scut} \t"
                      f"{imgs_sec:.1f}FPS"
                      )
                
                running_loss = 0
                model.train()
        
        
        ## end epoch...
        #######################################################
        if lr_sched is not None:
            lr_sched.step()
            print(f'lr={lr_sched.get_last_lr()[0]:0.3g}')
    
    ## END TRAIN LOOP...
    ###########################################################
    Ts.extend(T)
    Ys.extend(Y)
    Ss.extend(S)
    
    ## break CV loop ?
    if not cv_complete:
        break
    
## END K-fold CV LOOP...
###########################################################

## save best model ?
if SAVE:
    torch.save(best_model.state_dict(), MODEL_PATH + f'{ITEM.lower()}.pt')

Ts,Ys,Ss = np.array(Ts), np.array(Ys), np.array(Ss)
idx = np.argsort(Ys)
Ts,Ys,Ss = Ts[idx],Ys[idx],Ss[idx]
    
## compute final K-fold metrics...
if cv_complete:
    
    ap = average_precision_score(Ts, Ys)
    pp,rr,tt = precision_recall_curve(Ts, Ys)
    f1 = 2*pp*rr/(pp+rr+1e-16)
    f1max = f1.max()
    pm,rm = pp[f1.argmax()],rr[f1.argmax()]
    CC = tt[f1.argmax()]
    
    cuta = 0.5
    ia = abs(tt-cuta).argmin()
    f1a,pa,ra = f1[ia],pp[ia],rr[ia]
    cutb = (CC+0.5)/2
    ib = abs(tt-cutb).argmin()
    f1b,pb,rb = f1[ib],pp[ib],rr[ib]
    
    print(f"\nFINAL K-fold metrics:\n"
          f"\tAP: {ap:.3f}\n"
          f"\tF1: {f1a:.3f} ({pa:.3f}/{ra:.3})\tcut={cuta:0.3g}\n"
          f"\tF1: {f1b:.3f} ({pb:.3f}/{rb:.3})\tcut={cutb:0.3g}\n"
          f"\tF1: {f1max:.3f} ({pm:.3f}/{rm:.3})\tcut={CC:0.3g}\n"
          )
    ##
    if PLOT:
        plt.plot(rr,pp,'g')
        plt.plot(rr,f1,'c')
        plt.title('PR curve')
        plt.show()
    
    if SAVE:
        np.savetxt(MODEL_PATH + 'P.txt', pp, fmt='%0.6f')
        np.savetxt(MODEL_PATH + 'R.txt', rr, fmt='%0.6f')

## sideline FPs and FNs for easier inspection/correction...
if FPFN:
    # conf = CC
    conf = (CC+0.5)/2
    eps = 0.15
    yi = (Ys>conf).astype(np.int32)
    zi = abs(Ys-conf)<eps
    fni, fpi = Ts>yi, Ts<yi
    fni[zi], fpi[zi] = False, False
    # fpi.sum(), fni.sum()
    fns, fps = Ss[fni], Ss[fpi]
    copy_files(fps, PATH_FP)
    copy_files(fns, PATH_FN)
    write_lines([path_leaf(f) for f in fps], FILE_FP)
    write_lines([path_leaf(f) for f in fns], FILE_FN)
    print(f'\n{len(fps)} FALSE POSITIVES.... copied to:\n\t{PATH_FP}')
    print(f'{len(fns)} FALSE NEGATIVES.... copied to:\n\t{PATH_FN}')
    