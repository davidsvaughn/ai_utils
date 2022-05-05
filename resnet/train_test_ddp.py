import os, sys
import argparse
import time
import random
import math
from pathlib import Path
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import copy
from sklearn.metrics import average_precision_score, precision_recall_curve
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Iterator, Sequence
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Sampler
from torch.utils.data.sampler import SubsetRandomSampler

# packages for distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

#
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ------ Setting up the distributed environment -------
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # this function is responsible for synchronizing and successfully communicate across multiple process
    # involving multiple GPUs.

def cleanup():
    dist.destroy_process_group()
    
def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def scale_img_1d(img, scale):
    if scale<0:
        scale = -scale
        q = scale/min(img.shape[1:])
    else:
        q = scale/max(img.shape[1:])
    if q<1:
        h,w = int(img.shape[1]*q), int(img.shape[2]*q) 
        return F.interpolate(img.unsqueeze(0), size=[h,w]).squeeze()
    return img

def scale_img_2d(img, scale):
    area = img.shape[1] * img.shape[2]
    q = np.sqrt(scale/area)
    if q<1:
        h,w = int(img.shape[1]*q), int(img.shape[2]*q) 
        return F.interpolate(img.unsqueeze(0), size=[h,w]).squeeze()
    return img

def scale_img(img, scale, rand=None):
    if rand is not None:
        s2 = int(rand * scale)
        scale = random.randint(min(s2,scale),max(s2,scale))
    if scale>10000:
        return scale_img_2d(img, scale)
    return scale_img_1d(img, scale)

def random_rot90(x, p):
    if random.random()<p:
        k = random.randint(0,1)*2-1
        x = torch.rot90(x, k, [1,2])
    return x

class ImageCache(object):
    def __init__(self, scale=None):
        self.scale = scale
        self.data = {}
    
    def image_loader(self, path):
        with open(path, "rb") as f:
            img = np.array(Image.open(f).convert("RGB"))
            # img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            ####
            if self.scale is not None:
                img = torch.FloatTensor(img.transpose([2,0,1]))
                img = scale_img(img, self.scale).numpy().transpose([1,2,0]).astype(np.uint8) ## dsv******
            ####
            return img

    def load_image(self, path):
        if path not in self.data:
            self.data[path] = self.image_loader(path)
        return self.data[path]

class Collate(object):
    pad = 10
    rot = 0
    training = True
    null = False
    scale = None
    rand_scale = 0.5

    @staticmethod
    def disable():
        Collate.null = True

    @staticmethod
    def restore():
        Collate.null = False
    
    @staticmethod
    def train():
        Collate.training = True

    @staticmethod
    def eval():
        Collate.training = False
    
    ## used for TRAIN SET...
    @staticmethod
    def my_collate(batch):
        # data = [torch.FloatTensor(np.array(item[0]).transpose([2,0,1]) * (1/255)) for item in batch]
        data = [item[0] for item in batch]
        labels = torch.LongTensor([item[1] for item in batch])
        paths = [item[2] for item in batch]
        idx = torch.LongTensor([item[3] for item in batch])

        if Collate.null:
            return [None, None, paths, idx]
        
        ## resize...?
        # if Collate.scale is not None:
        #     data = [scale_img(img, Collate.scale, rand=Collate.rand_scale if Collate.training else None) for img in data]
         
        # ## randomly rotate +-90 deg...?
        # if Collate.training and Collate.rot>0:
        #     data = [random_rot90(img, Collate.rot) for img in data]

        ## orient all same....
        if Collate.training and Collate.rot>-1:
            szs = np.array([d.shape[1:] for d in data])
            q = (szs[:,0]>szs[:,1]).mean()
            v = 1 if np.random.rand()<q else -1
            data = [(random_rot90(img,1) if (v*img.shape[1])<(v*img.shape[2]) else img) for img in data]

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

        return [data, labels, paths, idx]

class DistributedSubsetRandomSampler(Sampler[int]):
    indices: Sequence[int]
    
    def __init__(self, indices: Sequence[int], num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=True) -> None:
        self.indices = indices
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if drop_last and len(self.indices) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.indices) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.indices) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        # for i in torch.randperm(len(self.indices), generator=self.generator):
        #     yield self.indices[i]
        ########################
        # deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        x = torch.randperm(len(self.indices), generator=g)
        if not self.shuffle:
            x = x.sort()[0]
        
        ## force each rank to sample the same subset of data every time
        ## for space efficient caching !!!!
        x = x[x%self.num_replicas==self.rank]
        # x = x[:self.num_samples]

        idx = x.tolist()
        if self.drop_last:
            idx = idx[:self.num_samples]
        else:
            padding_size = self.num_samples - len(idx)
            if padding_size <= len(idx):
                idx += idx[:padding_size]
            else:
                idx += (idx * math.ceil(padding_size / len(idx)))[:padding_size]

        #return iter(idx)
        for i in idx:
            yield self.indices[i]

    def __len__(self) -> int:
        # return len(self.indices)
        return self.num_samples
    
    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): _epoch number.
        """
        self.epoch = epoch
      
class ImageFolderWithPathsAndIndex(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPathsAndIndex, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path... and index....
        # tuple_with_path = (original_tuple + (path,))
        # tuple_with_path_and_index = (tuple_with_path + (index,))
        tuple_with_path_and_index = (original_tuple + (path,) + (index,))
        return tuple_with_path_and_index

class Transforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))['image']

def load_split_train_test(datadir, args, rank, seed, k=5, test_fold=0, loader=None):
    ###############################################################################
    ## original transforms
    # train_transforms = transforms.Compose([
    #                                     transforms.RandomHorizontalFlip(),
    #                                     # transforms.RandomRotation(degrees=(90, -90)),
    #                                     # transforms.RandomResizedCrop(512, scale=(0.1, 0.5)),
    #                                     transforms.ToTensor(),
    #                                     # transforms.ColorJitter(*cj),
    #                                     # transforms.Normalize(mean=[0.436, 0.45 , 0.413], std=[0.212, 0.208, 0.221]),
    #                                     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #                                     ])
    # test_transforms = transforms.Compose([
    #                                     transforms.ToTensor(),
    #                                     # transforms.ColorJitter(*cj),
    #                                     # transforms.Normalize(mean=[0.436, 0.45 , 0.413], std=[0.212, 0.208, 0.221]),
    #                                     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #                                   ])
    
    
    ###############################################################################
    ## albumentations....
    train_transforms = A.Compose(
        [
            A.Flip(p=0.5),
            A.RandomRotate90(p=0.25),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=[0.5,1], rotate_limit=15, p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    test_transforms = A.Compose(
        [
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    train_transforms = Transforms(train_transforms)
    test_transforms = Transforms(test_transforms)
    
    ###############################################################################
    
    train_data = ImageFolderWithPathsAndIndex(datadir, transform=train_transforms, loader=loader)
    test_data = ImageFolderWithPathsAndIndex(datadir, transform=test_transforms, loader=loader)
    
    ## train/test split
    num_train = len(train_data)
    idx = np.random.RandomState(seed=seed).permutation(num_train)
    n = int(np.ceil(num_train/k))
    test_idx = idx[n*test_fold:n*(test_fold+1)]
    train_idx = np.setdiff1d(idx, test_idx)

    ## train sampler
    train_sampler = DistributedSubsetRandomSampler(train_idx, num_replicas=args.world_size, rank=rank)
    trainloader = torch.utils.data.DataLoader(train_data, 
                                              sampler=train_sampler,
                                              batch_size=args.batch_size,
                                              pin_memory=True,
                                              collate_fn=Collate.my_collate,
                                              )
    
    ## test sampler
    # test_batch_size = 1
    test_batch_size = args.batch_size // 2
    test_sampler = DistributedSubsetRandomSampler(test_idx, num_replicas=args.world_size, rank=rank)
    testloader = torch.utils.data.DataLoader(test_data, 
                                             sampler=test_sampler,
                                             batch_size=test_batch_size,
                                             pin_memory=True,
                                             collate_fn=Collate.my_collate,
                                             )

    test_path_sampler = SubsetRandomSampler(test_idx)
    testpathloader = torch.utils.data.DataLoader(test_data,
                                                sampler=test_path_sampler,
                                                batch_size=test_batch_size,
                                                collate_fn=Collate.my_collate,
                                                )
                        
    return trainloader, testloader, testpathloader

def gather_tensors(t, device, world_size):
    if world_size==1:
        return t
    out_t = [torch.zeros_like(t, device=device) for _ in range(world_size)]
    dist.all_gather(out_t, t)
    return torch.cat(out_t, 0)

def train_model(rank, args):
    # global y,Y_test
    #######################
    ddp = args.world_size>1
    if ddp:
        print(f"Running Distributed ResNet on rank {rank}.")
        setup(rank, args.world_size)
    torch.cuda.set_device(rank)
    PLOT = True

    ##################################################################################
    ## WOOD DAMAGE ##

    LOCAL_ROOT  = '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/tags/'
    REMOTE_ROOT = '/home/ubuntu/data/wood_damage/tags/'
    
    ITEM, scale, args.res, fc, drops, print_every, rot, SEED  = 'Deteriorated', 1024, 50, 256, [0.66,0.33], 300, 0, 657657
    cv_complete = True
    K, alpha = 5, 0.5
    args.batch_size = 32
    args.epochs = 32
    
    
    # ITEM, scale, args.res, fc, drops, print_every, rot, SEED  = 'Vegetation', 1024, 34, 64, [0.66,0.33], 200, 0.25, 191919
    # cv_complete = True
    # K, alpha = 5, 0.25
    # args.batch_size = 32
    # args.epochs = 32

    ##################################################################################
    ## INSULATOR TYPE ##

    # LOCAL_ROOT  = '/home/david/code/phawk/data/generic/transmission/master/attribs/'
    # REMOTE_ROOT = '/home/ubuntu/data/attribs/'

    # ITEM, scale, args.res, fc, drops, print_every, rot, SEED  = 'Insulator_Type', 480, 18, 64, [0.66,0.33], 300, 0.25, 45245
    # cv_complete = False
    # K, alpha = 5, 0.5
    # args.batch_size = 32
    # args.epochs = 20

    ###############################
    ## INSULATOR MATERIAL ##

    # ITEM, scale, args.res, fc, drops, print_every, rot, SEED  = 'Insulator_Material', 480, 18, 64, [0.66,0.66], 320, 0.25, 1111
    # cv_complete = False
    # K, alpha = 5, 0.5
    # args.batch_size = 32
    # args.epochs = 20
    
    
    ##################################################################################
    SAVE = True
    home = str(Path.home())
    ROOT = LOCAL_ROOT if home in LOCAL_ROOT else REMOTE_ROOT
    PLOT = PLOT if home in LOCAL_ROOT else False
    DATA_PATH = os.path.join(ROOT, ITEM)
    MODEL_PATH = os.path.join(ROOT, 'models', ITEM)
    MODEL_FILE = os.path.join(MODEL_PATH, f'{ITEM}.pt')
    MODEL_CHKPT = os.path.join(MODEL_PATH, f'{ITEM}_chk.pt')
    if rank<1:
        make_dirs(MODEL_PATH)

    num_class = len(os.listdir(DATA_PATH))
    res, N, LR = args.res, args.freeze, args.lr
    Collate.scale, Collate.rot = scale, rot

    # set_random_seeds(random_seed=SEED)
    if rank<1:
        Ts,Ys,Ss = [],[],[]
        print(f'SEED={SEED}')
    
    for test_fold in range(K):
        
        MODEL_CHKPT = os.path.join(MODEL_PATH, f'{ITEM}_{test_fold+1}.pt')

        set_random_seeds(random_seed=SEED)

        if rank<1:
            print(f'\nCV FOLD {test_fold+1}/{K}')

        ## load RESNET
        res, N = args.res, args.freeze
        if res==50:
            model = models.resnet50(pretrained=True)
        elif res==18:
            model = models.resnet18(pretrained=True)
        elif res==101:
            model = models.resnet101(pretrained=True)
        else: # 34
            model = models.resnet34(pretrained=True)
        
        ## freeze layers 1-N in the total 10 layers of Resnet
        ct = 0
        drop1, drop2 = drops
        if rank<1 and test_fold==0:
            print('MODEL:')
        for name, child in model.named_children():
            ct += 1
            if ct < N+1:
                for param in child.parameters():
                    param.requires_grad = False
                if rank<1 and test_fold==0:
                    print(f'{ct} {name}\tFROZEN')
            else:
                if drop1>0:
                    child.register_forward_hook(lambda m, inp, out: F.dropout(out, p=drop1, training=m.training))
                if rank<1 and test_fold==0:
                    print(f'{ct} {name}')
        
        ## customize output layer..
        num_ftrs = model.fc.in_features
        if fc>0:
            model.fc = nn.Sequential(nn.Linear(num_ftrs, fc),
                                      nn.ReLU(),
                                      nn.Dropout(drop2),
                                      nn.Linear(fc, num_class),
                                      nn.LogSoftmax(dim=1))
        else:
            model.fc = nn.Sequential(nn.Linear(num_ftrs, num_class), nn.LogSoftmax(dim=1))
        
        if ddp: # wraps the network around distributed package
            model = DDP(model.to(rank), device_ids=[rank])
        else:
            model = model.to(rank)
        
        ## Loss and Optimizer
        criterion = nn.NLLLoss().to(rank)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, amsgrad=True)

        gamma = np.exp(np.log(alpha)/args.epochs)
        lr_sched = lr_scheduler.ExponentialLR(optimizer, gamma=gamma) # 0.98
        
        # image_cache = ImageCache()
        image_cache = ImageCache(scale=scale)
        trainloader, testloader, testpathloader = load_split_train_test(DATA_PATH, args, rank, seed=SEED, k=K, test_fold=test_fold, loader=image_cache.load_image)

        ## get paths...
        Collate.disable()
        if rank<1:
            pathmap = {}
            for _, _, paths, idx in testpathloader:
                idx = idx.cpu().numpy()
                for i,path in zip(idx,paths):
                    pathmap[i] = path
            print('Starting training...')
        Collate.restore()

        # Training
        device = next(model.parameters()).device
        running_loss = 0
        best_f1, best_ap, best_acc = 0, 0, 0
        train_losses, test_losses = [], []
        pe = print_every
        
        for epoch in range(args.epochs):
            t0 = t1 = time.time()
            frames = 0
            train_secs, test_secs = 0,0
            train_imgs, test_imgs = 0,0
            trainloader.sampler.set_epoch(epoch)
            
            ## test metrics more frequently in later epochs....
            # if epoch/args.epochs==0.5: pe /= 2
                
            for inputs, labels, _,_ in trainloader:
                frames += args.batch_size * args.world_size
                inputs, labels = inputs.to(rank), labels.to(rank)
                
                optimizer.zero_grad()
                logps = model.forward(inputs.float())
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                if frames>=pe:

                    if rank<1:
                        t,y,p,test_loss = [],[],[],0
                        train_imgs += frames
                        train_secs += time.time()-t0
                        t0 = time.time()

                    model.eval()
                    Collate.eval()
                    frames = 0

                    with torch.no_grad():

                        for inputs, labels, paths, idx in testloader:
                            inputs, labels, idx = inputs.to(rank), labels.to(rank), idx.to(rank)
                            logps = model.forward(inputs.float())
                            ps = torch.exp(logps)

                            ## gather outputs from all DDP processes...
                            ps = gather_tensors(ps, device, args.world_size)
                            targets = gather_tensors(labels, device, args.world_size)
                            idx = gather_tensors(idx, device, args.world_size)

                            if rank<1:
                                frames += len(targets)
                                batch_loss = criterion(logps, labels)
                                test_loss += batch_loss.item()
                                # y.extend(ps[:,1].cpu().numpy())
                                y.append(ps.cpu().numpy())
                                t.extend(targets.cpu().numpy())
                                for i in idx.cpu().numpy():
                                    p.append(pathmap[i])
                    
                    ##################################################
                    if rank<1:
                        ## timings
                        test_imgs += frames
                        test_secs += time.time()-t0
                        t0, frames = time.time(), 0
                        fps_train = int(train_imgs/train_secs)
                        fps_test = int(test_imgs/test_secs)

                        update_model = False
                        t,p = np.array(t), np.array(p)
                        y = np.vstack(y)
                        ##########################################
                        ## https://stats.stackexchange.com/questions/534431/calculating-area-under-the-precision-recall-curve-with-multiclass

                        if num_class>2:
                            precision, recall = dict(), dict()
                            aps = []
                            Y_test = (t[:,None]==np.arange(num_class)[None,:])*1
                            # Y_pred = (y.argmax(1)[:,None]==np.arange(num_class)[None,:])*1
                            # # Y_pred = (abs(y-y.max(1)[:,None])<0.00001).astype(np.int32)
                            
                            for i in range(num_class):
                                yi, ti = y[:,i], Y_test[:,i] # (t==i)+0
                                precision[i], recall[i], _ = precision_recall_curve(ti, yi)
                                aps.append(average_precision_score(ti, yi))
                                
                            # A "micro-average": quantifying score on all classes jointly
                            pp,rr,_ = precision_recall_curve(Y_test.ravel(), y.ravel())
                            ap = average_precision_score(Y_test, y, average="micro")
                            aps = np.array(aps).round(2).tolist()
                            ap = np.min(aps)
                            
                            ## global accuracy
                            acc = (Y_test.argmax(1) == y.argmax(1)).mean()
                            
                            ## chance adjusted accuracy
                            e = 10
                            acc_e = np.mean([(Y_test.argmax(1) == np.random.randint(0,3,len(y))).mean() for _ in range(e)])
                            acc_k = (acc-acc_e)/(1-acc_e)

                            xxx = ' \t'
                            if ap > best_ap:
                                best_ap = ap
                                xxx = '*\t'
                                # update_model = True
                            eee = ' \t'
                            if acc_k > best_acc:
                                best_acc = acc_k
                                eee = '*\t'
                                update_model = True
                                
                            if update_model:
                                best_model = copy.deepcopy(model)
                                if SAVE:
                                    torch.save(best_model.state_dict(), MODEL_CHKPT)
                                if PLOT:
                                    plt.plot(rr,pp,'g')
                                    plt.title('PR Curve')
                                    plt.show()
                                ## FPs/FNs...
                                T,Y,S,CC = Y_test,y,p,0
                                ##
                                # if epoch>7: sys.exit()
                                
                            print(f"Epoch {epoch+1}/{args.epochs}...\t"
                                  f"ACC: {acc_k:.3f}{eee}"
                                  f"AP: {aps}\t{ap:.3f}{xxx}"
                                  f"FPS:{fps_train}/{fps_test}"
                                )

                        ##########################################
                        else:
                            y = y[:,1]
                            idx = np.argsort(-y)
                            t,y,p = t[idx],y[idx],p[idx]
                            
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
                            xxx = ' \t'
                            if ap > best_ap:
                                best_ap = ap
                                xxx = '*\t'
                                update_model = True
                            if update_model:
                                best_model = copy.deepcopy(model)
                                if SAVE:
                                    torch.save(best_model.state_dict(), MODEL_CHKPT)
                                if PLOT:
                                    plt.plot(rr,pp,'g')
                                    plt.title('PR Curve')
                                    plt.show()
                                ## FPs/FNs...
                                T,Y,S,CC = t,y,p,cc
                            train_losses.append(running_loss/len(trainloader))
                            test_losses.append(test_loss/len(testloader))
                            scut = f"cut={cc:0.2g}"
                            print(f"Epoch {epoch+1}/{args.epochs}...\t"
                                f"LOSS: {running_loss/pe:.4f} "
                                f"/ {test_loss/len(testloader):.4f} \t"
                                f"f1: {f1mid:.3f} ({pmid:.3f}/{rmid:.3f}){exp}"
                                f"AP: {ap:.3f}{xxx}"
                                f"f1max: {ff1m:.3f} {scut} \t"
                                f"FPS:{fps_train}/{fps_test}"
                                )
                    ### end eval ##########
                    model.train()
                    Collate.train()
                    frames, running_loss = 0,0
                    
            ## end epoch...
            #######################################################
            if rank<1:
                print(f'{(time.time()-t1)/60:0.2f}min')

            if lr_sched is not None:
                lr_sched.step()
                if rank<1:
                    print(f'lr={lr_sched.get_last_lr()[0]:0.3g}')
                
        ## END TRAIN LOOP...
        ###########################################################
        if rank<1:
            Ss.extend(S)
            if num_class>2:
                Ts.append(T)
                Ys.append(Y)
            else:
                Ts.extend(T)
                Ys.extend(Y)
            if SAVE:
                torch.save(best_model.state_dict(), MODEL_FILE)
                print(f'Best Model saved to:\n\t{MODEL_FILE}')
        
        ## break CV loop ?
        if not cv_complete: break
        # if test_fold==3: break
        
    ## END K-FOLD LOOP...
    ###########################################################
    
    if rank<1:
        if num_class>2:
            Ts, Ys = np.vstack(Ts), np.vstack(Ys)
            Ss = np.array(Ss)
            precision, recall = dict(), dict()
            aps = []
            for i in range(num_class):
                yi, ti = Ys[:,i], Ts[:,i] # (t==i)+0
                precision[i], recall[i], _ = precision_recall_curve(ti, yi)
                aps.append(average_precision_score(ti, yi))
            # A "micro-average": quantifying score on all classes jointly
            pp,rr,_ = precision_recall_curve(Ts.ravel(), Ys.ravel())
            ap = average_precision_score(Ts, Ys, average="micro")
            aps = np.array(aps).round(2).tolist()
            ap = np.min(aps)

            print(f"\nK-FOLD METRICS:\n"
                f"AP: {aps}\t{ap:.3f}"
                )
            if PLOT:
                plt.plot(rr,pp,'g')
                plt.title('K-FOLD PR Curve')
                plt.show()

        else:
            Ts,Ys,Ss = np.array(Ts), np.array(Ys), np.array(Ss)
            idx = np.argsort(Ys)
            Ts,Ys,Ss = Ts[idx],Ys[idx],Ss[idx]
            
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
    if ddp:
        cleanup()


def run_train_model(train_func, world_size):

    parser = argparse.ArgumentParser("PyTorch - Training ResNet101 on CIFAR10 Dataset")
    parser.add_argument('--world_size', type=int, default=world_size, help='total number of processes')
    parser.add_argument('--lr', default=0.001, type=float, help='Default Learning Rate')
    parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
    parser.add_argument('--epochs', type=int, default=20, help='Total number of epochs for training')
    parser.add_argument('--res', type=int, default=34, help='ResNet model (18,34,50,101)')
    parser.add_argument('--freeze', type=int, default=7, help='')
    args = parser.parse_args()
    print(args)

    # this is responsible for spawning 'nprocs' number of processes of the train_func function with the given arguments as 'args'
    if world_size>1:
        mp.spawn(train_func, args=(args,), nprocs=args.world_size, join=True)
    else:
        train_func(rank=0, args=args)


if __name__ == "__main__":
    # since this example shows a single process per GPU, the number of processes is simply replaced with the
    # number of GPUs available for training.
    n_gpus = torch.cuda.device_count()
    run_train_model(train_model, n_gpus)

