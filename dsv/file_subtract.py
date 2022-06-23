import os,sys
from glob import glob
import ntpath
import numpy as np
from shutil import copyfile
import pathlib
import datetime
import cv2

jpg,txt = '.jpg','.txt'

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_filenames(path, ext=jpg):
    pattern = os.path.join(path, f'*{ext}')
    return np.array([path_leaf(f) for f in glob(pattern)])

def empty_file(fn):
    with open(fn, 'w') as f:
        f.write('')
        
#####################################

## remove all pp1 images from pp2

pp1 = ['/home/david/code/phawk/data/generic/transmission/damage/wood_damage/images',
       '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/tags/Deteriorated/0',
       '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/tags/Deteriorated/1',
       '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/tags/fpfn/Deteriorated/del',
       '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/tags/Vegetation/0',
       '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/tags/Vegetation/1',
       '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/tags/fpfn/Vegetation/del',
       ]

pp2 = ['/home/david/code/phawk/data/generic/transmission/damage/wood_damage/stuff/dist',
      '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/stuff/dump_aep',
      '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/stuff/dump_claire',
      '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/stuff/dump_master',
      '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/stuff/Deteriorated/maybe'
      ]

#################################
item = 'Deteriorated'
# item = 'Vegetation'

## step1
pp1 = [f'/home/david/code/phawk/data/generic/transmission/damage/wood_damage/stuff/{item}/del',]
pp2 = [f'/home/david/code/phawk/data/generic/transmission/damage/wood_damage/stuff/{item}/1',
       f'/home/david/code/phawk/data/generic/transmission/damage/wood_damage/stuff/{item}/0',
       f'/home/david/code/phawk/data/generic/transmission/damage/wood_damage/stuff/{item}/maybe'
       ]
## step2
pp1 = [f'/home/david/code/phawk/data/generic/transmission/damage/wood_damage/stuff/{item}/1',]
pp2 = [f'/home/david/code/phawk/data/generic/transmission/damage/wood_damage/stuff/{item}/0',]

#############

# pp1 = [f'/home/david/code/phawk/data/generic/transmission/damage/wood_damage/images',]
# pp2 = [f'/home/david/code/phawk/data/generic/transmission/damage/wood_damage/labels',]

# ## find image/label mismatch
# n = 0
# for p1 in pp1:
#     images = np.array([f.replace(jpg,'') for f in get_filenames(p1)])
#     for p2 in pp2:
#         labels = np.array([f.replace(txt,'') for f in get_filenames(p2, txt)])
#         idx = ~np.in1d(labels, images)
#         print(labels[idx])

# sys.exit()

#################################

ITEM = 'Insulator_Material'
# ITEM = 'Insulator_Type'

pp1 = [f'/home/david/code/phawk/data/generic/transmission/master/attribs/{ITEM}/0',
       f'/home/david/code/phawk/data/generic/transmission/master/attribs/{ITEM}/1',
       f'/home/david/code/phawk/data/generic/transmission/master/attribs/{ITEM}/2',
       ]
pp2 = [f'/home/david/code/phawk/data/generic/transmission/master/attribs/test/{ITEM}',
       f'/home/david/code/phawk/data/generic/transmission/master/attribs/add/{ITEM}/0',
       f'/home/david/code/phawk/data/generic/transmission/master/attribs/add/{ITEM}/1',
       f'/home/david/code/phawk/data/generic/transmission/master/attribs/add/{ITEM}/2',
       ]

#################################

## remove all pp1 images from pp2
n = 0
for p1 in pp1:
    images1 = get_filenames(p1)
    for p2 in pp2:
        images2 = get_filenames(p2)
        idx = np.in1d(images2, images1)
        for f in images2[idx]:
            os.remove(os.path.join(p2, f))
            n+=1

print(f'Removed {n} files')

sys.exit()

#####################################
## BGR ==> RGB 

def isint(s):
    try:
        int(s)
    except:
        return False
    return True

def fix_files(file_path):
    files = glob(os.path.join(file_path, '*.jpg'))
    t = datetime.datetime(2021, 6, 12)
    for fn in files:
        fname = pathlib.Path(fn)
        mtime = datetime.datetime.fromtimestamp(fname.stat().st_mtime)
        if mtime<t:
            x = cv2.imread(fn)
            cv2.imwrite(fn, x[:,:,::-1])
            
def BGR2RGB(file_path):
    files = glob(os.path.join(file_path, '*.jpg'))
    n = 0
    for fn in files:
        fname = path_leaf(fn)
        if '_' in fname:
            s = fname.split('_')[0]
            if isint(s):
                x = cv2.imread(fn)
                cv2.imwrite(fn, x[:,:,::-1])
                n+=1
    print(f'Flipped {n} files')

# # pth = '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/stuff/dist'
# # pth = '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/tags/Deteriorated/0'
# # pth = '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/tags/Deteriorated/1'
# # pth = '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/tags/Vegetation/0'
# pth = '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/tags/Vegetation/1'
# BGR2RGB(pth)
# sys.exit()

#####################################
## match images and labels.....

# p = '/home/david/code/phawk/data/generic/transmission/damage/insulator_damage/'
# imgp = p + 'images'
# labp = p + 'labels'

# images = np.array([f.replace(jpg,'') for f in get_filenames(imgp)])
# labels = np.array([f.replace(txt,'') for f in get_filenames(labp, txt)])

# images.sort()
# labels.sort()

# idx = np.in1d(images, labels)
# print(images[~idx])

# sys.exit()

#####################################
## subtract folders

p1 = '/home/david/code/phawk/data/generic/transmission/claire/images_for_hasty_2/'
p2 = '/home/david/code/phawk/data/generic/transmission/master/images/'

fns = get_filenames(p1)
for fn in fns:
    f = p2 + fn
    os.remove(f)

sys.exit()

######################################

# import torch

# z = 0.1
# ny,nx = 5,12
# vy,vx = torch.meshgrid([torch.arange(ny),torch.arange(nx)])
# # g = torch.stack((vx, vy), 2).view((1, 1, ny, nx, 2))
# xx = abs(vx-(nx-1)/2)==(nx-1)/2
# yy = abs(vy-(ny-1)/2)==(ny-1)/2
# h = (xx+yy)*1
# H = h*z + 1-h

# H = H.view((1,1,ny,nx))
# print(H)

# sys.exit()

######################################

## remove train set (repeats) from dirty (damage & control)....
tfiles = get_filenames(p_train)
for tf in tfiles:
    ff = [p1d+tf, p1c+tf]
    for f in ff:
        if os.path.exists(f):
            print(f'deleting: {f}')
            os.remove(f)
            
# sys.exit()

## remove damage from control....
dfiles = get_filenames(p1d)
for df in dfiles:
    cf = p1c+df
    if os.path.exists(cf):
        print(f'deleting: {cf}')
        os.remove(cf)

# sys.exit()

## copy clean damage files to clean folder...
dfiles = get_filenames(p1d)
for fn in dfiles:
    src = p_src_clean+fn
    dst = p3d+fn
    copyfile(src, dst)

## copy clean control files to clean folder...
cfiles = get_filenames(p1c)
for fn in cfiles:
    src = p_src_clean+fn
    dst = p3c+fn
    copyfile(src, dst)
    ## empty labels (just for control group)
    empty_file(dst.replace(jpg, txt))