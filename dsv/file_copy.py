import os,sys
from glob import glob
import ntpath
import numpy as np
from shutil import copyfile
import pathlib
import datetime
import cv2

jpg,txt = '.jpg','.txt'
JPG = '.JPG'

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_filenames(path, ext=jpg):
    pattern = os.path.join(path, f'*{ext}')
    return np.array([path_leaf(f) for f in glob(pattern)])

def read_lines(fn):
    with open(fn, 'r') as f:
        lines = [n.strip() for n in f.readlines()]
    return np.array([line for line in lines if len(line)>0])

def write_lines(lines, fn):
    with open(fn, 'w') as f:
        for line in lines:
            f.write("%s\n" % line)

def empty_file(fn):
    with open(fn, 'w') as f:
        f.write('')
        
#####################################

## match images and labels.....

# p1 = '/home/david/code/phawk/data/solar/thermal/damage/orig_2021/'
# p2 = '/home/david/code/phawk/data/solar/thermal/damage/'

# imgp1 = p1 + 'images/'
# labp1 = p1 + 'labels/'
# imgp2 = p2 + 'images/'
# labp2 = p2 + 'labels/'

# labels = read_lines(p1 + 'keep.txt')
# images = np.array([f.replace(txt,jpg) for f in labels])

# for f in labels:
#     copyfile(labp1+f, labp2+f)

# for f in images:
#     copyfile(imgp1+f, imgp2+f)

# sys.exit()
##########################################################

p1 = '/home/david/code/phawk/data/generic/transmission/rgb/master/inspect/'
p2 = '/home/david/code/phawk/data/generic/transmission/rgb/master/new/'
p3 = '/home/david/code/phawk/data/generic/transmission/rgb/master/images/'

pa = '/home/david/code/phawk/data/generic/transmission/rgb/claire/images/'
pb = '/home/david/code/phawk/data/generic/transmission/rgb/aep/images/'
pc = '/home/david/code/phawk/data/generic/transmission/rgb/pge/Hoogeveen_738/'

pp = [pa, pb, pc]


fns = get_filenames(p1, jpg)
for fn in fns:
    dd = p3 + fn
    if os.path.exists(dd): continue
    dd = p3 + fn.replace(jpg, '-2.jpg')
    if os.path.exists(dd): continue

    for p in pp:
        ff = get_filenames(p, jpg)
        if fn in ff:
            src = p + fn
            dst = p2 + fn
            copyfile(src, dst)
    