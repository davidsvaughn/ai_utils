import os, sys, re
import json
import cv2
import pandas as pd
from PIL import Image
# from utils import *

import glob
import os
import shutil
from shutil import copyfile
from pathlib import Path
import numpy as np
from PIL import ExifTags
from tqdm import tqdm

## convert labels exported from hasty (in hasty format, not COCO) to yolo format

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def write_lines(fn, lines):
    with open(fn, 'w') as f:
        for line in lines:
            f.write(f'{line}\n')
            
def increment_path(path, exist_ok=False, sep='_', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path
            
def XYXY2xywh(xyxy,W,H):
    x1,y1,x2,y2 = xyxy
    w,h = (x2-x1), (y2-y1)
    x,y = x1+w/2, y1+h/2
    return np.array([x/W,y/H,w/W,h/H])
    
# def convert_hasty2yolo(json_file, save_dir):
    
    # img_path = f"{Path(save_dir) / 'images'}" + '/'
    # lab_path = Path(save_dir) / 'labels'
    # lab_path = increment_path(lab_path)
    # mkdirs(f'{lab_path}')

#     # Import json
#     with open(json_file) as f:
#         data = json.load(f)
    
#     # save class names
#     classes_file = f"{Path(save_dir) / 'classes.txt'}"
#     classes = [d['name'] for d in data['categories']]
#     write_lines(classes_file, classes)
    
#     # Create image dict
#     images = {'%g' % x['id']: x for x in data['images']}

#     # Write labels file
#     for x in tqdm(data['annotations'], desc=f'Annotations {json_file}'):
#         if x['iscrowd']:
#             continue

#         img = images['%g' % x['image_id']]
#         h, w, f = img['height'], img['width'], img['file_name']
        
#         print(f)

#         # The COCO box format is [top left x, top left y, width, height]
#         box = np.array(x['bbox'], dtype=np.float64)
#         box[:2] += box[2:] / 2  # xy top-left corner to center
#         box[[0, 2]] /= w  # normalize x
#         box[[1, 3]] /= h  # normalize y

#         # Write
#         if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
#             cls = x['category_id'] - 1  # class
#             line = cls, *(box)  # cls, box or segments
#             with open((lab_path / f).with_suffix('.txt'), 'a') as file:
#                 file.write(('%g ' * len(line)).rstrip() % line + '\n')


if __name__ == '__main__':
    
    json_file = '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/hasty/wood_dam_ex2.json'
    save_dir = '/home/david/code/phawk/data/generic/transmission/damage/wood_damage'
    
    ##########################
    
    # convert_hasty2yolo(json_file, save_dir)
    
    # Import json
    lab_path = Path(save_dir) / 'labels'
    img_path = f"{Path(save_dir) / 'images'}" + '/'
    mkdirs(f'{lab_path}')
    with open(json_file) as f:
        data = json.load(f)
    
    # save class names
    classes_file = f"{Path(save_dir) / 'classes.txt'}"
    classes = [d['class_name'] for d in data['label_classes']]
    write_lines(classes_file, classes)
    
    tags = {}
    all_files = []
    
    for img in data['images']:
        
        h, w, f = img['height'], img['width'], img['image_name']
        img_tags, labs = img['tags'], img['labels']
        
        all_files.append(f)
    
        ## store image tags
        for t in img_tags:
            if t not in tags:
                tags[t] = []
            tags[t].append(f)
        
        # save object detections in yolo format
        for lab in labs:
            box = np.array(lab['bbox'], dtype=np.float64)
            box = XYXY2xywh(box,w,h)
            # Write
            if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
                cls = classes.index(lab['class_name'])
                line = cls, *(box)  # cls, box or segments
                with open((lab_path / f).with_suffix('.txt'), 'a') as file:
                    file.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
                    
    ################################
    ## setup image tag classification...
    all_files = np.array(all_files)
    all_files.sort()
    
    for tag in tags.keys():
        pos_files = np.array(tags[tag])
        pos_files.sort()
        idx = ~np.in1d(all_files, pos_files)
        neg_files = all_files[idx]
        
        tag_path = os.path.join(save_dir, 'tags', tag)
        pos_path = os.path.join(tag_path, '1')
        neg_path = os.path.join(tag_path, '0')
        mkdirs(pos_path)
        mkdirs(neg_path)
        
        for f in pos_files:
            src = os.path.join(img_path, f)
            dst = os.path.join(pos_path, f)
            copyfile(src, dst)
        for f in neg_files:
            src = os.path.join(img_path, f)
            dst = os.path.join(neg_path, f)
            copyfile(src, dst)
        
    