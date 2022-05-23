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

def read_lines(fn):
    with open(fn, 'r') as f:
        lines = [n.strip() for n in f.readlines()]
    return np.array([line for line in lines if len(line)>0])
            
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

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
    
def save_one_box(xyxy, im, file='image.jpg', gain=1.02, pad=5, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    b = xyxy2xywh(xyxy[None,:])  # boxes
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b)[0]
    xyxy = xyxy.clip([0,0,0,0], np.hstack([im.shape[:2], im.shape[:2]])[::-1])
    crop = im[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), ::(1 if BGR else -1)]
    if save:
        s = f'_crop-{int(xyxy[1])}-{int(xyxy[0])}.jpg'
        file = str(file).replace('.jpg', s)
        cv2.imwrite(file, crop)     
    return crop


if __name__ == '__main__':
    
    json_file = '/home/david/code/phawk/data/generic/transmission/rgb/master/hasty/master_exp_5.json'
    
    img_path = '/home/david/code/phawk/data/generic/transmission/rgb/master/images/'
    save_dir = '/home/david/code/phawk/data/generic/transmission/rgb/master/model/model5'
    
    # save_dir = '/home/david/code/phawk/data/generic/transmission/rgb/master'
    # img_path = f"{Path(save_dir) / 'images'}" + '/'
    

    attrib_classes = None
    attrib_classes = {'Insulator': ['Dead-end']}
    
    # save_crop = True
    save_crop = False
    ##########################
    
    lab_path = Path(save_dir) / 'labels'
    lab_path = increment_path(lab_path)
    mkdirs(f'{lab_path}')
    
    # Import json
    with open(json_file) as f:
        data = json.load(f)

    ## classes
    json_classes = np.array([d['class_name'] for d in data['label_classes']])
    classes_file = f"{Path(save_dir) / 'classes.txt'}"
    if os.path.exists(classes_file):
        classes = read_lines(classes_file)
        idx = ~np.in1d(json_classes, classes)
        if idx.sum()>0:
            print(f'Removing classes: {json_classes[idx]}')
        idx = ~np.in1d(classes, json_classes)
        if idx.sum()>0:
            print(f'Adding classes: {classes[idx]}')
    else:
        classes = json_classes
        write_lines(classes_file, classes)
    classes = list(classes)
    
    ## label attributes
    attrib_values = {}
    for att in data['attributes']:
        if att['type'] == 'SELECTION':
            if att['name'].startswith('General'):
                continue
            attrib_values[att['name']] = ['None'] + att['values']
    
    class_attribs = {}
    for d in data['label_classes']:
        for att in d['attributes']:
            if att in attrib_values:
                class_name = d['class_name']
                if class_name not in class_attribs:
                    class_attribs[class_name] = []
                class_attribs[class_name].append(att)
            
    if save_crop:    
        for k,vals in attrib_values.items():
            att = k.replace(' ','_') 
            att_path = os.path.join(save_dir, 'attribs', att)
            mkdirs(att_path)
            att_file = os.path.join(save_dir, 'attribs', f'{att}.txt')
            write_lines(att_file, vals)
            for i,_ in enumerate(vals):
                val_path = os.path.join(att_path, f'{i}')
                mkdirs(val_path)
    
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
            
        im = cv2.imread(img_path + f)
        
        # save object detections in yolo format
        for lab in labs:
            XYXY = np.array(lab['bbox'], dtype=np.int32)
            box = XYXY2xywh(XYXY,w,h)
            # Write
            if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
                class_name = lab['class_name']

                ## class attributes...
                if class_name in class_attribs:
                    for attrib in class_attribs[class_name]:
                        attrib_val = lab['attributes'][attrib] if attrib in lab['attributes'] else 'None'
                        
                        if attrib_classes is not None and class_name in attrib_classes:
                            if attrib_val in attrib_classes[class_name]:
                                class_name = f'{attrib_val} {class_name}'
                                if class_name not in classes:
                                    print(f'ERROR: {class_name} not in class list')
                                    sys.exit()

                        if save_crop:
                            att = attrib_values[attrib].index(attrib_val)
                            att_path = os.path.join(save_dir, 'attribs', attrib.replace(' ','_'), f'{att}')
                            save_one_box(XYXY, im, file=os.path.join(att_path, f), BGR=True)
                
                ## write to yolo label file
                if class_name not in classes:
                    continue
                cls = classes.index(class_name)
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
        
    