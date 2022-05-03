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
    
    json_file = '/home/david/code/phawk/data/generic/transmission/master/hasty/master_exp_4.json'
    save_dir = '/home/david/code/phawk/data/generic/transmission/master'
    
    ##########################
    
    # convert_hasty2yolo(json_file, save_dir)
    
    img_path = f"{Path(save_dir) / 'images'}" + '/'
    lab_path = Path(save_dir) / 'labels'
    lab_path = increment_path(lab_path)
    mkdirs(f'{lab_path}')
    
    # Import json
    with open(json_file) as f:
        data = json.load(f)
    
    ## label attributes
    attrib_values = {}
    for att in data['attributes']:
        if att['type'] == 'SELECTION':
            if att['name'].startswith('General'):
                continue
            attrib_values[att['name']] = ['None'] + att['values']
    
    ## save class names
    classes_file = f"{Path(save_dir) / 'classes.txt'}"
    classes = [d['class_name'] for d in data['label_classes']]
    write_lines(classes_file, classes)
    
    class_attribs = {}
    for d in data['label_classes']:
        for att in d['attributes']:
            if att in attrib_values:
                class_name = d['class_name']
                if class_name not in class_attribs:
                    class_attribs[class_name] = []
                class_attribs[class_name].append(att)
                
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
                cls = classes.index(class_name)
                line = cls, *(box)  # cls, box or segments
                with open((lab_path / f).with_suffix('.txt'), 'a') as file:
                    file.write(('%g ' * len(line)).rstrip() % line + '\n')
                ##
                if class_name in class_attribs:
                    for attrib in class_attribs[class_name]:
                        attrib_val = lab['attributes'][attrib] if attrib in lab['attributes'] else 'None'
                        att = attrib_values[attrib].index(attrib_val)
                        att_path = os.path.join(save_dir, 'attribs', attrib.replace(' ','_'), f'{att}')
                        save_one_box(XYXY, im, file=os.path.join(att_path, f), BGR=True)
                        
    sys.exit()
                    
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
        
    