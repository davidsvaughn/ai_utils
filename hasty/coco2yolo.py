import os,sys
import json
import glob
import re
from pathlib import Path
import numpy as np
from tqdm import tqdm
import ntpath

# import cv2
# import pandas as pd
# from PIL import Image
# from utils import *

## Script to convert labels exported from hasty (in COCO format, *NOT* hasty format) to YOLO format...
## -- this is done before we can build training manifest file for AWS

txt,jpg,JPG = '.txt','.jpg','.JPG'

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def write_lines(fn, lines):
    with open(fn, 'w') as f:
        for line in lines:
            f.write(f'{line}\n')

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_filenames(path, ext=jpg, noext=False):
    pattern = os.path.join(path, f'*{ext}')
    x = np.array([path_leaf(f) for f in glob.glob(pattern)])
    if noext:
        x = np.array([f.replace(ext,'') for f in x])
    return x

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

def make_empty_file(fn):
    with open(fn, 'w') as f:
        f.write('')
        
def make_empty_labels(img_path, lab_path):
    img_files = get_filenames(img_path, jpg, True)
    lab_files = get_filenames(lab_path, txt, True)
    idx = ~np.in1d(img_files, lab_files)
    files = img_files[idx]
    for fn in files:
        make_empty_file(os.path.join(lab_path, f'{fn}{txt}'))
    
def convert_hasty2yolo(json_file, save_dir=None, img_path=None, empty=False):
    if save_dir is None:
        save_dir = ntpath.split(json_file)[0]
    
    lab_path = Path(save_dir) / 'labels'
    lab_path = increment_path(lab_path)
    mkdirs(f'{lab_path}')
    
    if img_path is None:
        img_path = Path(save_dir) / 'images'
    
    # Import json
    with open(json_file) as f:
        data = json.load(f)
    
    # save class names
    classes_file = f"{Path(save_dir) / 'classes.txt'}"
    classes = [d['name'] for d in data['categories']]
    write_lines(classes_file, classes)
    
    # Create image dict
    images = {'%g' % x['id']: x for x in data['images']}

    # Write labels file
    for x in tqdm(data['annotations'], desc=f'Annotations {json_file}'):
        if x['iscrowd']:
            continue

        img = images['%g' % x['image_id']]
        h, w, f = img['height'], img['width'], img['file_name']
        
        # print(f)

        # The COCO box format is [top left x, top left y, width, height]
        box = np.array(x['bbox'], dtype=np.float64)
        box[:2] += box[2:] / 2  # xy top-left corner to center
        box[[0, 2]] /= w  # normalize x
        box[[1, 3]] /= h  # normalize y

        # Write
        if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
            cls = x['category_id'] - 1  # class
            line = cls, *(box)  # cls, box or segments
            with open((lab_path / f).with_suffix('.txt'), 'a') as file:
                file.write(('%g ' * len(line)).rstrip() % line + '\n')
                
    if empty:
        make_empty_labels(img_path, lab_path)


if __name__ == '__main__':

    hasty_json_file = '/home/david/code/phawk/data/generic/transmission/rgb/damage/wood_damage/hasty/woodex1.json'
    output_dir = '/home/david/code/phawk/data/generic/transmission/rgb/damage/wood_damage'
    convert_hasty2yolo(hasty_json_file, output_dir, empty=True)
    
    sys.exit()
    
    ##
    import argparse    
    parser = argparse.ArgumentParser(description='Convert hasty labels (COCO format) to YOLO labels')
    parser.add_argument('--input', help='json label file from hasty (COCO format)', type=str)
    parser.add_argument('--output', help='output directory', type=str)
    parser.add_argument('--images', help='images directory', type=str)
    parser.add_argument('--empty', help='make empty label files', action='store_true')
    args = parser.parse_args()
    
    convert_hasty2yolo(args.input, save_dir=args.output, img_path=args.images, empty=args.empty)
    