import os, sys
import json
import cv2
import pandas as pd
from PIL import Image
# from utils import *

import glob
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import ExifTags
from tqdm import tqdm

## convert labels exported from hasty (in COCO format, not hasty) to yolo format

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def write_lines(fn, lines):
    with open(fn, 'w') as f:
        for line in lines:
            f.write(f'{line}\n')
    
def convert_hasty2yolo(json_file, save_dir):
    global data

    # Import json
    fn = Path(save_dir) / 'labels'
    mkdirs(f'{fn}')
    # fn.mkdir()
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
        
        print(f)

        # The COCO box format is [top left x, top left y, width, height]
        box = np.array(x['bbox'], dtype=np.float64)
        box[:2] += box[2:] / 2  # xy top-left corner to center
        box[[0, 2]] /= w  # normalize x
        box[[1, 3]] /= h  # normalize y

        # Write
        if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
            cls = x['category_id'] - 1  # class
            line = cls, *(box)  # cls, box or segments
            with open((fn / f).with_suffix('.txt'), 'a') as file:
                file.write(('%g ' * len(line)).rstrip() % line + '\n')


if __name__ == '__main__':
    
    # hasty_json_file = '/home/david/code/phawk/data/fpl/damage/rgb/resnet/aitower/labelsT/all_1/hasty_import_export/drc_export_4.json'
    # output_dir = '/home/david/code/phawk/data/fpl/damage/rgb/resnet/aitower/labelsT/all_1'
    
    # hasty_json_file = '/home/david/code/phawk/data/generic/hasty_import_export/generic_1.json'
    # output_dir = '/home/david/code/phawk/data/generic'
    
    # hasty_json_file = '/home/david/code/phawk/data/solar/indivillage/hasty/indi_1.json'
    # output_dir = '/home/david/code/phawk/data/solar/indivillage'
    
    hasty_json_file = '/home/david/code/phawk/data/solar/indivillage/hasty/prog36.json'
    output_dir = '/home/david/code/phawk/data/solar/indivillage'
    
    convert_hasty2yolo(hasty_json_file, output_dir)
    