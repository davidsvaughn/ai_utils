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
    
def convert_hasty2yolo(json_file, save_dir):

    # Import json
    fn = Path(save_dir) / 'labels'
    fn.mkdir()
    with open(json_file) as f:
        data = json.load(f)

    # Create image dict
    images = {'%g' % x['id']: x for x in data['images']}

    # Write labels file
    for x in tqdm(data['annotations'], desc=f'Annotations {json_file}'):
        if x['iscrowd']:
            continue

        img = images['%g' % x['image_id']]
        h, w, f = img['height'], img['width'], img['file_name']

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
    
    hasty_json_file = '/home/david/code/phawk/data/aep/final/aep_hasty.json'
    output_dir = '/home/david/code/phawk/data/aep/final/output'
    
    convert_hasty2yolo(coco_json_file, output_dir)
    