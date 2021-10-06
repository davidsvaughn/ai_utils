import os, sys
import json
# import cv2
# import pandas as pd
# from PIL import Image
# from utils import *

import glob
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import ExifTags
from tqdm import tqdm

'''
Convert COCO predictions output by detectron2 --> YOLO format for metrics computation (against target YOLO labels)
'''


def convert_coco_json(target_json_file, save_dir, pred_json_file='coco_instances_results.json'):
    pred_json_file = os.path.join(save_dir, pred_json_file)

    # Import json
    fn = Path(save_dir) / 'labels' #/ json_file.stem.replace('instances_', '')  # folder name
    fn.mkdir()
    with open(target_json_file) as f:
        data = json.load(f)

    # Create image dict
    images = {'%g' % x['id']: x for x in data['images']}

    with open(pred_json_file) as f:
        preds = json.load(f)

    # Write labels file
    for x in preds:

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
            conf = x['score']  # confidence
            line = cls, *(box), conf  # cls, box or segments
            with open((fn / f).with_suffix('.txt'), 'a') as file:
                file.write(('%g ' * len(line)).rstrip() % line + '\n')


if __name__ == '__main__':

    data_root = '/home/david/code/phawk/detectron/data/T/'
    # img_path = data_root + 'images'
    # train_json_file = data_root + 'train.json'
    val_json_file = data_root + 'val.json'
    test_json_file = data_root + 'test.json'

    MODEL_NAME = 'model2'
    SCORE_THRESH_TEST = 50

    convert_coco_json(val_json_file, f'/home/david/code/phawk/detectron/data/T/models/{MODEL_NAME}/eval_val_{SCORE_THRESH_TEST}/')
    convert_coco_json(test_json_file, f'/home/david/code/phawk/detectron/data/T/models/{MODEL_NAME}/eval_test_{SCORE_THRESH_TEST}/')