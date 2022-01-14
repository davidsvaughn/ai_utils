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

## filter labels exported from hasty (in hasty format) by class, for re-importing

keepers = ['FoundationScrew']


hasty_input_file = '/home/david/code/phawk/data/solar/hasty/all_labels.json'
hasty_output_file = '/home/david/code/phawk/data/solar/hasty/found_screws.json'


with open(hasty_input_file) as f:
    data = json.load(f)
    

for img in data['images']:
    for i in range(len(img['labels'])-1,-1,-1):
        name = img['labels'][i]['class_name']
        if name not in keepers:
            del img['labels'][i]

with open(hasty_output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
