import os, sys
import json
import cv2
import ntpath
import datetime
import numpy as np
from glob import glob
import pandas as pd
import shutil
from pathlib import Path


## For importing annotations into hasty exported from PA...


jpg,txt = '.jpg','.txt'

root = '/home/david/code/phawk/data/generic/transmission/nisource/'
src = root + 'nisource_labels.txt'
classes_file = root + 'classes.txt'
data_root = root + 'images/'

project_name = 'Transmission Damage'       ## name of hasty.ai project being imported into...
output_file = root + 'transdam_import.json'    ## json output file name


def read_lines(fn):
    with open(fn, 'r') as f:
        lines = [n.strip() for n in f.readlines()]
    return [line for line in lines if len(line)>0]

def write_lines(fn, lines):
    with open(fn, 'w') as f:
        for line in lines:
            f.write(f'{line}\n')

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_filenames(path, ext=jpg):
    pattern = os.path.join(path, f'*{ext}')
    return np.array([path_leaf(f) for f in glob(pattern)])

def memoize_img_dim(img_file):
    img_dim_file = img_file.replace('.jpg','.npy')
    if os.path.isfile(img_dim_file):
        img_dim = np.load(img_dim_file)
    else:
        img = cv2.imread(img_file)
        img_dim = np.array(img.shape[:2][::-1])
        np.save(img_dim_file, img_dim)
    width,height = img_dim
    return int(height), int(width)

## initialize json object
json_data = dict()
json_data['project_name']= project_name
json_data['create_date']= datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
json_data['export_format_version']= '1.1'

## attributes #################################################################
attribs = []

att = dict()
att['name'] = 'Insulator Condition'
att['type'] = 'SELECTION'
att['values'] = ['Damaged', 'Flashed', 'Misaligned', 'Rusted']
attribs.append(att)

att = dict()
att['name'] = 'Wood Condition'
att['type'] = 'SELECTION'
att['values'] = ['Damaged', 'Deteriorated', 'Cracked', 'Twisted', 'Vegetation']
attribs.append(att)

att = dict()
att['name'] = 'Metal Condition'
att['type'] = 'SELECTION'
att['values'] = ['Damaged', 'Deteriorated', 'Rusted']
attribs.append(att)

att = dict()
att['name'] = 'General Condition'
att['type'] = 'SELECTION'
att['values'] = ['Damaged']
attribs.append(att)

json_data['attributes']= attribs

## add label classes ##########################################################
with open(classes_file,'r') as f:
    classes = [s.strip() for s in f.readlines()]
classes.sort()
label_classes = []
label_attribs = {}
for i,label in enumerate(classes):
    att = None
    if label.startswith('Wood'):
        att = 'Wood Condition'
    elif 'Insulator' in label or 'Surge' in label:
        att = 'Insulator Condition'
    elif 'Ground' in label or 'Raptor' in label or 'Riser' in label or 'Signage' in label:
        att = 'General Condition'
    else:
        att = 'Metal Condition'
    label_attribs[label] = att
    label_classes.append({'class_name':label, 'class_type': 'object', 'attributes': [att], 'norder':i+1})
json_data['label_classes'] = label_classes

# loop over images/labels
json_data['images'] = []
file_number = 0
fns, uids, labels = [],[],[]
img_dict, last_uid = None,''

lines = read_lines(src)[1:]
lines.sort()
for j,line in enumerate(lines):
    
    # if j % 25 == 0: print(j)
        
    toks = line.split(',')
    uid = toks[0].strip("\"")
    img_name = uid + jpg
    img_path = data_root + img_name
    if uid != last_uid:
        
        if len(last_uid)>0:
            img_dict['labels'] = labels.copy()
            json_data['images'].append(img_dict)
        
        last_uid = uid
        img_dict = {}
        labels = []
        height,width = memoize_img_dim(img_path) ## this is MUCH faster if you run multiple times.....
        img_dict['image_name'] = img_name
        img_dict['height'] = height
        img_dict['width'] = width
        img_dict['dataset_name'] = 'Default'
        img_dict['tags'] = []
        
    comp = toks[2].strip().strip("\"")
    cond = toks[1].strip().strip("\"")
    label = comp
    attribs = {}
    if len(cond)>0:
        if label.startswith('Porc'):
            continue
        attribs[label_attribs[label]] = cond
        # cond = cond.replace('/ ','/')
        # label = label + ' - ' + cond
    else: # SKIP if no condition!
        continue
    
    # print(label)
    
    m = ','.join(toks[4:])
    m = m.replace("\"\"","\"").strip("\"")
    d = json.loads(m)
    xs = [dd['x'] for dd in d]
    ys = [dd['y'] for dd in d]
    xs.sort()
    ys.sort()
    x1,x2 = np.round(xs[0]),np.round(xs[-1])
    y1,y2 = np.round(ys[0]),np.round(ys[-1])
    x1,y1 = max(x1,0),max(y1,0)
    x2,y2 = min(x2,width), min(y2,height)
        
    bbox_dict = {}
    bbox_dict['class_name'] = label
    bbox_dict['attributes'] = attribs
    bbox_dict['bbox'] = [int(x1),int(y1),int(x2),int(y2)]
    labels.append(bbox_dict)
    
    if x1==x2 or y1==y2:
        print(img_name)
        print(f"   {bbox_dict['bbox']}")
    
    # if x2>width or y2>height:
    #     print(img_name)
    #     print(f"   {bbox_dict['bbox']}")
    #     print(f'   {width} , {height}')
    #     # sys.exit()
        
##############
## run once.... to get labels.....
# labels.sort()
# labels = np.unique(labels)
# write_lines(classes_file, labels)

## last one...
img_dict['labels'] = labels.copy()
json_data['images'].append(img_dict)

## save/print ##########################

## print to screen for testing.....
# print(json.dumps(json_data, indent=2))

# sys.exit()

json_save_path = output_file
with open(json_save_path,'w') as fw:
   json.dump(json_data, fw, indent=2)
print('Done.')
