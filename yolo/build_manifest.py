import os, sys
import json, ujson
import random
import numpy as np
import glob
import ntpath
import pandas as pd
from datetime import date
from collections import Counter
from skmultilearn.model_selection import IterativeStratification


''' Set these file paths appropriately... '''

DATA_DIR            = '/home/david/code/phawk/data/generic/transmission/rgb/master/model/large2/'
S3_IMG_BUCKETS      = ['s3://ph-dvaughn-dev/transmission/data/master/images/',]

MANIFEST_FILE       = DATA_DIR + 'manifest.txt'
MANIFEST_STATS_FILE = DATA_DIR + 'manifest_stats.txt'
CATEGORIES_FILE     = DATA_DIR + 'categories.json'
CLASSES_FILE        = DATA_DIR + 'classes.txt'
LABEL_DIRS          = [DATA_DIR + 'labels',]
IMAGE_DIRS          = ['/home/david/code/phawk/data/generic/transmission/rgb/master/images',]

## insulator_damage...
# DATA_DIR            = '/home/david/code/phawk/data/generic/transmission/rgb/damage/insulator_damage/' ## local save directory
# MANIFEST_FILE       = DATA_DIR + 'manifest.txt'
# MANIFEST_STATS_FILE = DATA_DIR + 'manifest_stats.txt'
# CATEGORIES_FILE     = DATA_DIR + 'categories.json'
# CLASSES_FILE        = DATA_DIR + 'classes.txt'
# LABEL_DIRS          = [DATA_DIR + 'labels',]
# S3_IMG_BUCKETS      = ['s3://ai-labeling/transmission/images/insulators/',]

## CLASSES_FILE        = '/home/david/code/phawk/data/solar/indivillage/classes.txt'
## LABEL_DIRS          = ['/home/david/code/phawk/data/solar/indivillage/labels',]

# DATA_DIR            = '/home/david/code/phawk/data/generic/transmission/rgb/damage/wood_damage/' ## local save directory
# S3_IMG_BUCKETS      = ['s3://ai-labeling/transmission/images/wood/',]

# MANIFEST_FILE       = DATA_DIR + 'manifest.txt'
# MANIFEST_STATS_FILE = DATA_DIR + 'manifest_stats.txt'
# CATEGORIES_FILE     = DATA_DIR + 'categories.json'
# CLASSES_FILE        = DATA_DIR + 'classes.txt'
# LABEL_DIRS          = [DATA_DIR + 'labels',]
# IMAGE_DIRS          = [DATA_DIR + 'images',]


'''
TRAIN/TEST/VAL split weightings:
- DON'T need to normalize... just give *relative* weightings ** ( the code will normalize so sum[weights]==1 )
- If TEST or VAL weight==0, then TEST SET == VAL SET
'''
SPLITS = [20,4,0]  ## [TRAIN,TEST,VAL] relative proportions

## make empty label files for images with no label file...
MAKE_EMPTY_LABELS = False
EMPTY_SPLIT = -1
# EMPTY_SPLIT = 0.8

''' ability to filter out certain classes '''
BLACKLIST = None
# BLACKLIST = [2, 3, 5, 7, 15, 17, 19, 22, 23]
    
txt,jpg = '.txt','.jpg'
# jpg = '.JPG'

################################################################

def empty_file(fn):
    with open(fn, 'w') as f:
        f.write('')
    
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def read_lines(fn):
    with open(fn, 'r') as f:
        lines = [n.strip() for n in f.readlines()]
    return [line for line in lines if len(line)>0]

def get_filenames(path, ext=jpg, noext=False):
    pattern = os.path.join(path, f'*{ext}')
    x = np.array([path_leaf(f) for f in glob.glob(pattern)])
    if noext:
        x = np.array([f.replace(ext,'') for f in x])
    return x

def get_labels(fn, blacklist=None):
    with open(fn, 'r') as f:
        lines = f.readlines()
    labs = np.array( [np.array(line.strip().split(' ')).astype(float).round(6) for line in lines])
    if len(labs)==0:
        return labs.tolist()
    
    ## filter out blacklisted classes
    if blacklist is not None and len(blacklist)>0:
        idx = ~np.in1d(labs[:,0], blacklist) ## label rows that are NOT in blacklist
        labs = labs[idx]
        
    labs[:,1:] = labs[:,1:].clip(0,1)
    labs = [[int(x[0])] + x[1:] for x in labs.tolist()]
    return labs

def make_empty_labels():
    if not MAKE_EMPTY_LABELS:
        return
    for lab_dir,img_dir in zip(LABEL_DIRS, IMAGE_DIRS):
        img_files = get_filenames(img_dir, jpg, True)
        lab_files = get_filenames(lab_dir, txt, True)
        idx = ~np.in1d(img_files, lab_files)
        files = img_files[idx]
        for fn in files:
            empty_file(os.path.join(lab_dir, f'{fn}{txt}'))

## lab_files: yolo label files
def label_matrix(lab_files, dim=None):
    Y = []
    for f in lab_files:
        labs = get_labels(f)
        Y.append([y[0] for y in labs])
    if dim is None:
        n = max([max(y) for y in Y if len(y)>0]) + 1
    else:
        n = dim
    Z = []
    for y in Y:
        z = np.zeros(n)
        for k,v in Counter(y).items():
            z[k] = v
        Z.append(z)
    return np.array(Z)

## perform binary (2-way) stratified split on multilabel data...
def binary_split(X, Y, split, order=2):
    # global set1, set2
    split = np.array(split/split.sum())
    strat = IterativeStratification(order=order, n_splits=len(split), sample_distribution_per_fold=split.tolist())
    idx1, idx2 = next(strat.split(X, Y))
    ## switch if out of order with split...
    if np.sign(split[0]-split[1]) != np.sign(len(idx1)-len(idx2)):
        idx1, idx2 = idx2, idx1
    set1 = X[idx1, :], Y[idx1, :]
    set2 = X[idx2, :], Y[idx2, :]
    
    ##############################
    # if EMPTY_SPLIT<=0 or split[0]==0 or split[1]==0:
    #     return set1, set2
    
    ##############################
    ## split empty labels according to EMPTY_SPLIT...
    # X1,Y1 = set1
    # X2,Y2 = set2
    # idx1 = (Y1.sum(1)==0)# set1 empty label ids
    # x1,y1 = X1[idx1],Y1[idx1]# set1 empty labels
    # X1,Y1 = X1[~idx1],Y1[~idx1]
    # idx2 = (Y2.sum(1)==0)# set2 empty label ids
    # x2,y2 = X2[idx2],Y2[idx2]# set2 empty labels
    # X2,Y2 = X2[~idx2],Y2[~idx2]
    # ##
    # xe = np.vstack([x1,x2])# all empty labels
    # ye = np.vstack([y1,y2])
    # ...
    
    # put empty labels in set1.... ie -> train set
    X1,Y1 = set1
    X2,Y2 = set2
    idx = (Y2.sum(1)==0)
    
    x2,y2 = X2[idx],Y2[idx]
    X2,Y2 = X2[~idx],Y2[~idx]
    X1 = np.vstack([X1,x2])
    Y1 = np.vstack([Y1,y2])
    set1 = (X1,Y1)
    set2 = (X2,Y2)
    ##############################################
    return set1, set2
    
def train_test_val_split(X, Y, splits, order=2):
    # global split1, split2
    splits = np.array(splits)
    ## since strat_split only does binary (2-way) split
    ## do it twice to get train/test/val...
    split1 = np.array([splits[:2].sum(), splits[-1]])
    split2 = splits[:2]
    
    # X = X[:,None]
    ## first separate val_set (smallest set) from rest of the data
    data_set, val_set = binary_split(X, Y, split=split1, order=order)
    
    ## then separate data_set into train_set/test_set...
    train_set, test_set = binary_split(*data_set, split=split2, order=order)
    
    return train_set, test_set, val_set

def get_s3_bucket_1(path):
    idx = LABEL_DIRS.index(path)
    s3bucket = S3_IMG_BUCKETS[idx]
    return s3bucket

def build_json_string(label_file, name='train', blacklist=None):
    y = get_labels(label_file, blacklist)
    path, x = ntpath.split(label_file)

    ###############################
    #### sometimes...  .jpg ==> .JPG
    i = LABEL_DIRS.index(path)
    img_path = IMAGE_DIRS[i]
    img_file = os.path.join(img_path, x.replace('.txt','.jpg'))
    # img_file = label_file.replace('/labels/','/images/').replace('.txt','.jpg')
    if not os.path.exists(img_file):
        img_file = img_file.replace('.jpg','.JPG')
    if not os.path.exists(img_file):
        print(f'IMAGE FILE NOT FOUND: {img_file}')
        sys.exit()
    _, imf = ntpath.split(img_file)
    ###############################

    s3bucket = get_s3_bucket_1(path)
    s3url = s3bucket + imf
        
    s = {'s3Url' : s3url,
         'annotations' : y,
         'datasets' : [name]
         }
    return json.dumps(s).replace("/", "\\/").replace(' ','')
    
def build_set(X_files, name='train', blacklist=None):
    entries = []
    for x in X_files.squeeze():
        e = build_json_string(x, name, blacklist)
        entries.append(e)
    return entries

def make_categories():
    if not os.path.exists(CLASSES_FILE):
        print(f'{CLASSES_FILE} not found.  Cannot make categories.json')
        return
    
    classes = np.array(read_lines(CLASSES_FILE))
    categories = []
    class_ids = {}
    
    # Supercategory name.... currently not used, but possible in future....
    supercategory = "Component"
    
    for i, class_name in enumerate(classes):
        categories.append({
            "id": i + 1,
            "name": class_name,
            "supercategory": supercategory
        })
        class_ids[class_name] = i
    today = date.today()
    coco_json = {
        "info": {
            "description": "",
            "url": "",
            "version": "1.0",
            "year": today.strftime("%Y"),
            "contributor": "me",
            "date_created": today.strftime("%Y/%m/%e")
        },
        "licenses": [
            {
                "url": "",
                "id": "12345678",
                "name": "Development"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": categories
    }
    with open(CATEGORIES_FILE, 'w') as f:
        f.write(ujson.dumps(coco_json))

def make_manifest():  
    # global lab_files
        
    ## load all labels
    lab_files = []
    for label_dir in LABEL_DIRS:
        lab_files.extend(glob.glob('{}/*.txt'.format(label_dir)))   
    
    names = np.array(read_lines(CLASSES_FILE))
    
    # random.seed(1234)
    random.shuffle(lab_files)
    X = np.array(lab_files)[:,None]
    Y = label_matrix(lab_files, dim=len(names))
    
    ## filter out any blacklisted classes
    blacklist = None
    if BLACKLIST is not None and len(BLACKLIST)>0:
        blacklist = np.array(BLACKLIST)
        Y[:,blacklist] = 0
        idx = np.where((Y.sum(1)>0))[0]
        X,Y = X[idx], Y[idx]
    
    ## get stratified splits
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = train_test_val_split(X, Y, SPLITS, order=2)
    
    ## fix if empty test/val... (then: test==val)
    if SPLITS[1]==0 or SPLITS[2]==0:
        if len(X_test)==0:
            X_test = X_val
        elif len(X_val)==0:
            X_val = X_test
    
    ## convert to manifest entries
    train_entries = build_set(X_train, name='train', blacklist=blacklist)
    test_entries = build_set(X_test, name='test', blacklist=blacklist)
    val_entries = build_set(X_val, name='val', blacklist=blacklist)
    
    ## save manifest
    with open(MANIFEST_FILE, "w") as f:
        f.writelines('{}\n'.format(s) for s in train_entries)
        f.writelines('{}\n'.format(s) for s in test_entries)
        f.writelines('{}\n'.format(s) for s in val_entries)
    
    ## print stats
    df_image_counts = pd.DataFrame({'TRAIN': y_train.astype(np.bool).sum(0),
                                    'TEST': y_test.astype(np.bool).sum(0),
                                    'VAL': y_val.astype(np.bool).sum(0)}, 
                                   index = names) 
    
    df_target_counts = pd.DataFrame({'TRAIN': y_train.astype(np.int32).sum(0),
                                     'TEST': y_test.astype(np.int32).sum(0),
                                     'VAL': y_val.astype(np.int32).sum(0)}, 
                                    index = names)
    
    original_stdout = sys.stdout
    for i in range(2):
        with open(MANIFEST_STATS_FILE, 'w') as f:
            if i==1: sys.stdout = f
            #####
            print('\nIMAGE COUNTS PER SET/CLASS:')
            print(df_image_counts.to_markdown())
            print('\nTARGET COUNTS PER SET/CLASS:')
            print(df_target_counts.to_markdown())
            print('\nIMAGE COUNTS PER SET:')
            print('TRAIN\t| {}'.format(y_train.shape[0]))
            print('TEST \t| {}'.format(y_test.shape[0]))
            print('VAL  \t| {}'.format(y_val.shape[0]))
            #####
            if i==1: sys.stdout = original_stdout

# ###########################
if __name__ == "__main__":
    make_empty_labels()
    make_manifest()
    make_categories()
    print('Done!')
    