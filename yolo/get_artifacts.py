import os, sys
import json
import random
import glob
import boto3
import ntpath
import numpy as np
import pandas as pd
from concurrent.futures.thread import ThreadPoolExecutor
from json.decoder import JSONDecodeError
from manifest_entry import ManifestEntry
from inference_response import COCOCategory
from collections import Counter

DOWNLOAD_IMAGES = False
# DOWNLOAD_LABELS = False
## sometimes image train set is very large.....
DOWNLOAD_TRAIN = True

'''
Use this file to download saved model artifacts (and data) after a training run... 
It will arrange the files in the same structure as 'training_start.py' script in ai_docker.
'''
''' Set the following 3 file paths appropriately.... This example is for FPL Thermal Damage... '''

## Solar RGB
# DATA_DIR        = '/home/david/code/phawk/data/solar/indivillage'
# MODEL_DIR       = '/home/david/code/phawk/data/solar/indivillage/models/model2'
# MODEL_BUCKET    = 's3://ai-inference-dev-model-catalog/model/yolo-v5-full-scale/solar-construction-2048-3200-jan-14-22-p3dn/'

## Solar Thermal
# DATA_DIR        = '/home/david/code/phawk/data/solar/thermal'
# MODEL_DIR       = '/home/david/code/phawk/data/solar/thermal/models/model1'
# MODEL_BUCKET    = 's3://ai-inference-dev-model-catalog/model/yolo-v5-full-scale/nextera-thermal-1/'

## FPL Damage
# DATA_DIR        = '/home/david/code/phawk/data/generic/distribution/models/rgb/damage'
# # MODEL_DIR       = '/home/david/code/phawk/data/generic/distribution/models/rgb/damage/oct15a'
# # MODEL_BUCKET    = 's3://ai-inference-dev-model-catalog/model/yolo-v5-full-scale/dist-rgb-damage-2048-2560-oct15a/'
# MODEL_DIR       = '/home/david/code/phawk/data/generic/distribution/models/rgb/damage/oct19a'
# MODEL_BUCKET    = 's3://ai-inference-dev-model-catalog/model/yolo-v5-full-scale/generic-rgb-damage-2560-3008-oct19a/'

## Claire Transmission Data
# DATA_DIR        = '/home/david/code/phawk/data/generic/transmission/claire'
# MODEL_DIR       = '/home/david/code/phawk/data/generic/transmission/claire'
# MODEL_BUCKET    = 's3://ai-labeling/transmission/manifests/extracted_from_image_filenames/'
# DOWNLOAD_TRAIN  = True

########################################################
## Transmission ##

# # Transmission Master
# DATA_DIR        = '/home/david/code/phawk/data/generic/transmission/rgb/master'
# MODEL_DIR       = '/home/david/code/phawk/data/generic/transmission/rgb/master/model/model5'
# MODEL_BUCKET    = 's3://ai-inference-dev-model-catalog/model/yolo-v5-full-scale/transmission-master-3008-5a/'

# Transmission Edge-Small
DATA_DIR        = '/home/david/code/phawk/data/generic/transmission/rgb/master'
MODEL_DIR       = '/home/david/code/phawk/data/generic/transmission/rgb/master/model/small2'
MODEL_BUCKET    = 's3://ai-inference-dev-model-catalog/model/yolo-v5-full-scale/transmission-small-720-b/'

## Insulator Damage
# DATA_DIR        = '/home/david/code/phawk/data/generic/transmission/damage/insulator_damage'
# MODEL_DIR       = '/home/david/code/phawk/data/generic/transmission/damage/insulator_damage/models/model1'
# MODEL_BUCKET    = 's3://ai-inference-dev-model-catalog/model/yolo-v5-full-scale/insulator-damage-1280-5m6-freeze3/'

## Wood Damage
# DATA_DIR        = '/home/david/code/phawk/data/generic/transmission/damage/wood_damage'
# MODEL_DIR       = '/home/david/code/phawk/data/generic/transmission/damage/wood_damage/models/model2'
# MODEL_BUCKET    = 's3://ai-inference-dev-model-catalog/model/yolo-v5-full-scale/wood-damage-2048-a/'

## Pseudo-Thermal
# DATA_DIR        = '/home/david/code/phawk/data/generic/transmission/thermal/models/pseudo1'
# MODEL_DIR       = '/home/david/code/phawk/data/generic/transmission/thermal/models/pseudo1'
# MODEL_BUCKET    = 's3://ai-inference-dev-model-catalog/model/yolo-v5-full-scale/transmission-pseudo-thermal-1/'

# Transmission Thermal
# DATA_DIR        = '/home/david/code/phawk/data/generic/transmission/thermal'
# MODEL_DIR       = '/home/david/code/phawk/data/generic/transmission/thermal/models/model1'
# MODEL_BUCKET    = 's3://ai-inference-dev-model-catalog/model/yolo-v5-full-scale/transmission-thermal-test1/'


########################################################
## Solar ##

## Pseudo-Thermal
# DATA_DIR        = '/home/david/code/phawk/data/solar/thermal/component'
# MODEL_DIR       = '/home/david/code/phawk/data/solar/thermal/component/models/model3'
# MODEL_BUCKET    = 's3://ai-inference-dev-model-catalog/model/yolo-v5-full-scale/solar-pseudo-thermal-3/'

## Thermal-Damage
# DATA_DIR        = '/home/david/code/phawk/data/solar/thermal/damage'
# MODEL_DIR       = '/home/david/code/phawk/data/solar/thermal/damage/models/model3'
# MODEL_BUCKET    = 's3://ai-inference-dev-model-catalog/model/yolo-v5-full-scale/solar-thermal-damage-3/'

#-----------------------------------------------
IMG_DIR = '{}/images'.format(DATA_DIR)
LAB_DIR = '{}/labels'.format(DATA_DIR)
# LAB_DIR = '{}/labels'.format(MODEL_DIR)

if MODEL_BUCKET:
    MANIFEST_URL    = MODEL_BUCKET + 'manifest.txt'
    MODEL_WTS_URL   = MODEL_BUCKET + 'weights.pt'
    CFG_URL         = MODEL_BUCKET + 'hyp.yaml'
    CAT_URL         = MODEL_BUCKET + 'categories.json'
    OUTPUT_LOG      = MODEL_BUCKET + 'output.log'
    JOB_JSON        = MODEL_BUCKET + 'job.json'
    TESTLOG_URL     = MODEL_BUCKET + 'test.txt'

#-------------------------------------------------

s3_client = boto3.client('s3')
boto3.setup_default_session(profile_name='ph-ai-dev')  # To switch between different AWS accounts

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

if not os.path.exists(LAB_DIR):
    os.makedirs(LAB_DIR)

def download_s3_file(s3_url, dst_dir=DATA_DIR, filename=None, silent=False, overwrite=False):
    if filename is None:
        filename = s3_url.split("/").pop()
    out_path = os.path.join(dst_dir, filename)
    if os.path.exists(out_path) and not overwrite:
        return out_path 
    try:
        if not silent:
            print(f'Downloading file from: {s3_url} to {out_path}')
        s3_url_components = s3_url.replace("s3://", "")
        bucket = s3_url_components.replace("s3://", "").split("/")[0]
        key = s3_url_components.replace(f'{bucket}/', "")
        with open(out_path, 'wb') as f:
            s3_client.download_fileobj(bucket, key, f)
        return out_path
    except Exception as e:
        # Print the URL of the failed file and re-raise the exception.
        print(f"Failed to download file from S3: {s3_url}")
        # raise e

def parse_manifest(manifest_path):
    # Initialize train/val/test sets.
    train_set = set()
    val_set = set()
    test_set = set()

    # Read in the manifest.
    manifest = []
    with open(manifest_path, 'r') as f:
        try:
            line_number = 0
            line = f.readline()
            while line is not None:
                line_json = json.loads(line)
                entry = ManifestEntry(json=line_json)
                if entry is not None:
                    entry.annotations = line_json['annotations']
                    manifest.append(entry)

                    # Split up the entries into train/val/test sets.
                    if "train" in entry.datasets:
                        train_set.add(entry)
                    if "val" in entry.datasets:
                        val_set.add(entry)
                    if "test" in entry.datasets:
                        test_set.add(entry)

                # Read the next line.
                line_number += 1
                line = f.readline()
        except JSONDecodeError:
            pass  # Do nothing. Reached end of file.
    print(f'Finished reading manifest: {len(manifest)} lines.')
    return list(train_set), list(val_set), list(test_set), manifest

def parse_categories(categories_path):
    categories = []
    with open(categories_path, 'r') as f:
        categories_dict = json.loads(f.read())
    cats = categories_dict['categories']
    cats = sorted(cats, key=lambda i: i['id'])
    for cat in cats:
        categories.append(COCOCategory(json=cat))
    print(f'Finished reading categories: {len(categories)} classes.')
    return categories

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def download_image(entry: ManifestEntry):
    s3_url = entry.s3Url
    return download_s3_file(s3_url, dst_dir=IMG_DIR, silent=True)

def download_images(img_set):
    executor = ThreadPoolExecutor(max_workers=64)
    for entry in img_set:
        executor.submit(download_image, entry)
    executor.shutdown(wait=True)
        
def write_labels(img_set, lab_dir=LAB_DIR, overwrite=True):
    for entry in img_set:
        label_file_path = f'{lab_dir}/{entry.resource_name()}.txt'
        if os.path.exists(label_file_path):
            if overwrite:
                os.remove(label_file_path)
            else:
                continue
        with open(label_file_path, 'a') as f:
            for annotation in entry.annotations:
                class_id, x, y, w, h = annotation
                f.write(f'{class_id} {x} {y} {w} {h}\n') 

def write_set(set_path, img_set):
    if os.path.exists(set_path):
        os.remove(set_path)
    with open(set_path, 'a') as f:
        for entry in img_set:
            f.write(f'{IMG_DIR}/{entry.image_file_name()}\n')
            
def write_class_names(categories, dst_dir=DATA_DIR):
    class_names_filename = f'{dst_dir}/classes.txt'
    if os.path.exists(class_names_filename):
        os.remove(class_names_filename)
    labels = []
    with open(class_names_filename, 'a') as f:
        for category in categories:
            name = category.name
            labels.append(name)
            f.write(f'{name}\n')
    return labels

def write_data_yaml(labels, dst_dir=DATA_DIR):
    data_yaml_filename = f"{dst_dir}/data.yaml"
    if os.path.exists(data_yaml_filename):
        os.remove(data_yaml_filename)
    with open(data_yaml_filename, 'a') as f:
        # f.write(f"train: {DATA_DIR}/train.txt\n")
        f.write(f"val: {dst_dir}/val.txt\n")
        f.write(f"test: {dst_dir}/test.txt\n\n")
        f.write(f'# Number of classes in dataset:\nnc: {len(labels)}\n\n')
        f.write(f'# Class names:\nnames: {json.dumps(labels)}')

## input: manifest entries
def label_matrix(img_set, dim=None):
    Y = []
    for m in img_set:
        labs = m.annotations
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

###################################################################

if __name__ == "__main__":
    
    manifest_path = download_s3_file(MANIFEST_URL, dst_dir=MODEL_DIR, overwrite=False)# True
    train_set, val_set, test_set, all_set = parse_manifest(manifest_path)
    
    ## DOWNLOAD MODEL FILES....
    if CAT_URL:
        categories_path = download_s3_file(CAT_URL, dst_dir=MODEL_DIR, overwrite=False)
        categories = parse_categories(categories_path)
        labels = write_class_names(categories, dst_dir=MODEL_DIR)
        write_data_yaml(labels, dst_dir=MODEL_DIR)
        
    if MODEL_BUCKET:
        model_path = download_s3_file(MODEL_WTS_URL, dst_dir=MODEL_DIR, overwrite=False)# True
        cfg_path = download_s3_file(CFG_URL, dst_dir=MODEL_DIR, overwrite=True)
        download_s3_file(TESTLOG_URL, dst_dir=MODEL_DIR, filename='testlog.txt', overwrite=True)
        download_s3_file(OUTPUT_LOG, dst_dir=MODEL_DIR, filename='output.log', overwrite=True)
        download_s3_file(JOB_JSON, dst_dir=MODEL_DIR, filename='job.json', overwrite=True)
    
    ## DOWNLOAD IMAGES....
    img_sets = []
    img_sets.append((test_set, 'test'))
    img_sets.append((val_set, 'val'))
    if DOWNLOAD_TRAIN:
        img_sets.append((train_set, 'train'))
    
    for img_set in img_sets:
        img_set, name = img_set
        
        if DOWNLOAD_IMAGES:
            print(f'Downloading {name} set... {len(img_set)} images...')
            download_images(img_set)
            print('...done.')
        
        ## filter test manifest entries on downloaded images
        img_files = [path_leaf(f) for f in glob.glob('{}/*.[jJ][pP][gG]'.format(IMG_DIR))]
        img_set = [e for e in img_set if e.image_file_name() in img_files]
        
        ## save image filenames to file
        write_labels(img_set)
        img_set_path = f'{MODEL_DIR}/{name}.txt'
        write_set(img_set_path, img_set)

    ######################################
    ## print stats
    dim = len(labels)
    y_train = label_matrix(train_set, dim)
    y_val = label_matrix(val_set, dim)
    y_test = label_matrix(test_set, dim)
    
    df_image_counts = pd.DataFrame({'TRAIN': y_train.astype(bool).sum(0),
                                    'TEST': y_test.astype(bool).sum(0),
                                    'VAL': y_val.astype(bool).sum(0)}, 
                                   index = labels) 
    
    df_target_counts = pd.DataFrame({'TRAIN': y_train.astype(np.int32).sum(0),
                                     'TEST': y_test.astype(np.int32).sum(0),
                                     'VAL': y_val.astype(np.int32).sum(0)}, 
                                    index = labels)
    
    stats_path = f'{MODEL_DIR}/manifest_stats.txt'
    original_stdout = sys.stdout
    for i in range(2):
        with open(stats_path, 'w') as f:
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