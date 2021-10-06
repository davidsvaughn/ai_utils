#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 17:36:38 2021

@author: kostya
"""

'''
This is just an example script I downloaded from hasty.ai
apparently it shows how to use the API to import labels.
I'm planning to use this as a starting point.
-dvaughn
'''

import json

from PIL import Image, ImageDraw
# install hasty library, pip install hasty
from hasty import Client
import numpy as np
import pycocotools.mask as maskUtils
from tqdm import tqdm

# PASTE YOUR KEY here
API_KEY = "lHltDKNGzok0GXGDhP6ptYPKm2TM1lk1HE....."

# Read COCO file
coco_file = '/Users/kostya/Documents/Hasty/Datasets/COCO/annotations/instances_train2017.json'
coco_data = json.load(open(coco_file))

# Create an instance of hasty helper
h = Client(api_key=API_KEY)


# Create a new project, or retrieve an existed one
project_id = None
if project_id is None: 
    workspaces = h.get_workspaces()
    workspace = workspaces[0]
    hasty_project = h.create_project(workspace, coco_data['info']['description'])
else:
    hasty_project = h.get_project(project_id)


# Add classes
class_mapping = {}
for c in coco_data['categories']:
    if c["id"] in class_mapping:
        continue
    new_c = hasty_project.create_label_class(name=c["name"])
    class_mapping[c["id"]] = new_c.id

# Create dataset
dataset = hasty_project.create_dataset("train")

# Upload images
image_mapping = {}
image_sizes = {}
for i in tqdm(coco_data["images"]):
    image_sizes[i['id']] = i['height'], i['width']
    if i['id'] in image_mapping:
        continue
    new_image = hasty_project.upload_from_url(dataset, i['file_name'], i['coco_url'])
    image_mapping[i['id']] = new_image.id
    


def polygons2mask(polygons):
    minX = min([p[0] for polygon in polygons for p in polygon])  
    maxX = max([p[0] for polygon in polygons for p in polygon])
    minY = min([p[1] for polygon in polygons for p in polygon]) 
    maxY = max([p[1] for polygon in polygons for p in polygon])

    img = Image.new('L', (maxX - minX, maxY - minY), 0)
    for polygon in polygons:
        ImageDraw.Draw(img).polygon([(p[0] - minX, p[1] - minY) for p in polygon], outline=1, fill=1)
    mask = np.array(img)
    bbox = [minX, minY, maxX, maxY]
    return bbox, mask

def rle_encoding(x, transpose=False):
    """
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    """
    if transpose:
        dots = np.where(x.T.flatten() == 1)[0]  # Order down-then-right
    else:
        dots = np.where(x.flatten() == 1)[0]  # Order right-then-down
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def coco_poly2hasty(segmentaions):
    polygons = []
    for segm in segmentaions:
        poly = []
        for i in range(int(len(segm)/2)):
            poly.append([int(segm[2*i]), int(segm[2*i+1])])
        polygons.append(poly)
    return polygons


# preprocess annotations
coco_images = {}
for l in tqdm(coco_data['annotations']):
    image_id = l["image_id"]
    if image_id not in coco_images:
        coco_images[image_id] = []
    class_id = class_mapping[l["category_id"]]
    if type(l['segmentation']) == list:
        if len(l["segmentation"]) != 1:
            polies = coco_poly2hasty(l["segmentation"])
            bbox, mask = polygons2mask(polies)
            rle_mask = [int(r) for r in rle_encoding(mask)]
            coco_images[image_id].append({"class_id": class_id,
                                          "bbox": bbox,
                                          "mask": rle_mask})
        else:
            segm = l["segmentation"][0]
            polygon = [[int(segm[j*2]), int(segm[j*2+1])] for j in range(int(len(segm)/2))]
            coco_images[image_id].append({"class_id": class_id,
                                          "polygon": polygon})
    else:
        height, width = image_sizes[i['id']]
        if type(l['segmentation']['counts']) == list:
            rle = maskUtils.frPyObjects([l['segmentation']], height, width)
        else:
            rle = [l['segmentation']]
        m = maskUtils.decode(rle)
        bbox = l['bbox']
        rle_mask = [int(r) for r in rle_encoding(m[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]])]
        
        
        coco_images[image_id].append({"mask": rle_mask,
                                      "bbox": bbox,
                                      "class_id": class_id})

# import labels to hasty
images_with_labels = []
failed_images = []
for image_id, labels in tqdm(coco_images.items(), total=len(coco_images)):
    try:
        if image_id in images_with_labels:
            continue
        hasty_img_id = image_mapping[image_id]
        h_image = hasty_project.get_image(hasty_img_id)
        h_image.create_labels(labels)
        h_image.set_status("TO REVIEW")
        images_with_labels.append(image_id)
    except:
        failed_images.append(image_id)
