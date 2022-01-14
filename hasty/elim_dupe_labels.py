#!/usr/bin/env python3
import sys,os
import json
import random
# pip install hasty
from hasty import Client
import numpy as np


'''
Example script to change the status of all images with a particular label...
To allow filtering of images by label in the web interface....

First create API key here....
https://hasty.readthedocs.io/en/latest/quick_start.html#authentication
'''

## accepts a single box and a list of boxes...
## returns array of iou values between 'box' and all elements in 'boxes'
def _bbox_iou(box, boxes):  
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(box[0], boxes[0])
    yA = np.maximum(box[1], boxes[1])
    xB = np.minimum(box[2], boxes[2])
    yB = np.minimum(box[3], boxes[3])
    
    interW = xB - xA
    interH = yB - yA
    
    # Correction: reject non-overlapping boxes
    z = (interW>0) * (interH>0)
    interArea = z * interW * interH
    
    boxAArea = (box[2] - box[0]) * (box[3] - box[1])
    boxBArea = (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

def bbox_iou(boxes1, boxes2=None):
    if boxes2 is None:
        boxes2 = boxes1.copy()
    M = np.array([_bbox_iou(box1.T, boxes2.T) for box1 in boxes1])
    np.fill_diagonal(M, 0)
    return M

# PASTE YOUR KEY here
API_KEY = 'Y5Q61FQ1x4dR31RjT4Kf0_HCuwc9m2SwqdquZVXTIh0xqn6w3DHYEz6IgK5RvFpGSFL27sTjEUQkdLdPWbxyQw'

# Create an instance of hasty helper
h = Client(api_key=API_KEY)

# Get workspaces that the user account can have access to
print(h.get_workspaces())

# Get Projects
print(h.get_projects())

# Get project by id
# pid = '9f00c529-5486-4277-ad77-5ac2623e9271' ## Solar Construction Project id
pid = 'ac8d612c-da2f-49d6-964e-9d3149d25ff3' ## Solar Construction 2 Project id
proj = h.get_project(pid)
print(proj)

## Get label ids in hasty ...
hasty_classes = proj.get_label_classes()
lab_dict = {lab.id : lab.name for lab in hasty_classes}
[print(f'{lc.id}\t{lc.name}') for lc in hasty_classes];

## Status types are....
NEW  = 'NEW'
DONE = 'DONE'
SKIP = 'SKIPPED'
PROG = 'IN PROGRESS'
REV  = 'TO REVIEW'
AUTO = 'AUTO-LABELLED'

SRC_STATUS = None
# SRC_STATUS = PROG

# Retrieve the list of projects images
images = list(proj.get_images(image_status=SRC_STATUS))
# random.shuffle(images)

j=0
for i,img in enumerate(images):
    # if i<140: continue
    if i%10==0: print(f'{j}/{i}/{len(images)}\t{img.name}')
    
    labels = img.get_labels()
    if len(labels)==0: continue

    boxes = np.array([lab.bbox for lab in labels])
    M = bbox_iou(boxes)
    x,y = np.where(M==1)
    if len(x)==0: continue

    z = [yy for xx,yy in zip(x,y) if xx<yy]
    xids = [labels[j].id for j in z]
    img.delete_labels(xids)
    j+=1
        