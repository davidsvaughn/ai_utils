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

# PASTE YOUR KEY here
API_KEY = 'Y5Q61FQ1x4dR31RjT4Kf0_HCuwc9m2SwqdquZVXTIh0xqn6w3DHYEz6IgK5RvFpGSFL27sTjEUQkdLdPWbxyQw'

# Create an instance of hasty helper
h = Client(api_key=API_KEY)

# Get workspaces that the user account can have access to
print(h.get_workspaces())

# Get Projects
print(h.get_projects())

# Get project by id
pid = 'ad88c3e7-aad2-4e2f-a0c3-e78c38845c6f' ## Solar Construction Project id
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
# SRC_STATUS = NEW
##
# DST_STATUS = PROG
DST_STATUS = SKIP

# Retrieve the list of projects images
images = list(proj.get_images(image_status=SRC_STATUS))
# random.shuffle(images)

def label_counts(labels, lab_dict):
    cnt = { name:0 for name in lab_dict.values() }
    for lab in labels:
        cnt[lab_dict[lab.class_id]]+=1
    return cnt

LABEL_CLASS = 'Modules'
LABEL_COUNT = 400

j=0
for i,img in enumerate(images):
    if i<140: continue
    if i%10==0: print(f'{j}/{i}/{len(images)}\t{img.name}')
    
    if img.name[:4] in ['Clay', 'Nutm', 'Sear', 'Site']:
        img.set_status(SKIP)
        continue
    
    labels = img.get_labels()
    cnt = label_counts(labels, lab_dict)
    if cnt[LABEL_CLASS] >= LABEL_COUNT:
        img.set_status(REV)
        continue
    
    img.set_status(DONE)
    j+=1
        