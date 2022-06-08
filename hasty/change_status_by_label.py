#!/usr/bin/env python3
import sys,os
import json
# install hasty library, pip install hasty
from hasty import Client
import numpy as np
# from PIL import Image, ImageDraw

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

# Create a new project, or retrieve an existed one
# project_id = None
# hasty_project = h.get_project(project_id)

# Get workspaces that the user account can have access to
print(h.get_workspaces())

# Get Projects
print(h.get_projects())

# Get project by id
# pid = 'ad88c3e7-aad2-4e2f-a0c3-e78c38845c6f' ## Solar Construction
# pid = 'ac8d612c-da2f-49d6-964e-9d3149d25ff3' ## Solar Construction 2
# pid = 'a55dadc8-1808-47a6-9ba5-dad69723efe5' ## Transmission Master
# pid = '2da96c3c-0bd6-429b-8076-e5aa53ba7940' ## Insulator Damage
pid = 'bb5e2e2d-0645-4b91-9f19-9e34b0645e4b' ## Wood Damage
# pid = 'bf104acc-44f9-4103-9379-c4096e570f6c' ## Solar Thermal Damage

proj = h.get_project(pid)
print(proj)

# Get label classes
label_classes = proj.get_label_classes()
[print(f'{lc.id}\t{lc.name}') for lc in label_classes];

# sys.exit()

def image_has_label(image, lab_id):
    labels = image.get_labels()
    return np.any([lab.class_id==lab_id for lab in labels])

def filter_labels(image, label_ids):
    labels = image.get_labels()
    return [lab for lab in labels if lab.class_id in label_ids]

ID1 = '6fb4a67b-498b-4d7f-b77a-c267eb699359' # Crack

LABEL_IDS = [ID1]

## Status types are....
NEW  = 'NEW'
SKIP = 'SKIPPED'
PROG = 'IN PROGRESS'
REV  = 'TO REVIEW'
AUTO = 'AUTO-LABELLED'
DONE = 'DONE'

SRC_STATUS = DONE
DST_STATUS = PROG

# Retrieve the list of projects images
images = list(proj.get_images(image_status=SRC_STATUS))

j=0
for i,image in enumerate(images):
    
    if i%10==0: print(f'{j}/{i}/{len(images)}')
    
    labs = filter_labels(image, LABEL_IDS)
    if len(labs)==0: continue
    j+=1
    image.set_status(DST_STATUS)
        