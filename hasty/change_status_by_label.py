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
pid = 'e7d489a2-a79f-46a6-b285-4df46238ff21'
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

ID1 = 'ef970fc5-a315-4a46-9458-8554913581c9' # Fuse_Switch_Porcelain
ID2 = 'e4904f99-3e70-451e-aa98-c9e7489ff97a' # Fuse_Switch_Polymer
ID3 = 'cb88e25d-9d38-4c53-87f7-60a87cf69322' # Porcelain_Dead-end_Insulator
ID4 = 'dac2a2d8-a9ae-42e4-a6e3-3bd9c6e55207' # Porcelain_Insulator
ID5 = '6ef2859e-e08c-4693-ac16-eb2248c9deb0' # Surge_Arrester

LABEL_IDS = [ID3, ID4]

## Status types are....
## 'NEW', 'DONE', 'SKIPPED', 'IN PROGRESS', 'TO REVIEW', 'AUTO-LABELLED'
SRC_STATUS = 'NEW' 
DST_STATUS = 'IN PROGRESS'

# Retrieve the list of projects images
images = list(proj.get_images(image_status=SRC_STATUS))

j=0
for i,image in enumerate(images):
    
    if i%10==0: print(f'{j}/{i}/{len(images)}')
    
    labs = filter_labels(image, LABEL_IDS)
    if len(labs)==0: continue
    j+=1
    image.set_status(DST_STATUS)
        