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
API_KEY = 'Y5Q61FQ1x4dR31RjT4Kf0_HCuwc9m2SwqdquZVXTI...'

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
pid = 'e7f563bd-bd81-416a-a12d-dca45a32f9be'
proj = h.get_project(pid)
print(proj)

# Get label classes
label_classes = proj.get_label_classes()
[print(f'{lc.id}\t{lc.name}') for lc in label_classes];

# sys.exit()

# Retrieve the list of projects images
images = list(proj.get_images())

def image_has_label(image, lab_id):
    labels = image.get_labels()
    return np.any([lab.class_id==lab_id for lab in labels])

lab_id = 'cb88e25d-9d38-4c53-87f7-60a87cf69322' # Porcelain_Dead-end_Insulator
# lab_id = '????' # Concrete_Pole

for i,image in enumerate(images):
    # labels = image.get_labels()
    # print(''); [print(lab) for lab in labels];
    
    if i%10==0: print(i)
    filt = image_has_label(image, lab_id)
    
    if filt:
        labels = image.get_labels()
        print(''); [print(lab.class_id) for lab in labels];
        
        ## 'NEW', 'DONE', 'SKIPPED', 'IN PROGRESS', 'TO REVIEW'
        status = 'IN PROGRESS'
        image.set_status(status)
        