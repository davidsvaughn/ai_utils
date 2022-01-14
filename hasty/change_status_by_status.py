#!/usr/bin/env python3
import sys,os
import json
# pip install hasty
from hasty import Client
import numpy as np

'''
Example script to change the status of all images with a particular status to a new status...

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
# pid = 'ad88c3e7-aad2-4e2f-a0c3-e78c38845c6f' ## Solar Construction
pid = 'ac8d612c-da2f-49d6-964e-9d3149d25ff3' ## Solar Construction 2
proj = h.get_project(pid)
print(proj)

## Status types are....
NEW  = 'NEW'
DONE = 'DONE'
SKIP = 'SKIPPED'
PROG = 'IN PROGRESS'
REV  = 'TO REVIEW'
AUTO = 'AUTO-LABELLED'

# SRC_STATUS = None
SRC_STATUS = [DONE, PROG, REV] 

DST_STATUS = SKIP

# LAB_ID1 = 'ef970fc5-a315-4a46-9458-8554913581c9' # Fuse_Switch_Porcelain
# LAB_ID2 = 'e4904f99-3e70-451e-aa98-c9e7489ff97a' # Fuse_Switch_Polymer

def image_has_label(image, lab_id):
    labels = image.get_labels()
    return np.any([lab.class_id==lab_id for lab in labels])

# Retrieve the list of projects images
images = list(proj.get_images(image_status=SRC_STATUS))

j = 0
for i,image in enumerate(images):
    # print(image.name)

    if i%10==0: print(f'{j}/{i}/{len(images)}')
    
    # if image_has_label(image, LAB_ID1) or image_has_label(image, LAB_ID2):
    # if image.name.startswith('Davey_DJI_0_'):
        
    image.set_status(DST_STATUS)
    j += 1
    # if j>53: break
    
