#!/usr/bin/env python3
import sys,os
import json
import random
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

# Get workspaces that the user account can have access to
print(h.get_workspaces())

# Get Projects
print(h.get_projects())

# Get project by id
pid = 'e7d489a2-a79f-46a6-b285-4df46238ff21'
proj = h.get_project(pid)
print(proj)

## Status types are....
## 'NEW', 'DONE', 'SKIPPED', 'IN PROGRESS', 'TO REVIEW', 'AUTO-LABELLED'
SRC_STATUS = 'NEW' 
DST_STATUS = 'IN PROGRESS'

# Retrieve the list of projects images
images = list(proj.get_images(image_status=SRC_STATUS))
random.shuffle(images)

MIN_LABELS = 4
j=0

for i,image in enumerate(images):

    if i%10==0: print(f'{j}/{i}/{len(images)}')
    
    labels = image.get_labels()
    
    if len(labels)>=MIN_LABELS:
        image.set_status(DST_STATUS)
        j+=1
        