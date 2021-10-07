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
API_KEY = 'Y5Q61FQ1x4dR31RjT4Kf0_HCuwc9m2Swqdqu...'

# Create an instance of hasty helper
h = Client(api_key=API_KEY)

# Get workspaces that the user account can have access to
print(h.get_workspaces())

# Get Projects
print(h.get_projects())

# Get project by id
pid = 'e7f563bd-bd81-416a-a12d-dca45a32f9be' ## <---- change this to your project id....
proj = h.get_project(pid)
print(proj)


## Status types are....
## 'NEW', 'DONE', 'SKIPPED', 'IN PROGRESS', 'TO REVIEW', 'AUTO-LABELLED'
SRC_STATUS = 'TO REVIEW' 
DST_STATUS = 'NEW'


# Retrieve the list of projects images
images = proj.get_images(image_status=SRC_STATUS)

for i,image in enumerate(images):

    if i%10==0: print(f'{i}/{len(images)}')
    
    image.set_status(DST_STATUS)
    
    if i>50: break
        