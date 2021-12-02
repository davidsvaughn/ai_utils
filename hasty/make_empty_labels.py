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

txt,jpg,JPG = '.txt','.jpg','.JPG'

# PASTE YOUR KEY here
API_KEY = 'Y5Q61FQ1x4dR31RjT4Kf0_HCuwc9m2SwqdquZVXTIh0xqn6w3DHYEz6IgK5RvFpGSFL27sTjEUQkdLdPWbxyQw'


def empty_file(fn):
    with open(fn, 'w') as f:
        f.write('')
            
# Create an instance of hasty helper
h = Client(api_key=API_KEY)

# Get workspaces that the user account can have access to
print(h.get_workspaces())

# Get Projects
print(h.get_projects())

# Get project by id
pid = '77ac6463-fcf4-4f82-839c-7a159b5495d2'
proj = h.get_project(pid)
print(proj)

## Status types are....
## 'NEW', 'DONE', 'SKIPPED', 'IN PROGRESS', 'TO REVIEW', 'AUTO-LABELLED'
SRC_STATUS = ['DONE', 'TO REVIEW'] 

# Retrieve the list of projects images
images = list(proj.get_images(image_status=SRC_STATUS))

label_path = '/home/david/code/phawk/data/generic/labels/'

j=0
for i,image in enumerate(images):
    
    if i%10==0: print(f'{j}/{i}/{len(images)}')
    
    labs = image.get_labels()
    if len(labs)==0:
        j+=1
        empty_file( label_path + image.name.replace(jpg,txt).replace(JPG,txt) )