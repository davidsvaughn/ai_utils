#!/usr/bin/env python3
import sys,os
import json
import random
# install hasty library, pip install hasty
from hasty import Client
import numpy as np
# from PIL import Image, ImageDraw

class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self
        
'''
Example script to ...

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

# sys.exit()

# Get project by id
pid = 'a55dadc8-1808-47a6-9ba5-dad69723efe5' ## <---- change this to your project id....
proj = h.get_project(pid)
print(proj); print('')

# Get label classes
label_classes = proj.get_label_classes()
[print(f'{lc.id}\t{lc.name}') for lc in label_classes]; print('')

# Get label attributes
attribs = proj.get_attributes()
[print(f'{at.id}\t{at.name}\t{at.values}') for at in attribs]; print('')

att_map = adict()
att_map.ids = adict()
att_map.vals = adict()
for at in attribs:
    # print(f'{at.id}\t{at.name}\t{at.values}')
    att_map.ids[at.name] = at.id
    att_map.vals[at.name] = adict()
    for kv in at.values:
        att_map.vals[at.name][kv['value']] = kv['id']

# Get attributes-->class mapping
att2class = proj.get_attribute_classes()
[print(f'{a2c}') for a2c in att2class]; print('')


## Status types are....
NEW  = 'NEW'
SKIP = 'SKIPPED'
PROG = 'IN PROGRESS'
REV  = 'TO REVIEW'
AUTO = 'AUTO-LABELLED'
DONE = 'DONE'

SRC_STATUS = DONE


LAB_ID1 = '77864178-6ec9-44bc-b445-2f87764aaacb' # Insulator
LAB_ID2 = '9e3e8453-91ec-4133-bc61-77cf857bcab2' # Dead-end Insulator

LABEL_IDS = [LAB_ID1]

def image_has_label(image, lab_id):
    labels = image.get_labels()
    return np.any([lab.class_id==lab_id for lab in labels])

def filter_labels(image, label_ids):
    labels = image.get_labels()
    return [lab for lab in labels if lab.class_id in label_ids]

def label_has_attribute(label, attrib_name, attrib_value):
    atts = label.get_attributes()
    for att in atts:
        if att.id == att_map.ids[attrib_name]:
            if att.value == att_map.vals[attrib_name][attrib_value]:
                return True
    return False

def set_label_attribute(label, attrib_name, attrib_value):
    label.set_attribute(att_map.ids[attrib_name], att_map.vals[attrib_name][attrib_value])
    
            
# Retrieve the list of projects images
images = list(proj.get_images(image_status=SRC_STATUS))
# random.shuffle(images)


ATT1 = ('Insulator Type', 'Strut')
ATT2 = ('Insulator Material', 'Porcelain')

j = 0
for i,image in enumerate(images):

    if i%10==0: print(f'{i}/{len(images)}')
    
    labs = filter_labels(image, LABEL_IDS)
    if len(labs)==0: 
        continue
    
    for lab in labs:
        if label_has_attribute(lab, 'Insulator Type', 'Dead-end'):
            lab.edit(LAB_ID2, lab.bbox)
    
    # for lab in labs:
    #     lab.edit(LAB_ID2, lab.bbox)
    #     set_label_attribute(lab, *ATT1)
    #     set_label_attribute(lab, *ATT2)
