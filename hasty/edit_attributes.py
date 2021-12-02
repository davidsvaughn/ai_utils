#!/usr/bin/env python3
import sys,os
import json
import random
# install hasty library, pip install hasty
from hasty import Client
import numpy as np
# from attrdict import AttrDict as adict
# from PIL import Image, ImageDraw

class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self
        
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

# sys.exit()

# Get project by id
pid = 'e7d489a2-a79f-46a6-b285-4df46238ff21' ## <---- change this to your project id....
proj = h.get_project(pid)
print(proj); print('')

# Get label classes
label_classes = proj.get_label_classes()
[print(f'{lc.id}\t{lc.name}') for lc in label_classes]; print('')

# Get label classes
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
## 'NEW', 'DONE', 'SKIPPED', 'IN PROGRESS', 'TO REVIEW', 'AUTO-LABELLED'
# SRC_STATUS = 'DONE' 
SRC_STATUS = 'IN PROGRESS' 
# DST_STATUS = 'IN PROGRESS'

LAB_ID1 = 'ef970fc5-a315-4a46-9458-8554913581c9' # Fuse_Switch_Porcelain
LAB_ID2 = 'e4904f99-3e70-451e-aa98-c9e7489ff97a' # Fuse_Switch_Polymer
LABEL_IDS = [LAB_ID1, LAB_ID2]

def image_has_label(image, lab_id):
    labels = image.get_labels()
    return np.any([lab.class_id==lab_id for lab in labels])

def filter_labels(image, label_ids):
    labels = image.get_labels()
    return [lab for lab in labels if lab.class_id in label_ids]
            
# Retrieve the list of projects images
images = list(proj.get_images(image_status=SRC_STATUS))
# random.shuffle(images)

att_reset = []
att_reset.append((('open', True), ('fuse_switch_condition', 'open')))
att_reset.append((('threaded', True), ('fuse_switch_condition', 'threaded')))
att_reset.append(('fuse_switch_condition', 'other'))

j = 0
for i,image in enumerate(images):

    if i%10==0: print(f'{i}/{len(images)}')
    
    labs = filter_labels(image, LABEL_IDS)
    if len(labs)==0: 
        continue
    
    for lab in labs:
        
        atts = lab.get_attributes()
        # print(atts)
        
        for L,R in att_reset:
            if not isinstance(L,tuple):
                lab.set_attribute(att_map.ids[L], att_map.vals[L][R])
                break
            y = (x for x in atts if x.id == att_map.ids[L[0]])
            att = next(y, None)
            if att is None: continue
            if att.value == L[1]:
                lab.set_attribute(att_map.ids[R[0]], att_map.vals[R[0]][R[1]])
                break
        