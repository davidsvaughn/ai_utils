import os, sys
import json
import cv2
import ntpath
import datetime
import numpy as np
from glob import glob

## PREPARE YOLO DATA FOR IMPORT INTO HASTY

# location of YOLO annotation info.... set these values appropriately....

data_root = '/home/david/code/phawk/data/generic/transmission/nisource/detect/claire/'
yolo_classes_file = data_root + 'classes.txt'
directory_labels  = data_root + 'labels'
# directory_images  = data_root + 'images'
directory_images  = '/home/david/code/phawk/data/generic/transmission/nisource/images'


project_name = 'Transmission NiSource'       ## name of hasty.ai project being imported into...
output_file = 'claire_import.json'    ## json output file name

###############################################################################

jpg,txt = '.jpg','.txt'

def memoize_img_dim(img_file):
    img_dim_file = img_file.replace('.jpg','.npy')
    if os.path.isfile(img_dim_file):
        img_dim = np.load(img_dim_file)
    else:
        img = cv2.imread(img_file)
        img_dim = np.array(img.shape[:2][::-1])
        np.save(img_dim_file, img_dim)
    width,height = img_dim
    return int(height), int(width)

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_filenames(path, ext=jpg):
    pattern = os.path.join(path, f'*{ext}')
    return np.array([path_leaf(f) for f in glob(pattern)])


## initialize json object
json_data = dict()
json_data['project_name']= project_name
json_data['create_date']= datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
json_data['export_format_version']= '1.1'
json_data['attributes']= []

# add label classes
with open(yolo_classes_file,'r') as f:
    classes = [s.strip() for s in f.readlines()]
label_classes = []
for i,label in enumerate(classes):
    label_classes.append({'class_name':label, 'class_type': 'object' })
    # label_classes.append({'class_name':label, 'class_type': 'object', 'attributes': []})
    # label_classes.append({'class_name':label, 'class_type': 'object', 'norder':i+1, 'attributes': []})
json_data['label_classes'] = label_classes

# loop over images/labels
json_data['images'] = []
file_number = 0
img_files = get_filenames(directory_images)
for filename in img_files:
    file_number = file_number+1
    
    img_path = (os.path.join(directory_images, filename))
    base = os.path.basename(img_path)
    file_name_without_ext = os.path.splitext(base)[0] # name of the file without the extension
    yolo_annotation_path  = os.path.join(directory_labels, file_name_without_ext+ "." + 'txt')
    
    if file_number % 100 == 0:
        print(file_number)
    
    img_name = os.path.basename(img_path) # name of the file without the extension
    img_dict = {}
    
    # height,width = cv2.imread(img_path).shape[:2]
    height,width = memoize_img_dim(img_path) ## this is MUCH faster if you run multiple times.....
        
    img_dict['image_name'] = img_name
    img_dict['height'] = height
    img_dict['width'] = width
    img_dict['dataset_name'] = 'Default'
    img_dict['tags'] = []
    
    if not os.path.isfile(yolo_annotation_path):
        labs = []
    else:
        with open(yolo_annotation_path,'r') as f2:
            labs = f2.readlines() 
        
    labels = []
    for i,line in enumerate(labs): # for loop runs for number of annotations labelled in an image
        line = line.split(' ')
        bbox_dict = {}
        class_id, x_yolo,y_yolo,width_yolo,height_yolo= line[0:5]
        x_yolo,y_yolo,width_yolo,height_yolo,class_id= float(x_yolo),float(y_yolo),float(width_yolo),float(height_yolo),int(class_id)
        h,w = abs(height_yolo*height),abs(width_yolo*width)
        x1 = int(round(x_yolo*width -(w/2)))
        y1 = int(round(y_yolo*height -(h/2)))
        x2 = int(round(x_yolo*width +(w/2)))
        y2 = int(round(y_yolo*height +(h/2)))
        x1,y1 = max(x1,0),max(y1,0)
        x2,y2 = min(x2,width), min(y2,height)
        bbox_dict['class_name'] = classes[int(class_id)]
        bbox_dict['attributes'] = {}
        bbox_dict['bbox'] = [x1,y1,x2,y2]
        labels.append(bbox_dict)
    
    img_dict['labels'] = labels.copy()
    json_data['images'].append(img_dict)
    
    ## testing....
    # if file_number > 20: break

## save/print ##########################

## print to screen for testing.....
# print(json.dumps(json_data, indent=2))

json_save_path = data_root + output_file
with open(json_save_path,'w') as fw:
   json.dump(json_data, fw, indent=2)
print('Done.')
   
