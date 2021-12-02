import os, sys
import json
import cv2
import random
import time
import ntpath
import numpy as np
from glob import glob
from shutil import copyfile

'''
convert YOLO data format --> COCO data format for detectron2 training
'''

### https://github.com/PrabhjotKaurGosal/ObjectDetectionScripts/blob/main/CovertAnnotations_YOLO_to_COCO_format.ipynb

jpg,txt = '.jpg','.txt'

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_filenames(path, ext=jpg):
    pattern = os.path.join(path, f'*{ext}')
    return np.array([path_leaf(f) for f in glob(pattern)])

def read_lines(fn):
    with open(fn, 'r') as f:
        lines = [n.strip() for n in f.readlines()]
    return [line for line in lines if len(line)>0]

def get_img_dim(img_file):
    img_dim_file = img_file.replace('.jpg','.npy')
    if os.path.isfile(img_dim_file):
        img_dim = np.load(img_dim_file)#, allow_pickle=True)
    else:
        img = cv2.imread(img_file)
        img_dim = np.array(img.shape[:2][::-1])
        np.save(img_dim_file, img_dim)
    width,height = img_dim
    return int(width), int(height)


# data_path = '/home/david/code/phawk/data/fpl/damage/rgb/resnet/aitower/labelsT/all_1/'
# class_file = data_path + 'classes.txt'
# img_path   = data_path + 'images/'
# label_path = data_path + 'labels/'
# output_path  = data_path + 'coco/'

data_path = '/home/david/code/phawk/data/generic/damage/'
class_file = data_path + 'classes.txt'
img_path   = data_path + 'images/'
label_path = data_path + 'labels/'
output_path  = data_path + 'coco/'

mkdirs(output_path)

# for dset in ['train']:
for dset in ['test', 'train']:
# for dset in ['val', 'test', 'train']:

    dset_file = data_path + f'{dset}.txt'
    coco_format_save_path = output_path + f'{dset}.json'
    
    #Read the categories file and extract all categories
    with open(class_file,'r') as f1:
        classes = f1.readlines()
    classes = [s.strip() for s in classes]
    categories = []
    for j,label in enumerate(classes):
        label = label.strip()
        categories.append({'id':j+1,'name':label,'supercategory': 'Component'})
        
    write_json_context = dict()
    write_json_context['info'] = {'description': '', 'url': '', 'version': '', 'year': 2021, 'contributor': '', 'date_created': '2021-02-12 11:00:08.5'}
    write_json_context['licenses'] = [{'id': 1, 'name': None, 'url': None}]
    write_json_context['categories'] = categories
    write_json_context['images'] = []
    write_json_context['annotations'] = []
    
    file_number,num_bboxes = 1,1
    image_files = read_lines(dset_file)
    for img_file in image_files:
        
        img_file = ntpath.basename(img_file)
        
        #####################################
        lab_file = img_file.replace(jpg, txt)
        lab_path = label_path + lab_file
        img_context = {}
        
        # height,width = cv2.imread(img_path).shape[:2]
        width,height = get_img_dim(img_path + img_file)
        
        ####
        ## copy image file
        dst_path   = data_path + f'coco/{dset}/'
        src = img_path + img_file
        dst = dst_path + img_file
        copyfile(src, dst)
        ####
        
        img_context['file_name'] = img_file
        img_context['height'] = height
        img_context['width'] = width
        img_context['date_captured'] = '2021-9-29 11:00:08.5'
        img_context['id'] = file_number # image id
        img_context['license'] = 1
        img_context['coco_url'] =''
        img_context['flickr_url'] = ''
        # write_json_context['images'].append(img_context)
        
        with open(lab_path,'r') as f:
            labels = f.readlines() 
    
        bboxes = []
        for i,line in enumerate(labels):
            line = line.split(' ')
            bbox_dict = {}
            class_id, x_yolo,y_yolo,width_yolo,height_yolo= line[0:]
            x_yolo,y_yolo,width_yolo,height_yolo,class_id= float(x_yolo),float(y_yolo),float(width_yolo),float(height_yolo),int(class_id)
            
            bbox_dict['id'] = num_bboxes
            bbox_dict['image_id'] = file_number
            bbox_dict['category_id'] = class_id+1
            bbox_dict['iscrowd'] = 0 # There is an explanation before
            h,w = int(round(abs(height_yolo*height))), int(round(abs(width_yolo*width)))
            bbox_dict['area']  = h * w
            x_coco = int(round(x_yolo*width -(w/2)))
            y_coco = int(round(y_yolo*height -(h/2)))
            if x_coco <0: #check if x_coco extends out of the image boundaries
                x_coco = 1
            if y_coco <0: #check if y_coco extends out of the image boundaries
                y_coco = 1
            bbox_dict['bbox'] = [x_coco,y_coco,w,h]
            bbox_dict['segmentation'] = [[x_coco, y_coco, x_coco+w, y_coco, x_coco+w, y_coco+h, x_coco, y_coco+h]]
            # write_json_context['annotations'].append(bbox_dict)
            bboxes.append(bbox_dict)
            num_bboxes+=1
            
        if len(bboxes)==0:
            continue
            # print(f'NO BBOXES!!!!')
            # sys.exit()
        #####################################################
        
        print(f'{file_number}/{len(image_files)}\t{lab_file}')
        file_number = file_number+1
        # copyfile(img_path, img_dst_path)
        write_json_context['images'].append(img_context)
        write_json_context['annotations'].extend(bboxes)
            
    with open(coco_format_save_path,'w') as fw:
        json.dump(write_json_context, fw, indent=2)
        
    print(f'{file_number}\t\t{num_bboxes}')
