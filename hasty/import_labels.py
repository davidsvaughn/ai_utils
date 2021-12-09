import sys,os
import json
# install hasty library: pip install hasty
from hasty import Client
import numpy as np
import cv2
import ntpath
from glob import glob

'''
Example script to change the status of all images with a particular label...
To allow filtering of images by label in the web interface....

First create API key here....
https://hasty.readthedocs.io/en/latest/quick_start.html#authentication
'''

pid = 'ad88c3e7-aad2-4e2f-a0c3-e78c38845c6f' ## Solar Construction Project id

data_root = '/home/david/code/phawk/data/solar/indivillage/'
directory_images  = data_root + 'images'
directory_labels  = data_root + 'labels'
yolo_classes_file = data_root + 'classes.txt'

## PASTE YOUR KEY here
API_KEY = 'Y5Q61FQ1x4dR31RjT4Kf0_HCuwc9m2SwqdquZVXTIh0xqn6w3DHYEz6IgK5RvFpGSFL27sTjEUQkdLdPWbxyQw'

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

def get_labels(lab_path):
    if not os.path.isfile(lab_path):
        labs = []
    else:
        with open(lab_path,'r') as f:
            labs = [s.strip() for s in f.readlines()]
    return np.array([lab.split(' ') for lab in labs]).astype(np.float32)
    
def xywhn2xyxy(x, w=640, h=640):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2)  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2)  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2)  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2)  # bottom right y
    return y.round().astype(np.int32)

## filter out degenerate label boxes (h,w == 0)
def check(x,y):
    w = x[:,2]-x[:,0]
    h = x[:,3]-x[:,1]
    idx = np.minimum(w,h)>0
    return x[idx],y[idx]

###############################################################################

## Create an instance of hasty helper
h = Client(api_key=API_KEY)

## Get workspaces that the user account can have access to
# print(h.get_workspaces())
## Get Projects
# print(h.get_projects())

## Get project by id
# pid = 'ad88c3e7-aad2-4e2f-a0c3-e78c38845c6f' ## Solar Construction
proj = h.get_project(pid)
print(proj)

## Get yolo class list
with open(yolo_classes_file,'r') as f:
    yolo_classes = [s.strip() for s in f.readlines()]
    
## Get label ids in hasty ... assuming have been manually added to project already...
## ...planning to automate this too, in future....
hasty_classes = proj.get_label_classes()
lab_dict = {lab.name : lab.id for lab in hasty_classes}
[print(f'{lc.id}\t{lc.name}') for lc in hasty_classes];

## Status types are....
## 'NEW', 'DONE', 'SKIPPED', 'IN PROGRESS', 'TO REVIEW', 'AUTO-LABELLED'
SRC_STATUS = None

## Retrieve the list of projects images...
print('fetching images from hasty....') 
images = list(proj.get_images(image_status=SRC_STATUS))
print('...done')

## Loop through local label files, upload labels to hasty...
img_dict = {im.name : im for im in images}
img_files = get_filenames(directory_images)
img_files.sort()
file_number = 0
for filename in img_files:
    file_number+=1
    ############################
    if file_number<815: continue
    ############################
    
    img_path = (os.path.join(directory_images, filename))
    base = os.path.basename(img_path)
    file_name_without_ext = os.path.splitext(base)[0] # name of the file without the extension
    lab_path  = os.path.join(directory_labels, file_name_without_ext+ "." + 'txt')
    img_name = os.path.basename(img_path) # name of the file without the path
    height,width = memoize_img_dim(img_path)
    labs = get_labels(lab_path)
    if len(labs)==0:
        print(f'{img_name} has no labels')
        continue
    
    y,x = labs[:,0].astype(np.int32), xywhn2xyxy(labs[:,1:], width, height)
    ## filter out bad label boxes...
    x,y = check(x,y)
    
    if file_number % 25 == 0: print(f'{file_number}/{len(img_files)}\t{img_name}')
    
    ## get hasty image object
    img = img_dict[img_name]
    
    ## create labels for hasty
    labels = [ {'class_id': lab_dict[yolo_classes[yy]], 'bbox': xx.tolist() } for yy,xx in zip(y,x)]
    
    ## split into chunks - max label batch size is 100...
    n = 100
    chunks = [labels[i:i + n] for i in range(0, len(labels), n)]
    
    ## add labels to hasty
    for chunk in chunks:
        labs = img.create_labels(chunk)

print('Done!')
