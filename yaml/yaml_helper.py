import os,sys
import numpy as np
import ntpath
from glob import glob
from shutil import copyfile
import yaml

jpg,txt = '.jpg','.txt'

dst_path = '/home/david/code/phawk/data/generic/transmission/rgb/damage/models/model1/'

###############################################################################

class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self
        
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_filenames(path, ext=jpg):
    pattern = os.path.join(path, f'*{ext}')
    return [path_leaf(f) for f in glob(pattern)]

def read_lines(fn):
    with open(fn, 'r') as f:
        lines = [n.strip() for n in f.readlines()]
    return [line for line in lines if len(line)>0]

def write_lines(lines, fn):
    with open(fn, 'w') as f:
        for line in lines:
            f.write("%s\n" % line)

def save_yaml(d, fn, sort_keys=False):
    with open(fn, 'w') as f:
        yaml.dump(d, f, sort_keys=sort_keys)
  
def load_yaml(fn):
    with open(fn, 'r') as f:
        return yaml.full_load(f)

###############################################################################
## Master
CFG, FN = [], 'config.yaml'

## load ? 
# CFG = load_yaml(dst_path+FN)
# print(CFG)
# sys.exit()

###############################################################################
## Component

name = 'component'
src_path = '/home/david/code/phawk/data/generic/transmission/rgb/master/model/model5/'
weights = 'weights.pt'
classes = 'classes.txt'

cfg = adict()
cfg.model = name
cfg.weights = f'{name}.pt'
cfg.model_type    = 'yolo'
cfg.scale   = 3008
cfg.classes = read_lines(src_path + classes)
conf    = [0.5,0.05,0.15,0.5,0.5,0.1,0.02,0.1,0.5,0.1,0.5,0.5,0.5,0.1,0.25]
cfg.conf_thres = {k:v for k,v in zip(cfg.classes, conf)}

copyfile(src_path+weights, dst_path+cfg.weights)
cfg = dict(cfg)
CFG.append(cfg)

fn = f'{name}.yaml'
save_yaml(cfg, dst_path+fn)

###########################################
## Insulator_Damage

name = 'insulator_damage'
src_path = '/home/david/code/phawk/data/generic/transmission/rgb/damage/insulator_damage/models/model1/'
weights = 'weights.pt'
classes = 'classes.txt'

cfg = adict()
cfg.model = name
cfg.weights = f'{name}.pt'
cfg.model_type    = 'yolo'
cfg.label_type    = 'condition'
cfg.scale   = 1280
cfg.classes = read_lines(src_path + classes)
cfg.input_classes = ['Insulator', 'Dead-end Insulator']
conf    = [0.25,0.15]
cfg.conf_thres = {k:v for k,v in zip(cfg.classes, conf)}

copyfile(src_path+weights, dst_path+cfg.weights)
cfg = dict(cfg)
CFG.append(cfg)

fn = f'{name}.yaml'
save_yaml(cfg, dst_path+fn)

###########################################
## Insulator_Material

name = 'insulator_material'
src_path = '/home/david/code/phawk/data/generic/transmission/rgb/master/attribs/models/Insulator_Material/model5/'
weights = 'Insulator_Material.pt'

cfg = adict()
cfg.model = name
cfg.weights = f'{name}.pt'
cfg.classes = ['Glass','Polymer','Porcelain']
cfg.input_classes = ['Insulator', 'Dead-end Insulator']
cfg.model_type    = 'resnet34'
cfg.label_type    = 'attribute'
cfg.scale   = 480
cfg.fc      = 64
mu      = [0.485, 0.456, 0.406]
sig     = [0.229, 0.224, 0.225]
cfg.norm = {'mean':mu, 'std':sig}


copyfile(src_path+weights, dst_path+cfg.weights)
cfg = dict(cfg)
CFG.append(cfg)

fn = f'{name}.yaml'
save_yaml(cfg, dst_path+fn)

###########################################
## Insulator_Type

name = 'insulator_type'
src_path = '/home/david/code/phawk/data/generic/transmission/rgb/master/attribs/models/Insulator_Type/model3/'
weights = 'Insulator_Type.pt'

cfg = adict()
cfg.model = name
cfg.weights = f'{name}.pt'
cfg.classes = ['Pin','Post','Strain']
cfg.input_classes = ['Insulator', 'Dead-end Insulator']
cfg.model_type    = 'resnet34'
cfg.label_type    = 'attribute'
cfg.scale   = 480
cfg.fc      = 64
mu      = [0.485, 0.456, 0.406]
sig     = [0.229, 0.224, 0.225]
cfg.norm = {'mean':mu, 'std':sig}

copyfile(src_path+weights, dst_path+cfg.weights)
cfg = dict(cfg)
CFG.append(cfg)

fn = f'{name}.yaml'
save_yaml(cfg, dst_path+fn)

###########################################
## Wood_Damage

name = 'wood_damage'
src_path = '/home/david/code/phawk/data/generic/transmission/rgb/damage/wood_damage/models/model2/'
weights = 'weights.pt'
classes = 'classes.txt'

cfg = adict()
cfg.model = name
cfg.weights = f'{name}.pt'
cfg.model_type    = 'yolo'
cfg.label_type    = 'condition'
cfg.scale   = 2048
cfg.classes = read_lines(src_path + classes)
cfg.input_classes = ['Wood Pole', 'Wood Crossarm']
conf    = [0.55,0.15,0.1]
cfg.conf_thres = {k:v for k,v in zip(cfg.classes, conf)}

copyfile(src_path+weights, dst_path+cfg.weights)
cfg = dict(cfg)
CFG.append(cfg)

fn = f'{name}.yaml'
save_yaml(cfg, dst_path+fn)

###########################################
## Wood_Deteriorated

name = 'wood_deteriorated'
# src_path = '/home/david/code/phawk/data/generic/transmission/rgb/damage/wood_damage/tags/models/Deteriorated_6/'
src_path = '/home/david/code/phawk/data/generic/transmission/rgb/damage/wood_damage/tags/models/Deteriorated_7/'
weights = 'Deteriorated.pt'

cfg = adict()
cfg.model = name
cfg.weights = f'{name}.pt'
cfg.classes = ['Normal','Deteriorated']
cfg.input_classes = ['Wood Pole', 'Wood Crossarm']
cfg.model_type    = 'resnet50'
cfg.label_type    = 'condition'
cfg.scale   = 1024
cfg.fc      = 64
# cfg.conf_thres = 0.45
mu      = [0.485, 0.456, 0.406]
sig     = [0.229, 0.224, 0.225]
cfg.norm = {'mean':mu, 'std':sig}

copyfile(src_path+weights, dst_path+cfg.weights)
cfg = dict(cfg)
CFG.append(cfg)

fn = f'{name}.yaml'
save_yaml(cfg, dst_path+fn)

###########################################
## Wood_Vegetation

name = 'wood_vegetation'
src_path = '/home/david/code/phawk/data/generic/transmission/rgb/damage/wood_damage/tags/models/Vegetation_4/'
weights = 'Vegetation.pt'

cfg = adict()
cfg.model = name
cfg.weights = f'{name}.pt'
cfg.classes = ['Normal','Vegetation']
cfg.input_classes = ['Wood Pole', 'Wood Crossarm']
cfg.model_type    = 'resnet34'
cfg.label_type    = 'condition'
cfg.scale   = 1024
cfg.fc      = 64
# cfg.conf_thres = 0.7
mu      = [0.485, 0.456, 0.406]
sig     = [0.229, 0.224, 0.225]
cfg.norm = {'mean':mu, 'std':sig}

copyfile(src_path+weights, dst_path+cfg.weights)
cfg = dict(cfg)
CFG.append(cfg)

fn = f'{name}.yaml'
save_yaml(cfg, dst_path+fn)

###############################################################################

save_yaml(CFG, dst_path+FN)

CFG = load_yaml(dst_path+FN)
print(CFG)
    