import os,sys
import numpy as np
import ntpath
from glob import glob
from shutil import copyfile
import yaml

jpg,txt = '.jpg','.txt'

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
## Distribution Component

src_path = '/home/david/code/phawk/data/generic/distribution/models/rgb/component/'
dst_path = src_path

name = 'distribution component'
weights = 'weights.pt'
classes = 'classes.txt'

cfg = adict()
cfg.model = name
cfg.weights = 'weights.pt'
cfg.model_type = 'yolo'
cfg.scale   = 2048
cfg.classes = read_lines(src_path + classes)
conf    = [0.5,0.5,0.25,0.1,0.65,0.3,0.5,0.5,0.5,0.6,0.4,0.4,0.1,0.5,0.5,0.25,0.45,0.6,0.25,0.5,0.6,0.5,0.75,0.6,0.15,0.4]
cfg.conf_thres = {k:v for k,v in zip(cfg.classes, conf)}

# copyfile(src_path+weights, dst_path+cfg.weights)
cfg = dict(cfg)

fn = f'config.yaml'
save_yaml(cfg, dst_path+fn)
