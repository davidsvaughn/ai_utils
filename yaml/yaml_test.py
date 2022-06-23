import os,sys
import numpy as np
import ntpath
from glob import glob
from shutil import copyfile
import yaml
import inspect

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

def save_yaml(d, fn):
    with open(fn, 'w') as f:
        yaml.dump(d, f)
  
def load_yaml(fn):
    with open(fn, 'r') as f:
        d = yaml.full_load(f)
    return d

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

##################################################

yf = dst_path + 'component.yaml'
cfg = load_yaml(yf)

