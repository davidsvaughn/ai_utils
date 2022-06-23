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

# p = '/home/david/code/phawk/data/generic/transmission/rgb/damage/models/model1/'
p = '/home/david/code/phawk/data/generic/transmission/rgb/damage/models/model1/yaml/'

f_out = 'config.yaml'

yfiles = get_filenames(p, ext='.yaml')
yfiles.sort()
if f_out in yfiles:
    yfiles.remove(f_out)

CFG = []
for yfile in yfiles:
    cfg = adict(load_yaml(p+yfile))
    if 'input_classes' in cfg:
        CFG.append(cfg)
    else:
        cfg1 = cfg

classes = cfg1.filter_classes if 'filter_classes' in cfg1 else cfg1.classes
attrib_map = {s:[] for s in classes}
for cfg in CFG:
    if cfg.label_type == 'attribute':
        for ic in cfg.input_classes:
            if ic in classes:
                attrib_map[ic].append(cfg.classes)
                
condish_map = {s:set() for s in classes}
# condish_map = {s:set(['Not Normal']) for s in classes}
for cfg in CFG:
    if cfg.label_type == 'condition':
        for ic in cfg.input_classes:
            if ic in classes:
                condish_map[ic].update(cfg.classes)

new_classes = []
for name in classes:
    X = ['']
    for A in attrib_map[name]:
        Y = []
        for a in A:
            for x in X:
                Y.append(f'{x}{a} ')
        X = Y
    comp = []
    for x in X:
        # new_classes.append(f'{x}{name}')
        # print(f'{x}{name}')
        comp.append(f'{x}{name}')
    
    for cond in condish_map[name]:
        if cond == 'Normal':
            continue
        for name in comp:
            new_classes.append(f'{name} - {cond}')
        
new_classes.sort()
print(new_classes)
# sys.exit()
        
write_lines(new_classes, p+'classes.txt')

## insert primary stage config at top
CFG.insert(0, cfg1)
CFG = [dict(cfg) for cfg in CFG]

# save_yaml(CFG, p+f_out)
