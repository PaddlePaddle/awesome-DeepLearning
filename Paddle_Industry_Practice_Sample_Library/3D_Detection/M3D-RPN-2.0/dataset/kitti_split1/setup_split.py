from importlib import import_module
from getopt import getopt
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import pprint
import sys
import os
import cv2
import math
import shutil
import re

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.util import *

split = 'kitti_split1'

# base paths
base_data = os.path.join(os.getcwd(), 'dataset')

kitti_raw = dict()
kitti_raw['cal'] = os.path.join(base_data, 'kitti', 'training', 'calib')
kitti_raw['ims'] = os.path.join(base_data, 'kitti', 'training', 'image_2')
kitti_raw['lab'] = os.path.join(base_data, 'kitti', 'training', 'label_2')
kitti_raw['pre'] = os.path.join(base_data, 'kitti', 'training', 'prev_2')

kitti_tra = dict()
kitti_tra['cal'] = os.path.join(base_data, split, 'training', 'calib')
kitti_tra['ims'] = os.path.join(base_data, split, 'training', 'image_2')
kitti_tra['lab'] = os.path.join(base_data, split, 'training', 'label_2')
kitti_tra['pre'] = os.path.join(base_data, split, 'training', 'prev_2')

kitti_val = dict()
kitti_val['cal'] = os.path.join(base_data, split, 'validation', 'calib')
kitti_val['ims'] = os.path.join(base_data, split, 'validation', 'image_2')
kitti_val['lab'] = os.path.join(base_data, split, 'validation', 'label_2')
kitti_val['pre'] = os.path.join(base_data, split, 'validation', 'prev_2')

tra_file = os.path.join(base_data, split, 'train.txt')
val_file = os.path.join(base_data, split, 'val.txt')

# mkdirs
mkdir_if_missing(kitti_tra['cal'])
mkdir_if_missing(kitti_tra['ims'])
mkdir_if_missing(kitti_tra['lab'])
mkdir_if_missing(kitti_tra['pre'])
mkdir_if_missing(kitti_val['cal'])
mkdir_if_missing(kitti_val['ims'])
mkdir_if_missing(kitti_val['lab'])
mkdir_if_missing(kitti_val['pre'])


print('Linking train')
text_file = open(tra_file, 'r')

imind = 0

for line in text_file:

    parsed = re.search('(\d+)', line)

    if parsed is not None:

        id = str(parsed[0])
        new_id = '{:06d}'.format(imind)

        if not os.path.exists(os.path.join(kitti_tra['cal'], str(new_id) + '.txt')):
            os.symlink(os.path.join(kitti_raw['cal'], str(id) + '.txt'), os.path.join(kitti_tra['cal'], str(new_id) + '.txt'))

        if not os.path.exists(os.path.join(kitti_tra['ims'], str(new_id) + '.png')):
            os.symlink(os.path.join(kitti_raw['ims'], str(id) + '.png'), os.path.join(kitti_tra['ims'], str(new_id) + '.png'))

        if not os.path.exists(os.path.join(kitti_tra['pre'], str(new_id) + '_01.png')):
            os.symlink(os.path.join(kitti_raw['pre'], str(id) + '_01.png'), os.path.join(kitti_tra['pre'], str(new_id) + '_01.png'))

        if not os.path.exists(os.path.join(kitti_tra['pre'], str(new_id) + '_02.png')):
            os.symlink(os.path.join(kitti_raw['pre'], str(id) + '_02.png'), os.path.join(kitti_tra['pre'], str(new_id) + '_02.png'))

        if not os.path.exists(os.path.join(kitti_tra['pre'], str(new_id) + '_03.png')):
            os.symlink(os.path.join(kitti_raw['pre'], str(id) + '_03.png'), os.path.join(kitti_tra['pre'], str(new_id) + '_03.png'))

        if not os.path.exists(os.path.join(kitti_tra['lab'], str(new_id) + '.txt')):
            os.symlink(os.path.join(kitti_raw['lab'], str(id) + '.txt'), os.path.join(kitti_tra['lab'], str(new_id) + '.txt'))

        imind += 1

text_file.close()

print('Linking val')
text_file = open(val_file, 'r')

imind = 0

for line in text_file:

    parsed = re.search('(\d+)', line)

    if parsed is not None:

        id = str(parsed[0])
        new_id = '{:06d}'.format(imind)

        if not os.path.exists(os.path.join(kitti_val['cal'], str(new_id) + '.txt')):
            os.symlink(os.path.join(kitti_raw['cal'], str(id) + '.txt'), os.path.join(kitti_val['cal'], str(new_id) + '.txt'))

        if not os.path.exists(os.path.join(kitti_val['ims'], str(new_id) + '.png')):
            os.symlink(os.path.join(kitti_raw['ims'], str(id) + '.png'), os.path.join(kitti_val['ims'], str(new_id) + '.png'))

        if not os.path.exists(os.path.join(kitti_val['pre'], str(new_id) + '_01.png')):
            os.symlink(os.path.join(kitti_raw['pre'], str(id) + '_01.png'), os.path.join(kitti_val['pre'], str(new_id) + '_01.png'))
        
        if not os.path.exists(os.path.join(kitti_val['pre'], str(new_id) + '_02.png')):
            os.symlink(os.path.join(kitti_raw['pre'], str(id) + '_02.png'), os.path.join(kitti_val['pre'], str(new_id) + '_02.png'))
        
        if not os.path.exists(os.path.join(kitti_val['pre'], str(new_id) + '_03.png')):
            os.symlink(os.path.join(kitti_raw['pre'], str(id) + '_03.png'), os.path.join(kitti_val['pre'], str(new_id) + '_03.png'))

        if not os.path.exists(os.path.join(kitti_val['lab'], str(new_id) + '.txt')):
            os.symlink(os.path.join(kitti_raw['lab'], str(id) + '.txt'), os.path.join(kitti_val['lab'], str(new_id) + '.txt'))

        imind += 1

text_file.close()

print('Done')
