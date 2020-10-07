#!/usr/bin/env python
# coding: utf-8

# ### This code combines isolated heart beats and their annotations according to its class separately.
# 
# Same notebook is in form of script in previous directory.

# In[ ]:


from glob import glob
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math

# Prevent progress bar from printing to new line
from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, position=0, leave=True)


# In[ ]:


ROOT_DIR = os.path.abspath("../")
DATA_PATH = os.path.join(ROOT_DIR, "datasets", "mit-bih-arrhythmia-database-1.0.0/")
DATA_PATH

def get_records(patients_idxs):
    """ Get paths for data in data/mit/ directory """
    # Download if doesn't exist

    # There are 3 files for each record
    # *.atr is one of them
    #paths = glob('{}/*.atr'.format(DATA_PATH))

    # Get rid of the extension
    #paths = [path[:-4] for path in paths]
    #paths.sort()
    
    paths = [os.path.join(DATA_PATH, path) for path in patients_idxs]
    
    return paths

def segmentation(records, type, output_label):

    """'N' for normal beats. Similarly we can give the input 'L' for left bundle branch block beats. 'R' for right bundle branch block
        beats. 'A' for Atrial premature contraction. 'V' for ventricular premature contraction. '/' for paced beat. 'E' for Ventricular
        escape beat."""
    X, y = [], []
    kernel = np.ones((4, 4), np.uint8)
    count = 1

    '''
    max_values = []
    min_values = []
    mean_values = []
    for e in tqdm(records):
        signals, fields = wfdb.rdsamp(e, channels=[0])
        mean_values.append(np.mean(signals))

    mean_v = np.mean(np.array(mean_values))
    std_v = 0
    count = 0
    for e in tqdm(records):
        signals, fields = wfdb.rdsamp(e, channels=[0])
        count += len(signals)
        for s in signals:
            std_v += (s[0] - mean_v)**2

    std_v = np.sqrt(std_v/count)'''

    mean_v = -0.33859
    std_v = 0.472368
    floor = mean_v - 3*std_v
    ceil = mean_v + 3*std_v

    for e in records:
        signals, fields = wfdb.rdsamp(e, channels = [0])

        ann = wfdb.rdann(e, 'atr')
        good = [type]
        ids = np.in1d(ann.symbol, good)
        imp_beats = ann.sample[ids]
        beats = (ann.sample)
        for i in tqdm(imp_beats):
            beats = list(beats)
            j = beats.index(i)
            if(j!=0 and j!=(len(beats)-1)):
                data = (signals[beats[j]-96: beats[j]+96, 0])
                data = np.array(data)
                if data.shape[0] == 192:
                    X.append(data)
                    y.append(output_label)
                
    return np.array(X), np.array(y)



pts_train = ['222',
 '107',
 '101',
 '118',
 '116',
 '115',
 '108',
 '106',
 '214',
 '105',
 '219',
 '205',
 '102',
 '232',
 '220',
 '114',
 '228',
 '117',
 '121',
 '100',
 '231',
 '234',
 '124',
 '122',
 '202',
 '217',
 '212',
 '233']

pts_val = ['210', '223', '200', '103', '113', '207', '119', '213', '112', '109']
pts_test = ['104', '111', '123', '201', '203', '208', '209', '215', '221', '230']

print("Patient instances in train, val and test: ", len(pts_train), len(pts_val), len(pts_test))

train_records = get_records(pts_train)
val_records = get_records(pts_val)
test_records = get_records(pts_test)


# In[ ]:


# See https://archive.physionet.org/physiobank/annotations.shtml
# Get beats from https://arxiv.org/pdf/1804.06812.pdf

# From the MIT-BIH  database, we  includednormal  beat  (NOR)  and  seven  types
# of  ECG  arrhythmias  including  prema-ture ventricular contraction (PVC),
# paced beat (PAB), right bundle branchblock beat (RBB), left bundle branch block beat (LBB),
# atrial premature con-traction (APC), ventricular flutter wave (VFW), and ventricular escape beat(VEB). 

# NOR PVC PAB RBB LBB APC VFW VEB

# 'N' for normal beats
# 'L' for left bundle branch block beats LBB
# 'R' for right bundle branch block RBB
# 'A' for Atrial premature contraction APC
# 'V' for premature ventricular contraction PVC 
# '/' for paced beat PAB
# 'E' for Ventricular escape beat VEB
# '!' for Ventricular flutter wave VFW

labels = ['N', 'L', 'R', 'A', 'V', '/', 'E', '!']
#labels = ['N', 'L','R','V','/','A','f','F','j','a','E','J','e','S']
#output_dirs = ['NOR/', 'LBBB/', 'RBBB/', 'APC/', 'PVC/', 'PAB/', 'VEB/', 'VFW/']
output_labels = [0, 1, 1, 1, 1, 1, 1, 1]


# In[ ]:


# Process data
# If multi-class, manual split of 60 20 20 is applied to classes VEB and VFW since this symptom is only in
# 107620 images for multi-class 

print("Process training data...")
x_train, y_train = [], []
for type, output_label in tqdm(zip(labels, output_labels)):
    X, y= segmentation(train_records, type, output_label)
    if X.shape[0] != 0:
        x_train.append(X)
        y_train.append(y)
        
x_train = np.concatenate([x for x in x_train], axis=0)
y_train = np.concatenate([y for y in y_train], axis=0)
print(x_train.shape, y_train.shape)


# In[ ]:


print("Process val data...")
x_val, y_val = [], []
for type, output_label in tqdm(zip(labels, output_labels)):
    X, y= segmentation(val_records, type, output_label)
    if X.shape[0] != 0:
        x_val.append(X)
        y_val.append(y)
        
x_val = np.concatenate([x for x in x_val], axis=0)
y_val = np.concatenate([y for y in y_val], axis=0)
print(x_val.shape, y_val.shape)


# In[ ]:


print("Process test data...")
x_test, y_test = [], []
for type, output_label in tqdm(zip(labels, output_labels)):
    X, y= segmentation(test_records, type, output_label)
    if X.shape[0] != 0:
        x_test.append(X)
        y_test.append(y)
        
x_test = np.concatenate([x for x in x_test], axis=0)
y_test = np.concatenate([y for y in y_test], axis=0)
print(x_test.shape, y_test.shape)


# In[ ]:


print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)
#(60051, 192) (60051, 1)
#(23925, 192) (23925, 1)
#(23628, 192) (23628, 1)

# In[ ]:


np.savez(os.path.join(ROOT_DIR, "datasets", "beats_and_labels.npz"), 
         name1=x_train, name2=y_train, name3=x_val, name4=y_val,
        name5=x_test, name6=y_test)


# Sanity check

# In[ ]:


# Load data
data = np.load(os.path.join(ROOT_DIR, "datasets", "beats_and_labels.npz"))
x1 = data['name1']
y1 = data['name2']
x2 = data['name3']
y2 = data['name4']

print(x1.shape, y1.shape, x2.shape, y2.shape)
