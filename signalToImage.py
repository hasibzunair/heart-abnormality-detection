"""
This code converts 1D heart beats to 2D images and saves each
beat according to its class separately.
"""

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

ROOT_DIR = os.path.abspath("./")
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

def segmentation(records, type, class_counter, output_dir=''):

    """'N' for normal beats. Similarly we can give the input 'L' for left bundle branch block beats. 'R' for right bundle branch block
        beats. 'A' for Atrial premature contraction. 'V' for ventricular premature contraction. '/' for paced beat. 'E' for Ventricular
        escape beat."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
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

                results.append(data)

                plt.axis([0, 192, floor, ceil])
                plt.plot(data, linewidth=0.5)
                plt.xticks([]), plt.yticks([])
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)

                filename = output_dir + 'fig_{}_{}'.format(class_counter, count) + '.png'
                plt.savefig(filename)
                plt.close()
                im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                im_gray = cv2.erode(im_gray, kernel, iterations=1)
                im_gray = cv2.resize(im_gray, (192, 128), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(filename, im_gray)
                #print('img writtten {}'.format(filename))
                count += 1
                
    return results



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
output_dirs = ['0/', '1/', '1/', '1/', '1/', '1/', '1/', '1/']

# Process data
# If multi-class, manual split of 60 20 20 is applied to classes VEB and VFW since this symptom is only in
# 107620 images for multi-class 

class_counter = 0
print("Process training data...")
for type, output_dir in tqdm(zip(labels, output_dirs)):
    output_dir = os.path.join("train", output_dir)
    segmentation(train_records, type, class_counter, output_dir='./datasets/isolated-beat-images/'+output_dir)
    class_counter+=1
    
class_counter = 0
print("Process val data...")
for type, output_dir in tqdm(zip(labels, output_dirs)):
    output_dir = os.path.join("val", output_dir)
    segmentation(val_records, type, class_counter, output_dir='./datasets/isolated-beat-images/'+output_dir)
    class_counter+=1

class_counter = 0
print("Process test data...")
for type, output_dir in tqdm(zip(labels, output_dirs)):
    output_dir = os.path.join("test", output_dir)
    segmentation(test_records, type, class_counter, output_dir='./datasets/isolated-beat-images/'+output_dir)
    class_counter+=1

    
