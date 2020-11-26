# Heart abnormality detection using 1D and 2D neural networks

Implemented both 1D and 2D approaches for the task of heart abnormality detection from two-channel ambulatory ECG recordings, obtained from MIT-BIH Arrhythmia Database. For each method and patient level training, validation and test split is applied beforehand. ECG recordings are converted to images which are input to the 2D approach.

### Setup

```
# Clone this repository
git clone https://github.com/hasibzunair/heart-abnormality-det.git
cd heart-abnormality-det/
```
Install the dependencies. TF is version 2.2.
```
pip install wfdb
pip install opencv-python
pip install -U efficientnet

conda install -c anaconda tensorflow-gpu
conda install -c conda-forge keras==2.3.1
```

### Usage

* Download the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
* Make a folder named  `datasets` and unzip file inside.
* Run `get_signals.py` to make the patient level train, val and test split 1D heart beats and annotations.
* Run `signals_to_images.py` to make the patient level train, val and test split of beats converted to images and their annotations.

### Dataset

Patient level split: 28, 10, 10. See code for patient indexes.

| Data Type  | Samples | 
| ------------- | ------------- | 
| Train  | 60051  | 
| Val  | 23925  | 
| Test  | 23628  | 

To test on unique heart signatures and avoid leaking information across datasets, we do a patient level split resulting in no overlap between datasets.

### Models
TODO

### Results
TODO


