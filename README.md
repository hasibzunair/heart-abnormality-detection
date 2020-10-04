# ECG image classification

### Setup

```
# Clone this repository
git clone https://github.com/hasibzunair/ecg-heart.git
cd ecg-heart/
```
Install the dependencies

* wfdf
* cv2
* split-folders, tqdm

### Usage

* Download the (MIT-BIH Arrhythmia Database)[https://physionet.org/content/mitdb/1.0.0/]
* Unzip file inside `datasets/` directory
* Remove 102-0.atr. See release info in the above link for details.
* Run `signalToImage.py` to convert ECG beats to 2D images.

### Datasets

```splitfolders.ratio("datasets/MIT-BIH_AD/", output="datasets/mit-bih-ad/", seed=1337, ratio=(.6, .2, .2), group_prefix=None)```

Dataset statistics

| 1  | 2 | 3 |
| ------------- | ------------- | ------------- |
| a  | c  | e |
| b  | d  | f |

### Models
TODO

### Results
TODO


