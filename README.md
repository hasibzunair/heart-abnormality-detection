# ECG image classification

### Setup

```
# Clone this repository
git clone https://github.com/hasibzunair/ecg-heart.git
cd ecg-heart/
```
Install the dependencies
```
pip install wfdb
pip install opencv-python
pip install split-folders
pip install tqdm
```

### Usage

* Download the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
* Make a folder named  `datasets` and unzip file inside
* Remove 102-0.atr from the folder `mit-bih-arrhythmia-database-1.0.0`. See release info in the above link for details.
* Run `signalToImage.py` to convert ECG beats to 2D images.

### Datasets

```splitfolders.ratio("datasets/MIT-BIH_AD/", output="datasets/mit-bih-ad/", seed=1337, ratio=(.6, .2, .2), group_prefix=None)```

Dataset statistics

| Data Type  | 2 | 3 |
| ------------- | ------------- | ------------- |
| a  | c  | e |
| b  | d  | f |
| b  | d  | f |

### Models
TODO

### Results
TODO


