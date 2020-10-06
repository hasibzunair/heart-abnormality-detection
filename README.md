# ECG image classification

### Setup

```
# Clone this repository
git clone https://github.com/hasibzunair/ecg-heart.git
cd ecg-heart/
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
* Run `preprocess_data.ipynb` to get the patient level train, val and test split 1D heart beats and annotations.

### Dataset

TODO - > Patient record names for train val test. Patient level split: 28, 10, 10.

| Data Type  | Normal | Abnormal | Total Samples |
| ------------- | ------------- | ------------- | ------------- |
| Train  | 40606  | 19954 | 60560 |
| Val  | 16727  | 7094 | 23821 | 
| Test  | 17462  | 7256 | 24718 |

### Models
TODO

### Results
TODO


