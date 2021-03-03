# movie recommendation system [[简体中文](./README.md)]

## Dependent packages
- os
- numpy
- matplotlib
- PIL
- random
- paddlepaddle==2.0.0


## Structure
```
|-data: store dataset
|-nets: store the network
    |-DSSM.py: DSSM network definition script
|-train.py: script for training DSSM using Movielens dataset
|-movielens_dataset.py: script for processing movielens dataset

```

## Dataset preparation
1. Download the [dataset](https://aistudio.baidu.com/aistudio/datasetdetail/3233) to the data directory
2. Unzip the dataset
‘’‘
cd data
unzip -q ml-1m.zip
’‘’

## Train
1. Train DSSM model
'''
python3 train.py
'''

