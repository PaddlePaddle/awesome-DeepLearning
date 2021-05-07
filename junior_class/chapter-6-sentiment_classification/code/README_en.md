# Sentiment Classification [[简体中文](./README.md)]

## Dependent packages
- re
- os
- numpy
- paddle
- random
- tarfile
- requests

**note**: please install paddle with version 2.0. if you have not installed it, please refer to
  [ the quick install](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/pip/windows-pip.html) 。

## Structure
```buildoutcfg
|-data: The dir of saving dataset or trained model
|-model: 
    |-sentiment_classifier: The implement of sentiment classification
|-utils: 
    |-data_processor.py: the operations related data processing
|-train.py: the script of training model
|-evaluate.py: the script of evaluating model
```

## Training Model
>python train.py

## Evaluating Model
>python evaluate.py