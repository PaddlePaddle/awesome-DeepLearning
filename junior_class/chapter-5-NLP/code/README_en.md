# Training Word Embedding with Word2vec  [[简体中文](./README.md)]
## Dependent packages
- os
- math
- random
- requests
- numpy
- paddle

**note**: please install paddle with version 2.0. if you have not installed it, please refer to
  [ the quick install](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/pip/windows-pip.html) 。
    
## Structure
```buildoutcfg
|-data: the dir of saving dataset
|-model: 
    |-word2vec: the implement of skip gram
|-utils: 
    |-data_processor.py: the operations related data processing
    |-utils: some tool methods
|-train.py: the script of training model
```

## Training Model
>python train.py