[简体中文](README.md) | English

# Handwritten digit recognition

## Dependent packages
* os
* json
* gzip
* numpy
* random
* time
* paddle=2.x

## Structure
```
|-datasets: store data and data read script
    |-mnist.json.gz: mnist data file
    |-generate.py: script to read data
|-nets: network structure
    |-fnn.py: two feedforward neural networks of single layer and multilayer are defined
|-train.py: training startup script
```

## Results
|structure  |active   |Regularization     |optimizer|epoch  |lr    |bs    |acc   |
|:--:       |:--:     |:--:     |:--:     |:--:   |:--:  |:--:  |:--:  |
|single     |sigmoid  | N       |SGD      |10     |0.1   |32    |85.03%|
|single     |sigmoid  | N       |SGD      |10     |1     |32    |95.87%|
|single     |relu     | N       |SGD      |10     |0.1   |32    |96.18%|
|multi      |relu     | N       |SGD      |10     |0.1   |32    |97.10%|
|multi      |relu     | Y       |SGD      |10     |0.1   |32    |97.18%|

> The experimental results are taken from AIStudio.

## Train
Start the training directly using the train.py script.
```
python3 train.py
```