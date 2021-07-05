# Image classification based on Transformer [[简体中文](./README.md)]

## Dependent packages
- os
- numpy
- opencv
- pillow
- paddlepaddle==2.0.0

## Structure
'''
|-data: Store the ImageNet verification set
|-transform.py: data preprocessing script
|-dataset.py: script to read data
|-model.py: The script defines the network structure of ViT and DeiT
|-eval.py: Script to start model evaluation

'''

## Dataset preparation
- Enter insatallation dir

  ```
  cd path_to_Transformer-classification
  ```

- Download [ImageNet Verification Set](https://aistudio.baidu.com/aistudio/datasetdetail/93561) to the `data` directory.

  Please organize data dir as below

  ```
  data/ILSVRC2012_val
  |_ val
  |_ val_list.txt
  ```

## Model preparation

- Enter insatallation dir

  ```
  cd path_to_Transformer-classification
  ```

- Download the model weight files of ViT and DeiT to the `model_file` directory

  ```
  cd model_file
  wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_384_pretrained.pdparams
  wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_distilled_patch16_384_pretrained.pdparams
  cd ../
  ```

## Model evaluation

The model evaluation process can be started as follows

```bash
python3 eval.py 
    --model ViT  \
    --data data/ILSVRC2012_val
```

Among them:

+ `model`: Model name, The default value is `ViT`, which can be changed to `DeiT`;
+ `data`: The directory to save the ImageNet verification set, the default value is `data/ILSVRC2012_val`.

