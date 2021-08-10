# Image classification based on Swin Transformer [[简体中文](./README.md)]

## Dependent packages
- os
- numpy
- opencv
- pillow
- paddlepaddle==2.1.1

## Project Introduction
```
|-data: Store the ImageNet verification set
|-transform.py: data preprocessing script
|-dataset.py: script to read data
|-swin_transformer.py: The script defines the network structure of Swin Transformer
|-eval.py: Script to start model evaluation
```

**Model introduction**

Swin Transformer is a new Transformer model of computer vision, from the paper "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows". This model can be used as the backbone of computer vision tasks.[Paper](https://arxiv.org/pdf/2103.14030.pdf)


## Dataset preparation

- Enter insatallation dir

  ```
  cd Swin_Transformer_for_image_classification
  ```

- Download [ImageNet Verification Set](https://aistudio.baidu.com/aistudio/datasetdetail/93561) and unzip it to the `data` directory

  ```
  mkdir data && cd data
  tar -xvf ILSVRC2012_val.tar
  cd ../
  ```
  
- Please organize data dir as below

  ```
  data/ILSVRC2012_val
  |_ val
  |_ val_list.txt
  ```

## Model preparation

- Download the model weight files of SwinaTaransformer to the `model_file` directory

  ```
  mkdir model_file && cd model_file
  wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams
  cd ../
  ```

## Model evaluation

The model evaluation process can be started as follows

```bash
python3 eval.py 
    --model SwinTransformer  \
    --data data/ILSVRC2012_val
```

Among them:

+ `model`: Model name;
+ `data`: The directory to save the ImageNet verification set, the default value is `data/ILSVRC2012_val`.
