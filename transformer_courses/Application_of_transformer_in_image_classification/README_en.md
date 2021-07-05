# Image classification based on Transformer [[简体中文](./README.md)]

## Dependent packages
- os
- numpy
- opencv
- pillow
- paddlepaddle==2.0.0

## Project Introduction
```
|-data: Store the ImageNet verification set
|-transform.py: data preprocessing script
|-dataset.py: script to read data
|-model.py: The script defines the network structure of ViT and DeiT
|-eval.py: Script to start model evaluation
```

**Model introduction**

ViT(Vision Transformer) series models were proposed by Google in 2020. These models only use the standard transformer structure, completely abandon the convolution structure, splits the image into multiple patches and then inputs them into the transformer, showing the potential of transformer in the CV field.[Paper](https://arxiv.org/abs/2010.11929)

DeiT(Data-efficient Image Transformers) series models were proposed by Facebook at the end of 2020. Aiming at the problem that the ViT models need large-scale dataset training, the DeiT improved them, and finally achieved 83.1% Top1 accuracy on ImageNet. More importantly, using convolution model as teacher model, and performing knowledge distillation on these models, the Top1 accuracy of 85.2% can be achieved on the ImageNet dataset.[Paper](https://arxiv.org/abs/2012.12877)

## Dataset preparation

- Enter insatallation dir

  ```
  cd path_to_Transformer-classification
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

- Download the model weight files of ViT and DeiT to the `model_file` directory

  ```
  mkdir model_file && cd model_file
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

