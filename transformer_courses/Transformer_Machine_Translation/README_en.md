# Transformer based Machine Translation[[简体中文](./README.md)]

## Dependent packages

- python3
- subword_nmt==0.3.7
- attrdict==2.0.1
- paddlenlp==2.0.0rc22


## Project Introduction
```
|-data: Store the ImageNet verification set
|-transform.py: data preprocessing script
|-dataset.py: script to read data
|-model.py: The script defines the network structure of ViT and DeiT
|-eval.py: Script to start model evaluation
```

```
|-data_process.py: data cleaning script
|-bpe_process.py: jieba segmentation script
|-bpe_process2.py: bpe preprocessing script
|-dataloader.py: dataloader iterator
|-train.py: Script to start model training
|-predict.py: Script to start model prediction
```

**Model introduction**

Transformer is a classic work of NLP proposed by Google team in June 17. It was proposed by Ashish Vaswani and others in the paper attention is all you need published in 2017. The performance of transformer in machine translation task is better than that of RNN and CNN. Only encoder decoder and attention mechanism can achieve good results. The biggest advantage is that it can be parallelized efficiently.[paper link](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

## Install dependencies

- go to repo directory

  ```
  cd Transformer_Machine_Translation
  ```
- install dependency

  ```
  pip install -r requirements.txt
  ```

## data preparation

- Download[2015-01.tgz dataset](https://wit3.fbk.eu/2015-01) to root dir

  ```
  tar -xvf 2015-01.tgz
  tar -xvf 2015-01/texts/zh/en/zh-en.tgz
  cd path_to_Transformer-classification
  ```

## BPE subword processing

  ```
  python data_process.py
  sh subword.sh
  python bpe_process2.py
  ```
## Model Training

- go repo directory

  ```bash
  cd path_to_Transformer-classification
  ```

  ```bash
  python train.py
  ```

## Model Prediction

The model prediction process can be started as follows

  ```bash
  python predict.py
  ```