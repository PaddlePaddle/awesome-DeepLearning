# ELECTRA based Punctuation restoration [[简体中文](./README.md)]

## Dependent packages

- python3
- subword_nmt==0.3.7
- attrdict==2.0.1
- paddlenlp==2.0.0rc22


## Project Introduction

```
|-data_transfer.py: Extract the test set and training set data from xml format into txt format
|-data_process.py: Data set preprocessing, and build training and test data sets separately
|-dataloader.py: dataloader iterator script: load dataset, build dataloader, load pre-training model, set AdamW optimizer, cross entropy loss function and evaluation method
|-train.py: The ELECTRA training is defined in the script
|-predict.py: Start the model prediction script and store the prediction structure in a txt file
```

**Model Introduction**

ELECTRA is proposed in the paper ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS published by Kevin Clark et al. (Standfold and Google Brain) at ICLR 2020. Its biggest contribution is to propose a new pre-training task Replaced Token Detection (RTD) and framework ELECTRA. ELECTRA's RTD task is better than MLM's pre-training task, and a GAN-like framework that is very suitable for NLP is introduced. The biggest advantage is to design a more efficient model structure and self-supervised pre-training task based on the existing computing power resources. [Paper link](https://arxiv.org/abs/2003.10555)

**Task Introduction**

This experiment uses Discriminator to do the punctuation restoration task. Punctuation prediction is essentially a sequence labeling task. The punctuation marks predicted in this experiment are comma, period, and question mark. You can also add other types of punctuation if you like.

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

- Download [IWSLT12.zip data set](https://aistudio.baidu.com/aistudio/datasetdetail/98318) and unzip it to the `data` directory

  ``` 
  mkdir data && cd data
  tar -xvf IWSLT12.zip
  cd ../
  ```
- Please organize the data set in the following format 

  ```
  data/IWSLT12
  |_ IWSLT12.TED.MT.tst2011.en-fr.en.xml
  |_ IWSLT12.TED.SLT.tst2011.en-fr.en.system0.comma.xml
  |_ IWSLT12.TALK.dev2010.en-fr.en.xml
  |_ IWSLT12.TED.MT.tst2012.en-fr.en.xml
  |_ train.tags.en-fr.en.xml
  ```
## Data Preprocessing

  ```bash
  python data_transfer.py  
  python data_process.py  
  ```
## Model Preparation

- Enter into repo  

  ```bash
  cd Transformer_Punctuation_Restoration
  ```

  ```bash
  python dataloader.py
  ```

## Model Training

- Enter into repo  

  ```bash
  cd Transformer_Punctuation_Restoration
  ```

  ```bash
  python train.py
  ```

## Model Evaluation

- The model evaluation process can be started by

  ```bash
  python predict.py
  ```