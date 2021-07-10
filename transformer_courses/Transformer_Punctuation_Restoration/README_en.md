# ELECTRA based Punctuation restoration [[简体中文](./README.md)]

## Dependent packages

- python3
- paddlenlp==2.0.0rc22 
- paddlepaddle==2.1.1
- pandas
- attrdict==2.0.1
- ujson
- tqdm
- paddlepaddle-gpu 


## Project Introduction

```
|-data_transfer.py: Extract the test set and training set data from xml format into txt format
|-data_process.py: Data set preprocessing, and build training and test data sets separately
|-dataloader.py: Contains methods to build dataloader
|-train.py: Build dataloader, load pre-training model, set AdamW optimizer, cross entropy loss function and evaluation method in this script, and define ELECTRA training
|-predict.py: Start the model prediction script and store the prediction result in a txt file
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

## Data Preparation

- Download [IWSLT12.zip data set](https://aistudio.baidu.com/aistudio/datasetdetail/98318) and unzip it to the `data` directory

  ``` 
  mkdir data && cd data
  unzip IWSLT12.zip
  cd ../
  ```
- Please organize the data set in the following format 

  ```
  data
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

## Model Training & Evaluation
- After using `electra.base.yaml` to configure the training hyperparameters, enter the model training. Evaluate the model after training.
- Enter into repo  

  ```bash
  cd Transformer_Punctuation_Restoration
  ```

  ```bash
  python train.py
  ```

## Model Prediction

- Select the model parameters in `checkpoint` and configure them in `electra.base.yaml`, we can start the model's prediction on the test set in the following way. The final prediction result can be output to a txt file.

  ```bash
  python predict_py.py
  ```