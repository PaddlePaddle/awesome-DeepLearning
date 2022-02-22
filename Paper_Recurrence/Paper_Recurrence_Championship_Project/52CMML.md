# CMML复现项目



## 解压原始COCO2014图像数据集


```python
%cd /home/aistudio/data/data28191/
!unzip val2014.zip
!unzip train2014_1.zip
!unzip train2014_2.zip
```

    /home/aistudio/data/data28191
    Archive:  val2014.zip
    replace val2014/COCO_val2014_000000000042.jpg? [y]es, [n]o, [A]ll, [N]one, [r]ename: ^C
    Archive:  train2014_1.zip
    replace train2014_1/COCO_train2014_000000000009.jpg? [y]es, [n]o, [A]ll, [N]one, [r]ename: ^C
    Archive:  train2014_2.zip
    replace train2014_2/COCO_train2014_000000293142.jpg? [y]es, [n]o, [A]ll, [N]one, [r]ename: ^C



```python
%cd /home/aistudio/data/data110550/
!tar -xvf CMML-data.tar
```

    /home/aistudio/data/data110550
    CMML-data/
    CMML-data/unsupervise_index.npy
    CMML-data/coco_text.npy
    CMML-data/coco_label.npy
    CMML-data/val_index.npy
    CMML-data/train_index.npy
    CMML-data/coco_images.pkl
    CMML-data/test_index.npy



```python
%cd /home/aistudio/work/
!mkdir data
!mkdir data/images
!find /home/aistudio/data/data28191/train2014_1/ -name "*.jpg" -exec mv {} data/images/ \;
!find /home/aistudio/data/data28191/train2014_2/ -name "*.jpg" -exec mv {} data/images/ \;
!find /home/aistudio/data/data28191/val2014/ -name "*.jpg" -exec mv {} data/images/ \;
```

    /home/aistudio/work
    mkdir: cannot create directory ‘data’: File exists
    mkdir: cannot create directory ‘data/images’: File exists



```python
%cd /home/aistudio/work/
!mv /home/aistudio/data/data110550/CMML-data/* data/
```

    /home/aistudio/work



```python
# 训练

%cd /home/aistudio/work/
!python Deep_attention_strong_weak_train.py
```

    /home/aistudio/work
    W0929 09:03:31.705452 26077 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0929 09:03:31.710263 26077 device_context.cc:422] device: 0, cuDNN Version: 7.6.
    Namespace(Attentionparameter='128, 64, 32, 1', Imgpredictpara='128, 20', Predictpara='128, 20', Textfeaturepara='2912, 256, 128', Textpredictpara='128, 20', batchsize=4, epochs=20, img_lr_supervise=0.0001, img_supervise_epochs=0, imgbatchsize=32, imgfilename='data/images/', imgfilenamerecord='data/coco_images.pkl', labelfilename='data/coco_label.npy', lambda1=0.01, lambda2=1, lr_supervise=0.0001, savepath='models/', superviseunsuperviseproportion='3, 7', text_lr_supervise=0.0001, text_supervise_epochs=1, textbatchsize=32, textfilename='data/coco_text.npy', traintestproportion=0.667, use_gpu=True, visible_gpu='0', weight_decay=0)
    [2021-09-29 09:03:37,273][Train.py][line:38][INFO] start training!
    train text supervise data: 1
    test text data:
    /home/aistudio/work/Model/Test.py:136: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      text_predict = np.array(text_predict)
    /home/aistudio/work/Model/Test.py:137: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      truth = np.array(truth)
    [2021-09-29 09:06:36,328][Train.py][line:152][INFO] Epoch:[1/2]
    acc=0.649
    train supervise data: 1
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/layer/norm.py:641: UserWarning: When training, we now always track global mean and variance.
      "When training, we now always track global mean and variance.")
    val:  1000
    test data:
    [2021-09-29 09:10:59,845][Train.py][line:336][INFO] Epoch:[1/21]
    acc1=0.77727, acc2=0.73102, acc3=0.66500
    coverage1=3.25600, coverage2=3.65600, coverage3=5.76700
    example_auc1=0.92795, example_auc2=0.91166, example_auc3=0.83278
    macro_auc1=0.85034, macro_auc2=0.86388, macro_auc3=0.68764
    micro_auc1=0.91109, micro_auc2=0.91015, micro_auc3=0.82107
    ranking_loss1=0.07205, ranking_loss2=0.08834, ranking_loss3=0.16722
    coverage1 :  3.2560000000000002
    train supervise data: 2
    val:  1000
    test data:
    [2021-09-29 09:15:26,404][Train.py][line:336][INFO] Epoch:[2/21]
    acc1=0.79048, acc2=0.72424, acc3=0.71093
    coverage1=2.98700, coverage2=3.68900, coverage3=4.16400
    example_auc1=0.93560, example_auc2=0.90805, example_auc3=0.88952
    macro_auc1=0.88504, macro_auc2=0.87608, macro_auc3=0.79022
    micro_auc1=0.92677, micro_auc2=0.90701, micro_auc3=0.87415
    ranking_loss1=0.06440, ranking_loss2=0.09195, ranking_loss3=0.11048
    coverage1 :  2.987
    train supervise data: 3
    val:  1000
    test data:
    [2021-09-29 09:19:52,729][Train.py][line:336][INFO] Epoch:[3/21]
    acc1=0.80897, acc2=0.76179, acc3=0.74894
    coverage1=2.87200, coverage2=3.33300, coverage3=3.41200
    example_auc1=0.94077, example_auc2=0.92123, example_auc3=0.91810
    macro_auc1=0.89109, macro_auc2=0.88067, macro_auc3=0.84877
    micro_auc1=0.93154, micro_auc2=0.92275, micro_auc3=0.90558
    ranking_loss1=0.05923, ranking_loss2=0.07877, ranking_loss3=0.08190
    coverage1 :  2.872
    train supervise data: 4
    val:  1000
    test data:
    [2021-09-29 09:24:16,289][Train.py][line:336][INFO] Epoch:[4/21]
    acc1=0.83500, acc2=0.78392, acc3=0.77418
    coverage1=2.66000, coverage2=2.95000, coverage3=3.08200
    example_auc1=0.94968, example_auc2=0.93594, example_auc3=0.93163
    macro_auc1=0.90329, macro_auc2=0.89601, macro_auc3=0.86826
    micro_auc1=0.93859, micro_auc2=0.93073, micro_auc3=0.91827
    ranking_loss1=0.05032, ranking_loss2=0.06406, ranking_loss3=0.06837
    coverage1 :  2.66
    train supervise data: 5
    val:  1000
    test data:
    [2021-09-29 09:28:43,451][Train.py][line:336][INFO] Epoch:[5/21]
    acc1=0.83692, acc2=0.77606, acc3=0.78435
    coverage1=2.58500, coverage2=3.06800, coverage3=2.97800
    example_auc1=0.95160, example_auc2=0.93213, example_auc3=0.93511
    macro_auc1=0.90650, macro_auc2=0.88977, macro_auc3=0.87650
    micro_auc1=0.94157, micro_auc2=0.92677, micro_auc3=0.92369
    ranking_loss1=0.04840, ranking_loss2=0.06787, ranking_loss3=0.06489
    coverage1 :  2.585
    train supervise data: 6
    val:  1000
    test data:
    [2021-09-29 09:33:11,670][Train.py][line:336][INFO] Epoch:[6/21]
    acc1=0.84046, acc2=0.78510, acc3=0.79722
    coverage1=2.58400, coverage2=3.06300, coverage3=2.88500
    example_auc1=0.95123, example_auc2=0.93257, example_auc3=0.93889
    macro_auc1=0.90038, macro_auc2=0.88753, macro_auc3=0.88097
    micro_auc1=0.93648, micro_auc2=0.92578, micro_auc3=0.92793
    ranking_loss1=0.04877, ranking_loss2=0.06743, ranking_loss3=0.06111
    coverage1 :  2.584
    train supervise data: 7
    val:  1000
    test data:
    [2021-09-29 09:37:47,348][Train.py][line:336][INFO] Epoch:[7/21]
    acc1=0.84092, acc2=0.78606, acc3=0.80688
    coverage1=2.62200, coverage2=2.96700, coverage3=2.85300
    example_auc1=0.95069, example_auc2=0.93549, example_auc3=0.94115
    macro_auc1=0.89979, macro_auc2=0.89152, macro_auc3=0.88274
    micro_auc1=0.93460, micro_auc2=0.92794, micro_auc3=0.92864
    ranking_loss1=0.04931, ranking_loss2=0.06451, ranking_loss3=0.05885
    train supervise data: 8
    ^C
    Traceback (most recent call last):
      File "Deep_attention_strong_weak_train.py", line 84, in <module>
        weight_decay = args.weight_decay, batchsize = args.batchsize, textbatchsize = args.textbatchsize, imgbatchsize = args.imgbatchsize, cuda = cuda, savepath = args.savepath,lambda1=args.lambda1,lambda2=args.lambda2)
      File "/home/aistudio/work/Model/Train.py", line 188, in train
        for batch_index, (x, y) in enumerate(data_loader(), 1):
      File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py", line 197, in __next__
        data = self._reader.read_next_var_list()
    KeyboardInterrupt



```python
# 调用预训练模型进行测试

%cd /home/aistudio/work/
!mv /home/aistudio/work/data/pretrain_model/* /home/aistudio/work/models/

!python predict.py
```

    /home/aistudio/work
    W0929 10:05:58.168318 32023 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0929 10:05:58.173205 32023 device_context.cc:422] device: 0, cuDNN Version: 7.6.
    Namespace(Attentionparameter='128, 64, 32, 1', Imgpredictpara='128, 20', Predictpara='128, 20', Textfeaturepara='2912, 256, 128', Textpredictpara='128, 20', batchsize=4, epochs=0, img_lr_supervise=0.0001, img_supervise_epochs=0, imgbatchsize=32, imgfilename='data/images/', imgfilenamerecord='data/coco_images.pkl', labelfilename='data/coco_label.npy', lambda1=0.01, lambda2=1, lr_supervise=0.0001, savepath='models/', superviseunsuperviseproportion='3, 7', text_lr_supervise=0.0001, text_supervise_epochs=0, textbatchsize=32, textfilename='data/coco_text.npy', traintestproportion=0.667, use_gpu=True, visible_gpu='0', weight_decay=0)
    [2021-09-29 10:06:03,829][Train.py][line:38][INFO] start training!
    test :  11654
    test data:
    /home/aistudio/work/Model/Test.py:70: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      total_predict = np.array(total_predict)
    /home/aistudio/work/Model/Test.py:71: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      img_predict = np.array(img_predict)
    /home/aistudio/work/Model/Test.py:72: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      text_predict = np.array(text_predict)
    /home/aistudio/work/Model/Test.py:73: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      truth = np.array(truth)
    [2021-09-29 10:08:23,210][Train.py][line:382][INFO] Test:
    acc1=0.83574
    coverage1=2.71366
    example_auc1=0.94555
    macro_auc1=0.89441
    micro_auc1=0.93248
    ranking_loss1=0.05445
    [2021-09-29 10:08:23,210][Train.py][line:384][INFO] finish training!



```python

```
