```python
# 解压数据集
%cd
!unzip /home/aistudio/data/data7122/val2017.zip
!unzip /home/aistudio/data/data7122/train2017.zip
# !unzip /home/aistudio/data/data7122/test2017.zip
!unzip /home/aistudio/data/data7122/annotations_trainval2017.zip
# !unzip /home/aistudio/data/data7122/annotations_trainval2017.zip
# !unzip /home/aistudio/data/data7122/PaddlePaddle_baseline_model.zip
# !mv PaddlePaddle_baseline_model weight
!unzip /home/aistudio/data/data105567/yolact-paddle.zip
```

    /home/aistudio
    unzip:  cannot find or open /home/aistudio/data/data7122/val2017.zip, /home/aistudio/data/data7122/val2017.zip.zip or /home/aistudio/data/data7122/val2017.zip.ZIP.
    unzip:  cannot find or open /home/aistudio/data/data7122/train2017.zip, /home/aistudio/data/data7122/train2017.zip.zip or /home/aistudio/data/data7122/train2017.zip.ZIP.
    unzip:  cannot find or open /home/aistudio/data/data7122/annotations_trainval2017.zip, /home/aistudio/data/data7122/annotations_trainval2017.zip.zip or /home/aistudio/data/data7122/annotations_trainval2017.zip.ZIP.
    unzip:  cannot find or open /home/aistudio/data/data7122/PaddlePaddle_baseline_model.zip, /home/aistudio/data/data7122/PaddlePaddle_baseline_model.zip.zip or /home/aistudio/data/data7122/PaddlePaddle_baseline_model.zip.ZIP.
    mv: cannot stat 'PaddlePaddle_baseline_model': No such file or directory



```python
# unzip /home/aistudio/data/data7122/val2017.zip &&
# unzip /home/aistudio/data/data7122/train2017.zip &&
# unzip /home/aistudio/data/data7122/annotations_trainval2017.zip &&
# unzip /home/aistudio/data/data7122/PaddlePaddle_baseline_model.zip &&
# mv PaddlePaddle_baseline_model weight
```


```python
# 安装依赖
!pip install cython
!pip install pycocotools
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Requirement already satisfied: cython in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (0.29)
    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting pycocotools
      Downloading https://mirror.baidu.com/pypi/packages/de/df/056875d697c45182ed6d2ae21f62015896fdb841906fe48e7268e791c467/pycocotools-2.0.2.tar.gz
    Requirement already satisfied: setuptools>=18.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools) (56.2.0)
    Requirement already satisfied: cython>=0.27.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools) (0.29)
    Requirement already satisfied: matplotlib>=2.1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools) (2.2.3)
    Requirement already satisfied: numpy>=1.7.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools) (1.20.3)
    Requirement already satisfied: six>=1.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools) (1.15.0)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools) (0.10.0)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools) (2019.3)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools) (2.4.2)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools) (2.8.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools) (1.1.0)
    Building wheels for collected packages: pycocotools
      Building wheel for pycocotools (setup.py) ... [?25ldone
    [?25h  Created wheel for pycocotools: filename=pycocotools-2.0.2-cp37-cp37m-linux_x86_64.whl size=278365 sha256=a08c5203e344c1ae8c0e9484337969b8ddc7c65403fc4452753e65ebcfa17638
      Stored in directory: /home/aistudio/.cache/pip/wheels/fb/44/67/8baa69040569b1edbd7776ec6f82c387663e724908aaa60963
    Successfully built pycocotools
    Installing collected packages: pycocotools
    Successfully installed pycocotools-2.0.2



```python
# 开始训练
%cd
%cd yolact-paddle/
!python train.py --config=yolact_plus_resnet50_config
```


```python
# 继续训练，在48ww时达到精度
%cp /home/aistudio/model_470000.pdopt weights/model_470000.pdopt
!python train.py --config=yolact_plus_resnet50_config \
--resume=/home/aistudio/yolact_plus_resnet50_32_470000.pth \
--start_iter=470000
```


```python
# 验证精度
%cd yolact-paddle
!python eval.py --config=yolact_plus_resnet50_config \
--trained_model=weights/yolact_plus_resnet50_32_480000.pth \
--output_coco_json

!python run_coco_eval.py --eval_type=mask
```

    /home/aistudio/yolact-paddle
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import MutableMapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Iterable, Mapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Sized
    Multiple GPUs detected! Turning off JIT.
    use_jit False
    image_path /home/aistudio/train2017/ info_file /home/aistudio/annotations/instances_train2017.json
    loading annotations into memory...
    Done (t=22.81s)
    creating index...
    index created!
    image_path /home/aistudio/val2017/ info_file /home/aistudio/annotations/instances_val2017.json
    loading annotations into memory...
    Done (t=0.57s)
    creating index...
    index created!
    W0820 22:45:23.783344   267 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0820 22:45:23.787801   267 device_context.cc:422] device: 0, cuDNN Version: 7.6.
    Initializing weights...
    Begin training!

    [  0]       0 || lr: 0.0001000 | B: 5.622 | C: 18.577 | M: 6.139 | S: 55.199 | I: 1.312 | T: 86.850 || ETA: 15 days, 2:38:57 || timer: 1.632
    [  0]      10 || lr: 0.0001180 | B: 4.626 | C: 18.488 | M: 5.371 | S: 48.367 | I: 1.558 | T: 78.409 || ETA: 7 days, 10:44:05 || timer: 1.211
    [  0]      20 || lr: 0.0001360 | B: 4.971 | C: 18.772 | M: 5.524 | S: 35.059 | I: 1.477 | T: 65.804 || ETA: 7 days, 4:21:48 || timer: 0.668
    [  0]      30 || lr: 0.0001540 | B: 5.447 | C: 18.723 | M: 5.648 | S: 24.714 | I: 1.405 | T: 55.937 || ETA: 7 days, 4:08:41 || timer: 0.736
    ^C
    Stopping early. Saving network...



```python
2>=1
```




    True
