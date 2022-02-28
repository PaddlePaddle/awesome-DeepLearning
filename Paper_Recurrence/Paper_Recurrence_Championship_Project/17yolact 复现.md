##  yolact 复现

repo 地址：https://github.com/jay-z20/yolact-paddle

##  按照  `run.sh` 脚本代码在终端运行，预测 `test-dev` 结果

### 环境配置


```python
# 安装依赖包
!pip install pycocotools
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting pycocotools
      Downloading https://mirror.baidu.com/pypi/packages/de/df/056875d697c45182ed6d2ae21f62015896fdb841906fe48e7268e791c467/pycocotools-2.0.2.tar.gz
    Requirement already satisfied: setuptools>=18.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools) (56.2.0)
    Requirement already satisfied: cython>=0.27.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools) (0.29)
    Requirement already satisfied: matplotlib>=2.1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools) (2.2.3)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools) (2.4.2)
    Requirement already satisfied: six>=1.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools) (1.15.0)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools) (2019.3)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools) (1.1.0)
    Requirement already satisfied: numpy>=1.7.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools) (1.20.3)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools) (2.8.0)
    Building wheels for collected packages: pycocotools
      Building wheel for pycocotools (setup.py) ... [?25ldone
    [?25h  Created wheel for pycocotools: filename=pycocotools-2.0.2-cp37-cp37m-linux_x86_64.whl size=278368 sha256=9ffe255cc6e56949b8ae94e75dfd5c4547b14c8a33297c26e2d237245620443e
      Stored in directory: /home/aistudio/.cache/pip/wheels/fb/44/67/8baa69040569b1edbd7776ec6f82c387663e724908aaa60963
    Successfully built pycocotools
    Installing collected packages: pycocotools
    Successfully installed pycocotools-2.0.2



```python
# 预测 test-dev


!cd yolact-paddle1/
!python eval.py --trained_model ./weights/yolact_resnet50_54_800000.dpparams --output_coco_json  --dataset=coco2017_testdev_dataset --cuda=True
```


```python
!cd yolact-paddle1/
!pwd
```

    /home/aistudio



```python
# 垃圾占用大小达到0.0GB时释放内存垃圾，即一旦出现垃圾则马上释放。
export FLAGS_eager_delete_tensor_gb=0.0

#启用快速垃圾回收策略，不等待cuda kernel 结束，直接释放显存
export FLAGS_fast_eager_deletion_mode=1

#该环境变量设置只占用0%的显存
export FLAGS_fraction_of_gpu_memory_to_use=0
```


```python
lrs = []
lr_warmup_until = 500
lr_warmup_init = 1e-4
lr = 1e-3
for iteration in range(lr_warmup_until):
   r = (lr - lr_warmup_init) * (iteration / lr_warmup_until) + lr_warmup_init
   lrs.append(r)
```


```python
lrs[::10][:10]
```




    [0.0001,
     0.00011800000000000001,
     0.000136,
     0.000154,
     0.000172,
     0.00019,
     0.00020800000000000001,
     0.00022600000000000002,
     0.00024400000000000002,
     0.00026199999999999997]




```python
step_index = 0
lr_steps = [280000, 600000, 700000, 750000]
gamma = 0.1
lr = 1e-3

lr2 = []
iteration = lr_steps[1]
while step_index < len(lr_steps) and iteration >= lr_steps[step_index]:
    step_index += 1
    #lr2.append(lr * (gamma ** step_index))
    print("lr:",lr * (gamma ** step_index))
iteration += 1
```

    lr: 0.0001
    lr: 1.0000000000000003e-05



```python
milestones=(280000, 600000, 700000, 750000)
step_per_epoch = 200
boundary = [int(step_per_epoch) * i for i in milestones]
boundary
```




    [56000000, 120000000, 140000000, 150000000]




```python
import paddle.optimizer as optim
```


```python
class PiecewiseDecay(object):
    """
    Multi step learning rate decay

    Args:
        gamma (float | list): decay factor
        milestones (list): steps at which to decay learning rate
    """

    def __init__(self,
                 gamma=0.1,
                 milestones=[280000, 360000, 400000],
                 values=None,
                 use_warmup=True):
        super(PiecewiseDecay, self).__init__()
        if type(gamma) is not list:
            self.gamma = []
            for i in range(len(milestones)):
                self.gamma.append(gamma / 10**i)
        else:
            self.gamma = gamma
        self.milestones = milestones
        self.values = values
        self.use_warmup = use_warmup

    def __call__(self,
                 base_lr=None,
                 boundary=None,
                 value=None,
                 step_per_epoch=None):
        if boundary is not None and self.use_warmup:
            boundary.extend([int(step_per_epoch) * i for i in self.milestones])
        else:
            # do not use LinearWarmup
            boundary = [int(step_per_epoch) * i for i in self.milestones]
            value = [base_lr]  # during step[0, boundary[0]] is base_lr

        # self.values is setted directly in config
        if self.values is not None:
            assert len(self.milestones) + 1 == len(self.values)
            return optim.lr.PiecewiseDecay(boundary, self.values)

        # value is computed by self.gamma
        value = value if value is not None else [base_lr]
        for i in self.gamma:
            value.append(base_lr * i)

        return optim.lr.PiecewiseDecay(boundary, value)

class LinearWarmup(object):
    """
    Warm up learning rate linearly

    Args:
        steps (int): warm up steps
        start_factor (float): initial learning rate factor
    """

    def __init__(self, steps=500, start_factor=1e-4):
        super(LinearWarmup, self).__init__()
        self.steps = steps
        self.start_factor = start_factor

    def __call__(self, base_lr, step_per_epoch):
        boundary = []
        value = []
        for i in range(self.steps + 1):
            if self.steps > 0:
                alpha = i / self.steps
                lr = (base_lr - self.start_factor) * alpha + self.start_factor
                value.append(lr)
            if i > 0:
                boundary.append(i)
        return boundary, value

class LearningRate(object):
    """
    Learning Rate configuration

    Args:
        base_lr (float): base learning rate
        schedulers (list): learning rate schedulers
    """
    __category__ = 'optim'

    def __init__(self,
                 schedulers=[PiecewiseDecay(gamma= 0.1, milestones=(280000, 600000, 700000, 750000)), LinearWarmup(steps=500,start_factor=1e-4)]):
        super(LearningRate, self).__init__()
        self.schedulers = schedulers

    def __call__(self, step_per_epoch):
        assert len(self.schedulers) >= 1
        # warmup
        boundary, value = self.schedulers[1](1e-3, step_per_epoch)
        # decay
        decay_lr = self.schedulers[0](1e-3, boundary, value,
                                      step_per_epoch)
        return decay_lr
```


```python
sc = LearningRate()(1)
lrs = []
for i in range(750000+500):
    lr = sc.get_lr()
    lrs.append(lr)
    #print(i,lr)
    sc.step()
```


```python
lrs[750000-5:750000+5]
```




    [1e-06,
     1e-06,
     1e-06,
     1e-06,
     1e-06,
     1.0000000000000001e-07,
     1.0000000000000001e-07,
     1.0000000000000001e-07,
     1.0000000000000001e-07,
     1.0000000000000001e-07]




```python
sc = LearningRate()(1000)
```


```python
sc.step()
```


```python
sc.get_lr()
```




    0.00010180000000000001




```python
import paddle
import numpy as np
```


```python
a = paddle.to_tensor(float("inf"))
a
```




    Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
           [inf.])




```python
np.isinf(a.cpu().numpy())[0]
```




    True




```python
np.isnan(a.cpu().numpy())[0]
```




    False




```python
paddle.isfinite(a)
```




    Tensor(shape=[1], dtype=bool, place=CUDAPlace(0), stop_gradient=True,
           [False])




```python
a > 100
```




    Tensor(shape=[1], dtype=bool, place=CUDAPlace(0), stop_gradient=True,
           [True])




```python
if paddle.isfinite(a):
    print('aaa')
```


```python
paddle.isfinite(a).item()
```




    False




```python
b = paddle.to_tensor(2.3)
```


```python
paddle.isfinite(b)
```




    Tensor(shape=[1], dtype=bool, place=CUDAPlace(0), stop_gradient=True,
           [True])




```python
if paddle.isfinite(b):
    print('aaa')
```

    aaa



```python
import paddle
import numpy as np
```


```python
a = paddle.rand((4,2))
```


```python
a[-6:,:]
```




    Tensor(shape=[4, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
           [[0.68550843, 0.17667589],
            [0.32178292, 0.78345305],
            [0.06278194, 0.54712743],
            [0.38716763, 0.96911567]])




```python
b = np.array(1)
b
```




    array(1)




```python
b.item()
```




    1
