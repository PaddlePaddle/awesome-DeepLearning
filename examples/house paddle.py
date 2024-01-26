#!/usr/bin/env python
# coding: utf-8

# In[2]:


{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: \n",
    "# Also add the following code, \n",
    "# so that every time the environment (kernel) starts, \n",
    "# just run the following code: \n",
    "import sys \n",
    "sys.path.append('/home/aistudio/external-libraries')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([6.320e-03, 1.800e+01, 2.310e+00, ..., 3.969e+02, 7.880e+00,\n",
       "       1.190e+01])"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 导入需要用到的package\r\n",
    "import numpy as np\r\n",
    "import json\r\n",
    "# 读入训练数据\r\n",
    "datafile = '/home/aistudio/data/data7804/housing.data'\r\n",
    "data = np.fromfile(datafile, sep=' ')\r\n",
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 读入之后的数据被转化成1维array，其中array的第0-13项是第一条数据，第14-27项是第二条数据，以此类推.... \r\n",
    "# 这里对原始数据做reshape，变成N x 14的形式\r\n",
    "feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', \r\n",
    "                 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]\r\n",
    "feature_num = len(feature_names)\r\n",
    "data = data.reshape([data.shape[0] // feature_num, feature_num])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>\n",
    "Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(14,)\n",
      "[6.320e-03 1.800e+01 2.310e+00 0.000e+00 5.380e-01 6.575e+00 6.520e+01\n",
      " 4.090e+00 1.000e+00 2.960e+02 1.530e+01 3.969e+02 4.980e+00 2.400e+01]\n"
     ]
    }
   ],
   "source": [
    "# 查看数据\r\n",
    "x = data[0]\r\n",
    "print(x.shape)\r\n",
    "print(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(404, 14)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ratio = 0.8\r\n",
    "offset = int(data.shape[0] * ratio)\r\n",
    "training_data = data[:offset]\r\n",
    "training_data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 计算train数据集的最大值，最小值，平均值\r\n",
    "maximums, minimums, avgs = \\\r\n",
    "                     training_data.max(axis=0), \\\r\n",
    "                     training_data.min(axis=0), \\\r\n",
    "     training_data.sum(axis=0) / training_data.shape[0]\r\n",
    "# 对数据进行归一化处理\r\n",
    "for i in range(feature_num):\r\n",
    "    #print(maximums[i], minimums[i], avgs[i])\r\n",
    "    data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def load_data():\r\n",
    "    # 从文件导入数据\r\n",
    "    datafile = '/home/aistudio/data/data7804/housing.data'\r\n",
    "    data = np.fromfile(datafile, sep=' ')\r\n",
    "\r\n",
    "    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数\r\n",
    "    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \\\r\n",
    "                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]\r\n",
    "    feature_num = len(feature_names)\r\n",
    "\r\n",
    "    # 将原始数据进行Reshape，变成[N, 14]这样的形状\r\n",
    "    data = data.reshape([data.shape[0] // feature_num, feature_num])\r\n",
    "\r\n",
    "    # 将原数据集拆分成训练集和测试集\r\n",
    "    # 这里使用80%的数据做训练，20%的数据做测试\r\n",
    "    # 测试集和训练集必须是没有交集的\r\n",
    "    ratio = 0.8\r\n",
    "    offset = int(data.shape[0] * ratio)\r\n",
    "    training_data = data[:offset]\r\n",
    "\r\n",
    "    # 计算训练集的最大值，最小值，平均值\r\n",
    "    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \\\r\n",
    "                                 training_data.sum(axis=0) / training_data.shape[0]\r\n",
    "\r\n",
    "    # 对数据进行归一化处理\r\n",
    "    for i in range(feature_num):\r\n",
    "        #print(maximums[i], minimums[i], avgs[i])\r\n",
    "        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])\r\n",
    "\r\n",
    "    # 训练集和测试集的划分比例\r\n",
    "    training_data = data[:offset]\r\n",
    "    test_data = data[offset:]\r\n",
    "    return training_data, test_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 获取数据\r\n",
    "training_data, test_data = load_data()\r\n",
    "x = training_data[:, :-1]\r\n",
    "y = training_data[:, -1:]\r\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.         0.18       0.07344184 0.         0.31481481 0.57750527\n",
      " 0.64160659 0.26920314 0.         0.22755741 0.28723404 1.\n",
      " 0.08967991]\n",
      "[0.42222222]\n"
     ]
    }
   ],
   "source": [
    "# 查看数据\r\n",
    "print(x[0])\r\n",
    "print(y[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, 0.0]\r\n",
    "w = np.array(w).reshape([13, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\r\n",
    "get_ipython().run_line_magic('matplotlib', 'inline')\r\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "class Network(object):\r\n",
    "    def __init__(self, num_of_weights):\r\n",
    "        # 随机产生w的初始值\r\n",
    "        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子\r\n",
    "        np.random.seed(0)\r\n",
    "        self.w = np.random.randn(num_of_weights, 1)\r\n",
    "        self.b = 0.\r\n",
    "        \r\n",
    "    def forward(self, x):\r\n",
    "        z = np.dot(x, self.w) + self.b\r\n",
    "        return z\r\n",
    "    \r\n",
    "    def loss(self, z, y):\r\n",
    "        error = z - y\r\n",
    "        num_samples = error.shape[0]\r\n",
    "        cost = error * error\r\n",
    "        cost = np.sum(cost) / num_samples\r\n",
    "        return cost\r\n",
    "    \r\n",
    "    def gradient(self, x, y):\r\n",
    "        z = self.forward(x)\r\n",
    "        gradient_w = (z-y)*x\r\n",
    "        gradient_w = np.mean(gradient_w, axis=0)\r\n",
    "        gradient_w = gradient_w[:, np.newaxis]\r\n",
    "        gradient_b = (z - y)\r\n",
    "        gradient_b = np.mean(gradient_b)        \r\n",
    "        return gradient_w, gradient_b\r\n",
    "    \r\n",
    "    def update(self, gradient_w, gradient_b, eta = 0.01):\r\n",
    "        self.w = self.w - eta * gradient_w\r\n",
    "        self.b = self.b - eta * gradient_b\r\n",
    "        \r\n",
    "    def train(self, x, y, iterations=100, eta=0.01):\r\n",
    "        losses = []\r\n",
    "        for i in range(iterations):\r\n",
    "            z = self.forward(x)\r\n",
    "            L = self.loss(z, y)\r\n",
    "            gradient_w, gradient_b = self.gradient(x, y)\r\n",
    "            self.update(gradient_w, gradient_b, eta)\r\n",
    "            losses.append(L)\r\n",
    "            if (i+1) % 10 == 0:\r\n",
    "                print('iter {}, loss {}'.format(i, L))\r\n",
    "        return losses"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 获取数据\r\n",
    "train_data, test_data = load_data()\r\n",
    "x = train_data[:, :-1]\r\n",
    "y = train_data[:, -1:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 创建网络\r\n",
    "net = Network(13)\r\n",
    "num_iterations=1000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "iter 9, loss 5.143394325795511\n",
      "iter 19, loss 3.097924194225988\n",
      "iter 29, loss 2.082241020617026\n",
      "iter 39, loss 1.5673801618157397\n",
      "iter 49, loss 1.2966204735077431\n",
      "iter 59, loss 1.1453399043319765\n",
      "iter 69, loss 1.0530155717435201\n",
      "iter 79, loss 0.9902292156463155\n",
      "iter 89, loss 0.9426576903842504\n",
      "iter 99, loss 0.9033048096880774\n",
      "iter 109, loss 0.868732003041364\n",
      "iter 119, loss 0.837229250968144\n",
      "iter 129, loss 0.807927474161227\n",
      "iter 139, loss 0.7803677341465797\n",
      "iter 149, loss 0.7542920908532763\n",
      "iter 159, loss 0.7295420168915829\n",
      "iter 169, loss 0.7060090054240882\n",
      "iter 179, loss 0.6836105084697767\n",
      "iter 189, loss 0.6622781710179412\n",
      "iter 199, loss 0.6419520361168637\n",
      "iter 209, loss 0.622577651786949\n",
      "iter 219, loss 0.6041045903195837\n",
      "iter 229, loss 0.5864856570315078\n",
      "iter 239, loss 0.5696764374763879\n",
      "iter 249, loss 0.5536350125932015\n",
      "iter 259, loss 0.5383217588525027\n",
      "iter 269, loss 0.5236991929680567\n",
      "iter 279, loss 0.509731841376165\n",
      "iter 289, loss 0.4963861247069634\n",
      "iter 299, loss 0.48363025234390233\n",
      "iter 309, loss 0.47143412454019784\n",
      "iter 319, loss 0.45976924072044867\n",
      "iter 329, loss 0.44860861316590983\n",
      "iter 339, loss 0.4379266855659793\n",
      "iter 349, loss 0.4276992560632111\n",
      "iter 359, loss 0.4179034044959738\n",
      "iter 369, loss 0.4085174235863553\n",
      "iter 379, loss 0.39952075384787633\n",
      "iter 389, loss 0.39089392200622347\n",
      "iter 399, loss 0.382618482740513\n",
      "iter 409, loss 0.3746769635645124\n",
      "iter 419, loss 0.36705281267772816\n",
      "iter 429, loss 0.35973034962581096\n",
      "iter 439, loss 0.35269471861856694\n",
      "iter 449, loss 0.3459318443621334\n",
      "iter 459, loss 0.33942839026966587\n",
      "iter 469, loss 0.33317171892221653\n",
      "iter 479, loss 0.3271498546584252\n",
      "iter 489, loss 0.3213514481781961\n",
      "iter 499, loss 0.31576574305173283\n",
      "iter 509, loss 0.3103825440311682\n",
      "iter 519, loss 0.30519218706757245\n",
      "iter 529, loss 0.30018551094136725\n",
      "iter 539, loss 0.29535383041913843\n",
      "iter 549, loss 0.29068891085453674\n",
      "iter 559, loss 0.28618294415539336\n",
      "iter 569, loss 0.28182852604338504\n",
      "iter 579, loss 0.27761863453655344\n",
      "iter 589, loss 0.27354660958874766\n",
      "iter 599, loss 0.2696061338236152\n",
      "iter 609, loss 0.26579121430413205\n",
      "iter 619, loss 0.26209616528184804\n",
      "iter 629, loss 0.25851559187303397\n",
      "iter 639, loss 0.25504437461176843\n",
      "iter 649, loss 0.2516776548326958\n",
      "iter 659, loss 0.2484108208387405\n",
      "iter 669, loss 0.24523949481147198\n",
      "iter 679, loss 0.24215952042409844\n",
      "iter 689, loss 0.2391669511192288\n",
      "iter 699, loss 0.2362580390155805\n",
      "iter 709, loss 0.2334292244097483\n",
      "iter 719, loss 0.2306771258409729\n",
      "iter 729, loss 0.22799853068858245\n",
      "iter 739, loss 0.22539038627340982\n",
      "iter 749, loss 0.22284979143604464\n",
      "iter 759, loss 0.22037398856623477\n",
      "iter 769, loss 0.21796035605914357\n",
      "iter 779, loss 0.2156064011754777\n",
      "iter 789, loss 0.21330975328373866\n",
      "iter 799, loss 0.2110681574640261\n",
      "iter 809, loss 0.20887946845393043\n",
      "iter 819, loss 0.20674164491810018\n",
      "iter 829, loss 0.2046527440240648\n",
      "iter 839, loss 0.20261091630783168\n",
      "iter 849, loss 0.20061440081366638\n",
      "iter 859, loss 0.1986615204933024\n",
      "iter 869, loss 0.19675067785062839\n",
      "iter 879, loss 0.19488035081864621\n",
      "iter 889, loss 0.19304908885621125\n",
      "iter 899, loss 0.1912555092527351\n",
      "iter 909, loss 0.1894982936296714\n",
      "iter 919, loss 0.18777618462820625\n",
      "iter 929, loss 0.18608798277314595\n",
      "iter 939, loss 0.18443254350353405\n",
      "iter 949, loss 0.18280877436103968\n",
      "iter 959, loss 0.18121563232764165\n",
      "iter 969, loss 0.17965212130459232\n",
      "iter 979, loss 0.1781172897250724\n",
      "iter 989, loss 0.1766102282933619\n",
      "iter 999, loss 0.17513006784373505\n"
     ]
    }
   ],
   "source": [
    "# 启动训练\r\n",
    "losses = net.train(x,y, iterations=num_iterations, eta=0.01)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 画出损失函数的变化趋势\r\n",
    "plot_x = np.arange(num_iterations)\r\n",
    "plot_y = np.array(losses)\r\n",
    "plt.plot(plot_x, plot_y)\r\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "PaddlePaddle 2.1.0 (Python 3.5)",
   "language": "python",
   "name": "py35-paddle1.2.0"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}


# In[ ]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
get_ipython().system('ls /home/aistudio/work')


# In[ ]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')


# In[ ]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
