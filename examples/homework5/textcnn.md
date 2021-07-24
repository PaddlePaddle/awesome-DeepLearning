一、textCNN模型结构
  这是一篇短文，文中用很精炼的话语将CNN结构进行了描述，在图像CNN的模型上做了一些改变，改成适合处理文本任务的模型。论文中的结构图如下：
![](https://ai-studio-static-online.cdn.bcebos.com/91bb7ce9345c4832afc9ae90ff1c4114e553934be4ce454bac6deffd0a1d5a21)


  共分为了四个层：输入层、卷积层、池化层和全连接+softmax输出层。其中的n nn为最大句子长度，k kk为词向量的维度。输入是各句子各个词的词向量，输出的是代表句子特征的句向量。

  textCNN的详细的结构如下图所示



Embedding：第一层是图中最左边的7×5的句子矩阵，每行是词向量，词向量维度=5。
Convolution：然后经过 kernel_sizes=(2,3,4) 的一维卷积层，每个kernel_size 的out_channel=2。
MaxPolling：第三层是一个1-max pooling层，这样不同长度句子经过pooling层之后都能变成定长的表示。然后将这些定长的特征表示进行concatenate。
FullConnection and Softmax：最后接一层全连接的 softmax 层，输出每个类别的概率。
二、textCNN与用于图像的CNN的不同
  由于该模型是用于文本的(而非CNN的传统处理对象：图像)，因此在cnn的操作上相对应地做了一些小调整：

对于文本任务，输入层自然使用了word embedding来做input data representation。
接下来是卷积层，大家在图像处理中经常看到的卷积核都是正方形的，比如4*4，然后在整张image上沿宽和高逐步移动进行卷积操作。但是NLP中输入的“image”是一个词矩阵，比如n个words，每个word用200维的vector表示的话，这个"image"就是n*200的矩阵，卷积核只在高度上已经滑动，在宽度上和word vector的维度一致（=200），也就是说每次窗口滑动过的位置都是完整的单词，不会将几个单词的一部分“vector”进行卷积，这也保证了word作为语言中最小粒度的合理性。（当然，如果研究的粒度是character-level而不是word-level，需要另外的方式处理）
由于卷积核和word embedding的宽度一致，一个卷积核对于一个sentence，卷积后得到的结果是一个vector， shape=（sentence_len - filter_window + 1, 1），那么，在max-pooling后得到的就是一个Scalar。所以，这点也是和图像卷积的不同之处，需要注意一下。
正是由于max-pooling后只是得到一个scalar，在nlp中，会实施多个filter_window_size（比如3,4,5个words的宽度分别作为卷积的窗口大小），每个window_size又有num_filters个（比如64个）卷积核。一个卷积核得到的只是一个scalar太孤单了，智慧的人们就将相同window_size卷积出来的num_filter个scalar组合在一起，组成这个window_size下的feature_vector。
最后再将所有window_size下的feature_vector也组合成一个single vector，作为最后一层softmax的输入。
三、论文中的参数


请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
