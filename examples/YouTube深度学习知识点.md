# YouTube 深度学习视频推荐系统知识点

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/f821e16271344e5b8b4deb04bbe64187a004cfdb5c004c028121b63cf4709793" width="700" ></center>
<center><br></br></center>
<br></br>


作为全球最大的UGC的视频网站，需要在百万量级的视频规模下进行个性化推荐。由于候选视频集合过大，考虑online系统延迟问题，不宜用复杂网络直接进行推荐，Covington等人利用用户信息、情境信息，提炼出了一种深度神经网络模型用于YouTube视频推荐。该系统有两个关键过程，分别是候选生成和排序。

* 第一层是Candidate Generation Model完成候选视频的快速筛选，这一步候选视频集合由百万降低到了百的量级。
* 第二层是用Ranking Model完成几百个候选视频的精排
首先介绍candidate generation模型的架构:

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/1e27c28c52a54199b93f22dca98107454fc555ee053e42e5bd9629c64f8d0edb" width="700" ></center>
<center><br>Youtube Candidate Generation Model</br></center>
<br></br>

自底而上看这个网络，最底层的输入是用户观看过的video的embedding向量，以及搜索词的embedding向量。

先用word2vec方法对video和search token做了embedding之后再作为输入的，特征向量里面还包括了用户的地理位置的embedding，年龄，性别等。然后把所有这些特征concatenate起来，喂给上层的ReLU神经网络。三层神经网络过后，我们看到了softmax函数。这里Youtube把这个问题看作为用户推荐next watch的问题，所以输出应该是一个在所有candidate video上的概率分布，自然是一个多分类问题。

得到了几百个候选集合，下一步就是利用ranking模型进行精排序，下面是ranking深度学习网络的架构图。
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/fc2a98dc7f06463f9f35c270fea0706a27c39e6e2728487ea1d882edf1670ada" width="700" ></center>
<center><br>Youtube Ranking Model</br></center>
<br></br>
从左至右的特征依次是

* impression video ID embedding: 当前要计算的video的embedding
* watched video IDs average embedding: 用户观看过的最后N个视频embedding的average pooling
* language embedding: 用户语言的embedding和当前视频语言的embedding
* time since last watch: 自上次观看同channel视频的时间
* #previous impressions: 该视频已经被曝光给该用户的次数



* 把推荐问题转换成多分类问题，在预测next watch的场景下，每一个备选video都会是一个分类，因此总共的分类有数百万之巨，这在使用softmax训练时无疑是低效的，YouTube采取了负采样（negative sampling）并用importance weighting的方法对采样进行calibration。
* 在model serving过程中对几百万个候选集逐一跑一遍模型的时间开销显然太大了，因此在通过candidate generation model得到user 和 video的embedding之后，通过最近邻搜索的方法的效率高很多。所以在model serving过程中对几百万个候选集逐一跑一遍模型的时间开销显然太大了，因此在通过candidate generation model得到user 和 video的embedding之后，通过最近邻搜索的方法的效率高很多。
* 在处理测试集的时候，YouTube不采用经典的随机留一法（random holdout），而是把用户最近的一次观看行为作为测试集，只留最后一次观看行为做测试集主要是为了避免引入future information，产生与事实不符的数据穿越。

