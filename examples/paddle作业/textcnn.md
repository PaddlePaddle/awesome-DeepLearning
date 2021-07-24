# **TextCNN**

-----------------
Yoon Kim在论文(2014 EMNLP) Convolutional Neural Networks for Sentence Classification提出TextCNN。

将卷积神经网络CNN应用到文本分类任务，利用多个不同size的kernel来提取句子中的关键信息（类似于多窗口大小的ngram），从而能够更好地捕捉局部相关性。

<br></br>
<center> <img src="https://img2018.cnblogs.com/blog/1182656/201809/1182656-20180919171920103-1233770993.png" width="500" hegiht="">
  
</center>  
<br></br>



TextCNN利用cnn的分类能力，先对text文本进行格式规整，然后进行输入，通过三种不同规格的卷积核进行卷积，卷积核的维度和输入的规格有关，确保卷积核可以顺序的遍历整个文本信息，通过不同大小的卷积核来提取不同特征，随后通过池化将每个卷积核的提取结果提取到同样的维度并拼接，最后通过激活函数进行分类。网络中词向量可以有训练好的也可以在网络中自己训练。

以下是供参考的参数设置：
<br></br>
<center> <img src="https://ai-studio-static-online.cdn.bcebos.com/1e4dec4347da444db9d78a4ae98df64766f7bfdb0db4496bab17b0603cf31e46" width="500" hegiht="">
  
</center>  
<br></br>
