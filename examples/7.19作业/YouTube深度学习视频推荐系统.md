## YouTube深度学习视频推荐系统

## 概念：

**推荐系统**：用来预测使用者对于他们还没有见到或了解的事物的喜好。由于网络信息的复杂性和动态性，推荐系统成为解决信息过载问题的有效途径。

**深度学习**：是通过组合低层的特征，形成更加抽象的高层表示属性或特征，以发现数据的分布式特征表示。目前已应用于语音识别、图像处理、自然语言处理等诸多方面。同时，目前的研究已经证明其可以用于检索和推荐任务中。

将深度学习应用于推荐系统中，由于其最先进的性能和高质量的建议，正在得到发展。与传统的推荐模式相比，深度学习可以更好地理解用户的需求，项目的特点以及它们之间的历史交互。

## 流程：

<img src="C:\Users\spade-卿\AppData\Roaming\Typora\typora-user-images\image-20210724171141615.png" alt="image-20210724171141615" style="zoom:67%;" />

## 原理：

假设用户的兴趣标签及对应的标签权重如下，其中![640?wx_fmt=png](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/cge06IdYQyfal2eHibRnne77SicjcgVdibYGia5h5wZd4WmPkHkaGic2tseEFzvTlCHAfUfCIht27wLocAA9Oo2Z2Aw/640?wx_fmt=png)  是标签， ![640?wx_fmt=png](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/cge06IdYQyfal2eHibRnne77SicjcgVdibYAmpKbY4VN40gsKCBicf2FmTAgI2eLHQiaiciafEauic8AicJd2mCKoNj38vw/640?wx_fmt=png)是用户对标签的偏好权重。

假设标签 ![640?wx_fmt=png](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/cge06IdYQyfal2eHibRnne77SicjcgVdibYpd6VSS3x6GKR2DrEjlIweeTo6ks2PnnGSKyCo8QtDuJUWicrBvolPgw/640?wx_fmt=png) 关联的视频分别为：

![640?wx_fmt=png](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/cge06IdYQyfal2eHibRnne77SicjcgVdibYwQsfje45N4oVbzqW3a8yguia4Dt7LDZLSh8jtwg7iaicoPl6OQmaiaanTg/640?wx_fmt=png)



![640?wx_fmt=png](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/cge06IdYQyfal2eHibRnne77SicjcgVdibYZ4fhT0Lo3O74vR4edLoeiaHBnRXtXWrJ4CiacRmx5kLDg8fNoWwJLt6Q/640?wx_fmt=png)

​                                                                .......



![640?wx_fmt=png](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/cge06IdYQyfal2eHibRnne77SicjcgVdibYePc2WouTtEOMPqUMjhE9pG8Cdoc8sic6lIibiaz0pjIgicj8vgVibudgicfQ/640?wx_fmt=png)



其中![640?wx_fmt=png](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/cge06IdYQyfal2eHibRnne77SicjcgVdibYS27QpTTg5gzgiabbVLb2hJGranYasibCnQVq4iceArb7j0ibUkMqpFXE5w/640?wx_fmt=png) 、![640?wx_fmt=png](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/cge06IdYQyfal2eHibRnne77SicjcgVdibYicx7icjWsjVLjNGAfqtvlibvBSkSmDFq1QZpu3VM6LPu6cX9ouxXVsn5A/640?wx_fmt=png)分别是标的物及对应的权重，那么

![640?wx_fmt=png](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/cge06IdYQyfal2eHibRnne77SicjcgVdibYVqJA8oc9W4UvicRDwK7PCdLmDvKvib8ud2UuAQribLicDYr5P7rpoTcibmQ/640?wx_fmt=png)



上式中U是用户对视频的偏好集合，我们这里将视频    看成向量空间的基，所以有上面的公式。不同的标签可以关联到相同的视频(因为不同的视频可以有相同的标签)，上式中最后一个等号右边需要合并同类项，将相同基前面的系数相加。合并同类项后，视频(基)前面的数值就是用户对该视频的偏好程度了，我们对这些偏好程度降序排列，就可以为用户做topN推荐了。

 上面只是基于用户兴趣画像来为用户做推荐的算法原理，实际业务中，用户的兴趣有长期兴趣、短期兴趣，同时还需要考虑给用户提供多样性的推荐及根据用户播放过程中的实时反馈调整推荐结果，所以实际工程上会非常复杂，这一块我们会在第三节的架构及工程实现、第四节的召回和排序中详细说明。




## 作用：

1、完全个性化推荐

完全个性化推荐是为每个用户生成不一样的推荐结果，下图是电视猫小视频实时个性化推荐，基于用户的(标签)兴趣画像，为用户推荐跟用户兴趣偏好相似的视频，用户可以无限右滑(由于电视猫是客厅端的视频软件，靠遥控器交互，所以产品交互方式上跟头条等手机端的下拉交互是不一样的)获取自己感兴趣的推荐结果，整个算法会根据用户的兴趣变化实时为用户更新推荐结果。

2、相似视频推荐

短视频相似推荐基于视频标签构建视频之间的相似度，为每个视频推荐相似的视频。

3、主题推荐

主题推荐根据用户播放行为历史，构建用户兴趣画像，这里是基于节目的标签来构建用户画像，基于用户画像标签为用户推荐最感兴趣的标签关联的节目。



## 优缺点

缺点：

- 因为只引入了用户和商品之间的行为数据，导致头部内容更容易和其他内容相似，而长尾内容因为稀疏的行为向量，很难被推荐出去
- 数据单一、泛化能力差

优点：

- 可解释性强
- 工程需求量大