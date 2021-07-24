# MMoE

为了进行不相关任务的多任务学习，很多人做了很多工作都见效甚微，然后后来就有了Google的这个相当新颖的模型MMoE。

 ![Image text](https://upload-images.jianshu.io/upload_images/3866322-89f028aed28ebba0.png?imageMogr2/auto-orient/strip|imageView2/2/w/1177/format/webp)

它的脑洞大开之处在于跳出了Shared Bottom那种将整个隐藏层一股脑的共享的思维定式，而是将共享层有意识的（按照数据领域之类的）划分成了多个Expert，并引入了gate机制，得以个性化组合使用共享层。

观察一下上面Shared Bottom的模型结构图和MMoE的图，不难发现，MMoE实际上就是把Shared Bottom层替换成了一个双Gate的MoE层：

 ![Image text](https://upload-images.jianshu.io/upload_images/3866322-62300d0cb1a621bb.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

我们先来看一下原始的Shared Bottom的方式，假设input为x共享的底层网络为f(x), 然后将其输出喂到各任务独立输出层h^k(x)，其中k 表示第k 个任务的独立输出单元，那么，第k个任务的输出y^k即可表示为:

 ![Image text](https://math.jianshu.com/math?formula=y%5Ek%20%3D%20h%5Ek(f(x)))

而MoE共享层将这个大的Shared Bottom网络拆分成了多个小的Expert网络（如图所示，拆成了三个，并且保持参数个数不变，显然分成多少个Expert、每个多少参数，都是可以根据实际情况自己设定的）。我们把第i个Expert网络的运算记为f_i(x),然后Gate操作记为g(x)，他是一个n元的softmax值（n是Expert的个数，有几个Expert，就有几元），之后就是常见的每个Expert输出的加权求和，假设MoE的输出为y,那么可以表示为：

 ![Image text](https://math.jianshu.com/math?formula=y%20%3D%20%5Csum_%7Bi%3D1%7D%5En%20g(x)_if_i(x))

如果只是这样的话，要完成多任务还得像Shared Bottom那样再外接不同的输出层，这样一搞似乎这个MoE层对多任务来说就没什么卵用了，因为它无法根据不同的任务来调整各个Expert的组合权重。所以论文的作者搞了多个Gate，每个任务使用自己独立的Gate，这样便从根源上，实现了网络参数会因为输入以及任务的不同都产生影响。
于是，我们将上面MoE输出稍微改一下，用g^k(x)表示第k个任务的们就得到了MMoE的输出表达：

 ![Image text](https://math.jianshu.com/math?formula=y%5Ek%20%3D%20%5Csum_%7Bi%3D1%7D%5Eng%5Ek(x)_if_i(x))

*Gate
把输入通过一个线性变换映射到nums_expert维，再算个softmax得到每个Expert的权重
*Expert
简单的基层全连接网络，relu激活，每个Expert独立权重

# sharebottom

说起多任务学习，最为常规的思路就是共享底部最抽象的表示层，然后在上层分化出不同的任务：

 ![Image text](https://upload-images.jianshu.io/upload_images/3866322-f1184b0e532c4474.png?imageMogr2/auto-orient/strip|imageView2/2/w/706/format/webp)

这实际跟迁移学习有点类似，在图像领域甚是常见，因为图像识别的底层特征往往代表一些像素纹理之类的抽象特征，而跟具体的任务不是特别相关，因此这种低冲突的表示层共享是比较容易出效果的，并且可以减少多任务的计算量。

 ![Image text](https://upload-images.jianshu.io/upload_images/3866322-ed02d72414f7c27d.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

 ![Image text](https://upload-images.jianshu.io/upload_images/3866322-a31aca25421bed27.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

比如说可以很轻松的合并一个识别猫的任务和一个识别狗的任务，因为这两个任务所需要学习的表示很相似，因此同时学好这两个任务是可能的。 但是对于差别比较大的任务来说，比如用这种简单的共享底层表示的方式将一个识别车子的任务和一个识别狗的任务合到一起，模型就不行了

 ![Image text](https://upload-images.jianshu.io/upload_images/3866322-d0415a298fc87ffb.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)
 
 从直觉上，能感觉识别车和识别狗的任务相对猫狗的识别任务差异大了很多，因此Shared Bottom 的方式就不那么有效了。说明任务越相关，这种方式训练效果越好，若是不太相关的任务，效果就有些差强人意了。