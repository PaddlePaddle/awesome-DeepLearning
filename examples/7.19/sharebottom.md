# sharebottom

说起多任务学习，最为常规的思路就是共享底部最抽象的表示层，然后在上层分化出不同的任务：

 ![Image text](https://upload-images.jianshu.io/upload_images/3866322-f1184b0e532c4474.png?imageMogr2/auto-orient/strip|imageView2/2/w/706/format/webp)

这实际跟迁移学习有点类似，在图像领域甚是常见，因为图像识别的底层特征往往代表一些像素纹理之类的抽象特征，而跟具体的任务不是特别相关，因此这种低冲突的表示层共享是比较容易出效果的，并且可以减少多任务的计算量。

 ![Image text](https://upload-images.jianshu.io/upload_images/3866322-ed02d72414f7c27d.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

 ![Image text](https://upload-images.jianshu.io/upload_images/3866322-a31aca25421bed27.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

比如说可以很轻松的合并一个识别猫的任务和一个识别狗的任务，因为这两个任务所需要学习的表示很相似，因此同时学好这两个任务是可能的。 但是对于差别比较大的任务来说，比如用这种简单的共享底层表示的方式将一个识别车子的任务和一个识别狗的任务合到一起，模型就不行了

 ![Image text](https://upload-images.jianshu.io/upload_images/3866322-d0415a298fc87ffb.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)
 
 从直觉上，能感觉识别车和识别狗的任务相对猫狗的识别任务差异大了很多，因此Shared Bottom 的方式就不那么有效了。说明任务越相关，这种方式训练效果越好，若是不太相关的任务，效果就有些差强人意了。