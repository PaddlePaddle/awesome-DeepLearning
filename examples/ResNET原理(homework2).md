ResNET

它使用了一种连接方式叫做“shortcut connection”，顾名思义，shortcut 就是“抄近道”的意思，看下图我们就能大致理解：

![image-20210717110814008](C:\Users\86133\Desktop\images\image-20210717110814008.png)

<center>图 1 shortcut connection 示意图

在 ResNet 的论文中也给出了两种“抄近道”的方式：

![image-20210717110859631](C:\Users\86133\Desktop\images\image-20210717110859631.png)

<center>图 2 两种 ResNet 的设计

这两种结构分别针对 ResNet34（左图）和 ResNet50/101/152（右图），一般称整个结构为一个“building block”。其中右图又称为“bottleneck  design”，目的一目了然，就是为了降低参数的数目。ResNet 的整体结构如下图所示：

![image-20210717111014196](C:\Users\86133\Desktop\images\image-20210717111014196.png)

<center>图 3 ResNet 的整体网络结构