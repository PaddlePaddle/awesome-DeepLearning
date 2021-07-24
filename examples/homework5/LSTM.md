RNN基本原理

前言

当我们处理与事件发生的时间轴有关系的问题时，比如自然语言处理，文本处理，文字的上下文是有一定的关联性的；时间序列数据，如连续几天的天气状况，当日的天气情况与过去的几天有某些联系；又比如语音识别，机器翻译等。在考虑这些和时间轴相关的问题时，传统的神经网络就无能为力了，因此就有了RNN（recurrent neural network，循环神经网络），了解RNN先了解DNN基本原理。同样这里介绍RNN基本原理，也是为了铺垫我们的重点LSTM网络（long short term memory，长短时记忆神经网络）。

定义

递归神经网络（RNN）是两种人工神经网络的总称。一种是时间递归神经网络（recurrent neural network），另一种是结构递归神经网络（recursive neural network）。时间递归神经网络的神经元间连接构成矩阵，而结构递归神经网络利用相似的神经网络结构递归构造更为复杂的深度网络。RNN一般指代时间递归神经网络。

网络结构
![](https://ai-studio-static-online.cdn.bcebos.com/fe502969d9d24bc78a88d44d03c609bcfb57b085d6b047adb89a6db307784d86)


图1.DNN基础结构
对于这样一个DNN正向传播的基础结构来说，我们的整个过程就是，将输入x与权重矩阵w结合，以wx + b的形式输入隐藏层（Layer L2），经过激活函数f(x)的处理，得到输出结果a1, a2, a3， 然后与对应的权重、偏置结合，作为输出层（Layer L3）的输入，经过激活函数，得到最终输出结果。

![](https://ai-studio-static-online.cdn.bcebos.com/32975253c1fc447d8bd89984ba3c425679c11a99dd87484591c7ef52fcaa3358)


图2.RNN基础结构


其中g1常用tanh和Relu激活函数，g2常用sigmoid或者softmax.

BP

循环神经网络的训练类似于传统神经网络的训练。我们也使用反向传播算法，但是有所变化。因为循环神经网络在所有时刻的参数是共享的，但是每个输出的梯度不仅依赖当前时刻的计算，还依赖之前时刻的计算。例如，为了计算时刻 t = 4 的梯度，我们还需要反向传播3步，然后将梯度相加。这个被称为Backpropagation Through Time（BPTT）。

这与我们在深度前馈神经网络中使用的标准反向传播算法基本相同。主要的差异就是我们将每时刻 W 的梯度相加。在传统的神经网络中，我们在层之间并没有共享参数，所以我们不需要相加。

反向传播基本和DNN中BP算法一致，这里不做赘述了，从y^t>
与真实值y<t>做损失函数计算，所有RNN-cellde 损失函数做成本函数计算，以此为目标函数，基于最优化方法，常用梯度下降法进行目标优化，从而最终得到我们的最终结果。

小结
  
RNN 的关键点之一就是他们可以用来连接先前的信息到当前的任务上，例如使用过去的视频段来推测对当前段的理解。但是当相关信息和当前预测位置之间的间隔变得非常大，RNN 会丧失学习到连接如此远的信息的能力。我们仅仅需要明白的是利用BPTT算法训练出来的普通循环神经网络很难学习长期依赖（例如，距离很远的两步之间的依赖），原因就在于梯度消失/发散问题。但是RNN 绝对可以处理这样的长期依赖问题，人们可以仔细挑选参数来解决这类问题中的最初级形式，但在实践中，RNN 肯定不能够成功学习到这些知识。训练和参数设计十分复杂。LSTM就是专门设计出来解决这个问题的。

LSTM
LSTM网络
long short term memory，即我们所称呼的LSTM，是为了解决长期以来问题而专门设计出来的，所有的RNN都具有一种重复神经网络模块的链式形式。在标准RNN中，这个重复的结构模块只有一个非常简单的结构，例如一个tanh层。
![](https://ai-studio-static-online.cdn.bcebos.com/1a9578031c8a4e0484aae496a1af02c32096c5265b034b8a8b47e0abc363c70a)
  


图3.RNNcell
LSTM 同样是这样的结构，但是重复的模块拥有一个不同的结构。不同于单一神经网络层，这里是有四个，以一种非常特殊的方式进行交互。
![](https://ai-studio-static-online.cdn.bcebos.com/163d4f8f03924ab99393214dee80beb08c285f65440849a18ff7dbb3256590f8)
  


图4.LSTMcell
LSTM核心思想
LSTM的关键在于细胞的状态整个(绿色的图表示的是一个cell)，和穿过细胞的那条水平线。

细胞状态类似于传送带。直接在整个链上运行，只有一些少量的线性交互。信息在上面流传保持不变会很容易。
![](https://ai-studio-static-online.cdn.bcebos.com/8e8cb7deeae74e25a0182f3453e0d1905be8fec82eeb4b12a409277fa14674bb)
  


图5.LSTMcell内部结构图
若只有上面的那条水平线是没办法实现添加或者删除信息的。而是通过一种叫做 门（gates） 的结构来实现的。

门 可以实现选择性地让信息通过，主要是通过一个 sigmoid 的神经层 和一个逐点相乘的操作来实现的。
![](https://ai-studio-static-online.cdn.bcebos.com/f518225e39f2453cac1be42d338e11b51fb9d5ecef654da8809f9b05d326fc24)
  


图6.信息节点
sigmoid 层输出（是一个向量）的每个元素都是一个在 0 和 1 之间的实数，表示让对应信息通过的权重（或者占比）。比如， 0 表示“不让任何信息通过”， 1 表示“让所有信息通过”。

LSTM通过三个这样的基本结构来实现信息的保护和控制。这三个门分别输入门、遗忘门和输出门。

深入理解LSTM
  
遗忘门
  
在我们 LSTM 中的第一步是决定我们会从细胞状态中丢弃什么信息。这个决定通过一个称为忘记门层完成。该门会读取h t−1和xt ,输出一个在 0到 1之间的数值给每个在细胞状态Ct−1中的数字。1 表示“完全保留”，0 表示“完全舍弃”。

![](https://ai-studio-static-online.cdn.bcebos.com/ade405439ede41e0a3767f5d09ac642cd6ba3a943e0a4e3189ab82ff5d97346a)
  

其中ht−1表示的是上一个cell的输出，xt表示的是当前细胞的输入。σσ表示sigmod函数。

输入门
下一步是决定让多少新的信息加入到 cell 状态 中来。实现这个需要包括两个 步骤：首先，一个叫做“input gate layer ”的 sigmoid 层决定哪些信息需要更新；一个 tanh 层生成一个向量，也就是备选的用来更新的内容，C^t 。在下一步，我们把这两部分联合起来，对 cell 的状态进行一个更新。

![](https://ai-studio-static-online.cdn.bcebos.com/148c53f64416435596be0802d327d9548f0cd2f5eccd4e0a8156ffcdb15286bc)
  

现在是更新旧细胞状态的时间了，Ct−1更新为Ct。前面的步骤已经决定了将会做什么，我们现在就是实际去完成。

我们把旧状态与ft相乘，丢弃掉我们确定需要丢弃的信息。接着加上it∗C~t。这就是新的候选值，根据我们决定更新每个状态的程度进行变化。

输出门
最终，我们需要确定输出什么值。这个输出将会基于我们的细胞状态，但是也是一个过滤后的版本。首先，我们运行一个 sigmoid 层来确定细胞状态的哪个部分将输出出去。接着，我们把细胞状态通过 tanh 进行处理（得到一个在 -1 到 1 之间的值）并将它和 sigmoid 门的输出相乘，最终我们仅仅会输出我们确定输出的那部分。
![](https://ai-studio-static-online.cdn.bcebos.com/5f0b1ead93824502960c4f8028e5950e2d33b35c5df343bb9479ad14e7cd15b1)
  



请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
