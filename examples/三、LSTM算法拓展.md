三、LSTM算法拓展:

实验原理讲解:

LSTM:



LSTM 的控制流程与 RNN 相似，它们都是在前向传播的过程中处理流经细胞的数据，不同之处在于 LSTM 中细胞的结构和运算有所变化。



![img](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/BnSNEaficFAb1h6NCo05atBXvdu7Q8PibphuFVzUdDe1fiaRE2DsoOic94MyicU5tHGtIZCFW4J8O0EQ3jg79XicMUtQ/640?wx_fmt=png)

LSTM 的细胞结构和运算



这一系列运算操作使得 LSTM具有能选择保存信息或遗忘信息的功能。咋一看这些运算操作时可能有点复杂，但没关系下面将带你一步步了解这些运算操作。



核心概念:



LSTM 的核心概念在于细胞状态以及“门”结构。细胞状态相当于信息传输的路径，让信息能在序列连中传递下去。你可以将其看作网络的“记忆”。理论上讲，细胞状态能够将序列处理过程中的相关信息一直传递下去。



因此，即使是较早时间步长的信息也能携带到较后时间步长的细胞中来，这克服了短时记忆的影响。信息的添加和移除我们通过“门”结构来实现，“门”结构在训练过程中会去学习该保存或遗忘哪些信息。



Sigmoid：



门结构中包含着 sigmoid 激活函数。Sigmoid 激活函数与 tanh 函数类似，不同之处在于 sigmoid 是把值压缩到 0~1 之间而不是 -1~1 之间。这样的设置有助于更新或忘记信息，因为任何数乘以 0 都得 0，这部分信息就会剔除掉。同样的，任何数乘以 1 都得到它本身，这部分信息就会完美地保存下来。这样网络就能了解哪些数据是需要遗忘，哪些数据是需要保存。



![img](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_gif/BnSNEaficFAb1h6NCo05atBXvdu7Q8Pibpb150JKZMhIL5F7alYwsrYfKx72WicmDkdzQ5S6nRomo0DPWBwkB2dlQ/640?wx_fmt=gif)

Sigmoid 将值压缩到 0~1 之间



接下来了解一下门结构的功能。LSTM 有三种类型的门结构：遗忘门、输入门和输出门。



遗忘门：



遗忘门的功能是决定应丢弃或保留哪些信息。来自前一个隐藏状态的信息和当前输入的信息同时传递到 sigmoid 函数中去，输出值介于 0 和 1 之间，越接近 0 意味着越应该丢弃，越接近 1 意味着越应该保留。



![img](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_gif/BnSNEaficFAb1h6NCo05atBXvdu7Q8PibpmDGVicLxUDdUbXlRayNoMS6TSwlPNBamo5TuZPKZZET7hgfXPbWgZPg/640?wx_fmt=gif)

遗忘门的运算过程



输入门：



输入门用于更新细胞状态。首先将前一层隐藏状态的信息和当前输入的信息传递到 sigmoid 函数中去。将值调整到 0~1 之间来决定要更新哪些信息。0 表示不重要，1 表示重要。



其次还要将前一层隐藏状态的信息和当前输入的信息传递到 tanh 函数中去，创造一个新的侯选值向量。最后将 sigmoid 的输出值与 tanh 的输出值相乘，sigmoid 的输出值将决定 tanh 的输出值中哪些信息是重要且需要保留下来的。



![img](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_gif/BnSNEaficFAb1h6NCo05atBXvdu7Q8PibptMibLu41FZ8MibYStbhH6YOw2QDS3OcnZbkY6JDkvperguweVwR1TviaA/640?wx_fmt=gif)

输入门的运算过程



细胞状态：



下一步，就是计算细胞状态。首先前一层的细胞状态与遗忘向量逐点相乘。如果它乘以接近 0 的值，意味着在新的细胞状态中，这些信息是需要丢弃掉的。然后再将该值与输入门的输出值逐点相加，将神经网络发现的新信息更新到细胞状态中去。至此，就得到了更新后的细胞状态。



![img](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_gif/BnSNEaficFAb1h6NCo05atBXvdu7Q8PibpseuYuxGJ27Xcxp1EIuYLichXiaQTk7dnp5MV86PiabHxQLP9QbPhyFdBA/640?wx_fmt=gif)

细胞状态的计算



输出门：



输出门用来确定下一个隐藏状态的值，隐藏状态包含了先前输入的信息。首先，我们将前一个隐藏状态和当前输入传递到 sigmoid 函数中，然后将新得到的细胞状态传递给 tanh 函数。



最后将 tanh 的输出与 sigmoid 的输出相乘，以确定隐藏状态应携带的信息。再将隐藏状态作为当前细胞的输出，把新的细胞状态和新的隐藏状态传递到下一个时间步长中去。



![img](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_gif/BnSNEaficFAb1h6NCo05atBXvdu7Q8PibpIoYhCNTzuaS4XcPosgoW6owdgEePRbnnAo5Zicibc2WZUqCKGDibdqQvw/640?wx_fmt=gif)

 输出门的运算过程



