## CNN-DSSM
针对 DSSM 词袋模型丢失上下文信息的缺点，CLSM（convolutional latent semantic model）应运而生，又叫 CNN-DSSM。CNN-DSSM 与 DSSM 的区别主要在于输入层和表示层。

输入层输出层如图所示。
优点：

CNN-DSSM 通过卷积层提取了滑动窗口下的上下文信息，又通过池化层提取了全局的上下文信息，上下文信息得到较为有效的保留。

缺点：

对于间隔较远的上下文信息，难以有效保留。 举个例子，I grew up in France... I speak fluent French，显然 France 和 French 是具有上下文依赖关系的，但是由于 CNN-DSSM 滑动窗口（卷积核）大小的限制，导致无法捕获该上下文信息。

## LSTM-DSSM
LSTM(（Long-Short-Term Memory）是一种 RNN 特殊的类型，可以学习长期依赖信息。我们分别来介绍它最重要的几个模块：

**细胞状态**
细胞状态这条线可以理解成是一条信息的传送带，只有一些少量的线性交互。在上面流动可以保持信息的不变性。

**遗忘门**
遗忘门由 Gers 提出，它用来控制细胞状态 cell 有哪些信息可以通过，继续往下传递。如下图所示，上一层的输出 h(t-1) concat 上本层的输入 xt，经过一个 sigmoid 网络（遗忘门）产生一个从 0 到 1 的数值 ft，然后与细胞状态 C(t-1) 相乘，最终决定有多少细胞状态可以继续往后传递。

**输入门**
输入门决定要新增什么信息到细胞状态，这里包含两部分：一个 sigmoid 输入门和一个 tanh 函数。sigmoid 决定输入的信号控制，tanh 决定输入什么内容。如下图所示，上一层的输出 h(t-1) concat 上本层的输入 xt，经过一个 sigmoid 网络（输入门）产生一个从 0 到 1 的数值 it，同样的信息经过 tanh 网络做非线性变换得到结果 Ct，sigmoid 的结果和 tanh 的结果相乘，最终决定有哪些信息可以输入到细胞状态里。

**输出门**
输出门决定从细胞状态要输出什么信息，这里也包含两部分：一个 sigmoid 输出门和一个 tanh 函数。sigmoid 决定输出的信号控制，tanh 决定输出什么内容。如下图所示，上一层的输出 h(t-1) concat 上本层的输入 xt，经过一个 sigmoid 网络（输出门）产生一个从 0 到 1 的数值 Ot，细胞状态 Ct 经过 tanh 网络做非线性变换，得到结果再与 sigmoid 的结果 Ot 相乘，最终决定有哪些信息可以输出，输出的结果 ht 会作为这个细胞的输出，也会作为传递个下一个细胞。

LSTM-DSSM 其实用的是 LSTM 的一个变种——加入了peep hole的 LSTM。结构如图。

## MMoE
主要是解决传统的 multi-task 网络 (主要采用 Shared-Bottom Structure) 可能在任务相关性不强的情况下效果不佳的问题, 有研究揭示了 multi-task 模型的效果高度依赖于任务之间的相关性;
MMoE 借鉴 MoE 的思路, 引入多个 Experts (即多个 NN 网络) 网络, 然后再对每个 task 分别引入一个 gating network, gating 网络针对各自的 task 学习 experts 网络的不同组合模式, 即对 experts 网络的输出进行自适应加权. 说实话, 这一点非常像 Attention, Experts 网络学习出 embedding 序列, 而 gating 网络学习自适应的权重并对 Experts 网络的输出进行加权求和, 得到对应的结果之后再分别输入到各个 task 对应的 tower 网络中. 注意 gating 网络的数量和任务的数量是一致的.

    def call(self, inputs, **kwargs):
        """
        """
        gate_outputs = []
        final_outputs = []

        # f_{i}(x) = activation(W_{i} * x + b), where activation is ReLU according to the paper
        """
		inputs 输入 Tensor 的大小为 [B, I],
		self.expert_kernels 的大小为 [I, E, N],
		其中 I 为输入 embedding 大小, E 为 Experts 网络的输出大小, N 为 Experts 的个数
		tf.tensordot(a, b, axes=1) 相当于 tf.tensordot(a, b, axes=[[1],[0]]),
		因此 expert_outputs 的大小为 [B, E, N] 
		"""
        expert_outputs = K.tf.tensordot(a=inputs, b=self.expert_kernels, axes=1)
        # Add the bias term to the expert weights if necessary
        if self.use_expert_bias:
            expert_outputs = K.bias_add(x=expert_outputs, bias=self.expert_bias)
        """
        加上 Bias 以及通过激活函数 (relu) 后, expert_outputs 大小仍为 [B, E, N]
		"""
        expert_outputs = self.expert_activation(expert_outputs)

        # g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper
        """
		针对 K 个 Task 分别学习各自的 Gate 网络, 这里采用 for 循环实现,
		其中 inputs 的大小为 [B, I],
		gate_kernel 的大小为 [I, N], 其中 I 为输入 embedding 的大小,
		而 N 为 Experts 的个数. 因此 K.dot 对 inputs 和 gate_kernel 进行矩阵乘法,
		得到 gate_output 的大小为 [B, N].
		注意 gate_activation 为 softmax, 因此经过 Bias 以及 gate_activation 后,
		gate_output 的大小为 [B, N], 保存着各 Experts 网络的权重系数
		"""
        for index, gate_kernel in enumerate(self.gate_kernels):
            gate_output = K.dot(x=inputs, y=gate_kernel)
            # Add the bias term to the gate weights if necessary
            if self.use_gate_bias:
                gate_output = K.bias_add(x=gate_output, bias=self.gate_bias[index])
            gate_output = self.gate_activation(gate_output)
            gate_outputs.append(gate_output)

        # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x))
        """
		gate_outputs 为大小等于 K (任务个数) 的列表, 其中 gate_output 的大小等于 [B, N],
		而 expert_outputs 的大小为 [B, E, N];
		因此, 首先对 gate_output 使用 expand_dims, 按照 axis=1 进行, 得到
		expanded_gate_output 大小为 [B, 1, N];
		K.repeat_elements 将 expanded_gate_output 扩展为 [B, E, N],
		之后再乘上 expert_outputs, 得到 weighted_expert_output 大小为 [B, E, N];
		此时每个 Experts 网络都乘上了对应的系数, 最后只需要对各个 Experts 网络的输出进行加权
		求和即可, 因此 K.sum(weighted_expert_output, axis=2) 的结果大小为 [B, E];
		"""
        for gate_output in gate_outputs:
            expanded_gate_output = K.expand_dims(gate_output, axis=1) ## [B, 1, N]
            weighted_expert_output = expert_outputs * K.repeat_elements(expanded_gate_output, self.units, axis=1)  ## [B, E, N]
            final_outputs.append(K.sum(weighted_expert_output, axis=2)) ## [B, E]

        return final_outputs

## ShareBottom
特征embedding可以使用end2end和预训练两种方式。预训练可以使用word2vec，GraphSAGE等工业界落地的算法，训练全站id embedding特征，在训练dnn或则multi task的过程中fine-tune。end2end训练简单，可以很快就将模型train起来，直接输入id特征，模型从头开始学习id的embedding向量。这种训练方式最致命的缺陷就是训练不充分，某些id在训练集中出现次数较少甚至没有出现过，在inference阶段遇到训练过程中没有遇到的id，就直接走冷启了。这在全站item变化比较快的情况下，这种方式就不是首选的方式。


## Youtube
一个是用于候选集生成（candidate generation），另一个则是用于排序。
使用用户的历史作为输入，候选集生成网络（candidate generation network）显著地减少了视频的数量，并且可以从一个大型语料库中选取一组最相关的视频集。生成的候选集对用户来说是最为相关的，此神经网络的目的仅仅是为了通过协同过滤来提供一个宽泛的个性化服务。
在这一步中，我们拥有了更少量的候选结果，这些结果与用户需求更加接近。我们现在的目的是仔细地分析所有候选结果，这样我们就可以做出最好的决策。此任务是由排序网络（ranking network）来完成的，它可以根据一个期望的目标函数为每一个视频都分配一个分数，这个目标函数是使用数据来对有关用户行为的视频和信息来进行描述的。


使用两阶段法（two-stage approach），我们就能够从很大的视频语料库中做出视频推荐，然而可以确信的是，这些推荐结果中只有少量是个性化的，而且是被用户真正进行应用的。这一设计也能使我们把其它资源生成的结果和这些候选结果混合在一起。
推荐任务就像是一个极端的多类别分类问题，预测问题变成了一个在给定的时间 t 下，基于用户（U）和语境（C），对语料库（V）中数百万的视频类别（i）中的一个特定视频（wt）进行精准分类的问题。

